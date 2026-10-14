"""Local re-implementation of the YeNo x Builderr paper evaluator.

Built from yeno-evaluator-contract.md. Every rule the contract states is
implemented here; where the contract is silent, the choice made is the more
conservative one and is marked "ASSUMPTION".

Market data format (one dict per 5-minute market, see sim/prep_*.py):
    slug, start, end, winner ("YES"/"NO"),
    t  (N,) receive times of YES book states (time-sorted),
    bp, bs, ap, as  (N, L) YES bid/ask ladders, NaN padded, best first.
    ref_t, ref_p   receive time / value of the public BTC reference,
    target         (receive time, opening target) or None.
The NO (Down) book is the exact mirror of YES on Polymarket's CLOB:
NO bids = 1 - YES asks, NO asks = 1 - YES bids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

OUTCOMES = ("YES", "NO")
START_CASH = 10.0
MAX_BUY = 5.0
TARGET = 1000.0
LATENCY = 0.25
STALE = 2.0
ENTRY_CUTOFF = 15.0
MIN_ORDER = 5.0
OVERLAY = 0.01
PROTOCOL = 0.07
DUST = 1e-6


def fees(fills: list[tuple[float, float]]) -> float:
    gross = sum(p * q for p, q in fills)
    protocol = sum(PROTOCOL * q * p * (1 - p) for p, q in fills)
    return round(protocol + 1e-12, 5) + round(gross * OVERLAY + 1e-12, 5)


def unit_cost(p: float) -> float:
    return p + PROTOCOL * p * (1 - p) + OVERLAY * p


def fill_buy(asks, budget: float, adverse: float = 0.0):
    """Spend up to `budget` (fee inclusive) walking the asks. Returns fills."""
    fills: list[tuple[float, float]] = []
    left = budget
    for price, size in asks:
        p = min(0.999, price + adverse)
        uc = unit_cost(p)
        q = min(size, left / uc)
        q = math.floor(q * 100 + 1e-9) / 100  # ASSUMPTION: 0.01-share granularity
        if q <= 0:
            break
        fills.append((p, q))
        left -= q * uc
        if q < size:
            break
    # fee rounding can push total a hair over budget: trim
    while fills and sum(p * q for p, q in fills) + fees(fills) > budget + 1e-9:
        p, q = fills[-1]
        q = round(q - 0.01, 2)
        if q <= 0:
            fills.pop()
        else:
            fills[-1] = (p, q)
    return fills


def fill_sell(bids, shares: float, adverse: float = 0.0):
    fills: list[tuple[float, float]] = []
    left = shares
    for price, size in bids:
        p = max(0.001, price - adverse)
        q = min(size, left)
        fills.append((p, q))
        left -= q
        if left <= DUST:
            break
    return fills


@dataclass
class Position:
    market_id: str
    outcome: str
    shares: float = 0.0
    buy_gross_usd: float = 0.0
    buy_fees_usd: float = 0.0
    sell_gross_usd: float = 0.0
    sell_fees_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "outcome": self.outcome,
            "shares": round(self.shares, 6),
            "buy_gross_usd": round(self.buy_gross_usd, 6),
            "buy_fees_usd": round(self.buy_fees_usd, 6),
            "sell_gross_usd": round(self.sell_gross_usd, 6),
            "sell_fees_usd": round(self.sell_fees_usd, 6),
        }


@dataclass
class Result:
    qualified: bool = False
    cash: float = START_CASH
    volume: float = 0.0
    cycles: int = 0
    max_drawdown: float = 0.0
    time_to_target_h: float | None = None
    flat_at_end: bool = True
    settled_positions: int = 0
    rejected: int = 0
    queries: int = 0
    losses: int = 0
    wins: int = 0
    cycle_pnls: list = field(default_factory=list)
    errors: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "qualified": self.qualified,
            "cash": round(self.cash, 4),
            "volume": round(self.volume, 2),
            "cycles": self.cycles,
            "maxDD": round(self.max_drawdown, 3),
            "hours": None if self.time_to_target_h is None else round(self.time_to_target_h, 2),
            "flat": self.flat_at_end,
            "settled": self.settled_positions,
            "rejected": self.rejected,
            "W/L": f"{self.wins}/{self.losses}",
        }


def run(
    markets: list[dict],
    decide: Callable[[dict], dict],
    *,
    query_interval: float = 0.1,
    adverse: float = 0.0,
    max_hours: float = 24.0,
    stop_at_target: bool = True,
) -> Result:
    """Run one evaluation path (a chronological list of markets)."""
    res = Result()
    cash = START_CASH
    peak = cash
    volume = 0.0
    cycles = 0
    pos: Position | None = None
    if not markets:
        return res
    t0 = markets[0]["start"]
    deadline = t0 + max_hours * 3600

    for mk in markets:
        if mk["start"] >= deadline:
            break
        mid = mk["slug"]
        end = mk["end"]
        T, BP, BS, AP, AS = mk["t"], mk["bp"], mk["bs"], mk["ap"], mk["as"]
        ref_t, ref_p = mk["ref_t"], mk["ref_p"]
        target = mk.get("target")
        ri = 0
        nref = len(ref_t)
        ref = None
        pending = None  # (exec_after, action dict)
        last_q = -math.inf

        def ladders(i: int):
            """(yes_bids, yes_asks, no_bids, no_asks) as [(price, size)] lists."""
            yb = [(float(p), float(q)) for p, q in zip(BP[i], BS[i]) if p == p and q > 0]
            ya = [(float(p), float(q)) for p, q in zip(AP[i], AS[i]) if p == p and q > 0]
            nb = [(round(1 - p, 4), q) for p, q in ya]
            na = [(round(1 - p, 4), q) for p, q in yb]
            return yb, ya, nb, na

        def observation(t: float, lad) -> dict[str, Any]:
            yb, ya, nb, na = lad
            reference = None
            if ref is not None:
                reference = {"btcMidUsd": ref[1], "observedAt": ref[0]}
                if target is not None and target[0] <= t:
                    reference["openingTargetUsd"] = target[1]
                    reference["targetObservedAt"] = target[0]
                    reference["targetProvisional"] = False
                else:
                    reference["targetProvisional"] = True
            return {
                "schemaVersion": 1,
                "timestamp": t,
                "market": {"id": mid, "secondsToClose": round(end - t, 3)},
                "account": {
                    "cashUsd": round(cash, 6),
                    "eligibleVolumeUsd": round(volume, 6),
                    "completedCycles": cycles,
                    "position": pos.as_dict() if pos else None,
                },
                "rules": {
                    "maximumBuyCashUsd": MAX_BUY,
                    "targetVolumeUsd": TARGET,
                    "evaluationWindowHours": 24,
                    "onePositionAtATime": True,
                    "buyFeesIncludedInMaximum": True,
                },
                "reference": reference,
                "books": {
                    "YES": {"bids": [list(x) for x in yb], "asks": [list(x) for x in ya], "minOrderSize": MIN_ORDER},
                    "NO": {"bids": [list(x) for x in nb], "asks": [list(x) for x in na], "minOrderSize": MIN_ORDER},
                },
            }

        for i in range(len(T)):
            t = float(T[i])
            if t >= end or t >= deadline:
                break
            while ri < nref and ref_t[ri] <= t:
                ref = (float(ref_t[ri]), float(ref_p[ri]))
                ri += 1
            lad = None

            # 1) execute a pending action against this later update of its own outcome
            if pending is not None and t >= pending[0]:
                act = pending[1]
                lad = ladders(i)
                k = 0 if act["_outcome"] == "YES" else 2
                bids, asks = lad[k], lad[k + 1]
                pending = None
                if act["action"] == "BUY":
                    if end - t <= ENTRY_CUTOFF or pos is not None:
                        res.rejected += 1
                    else:
                        budget = min(float(act["maxCashUsd"]), cash, MAX_BUY)
                        fills = fill_buy(asks, budget, adverse)
                        sh = sum(q for _, q in fills)
                        if sh < MIN_ORDER - 1e-9:
                            res.rejected += 1  # below minOrderSize (or no asks)
                        else:
                            g = sum(p * q for p, q in fills)
                            f = fees(fills)
                            cash -= g + f
                            pos = Position(mid, act["outcome"], sh, g, f)
                elif act["action"] == "SELL" and pos is not None:
                    fills = fill_sell(bids, pos.shares, adverse)
                    if not fills:
                        res.rejected += 1
                    else:
                        sh = sum(q for _, q in fills)
                        g = sum(p * q for p, q in fills)
                        f = fees(fills)
                        cash += g - f
                        pos.shares -= sh
                        pos.sell_gross_usd += g
                        pos.sell_fees_usd += f
                        if pos.shares <= DUST:
                            pnl = pos.sell_gross_usd - pos.sell_fees_usd - pos.buy_gross_usd - pos.buy_fees_usd
                            res.cycle_pnls.append(pnl)
                            if pnl >= 0:
                                res.wins += 1
                            else:
                                res.losses += 1
                            volume += pos.buy_gross_usd + pos.sell_gross_usd
                            cycles += 1
                            pos = None
                            peak = max(peak, cash)
                            res.max_drawdown = max(res.max_drawdown, peak - cash)
                            if volume >= TARGET - 1e-9 and res.time_to_target_h is None:
                                res.qualified = True
                                res.time_to_target_h = (t - t0) / 3600
                                if stop_at_target:
                                    res.cash, res.volume, res.cycles = cash, volume, cycles
                                    res.flat_at_end = True
                                    return res

            # 2) ask the bot when nothing is pending. Queries are triggered by a
            #    book update, so the book the bot sees is always fresh.
            if pending is None and t - last_q >= query_interval:
                if True:
                    last_q = t
                    res.queries += 1
                    if lad is None:
                        lad = ladders(i)
                    try:
                        d = decide(observation(t, lad))
                    except Exception:
                        res.errors += 1
                        d = {"action": "HOLD"}
                    a = d.get("action") if isinstance(d, dict) else None
                    if a == "BUY":
                        oc = d.get("outcome")
                        try:
                            mc = float(d.get("maxCashUsd", 0))
                        except (TypeError, ValueError):
                            mc = 0
                        if pos is None and oc in OUTCOMES and 0 < mc <= MAX_BUY + 1e-9 and mc <= cash + 1e-9:
                            pending = (t + LATENCY, {"action": "BUY", "outcome": oc, "maxCashUsd": mc, "_outcome": oc})
                        else:
                            res.rejected += 1
                    elif a == "SELL":
                        if pos is not None:
                            pending = (t + LATENCY, {"action": "SELL", "_outcome": pos.outcome})
                        else:
                            res.rejected += 1

        # market closed: unsold inventory settles (no volume credit, cycle lost)
        if pos is not None:
            win = 1.0 if pos.outcome == mk["winner"] else 0.0
            cash += pos.shares * win
            res.settled_positions += 1
            res.cycle_pnls.append(pos.sell_gross_usd - pos.sell_fees_usd + pos.shares * win - pos.buy_gross_usd - pos.buy_fees_usd)
            pos = None
            peak = max(peak, cash)
            res.max_drawdown = max(res.max_drawdown, peak - cash)

    res.cash, res.volume, res.cycles = cash, volume, cycles
    res.flat_at_end = pos is None
    return res
