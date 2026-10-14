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

