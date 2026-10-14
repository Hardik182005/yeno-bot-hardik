#!/usr/bin/env python3
"""YeNo x Builderr volume bot - adaptive multi-strategy.

Why adaptive
------------
Every $1 of volume pays the 1% YeNo overlay, so $1,000 of volume costs $10 in
overlay alone - the whole starting bankroll. A bot can only qualify if it earns
trading profit that offsets fees, and whether such profit exists depends on
the microstructure of the hidden evaluation data (reference feed latency,
recorder latency, volatility regime). Research on public tick data showed that
no single fixed rule is profitable in every regime.

So the bot runs a portfolio of strategy variants. All of them are paper-traded
*inside the bot* ("shadow trades") against the live observation stream with the
evaluator's own rules: fill at the first book seen >= 250 ms after the
decision, ladder walking, both fee layers, partial-cycle settlement with zero
volume. For every variant the bot measures

    r = realised PnL / eligible volume      (cost per $ of volume, usually < 0)

and only spends real orders on variants whose lower confidence bound on r fits
the remaining budget:

    LCB(r) >= -(cash - reserve) / volume still needed

Among qualifying variants it uses the best one; when ahead of the volume
schedule it becomes more selective to preserve cash, which is what ranks.

Variant families
----------------
* LAG   - reference moved, book has not caught up: buy the side that reprices.
* FAV   - House Bot v1 style: buy the reference favourite late, exit at
          break-even-or-better, stop loss, forced exit.
* LOCK  - settlement-TWAP lock-in: late in the window the 60 s settlement TWAP
          is mostly observed; buy when model probability exceeds the ask,
          exit into the close.

Standard library only. POST /decide on 127.0.0.1:8080 (HOST / PORT env vars).
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

OVERLAY = 0.01
PROTOCOL = 0.07
LATENCY = 0.25
ENTRY_CUTOFF = 15.0
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

DEFAULTS: dict[str, Any] = {
    "reserve_usd": 1.0,        # cash kept out of the volume budget
    "lcb_k": 2.0,              # confidence multiplier on the cost-per-volume estimate
    "min_cycles": 30,          # shadow cycles before a variant may trade live
    "rate_safety": 1.5,        # selected variants must trade this much faster than needed
    "rescue_hour": 12.0,       # after this hour, if far behind schedule ...
    "rescue_behind": 0.5,      # ... (volume < this share of plan), relax the evidence bar
    "brake_cash": 6.0,         # below this cash only variants with LCB >= 0 may trade
    "ban_live_cycles": 8,      # live cycles before the ratio kill-switch can fire
    "ban_live_loss": 1.5,      # ban a variant whose live trades lost this much in total
    "min_shadow_vol": 300.0,   # shadow volume ($) before a variant may trade live
    "risk_frac": 0.08,         # size so that one RMS-return loss is this share of cash
    "ban_margin": 0.01,        # ban if live PnL/vol is this much worse than allowed
    "decay_cycles": 400.0,     # effective memory of the statistics (cycles)
    "prior_r": -0.02,          # prior cost per $ volume (pessimistic)
    "prior_weight_usd": 40.0,  # prior strength in $ of volume
    "pace_hours": 21.0,        # plan to cross $1,000 by this hour
    "sigma_prior": 3.0,        # USD per sqrt(second)
    "sigma_halflife_s": 900.0,
    "max_ref_age": 2.5,
    "settle_guess": True,
    "variants": None,          # override the variant list (research)
    "live": True,              # False = shadow only (diagnostics)
}


# ---------------------------------------------------------------- math utils
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF (|error| < 1.2e-9)."""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def levels(rows: Any, descending: bool) -> list[tuple[float, float]]:
    out = []
    if isinstance(rows, list):
        for row in rows:
            try:
                p, s = float(row[0]), float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0.0 < p < 1.0 and s > 0 and math.isfinite(p) and math.isfinite(s):
                out.append((p, s))
    out.sort(reverse=descending)
    return out


def fees(fills: list[tuple[float, float]]) -> float:
    gross = sum(p * q for p, q in fills)
    protocol = sum(PROTOCOL * q * p * (1 - p) for p, q in fills)
    return round(protocol + 1e-12, 5) + round(gross * OVERLAY + 1e-12, 5)


def sim_buy(asks, budget: float):
    """Evaluator-style market buy for a fee-inclusive budget -> (shares, gross, fees)."""
    fills = []
    left = budget
    for p, s in asks:
        unit = p * (1 + OVERLAY) + PROTOCOL * p * (1 - p)
        q = math.floor(min(s, left / unit) * 100 + 1e-9) / 100
        if q <= 0:
            break
        fills.append((p, q))
        left -= q * unit
        if q < s:
            break
    while fills and sum(p * q for p, q in fills) + fees(fills) > budget + 1e-9:
        p, q = fills[-1]
        q = round(q - 0.01, 2)
        if q <= 0:
            fills.pop()
        else:
            fills[-1] = (p, q)
    sh = sum(q for _, q in fills)
    return sh, sum(p * q for p, q in fills), fees(fills)


def sim_sell(bids, shares: float):
    """Evaluator-style market sell -> (shares sold, gross, fees)."""
    fills = []
    left = shares
    for p, s in bids:
        q = min(s, left)
        fills.append((p, q))
        left -= q
        if left <= 1e-9:
            break
    return sum(q for _, q in fills), sum(p * q for p, q in fills), fees(fills)


# ---------------------------------------------------------------- variants
def default_variants() -> list[dict[str, Any]]:
    v: list[dict[str, Any]] = []
    # LAG: reference move over `lb` seconds, predicted mid move >= edge (price units)
    for lb in (1.5, 3.0):
        for edge in (0.06, 0.12, 0.2, 0.3, 0.45):
            for hold in (2.0, 5.0):
                v.append({"fam": "LAG", "lb": lb, "edge": edge, "hold": hold,
                          "tau_lo": 20.0, "tau_hi": 280.0, "stop": 0.10, "max_spread": 0.03})
    # FAV: House Bot v1 exactly, then volatility-scaled versions
    # House Bot v1 exactly, then a small neighbourhood around it
    v.append({"fam": "FAV", "house": True, "tau_lo": 15.0, "tau_hi": 90.0, "gap": 50.0,
              "max_ask": 0.90, "min_ask": 0.0, "stop": 0.10, "forced": 60.0, "max_loss": 0.40,
              "cycles": 2})
    for gap in (30.0, 80.0):
        for max_ask in (0.90, 0.95):
            v.append({"fam": "FAV", "house": True, "tau_lo": 15.0, "tau_hi": 90.0, "gap": gap,
                      "max_ask": max_ask, "min_ask": 0.0, "stop": 0.10, "forced": 60.0,
                      "max_loss": 0.40, "cycles": 2})
    for gap in (50.0,):
        for (tlo, thi, forced) in ((45.0, 150.0, 40.0), (15.0, 90.0, 30.0)):
            v.append({"fam": "FAV", "house": True, "tau_lo": tlo, "tau_hi": thi, "gap": gap,
                      "max_ask": 0.92, "min_ask": 0.0, "stop": 0.10, "forced": forced,
                      "max_loss": 0.40, "cycles": 3})
    for z, (tlo, thi), (lo, hi) in (
        (1.0, (60.0, 150.0), (0.55, 0.90)),
        (1.5, (60.0, 150.0), (0.70, 0.95)),
        (1.0, (25.0, 90.0), (0.55, 0.90)),
        (2.0, (25.0, 90.0), (0.80, 0.97)),
    ):
        v.append({"fam": "FAV", "house": False, "tau_lo": tlo, "tau_hi": thi, "z": z,
                  "min_ask": lo, "max_ask": hi, "stop": 0.10, "forced": max(tlo - 10.0, 8.0),
                  "max_loss": 0.40, "cycles": 3})
    # LOCK: model probability from the partly observed settlement TWAP vs ask
    for margin in (0.03, 0.06):
        for lo in (0.60, 0.85):
            v.append({"fam": "LOCK", "margin": margin, "min_ask": lo, "max_ask": 0.985,
                      "tau_lo": 16.0, "tau_hi": 55.0, "exit_tau": 4.0, "stop": 0.15})
    for i, x in enumerate(v):
        x["id"] = i
    return v


class Stats:
    """Exponentially decayed ratio estimator of PnL per $ of eligible volume."""

    def __init__(self, decay: float) -> None:
        self.k = 1.0 - 1.0 / decay
        self.n = 0.0
        self.pnl = 0.0
        self.vol = 0.0
        self.pnl2 = 0.0
        self.pv = 0.0
        self.vol2 = 0.0
        self.raw_n = 0
        self.raw_vol = 0.0
        self.rn = 0.0
        self.ret2 = 0.0
        self.dur = 0.0

    def add(self, pnl: float, vol: float, dur: float, cost: float = 0.0) -> None:
        k = self.k
        if cost > 0:
            ret = pnl / cost
            self.rn = self.rn * k + 1
            self.ret2 = self.ret2 * k + ret * ret
        self.n = self.n * k + 1
        self.pnl = self.pnl * k + pnl
        self.vol = self.vol * k + vol
        self.pnl2 = self.pnl2 * k + pnl * pnl
        self.pv = self.pv * k + pnl * vol
        self.vol2 = self.vol2 * k + vol * vol
        self.dur = self.dur * k + dur
        self.raw_n += 1
        self.raw_vol += vol

    def ret_sd(self) -> float:
        """Root-mean-square return per $ invested (PnL / buy cost)."""
        return math.sqrt(self.ret2 / self.rn) if self.rn > 0 else 0.3

    def rate_per_hour(self, hours: float) -> float:
        """Raw volume per hour of observation (shadow cycles are sequential per variant)."""
        return self.raw_vol / max(hours, 0.25)

    def ratio(self, prior_r: float, prior_w: float) -> tuple[float, float]:
        """(posterior mean cost-per-volume, standard error)."""
        vol = self.vol + prior_w
        r = (self.pnl + prior_r * prior_w) / vol if vol > 0 else prior_r
        if self.n < 2 or self.vol <= 0:
            return r, 1.0
        # residual variance of pnl - r*vol per cycle
        rv = (self.pnl2 - 2 * r * self.pv + r * r * self.vol2) / self.n
        se = math.sqrt(max(rv, 0.0) * self.n) / vol
        return r, se


class Bot:
    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown params {unknown}")
        self.p = {**DEFAULTS, **params}
        self.variants = self.p["variants"] or default_variants()
        self.stats = [Stats(self.p["decay_cycles"]) for _ in self.variants]
        self.shadow: list[dict | None] = [None] * len(self.variants)
        self.cycles_in_market = [0] * len(self.variants)
        self.cool_until = [-math.inf] * len(self.variants)
        self.ref: deque = deque(maxlen=3000)        # (observedAt, price)
        self.var_ps = self.p["sigma_prior"] ** 2
        self.first_t: float | None = None
        self.market_id: str | None = None
        self.market_end = 0.0
        self.target: float | None = None
        self.basis = 0.0            # own reference TWAP minus published target
        self.basis_n = 0
        self.last_mid = 0.5
        self.live_vid: int | None = None
        self.live_entry_t = 0.0
        self.live_entry_ask = 0.0
        self.live_pending_buy = False
        self.had_position = False
        self.live_cash_at_buy = 0.0
        self.live_vol_at_buy = 0.0
        self.live_stats = [Stats(10_000.0) for _ in self.variants]
        self.banned = [False] * len(self.variants)
        self.hour_mark = None
        self.lock = threading.Lock()
        self.log: list = []

    # ================================================================ public
    def decide(self, obs: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            try:
                return self._decide(obs)
            except Exception:
                return {"action": "HOLD"}

    # ================================================================ state
    def _update_reference(self, ref: Any, now: float) -> None:
        if not isinstance(ref, dict):
            return
        try:
            px = float(ref["btcMidUsd"])
            at = float(ref.get("observedAt", now))
        except (KeyError, TypeError, ValueError):
            return
        if not (math.isfinite(px) and px > 0 and math.isfinite(at)) or at > now + 1e-6:
            return
        if self.ref and at <= self.ref[-1][0] + 1e-9:
            return
        if self.ref:
            t0, p0 = self.ref[-1]
            dt = at - t0
            if 0.2 <= dt <= 10:
                r2 = min((px - p0) ** 2 / dt, 25 * self.var_ps)
                a = 1 - 0.5 ** (dt / self.p["sigma_halflife_s"])
                self.var_ps += a * (r2 - self.var_ps)
        self.ref.append((at, px))

    def _ref_at(self, t: float) -> float | None:
        for at, px in reversed(self.ref):
            if at <= t:
                return px
        return None

    def _twap(self, t0: float, t1: float) -> tuple[float | None, float]:
        """Time-weighted mean of the reference over [t0, t1] and covered fraction."""
        pts = []
        for a, p in reversed(self.ref):  # newest first; stop once safely before the window
            if a <= t1:
                pts.append((a, p))
                if a < t0 - 5:
                    break
        pts.reverse()
        if not pts:
            return None, 0.0
        acc = cover = 0.0
        prev_t, prev_p = None, None
        for a, p in pts:
            if prev_t is not None:
                lo, hi = max(prev_t, t0), min(a, t1)
                if hi > lo and a - prev_t <= 5:
                    acc += prev_p * (hi - lo)
                    cover += hi - lo
            prev_t, prev_p = a, p
        if prev_t is not None and prev_t < t1:
            lo = max(prev_t, t0)
            if t1 - prev_t <= 5 and t1 > lo:
                acc += prev_p * (t1 - lo)
                cover += t1 - lo
        if cover <= 0:
            return None, 0.0
        return acc / cover, cover / max(t1 - t0, 1e-9)

    def _sigma(self) -> float:
        return min(max(math.sqrt(self.var_ps), 0.5), 60.0)

    def _new_market(self, mid: str, now: float, tau: float, ref: Any) -> None:
        self.market_id = mid
        self.market_end = now + tau
        self.cycles_in_market = [0] * len(self.variants)
        self.target = None

    def _observe_target(self, ref: Any, now: float) -> None:
        if not isinstance(ref, dict) or ref.get("targetProvisional", True):
            return
        try:
            k = float(ref["openingTargetUsd"])
            kt = float(ref.get("targetObservedAt", now))
        except (KeyError, TypeError, ValueError):
            return
        if not math.isfinite(k) or k <= 0 or kt > now + 1e-6:
            return
        if self.target is None:
            # estimate the reference/target source basis from our own TWAP of the window
            start = self.market_end - 300.0
            own, cover = self._twap(start - 60.0, start)
            if own is not None and cover > 0.8:
                b = own - k
                if abs(b - self.basis) < 300:
                    w = 1.0 / (1 + min(self.basis_n, 20))
                    self.basis += w * (b - self.basis)
                    self.basis_n += 1
        self.target = k

    # ================================================================ model
    @staticmethod
    def _sens(tau: float, sigma: float) -> tuple[float, float]:
        if tau >= 60:
            return sigma * math.sqrt(tau - 40.0), 1.0
        tau = max(tau, 0.5)
        return sigma * math.sqrt(tau ** 3 / 10800.0), tau / 60.0

    def _lag_move(self, lb: float, tau: float, mid: float, sigma: float) -> float:
        if not self.ref:
            return 0.0
        then = self._ref_at(self.ref[-1][0] - lb)
        if then is None:
            return 0.0
        move = self.ref[-1][1] - then
        sd, w = self._sens(tau, sigma)
        z = norm_ppf(min(max(mid, 0.005), 0.995))
        return norm_cdf(z + move * w / sd) - mid

    def _p_up(self, now: float, tau: float, sigma: float) -> float | None:
        """Model probability that the settlement TWAP ends >= target."""
        if self.target is None or not self.ref:
            return None
        k = self.target + (self.basis if self.basis_n >= 3 else 0.0)
        s = self.ref[-1][1]
        if tau >= 60:
            mean, sd = s, sigma * math.sqrt(tau - 40.0)
        else:
            a, cover = self._twap(self.market_end - 60.0, now)
            if a is None or cover < 0.7:
                return None
            f = (60.0 - tau) / 60.0
            mean = f * a + (1 - f) * s
            sd = sigma * math.sqrt(max(tau, 0.5) ** 3 / 10800.0) + 0.3 * sigma
        return norm_cdf((mean - k) / max(sd, 1e-6))

    # ================================================================ variants
    def _entry(self, v, ctx) -> str | None:
        """Side to buy for variant v now, or None."""
        tau = ctx["tau"]
        if not (v["tau_lo"] < tau <= v["tau_hi"]) or tau <= ENTRY_CUTOFF + 0.5:
            return None
        fam = v["fam"]
        if fam == "LAG":
            e = self._lag_move(v["lb"], tau, ctx["mid"], ctx["sigma"])
            if abs(e) < v["edge"]:
                return None
            side = "YES" if e > 0 else "NO"
            asks, bids = ctx["ladders"][side]
            if not asks or not bids or asks[0][0] - bids[0][0] > v["max_spread"] + 1e-9:
                return None
            if not (0.04 <= asks[0][0] <= 0.96):
                return None
            return side
        if fam == "FAV":
            if self.cycles_in_market[v["id"]] >= v["cycles"]:
                return None
            if self.target is None or not self.ref:
                return None
            s = self.ref[-1][1]
            if v["house"]:
                gap = s - self.target
                if abs(gap) < v["gap"]:
                    return None
            else:
                gap = s - self.target - (self.basis if self.basis_n >= 3 else 0.0)
                sd, _ = self._sens(tau, ctx["sigma"])
                if abs(gap) / sd < v["z"]:
                    return None
            side = "YES" if gap >= 0 else "NO"
            asks, bids = ctx["ladders"][side]
            if not asks or not bids:
                return None
            if not (v["min_ask"] <= asks[0][0] <= v["max_ask"]):
                return None
            # immediate round-trip loss cap (House Bot v1)
            sh, g, f = sim_buy(asks, 5.0)
            if sh <= 0:
                return None
            ss, sg, sf = sim_sell(bids, sh)
            if ss < sh - 1e-9 or sg - sf - g - f < -v["max_loss"]:
                return None
            return side
        if fam == "LOCK":
            pu = ctx["p_up"]
            if pu is None:
                return None
            for side, pw in (("YES", pu), ("NO", 1 - pu)):
                asks, bids = ctx["ladders"][side]
                if not asks or not bids:
                    continue
                a = asks[0][0]
                if v["min_ask"] <= a <= v["max_ask"] and pw - a >= v["margin"]:
                    return side
            return None
        return None

    def _exit(self, v, pos, ctx) -> bool:
        """Should a position opened by variant v be sold now?"""
        tau = ctx["tau"]
        side = pos["side"]
        asks, bids = ctx["ladders"][side]
        if not bids:
            return False
        held = ctx["now"] - pos["t"]
        sold_sh, sg, sf = sim_sell(bids, pos["shares"])
        pnl = sg - sf - pos["gross"] - pos["fees"]
        fam = v["fam"]
        if tau <= 5.0:
            return True
        if fam == "LAG":
            if bids[0][0] <= pos["ask"] - v["stop"]:
                return True
            if held >= v["hold"] or held >= 20:
                return True
            return pnl >= 0 and sold_sh >= pos["shares"] - 1e-9 and held >= 1.0
        if fam == "FAV":
            if sold_sh >= pos["shares"] - 1e-9 and pnl >= -1e-9:
                return True
            if pnl <= -v["stop"] * pos["shares"] + 1e-9:
                return True
            return tau <= v["forced"]
        if fam == "LOCK":
            if bids[0][0] <= pos["ask"] - v["stop"]:
                return True
            return tau <= v["exit_tau"] or bids[0][0] >= 0.995
        return True

    # ================================================================ shadows
    def _advance_shadows(self, ctx) -> None:
        now = ctx["now"]
        for i, v in enumerate(self.variants):
            sh = self.shadow[i]
            if sh is None:
                continue
            if sh["state"] == "buy":
                if now < sh["due"]:
                    continue
                if ctx["tau"] <= ENTRY_CUTOFF:
                    self.shadow[i] = None
                    continue
                asks, _ = ctx["ladders"][sh["side"]]
                shares, g, f = sim_buy(asks, sh["budget"])
                if shares < 5.0:
                    self.shadow[i] = None
                    continue
                sh.update(state="open", shares=shares, gross=g, fees=f, t=sh["t"], ask=asks[0][0])
                continue
            if sh["state"] == "open":
                if self._exit(v, sh, ctx):
                    sh["state"] = "sell"
                    sh["due"] = now + LATENCY
                continue
            if sh["state"] == "sell" and now >= sh["due"]:
                _, bids = ctx["ladders"][sh["side"]]
                sold, sg, sf = sim_sell(bids, sh["shares"])
                if sold <= 0:
                    sh["state"] = "open"
                    continue
                sh["shares"] -= sold
                sh["sgross"] = sh.get("sgross", 0.0) + sg
                sh["sfees"] = sh.get("sfees", 0.0) + sf
                if sh["shares"] <= 1e-6:
                    pnl = sh["sgross"] - sh["sfees"] - sh["gross"] - sh["fees"]
                    self.stats[i].add(pnl, sh["gross"] + sh["sgross"], now - sh["t"], sh["gross"] + sh["fees"])
                    self.shadow[i] = None
                    self.cool_until[i] = now + 1.0
                else:
                    sh["state"] = "open"

    def _settle_shadows(self) -> None:
        """Market changed: unsold shadow inventory settles with zero volume."""
        for i, sh in enumerate(self.shadow):
            if sh is None or sh["state"] == "buy":
                self.shadow[i] = None
                continue
            won = (self.last_mid >= 0.5) == (sh["side"] == "YES") if self.p["settle_guess"] else False
            pnl = sh.get("sgross", 0.0) - sh.get("sfees", 0.0) + (sh["shares"] if won else 0.0) \
                - sh["gross"] - sh["fees"]
            self.stats[i].add(pnl, 0.0, 300.0, sh["gross"] + sh["fees"])
            self.shadow[i] = None

    # ================================================================ selection
    def _score(self, i: int) -> tuple[float, float]:
        r, se = self.stats[i].ratio(self.p["prior_r"], self.p["prior_weight_usd"])
        return r, r - self.p["lcb_k"] * se

    def _allowed_cost(self, cash: float, volume: float) -> float:
        need = max(1000.0 - volume, 10.0)
        return max(cash - self.p["reserve_usd"], 0.0) / need

    # ================================================================ main
    def _decide(self, obs: dict[str, Any]) -> dict[str, Any]:
        p = self.p
        now = float(obs["timestamp"])
        if self.first_t is None:
            self.first_t = now
        market = obs["market"]
        account = obs["account"]
        tau = float(market["secondsToClose"])
        mid_id = str(market["id"])
        ref = obs.get("reference")
        self._update_reference(ref, now)
        if mid_id != self.market_id:
            if self.market_id is not None:
                self._settle_shadows()
            self._new_market(mid_id, now, tau, ref)
        self.market_end = now + tau
        self._observe_target(ref, now)

        books = obs.get("books") or {}
        yes, no = books.get("YES") or {}, books.get("NO") or {}
        lad = {
            "YES": (levels(yes.get("asks"), False), levels(yes.get("bids"), True)),
            "NO": (levels(no.get("asks"), False), levels(no.get("bids"), True)),
        }
        ya, yb = lad["YES"]
        if ya and yb:
            mid = (ya[0][0] + yb[0][0]) / 2
        elif yb:
            mid = max(yb[0][0], 0.5)
        elif ya:
            mid = min(ya[0][0], 0.5)
        else:
            mid = self.last_mid
        self.last_mid = mid
        sigma = self._sigma()
        ref_ok = bool(self.ref) and now - self.ref[-1][0] <= p["max_ref_age"]
        ctx = {"now": now, "tau": tau, "mid": mid, "sigma": sigma, "ladders": lad,
               "p_up": self._p_up(now, tau, sigma) if ref_ok else None}

        # ---- shadows: progress existing, then open new ones
        self._advance_shadows(ctx)
        if ref_ok and ya and yb:
            for i, v in enumerate(self.variants):
                if self.shadow[i] is None and now >= self.cool_until[i]:
                    side = self._entry(v, ctx)
                    if side:
                        self.shadow[i] = {"state": "buy", "side": side, "due": now + LATENCY,
                                          "budget": 5.0, "t": now}
                        self.cycles_in_market[i] += 1

        if not p["live"]:
            return {"action": "HOLD"}

        # ---- live position management
        position = account.get("position")
        cash_now = float(account.get("cashUsd", 0))
        vol_now = float(account.get("eligibleVolumeUsd", 0))
        if position is None and self.had_position:
            if self.live_vid is not None:
                i = self.live_vid
                self.live_stats[i].add(cash_now - self.live_cash_at_buy, vol_now - self.live_vol_at_buy, 0.0)
                ls = self.live_stats[i]
                if ls.raw_n >= 2 and ls.pnl < -p["ban_live_loss"]:
                    self.banned[i] = True
                if ls.raw_n >= p["ban_live_cycles"] and ls.vol > 0:
                    if ls.pnl / ls.vol < -self._allowed_cost(cash_now, vol_now) - p["ban_margin"]:
                        self.banned[i] = True
            self.live_vid = None
        self.had_position = position is not None
        if position is not None:
            side = str(position.get("outcome"))
            shares = float(position.get("shares", 0))
            if shares <= 0:
                return {"action": "HOLD"}
            vid = self.live_vid if self.live_vid is not None else 0
            pos = {"side": side, "shares": shares, "t": self.live_entry_t, "ask": self.live_entry_ask,
                   "gross": float(position.get("buy_gross_usd", 0)) - float(position.get("sell_gross_usd", 0)),
                   "fees": float(position.get("buy_fees_usd", 0)) - float(position.get("sell_fees_usd", 0))}
            if self._exit(self.variants[vid], pos, ctx):
                return {"action": "SELL"}
            return {"action": "HOLD"}

        # ---- live entry: best approved variant whose signal fires now
        if not ref_ok or tau <= ENTRY_CUTOFF + 0.5:
            return {"action": "HOLD"}
        cash = cash_now
        volume = vol_now
        allowed = self._allowed_cost(cash, volume)
        hours = (now - self.first_t) / 3600.0
        need_rate = max(1000.0 - volume, 0.0) / max(p["pace_hours"] - hours, 1.0)
        plan = 1000.0 * min(1.0, hours / p["pace_hours"])
        # rescue: not qualifying scores nothing, so late and far behind, use the mean
        # estimate with a looser budget instead of the confidence bound
        rescue = hours >= p["rescue_hour"] and volume < p["rescue_behind"] * plan
        cands = []
        for i, v in enumerate(self.variants):
            if self.banned[i] or self.stats[i].raw_n < p["min_cycles"] or self.stats[i].raw_vol < p["min_shadow_vol"]:
                continue
            r, lcb = self._score(i)
            if rescue:
                lcb = r
                if r < -1.5 * allowed - 0.003:
                    continue
            elif lcb < -allowed:
                continue
            if cash < p["brake_cash"] and lcb < 0:
                continue
            cands.append((lcb, i, self.stats[i].rate_per_hour(hours)))
        cands.sort(reverse=True)
        # smallest prefix (best first) that is fast enough to meet the schedule
        chosen = []
        fastest = 0.0
        for lcb, i, rate in cands:
            chosen.append(i)
            fastest = max(fastest, rate)
            if fastest >= need_rate * p["rate_safety"]:
                break
        best = None
        for i in chosen:  # chosen is ordered best first
            side = self._entry(self.variants[i], ctx)
            if side:
                best = (0.0, i, side)
                break
        if best is None:
            return {"action": "HOLD"}
        _, i, side = best
        rsd = max(self.stats[i].ret_sd(), 0.02)
        budget = min(5.0, cash - 0.02, p["risk_frac"] * cash / rsd)
        asks, _ = lad[side]
        sh, g, f = sim_buy(asks, budget)
        if sh < 5.0 * 1.01:
            return {"action": "HOLD"}
        self.live_vid = i
        self.live_cash_at_buy = cash
        self.live_vol_at_buy = volume
        self.live_entry_t = now
        self.live_entry_ask = asks[0][0]
        return {"action": "BUY", "outcome": side, "maxCashUsd": round(budget, 4)}

    # ================================================================ diagnostics
    def report(self) -> list[dict[str, Any]]:
        out = []
        for i, v in enumerate(self.variants):
            r, lcb = self._score(i)
            s = self.stats[i]
            ls = self.live_stats[i]
            out.append({"id": i, "fam": v["fam"], "n": s.raw_n, "r": round(r, 4), "lcb": round(lcb, 4),
                        "live_n": ls.raw_n, "live_pnl": round(ls.pnl, 3), "banned": self.banned[i],
                        "vol": round(s.vol, 1), "pnl": round(s.pnl, 3),
                        "v": {k: x for k, x in v.items() if k not in ("fam", "id")}})
        return out


