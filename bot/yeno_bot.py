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
