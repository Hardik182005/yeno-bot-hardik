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
