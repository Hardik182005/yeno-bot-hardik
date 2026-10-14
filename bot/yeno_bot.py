#!/usr/bin/env python3
"""YeNo x Builderr volume bot - core mathematical foundations.
"""
from __future__ import annotations
import math
from typing import Any

OVERLAY = 0.01
PROTOCOL = 0.07
LATENCY = 0.25
ENTRY_CUTOFF = 15.0
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

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

def default_variants() -> list[dict[str, Any]]:
    v: list[dict[str, Any]] = []
    # LAG
    for lb in (1.5, 3.0):
        for edge in (0.06, 0.12, 0.2, 0.3, 0.45):
            for hold in (2.0, 5.0):
                v.append({"fam": "LAG", "lb": lb, "edge": edge, "hold": hold,
                          "tau_lo": 20.0, "tau_hi": 280.0, "stop": 0.10, "max_spread": 0.03})
    # FAV
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
    # LOCK: settlement-TWAP model probability
    for margin in (0.03, 0.06):
        for lo in (0.60, 0.85):
            v.append({"fam": "LOCK", "margin": margin, "min_ask": lo, "max_ask": 0.985,
                      "tau_lo": 16.0, "tau_hi": 55.0, "exit_tau": 4.0, "stop": 0.15})
    for i, x in enumerate(v):
        x["id"] = i
    return v

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
