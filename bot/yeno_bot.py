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
