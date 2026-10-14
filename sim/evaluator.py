"""Local re-implementation of the YeNo x Builderr paper evaluator.
Order book level parsing, ladder walking, and exact fee models.
"""
from __future__ import annotations
import math

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
    """Exact fee calculation matching evaluator contract (5-decimal rounding)."""
    gross = sum(p * q for p, q in fills)
    protocol = sum(PROTOCOL * q * p * (1 - p) for p, q in fills)
    return round(protocol + 1e-12, 5) + round(gross * OVERLAY + 1e-12, 5)

def unit_cost(p: float) -> float:
    return p + PROTOCOL * p * (1 - p) + OVERLAY * p

def fill_buy(asks, budget: float, adverse: float = 0.0):
    """Spend up to budget (fee inclusive) walking the asks. Returns fills."""
    fills: list[tuple[float, float]] = []
    left = budget
    for price, size in asks:
        p = min(0.999, price + adverse)
        uc = unit_cost(p)
        q = min(size, left / uc)
        q = math.floor(q * 100 + 1e-9) / 100
        if q <= 0:
            break
        fills.append((p, q))
        left -= q * uc
        if q < size:
            break
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
