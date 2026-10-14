"""Fetch and cache Binance BTCUSDT 1-second closes (used where a dataset has no BTC reference)."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "binance"
URL = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1s&startTime={}&limit=1000"


def closes(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (second, close) arrays for [start, end) unix seconds, hour-cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    secs, px = [], []
    for h in range(start - start % 3600, end, 3600):
        f = CACHE / f"{h}.npy"
        if not f.exists():
            rows = []
            t = h
            while t < h + 3600:
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen(URL.format(t * 1000), timeout=20) as r:
                            batch = json.loads(r.read())
                        break
                    except Exception:
                        time.sleep(1 + attempt)
                else:
                    raise RuntimeError(f"binance fetch failed at {t}")
                batch = [b for b in batch if b[0] // 1000 < h + 3600]
                if not batch:
                    break
                rows.extend((b[0] // 1000, float(b[4])) for b in batch)
                t = batch[-1][0] // 1000 + 1
            np.save(f, np.array(rows, dtype=np.float64).reshape(-1, 2))
        a = np.load(f)
        secs.append(a[:, 0])
        px.append(a[:, 1])
    s, p = np.concatenate(secs), np.concatenate(px)
    keep = (s >= start) & (s < end)
    return s[keep], p[keep]


def attach_reference(markets: list[dict], recv_lag: float = 1.1) -> None:
    """Give each market a Chainlink-like reference stream and a 60 s TWAP opening target.

    A Binance 1 s close for second s is treated as observed at s+1 and received
    at s+1+recv_lag (Chainlink reports in the PMData day arrive ~1.1 s late).
    """
    lo = int(min(m["start"] for m in markets)) - 120
    hi = int(max(m["end"] for m in markets)) + 5
    s, p = closes(lo, hi)
    recv = s + 1 + recv_lag
    for m in markets:
        sel = (recv >= m["start"] - 90) & (recv < m["end"])
        m["ref_t"], m["ref_p"] = recv[sel], p[sel]
        w = (s >= m["start"] - 60) & (s < m["start"])
        m["target"] = (m["start"] + 1 + recv_lag, float(p[w].mean())) if w.sum() >= 50 else None
        w2 = (s >= m["end"] - 60) & (s < m["end"])
        m["twap_end"] = float(p[w2].mean()) if w2.sum() >= 50 else None


def attach_raw(markets: list[dict]) -> None:
    """Store raw Binance 1 s closes per market so reference scenarios can be built later."""
    lo = int(min(m["start"] for m in markets)) - 120
    hi = int(max(m["end"] for m in markets)) + 5
    s, p = closes(lo, hi)
    for m in markets:
        sel = (s >= m["start"] - 120) & (s < m["end"])
        m["bn_s"], m["bn_p"] = s[sel], p[sel]
