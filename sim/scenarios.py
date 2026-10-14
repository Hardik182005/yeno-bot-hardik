"""Reference-feed scenarios. The official feed's source and latency are not published,
so every candidate bot is tested under all of these.

  cl     Chainlink Data Streams spot as received (~1.1 s late) + Chainlink 60 s TWAP target
  bnf    Binance 1 s close received 0.15 s after the second ends + Binance TWAP target
  bns    Binance 1 s close received 1.1 s after the second ends + Binance TWAP target
  bnf_k  Binance fast reference but Chainlink TWAP target (source mismatch / basis risk)
"""

from __future__ import annotations

import numpy as np

SCENARIOS = ("cl", "bnf", "bns", "bnf_k")


def _bn(m, lag):
    s, p = m["bn_s"], m["bn_p"]
    recv = s + 1 + lag
    sel = (recv >= m["start"] - 90) & (recv < m["end"])
    w = (s >= m["start"] - 60) & (s < m["start"])
    tgt = (m["start"] + 1 + lag + 0.05, float(p[w].mean())) if w.sum() >= 50 else None
    return recv[sel], p[sel], tgt


def apply(markets, mode):
    """mode is a scenario name, optionally '@D' to delay every book state by D seconds
    (models a book recorder that is slower than the BTC reference feed)."""
    delay = 0.0
    if "@" in mode:
        mode, d = mode.split("@")
        delay = float(d)
    out = []
    for m in markets:
        has_cl = "cl_t" in m
        if mode == "cl":
            if not has_cl:
                return None
            ref_t, ref_p, tgt = m["cl_t"], m["cl_p"], m["cl_target"]
        elif mode in ("bnf", "bns"):
            ref_t, ref_p, tgt = _bn(m, 0.15 if mode == "bnf" else 1.1)
        elif mode == "bnf_k":
            if not has_cl:
                return None
            ref_t, ref_p, _ = _bn(m, 0.15)
            tgt = m["cl_target"]
        else:
            raise ValueError(mode)
        s = dict(m)
        s["ref_t"], s["ref_p"], s["target"] = ref_t, ref_p, tgt
        if delay:
            keep = m["t"] + delay < m["end"]
            s["t"] = m["t"][keep] + delay
            for k in ("bp", "bs", "ap", "as"):
                s[k] = m[k][keep]
        out.append(s)
    return out
