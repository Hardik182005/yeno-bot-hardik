"""Build a research table: one row per (market, decision time) with book, reference and outcome."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "sim"
TAUS = list(range(290, 60, -2)) + list(range(60, 0, -1))  # seconds to close


def at(arr_t, t):
    return np.searchsorted(arr_t, t, side="right") - 1


def rows_for(m):
    out = []
    K = m["target"][1] if m["target"] else np.nan
    rt, rp = m["ref_t"], m["ref_p"]
    # 1 s returns of the reference received so far (for a causal volatility estimate)
    for tau in TAUS:
        t = m["end"] - tau
        i = at(m["t"], t)
        if i < 0:
            continue
        j = at(rt, t)
        if j < 1:
            continue
        S = rp[j]
        # causal vol: std of reference changes over the trailing 180 s
        w = (rt > t - 180) & (rt <= t)
        d = np.diff(rp[w])
        sig = float(np.sqrt(np.mean(d * d))) if len(d) > 20 else np.nan
        # partial TWAP over [end-60, t] using received values (step function)
        if tau < 60:
            ww = (rt > m["end"] - 60) & (rt <= t)
            A = float(rp[ww].mean()) if ww.sum() > 0 else S
        else:
            A = np.nan
        out.append(dict(
            slug=m["slug"], start=m["start"], tau=tau, S=S, K=K, sig=sig, A=A,
            ref_age=t - rt[j],
            yb=float(m["bp"][i, 0]), ya=float(m["ap"][i, 0]),
            ybs=float(m["bs"][i, 0]), yas=float(m["as"][i, 0]),
            win=1.0 if m["winner"] == "YES" else 0.0,
        ))
    return out


def main():
    files = sys.argv[1:] or sorted(str(p) for p in DATA.glob("*.pkl"))
    rows = []
    for f in files:
        for m in pickle.load(open(f, "rb")):
            r = rows_for(m)
            for x in r:
                x["src"] = Path(f).stem
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_parquet(DATA.parent / "research_rows.parquet")
    print(df.shape)


if __name__ == "__main__":
    main()
