"""Realised round-trip economics on tick-accurate data.

For every decision time t (0.5 s grid), and both outcomes, compute:
  fill  = side's best ask at the first book state >= t + 0.25 (BUY)
  exit  = side's best bid at the first state >= t + 0.25 + H  (SELL, after H s)
  net   = per-share PnL after both fee layers
alongside causal signals available at t. Saved for analysis.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim import scenarios  # noqa: E402

HS = [1, 2, 3, 5, 8, 12, 20]


def fee(p):
    return 0.07 * p * (1 - p) + 0.01 * p


def main():
    files = sys.argv[2:]
    mode = sys.argv[1]
    rows = []
    for f in files:
        ms = scenarios.apply(pickle.load(open(f, "rb")), mode)
        for m in ms:
            T = m["t"]
            yb, ya = m["bp"][:, 0].astype(float), m["ap"][:, 0].astype(float)
            ybs, yas = m["bs"][:, 0].astype(float), m["as"][:, 0].astype(float)
            rt, rp = m["ref_t"], m["ref_p"]
            grid = np.arange(m["start"] + 10, m["end"] - 18, 0.5)
            i0 = np.searchsorted(T, grid, side="right") - 1
            ok = i0 >= 0
            grid, i0 = grid[ok], i0[ok]
            jf = np.searchsorted(T, grid + 0.25, side="left")
            jr = np.searchsorted(rt, grid, side="right") - 1
            for k, lb in enumerate((1.0, 2.0, 3.0, 5.0)):
                pass
            d = {
                "slug": m["slug"], "tau": m["end"] - grid,
                "yb": yb[i0], "ya": ya[i0], "ybs": ybs[i0], "yas": yas[i0],
                "ref_age": grid - rt[np.maximum(jr, 0)],
                "S": rp[np.maximum(jr, 0)],
            }
            for lb in (1.0, 2.0, 3.0, 5.0, 10.0):
                jl = np.searchsorted(rt, rt[np.maximum(jr, 0)] - lb, side="right") - 1
                d[f"dS{int(lb)}"] = rp[np.maximum(jr, 0)] - rp[np.maximum(jl, 0)]
            # past book move (did the book already react?)
            for lb in (1.0, 2.0, 5.0):
                jb = np.searchsorted(T, grid - lb, side="right") - 1
                d[f"dm{int(lb)}"] = (yb[i0] + ya[i0]) / 2 - (yb[np.maximum(jb, 0)] + ya[np.maximum(jb, 0)]) / 2
            jf_ok = jf < len(T)
            jf = np.minimum(jf, len(T) - 1)
            d["fya"], d["fyb"] = ya[jf], yb[jf]  # YES ask / bid at fill
            for H in HS:
                je = np.searchsorted(T, grid + 0.25 + H, side="left")
                bad = (je >= len(T)) | ~jf_ok
                je = np.minimum(je, len(T) - 1)
                eyb, eya = yb[je], ya[je]
                # YES: buy at YES ask, sell at YES bid; NO: buy at 1-YES bid, sell at 1-YES ask
                ny = eyb - d["fya"] - fee(d["fya"]) - fee(eyb)
                nn = (1 - eya) - (1 - d["fyb"]) - fee(1 - d["fyb"]) - fee(1 - eya)
                ny[bad] = np.nan
                nn[bad] = np.nan
                d[f"ny{H}"], d[f"nn{H}"] = ny, nn
            rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    out = Path(__file__).resolve().parent.parent.parent / "data" / f"roundtrips_{mode}.parquet"
    df.to_parquet(out)
    print(df.shape, "->", out)


if __name__ == "__main__":
    main()
