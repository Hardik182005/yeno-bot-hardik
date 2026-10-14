"""Convert polyorderbooks 250 ms L2 snapshots (BTC 5m) into sim markets.

Usage: python sim/prep_polyob.py RAW_PARQUET OUT_FILE

Only the Up (YES) token ladders are stored; NO is mirrored in the evaluator.
BTC reference comes from Binance 1 s closes (sim/binance.py).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.binance import attach_reference  # noqa: E402

LEVELS = 12


def pad(lists, n):
    out = np.full((len(lists), n), np.nan, np.float32)
    for i, l in enumerate(lists):
        if l is not None and len(l):
            k = min(n, len(l))
            out[i, :k] = l[:k]
    return out


def main():
    raw, out_file = sys.argv[1], sys.argv[2]
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    df = con.execute(
        f"""select market_slug, epoch_us(captured_at)/1e6 t, epoch_us(market_end_at)/1e6 e, winner,
                   bid_prices, bid_sizes, ask_prices, ask_sizes
            from read_parquet('{raw}') where outcome = 'Up' order by market_slug, captured_at"""
    ).df()
    markets = []
    for slug, g in df.groupby("market_slug", sort=False):
        end = float(g.e.iloc[0])
        start = end - 300
        g = g[(g.t >= start) & (g.t < end)]
        if len(g) < 200:
            continue
        markets.append({
            "slug": slug, "start": start, "end": end,
            "winner": "YES" if g.winner.iloc[0] == "yes" else "NO",
            "t": g.t.to_numpy(np.float64),
            "bp": pad(g.bid_prices.tolist(), LEVELS), "bs": pad(g.bid_sizes.tolist(), LEVELS),
            "ap": pad(g.ask_prices.tolist(), LEVELS), "as": pad(g.ask_sizes.tolist(), LEVELS),
        })
    markets.sort(key=lambda m: m["start"])
    attach_reference(markets)
    with open(out_file, "wb") as f:
        pickle.dump(markets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(markets)} markets -> {out_file}")


if __name__ == "__main__":
    main()
