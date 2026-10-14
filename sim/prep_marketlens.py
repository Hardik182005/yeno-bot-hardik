"""Convert the marketlens one-day dataset (snapshots + level deltas) into sim markets.

Usage: python sim/prep_marketlens.py RAW_DIR OUT_FILE

Book = latest snapshot plus its delta chain (absolute level replacement), keyed by
exchange timestamp (+50 ms receive latency), sampled on a 100 ms grid. Crossed
states (stale levels the feed never removed) are dropped. Reference: Binance 1 s.
"""

from __future__ import annotations

import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.binance import attach_raw  # noqa: E402
from sim.prep_pmdata import GRID, _ladder  # noqa: E402

RECV = 0.05


def build(args):
    mid, start, end, winner, raw = args
    con = duckdb.connect()
    snaps = con.execute(
        f"""select collected_at_ms/1e3, snapshot_id, bids, asks from read_parquet('{raw}/snapshots.parquet')
            where market_id = ? order by collected_at_ms""", [mid]).fetchall()
    deltas = con.execute(
        f"""select platform_timestamp_ms/1e3 + {RECV}, snapshot_id, price, size, side
            from read_parquet('{raw}/deltas.parquet') where market_id = ?
            order by platform_timestamp_ms""", [mid]).fetchall()
    ev = [(t, 0, sid, b, a) for t, sid, b, a in snaps] + [(t, 1, sid, p, s, side) for t, sid, p, s, side in deltas]
    ev.sort(key=lambda x: (x[0], x[1]))
    bids, asks = {}, {}
    cur_sid = None
    out_t, out = [], []
    bucket = None
    last = None

    def emit(tb):
        nonlocal last
        bp, bs = _ladder(bids, True, None)
        ap, as_ = _ladder(asks, False, None)
        if bp[0] == bp[0] and ap[0] == ap[0] and bp[0] >= ap[0]:
            return  # crossed: stale level
        sig = (bp.tobytes(), bs.tobytes(), ap.tobytes(), as_.tobytes())
        if sig != last:
            last = sig
            out_t.append(tb)
            out.append((bp, bs, ap, as_))

    for e in ev:
        t = e[0]
        if t >= end:
            break
        b = int((t - start) // GRID)
        if bucket is not None and b != bucket and t > start:
            tb = start + (bucket + 1) * GRID
            if tb > start:
                emit(min(tb, t))
        bucket = b
        if e[1] == 0:
            cur_sid = e[2]
            bids = {x["price"]: x["size"] for x in (e[3] or []) if x["size"] > 0}
            asks = {x["price"]: x["size"] for x in (e[4] or []) if x["size"] > 0}
        elif e[2] == cur_sid:
            side = bids if e[5] == "BUY" else asks
            if e[4] <= 0:
                side.pop(e[3], None)
            else:
                side[e[3]] = e[4]
    if len(out) < 100:
        return None
    arr = [np.stack([o[k] for o in out]) for k in range(4)]
    return {"slug": f"ml-{int(start)}", "start": start, "end": end, "winner": winner,
            "t": np.array(out_t), "bp": arr[0], "bs": arr[1], "ap": arr[2], "as": arr[3]}


def main():
    raw, out_file = sys.argv[1], sys.argv[2]
    con = duckdb.connect()
    mk = con.execute(f"""select market_id, open_time_ms/1e3, close_time_ms/1e3, winning_outcome
                         from read_parquet('{raw}/markets.parquet') order by open_time_ms""").fetchall()
    jobs = [(m, a, b, "YES" if w == "Up" else "NO", raw) for m, a, b, w in mk]
    with Pool(8) as pool:
        markets = [m for m in pool.map(build, jobs, chunksize=4) if m is not None]
    attach_raw(markets)
    with open(out_file, "wb") as f:
        pickle.dump(markets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(markets)} markets -> {out_file}")


if __name__ == "__main__":
    main()
