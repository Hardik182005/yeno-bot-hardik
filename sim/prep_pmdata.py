"""Convert the PMData tick dataset (one day, Chainlink included) into sim markets.

Usage: python sim/prep_pmdata.py RAW_DIR OUT_FILE

The L2 stream carries the Up (YES) token book; the Down (NO) book is its exact
mirror on Polymarket's CLOB, so only YES ladders are stored. The book is
replayed tick by tick and sampled on a GRID-second receive-time grid (the last
state inside each bucket), between market open and close.
"""

from __future__ import annotations

import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np

GRID = 0.1
LEVELS = 12


def _epoch(col: str) -> str:
    return f"epoch_us({col})/1e6"


def _ladder(levels: dict, reverse: bool, cut: float | None):
    items = sorted(levels.items(), reverse=reverse)
    if cut is not None:  # drop stale levels on the wrong side of the exchange BBO
        items = [(p, s) for p, s in items if (p <= cut + 1e-9 if reverse else p >= cut - 1e-9)]
    items = items[:LEVELS]
    p = np.full(LEVELS, np.nan, np.float32)
    s = np.full(LEVELS, np.nan, np.float32)
    for i, (pp, ss) in enumerate(items):
        p[i], s[i] = pp, ss
    return p, s


def build_market(args):
    slug, start, end, winner, raw = args
    con = duckdb.connect()
    rows = con.execute(
        f"""select {_epoch('local_timestamp')} t, event_type, bid_prices, bid_sizes, ask_prices, ask_sizes,
                   pc_price, pc_size, pc_side, best_bid, best_ask
            from read_parquet('{raw}/l2/*/*.parquet') where market_slug = ?
            order by local_timestamp""",
        [slug],
    ).fetchall()
    bids: dict = {}
    asks: dict = {}
    bbo_bid = bbo_ask = None
    out_t, out = [], []
    bucket = None
    last_sig = None

    def emit(tb):
        nonlocal last_sig
        bp, bs = _ladder(bids, True, bbo_bid)
        ap, as_ = _ladder(asks, False, bbo_ask)
        sig = (bp.tobytes(), bs.tobytes(), ap.tobytes(), as_.tobytes())
        if sig != last_sig:
            last_sig = sig
            out_t.append(tb)
            out.append((bp, bs, ap, as_))

    for t, ev, bpr, bsz, apr, asz, pcp, pcs, pcside, bb, ba in rows:
        if t >= end:
            break
        b = int((t - start) // GRID)
        if bucket is not None and b != bucket and t > start:
            tb = start + (bucket + 1) * GRID  # state as of the end of the bucket
            if tb > start:
                emit(min(tb, t))
        bucket = b
        if ev == "book":
            bids = {p: s for p, s in zip(bpr or [], bsz or []) if s > 0}
            asks = {p: s for p, s in zip(apr or [], asz or []) if s > 0}
            bbo_bid = bbo_ask = None
        elif ev == "price_change":
            side = bids if pcside == "BUY" else asks
            if pcs <= 0:
                side.pop(pcp, None)
            else:
                side[pcp] = pcs
            bbo_bid = bb if bb is not None and bb > 0 else None
            bbo_ask = ba if ba is not None and ba < 1 else None
    if not out:
        return None
    arr = [np.stack([o[k] for o in out]) for k in range(4)]
    return {
        "slug": slug, "start": start, "end": end, "winner": winner,
        "t": np.array(out_t), "bp": arr[0], "bs": arr[1], "ap": arr[2], "as": arr[3],
    }


def main():
    RAW = Path(sys.argv[1])
    out_file = Path(sys.argv[2])
    con = duckdb.connect()
    mk = con.execute(
        f"select market_slug, {_epoch('start_time')}, {_epoch('end_time')}, winning_outcome "
        f"from read_parquet('{RAW}/markets.parquet') order by start_time"
    ).fetchall()
    # Chainlink spot (receive time, price) and 60 s TWAP keyed by observation time
    cl = con.execute(
        f"""select {_epoch('receiveMicrosecondTimestamp')} r, {_epoch('observationsTimestamp')} o,
                   cast(price as double)/1e18 p
            from read_parquet('{RAW}/chainlink_streams/*/*.parquet') order by r"""
    ).fetchnumpy()
    tw = con.execute(
        f"""select {_epoch('observationsTimestamp')} o, min({_epoch('receiveMicrosecondTimestamp')}) r,
                   last(cast(price as double)/1e18) p
            from read_parquet('{RAW}/chainlink_streams_twap60s/*/*.parquet') group by 1 order by 1"""
    ).fetchnumpy()
    jobs = [(s, a, b, "YES" if w == "yes" else "NO", str(RAW)) for s, a, b, w in mk]
    with Pool(8) as pool:
        markets = [m for m in pool.map(build_market, jobs, chunksize=4) if m is not None]
    for m in markets:
        sel = (cl["r"] >= m["start"] - 90) & (cl["r"] < m["end"])
        m["ref_t"] = cl["r"][sel]
        m["ref_p"] = cl["p"][sel]
        i = np.searchsorted(tw["o"], m["start"], side="right") - 1
        m["target"] = (float(tw["r"][i]), float(tw["p"][i])) if i >= 0 and tw["o"][i] == m["start"] else None
        # TWAP at the end, for diagnostics only (the bot never sees it)
        j = np.searchsorted(tw["o"], m["end"], side="right") - 1
        m["twap_end"] = float(tw["p"][j]) if j >= 0 else None
    with open(out_file, "wb") as f:
        pickle.dump(markets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(markets)} markets, {sum(len(m['t']) for m in markets)} book states -> {out_file}")


if __name__ == "__main__":
    main()
