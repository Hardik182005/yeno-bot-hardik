"""Run a bot through the local evaluator.

    python sim/run.py BOT_FILE [--data FILE ...] [--paths N] [--adverse 0.02]

BOT_FILE must expose either `make_bot()` returning an object with `.decide`,
a module-level `BOT` with `.decide`, or a module-level `decide` function.
A fresh bot is created for every path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.evaluator import run  # noqa: E402
from sim import scenarios  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sim"
_MARKETS: list = []


def load_bot(path: str, overrides: dict | None = None):
    spec = importlib.util.spec_from_file_location(f"bot_{abs(hash((path, json.dumps(overrides, sort_keys=True))))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_bot"):
        return mod.make_bot(**(overrides or {})).decide
    if hasattr(mod, "BOT"):
        return mod.BOT.decide
    return mod.decide


def load_markets(files):
    out = []
    for f in files:
        with open(f, "rb") as fh:
            out.extend(pickle.load(fh))
    out.sort(key=lambda m: m["start"])
    return out


def contiguous_days(markets):
    """Split into chronological runs of up to 288 consecutive markets (24 h)."""
    days, cur = [], []
    for m in markets:
        if cur and (m["start"] - cur[-1]["start"] > 600 or len(cur) == 288):
            days.append(cur)
            cur = []
        cur.append(m)
    if cur:
        days.append(cur)
    return days


def resampled_paths(markets, n, seed=7, block=12):
    """Block bootstrap: 24 h paths built from 1 h blocks of consecutive markets."""
    rng = random.Random(seed)
    blocks = []
    for d in contiguous_days(markets):
        for i in range(0, len(d) - block + 1, block):
            blocks.append(d[i:i + block])
    paths = []
    for _ in range(n):
        chosen = [rng.choice(blocks) for _ in range(288 // block)]
        path, t = [], 0.0
        for b in chosen:  # re-time blocks so the path is chronological
            base = b[0]["start"]
            for m in b:
                shift = t + (m["start"] - base) - m["start"]
                path.append(_shift(m, shift))
            t += 300 * len(b)
        paths.append(path)
    return paths


def _shift(m, dt):
    if dt == 0:
        return m
    s = dict(m)
    s["start"], s["end"] = m["start"] + dt, m["end"] + dt
    s["t"] = m["t"] + dt
    s["ref_t"] = m["ref_t"] + dt
    if m.get("target"):
        s["target"] = (m["target"][0] + dt, m["target"][1])
    s["slug"] = f"{m['slug']}@{int(dt)}"
    return s


def _job(args):
    bot_file, overrides, path, kw = args
    decide = load_bot(bot_file, overrides)
    r = run(path, decide, **kw)
    owner = getattr(decide, "__self__", None)
    if owner is not None and hasattr(owner, "report"):
        r.bot_report = owner.report()
    return r


def evaluate(bot_file, paths, overrides=None, workers=8, **kw):
    with Pool(workers) as pool:
        return pool.map(_job, [(bot_file, overrides, p, kw) for p in paths])


def report(results, label=""):
    vols = [r.volume for r in results]
    q = [r for r in results if r.qualified]
    cash_q = [r.cash for r in q]
    print(f"== {label} paths={len(results)} qualified={len(q)} "
          f"medianVol={statistics.median(vols):.1f} bestVol={max(vols):.1f} "
          f"medianCash={statistics.median(r.cash for r in results):.2f} "
          + (f"qualCash med={statistics.median(cash_q):.2f} min={min(cash_q):.2f} max={max(cash_q):.2f} "
             f"hours med={statistics.median(r.time_to_target_h for r in q):.1f}" if q else ""))
    return {"qualified": len(q), "n": len(results), "medianVol": statistics.median(vols),
            "medianQualCash": statistics.median(cash_q) if q else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bot")
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--paths", type=int, default=0, help="0 = real contiguous days only")
    ap.add_argument("--adverse", type=float, default=0.0)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--show-variants", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override KEY=VALUE for make_bot")
    ap.add_argument("--scenario", default="bnf", help="cl|bnf|bns|bnf_k, optionally @DELAY")
    args = ap.parse_args()
    files = args.data or sorted(str(p) for p in DATA_DIR.glob("*.pkl"))
    markets = scenarios.apply(load_markets(files), args.scenario)
    if markets is None:  # scenario needs Chainlink data: restrict to datasets that have it
        markets = scenarios.apply([m for m in load_markets(files) if "cl_t" in m], args.scenario)
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    paths = resampled_paths(markets, args.paths) if args.paths else contiguous_days(markets)
    res = evaluate(args.bot, paths, overrides or None, adverse=args.adverse,
                   query_interval=args.interval, stop_at_target=not args.no_stop)
    for p, r in zip(paths, res):
        print(p[0]["slug"], len(p), json.dumps(r.summary()))
        if args.show_variants and getattr(r, "bot_report", None):
            for row in sorted(r.bot_report, key=lambda x: -x["r"]):
                print("   ", json.dumps(row))
    report(res, f"{Path(args.bot).name} [{args.scenario}]")


if __name__ == "__main__":
    main()
