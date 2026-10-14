"""Contract and robustness tests for bot/yeno_bot.py (standard library only).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "yeno-sample-observation.json").read_text())


def load():
    spec = importlib.util.spec_from_file_location("yeno_bot", ROOT / "bot" / "yeno_bot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


def valid(decision, cash=10.0, has_position=False):
    if not isinstance(decision, dict):
        return False
    a = decision.get("action")
    if a == "HOLD":
        return True
    if a == "SELL":
        return has_position
    if a == "BUY":
        mc = decision.get("maxCashUsd")
        return (not has_position and decision.get("outcome") in ("YES", "NO")
                and isinstance(mc, (int, float)) and 0 < mc <= 5 and mc <= cash + 1e-9)
    return False


def stream(n=3000, seed=1, start=1789880000.0):
    """Synthetic but contract-shaped observations: random-walk BTC, moving books, markets roll."""
    rng = random.Random(seed)
    btc = 80000.0
    t = start
    for _ in range(n):
        t += rng.choice((0.1, 0.25, 0.5))
        btc += rng.gauss(0, 2.5)
        market_start = t - (t % 300)
        tau = 300 - (t - market_start)
        mid = min(0.97, max(0.03, 0.5 + rng.gauss(0, 0.2)))
        bid, ask = round(mid - 0.005, 2), round(mid + 0.005, 2)
        if ask <= bid:
            ask = round(bid + 0.01, 2)
        obs = copy.deepcopy(SAMPLE)
        obs["timestamp"] = t
        obs["market"] = {"id": f"m{int(market_start)}", "secondsToClose": round(tau, 3)}
        obs["reference"] = {"btcMidUsd": btc, "observedAt": t - 0.3, "openingTargetUsd": 80000.0,
                            "targetObservedAt": market_start + 1, "targetProvisional": False}
        obs["books"] = {
            "YES": {"bids": [[bid, 100]], "asks": [[ask, 100]], "minOrderSize": 5},
            "NO": {"bids": [[round(1 - ask, 2), 100]], "asks": [[round(1 - bid, 2), 100]], "minOrderSize": 5},
        }
        yield obs


class ContractTests(unittest.TestCase):
    def test_sample_observation(self):
        bot = MOD.make_bot()
        self.assertTrue(valid(bot.decide(copy.deepcopy(SAMPLE))))

    def test_malformed_inputs_hold(self):
        bot = MOD.make_bot()
        for bad in ({}, {"timestamp": "x"}, {"market": None}, [], None, {"timestamp": 1, "market": {}, "account": {}}):
            self.assertEqual(bot.decide(bad), {"action": "HOLD"})

    def test_degenerate_books_and_reference(self):
        bot = MOD.make_bot()
        cases = []
        o = copy.deepcopy(SAMPLE); o["books"]["YES"]["asks"] = []; cases.append(o)
        o = copy.deepcopy(SAMPLE); o["books"] = {}; cases.append(o)
        o = copy.deepcopy(SAMPLE); o["reference"] = None; cases.append(o)
        o = copy.deepcopy(SAMPLE); o["reference"]["targetProvisional"] = True; cases.append(o)
        o = copy.deepcopy(SAMPLE); o["reference"]["observedAt"] = o["timestamp"] + 50; cases.append(o)
        o = copy.deepcopy(SAMPLE); o["books"]["YES"]["bids"] = [["nan", 5], [2, 3], [0.4, -1]]; cases.append(o)
        for c in cases:
            self.assertTrue(valid(bot.decide(c)), c)

    def test_open_position_never_buys(self):
        bot = MOD.make_bot()
        o = copy.deepcopy(SAMPLE)
        o["account"]["position"] = {"market_id": "sample-btc-5m-market", "outcome": "YES", "shares": 5,
                                    "buy_gross_usd": 2.5, "buy_fees_usd": 0.03, "sell_gross_usd": 0, "sell_fees_usd": 0}
        for tau in (200, 60, 20, 5, 1):
            o["market"]["secondsToClose"] = tau
            d = bot.decide(copy.deepcopy(o))
            self.assertIn(d["action"], ("HOLD", "SELL"))
        # into the close it must try to get flat
        o["market"]["secondsToClose"] = 3
        self.assertEqual(bot.decide(copy.deepcopy(o)), {"action": "SELL"})

    def test_long_stream_valid_and_fast(self):
        bot = MOD.make_bot()
        worst = 0.0
        times = []
        for obs in stream(6000):
            t0 = time.perf_counter()
            d = bot.decide(obs)
            dt = time.perf_counter() - t0
            times.append(dt)
            worst = max(worst, dt)
            self.assertTrue(valid(d, obs["account"]["cashUsd"]), d)
        times.sort()
        p99 = times[int(len(times) * 0.99)]
        self.assertLess(p99, 0.02, f"p99 decide latency {p99*1000:.1f} ms")
        self.assertLess(worst, 0.2, f"worst decide latency {worst*1000:.1f} ms")

    def test_buy_respects_cash(self):
        bot = MOD.make_bot()
        for obs in stream(4000, seed=3):
            obs["account"]["cashUsd"] = 3.0
            d = bot.decide(obs)
            self.assertTrue(valid(d, 3.0), d)


class HttpTests(unittest.TestCase):
    def test_http_endpoint(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MOD.Handler)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/decide"
            req = urllib.request.Request(url, method="POST", data=json.dumps(SAMPLE).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=1) as r:
                payload = json.loads(r.read())
            self.assertLess(time.perf_counter() - t0, 0.2)
            self.assertTrue(valid(payload))
            bad = urllib.request.Request(url.replace("/decide", "/nope"), method="POST", data=b"{}")
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(bad, timeout=1)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
