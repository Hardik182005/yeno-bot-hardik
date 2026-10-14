#!/usr/bin/env python3
"""Check that a local YeNo /decide endpoint follows the public API contract.

This is a plumbing check, not a score or qualification run.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/decide")
    parser.add_argument("--observation", type=Path, default=Path("yeno-sample-observation.json"))
    args = parser.parse_args()

    request = urllib.request.Request(
        args.url,
        method="POST",
        data=args.observation.read_bytes(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1) as response:
        payload = json.loads(response.read())

    if not isinstance(payload, dict) or payload.get("action") not in {"HOLD", "BUY", "SELL"}:
        raise SystemExit(f"invalid decision: {payload!r}")
    if payload["action"] == "BUY":
        if payload.get("outcome") not in {"YES", "NO"}:
            raise SystemExit(f"BUY is missing a valid outcome: {payload!r}")
        amount = float(payload.get("maxCashUsd", 0))
        if not 0 < amount <= 5:
            raise SystemExit(f"BUY exceeds the $5 fee-inclusive cap: {payload!r}")
    print(json.dumps({"contractSmokePassed": True, "decision": payload}, indent=2))


if __name__ == "__main__":
    main()
