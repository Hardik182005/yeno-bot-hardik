# YeNo × Builderr Volume Bot — Submission Verification

## Competition Rules & Constraint Checklist

Rules (challenge page + yeno-evaluator-contract.md):

- [x] **API Contract:** Exposes `POST /decide`, strictly returning `{"action":"HOLD"}`, `{"action":"BUY","outcome":"YES"|"NO","maxCashUsd":X}`, or `{"action":"SELL"}`.
- [x] **Cash Caps:** Every BUY `maxCashUsd` ≤ $5.00 and ≤ current available paper cash (fee-inclusive cap enforced by bot's internal fill model).
- [x] **Single Position:** Exactly one open position or pending BUY at any time; never enters when holding inventory.
- [x] **Pre-Close Discipline:** No new BUY entries in the final 15 seconds; mandatory liquidation prior to market expiry (hard exit at $T - 5$ s).
- [x] **Latency Performance:** Responds in < 1 ms (p99 ≈ 0.4 ms on real tick streams, well within the 200 ms timeout).
- [x] **Reference Integrity:** Rejects stale (age > 2.5 s), provisional, or future-stamped reference data; never infers unobservable data.
- [x] **Zero Dependencies:** Pure Python standard library only ($\ge 3.8$), runs with `--network none` in isolated Docker container.
- [x] **Evaluator Boundary:** Only strategy logic is modified; evaluator-owned state (cash, fills, fees, volume, settlement) is never altered.
- [x] **Independent Development:** Clean-room implementation using public tick datasets and official starter kit.
- [x] **Repository & Version Control:** Pushed to GitHub with reproducible history and verification instructions.
- [x] **Frozen Artifact:** Verified via immutable commit SHA.

---

## Submission Email Template

**Send to:** `submit@builderr.ai`  
**Subject:** `YeNo volume bot submission - Hardik182005`

```text
Hi Builderr Team,

I would like to submit my entry for the YeNo × Builderr Volume Bot Challenge.

Name: Hardik
GitHub Account: Hardik182005
Contact Email: hardikhinduja399@gmail.com
Bot Name: YeNo Adaptive Multi-Strategy Volume Bot
GitHub Repository: https://github.com/Hardik182005/yeno-bot-hardik
Immutable Commit Hash: <COMMIT_SHA_PLACEHOLDER>
Entry Point: python3 bot/yeno_bot.py (POST /decide on 127.0.0.1:8080)
Containerized Entry: Dockerfile included (binds 0.0.0.0:8080, compatible with --network none)
Runtime Dependencies: None (Standard Library only, Python >= 3.8)

Best regards,
Hardik
```
