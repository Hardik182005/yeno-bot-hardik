# YeNo × Builderr Volume Bot — Adaptive Multi-Strategy Engine

[![Language](https://img.shields.io/badge/Language-Python%203.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(7%2F7)-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Compliant%20(--network%20none)-orange.svg)]()

A high-performance, standard-library-only `POST /decide` algorithmic trading bot designed for the **YeNo × Builderr Volume Bot Challenge** (BTC 5-minute Up/Down binary options on Polymarket/YeNo).

* **Starting Bankroll:** $10.00 paper cash
* **Target:** $1,000.00 executed cycle volume within 24 hours
* **Qualification Invariant:** 100% flat at the first clean crossing of $1,000 volume
* **Scoring:** Ranked by final paper cash at the crossing point
* **Execution Constraint:** Zero external dependencies, standard library only, sub-millisecond decision latency (p99 < 0.4 ms)

---

## 1. The Core Economic & Microstructure Challenge

In the YeNo competition environment, the evaluator imposes a **1.0% YeNo platform overlay fee** on every executed gross notional dollar, levied on both BUY and SELL sides:

$$\text{Fee}_{\text{overlay}} = 0.01 \times (\text{Gross}_{\text{BUY}} + \text{Gross}_{\text{SELL}})$$

Completing the target of **$1,000 of volume requires paying $10.00 in overlay fees alone** — which equals **100% of the starting $10.00 bankroll**!

In addition, each trade incurs:
1. **Polymarket Protocol Taker Fee:** 
   $$\text{Fee}_{\text{protocol}} = 0.07 \times \text{shares} \times p \times (1 - p)$$
   where $p$ is the fill price.
2. **Bid-Ask Spread & Slippage:** Crossing the L2 order book ladder at a minimum 250 ms execution latency.

$$\text{Final Cash} \approx \$10.00 + \$1,000 \times r$$

where $r = \frac{\text{Net PnL}}{\text{Volume}}$ is the net cost or profit per dollar of volume. 

A naive market-taking or random volume generator produces $r \approx -0.025$ to $-0.040$, going bankrupt within 30 to 50 cycles. **To qualify and rank high, the bot must generate organic trading alpha net of spread and protocol fees** to offset the 1% overlay and finish with positive cash.

---

## 2. System Architecture

Extensive backtesting on tick-level Polymarket order books revealed that **no single static rule remains profitable across all market regimes**. Market conditions shift between:
- Colocated vs. non-colocated book recorder delays (1.0 s to 3.0 s lead).
- Low vs. high volatility spikes.
- Binance spot vs. Chainlink 60-second TWAP settlement discrepancies ($25–$65 basis spread).

To solve this, our bot implements an **in-process adaptive portfolio engine**:

```
                         Incoming Observation JSON (POST /decide)
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  Sanitization & Stale-Data Filtering       │
                   │  (Rejects stale reference, corrupt books) │
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  In-Process Shadow Trading Engine         │
                   │  • 35 Strategy Variants paper-traded      │
                   │  • Simulated 250 ms L2 ladder fill        │
                   │  • Real-time Fee & Settlement tracking    │
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  Online Decayed Ratio Estimator           │
                   │  • Computes r = PnL / Volume per variant  │
                   │  • Welford variance & Standard Error      │
                   │  • Lower Confidence Bound: LCB(r) = r - k·SE│
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  Dynamic Budget Feasibility Filter        │
                   │  LCB(r) ≥ -(Cash - Reserve) / Vol_Needed  │
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  Schedule Pacing & Variant Selection      │
                   │  Selects top-ranked variants calibrated   │
                   │  to cross $1,000 by ~hour 21 (cash focus) │
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  Risk Controls & Execution Guardrails     │
                   │  • RMS position sizing (8% risk fraction) │
                   │  • $5 max cash per BUY                    │
                   │  • $6 cash preservation brake             │
                   │  • 15s entry cutoff / 5s pre-close exit   │
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                           Decision Output: HOLD / BUY / SELL
```

---

## 3. Strategy Families

The 35 internal shadow-traded variants belong to three distinct alpha families:

### Family A: Reference-Lag Scalper (`LAG`)
* **Premise:** High-frequency Binance spot feeds lead Polymarket book updates by 1.0–3.0 seconds during rapid directional moves.
* **Mechanism:** Tracks BTC reference momentum over 1.5 s and 3.0 s rolling windows. When Binance moves significantly and the Polymarket implied probability has not adjusted, it buys the underpriced side.
* **Exit:** Rapid hold duration (2–5 seconds), dynamic break-even-or-better exit, and strict stop-loss.

### Family B: Volatility-Adaptive Favorite (`FAV`)
* **Premise:** Generalizes and improves the House Bot v1 baseline. In trending regimes, the probability favorite tends to underprice tail continuation late in the 5-minute candle.
* **Mechanism:** Normalizes the reference-to-target distance by rolling realized volatility $\sigma$:
  $$z = \frac{P_{\text{ref}} - P_{\text{target}}}{\sigma \sqrt{\Delta t}}$$
  Includes basis-drift correction between Binance spot and Chainlink oracle feeds.
* **Exit:** Trailing profit lock, break-even exit, and forced liquidation before the 15 s cutoff.

### Family C: Analytical Settlement-TWAP Lock-In (`LOCK`)
* **Premise:** YeNo 5-minute markets settle on a **60-second Chainlink TWAP** ($t \in [240, 300]$ seconds).
* **Mechanism:** In the final 60 seconds, a fraction $\tau / 60$ of the settlement TWAP is already finalized and observable:
  $$\text{TWAP}_{60} = \frac{1}{60} \left( \int_0^\tau P(s) ds + (60 - \tau) \bar{P}_{\text{rem}} \right)$$
  The bot uses Acklam’s inverse normal approximation and Gaussian terminal variance to calculate the closed-form probability $P(\text{TWAP}_{60} \ge K)$. When this theoretical probability exceeds the best ask by a positive safety margin, it buys.
* **Exit:** Closes position into the final seconds as the book converges to 0.995+, or redeems at settlement.

---

## 4. Online Statistical Estimation & Risk Management

### 1. Online Decayed Ratio Estimator
To track the true cost-per-volume $r_i$ of variant $i$ without lag from distant regime changes, the bot maintains exponential memory decay ($\approx 400$ cycles):

$$\hat{r} = \frac{\sum_{j} w_j \Delta \text{PnL}_j}{\sum_{j} w_j \Delta \text{Vol}_j}$$

The lower confidence bound (LCB) is calculated via Welford’s online standard error:

$$\text{LCB}(r) = \hat{r} - k \cdot \text{SE}(r), \quad k = 2.0$$

### 2. Capital Budget Constraint
A variant is strictly prohibited from live trading unless its lower confidence bound satisfies the remaining bankroll budget:

$$\text{LCB}(r) \ge -\frac{\text{Cash} - \text{Reserve}}{\text{Target Volume} - \text{Completed Volume}}$$

* $\text{Reserve} = \$1.00$ (untouchable buffer).
* Requires at least 30 shadow cycles and $\ge \$300$ of shadow volume before first live execution.

### 3. Pacing for Cash, Not Speed
The competition ranks entrants by **cash at the moment $1,000 volume is achieved**. Racing to finish in 2 hours with mediocre variants wastes cash. The bot dynamically paces execution to cross $1,000 volume by hour ~21 (with $1.5\times$ safety margin), selecting only the highest-expected-cash variants.

### 4. Hard Safety Brakes
- **$6.00 Cash Brake:** If cash drops below $6.00, all loss-tolerant variants are disabled; only variants with $\text{LCB}(r) \ge 0.0$ may trade.
- **Live Variant Kill-Switch:** If any variant loses $> \$1.50$ in live execution or exhibits live performance significantly worse than shadow estimates, it is banned permanently from the run.
- **Position Sizing:** Dynamically scaled by return volatility so that a 1-RMS loss is bounded at $8\%$ of available cash. Hard-capped at $\min(\$5.00, \text{cash})$.
- **Terminal Liquidation:** Hard cutoff at $T - 15$ s for entries; mandatory exit at $T - 5$ s. Zero open positions carried into market expiry.

---

## 5. Local Evaluator & Replay Framework

The `sim/` module provides a byte-accurate re-implementation of the official YeNo evaluator contract:
- **250 ms L2 Latency:** Decisions are filled against book states observed $\ge 250$ ms later.
- **Ladder Walking:** Matches partial orders against depth levels; rejects undersized orders ($< \text{minOrderSize}$).
- **Fee Accounting:** 5-decimal place rounding matching the competition engine.
- **Datasets Supported:**
  * `PMData`: Tick-level L2 with synchronized Chainlink streams (Hugging Face `mileswangs/polymarket-btc-updown-5m-orderbook-tick-data`).
  * `PolyOrderBooks`: 250 ms resampled L2 order books (`polyorderbooks/polymarket-crypto-btc-5min-250ms-l2`).
  * `MarketLens`: L2 depth snapshots and order book deltas (`marketlens/polymarket-btc-5m-l2-depth`).
  * `Binance`: 1-second historical BTC/USDT klines.

---

## 6. Empirical Backtest & Benchmark Results

All experiments simulate complete 24-hour continuous sessions ($288 \times 5\text{-minute}$ markets) starting with $\$10.00$ paper cash.

### Comprehensive Scenario Matrix

| Scenario | Description | Paths | Qualification Rate | Median Final Cash | Min Cash | Max Cash | Median Hours to $1,000 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`bnf@1.5`** | Binance ref, 1.5s book lag (16-path bootstrap) | 16 | **100% (16/16)** | **$84.29** | $26.17 | $211.48 | 10.9 h |
| **`bnf@2.0`** | Binance ref, 2.0s book lag (real days) | 5 | **100% (5/5)** | **$167.78** | $29.99 | $222.04 | 10.2 h |
| **`bnf@3.0`** | Binance ref, 3.0s book lag | 5 | **100% (5/5)** | **$116.33** | $45.08 | $243.18 | 5.1 h |
| **`bnf@1.0`** | Binance ref, 1.0s book lag | 5 | **80% (4/5)** | **$105.45** | $16.82 | $149.82 | 16.5 h |
| **`bnf@2.0 + adverse`** | 2.0s lag + 2¢ adverse fill penalty | 5 | **60% (3/5)** | **$122.24** | $66.91 | $135.35 | 18.4 h |
| **`bnf@0` / Colocated** | Colocated book (no latency lead) | 5 | 0% (Capital Preserved) | $5.88 | $2.89 | $9.36 | — |
| **`cl@0` / Chainlink** | Pure Chainlink feed (zero lag) | 1 | 0% (Capital Preserved) | $10.04 | $10.04 | $10.04 | — |

*Note: In zero-edge scenarios (`bnf@0` and `cl@0`), the bot's statistical filters detect negative expectancy within 30 cycles and halt trading, successfully preserving the majority of capital ($5.88 to $10.04) rather than going bankrupt.*

### Comparison Against House Bot v1
- **House Bot v1:** Qualified on **0 of 5** local replay days (max volume reached: $127 in 24 h). The competition organizers also report **0 qualifiers in 30 development paths**.
- **YeNo Adaptive Bot:** Qualified on **16 of 16** bootstrap paths under realistic recorder delays, generating up to **+$212 profit on a $10 bankroll**.

---

## 7. Verification & Testing

### 1. Run the Unit Test Suite
Includes 7 contract, robustness, and latency tests:
```bash
python3 -m unittest discover -s tests -v
```
Expected output:
```text
test_buy_respects_cash (test_bot.ContractTests) ... ok
test_degenerate_books_and_reference (test_bot.ContractTests) ... ok
test_long_stream_valid_and_fast (test_bot.ContractTests) ... ok
test_malformed_inputs_hold (test_bot.ContractTests) ... ok
test_open_position_never_buys (test_bot.ContractTests) ... ok
test_sample_observation (test_bot.ContractTests) ... ok
test_http_endpoint (test_bot.HttpTests) ... ok

----------------------------------------------------------------------
Ran 7 tests in 1.300s

OK
```

### 2. Run the Official Smoke Test
Launch the bot in one terminal:
```bash
python3 bot/yeno_bot.py
```
Run the smoke test in another terminal:
```bash
python3 yeno-smoke-test.py
```
Output:
```json
{
  "contractSmokePassed": true,
  "decision": {
    "action": "HOLD"
  }
}
```

### 3. Docker Verification
Build and test inside a hardened, isolated container without network access:
```bash
docker build -t yeno-volume-bot .
docker run --rm --network none -p 8080:8080 yeno-volume-bot
```

---

## 8. Repository Structure

```
├── Dockerfile                      # Hardened container image (Python 3.12-slim, non-root)
├── README.md                       # Architecture, strategy formulas, benchmarks
├── SUBMISSION.md                   # Submission verification checklist & email draft
├── yeno-sample-observation.json     # Official schema observation payload
├── yeno-smoke-test.py              # Official protocol contract smoke test
├── bot/
│   └── yeno_bot.py                 # Core submission bot (standard library only)
├── sim/
│   ├── evaluator.py                # Byte-accurate local evaluator & L2 order matcher
│   ├── scenarios.py                # Market latency & feed perturbation models
│   ├── binance.py                  # Binance 1s spot kline client & cache
│   ├── prep_pmdata.py              # PMData tick dataset converter
│   ├── prep_polyob.py              # PolyOrderBooks 250ms L2 dataset converter
│   ├── prep_marketlens.py          # MarketLens L2 depth dataset converter
│   └── run.py                      # Multi-day simulation execution harness
├── research/
│   ├── features.py                 # Orderbook feature engineering
│   ├── roundtrips.py               # Simulated roundtrip trade generator
│   ├── matrix.sh                   # Comprehensive evaluation matrix runner
│   └── results/
│       ├── matrix.txt              # Multi-scenario empirical backtest logs
│       └── bootstrap_bnf1.5.txt    # 16-path block bootstrap logs
└── tests/
    └── test_bot.py                 # Invariant, contract, and latency test suite
```

---

## 9. Submission Metadata

* **Repository:** [https://github.com/Hardik182005/yeno-bot-hardik](https://github.com/Hardik182005/yeno-bot-hardik)
* **Author:** Hardik (`Hardik182005`)
* **Contact:** `148485624+Hardik182005@users.noreply.github.com`
* **Entry Point:** `python3 bot/yeno_bot.py` (`POST /decide` on port 8080)
* **Dependencies:** None (Pure Python $\ge 3.8$ standard library)
