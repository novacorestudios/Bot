# PAPER_TRADING_READINESS.md

**VERDICT: NOT READY.**

Not because a criterion was narrowly missed, but because the criteria that
matter have **not been measured at all**. This document exists to say that
precisely rather than vaguely.

---

## The gate

Per brief §43, a win rate above 50% is explicitly *not* sufficient. Every row
below must pass.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Positive net expectancy **after costs** | **FAIL** | NOT MEASURED — no backtest on real data |
| 2 | Acceptable maximum drawdown | **FAIL** | NOT MEASURED |
| 3 | Out-of-sample stability | **FAIL** | NOT MEASURED — no OOS split has been run |
| 4 | Walk-forward stability | **FAIL** | NOT MEASURED — harness exists, never run on real data |
| 5 | No single symbol or strategy dominating suspiciously | **FAIL** | NOT MEASURED — the matrices are empty |
| 6 | No obvious overfitting | **INCONCLUSIVE** | Nothing has been fitted, so nothing is overfitted — but that is not evidence of robustness either |
| 7 | Costs modelled and survivable under STRESS | **FAIL** | Model built and tested; never run against real prices |

**7 criteria. 0 pass.**

The reason is uniform: `fapi.binance.com`, `api.binance.com` and
`data.binance.vision` are refused by this environment's egress policy, so no
historical data exists to measure against. See
[`BACKTEST_REPORT.md`](BACKTEST_REPORT.md).

---

## What IS ready

Stated so the failure above is not read as a failure of the whole system.

| Area | Status | Evidence |
|---|---|---|
| Data acquisition pipeline | **READY** | Archive + REST sources, manifests, validation; 43 tests |
| Data validation and quality reporting | **READY** | Every corruption class caught; gaps reported not filled |
| Backtest fidelity to the live system | **READY** | Top-N universe, opportunity queue, preservation, matrices, liquidation; 32 tests |
| Realistic cost model | **READY** | Itemised costs, three scenarios, monotonic and adverse; 37 tests |
| Reproducibility | **READY** | Run ID, git commit, config hash, dataset fingerprint, seed |
| Risk engine | **READY** | Unchanged from V2; single construction site for order intents |
| Execution safety | **READY** | Order state machine, idempotent ids, mandatory stops |
| Live data path | **READY** | WebSocket-fed, staleness-gated (V2) |
| CI, security, typing | **READY** | All gates green; bandit 0 findings |

The apparatus is built. The measurement has not been taken.

---

## V3.1: the measurement is now trustworthy, the measurement still has not happened

Fourteen correctness issues were found and fixed after V3 — funding silently
zero, equity marked before the position existed, Sharpe off by ~7x, one trade
attributable to two strategies, and the documented CLI not using the
three-scenario runner. See
[`BACKTEST_AUDIT.md`](BACKTEST_AUDIT.md#v31-correctness-changes).

This changes nothing in the table above. Every row still fails for the same
reason: **no data**. What V3.1 changes is that when the data arrives, the
numbers it produces will mean what they say.

## What has to happen before paper trading

In order. Each step gates the next.

### 1. Obtain real data

```bash
python scripts/fetch_data.py --top 30 --intervals 1m,3m,5m,15m,1h \
    --start 2024-01-01 --end 2025-01-01 --out data
```

A year, covering at least one trending, one ranging and one high-volatility
regime. **Fetch every timeframe** — the strategies are multi-timeframe, and a
primary-only dataset produces zero trades that read as "no edge".

**Gate:** `data/reports/data_quality.json` shows no `UNUSABLE` datasets, and
you have read the coverage figures rather than assuming them.

### 2. Baseline run, all three scenarios

Parameters exactly as shipped — 75 USDT, 0.5% risk, 4 positions, 5× leverage,
top 25, 3600 s cap. **Change nothing.**

**Gate:** BASE shows positive net expectancy after costs. If it does not, stop —
the answer is that there is no edge, and that is a legitimate and valuable
result.

### 3. Read the cost ratio before the PnL

`costs / gross` above 1.0 means the edge was real in price terms and did not
survive contact with the exchange. At ~4 bps taker each way plus spread and
slippage, a round trip costs roughly 10–15 bps; a strategy whose average
winner is smaller than that cannot work at any hit rate.

**Gate:** the average winner comfortably exceeds the round-trip cost.

### 4. STRESS scenario

**Gate:** still positive under STRESS, or at minimum not catastrophic. A
strategy destroyed between BASE and CONSERVATIVE has an edge thinner than the
error bars on a cost model whose spread input is assumed.

### 5. Out-of-sample

Split 60/20/20. Fit nothing on the final 20%.

**Gate:** OOS expectancy has the same sign as in-sample, and is within a factor
of two.

### 6. Walk-forward

**Gate:** the majority of windows are positive, and no single window carries the
whole result.

### 7. Concentration check

**Gate:** no single symbol or strategy contributes more than ~40% of net PnL. A
result carried by one symbol is a result about that symbol.

### 8. Monte Carlo on the real trade sequence

**Gate:** risk of ruin at 75 USDT is acceptable to *you*. The 5th-percentile
outcome, not the median, is the one to plan around.

### 9. Only then: paper trade

On testnet, for weeks not days, comparing realised edge against expected edge
via the execution-quality report. If the model is optimistic, fix the model
before risking capital.

---

## The honest summary

The engineering is in good shape. The system will not let you trade on stale
data, will not open a position it cannot protect, will not size up into a
drawdown, and will not trade without a positive expected edge after costs.

**None of that is an edge.** It is the absence of specific, preventable ways to
lose money. Whether the strategies make money after friction is an empirical
question, the measurement has not been taken, and no amount of further
engineering in this environment will take it.

If the eventual answer is that there is no edge, **the correct action is not to
trade.** That would be a successful use of this system, not a failed one.

---

## V3.2 — readiness after the trust and timing patch

### What moved

| area | before V3.2 | after |
| --- | --- | --- |
| data trust | corrupt data reported TRUSTED | UNUSABLE data is REFUSED, and no flag overrides it |
| walk-forward | no trust check at all | the same gate, the same implementation |
| `opened_at` | the signal timestamp | the fill timestamp |
| max-hold cap | measured from the signal, closing early | measured from the fill |
| funding timing | an assumed 8-hour grid | the symbol's actual event times |
| walk-forward config | seed 0, implicit scenario | seed and scenario explicit and recorded |
| data provenance | nothing written per run | `<report>.data_quality.json` per run |

### What this does and does not license

It licenses **believing the measurement apparatus**. A `TRUSTED` banner now
means the bars were checked and passed; a duration now means time in the
market; a funding cost now comes from real events.

It does **not** license live capital. Nothing here measured profitability, and
the correctness of a measuring instrument is not evidence about what it will
measure. The gate to live trading is unchanged and unmet: a real backtest on
real Binance history, then walk-forward consistency, then paper trading that
matches the backtest's own execution assumptions.

**Status: ENGINEERING / CORRECTNESS VERIFIED. PROFITABILITY NOT MEASURED.**
