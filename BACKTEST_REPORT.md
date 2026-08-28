# BACKTEST_REPORT.md

**Status: NOT VERIFIED — no backtest on real market data has been run.**

---

## The answer to the question that was asked

> هل الاستراتيجيات الحالية لديها Edge حقيقي بعد Fees + Spread + Slippage + Funding؟
> *Do the current strategies have a real edge after fees, spread, slippage and funding?*

**UNKNOWN. Not "no", not "probably" — unknown, because the measurement has not
been taken.**

This environment has no route to Binance. `fapi.binance.com`, `api.binance.com`
and `data.binance.vision` are all refused by the sandbox's egress policy with a
403 to CONNECT:

```
$ curl -sS -o /dev/null -w "%{http_code}" https://data.binance.vision/
curl: (56) CONNECT tunnel failed, response 403

$ python scripts/fetch_data.py --symbols BTCUSDT --intervals 5m \
      --start 2024-01-01 --end 2024-01-03 --out /tmp/realtry
Cannot reach Binance at https://testnet.binancefuture.com:
  HTTP 403 ... Host not in allowlist
This script needs a host with a route to Binance. Nothing was downloaded
and nothing was written.
```

Per brief §49, what has been built is the pipeline that will answer the
question, exercised against fixtures. **Every performance figure in this
document is absent rather than estimated.**

---

## What this report does NOT contain, and why

| Figure the brief asks for (§28) | Value |
|---|---|
| Initial / Final capital | NOT MEASURED |
| Net PnL, Return % | NOT MEASURED |
| Total trades, Win rate, Loss rate | NOT MEASURED |
| Profit factor, Expectancy | NOT MEASURED |
| Average / median trade | NOT MEASURED |
| Max drawdown | NOT MEASURED |
| Sharpe, Sortino, Calmar | NOT MEASURED |
| Longest losing streak | NOT MEASURED |
| Average / median / P95 duration | NOT MEASURED |
| Fees, funding, slippage, gross/net PnL | NOT MEASURED |

Each of these has code that computes it and a test that the code is right.
None of them has a number, because a number here would be fabricated.

**A zero is not the same as an absence.** A win rate of `0.0` and an unmeasured
win rate look identical in a table and mean opposite things, so the report
generator emits `NOT MEASURED` and never `0`.

---

## What WAS verified

The machinery, on synthetic fixtures. This establishes that the pipeline runs
and that its parts are connected — nothing about profitability.

```
bars processed     3600 across 3 symbols, 5 timeframes
universe rows      2842   (top-N ranking, logged per §8)
trades             1
liquidations       0
matrix trades      1      (recorded with a real regime, not the SIDEWAYS default)
exec quality legs  2      (entry and exit both measured)
rejections         NO_SIGNAL 2481, INSUFFICIENT_CONSENSUS 339,
                   CONFLICTING_SIGNALS 14, LOW_OPPORTUNITY_SCORE 6,
                   NEGATIVE_EXPECTED_EDGE 1
```

**Read that rejection list, not the trade count.** Every analytical gate fired
at least once, including the expected-net-edge gate. One trade in 3600 bars is
the system being extremely selective on data that contains no real edge — which
is the correct behaviour, and is what "never force trading" looks like in
practice.

On a pure random walk the same engine takes **zero** trades. That is worth
stating plainly: given data with no edge, the system correctly declines to
trade it.

### What the fixtures are NOT

They are a plumbing test. The generator produces alternating trend and range
phases with momentum; any PnL from it is a property of the generator, not of a
market. No conclusion about edge can be drawn from them and none is drawn here.

---

## The pipeline that will produce the real answer

```bash
# 1. Fetch. Every timeframe the strategies read, or the run is meaningless.
python scripts/fetch_data.py --top 30 --intervals 1m,3m,5m,15m,1h \
    --start 2024-01-01 --end 2025-01-01 --out data

# 2. Read the quality report BEFORE believing anything downstream.
python -m json.tool data/reports/data_quality.json | head -60

# 3. Run all three execution scenarios.
CONFIG_FILE=config/config.backtest.yaml \
python -m tradebot.app.cli backtest --data data \
    --start 2024-01-01 --end 2025-01-01 --split 2024-10-01 \
    --report reports/backtest.json
```

Baseline parameters, per §40, unchanged from what ships:

| | |
|---|---|
| Initial capital | 75 USDT |
| Risk per trade | 0.5% |
| Max concurrent positions | 4 |
| Max leverage | 5x |
| Top markets | 25 |
| Max trade duration | 3600 s |
| Strategy parameters | as shipped, untouched |

---

## What the backtester now does that it did not before

From [`BACKTEST_AUDIT.md`](BACKTEST_AUDIT.md), which found 21 issues including
four critical ones, three of them silent:

- the universe is **ranked and cut to the top N every scan interval**, from
  point-in-time data, with the ranking logged (was: every symbol, every bar);
- opportunities are **queued and the free slots spent best-first** (was:
  alphabetical order decided who got the last slot);
- **capital preservation engages**, because the drawdown is now passed in (was:
  permanently NORMAL — the backtest measured a system with the brakes off);
- the **strategy × regime matrix records the real regime** (was: every trade
  recorded as `SIDEWAYS` with zero PnL, which would have produced a
  plausible-looking answer to §33);
- **liquidation is modelled** and checked before the stop;
- costs are **itemised** — fee, spread, slippage, latency, funding — and run
  under **three scenarios**.

---

## Known limitations of the measurement itself

These bound what the eventual answer can mean, and none of them is fixable by
more code:

1. **Spread is assumed, not measured.** Binance does not publish usable
   bookTicker history. For a scalper, spread is most of the cost — so the
   central input to the central question is a parameter. This is why three
   scenarios are reported and why the STRESS column is the one to read.
2. **Order-book depth is not simulated.** Slippage is parametric.
3. **Latency is a model**, scaled by the bar's own range. Real latency is not
   in kline data at any resolution.
4. **Maintenance margin is flat 0.4%**, not Binance's notional tiers.
5. **Survivorship bias** if `--top` is used: it ranks by *present-day* volume
   over a *historical* range. Documented in `DATA_PIPELINE.md`; avoidable only
   by supplying a listing snapshot from the start of the period.
6. **Wall time.** Roughly 57 bar-symbol evaluations per second after a 7×
   optimisation. A year of 5m bars across 25 symbols is ~2.6M evaluations —
   on the order of half a day. Budget for it.

---

## V3.1 correctness changes

A source-level audit after V3 found fourteen further issues, all fixed and all
regression-tested. Four mattered enough to change what a result would have meant:

* **Funding was silently zero in every backtest** — the store wrote Parquet and
  the loader read CSV. Any position held across a funding timestamp was
  under-costed.
* **Equity was marked at a price from before the position existed**, booking the
  entry gap as instant PnL.
* **Sharpe used a sampling interval 50x too long**, misstating it by roughly 7x.
* **One trade could be attributed to two different strategies**, so every
  per-strategy figure was quietly wrong.

Plus: the documented CLI did not use the three-scenario runner; stored exchange
filters were ignored in favour of permissive placeholders; funding was charged
every 8 hours from entry rather than at the exchange's timestamps; and
`strict=False` let damaged data produce a confident result.

Full detail, with the regression test for each, in
[`BACKTEST_AUDIT.md`](BACKTEST_AUDIT.md#v31-correctness-changes).

**None of this changes the verdict below.** The measurement still has not been
taken. What changed is that it would now be worth taking.

## Verdict

Using the vocabulary of §42:

**INCONCLUSIVE.**

Not `NEGATIVE` — nothing has been measured that would justify that. Not
`PROMISING` — nothing has been measured that would justify that either. The
data required to reach any of the other verdicts has not been obtainable from
this environment.

The honest position is that the *apparatus* for answering the question is now
built, audited and tested, and the *answer* requires someone to run it on a
host that can reach Binance.

**Do not paper trade on the strength of this document.** See
[`PAPER_TRADING_READINESS.md`](PAPER_TRADING_READINESS.md), which fails on the
criteria that matter for exactly this reason.

---

## V3.2 — what changed about the measurement

No result in this document was produced by a strategy change. V3.2 is a
correctness patch: it changes what the numbers *mean*, not what the system
trades.

Three of the fixes change the interpretation of any result produced before them:

**The trust line was not load-bearing.** `evaluate_trust()` read three
attributes off an object that did not have them, so structurally corrupt data
was reported `TRUSTED`. Any `TRUSTED` banner printed before V3.2 means only
that the timeframes were present and the metadata was found — it says nothing
about whether the bars themselves were sane. Re-run anything you intend to
quote.

**Trade durations were overstated by one decision interval.** `opened_at` held
the signal timestamp while the fill happened at the next bar's open. Every
duration statistic, and the 3600-second maximum-hold cap, measured from before
the position existed — so positions were force-closed one interval early.

**Funding was priced at zero on symbols that had a full funding history.** The
rate lookup snapped onto an assumed 8-hour grid and missed the real event
timestamps. Held positions were under-costed in the *edge estimate* (the
accounting charge itself was already correct as of V3.1).

Every run now writes `<report>.data_quality.json` alongside it, so any number in
this document can be traced to the state of the data behind it.

**Status: ENGINEERING / CORRECTNESS VERIFIED. PROFITABILITY NOT MEASURED.**
