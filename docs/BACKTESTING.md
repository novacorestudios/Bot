# BACKTESTING

> **No backtest has been run against real market data.** The development
> environment has no network route to Binance, so nothing in this repository
> constitutes a performance result. Everything below describes how to produce
> one, and how to avoid fooling yourself while doing it.

## Getting data

```bash
python scripts/fetch_data.py --top 30 \
    --intervals 1m,3m,5m,15m,1h \
    --start 2024-01-01 --end 2024-07-01 \
    --out data
```

Six months across thirty symbols is a reasonable minimum. The default source is
the bulk archive at <https://data.binance.vision>, which needs no credentials
and is roughly 30x fewer requests than REST for a long range.

**Fetch every timeframe the strategies read.** They are multi-timeframe, and a
dataset with only the primary produces zero trades that look like "no edge" —
see *A dataset that produces no trades* below.

See [`DATA_PIPELINE.md`](../DATA_PIPELINE.md) for the storage layout, the
manifests, and the biases (survivorship, listing boundaries) stated as biases.

The loader validates as it reads and **refuses** rather than repairs
out-of-order bars, impossible OHLC relationships and non-positive prices;
duplicates are dropped losslessly and recorded in the manifest, and gaps are
measured and reported rather than filled. Indicators computed across a silent
gap read the resulting price jump as a real move, which is the kind of error
that produces a confident, precise, wrong answer.

## Running

```bash
export PYTHONPATH=src
CONFIG_FILE=config/config.backtest.yaml \
  python -m tradebot.app.cli backtest \
    --data data/klines \
    --start 2024-01-01 --end 2024-07-01 \
    --split 2024-05-01 \
    --report reports/backtest.json
```

`--split` divides the run into in-sample and out-of-sample halves and reports
each. Monte Carlo runs automatically when there are at least 30 trades.

## The look-ahead rules

These are what separate a backtest worth reading from one that flatters itself.

1. **Decisions use closed bars only.** At bar *i* the strategies see bars
   `0..i`, all closed. The forming bar does not exist to them.
2. **Fills happen at the next bar's open.** A signal computed from bar *i*'s
   close cannot fill at that same close — in live trading that close is already
   history by the time the signal exists.
3. **Intrabar resolution is pessimistic.** When one bar's range touches both the
   stop and the target, the **stop** is assumed first. Bar data cannot say which
   came first, and assuming the favourable one inflates every result. This one
   choice frequently separates a "profitable" strategy from a losing one.
4. **Gaps through the stop fill at the open**, not at the stop price.

A test asserts rule 1 directly: running over the first half of a dataset must
produce exactly the trades the full run produced in that half. If a later bar
can change an earlier decision, the backtest is reading the future.

## What is simulated

Fees (maker/taker), spread, size-aware slippage, funding across funding
timestamps, position sizing, leverage, exchange filters (tick/step/min-notional),
stop/target/trailing/time exits, concurrent positions, correlation limits, the
full risk budget and every kill switch.

## What is NOT simulated

Stated plainly, because these are the gaps between a backtest and reality and
every one of them makes the backtest **more optimistic**:

* **Order-book depth.** Slippage is modelled parametrically from size and the
  bar's own range, not by walking a historical book — kline data contains no
  book at all. This is the largest single limitation, and it is why the three
  scenarios exist.
* **Historical spreads.** Binance does not publish bookTicker history at a
  volume or completeness worth relying on, so the spread is an assumption. For
  a scalper, spread is most of the cost, so an assumed spread is an assumed
  answer to the central question. Read the STRESS column accordingly.
* **Market impact beyond the modelled participation term.** Negligible at
  75 USDT, false at scale.
* **Exchange outages and rate limiting.** Rejections are modelled statistically
  in CONSERVATIVE and STRESS, not causally.
* **Maker fills.** Every fill is taker, matching the shipped `entry_order_type:
  MARKET`. A maker strategy would be a different system.

## Execution scenarios

Since V3 the same signals are executed under three sets of assumptions, and all
three are reported side by side. They are **not** three guesses at the truth;
they are a sensitivity test on the part of the model that is assumed rather
than measured.

| | spread | slippage | latency | rejects | partial fills |
|---|---|---|---|---|---|
| `BASE` | configured | size-aware | 250 ms | none | none |
| `CONSERVATIVE` | 2x | 2x + volatility | 750 ms | 1% | 5% at 60% |
| `STRESS` | 4x | 4x + volatility | 2500 ms | 5% | 15% at 40% |

**A strategy profitable under BASE and destroyed under CONSERVATIVE has an edge
thinner than the error bars on the cost model**, which is the same thing as
having no edge you can rely on. `ScenarioResults.survives_stress` asks that
question directly.

Reporting one scenario without the others is the easiest way to fabricate an
edge, so `run_scenarios()` returns all three and the API offers no way to
produce a single-scenario report.

### Latency

Bar data cannot resolve sub-bar timing, so latency is applied as an adverse
price adjustment proportional to the bar's own realised range: a fill delayed
into a fast market is worse than one delayed into a quiet one, and the bar's
range is the only evidence about speed that kline data holds. This is a model,
not a measurement.

### Liquidation

Modelled since V3, checked before the stop on every bar — the exchange does not
wait for our stop. The maintenance-margin model is simplified (a flat 0.4%
rather than Binance's notional tiers), but the point is that leverage risk is
**measured** rather than assumed away by the risk engine's entry check.

## Reproducibility

Every run carries a `RunContext`: run ID, git commit, config hash, dataset
fingerprint, seed and code version. The config hash covers the *whole* validated
config rather than a chosen subset, because the parameter someone forgot to
include is exactly the one that will differ between two runs meant to match.
The dataset fingerprint is order-independent and is taken over bar values, not
file bytes.

A result quoted without its run ID cannot be reproduced or compared, and is
therefore not evidence.

## A dataset that produces no trades

The strategies are multi-timeframe: momentum, breakout and VWAP read the fast,
entry and context series, not just the primary. Given a dataset with only the
primary timeframe they all return `INSUFFICIENT_DATA`, the pipeline rejects
every symbol as `NO_SIGNAL`, and the backtest completes successfully with
**zero trades** — a result that reads as "no edge" and actually means "no data".

The engine now checks timeframe coverage on startup and logs an error naming
the missing series; `BacktestResult.missing_timeframes` carries it too. If that
field is non-empty, a low trade count is a data problem and the run should not
be interpreted.

## The bootstrap problem

This is worth understanding before running anything, because it looks like a bug.

The edge filter estimates a win probability by shrinking a strategy's observed
rate toward a prior until enough trades exist. That is correct — six wins from
eight trades is noise, and sizing an account on it is how people lose money.

But it creates a deadlock. With no history the estimate sits at the 0.45 prior,
and at a reward:risk of 2.0 that makes **every** candidate negative-edge:

| Reward:risk | Prior win rate | Passes at any target size? |
|---|---|---|
| 1.6 | 0.450 | only at ≥ 2 % targets |
| 2.0 | 0.390 | **no** |
| 3.0 | 0.293 | **no** |

Nothing is taken, so no evidence accumulates, so the estimate never moves. The
system cannot start.

For **live trading that is the correct behaviour** — do not risk money on a
strategy that has never been measured. For a **backtest it is fatal**, because
measuring the win rate is the entire point.

So `config/config.backtest.yaml` sets `edge.bootstrap_enabled: true`. An
unproven strategy is then assumed to win at its exact break-even rate *after
costs*,

```
p_breakeven = (loss + costs) / (win + loss)
```

plus `bootstrap_win_rate_margin` (5 percentage points) — i.e. "assume this
strategy is just good enough to be worth measuring". As real trades arrive the
assumption is blended out in favour of the observed rate.

**Every such estimate is counted and reported.** A run that used them tells you
what *would* happen if the assumption held. It is not evidence that it does. The
workflow is:

1. Run with bootstrap on. Read the **measured** per-strategy win rates.
2. Feed those into a second run with bootstrap **off** (`EdgeCalculator.seed_from`).
3. If the edge survives without the assumption, it may be real.

`bootstrap_enabled` must stay `false` in `config.yaml`. A test asserts it.

## Reading a result

The report leads with capital and returns, but those are the least informative
numbers in it. Read in this order instead:

1. **Trade count.** Under 100, nothing else on the page means anything.
2. **Warnings.** They are generated for short samples, high cost ratios, severe
   drawdowns and bootstrap usage.
3. **Costs as % of gross.** Above ~50 % the edge is thin enough that a small
   error in the slippage model flips the result.
4. **Max drawdown**, measured on the equity curve — not on closed trades, which
   would miss an open position's excursion.
5. **Rejection counts.** These say what the system was *refusing* and why, which
   is usually more diagnostic than what it accepted.

`annualized_return` is deliberately reported as **0** for samples under seven
days rather than extrapolated: compounding a one-hour gain to a year produces
roughly 10³⁰⁰, and printing that as a "return" is worse than printing nothing.

## Out-of-sample: two modes, and only one is a holdout

`--split` alone gives `LIVE_LIKE_FORWARD`: **one continuous adaptive run**,
partitioned at the date. Cooldowns, strategy allocation and the edge
calculator's win-rate estimates keep adapting across the split, exactly as they
would live. Faithful to deployment — and **not a clean holdout**, because the
"out-of-sample" half was traded by a system that had already learned from the
in-sample half.

`--strict-oos` gives `STRICT_OOS`: a train engine runs to the split, its learned
statistics are frozen and seeded into a **fresh** engine, and only that engine
runs the test period. This is what "out-of-sample" is normally taken to mean,
and it is the harder test.

The run context records which mode produced a result. Calling the first one a
holdout is the mistake the two labels exist to prevent.

## Edge modes

| Mode | Win probability comes from | Use for |
|---|---|---|
| `LIVE_FAITHFUL` | bootstrap: an unproven strategy is *assumed* to win at break-even plus a margin | reproducing what the live system would do |
| `RESEARCH_STRICT` | measured evidence only; no evidence means no trades | asking whether an edge exists at all |

`--edge-mode RESEARCH_STRICT` cannot manufacture a positive edge. A report must
say which mode produced it, and the two must never be mixed across a split.

## Data trust

Decided **before** the run and carried with the result:

| Condition | Verdict |
|---|---|
| damaged data (out-of-order, impossible OHLC, negative prices) | **REFUSED** |
| a symbol missing a timeframe the strategies read | **REFUSED** |
| gaps, with `--allow-degraded` | UNTRUSTED |
| no stored `exchangeInfo` | UNTRUSTED |
| funding enabled but no funding history | UNTRUSTED |

An UNTRUSTED run may proceed and be inspected; its report carries a banner
saying the numbers are not evidence. `strict=False` no longer turns damaged data
into a confident result silently.

## Decision cadence

Decisions, fills, exits and marks run at the **finest timeframe the dataset
holds** — normally 1m — while strategies read their own closed 3m/5m/15m/1h
series. Evaluating on the 5m primary made each decision point stand in for
twenty live ones and systematically undercounted short-lived setups.

**15-second live decisions cannot be reconstructed from 1m OHLCV.** Four
sub-intervals of a minute are not recoverable from its open, high, low and
close. 1m is a floor on the discrepancy, not a removal of it.

## The cost ledger

Every trade carries a complete, non-overlapping accounting:

```
reference_gross_pnl          both legs at the prices the decision saw
  - spread_cost
  - entry_slippage
  - exit_slippage
  - latency_cost             = execution costs
  - entry_fee - exit_fee
  - funding
  ------------------------
  = net_pnl
```

`gross_pnl` is the PnL **as filled**, so execution costs are already inside it —
subtracting them from that number double-counts. Use `reference_gross_pnl` for
cost attribution. The identity is checked on every close and logged as an error
if it drifts.

## Funding

Charged at the exchange's **actual published timestamps**, for exactly the events
that fall strictly inside a position's life. A position opened at 07:59 pays the
08:00 event; one opened at 08:00 does not, because the exchange's snapshot did
not include it.

## Out-of-sample and walk-forward

A single backtest over one period proves almost nothing — parameters chosen by
looking at that period describe it rather than predict anything.

```bash
CONFIG_FILE=config/config.backtest.yaml \
  python -m tradebot.app.cli walkforward --data data/klines \
    --report reports/walkforward.json
```

```
|---- train ----|-embargo-|- test -|
         |---- train ----|-embargo-|- test -|
                  |---- train ----|-embargo-|- test -|
```

**This is a rolling train/test evaluation, not a 60/20/20
train/validation/test protocol.** The middle window is an **embargo gap** — it
is reserved and skipped, never evaluated. It exists because indicators look back
hundreds of bars, so a test window starting immediately after training would be
evaluated partly on indicator state computed from training data.

Calling it "validation" would imply a model-selection step that does not exist:
**no parameter optimisation is performed anywhere in this module.** Nothing is
fitted, searched or selected; the strategies run with identical parameters in
both windows, and "train" means only "the in-sample baseline for the efficiency
ratio".

Only the **test** windows count. What makes the result trustworthy is not a high
average but **consistency**: a strategy that made everything in one fold and
lost in five worked once. The **efficiency ratio** (out-of-sample ÷ in-sample)
measures how much fitted performance survived unseen data; well below 0.4 is the
signature of overfitting.

The analyzer deliberately does **not** search for parameters. Automated search
over a fixed dataset is precisely how overfitting is manufactured.

## Monte Carlo

A backtest reports one ordering of trades. That ordering flatters or damns the
result largely by luck, and drawdown — which decides whether an account survives
— is highly sensitive to it.

The trade sequence is resampled (bootstrap or shuffle) and the **tail** is
examined:

- **95th-percentile drawdown** — plan around this, not the median. If it exceeds
  `max_drawdown`, the strategy will trip its own kill switch in normal
  operation, which is a design failure regardless of expectancy.
- **Probability of a losing outcome** and **probability of ruin**.
- **95th-percentile losing streak** — check it against
  `max_consecutive_losses`.

Both methods assume trades are independent. They are not: adjacent trades share
a market regime, so real losing streaks cluster worse than either predicts.
**These figures understate tail risk**, and a strategy that only just passes
should be treated as failing.

## Parameter robustness

`parameter_robustness()` judges whether performance survives perturbation. A
strategy profitable only at exactly `roc_threshold: 0.0035` and losing at 0.0030
and 0.0040 was fitted to the parameter, not to the market. It requires a
majority of variants profitable and no catastrophic worst case.

## A finding about the shipped defaults

On synthetic data the shipped configuration produces **essentially no trades**.
The binding constraints, in order:

1. `NO_SIGNAL` — the strategies are built to fire in mutually exclusive
   conditions and individually stay silent most of the time.
2. `INSUFFICIENT_CONSENSUS` — regime gating permits only three or four
   strategies at once, and the aggregator wants two of those few to agree.
3. `LOW_OPPORTUNITY_SCORE` — the 70-point threshold is demanding.

Whether that is correct caution or over-tuning **cannot be settled with
generated prices**. It is a question for real data. The integration tests
exercise the fill/exit/PnL machinery with the gates explicitly opened, and
assert separately that the shipped thresholds are the binding constraint — so
the behaviour is pinned rather than hidden.

If a real-data backtest also produces near-zero trades, the levers, in order of
how much evidence should back changing them:

| Lever | Effect | Caution |
|---|---|---|
| `aggregator.min_agreeing_strategies` | 2 → 1 admits single-strategy signals | consensus is the main defence against a single strategy misfiring |
| `opportunity.min_score` | lower admits weaker setups | tune against measured outcomes, never to hit a trade count |
| `regime.strategy_weights` | permit more strategies per regime | the gating exists because strategies bleed in the wrong regime |
| `edge.min_expected_edge` | **do not lower this to trade more** | it is the one gate standing between the bot and negative-expectancy trading |

## Success criteria

From `IMPLEMENTATION_PLAN.md` §9. A backtest alone satisfies none of them:

1. Out-of-sample positive expectancy, profit factor > 1.15, drawdown within limit
2. Majority of walk-forward folds positive, no catastrophic fold
3. Monte Carlo 95th-percentile drawdown inside the configured limit
4. Performance stable under ±20 % parameter perturbation
5. ≥ 14 days of paper trading with metrics inside the backtest's confidence band

A strategy that passes the backtest and fails any of these is not ready.
