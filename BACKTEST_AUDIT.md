# BACKTEST_AUDIT.md

**Subject:** the existing backtesting engine, audited before any V3 work
**Commit audited:** `7835df1`
**Method:** every claim below was checked against the code and, where a defect
is asserted, reproduced by executing it. Reproduction commands are given.

---

## The headline

The backtester is **structurally honest but architecturally behind the live
engine.** Its look-ahead discipline is real and correct — next-bar fills,
closed-bars-only decisions, pessimistic intrabar resolution. That is the hard
part, and it was done properly.

The problem is drift. The live engine gained a universe scanner, an opportunity
queue, capital preservation, execution-quality feedback and performance
matrices in V2. **The backtester gained none of them**, and two of the wiring
gaps are silent: they do not error, they produce numbers that look plausible
and are wrong.

Concretely, a backtest run today would:

* evaluate **every symbol in the dataset on every bar**, with no Top-25
  selection — so §8 of the brief is not satisfied at all;
* take the **first** symbol that passes risk in timeline order rather than the
  best — the exact bug V2-6 fixed for live trading;
* run with **capital preservation permanently in NORMAL**, because the
  backtester never tells the risk engine what the drawdown is;
* record **every trade as `SIDEWAYS`** in the strategy×regime matrix.

None of these would be visible in the output. That is what makes them worth
fixing before any result is believed.

---

## Status summary

| Area | Status |
|---|---|
| Data model | IMPLEMENTED |
| Historical data loader | PARTIALLY IMPLEMENTED |
| Candle handling | IMPLEMENTED |
| Strategy invocation | IMPLEMENTED |
| Risk engine integration | PARTIALLY IMPLEMENTED |
| Execution simulation | PARTIALLY IMPLEMENTED |
| Fee model | IMPLEMENTED |
| Slippage model | PARTIALLY IMPLEMENTED |
| Spread model | PARTIALLY IMPLEMENTED |
| Funding model | PARTIALLY IMPLEMENTED |
| Position handling | IMPLEMENTED |
| Leverage | IMPLEMENTED |
| Liquidation handling | **MISSING** |
| Take profit | IMPLEMENTED |
| Stop loss | IMPLEMENTED |
| Time exit | IMPLEMENTED |
| Correlation | PARTIALLY IMPLEMENTED |
| Concurrent positions | IMPLEMENTED |
| Kill switches | PARTIALLY IMPLEMENTED |
| Capital preservation | **MISSING (silently inert)** |
| Dynamic Top 25 | **MISSING** |
| Opportunity Queue | **MISSING** |
| Execution quality feedback | **MISSING** |
| Performance matrices | **MISSING (silently wrong)** |
| Latency model | **MISSING** |
| Partial fills | **MISSING** (documented) |
| Reproducibility / run IDs | **MISSING** |
| Data quality report artefact | PARTIALLY IMPLEMENTED |

---

## What is genuinely right, and must not be broken

Stated first, because the V3 work must not damage it.

### The look-ahead discipline is correct

`_advance()` appends a bar only once `candle.close_time <= timestamp`, so at the
event for bar *N* the strategies see bars up to *N−1* and the forming bar does
not exist. `_next_open()` then fills at the first bar opening strictly after the
signal timestamp. The net effect is a **one-interval delay between signal and
fill**, which is one bar more conservative than the usual "fill at this bar's
open" convention.

### Intrabar resolution is pessimistic by default

When a bar's range touches both stop and target, `_exit_for()` assumes the
**stop**. Bar data genuinely cannot say which came first, and assuming the
favourable one is the single most common way a losing strategy is made to look
profitable. `intrabar: optimistic` exists as a config option but is not the
default — and its presence is useful, because the gap between the two settings
measures how much of a result rests on that assumption.

### Stops fill at the stop price *or worse*

`_stop_fill()` returns `min(stop, bar.open)` for a long — a gap through the stop
fills at the open, not at the stop. Most naive backtesters get this wrong and
book the stop price every time.

### The live pipeline is reused, not reimplemented

The backtester calls the real `SignalPipeline`, `RegimeDetector`, `MarketScorer`
and `RiskEngine`. A backtester that reimplements the trading logic tests the
reimplementation. This one does not.

### Results feed back exactly as they do live

`record_trade_closed` and `edge_calculator.record_result` are called on every
close, so cooldowns, strategy allocation, the strategy kill switch and the edge
calculator all evolve during the run rather than being frozen.

---

## CRITICAL findings

### B-1 — No Dynamic Top 25. Every symbol is evaluated on every bar

`_consider_entry()` is called for each `(timestamp, symbol)` event in the
timeline with no universe construction. There is no `MarketScanner`, no
`UniverseBuilder`, no ranking, no top-N cut.

```bash
grep -c "MarketScanner\|UniverseBuilder" src/tradebot/backtesting/engine.py   # 0
```

**Impact.** Brief §8 requires the universe to be rebuilt each cycle from
liquidity, volume, spread, volatility and market quality *using only
information available at that moment*, with `timestamp/rank/symbol/score/reason`
logged. None of that exists. A backtest over 40 symbols currently tests "trade
any of these 40 whenever a signal appears", which is **not the system being
sold**. It is also optimistic in a subtle way: the live engine can only ever act
on 25 symbols, and the scanner's job is partly to exclude symbols whose spread
or thinness would eat the edge.

### B-2 — No opportunity queue; first-past-the-post wins the slot

`_consider_entry()` opens a position the moment risk approves, inside the
per-symbol loop. Timeline order is `(timestamp, symbol)` sorted — so for bars
sharing a timestamp, **alphabetical symbol order decides who gets the last free
slot.**

```bash
grep -c "OpportunityQueue" src/tradebot/backtesting/engine.py   # 0
```

**Impact.** This is the identical defect V2-6 fixed for live trading
(AUDIT_REPORT.md H-4). A score-94 opportunity on `SOLUSDT` loses its slot to a
score-71 on `ADAUSDT` because A sorts before S. The backtest therefore measures
a worse strategy-selection policy than the live engine actually uses — and,
being alphabetical, does so with a bias that is stable across runs and so will
not average out.

### B-3 — Capital preservation is permanently inert

`RiskEngine.evaluate()` computes the preservation mode from
`context.drawdown`, `context.realized_pnl_today` and the kill switch's
consecutive-loss count. The backtester's `RiskContext` sets **none** of the
first two.

```bash
python -c "
import inspect
from tradebot.backtesting.engine import BacktestEngine
print('drawdown passed:', 'drawdown' in inspect.getsource(BacktestEngine._consider_entry))
"   # drawdown passed: False
```

**Impact.** `drawdown` defaults to `0.0`, so `CAUTIOUS`/`DEFENSIVE`/`HALTED` can
only ever be reached via consecutive losses. The risk-reduction ladder that is
supposed to protect a 75 USDT account through a bad run **does not run in the
backtest at all**, and the backtest's drawdown figure is therefore the drawdown
of a system with the brakes disconnected. This makes the result *pessimistic*
on drawdown and *optimistic* on trade count — but either way it is not the
system that would be deployed.

### B-4 — Every backtest trade is recorded as `SIDEWAYS` in the matrices

`_close_position()` calls `record_trade_closed()` without `regime=` or `pnl=`,
both of which were added in V2-10. Python fills the defaults.

```bash
python -c "
import ast, inspect, textwrap
from tradebot.backtesting.engine import BacktestEngine
t = ast.parse(textwrap.dedent(inspect.getsource(BacktestEngine._close_position)))
for n in ast.walk(t):
    if isinstance(n, ast.Call) and getattr(n.func,'attr','')=='record_trade_closed':
        print(sorted(k.arg for k in n.keywords))
"   # ['r_multiple', 'reason', 'volatility', 'won']  -> regime and pnl missing
```

**Impact.** Brief §33 asks for a strategy×regime table. Built from this data it
would show one populated column, `SIDEWAYS`, containing every trade — and it
would look like a real result rather than an empty one. The matrix cell PnL is
likewise always `0.0`. This is the worst kind of defect: it produces a
plausible-looking answer to the exact question §33 asks.

---

## HIGH findings

### B-5 — No liquidation modelling

Nothing computes a liquidation price or force-closes a position. `RiskEngine`
refuses entries whose estimated liquidation sits too close
(`min_liquidation_distance_multiple`), so a liquidation should be rare — but
"should be rare" is an assumption the backtest is supposed to test, not one it
is allowed to make. At 5× leverage a 20% adverse move liquidates, and the
backtest would simply keep the position open through it.

### B-6 — No latency between signal and fill beyond the bar boundary

Brief §17 asks for an explicit `signal → decision → submission → execution`
latency model. The current model is implicit: one bar. On a 5m primary
timeframe that is a *300-second* implied latency, which is far too generous a
gap to leave unparameterised — real latency is tens to hundreds of milliseconds,
and the difference matters most exactly where scalping edges live.

### B-7 — Spread and slippage are single fixed constants

`backtest.spread_bps` (1.0) and `backtest.slippage_bps` (1.5) are applied
uniformly to every symbol, every bar, every market condition. Brief §14/§15
require at minimum a documented conservative model and **three scenarios**
(BASE / CONSERVATIVE / STRESS). Neither the scenarios nor any size-, volatility-
or regime-awareness exists. A constant spread also means the scanner's
spread-quality score is constant, so it cannot discriminate between symbols —
which quietly disables part of the market-quality ranking.

### B-8 — Execution-quality feedback does not run

`ExecutionQuality` (V2-8) is not constructed. The live engine measures predicted
versus realised cost and feeds the bias back into the cost model; the backtester
does not, so the two systems price trades differently over a long run. Brief
§34/§35 require expected-vs-actual calibration explicitly.

---

## MEDIUM findings

| ID | Finding |
|---|---|
| B-9 | `_liquidity()` estimates 24h quote volume as `mean(last 20 bars) × 288`. The 288 is hardcoded for **5-minute** bars; on a 1m primary it under-states volume by 5×, and on 15m over-states it by 3×. Liquidity feeds the market score and the slippage model. |
| B-10 | `correlation_penalty=0.0` is hardcoded into `ScoringInputs`, so the market score never reflects correlation. The risk engine's correlation *gate* does run, so this affects ranking only — but ranking is what B-1 is about. |
| B-11 | `book_imbalance=0.0` always. Any strategy weighting book imbalance is running on a constant in backtest but a live signal in production. |
| B-12 | Equity is recorded every 50 bars (`index % 50 == 0`). Max drawdown is computed from that curve, so a drawdown that opens and recovers inside 50 bars is invisible. On 5m bars that is a 4-hour blind spot. |
| B-13 | Funding is charged on `quantity × entry_price`, not on the mark price at the funding timestamp. Directionally minor, but it is not what the exchange does. |
| B-14 | Funding is applied *after* the exit check, so a position closing on the same bar as a funding timestamp pays nothing. |
| B-15 | `_default_symbol_info()` invents permissive filters (tick 1e-8, min notional 5.0) when `exchangeInfo` was not saved. It warns, which is right, but §3 requires real tick/step/minQty/minNotional and the download script never saves them. |
| B-16 | No P95 trade duration. §23 asks for average, median, **P95** and max; `BacktestMetrics` has the first two and the max only implicitly. |
| B-17 | No run ID, git commit, config hash, dataset hash or seed in the output. §38 requires all of them for reproducibility. |

---

## LOW findings

| ID | Finding |
|---|---|
| B-18 | `_close_position` takes a `data` argument it never uses (`_ = data`). |
| B-19 | The data loader's `DataQuality` is returned to the caller but never written to disk as an artefact; §5 asks for a Data Quality Report. |
| B-20 | `load_dataset` picks `usable_timeframes[0]` as the discovery directory, which is the first *requested* timeframe, not the configured primary. Symbol discovery therefore depends on argument order. |
| B-21 | Funding is looked up by `(timestamp // interval) * interval`, which happens to align with Binance's 00:00/08:00/16:00 UTC schedule only because those are exact multiples of 8h from the epoch. Correct today, undocumented, and silently wrong if `funding_interval_hours` is ever changed. |

---

## Data pipeline: what exists

`scripts/download_data.py` fetches klines and funding over the REST API and
writes one Parquet/CSV per symbol/timeframe.

**What it does not do**, against brief §3–§5:

* no bulk-archive support (`data.binance.vision`), which is the only practical
  way to obtain multi-year 1m history and needs no credentials;
* no `exchangeInfo` capture, so tick size, step size, min qty and min notional
  are guessed (B-15);
* no dataset **manifest** — §4 requires symbol, interval, start, end, source,
  download timestamp and schema version per dataset, and none is written;
* no standalone **validation stage** or quality-report artefact (§5); the
  checks that exist run inside the loader at backtest time;
* no listing/delisting boundary handling, so survivorship bias is neither
  avoided nor documented.

---

## Environment limitation

`fapi.binance.com`, `api.binance.com` and `data.binance.vision` are all refused
by this sandbox's egress policy:

```bash
curl -sS -o /dev/null -w "%{http_code}" https://data.binance.vision/
# curl: (56) CONNECT tunnel failed, response 403
```

Per brief §49, the V3 work therefore builds the acquisition pipeline, exercises
it against fixtures and local datasets, and makes the download runnable by the
operator — and **no backtest result on real Binance data will be produced or
claimed from this environment.**

---

## What V3 must fix, in order

1. **B-1, B-2** — universe selection and the opportunity queue. Without these
   the backtest does not test the shipped system, so every downstream number is
   answering the wrong question.
2. **B-3, B-4** — the two silent wiring gaps, because they produce plausible
   wrong answers rather than errors.
3. **B-7, B-6** — the cost and latency model, plus BASE/CONSERVATIVE/STRESS.
   This is what decides whether an edge survives friction, which is the whole
   question V3 exists to answer.
4. **B-5** — liquidation, so leverage risk is measured rather than assumed away.
5. **B-9 to B-17** — correctness and reproducibility.

Only then is a baseline run worth reading.

---

# V3.1 CORRECTNESS CHANGES

A source-level audit after V3 found fourteen further issues. Every one was
verified against the code before being fixed, and every fix has a regression
test named for the defect it prevents. Tests live in
`tests/integration/test_v31_correctness.py` unless stated otherwise.

**None of these changes touches a strategy, a parameter or the trading thesis.**
The correct outcome of V3.1 is CORRECTNESS VERIFIED, not PROFITABILITY VERIFIED.

## P0

### 1 — Funding was silently zero in every backtest

**Problem.** `DataStore.write_funding()` wrote `data/funding/<SYMBOL>.parquet`
with columns `funding_time` / `funding_rate`. `_load_funding()` read
`data/funding/<SYMBOL>.csv` with columns `fundingTime` / `fundingRate`. Nothing
raised — the mismatch produced an empty mapping, so **funding was zero in every
run**, flattering any position held across a funding timestamp.

**Fix.** The loader reads Parquet first and CSV as legacy, accepts all three
known column layouts, and *logs an error* when a funding file exists but cannot
be read — a missing file is legitimate, an unreadable one is not.

**Test.** `TestIssue1FundingLoadsFromParquet` — writes funding through the real
store, loads through the real loader, and asserts the rates survive. Would fail
outright with the original code.

### 2 — The documented CLI did not use the V3 scenario runner

**Problem.** `tradebot backtest` called `BacktestEngine.run()` directly, so the
documented command produced a **single-scenario** run. `run_scenarios()` existed
and was tested in isolation, which proved nothing about what users actually ran.

**Fix.** `run_backtest()` now calls `run_scenarios()` and prints BASE /
CONSERVATIVE / STRESS side by side, with the full run context and the trust
verdict.

**Test.** `TestIssue2CLIUsesTheScenarioRunner` asserts the CLI function calls
`run_scenarios` and no longer constructs a `BacktestEngine`.

### 3 — Real exchange filters were stored and then ignored

**Problem.** The acquisition pipeline saved `exchangeInfo`, but the CLI called
`load_dataset()` without `symbol_infos`, so the loader fell back to permissive
placeholders (tick `1e-8`, min notional 5.0). The backtest could take positions
the exchange would reject — most often the small ones a 75 USDT account depends
on.

**Fix.** `load_dataset()` loads stored `exchangeInfo` automatically. Missing
filters no longer pass silently: they downgrade the run to **UNTRUSTED**.

**Note.** `exchangeInfo` is a *present-day metadata snapshot*. Binance changes
tick sizes, step sizes and minimum notionals over time, and there is no
historical filter endpoint — so filters applied to a 2024 backtest are 2026's.
This is a bounded but real inaccuracy, and it is stated rather than hidden.

**Test.** `TestIssue3RealExchangeInfo`.

### 4 — Equity was marked at a price from before the position existed

**Problem.** A signal is computed from bars closed at *T* and filled at the next
bar's open. `_record_equity` marked open positions at `series.last_price` — the
bar closed *before* the fill — so the entire open-to-previous-close gap was
booked as instant PnL. On a 1% gap at 5x, that is 5% of margin appearing at the
instant of entry, inflating the curve and understating the drawdown after it.

**Fix.** A position opened in the current cycle is not marked at all; its honest
mark-to-market is zero until a bar closes after it. A position is never marked
against a bar that closed at or before its fill.

**Test.** `TestIssue4EquityMarkingTiming`, parameterised over entries below, at
and above the previous close.

### 5 — Sharpe used a sampling interval 50x too long

**Problem.** `_result()` passed `primary_timeframe x 50` to `compute_metrics()`,
left over from when equity was sampled every 50 bars. Equity is now recorded
every cycle. Sharpe and annualised volatility scale with the square root of
samples per year, so a 50x error misstates Sharpe by about 7x.

**Fix.** The interval is measured from the equity curve itself, using the
**median** consecutive gap — a mean would be dragged by any data hole.

**Test.** `TestIssue5SamplingInterval`, for 1m/5m/15m plus a month-long gap.

### 6 — One trade could be attributed to two different strategies

**Problem.** `EdgeCalculator` used `contributing[0].strategy` while
`Opportunity.strategy` used the highest-confidence contributor. When the
aggregator's tuple order differed from confidence order, **edge statistics, risk
allocation and trade ownership referred to different strategies**, and every
per-strategy report was quietly wrong.

**Fix.** One model, on `AggregatedSignal`:
`primary_strategy` (highest confidence, ties broken on name so ordering cannot
change the answer), `contributing_strategies`, `contribution_weights`. Both call
sites delegate to it, and the weights are recorded on the intent and the trade
so a report can separate the trade's owner from its supporters.

**Test.** `TestIssue6Attribution` — the same contributors in both tuple orders
must attribute identically.

## P1

### 7 — Decisions ran only on the 5-minute timeline

**Problem.** The live engine evaluates every 15 seconds; the backtest evaluated
per 5m bar, so each decision point stood in for twenty live ones. Setups that
appear and resolve inside one 5m bar were invisible, **systematically
undercounting short-lived opportunities**.

**Fix.** Decisions, fills, exits and marks run at the finest timeframe the
dataset holds — normally 1m — while strategies keep reading their own closed
3m/5m/15m/1h series. Nothing is interpolated.

**Limitation, stated not hidden.** 15-second decisions **cannot** be
reconstructed from 1m OHLCV: four sub-intervals of a minute are not recoverable
from its open, high, low and close. 1m is a floor on the discrepancy, not a
removal of it.

**Test.** `TestIssue7DecisionCadence`.

### 8 — A continuous adaptive run was presented as a holdout

**Problem.** `--split` partitioned the trades of **one continuous run**, during
which cooldowns, allocation and win-rate estimates kept adapting across the
split. The "out-of-sample" half was traded by a system that had already learned
from the in-sample half.

**Fix.** Two explicitly labelled modes. `LIVE_LIKE_FORWARD` (the default) is the
continuous run and now says in its own output that it is **not a clean holdout**.
`--strict-oos` runs a train engine, freezes its learned statistics, seeds a
fresh engine with them and runs only the test period.

**Test.** `TestIssue8And12ModesAreDeclared`.

### 9 — "Validation" was defined and never used

**Problem.** `WalkForwardAnalyzer` computed a validation window and never
evaluated it, while the module described a train/validation/test protocol.

**Fix.** Option (B): renamed to **embargo** and documented as what it is — a
reserved gap that stops indicator state computed on training bars leaking into
the test window. The module now states plainly that **no parameter optimisation
is performed anywhere in it**, and that this is a rolling train/test evaluation,
not 60/20/20.

**Test.** `TestIssue9WalkForwardHonesty`.

### 10 — Costs could be double-counted

**Problem.** `gross_pnl` is computed from *filled* prices, which already contain
spread, slippage and latency. `slippage_cost` was also recorded as a separate
field, so any report doing `gross - fees - funding - slippage` double-counted.

**Fix.** An explicit ledger on `Trade`: `reference_gross_pnl` (both legs at
their reference prices), `entry_fee`, `exit_fee`, `spread_cost`,
`entry_slippage`, `exit_slippage`, `latency_cost`, `funding`. The identity

```
reference_gross_pnl - execution_costs - fees - funding == net_pnl
```

is checked on every close and logged as an error if it drifts.

**Test.** `TestIssue10CostLedger`, including a deliberately unbalanced ledger to
prove the check fires.

### 11 — Funding was charged every 8 hours from entry

**Problem.** `_apply_funding` charged on an interval measured from the position's
own open time, not at the exchange's published timestamps. A position opened at
07:59 was charged for 08:00 only at 15:59; one opened at 00:01 was charged for an
event it had missed. Both wrong, in opposite directions, neither visible.

**Fix.** Funding is charged for exactly the events whose timestamps fall strictly
inside the position's life. Opened exactly at an event: not charged, because the
exchange's snapshot did not include it.

**Test.** `TestIssue11FundingUsesRealEvents` — just before, exactly at and just
after an event; long and short; multiple events; and a double-sweep to prove no
event is charged twice.

## P2

### 12 — Research mode

**Problem.** The bootstrap assumes an unproven strategy wins at break-even plus
a margin. That is faithful to the live system, and it means a backtest partly
measures the assumption.

**Fix.** `--edge-mode RESEARCH_STRICT` disables it: a strategy with no measured
evidence does not trade. The mode is recorded in the run context, so a report
always states whether it is `LIVE_FAITHFUL` or `RESEARCH_STRICT`.

### 13 — Survivorship bias was a log line

**Problem.** `--top` ranks by present-day volume and applies that ranking to a
historical range.

**Fix.** `--top` now **refuses to run** without
`--i-understand-survivorship-bias`, and prints what the bias does. `--symbols-file`
accepts a point-in-time listing snapshot. The provenance is written beside the
data and carried in the run context as `POINT_IN_TIME_UNIVERSE` or
`PRESENT_DAY_UNIVERSE`.

### 14 — `strict=False` turned damaged data into a trusted result

**Problem.** The CLI loaded with `strict=False` and printed a warning. Nothing
stopped a confident number being produced from damaged data.

**Fix.** `backtesting/trust.py` decides before the run and the verdict travels
with it. Damaged data and missing timeframes are **REFUSED**; missing
`exchangeInfo` or funding **downgrade to UNTRUSTED**; gaps require an explicit
`--allow-degraded` and force UNTRUSTED. Every report from a non-trusted run
carries a banner saying the numbers are not evidence.

**Test.** `TestIssue14DataQualityGate`.

## Also fixed while verifying

`DataStore.manifests()` derived the data path from the manifest path with a
`with_suffix` chain that produced a path existing nowhere, so it always returned
an empty list — and **every run quoted the fingerprint of an empty set**, which
looks like a valid hash and identifies nothing. Now strips the `.manifest.json`
suffix directly.

## V3.1 verification — what actually ran

Every gate below was executed against the tree at the V3.1 commits. Results are
reported as they came back, including the one that could not run here.

| Gate | Command | Result |
| --- | --- | --- |
| Unit + integration suite | `pytest -q` | PASS — 1213 tests, 0 failures |
| V3.1 regression tests | `pytest tests/integration/test_v31_correctness.py -q` | PASS — 62 tests |
| Lint | `ruff check src tests scripts` | PASS |
| Format | `ruff format --check src tests scripts` | PASS |
| Types | `mypy src/tradebot` (strict via `pyproject.toml`) | PASS — 93 files, 0 errors |
| Static analysis | `bandit -r src -c pyproject.toml -q` | PASS — 0 findings |
| Dependency audit | `pip-audit -r requirements.txt --strict` | PASS — no known vulnerabilities |
| Secret scan | `python scripts/check_secrets.py` | PASS — 161 files clean |
| CLI end to end | `tradebot backtest --data ... --split ... --strict-oos` | PASS — exit 0 |
| Docker image build | `docker build -f docker/Dockerfile -t tradebot:ci .` | **NOT RUN HERE** — see below |
| Docker smoke tests | all six checks from the `build` job of `.github/workflows/ci.yml` | PASS on the stand-in image |

### The Docker qualification is real, not a formality

The shipped `docker/Dockerfile` **cannot be built in this environment**. Both
`docker.io`'s blob CDN and `deb.debian.org` return `403 Forbidden` through the
agent proxy, so the `# syntax=docker/dockerfile:1` frontend fetch and the
`apt-get update` layers both fail. The build was attempted and failed on the
frontend fetch, not on anything in the project.

What was verified instead is a **stand-in image**: same `src/`, `config/`,
`scripts/`, same `docker/entrypoint.sh`, same `ENTRYPOINT`/`CMD` contract, built
from a reachable base mirror without the multi-stage `apt` layers. That proves
the CLI contract the smoke test exercises — `validate-config` and `doctor` both
exit 0 in `PAPER` mode with no credentials — but it does **not** prove the
shipped Dockerfile builds. CI on GitHub Actions is the authority for that layer.

All six checks the `build` job runs after the build were executed against that
stand-in and passed:

1. `validate-config` in `PAPER` mode with no credentials — exit 0
2. `doctor` with no credentials and no network call — exit 0
3. default `CMD` is exactly `["run"]`, not a shell
4. `TRADING_MODE=LIVE` without the acknowledgement — exit **78** (`EX_CONFIG`)
   and `REFUSING TO START` on stderr
5. `TRADING_MODE=LIVE` against the testnet endpoint, acknowledgement set —
   exit **78**
6. the image runs as `tradebot`, not root

### Out-of-sample run on the small local dataset

The `--strict-oos` CLI run completed in 45s and took **zero trades** in both the
train and the test window. That is a genuine `NO TRADES TAKEN`, reported as such
rather than as a measured zero — one symbol over 25 hours does not clear the
opportunity gate. It exercises the code path; it is not evidence about the
strategies either way.

### One test the rename broke

`tests/unit/test_backtest.py::TestWalkForward::test_folds_tile_the_period_with_a_rolling_step`
still asserted on `fold.validation_start` / `fold.validation_end` after the
V3.1 rename to `embargo_start` / `embargo_end`, and failed with
`AttributeError`. The full suite was red until it was updated to the new names
and extended to assert `embargo_is_evaluated is False` — the property the
rename exists to make visible. The `WalkForwardConfig.validation_days` **config
key keeps its historical name** so existing config files still load; only the
`Fold` field, which describes what the window *is*, was renamed.

---

# V3.2 TRUST & TIMING CHANGES

A correctness-only patch, run before the first backtest on real Binance
history. No strategy was added, no parameter was tuned, no threshold was moved.

Two of these defects were **completely silent** — they produced plausible
numbers rather than errors — which is the reason the regression tests for them
assert on the shape of the code and not only on its output.

## P0 — the data trust gate was reading attributes that did not exist

**Problem.** `evaluate_trust()` decided whether a dataset was damaged by
reading `q.status`, `q.interval` and `q.missing_bars`. The objects it was
actually handed are `DataQuality` instances from `load_dataset()`, and that
class had none of those three attributes. The reads went through
`getattr(q, "status", None)`, so every guard that depended on them evaluated to
`False`. Not an error — a `None`.

The effect, reproduced before any code was changed:

```
DataQuality.usable         = False      <- the loader knew
getattr(q, 'status', None) = None       <- the gate could not ask
TRUST LEVEL ON CORRUPT DATA = TRUSTED
blockers = []   downgrades = []
```

That series had impossible OHLC, a 500-bar hole, seven duplicate timestamps and
40% coverage. The gate reported `TRUSTED`. Only the missing-timeframe and
missing-metadata checks — which read `data`, not `quality` — were ever live.

**Fix.** One contract, stated once, and typed so the compiler enforces it.
`DatasetQuality` is a `Protocol` in `backtesting/trust.py` naming exactly what
the gate needs. Two classes implement it — `DataQuality` at load time and
`ValidationReport` at acquisition time — and both now report the **same**
`QualityStatus` vocabulary (`OK` / `DEGRADED` / `UNUSABLE`) rather than each
having a private notion of "fine". `evaluate_trust()` takes
`Sequence[DatasetQuality]` and reads the attributes directly. The defensive
`getattr` defaults are gone, so the next schema drift is a mypy error rather
than a confident wrong number.

Detection was widened to everything the brief lists. `load_candles()` now
records, and `DataQuality.status` now grades:

| condition | status |
| --- | --- |
| impossible OHLC (high < low, or open/close outside the range) | UNUSABLE |
| any price at or below zero, on any of the four legs | UNUSABLE |
| negative volume | UNUSABLE |
| rows out of chronological order | UNUSABLE |
| a gap larger than the configured tolerance | UNUSABLE |
| no rows at all | UNUSABLE |
| duplicate timestamps (dropped, but the file was wrong) | DEGRADED |
| a gap within tolerance | DEGRADED |
| everything present and consistent | OK |

Two checks were genuinely absent before, not merely unreachable: the OHLC
consistency test compared `high` and `low` only against `close`, so a bar whose
**open** sat outside its own range passed; and non-positive prices were checked
on `open` and `close` but not on `high` or `low`.

The decision table the gate now applies, in full:

| condition | without `--allow-degraded` | with it |
| --- | --- | --- |
| no symbols loaded | REFUSED | REFUSED |
| an UNUSABLE series | REFUSED | REFUSED |
| a required timeframe absent | REFUSED | REFUSED |
| a DEGRADED series | REFUSED | UNTRUSTED |
| no exchangeInfo | UNTRUSTED | UNTRUSTED |
| funding enabled, no funding history | UNTRUSTED | UNTRUSTED |
| clean | TRUSTED | TRUSTED |

Two properties hold by construction and are tested directly:
`--allow-degraded` **cannot** rescue structurally corrupt data — before V3.2 it
could, turning a refusal into a running backtest over impossible bars — and **no
input condition reaches TRUSTED except a clean one**. An override can only move
REFUSED to UNTRUSTED.

**Tests.** `TestP0TheQualityContractIsShared` (6) and
`TestP0DamagedDataIsRefused` (12), including the four corruptions the brief
names driven through the real `tradebot backtest` process, and a property test
that sweeps every dirty-quality shape against both flag settings asserting none
of them reaches TRUSTED.

## P0 — walk-forward did not use the trust gate at all

**Problem.** `run_walkforward` called `load_dataset(..., strict=False)` and went
straight to the analyser. The dataset `tradebot backtest` refused would run to
completion under `tradebot walkforward` and print a verdict.

**Fix.** Both commands now go through one `_load_and_trust()` helper — the trust
logic is not duplicated, it is called twice. The walk-forward report carries the
`TrustReport`, prints `Data trust` in its summary, exposes `data_trust` in its
JSON, and prepends a warning when the level is not TRUSTED. A report produced by
a direct caller that skipped the gate reads `NOT EVALUATED`, never `TRUSTED`.

**Tests.** `TestP0WalkForwardUsesTheSameGate` (4), including a structural check
that `run_walkforward` contains no second copy of the rules.

## P0 — `opened_at` was the signal time, not the fill time

**Problem.** `_open_position` filled at the next decision bar's open — correct,
and deliberately so — then stamped the position `opened_at=timestamp`, the
timestamp of the bar that produced the *signal*. Every position appeared to have
been opened one decision interval before it existed.

That is not cosmetic. `opened_at` is the origin for trade duration, for the
3600-second maximum-hold cap (positions were force-closed one interval early,
every time), for funding eligibility, and for every timing statistic.

**Fix.** Three timestamps, named and separated:

* `signal_at` — the decision point; every bar behind it had already closed.
* `order_at` — submission. In the backtest this equals `signal_at`; the
  simulator's latency assumption is priced into the fill, not the clock.
* `filled_at` — the open of the next decision bar.

`Position.opened_at` and `Trade.opened_at` now **are** `filled_at`. `signal_at`
and `order_at` are explicit fields on both, defaulted so nothing older breaks,
and `Trade.signal_to_fill_sec` exposes the delay. `_next_open` became
`_next_fill`, returning the bar's `open_time` alongside its price, so the fill
timestamp is derived from the same bar as the fill price and is deterministic.

**Tests.** `TestP0OpenedAtIsTheFillTime` (9), including the brief's own example
(signal 12:00 → fill 12:01 → `opened_at == 12:01`), the maximum-hold boundary
either side of the cap, and an AST check that the `Position(...)` call passes
`filled_at` to `opened_at` and `timestamp` to `signal_at`.

## P1 — funding timing was computed from an assumed schedule

**Problem.** Execution accounting already charged real historical funding events
(V3.1). Two other places did not:

```python
interval = self.config.edge.funding_interval_hours * 3_600_000
bucket = (timestamp // interval) * interval
return data.funding_rates.get(bucket, 0.0)     # _funding_rate
return (interval_ms - (timestamp % interval_ms)) / 1000.0   # _seconds_to_funding
```

`_funding_rate` snapped the timestamp onto an assumed 8-hour grid and looked
that exact bucket up. Real funding timestamps do not land on that grid — the
bulk archive stores calculation times, and Binance runs 4-hour funding on many
symbols — so the lookup missed and returned `0.0`. A symbol with a complete
funding history was scored as if funding did not exist.
`_seconds_to_funding` never consulted the data at all; it produced a number from
the invented schedule and fed it to the edge model.

**Fix.** `data.funding_rates` is the single source of truth. `_funding_rate`
returns the most recently *settled* rate at or before the timestamp — which is
what `premiumIndex.lastFundingRate` means live — found by bisection over the
symbol's actual event times. `_seconds_to_funding` counts to the actual next
event. Where no event follows, it returns **infinity**: the edge model reads
that as "no funding falls inside the expected hold", which is what not knowing
should mean. No boundary is ever invented, and the trust gate separately
downgrades any run whose funding history is missing, so the silence is always
reported.

**Tests.** `TestP1FundingComesFromTheRealSchedule` (7), including an off-grid
event at `START + 3600000 + 137` that the old bucket lookup could not find, and
a structural check that `funding_interval_hours` no longer appears in either
method.

## P1 — walk-forward silently ran a different configuration

**Problem.** `WalkForwardAnalyzer.run` called
`BacktestEngine(self.config).run(data, start, end)` — no capital, no execution
assumptions, no seed. It therefore used the engine's fallbacks: the BASE
scenario and **seed 0**, while the headline backtest used the configured seed.
Two runs of "the same" system, quietly differing in their slippage draws.

**Fix.** Every execution input is now named in `WalkForwardAnalyzer.__init__`
and recorded in the report (`scenario`, `seed`, `initial_capital`). The
assumptions table is built exactly as `run_scenarios()` builds it, from the same
`config.backtest`. The CLI passes the same seed as the headline run.

What is **identical** by construction, because it is reached through one
`TunableConfig` and one `BacktestEngine`: fees, spread, slippage, latency,
rejection and partial-fill behaviour, funding, risk sizing, maximum concurrent
positions, leverage, the maximum-hold cap, and the strategy set.

What **intentionally differs**, stated in the docstring so the comparison is
never implicit: walk-forward runs **one** scenario per fold rather than all
three. Three scenarios across N folds costs 3N engine runs to answer a question
about consistency across time, which is what walk-forward is for; scenario
sensitivity is what `backtest` measures.

**Tests.** `TestP1WalkForwardMatchesTheBacktest` (4).

## P1 — the machine-readable data quality artifact

Every run now writes `<report>.data_quality.json` beside its report, with one
row per symbol/interval carrying `SYMBOL`, `INTERVAL`, `START`, `END`, `ROWS`,
`MISSING`, `DUPLICATES`, `GAPS`, `COVERAGE` and `QUALITY_STATUS`, plus the full
trust verdict and per-series detail. It is written **before** the refusal check,
so a refused run still leaves the evidence explaining why.

**Tests.** `TestP1TheQualityArtifact` (4), `TestTheRealCommandsRunEndToEnd` (3).

## V3.2 verification — what actually ran

Every gate below was executed against the tree at this commit. Results are
reported as they came back, including the one that could not run here.

| Gate | Command | Result |
| --- | --- | --- |
| Full suite | `pytest -q` | PASS — 1262 tests, 0 failures |
| V3.2 regressions | `pytest tests/integration/test_v32_correctness.py -q` | PASS — 49 tests |
| V3.1 regressions | `pytest tests/integration/test_v31_correctness.py -q` | PASS — 62 tests |
| Lint | `ruff check src tests scripts` | PASS |
| Format | `ruff format --check src tests scripts` | PASS — 140 files |
| Types | `mypy src/tradebot` (strict via `pyproject.toml`) | PASS — 93 files, 0 errors |
| Static analysis | `bandit -r src -c pyproject.toml -q` | PASS — 0 findings |
| Dependency audit | `pip-audit -r requirements.txt --strict` | PASS — no known vulnerabilities |
| Secret scan | `python scripts/check_secrets.py` | PASS — 161 files clean |
| `walkforward` end to end | `tradebot walkforward --data … --report …` | PASS — exit 0 |
| Docker image build | `docker build -f docker/Dockerfile -t tradebot:ci .` | **failed in the sandbox**, PASS on GitHub Actions — see below |
| Docker smoke tests | all six checks from the CI `build` job | PASS — on CI, and on a stand-in image locally |
| GitHub Actions | run 33187472334 on `c0f7fdc` | PASS — all six jobs green |

### The `walkforward` run, verbatim

The change is visible in the first four lines of output — none of which existed
before V3.2, because the command had no trust gate and recorded neither its
scenario nor its seed:

```
  Data trust          TRUSTED
  quality artifact    /tmp/wf.data_quality.json
Data trust             TRUSTED
Scenario / seed        BASE / 42
```

The verdict was `INCONCLUSIVE — only 1.0 days of data; a single fold needs 44
days`, which is the correct answer for the fixture dataset and not a result
about the strategies.

### The Docker build genuinely failed, and it is the environment

`docker build -f docker/Dockerfile -t tradebot:ci .` was attempted and returned
**exit 100**:

```
Err:1 http://deb.debian.org/debian trixie InRelease
  403  Forbidden [IP: 151.101.194.132 80]
E: Failed to fetch http://deb.debian.org/debian/dists/trixie/InRelease
The command '/bin/sh -c apt-get update && apt-get install -y ...' returned a non-zero code: 100
```

Both `deb.debian.org` and `docker.io`'s blob CDN return `403 Forbidden` through
this sandbox's egress proxy — confirmed directly with `curl`, outside Docker —
so the multi-stage build cannot fetch its base layers or its build toolchain.
Nothing in the project is implicated.

**The shipped Dockerfile is therefore NOT verified in this environment.** What
was verified is a stand-in image carrying the same `src/`, `config/`,
`scripts/` and `docker/entrypoint.sh`, with the same `ENTRYPOINT`/`CMD`
contract, built from a reachable base mirror without the apt layers. All six
checks the CI `build` job runs after the build pass against it:

1. `validate-config` in PAPER with no credentials — exit 0
2. `doctor` with no credentials and no network call — exit 0
3. default `CMD` is exactly `["run"]`
4. `TRADING_MODE=LIVE` without the acknowledgement — exit **78**, `REFUSING TO START`
5. `TRADING_MODE=LIVE` against testnet with the acknowledgement — exit **78**
6. the image runs as `tradebot`, not root

That proves the CLI contract the smoke tests exercise, and nothing about the
real build. GitHub Actions is the authority for that, and it has now answered.

### GitHub Actions, run 33187472334 on `c0f7fdc`

All six jobs green, and the Docker job built the **shipped** `docker/Dockerfile`
in 54 seconds, then passed every smoke step:

| Job | Result |
| --- | --- |
| Lint & type check | success |
| Tests (unit) | success |
| Tests (integration) | success — 10m01s, including the subprocess CLI tests |
| Tests (failure) | success |
| Security checks | success |
| Docker build | success — image built, all six smoke steps passed |

So the shipped Dockerfile does build. What the sandbox failure means is only
that this environment cannot reach `deb.debian.org` or `docker.io`'s blob CDN;
it was never evidence about the Dockerfile itself, and the stand-in image was
the substitute for a check that has now been run for real.

### Status

**ENGINEERING / CORRECTNESS VERIFIED.**
**PROFITABILITY NOT MEASURED.**

No strategy was added, no parameter was tuned, no threshold was moved. The
purpose of V3.2 was to make the instrument trustworthy before it is pointed at
real Binance history for the first time.

## Known issue (documented, not fixed) — rejection-counter merge collision

`engine.py` merges the pipeline's and the risk engine's rejection counters with
`{**self.pipeline.rejections, **self.risk.rejections}`. Dict unpacking is
last-wins, and two reasons are emitted by **both** counters — `LOW_OPPORTUNITY_SCORE`
(`signals/pipeline.py` and `risk/engine.py`) and `INVALID_STOP` (both). Where
both fire, the pipeline's count is silently replaced by the risk engine's. The
report then shows a plausible integer that is one stage's count presented as the
total, with nothing indicating the loss.

It has not bitten any run so far: the risk-side score check is guarded by
`preservation.enabled` and compares against a preservation-*raised* bar, which
in NORMAL mode equals the base threshold the pipeline already enforced, so the
risk stage records nothing. Both the June 2024 smoke run and the 7-day
diagnostic close their funnels to the exact trade count with zero residual,
which would not happen if a stage had been overwritten.

It remains a latent defect because the masking is state-dependent: on a run with
a real drawdown, preservation engages, the risk stage starts rejecting, and one
funnel stage becomes under-reported precisely when the run is most worth
studying. Severity P1 **reporting** — no effect on trading logic, PnL or the
trust gate; the counters are observational.

The fix is to keep the two counters separate in the report rather than merging
them, since summing would destroy the stage attribution the funnel depends on.
Deliberately not applied yet.
