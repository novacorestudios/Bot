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
