# FINAL_AUDIT.md

**Subject:** Dynamic Multi-Strategy Binance USDⓈ-M Futures Scalping Engine
**Baseline audited:** `582ba7f` — see [`AUDIT_REPORT.md`](AUDIT_REPORT.md)
**Hardening plan:** [`IMPLEMENTATION_PLAN_V2.md`](IMPLEMENTATION_PLAN_V2.md)
**Method:** every claim below is either the output of a command in this document
or is marked **NOT VERIFIED**. Nothing is asserted from reading code.

---

## The headline

**This system has never traded. It has no profitability evidence of any kind.**

There is no backtest on real data, no paper-trading record, and no live record,
because this environment has no route to Binance — every `api.binance.com` and
`fapi.binance.com` connection is refused by the sandbox proxy. That is a fact
about the environment, not a judgement about the code, and no amount of code
quality substitutes for it.

What *is* verified is narrower and worth stating precisely: the engine is
internally consistent, its safety properties are enforced by tests rather than
by convention, and the twenty-two defects found in the V1 audit are fixed.
**"Correct" is not "profitable."** A system can be flawlessly engineered and
still lose money, and until it has traded, that is exactly the position here.

---

## Verification status

| Claim | Status |
|---|---|
| Code passes lint, format, type and security gates | **VERIFIED** — commands below |
| Every test in the suite passes | **VERIFIED** — command below |
| The 5 CRITICAL findings are fixed | **VERIFIED** — each with a test |
| The 6 HIGH findings are fixed | **VERIFIED** — each with a test |
| The engine connects to Binance | **NOT VERIFIED** — network blocked |
| Strategies produce signals on real market data | **NOT VERIFIED** |
| Any backtest result, win rate or Sharpe ratio | **NOT MEASURED — none exists** |
| Paper trading works end to end against the exchange | **NOT VERIFIED** |
| The system is profitable | **NOT VERIFIED, AND NOT CLAIMED** |
| The container ENTRYPOINT/CMD contract | **VERIFIED** — exercised in a real container, both the correct and the broken form |
| The **shipped** `docker/Dockerfile` builds | **NOT VERIFIED** here — `deb.debian.org` is blocked; CI covers it |

---

## Gate results

Run from a clean checkout at the current commit:

```bash
make check          # lint + format + type + test + security
```

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check src tests scripts` | **All checks passed** |
| Format | `ruff format --check src tests` | **All files formatted** |
| Types | `mypy src/tradebot` (strict) | **No issues in 83 source files** |
| Security (code) | `bandit -r src/tradebot` | **0 issues, all severities** |
| Security (deps) | `pip-audit -r requirements.txt --strict` | **No known vulnerabilities** |
| Tests | `pytest -q` | **1024 passed, 0 failed, 0 skipped** |

`mypy` is strict and clean. That is worth one sentence of context: at the start
of V2 it reported 150 errors, and every one was fixed rather than suppressed —
there is no `# type: ignore` added in this work that hides a real defect, and
several of the 150 were genuine bugs.

`bandit` reports zero findings at every severity. The last one to go was an
`assert` in the exit evaluator: `python -O` strips asserts, so that line would
have become an `AttributeError` in production and nowhere else. It was replaced
with real type narrowing, not with a `# nosec`.

---

## The V1 findings, one by one

### CRITICAL

| ID | Finding | Status | Evidence |
|---|---|---|---|
| C-1 | The engine never starts the WebSocket layer | **FIXED** | `TradingEngine._start_streams` constructs and starts `MarketStream` and `UserStream`; `tests/integration/test_market_data_path.py` asserts it |
| C-2 | Mark-price stream crashes on its first real message | **FIXED** | `exchange/binance/parsers.py` dispatches on payload shape; `tests/unit/test_ws_parsers.py` parses the exact array payload that raised |
| C-3 | A database failure does not stop trading | **FIXED** | `database` is a critical component; `tests/unit/test_database_criticality.py` |
| C-4 | `pytest` cannot run from the repository root | **FIXED** | `pythonpath = ["src"]`; `tests/unit/test_build_environment.py` runs a subprocess import with `PYTHONPATH` unset |
| C-5 | CI hides failures with `\|\| true` | **FIXED** | both removed; `test_build_environment.py` greps the workflow to keep them out |

### HIGH

| ID | Finding | Status | Evidence |
|---|---|---|---|
| H-1 | No order state machine | **FIXED** | `execution/state.py`; `tests/unit/test_order_state.py` drives out-of-order updates from two sources |
| H-2 | Two configured exits not implemented | **FIXED** | `execution/exits.py`; `tests/integration/test_feedback_loops.py` asserts each flag is read |
| H-3 | `_exit_reason` ignores the price | **FIXED** | the local stop/target safety net; `tests/unit/test_exits.py` |
| H-4 | No opportunity queue | **FIXED** | `signals/queue.py`, drained best-first into free slots only |
| H-5 | No capital preservation modes | **FIXED** | `risk/preservation.py` |
| H-6 | No execution quality measurement | **FIXED** | `execution/quality.py`, fed back into the cost model |

### MEDIUM

| ID | Finding | Status |
|---|---|---|
| M-1 | No REST fallback when the WebSocket goes stale | **FIXED** — `_rest_backfill` |
| M-2 | Scanner re-fetches klines a stream would keep warm | **PARTIALLY FIXED** — the top-25 is stream-fed; the full-universe re-rank is still REST, which is correct: you cannot subscribe to 400 symbols to decide which 25 to subscribe to |
| M-3 | No performance matrices | **FIXED** — `risk/matrices.py` |
| M-4 | Reconciliation not triggered by a reconnect | **FIXED** — the reconcile loop waits on a reconnect event |
| M-5 | Paper broker's resting orders are polled | **NOT FIXED** — deliberate; see below |
| M-6 | `Repository` has no reconnect logic | **FIXED** — reconnect with backoff, buffer preserved |
| M-7 | No capital reserve | **FIXED** — `capital_reserve_fraction`, default 10% |

**M-5 is deliberately left as it is.** Making the paper broker event-driven
would mean writing a second execution path that only ever runs in paper mode —
and a simulator that diverges from the live path is worse than one that is
honestly slower, because it would let a bug pass paper testing and fail live.
The polling is a known, documented timing difference, not a hidden one.

### LOW

| ID | Finding | Status |
|---|---|---|
| L-1 | `mypy` has never passed | **FIXED** — strict and clean |
| L-2 | Dashboard polls 8 endpoints | **NOT FIXED** — cosmetic; the endpoint count grew to 12 with this work |
| L-3 | `ARCHITECTURE.md` describes a pipeline that did not exist | **FIXED** — it now describes the one that does |
| L-4 | No `Makefile` | **FIXED** |

---

## What changed, and why it matters

### The data path is now live (C-1, C-2, M-1, M-2, M-4)

Before: every decision was made from a REST poll on a 15-second loop, and the
WebSocket classes that shipped with the project had no caller. A scalper acting
on 15-second-old prices is trading a market that has already moved, and the
resulting losses look like strategy failure rather than what they are.

Now: `market/state.py` owns candles, book tickers, mark prices — and **freshness
per symbol**. Both transports write into it; nothing downstream knows or cares
which one supplied a number, but everything can ask how old it is.

The asymmetry is the important part: **a stale symbol is refused for ENTRY and
still permitted to EXIT.** Acting late on an exit is far better than not acting.

One design note worth recording: freshness deliberately ignores mark price. The
`!markPrice@arr@1s` stream ticks every second for every symbol on the exchange,
so counting it would make a symbol whose kline *and* book streams were both dead
look perfectly healthy — and keep taking entries on it.

### Orders have a governed lifecycle (H-1)

`execution/state.py` enforces three rules, each of which corresponds to a real
way a bot loses track of a real position:

1. **Terminal is final.** REST polling and the user stream race constantly; a
   stale `NEW` arriving after a `FILLED` must not resurrect the order.
2. **Filled quantity only moves forward.** Out-of-order updates would otherwise
   walk a fill backwards and make the engine size an exit for less than it holds.
3. **`INDETERMINATE` is a state, not an error.** A timed-out submission may or
   may not have reached the exchange. A duplicate position and an unprotected
   one are both worse than blocking entries until reconciliation settles it.

### The database is a real dependency (C-3)

The brief lists "continuing new entries during a database failure" among the
things the engine must never do. It now stops entries and never stops exits:
refusing to close a position because a log write failed would be far more
dangerous than the missing row. The repository reconnects with backoff, keeping
its buffer, so the audit trail resumes where it stopped instead of restarting at
recovery.

### Opportunities are ranked before they are spent (H-4)

Scan rank measures how *tradable* a symbol is. It is not a ranking of how good
this particular trade is. With four slots and twenty-five candidates, the old
first-past-the-post ordering routinely spent a slot on an adequate trade while a
much better one waited behind it. The queue scores everything first, drains
best-first into however many slots are actually free, and expires entries — a
signal computed on a 5-minute bar is not still valid ten minutes later.

**Zero free slots means zero trades.** The queue has no path to an order.

### Losing streaks now reduce size (H-5, M-7)

On 75 USDT a 10% drawdown is 7.50 USDT — fifteen trades' worth of risk at 0.5% —
and recovering it needs an 11% gain. Losses compound against you faster than
gains compound for you, so the response to a bad run has to be to risk less.

`NORMAL → CAUTIOUS → DEFENSIVE → HALTED`, with two properties that matter more
than the thresholds:

- **Escalation is immediate; relaxing must be earned.** Loosening requires both
  a minimum dwell time and recovery past a hysteresis band. Without the band, a
  drawdown oscillating around a threshold flips the mode every cycle and hands
  back full size in exactly the conditions that shrank the account.
- **No mode gates an exit.** Not even `HALTED`.

A capital reserve (10% by default) is never deployed. It is not idle money: it
pays funding, fees and adverse margin moves on positions already open.

### The cost model now learns (H-6)

Every input to the edge filter is an estimate. If those estimates are
systematically optimistic, the filter approves trades that were never profitable
— and the losses read as strategy failure rather than as the measurement error
they are. `execution/quality.py` measures the gap and feeds it back, so the
threshold a trade must clear rises to meet reality.

Median rather than mean, so one news-spike fill does not distort hours of
estimates. A minimum sample before any adjustment, because three fills are not
evidence of a bias. And the correction can only ever make the model **more**
careful: a lucky run of fills must never make the edge filter more permissive.

### The matrices diagnose; they do not select (M-3)

`risk/matrices.py` records strategy × regime and symbol × strategy. An aggregate
win rate cannot distinguish "this strategy is weak" from "this strategy keeps
being run in the wrong regime", and those call for opposite responses.

Feedback into selection is **off by default**, per the plan. On a 75 USDT
account taking a few trades a day, a cell takes weeks to fill, and suppressing a
combination on thin data is overfitting against your own history. Even switched
on, a matrix may only ever *reduce* a weight — a cell that looks excellent on a
dozen trades is far more likely to be luck than skill, and betting up on it is
how a small account turns a good run into a drawdown.

---

## The non-negotiable rules, and how each is enforced

Not by convention — by a test that fails if the property is broken.

| Rule | Enforcement |
|---|---|
| No strategy may bypass the Risk Engine | `OrderIntent` is constructed in exactly one file. A test greps the whole source tree and fails if a second constructor appears. |
| The AI layer has no order authority | An AST test asserts the AI package imports nothing from `execution` or `exchange`. |
| No position without a protective stop | If the stop cannot be placed, the position is closed immediately — a position that cannot be protected is worse than no position. |
| An exit is never blocked | Tests assert the close path reads no kill switch, no safe-mode flag, no preservation mode. |
| Never force a trade | Zero opportunities produces zero trades; zero free slots produces zero offers. There is no minimum trade count anywhere in the system. |
| No hardcoded coins | The universe is built from `exchangeInfo` every scan. A test asserts no symbol literal appears in the scanner or universe builder. |
| Max trade duration ≤ 60 minutes | Enforced by config validation and by the exit evaluator. |
| Expected net edge is mandatory | The edge gate is in the pipeline before risk, and rejects on cost, spread, slippage and funding. |
| No secrets in the repository | Credentials come only from the environment; `.env.example` carries no real values; a test asserts no key-shaped literal exists in the source. |

---

## Test suite

```bash
pytest -q
# ........................................................ [100%]
# 1024 passed
```

1024 test functions across 32 files, every one passing. The composition matters
more than the count, and V2's additions are deliberately weighted toward
integration:

**The single most important lesson from the V1 audit** is that C-1, H-2, H-4 and
M-1 all had the same root cause — components were built, unit-tested and then
never wired into the orchestrator. The WebSocket classes worked. Their tests
passed. Nothing called them.

So every V2 phase adds a test that asserts *the engine uses* the thing:

- `tests/integration/test_market_data_path.py` — the engine constructs and
  starts the streams, reads only through `MarketState`, and excludes stale
  symbols from entry
- `tests/integration/test_capital_discipline.py` — evaluation queues instead of
  trading inline; the risk engine consults the preservation mode
- `tests/integration/test_feedback_loops.py` — every `exit_on_*` flag is read;
  execution quality is recorded and fed back; the matrices are consulted

A unit test asserting a component works is worth less than an integration test
asserting the engine uses it.

---

## What has NOT been verified

Stated bluntly, because the brief requires it and because it is the most
important section in this document.

1. **No connection to Binance has ever been made.** The sandbox blocks it.
   Signing, rate limiting and filter arithmetic are tested against recorded
   payload shapes from the official documentation — never against the live API.
2. **No backtest has been run on real market data.** The backtester is tested
   for its own correctness (no look-ahead, pessimistic intrabar fills, correct
   metrics) against synthetic series. It has never been given real klines.
3. **There is no win rate, Sharpe ratio, profit factor or expectancy**, because
   there are no trades. Any number of that kind in this repository would be
   fabricated.
4. **Paper trading has not been run end to end.** The paper broker is unit
   tested; it has never been driven by a live feed.
5. **The shipped image has not been built here.** The sandbox blocks
   `deb.debian.org` and Docker Hub's blob CDN, so the real Dockerfile's `apt`
   layers cannot run. The ENTRYPOINT/CMD contract *was* verified in a real
   container built from a local stand-in that copies `entrypoint.sh`,
   `ENTRYPOINT`, `CMD`, `WORKDIR`, `PYTHONPATH` and the non-root user verbatim
   from the shipped Dockerfile — so what was proven is the argument contract and
   the LIVE refusal, not the shipped image's base layers.
6. **The strategies' parameters are starting points, not optimised values.**
   Every threshold in `config/config.yaml` is a guess informed by convention.
   None has been validated.

---

## What must happen before this touches real money

In order. Do not skip.

1. **Get real data.** Run the scanner and the backtester against downloaded
   klines for a period covering at least one trending, one ranging and one
   high-volatility regime.
2. **Backtest, then walk-forward.** In-sample results mean nothing on their own.
   The walk-forward harness exists; use it. Expect the honest outcome to be
   worse than the in-sample one.
3. **Read the break-even arithmetic first.** `p = (loss + costs) / (win + loss)`
   is on the dashboard for a reason. If a strategy's average winner is smaller
   than its round-trip cost, no hit rate saves it, and no amount of tuning will.
4. **Paper trade for weeks, not days**, on testnet, and compare realised edge
   against expected edge. The execution-quality report answers this directly.
   If the model is optimistic, fix the model before risking capital.
5. **Then, if and only if the evidence supports it**, go live with the smallest
   size the exchange permits — not with 75 USDT.

If step 2 or step 4 says the edge is not there, the correct outcome is **not to
trade**. That is a successful use of this system, not a failed one.

---

## Honest assessment

**What is good.** The risk engine is genuinely well-structured: one construction
site for order intents, no exit path through it, stop-distance-based sizing, and
a correlation engine that measures effective independent positions rather than
waving at pairwise correlation. The edge filter is honest about its own
uncertainty, including a bootstrap mode that says so out loud. Execution safety
— idempotent client order ids, indeterminate-state handling, the emergency close
on an unplaceable stop — is the part I would trust most.

**What is unproven.** Everything about whether it makes money. The strategies
are conventional implementations of conventional ideas; there is no reason to
assume an edge exists in them, and the honest prior for a retail scalping system
after costs is that it does not.

**What worries me most.** Not a bug — the cost structure. At 4 bps taker each
way plus spread and slippage, a round trip costs roughly 10–15 bps, so a
strategy needs to be right often enough to clear that before it makes a cent.
The engine measures this and refuses trades that do not clear it, which is the
right design. Whether enough opportunities clear it to be worth running is an
empirical question this environment cannot answer.

**The system will not stop you from losing money.** It will stop you from losing
it in the specific ways that are preventable by engineering: stale data,
unprotected positions, runaway drawdowns, orders it has lost track of, trades
with no audit trail. That is a real and worthwhile thing. It is not an edge.

---

## Reproducing this audit

```bash
git clone <repo> && cd Bot
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

make check              # every gate
.venv/bin/pytest -q     # the suite alone
.venv/bin/mypy src/tradebot
.venv/bin/bandit -q -r src/tradebot
.venv/bin/pip-audit -r requirements.txt --strict
```

Anything this document claims that those commands contradict is a defect in this
document. Report it.
