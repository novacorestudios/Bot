# AUDIT_REPORT.md

**Subject:** Bot-Claude-Binance-Futures-Scalping-Engine
**Commit audited:** `582ba7f`
**Method:** every finding below was reproduced by executing code, not by reading
it. The reproduction command is given for each.

> **Status: superseded.** This document records the state of the system as
> audited, and is kept unedited so the findings can be checked against what was
> actually fixed. For the resolution of each finding, and for the current
> verification status, see [`FINAL_AUDIT.md`](FINAL_AUDIT.md). The hardening
> plan is [`IMPLEMENTATION_PLAN_V2.md`](IMPLEMENTATION_PLAN_V2.md).

---

## Summary

| Severity | Count |
|---|---|
| CRITICAL | 5 |
| HIGH | 6 |
| MEDIUM | 7 |
| LOW | 4 |

The system is **structurally sound but operationally incomplete**. The risk
engine, strategy engine, backtester and safety gates are real and tested. The
gap is in the *live data path*: the engine polls REST for everything, the
WebSocket layer it ships with is never started, and several configured
behaviours are not wired to any code.

The blunt version: **this would run, and it would trade on data up to 15 seconds
old, and it would crash the moment a mark-price stream was connected.**

---

## CRITICAL

### C-1 — The trading engine never starts the WebSocket layer

`app/runner.py` declares `self.market_stream`, and stops it on shutdown, but
**never constructs or starts it**. `UserStream` is not referenced anywhere
outside its own module. All market data comes from REST polling on a
`signal_interval_sec` (15 s) loop, and order fills are discovered only by the
60-second reconciler.

```bash
grep -n "market_stream\|UserStream" src/tradebot/app/runner.py
# 78:  self.market_stream: Any = None      <- declared
# 310: if self.market_stream is not None:  <- stopped
#      ...never assigned, never started
```

**Impact.** Decisions are made on data that can be 15 seconds stale — an
eternity for a strategy holding positions for minutes. A stop fill is not known
for up to 60 seconds, during which the engine believes it holds a position it
does not. Both directly contradict the design in `docs/ARCHITECTURE.md`, which
describes a WebSocket-driven pipeline.

### C-2 — Mark-price stream crashes on its first real message

`ws.py::MarketStream._handle` begins with `data.get("e", "")`. The
`!markPrice@arr@1s` stream delivers a **list**, so the first message raises
`AttributeError` before reaching the list-handling branch further down — which
is therefore unreachable.

```bash
python -c "
import asyncio; from tradebot.exchange.binance.ws import MarketStream
async def noop(*a): pass
asyncio.run(MarketStream('wss://x', noop, on_mark=noop)._handle('!markPrice@arr@1s',
    [{'e':'markPriceUpdate','s':'BTCUSDT','p':'60000','r':'0.0001','T':0,'E':0}]))"
# AttributeError: 'list' object has no attribute 'get'
```

Undetected because C-1 means the stream is never connected. Fixing C-1 alone
would have produced an immediate crash loop in production.

### C-3 — A database failure does not stop trading

Requirement: no trading without an audit trail. `HealthMonitor` classifies
`database` as **non-critical**, so a database outage produces a warning and
trading continues.

```bash
python -c "
from tradebot.app.health import HealthMonitor
from tradebot.core.config import HealthConfig
m = HealthMonitor(HealthConfig())
[m.beat(c) for c in m.components]; m.fail('database','disk full')
print('safe_mode =', m.check().safe_mode)"   # False — should be True
```

### C-4 — `pytest` cannot run from the repository root

The package is never installed and `pyproject.toml` sets no `pythonpath`, so the
documented `pytest -q` fails. Every contributor must know to prefix
`PYTHONPATH=src`, and CI only works because it happens to set that variable.

```bash
env -u PYTHONPATH pytest tests/unit/test_types.py -q
# ModuleNotFoundError: No module named 'tradebot'
```

### C-5 — CI hides failures with `|| true`

```
.github/workflows/ci.yml:32  mypy src/tradebot || true
.github/workflows/ci.yml:73  pip-audit -r requirements.txt --strict || true
```

A type error or a newly-disclosed dependency vulnerability produces a green
build. "Red CI blocks merge" is stated in the plan and is not true in practice.

---

## HIGH

### H-1 — No order state machine

`OrderStatus` is an enum with no governed transitions. Nothing prevents an order
moving from `FILLED` back to `NEW` on a late REST response overtaking a
WebSocket update — a real ordering hazard once C-1 is fixed and two sources
report on the same order.

### H-2 — Two configured exit conditions are not implemented

`trade.exit_on_signal_flip` and `trade.exit_on_negative_edge` default to `true`
in `config.yaml`. Neither appears in `runner.py`. The configuration promises
behaviour the engine does not have.

### H-3 — `_exit_reason` ignores the price it is given

`runner.py:575` contains `_ = price`. Stop and target are delegated to resting
exchange orders, which is defensible, but there is no local safety net if a
protective order is missing or was cancelled.

### H-4 — No opportunity queue

Opportunities are evaluated and executed inline, one symbol at a time, in
scanner-rank order. There is no scoring across candidates, no expiry, and no
prioritisation — a score-94 opportunity found late in the loop can be blocked by
a score-71 one found earlier.

### H-5 — No capital preservation modes

Risk is static until a kill switch trips, at which point trading stops entirely.
There is no graduated response (reduce risk → reduce positions → raise
thresholds → halt).

### H-6 — No execution quality measurement

Slippage is recorded per fill for the kill switch, but expected-versus-actual
entry and exit prices are never compared or stored, so "is the loss from the
strategy or from execution?" is unanswerable.

---

## MEDIUM

| ID | Finding |
|---|---|
| M-1 | No REST fallback when the WebSocket goes stale — the engine only blocks entries; it could still poll to manage exits. |
| M-2 | `MarketScanner` re-fetches klines through REST every cycle for symbols that a WebSocket would keep warm. |
| M-3 | No strategy×regime or symbol×strategy performance matrices (requested in the brief, useful for diagnosis). |
| M-4 | Reconciliation runs on a fixed 60 s timer; it is not triggered by a WebSocket reconnect. |
| M-5 | The paper broker's resting orders are polled from the monitor loop rather than being event-driven, so paper and live differ in timing behaviour. |
| M-6 | `Repository` has no reconnect logic — once the engine is marked unavailable it stays unavailable until restart. |
| M-7 | No capital reserve: the full 75 USDT is treated as tradable. |

## LOW

| ID | Finding |
|---|---|
| L-1 | `mypy` has never passed; the debt is unmeasured. |
| L-2 | Dashboard polls 8 endpoints every 5 s where one aggregate would do. |
| L-3 | `docs/ARCHITECTURE.md` describes the WebSocket pipeline as if it exists. |
| L-4 | No `Makefile`/task runner, so the documented commands are long and easy to get wrong. |

---

## What is genuinely good and must not be broken

Stated explicitly so the hardening work does not damage it:

- **The risk engine.** `OrderIntent` is constructed in exactly one file; the
  engine has no exit path; sizing is stop-distance based; the correlation engine
  measures effective independent positions correctly.
- **The edge filter**, including the honest bootstrap mechanism and its loud
  reporting.
- **Execution safety**: idempotent client order ids, indeterminate-state
  handling, mandatory stops, the emergency close on an unplaceable stop.
- **Reconciliation**: adopt-and-protect, phantom resolution from `userTrades`.
- **The backtester's** no-look-ahead guarantees and pessimistic intrabar model.
- **624 tests** encoding real invariants.

None of the above is being rewritten.

---

## Root cause

One pattern explains C-1, H-2, H-4 and M-1: **components were built and tested
in isolation, then not all of them were wired into the orchestrator.** The
`WebSocket` classes, the `exit_on_*` flags and `UserStream` all have correct
implementations and passing unit tests; they simply have no caller.

The lesson for the fix: an integration test that asserts *the engine uses* a
component is worth more than a unit test asserting the component works.
