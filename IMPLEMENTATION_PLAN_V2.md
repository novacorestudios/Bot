# IMPLEMENTATION_PLAN_V2.md

Hardening plan for the findings in [`AUDIT_REPORT.md`](AUDIT_REPORT.md).

**Nothing is rewritten.** The risk engine, strategies, backtester, edge filter
and execution safety are sound and stay as they are. This plan wires up what is
disconnected, fixes what is broken, and adds the four missing subsystems.

Everything new is **configurable**, and every default is chosen to be *no more
aggressive than the current behaviour* — a hardening pass must not quietly
change trading behaviour under the guise of fixing bugs.

---

## Phase order

Ordered by dependency, and by "what would hurt most if it stayed broken".

### V2-1 — Build and test environment  *(fixes C-4, C-5, L-1, L-4)*

- install the package in editable mode; add `pythonpath = ["src"]` to
  `pyproject.toml` so `pytest` works from the root
- remove both `|| true` from CI; make `mypy` and `pip-audit` real gates
- add a `Makefile` so the documented commands are one word

*Exit:* `pytest -q` passes from a clean checkout with no `PYTHONPATH`.

### V2-2 — WebSocket correctness  *(fixes C-2)*

- rewrite the message parser to dispatch on payload *shape* before content, so
  a list can never reach `.get()`
- unit tests using the **documented payload shapes** for kline, bookTicker,
  markPrice (array **and** single), and the user-stream events

*Exit:* every documented payload parses; the array form is covered by a test
that would have caught C-2.

### V2-3 — Live market state  *(fixes C-1, M-1, M-2, M-4)*

Introduce `market/state.py` — a single `MarketState` owning candles, book
tickers, mark prices and **freshness per symbol**. Both WebSocket and REST write
into it; everything downstream reads only from it.

```
WebSocket ──┐
            ├──> MarketState ──> Scanner ──> Strategies ──> Opportunities
REST     ───┘   (freshness)
```

- start `MarketStream` and `UserStream` in `runner._build()`
- resubscribe on the top-25 rotation
- REST becomes: initial sync, metadata, account, recovery, reconciliation,
  history, and **fallback when a symbol goes stale**
- a WebSocket reconnect triggers reconciliation, not just a resubscribe

*Exit:* an integration test asserts the engine's data path is WebSocket-fed and
that stale symbols are excluded from entry.

### V2-4 — Order state machine and user stream  *(fixes H-1, and completes C-1)*

- `execution/state.py`: explicit legal transitions
  `CREATED → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED`, plus
  `CANCEL_REQUESTED → CANCELLED`, `REJECTED`, `EXPIRED`, `FAILED`
- illegal transitions are refused and logged, so a late REST response cannot
  move a `FILLED` order back to `NEW`
- `ORDER_TRADE_UPDATE` from the user stream drives fills; REST becomes
  confirmation rather than discovery

*Exit:* a test drives out-of-order updates from two sources and asserts the
terminal state is correct.

### V2-5 — Database as a critical dependency  *(fixes C-3, M-6)*

- reclassify `database` as critical → its failure enters safe mode
- add reconnect with backoff; recovery re-enables entries only after
  reconciliation

*Exit:* a test kills the database mid-run and asserts entries stop and resume
only after both recovery and reconciliation.

### V2-6 — Opportunity queue  *(fixes H-4)*

`signals/queue.py`: scored, expiring, deduplicated.

- opportunities carry an `expires_at` derived from data freshness
- the queue is drained **best-first**, not scanner-rank-first
- an opportunity whose data went stale is dropped, never executed

*Exit:* a test asserts best-first ordering and that stale entries expire unused.

### V2-7 — Capital preservation modes  *(fixes H-5, M-7)*

`risk/preservation.py`: `NORMAL → CAUTIOUS → DEFENSIVE → HALTED`, driven by
drawdown and consecutive losses. Each mode scales risk, max positions, minimum
score and minimum edge.

Plus a configurable **capital reserve** so the full balance is never tradable.

Defaults are set so `NORMAL` reproduces today's behaviour exactly.

*Exit:* a test walks equity down through every transition and back up.

### V2-8 — Execution quality and edge calibration  *(fixes H-6, and §26/§37)*

- record expected vs actual entry/exit and slippage per trade
- an `ExecutionQuality` score per symbol and strategy
- `EdgeCalibration`: predicted edge vs realised edge, so "is the model
  optimistic?" is answerable from data

*Exit:* a test asserts the calibration detects a systematically optimistic model.

### V2-9 — Missing exits  *(fixes H-2, H-3)*

Implement `exit_on_signal_flip` and `exit_on_negative_edge`; add a local
stop/target safety net so a missing protective order is not fatal.

*Exit:* tests for each exit path.

### V2-10 — Performance matrices  *(fixes M-3)*

Strategy×regime and symbol×strategy tables, surfaced on the dashboard. Recorded
for **diagnosis**, explicitly not fed back into selection — that would be
overfitting with extra steps.

### V2-11 — Full verification

Run everything; write `FINAL_AUDIT.md` with real, measured status.

---

## Rules for this work

1. **No behaviour change disguised as a fix.** Every new parameter defaults to
   the current behaviour. Where a default must change (database criticality),
   it is called out.
2. **No deletions.** If something is unused, wire it up or document why not.
3. **Integration over unit.** C-1 existed *because* the unit tests passed. Every
   phase adds a test that asserts the engine *uses* the thing.
4. **NOT VERIFIED stays NOT VERIFIED.** This plan adds no profitability
   evidence, and the sandbox still has no route to Binance.

---

## Explicitly out of scope

- changing strategy logic or risk parameters on a hunch
- adding strategies
- news trading
- giving the AI layer any authority
- anything that goes live
