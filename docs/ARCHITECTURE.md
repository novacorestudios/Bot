# ARCHITECTURE

## 1. Design principles

Ordered, and enforced in that order when they conflict:

1. **Correctness** — every number is derived, checked and reproducible.
2. **Safety** — the system's default action under uncertainty is *do not trade*.
3. **Testability** — no component reaches for the wall clock, the network or a
   global; everything is injected so it can be driven deterministically.
4. **Reliability** — crash and reconnect are normal events, not exceptions.
5. **Performance** — last, and only where measured.

Three structural rules follow from this and are enforced by code review and by
tests:

- **Only the Execution Engine may talk to the exchange for order actions.**
  A strategy that could place an order would be a strategy that could bypass
  risk. Strategies receive read-only market views and return data.
- **Every trade passes the Risk Engine.** There is exactly one function that
  turns an approved opportunity into an order intent, and it is inside the risk
  engine. The AI layer, the dashboard and the scanner have no path around it.
- **No position exists without a protective stop.** The execution engine's
  post-fill routine places the stop; if the stop cannot be placed, the position
  is closed immediately.

## 2. Component map

```
                   ┌──────────────────────────────────────────┐
                   │            Binance USDⓈ-M API            │
                   │   REST (signed)      WebSocket streams   │
                   └───────▲──────────────────────┬───────────┘
                           │                      │
                  orders / account          klines, book, mark
                           │                      │
                   ┌───────┴──────────────────────▼───────────┐
                   │           exchange/binance               │
                   │  rest.py  ws.py  ratelimit.py filters.py │
                   └───────▲──────────────────────┬───────────┘
                           │                      │
        ┌──────────────────┴───────┐    ┌─────────▼────────────┐
        │   execution/engine.py    │    │   market/state.py    │
        │   orders, state machine, │    │  candles, book, mark │
        │   reconcile, exits       │    │  + freshness/symbol  │
        └──────────────────▲───────┘    └─────────┬────────────┘
                           │                      │
                           │                      ▼
                           │            ┌──────────────────────┐
                           │            │ market/scanner.py    │
                           │            │  universe → score →  │
                           │            │  Top-N candidates    │
                           │            └─────────┬────────────┘
                           │                      ▼
                           │            ┌──────────────────────┐
                           │            │ market/regime.py     │
                           │            └─────────┬────────────┘
                           │                      ▼
                           │            ┌──────────────────────┐
                           │            │ strategies/*  (8)    │
                           │            │ regime-gated weights │
                           │            └─────────┬────────────┘
                           │                      ▼
                           │            ┌──────────────────────┐
                           │            │ signals/aggregator   │
                           │            │  consensus, conflict │
                           │            └─────────┬────────────┘
                           │                      ▼
                           │            ┌──────────────────────┐
                           │            │ signals/opportunity  │
                           │            │  score 0..100        │
                           │            └─────────┬────────────┘
                           │                      ▼
                           │            ┌──────────────────────┐
                           │            │ signals/edge.py      │
                           │            │ expected NET edge    │
                           │            └─────────┬────────────┘
                           │                      ▼
                           │            ┌──────────────────────┐
                           └────────────┤ risk/engine.py       │
                             order      │ correlation, budget, │
                             intent     │ sizing, leverage,    │
                                        │ kill switches        │
                                        └─────────┬────────────┘
                                                  ▼
                                        ┌──────────────────────┐
                                        │ database + audit log │
                                        │ telegram + dashboard │
                                        └──────────────────────┘
```

`ai/` sits beside the pipeline as an **observer and advisor**: it reads
snapshots, trades and metrics and can emit a *hint* (e.g. "regime looks
anomalous") that becomes one more input to scoring. It has no order path.

## 3. The decision pipeline, precisely

Per scan cycle (default 15 s for the fast loop, 5 min for the full re-rank).
Steps 2–7 read market data only through `MarketState`, which knows how old every
symbol's view is; a symbol whose feed has gone stale is refused for **entry**
and still permitted to **exit**.

1. **Universe** — `exchangeInfo` filtered to `PERPETUAL`, `TRADING`, quote
   `USDT`, excluding a configurable deny-list and any symbol whose filters make
   the account's minimum position unrepresentable.
2. **Scan** — for every symbol in the universe compute the component scores
   (liquidity, volume, recent volume, spread, volatility, ATR, momentum, trend
   strength, volume anomaly, breakout potential, mean-reversion potential,
   funding, structure, book imbalance, correlation penalty, estimated cost).
   Weighted sum → `market_score`. Sort → **Top N (default 25)**.
   *Nothing is hardcoded to include BTC/ETH/SOL.*
3. **Regime** — classify each candidate: `STRONG_TREND`, `WEAK_TREND`,
   `SIDEWAYS`, `HIGH_VOLATILITY`, `LOW_VOLATILITY`, `BREAKOUT`, `PANIC`.
4. **Strategies** — run only the strategies enabled for that regime, each with
   its regime weight. Each returns a `Signal` (`LONG`/`SHORT`/`WAIT`) with
   confidence, entry, stop, target, reason codes.
5. **Aggregate** — consensus score across strategies; strong disagreement →
   reject.
6. **Opportunity score** — blend market quality, consensus, momentum, volume,
   trend, liquidity, volatility, execution quality and R:R, minus cost and
   correlation penalties. Below `min_opportunity_score` → reject.
7. **Expected net edge** — `p·win − (1−p)·loss − fees − spread − slippage −
   funding`. `≤ min_expected_edge` → reject. This is the last purely-analytical
   gate and it rejects a great many technically valid signals by design.
8. **Queue** — every accepted opportunity is queued and scored rather than
   traded on arrival. The queue is drained **best-first** into however many
   position slots are actually free, and entries expire (`queue_ttl_sec`) so a
   signal computed on a 5-minute bar is never acted on ten minutes later.
   *Scan rank measures how tradable a symbol is, not how good this trade is.*
9. **Risk** — kill switches, **capital preservation mode**, cooldown, portfolio
   budget, exposure caps, correlation, then sizing and leverage. Produces an
   `OrderIntent` or a rejection with a reason code. Every outcome is written to
   the audit log.
10. **Execute** — idempotent client order id, filter-validated order, an
    explicit order state machine, protective stop, reconciliation.
11. **Monitor & exit** — a local stop/target safety net, the 60-minute cap,
    regime change, signal flip, a now-negative edge on *continuing to hold*,
    trailing stops, emergency.
12. **Record** — trade row, strategy metrics, execution quality (expected
    versus actual fill, fed back into the cost model), the performance
    matrices, Telegram, dashboard.

## 4. Concurrency model

Single process, `asyncio`, with these tasks:

| Task | Cadence | Notes |
|---|---|---|
| `ws_market` | continuous | kline + bookTicker + markPrice streams, auto-reconnect with jittered backoff, resubscribes on reconnect |
| `ws_user` | continuous | order/account updates; `listenKey` kept alive every 30 min |
| `scanner_full` | `scan_interval_sec` | full universe re-rank |
| `signal_loop` | `signal_interval_sec` | strategies over the Top-N |
| `position_monitor` | 1 s | exits, trailing, time limit |
| `reconciler` | `reconcile_interval_sec`, **or on a stream reconnect** | REST truth vs local state |
| `health` | 5 s | component heartbeats → safe mode |
| `persistence` | batched | DB writes off the hot path |

**Race protection:** a per-symbol `asyncio.Lock` serialises everything that
mutates that symbol's position, and an `IntentRegistry` refuses a second intent
for a symbol while one is in flight. Client order ids are deterministic, so a
retried submission collides on Binance's side instead of creating a second
position.

## 5. State and recovery

Local state is a cache, **Binance is the source of truth**. On start, and after
any connectivity loss, the engine runs:

```
connect → fetch account → fetch positionRisk → fetch openOrders
        → rebuild local state → diff against DB → resolve
        → only then allow new entries
```

Resolution rules:
- Position on Binance, not in DB → adopt it, attach a protective stop
  immediately, mark it `ADOPTED` in the audit log, and close it at the next
  acceptable opportunity (it has no known thesis).
- In DB, not on Binance → mark closed, reconcile PnL from `userTrades`.
- Quantity mismatch → trust Binance, log a `RECONCILIATION_MISMATCH` risk event.
- Orphan stop order with no position → cancel.

`ENTRIES_BLOCKED` is the state during reconciliation; exits are always allowed.

## 6. Modes

| Mode | Data | Orders | Clock |
|---|---|---|---|
| `BACKTEST` | historical files | simulated | virtual, bar-driven |
| `PAPER` | live WebSocket | simulated by `paper/broker.py` | real |
| `LIVE` | live WebSocket | real REST orders | real |

The gateway is selected behind one interface (`exchange/base.py`), so the same
engine code runs in all three. `LIVE` requires the mode env var, the explicit
risk-acknowledgement env var and the `--live` CLI flag to agree; anything else
falls back to `PAPER` and logs a `CRITICAL`.

## 7. Failure handling summary

| Failure | Response |
|---|---|
| REST timeout on order submit | query by client id before any retry; never blind-retry |
| `429` / `418` | backoff with `Retry-After`; repeated → kill switch, entries off |
| WS disconnect | reconnect with jittered backoff; the reconnect **requests reconciliation**, since a gap in the feed is a gap in our knowledge of our own orders. A symbol quiet for > `stream.stale_after_sec` is refused for entry and back-filled over REST; a global silence past `ws_stale_seconds` trips the kill switch |
| Clock skew (`-1021`) | resync offset from `/fapi/v1/time`, retry once |
| Partial fill | position tracked at actual filled quantity; stop sized to actual |
| Rejected order | risk event, symbol cooldown, no retry loop |
| Unexpected position | adopt + protect + flatten (§5) |
| DB failure | `database` is a **critical** component: new entries stop, exits and position management continue. Writes stay buffered; the repository reconnects with backoff and replays them |
| Crash / VPS restart | full reconciliation before entries re-enabled |

## 8. What this architecture deliberately does not do

- It does not let a machine-learning model place orders.
- It does not auto-tune live parameters. Parameter changes are a deploy.
- It does not chase a trade count. Zero opportunities means zero trades.
- It does not use leverage to compensate for a small account.
