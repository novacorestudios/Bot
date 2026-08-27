# IMPLEMENTATION_PLAN.md

**Project:** Dynamic Multi-Strategy Binance USDⓈ-M Futures Scalping Engine
**Status:** in development — **not authorised for live trading**
**Target first-live capital:** 75 USDT (only after every gate in §9 passes)

---

## 0. Statement of honesty (read first)

This document is a plan, not a claim of profitability.

- **No backtest has been run against real market data.** No profitability claim
  of any kind is made anywhere in this repository. Every performance figure
  would have to come from the operator's own run.
- 644 automated tests pass. They verify BEHAVIOUR — that stops are mandatory,
  that risk never exceeds budget, that a timed-out order is never re-sent, that
  the backtester does not read the future. None of them verify PROFIT, and no
  test can.
- **The development sandbox used to author this code has no network route to
  `fapi.binance.com`** (the egress proxy returns `403` on `CONNECT`). Therefore:
  - All Binance integration code is written against the **official Binance
    USDⓈ-M Futures API documentation** and is covered by unit tests using
    recorded/handcrafted response fixtures.
  - It has **not** been executed against the live Binance endpoints from this
    environment. The first real connectivity test must be run by the operator on
    a machine with egress to Binance, using `scripts/verify_connectivity.py`
    against **testnet first**.
  - Historical data must be downloaded by the operator with
    `scripts/download_data.py` (or from Binance Vision) before any meaningful
    backtest is possible. Backtests in CI run on **synthetic** data and exist
    only to prove the *engine* is correct, never to make claims about strategy
    profitability.
- There is no guaranteed profit. The purpose of the system is to *measure*
  whether a statistical edge exists after fees, spread, slippage and funding,
  and to refuse to trade when it does not.

---

## 1. Environment survey (done)

| Item | Result |
|---|---|
| Repo | `novacorestudios/Bot`, empty at start, branch `claude/binance-futures-scalping-engine-88tmfw` |
| Python | 3.11.15 |
| Node | 22.x (not required by the bot) |
| Docker | client present |
| PyPI reachable | yes |
| `fapi.binance.com` reachable | **no — 403 from egress proxy** |

Consequence: phases 1–8 (build + unit/integration + synthetic backtest) are fully
executable here. Phases involving real market data, paper trading against live
prices, and live trading must be executed by the operator on a VPS.

---

## 2. Repository structure

```
src/tradebot/
  core/          types, config, logging, clock, event bus, errors, math utils
  exchange/      exchange gateway protocol
    binance/     REST (signed), WebSocket, rate limiter, symbol filters, errors
  market/        candle store, indicators, universe, scanner, scoring, regime,
                 microstructure/cost model
  strategies/    base + registry + 8 strategies
  signals/       aggregator (consensus), opportunity score, expected-net-edge
  risk/          sizing, leverage, correlation, portfolio, kill switches,
                 cooldown, strategy allocation
  execution/     order manager, execution engine, reconciliation, exit engine
  portfolio/     position & PnL tracker
  backtesting/   data loader, engine, metrics, walk-forward, Monte Carlo, report
  paper/         simulated broker (latency, slippage, partial fills)
  database/      SQLAlchemy models, repositories, migrations
  notifications/ Telegram
  dashboard/     FastAPI mobile-friendly UI + JSON API
  ai/            advisory-only analysis layer (no order authority)
  app/           orchestrator, health monitor, safe mode, CLI
config/          YAML tunables per mode
tests/           unit / integration / failure-simulation
scripts/         data download, connectivity check, backtest/WFA runners
docs/            ARCHITECTURE, TRADING_ENGINE, RISK_MANAGEMENT, STRATEGIES,
                 BACKTESTING, PAPER_TRADING, DEPLOYMENT, SECURITY, API,
                 TROUBLESHOOTING
docker/          Dockerfile, compose, entrypoint, healthcheck
```

---

## 3. Dependencies

Runtime (pinned in `requirements.txt`):

| Package | Why |
|---|---|
| `aiohttp` | async REST + WebSocket client for Binance |
| `numpy` | indicator math (vectorised, no TA-Lib C dependency) |
| `pandas` | backtest data handling & reporting |
| `pydantic`, `pydantic-settings` | typed configuration, env binding, validation |
| `PyYAML` | tunable parameter files |
| `SQLAlchemy` (2.x async), `aiosqlite` | persistence; optional `asyncpg` for Postgres |
| `structlog` | structured JSON logging with secret redaction |
| `fastapi`, `uvicorn`, `jinja2` | dashboard + JSON API |
| `httpx` | Telegram client |

Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, `bandit`,
`pip-audit`.

Deliberately **not** used: TA-Lib (build friction on small VPS), `ccxt` (we want
exact control over Binance futures semantics, filters and error codes), any
scraping library.

---

## 4. Binance integration plan (official API only)

Base URLs (from official docs), all configurable:

- REST production: `https://fapi.binance.com`
- REST testnet: `https://testnet.binancefuture.com`
- WS market production: `wss://fstream.binance.com`
- WS market testnet: `wss://stream.binancefuture.com`

Endpoints used:

| Purpose | Endpoint | Method |
|---|---|---|
| Connectivity | `/fapi/v1/ping` | GET |
| Server time (clock skew) | `/fapi/v1/time` | GET |
| Symbol universe + filters | `/fapi/v1/exchangeInfo` | GET |
| Klines | `/fapi/v1/klines` | GET |
| 24h stats (scanner) | `/fapi/v1/ticker/24hr` | GET |
| Book ticker (spread) | `/fapi/v1/ticker/bookTicker` | GET |
| Order book depth | `/fapi/v1/depth` | GET |
| Mark price + funding | `/fapi/v1/premiumIndex` | GET |
| Funding history | `/fapi/v1/fundingRate` | GET |
| Account balance | `/fapi/v2/balance` | GET (signed) |
| Account/positions | `/fapi/v2/account`, `/fapi/v2/positionRisk` | GET (signed) |
| Leverage | `/fapi/v1/leverage` | POST (signed) |
| Margin type | `/fapi/v1/marginType` | POST (signed) |
| Place order | `/fapi/v1/order` | POST (signed) |
| Query order | `/fapi/v1/order` | GET (signed) |
| Cancel order | `/fapi/v1/order`, `/fapi/v1/allOpenOrders` | DELETE (signed) |
| Open orders | `/fapi/v1/openOrders` | GET (signed) |
| User trades (fills, fees) | `/fapi/v1/userTrades` | GET (signed) |
| Income (funding, realized) | `/fapi/v1/income` | GET (signed) |
| User data stream | `/fapi/v1/listenKey` | POST/PUT/DELETE (API-key) |

WebSocket streams: `<symbol>@kline_<interval>`, `<symbol>@bookTicker`,
`<symbol>@aggTrade`, `!markPrice@arr@1s`, plus the user data stream
(`ORDER_TRADE_UPDATE`, `ACCOUNT_UPDATE`, `listenKeyExpired`).

Rules the client enforces:

1. HMAC-SHA256 signature over the exact query string; `recvWindow` configurable
   (default 5000 ms); local clock offset measured against `/fapi/v1/time` and
   re-synced periodically.
2. **Weight-aware rate limiting.** A token-bucket tracks request weight per
   minute and order count per 10 s / per minute, seeded from `exchangeInfo`
   `rateLimits` and corrected from the `X-MBX-USED-WEIGHT-1M` and
   `X-MBX-ORDER-COUNT-*` response headers.
3. `418`/`429` → exponential backoff honouring `Retry-After`; repeated `418`
   trips a kill switch. `-1021` (timestamp) → resync clock and retry once.
   `-2019` (margin insufficient), `-4164` (min notional), `-1111` (precision)
   → non-retryable, logged as a risk event.
4. Every order carries a deterministic `newClientOrderId` derived from
   `(symbol, intent-id)` so a retry after a network timeout cannot create a
   duplicate position — on timeout we **query** before we re-send.
5. Order parameters are validated against the symbol's `PRICE_FILTER`,
   `LOT_SIZE`/`MARKET_LOT_SIZE`, `MIN_NOTIONAL`, `PERCENT_PRICE` filters and
   the leverage bracket **before** transmission.
6. Secrets come only from environment variables; the logger redacts any value
   matching the configured secret names, and signatures/keys are never logged.

---

## 5. Configuration schema

Two layers:

1. **Secrets & deployment** — environment variables only (`.env`, never
   committed). `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_CHAT_ID`, `DATABASE_URL`, `TRADING_MODE`, `BINANCE_TESTNET`.
2. **Tunables** — `config/*.yaml`, validated by pydantic models. Every threshold
   named in the brief is here, none in source: risk per trade, max daily loss,
   max drawdown, max positions, exposure caps, min signal score, min expected
   edge, max trade duration, max leverage, cooldowns, top-markets count, scan
   interval, scoring weights, strategy parameters, cost model assumptions.

`TRADING_MODE ∈ {BACKTEST, PAPER, LIVE}`. LIVE additionally requires
`I_UNDERSTAND_LIVE_TRADING_RISK=YES` **and** a `--live` CLI flag **and** a
non-testnet key; any backtest/paper config file that declares `mode` is refused
if it disagrees with the environment. Missing any one → process exits.

---

## 6. Database schema

SQLite by default (single-file, fine for one VPS), Postgres supported by URL.

| Table | Purpose |
|---|---|
| `trades` | one row per closed round-trip: symbol, strategy, direction, entry/exit, qty, leverage, SL, TP, duration, gross pnl, fees, funding, slippage, net pnl, regime, scores, exit reason |
| `positions` | live/open position state for crash recovery |
| `orders` | every order submitted, its client id, status transitions, fills |
| `fills` | individual executions with fee and commission asset |
| `signals` | every signal produced, whether accepted or rejected, with reason codes |
| `decisions` | **audit log**: full decision context per opportunity (see §35 of brief) |
| `market_snapshots` | compact per-scan snapshot of the ranked candidates |
| `strategy_metrics` | rolling performance per strategy |
| `risk_events` | every kill-switch trigger, limit breach, reconciliation mismatch |
| `equity_curve` | periodic equity samples for drawdown/Sharpe |
| `system_events` | connectivity, safe-mode entries/exits, restarts |

Retention: `market_snapshots` and `signals` pruned by configurable age to keep
the DB small.

---

## 7. Testing strategy

| Layer | What it proves |
|---|---|
| Unit | rounding to tick/step, min-notional, position sizing, leverage caps, SL/TP derivation, correlation matrix, kill-switch arithmetic, cost model, each indicator against hand-computed values, each strategy against constructed price paths |
| Integration | scanner→strategies→aggregator→risk→execution on a fake exchange; DB round-trips; reconciliation after simulated restart |
| Failure simulation | REST timeout, 429/418, WS disconnect mid-position, duplicate order response, partial fill, rejected order, unexpected position found on restart, DB unavailable, clock skew |
| Property/invariant | never a position without a stop; never exceed max positions/exposure; never size 0 or negative; net edge filter never passes a negative-edge trade |
| Backtest correctness | engine reproduces hand-calculated PnL on a scripted price series, including fees and funding |

CI gate: ruff + mypy + full pytest + bandit + pip-audit + docker build. Red CI
blocks merge.

---

## 8. Phase status

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Structure, config, docs, CI, Docker | **done** |
| 2 | Market data layer (REST/WS/indicators) | **done** — fixture-tested, NOT run against live Binance from here |
| 3 | Scanner + scoring + regime | **done** |
| 4 | Eight regime-gated strategies | **done** |
| 5 | Aggregator + opportunity score + edge filter | **done** |
| 6 | Risk engine, correlation, kill switches | **done** |
| 7 | Backtester, metrics, walk-forward, Monte Carlo | **done** — engine verified on synthetic data only |
| 8 | Paper broker | **done** — not yet run against live market data |
| 9 | Execution engine + reconciliation | **done** |
| 10 | Database, Telegram, dashboard, health | **done** |
| 11 | Docker + VPS documentation | **done** |
| 12 | Security audit | **done** — bandit clean, pip-audit clean, secret scan clean |
| 13 | End-to-end + structural tests | **done** — 644 tests |
| 14 | Paper-trading validation | **NOT DONE — requires the operator, ≥ 14 days** |
| 15 | LIVE | **NOT DONE — blocked on 14** |

### What "done" means here, and what it does not

Phases 1-13 are code, tests and documentation. They demonstrate the system
*behaves as specified*. They demonstrate nothing whatsoever about profitability.

Specifically **not** established:

- that any strategy has an edge on real data
- that the cost model matches real Binance fills
- that the engine survives a week of live market conditions
- that the trade frequency is workable in practice

Phases 14 and 15 are the ones that would establish those, and they cannot be
performed from this environment.

### Findings the operator should know before starting

Three things were discovered while building, and none of them are hidden in a
config file:

1. **The system could not bootstrap.** With no trade history the edge filter
   shrinks every win probability to the prior, which at a reward:risk of 2.0
   makes every candidate negative-edge — so nothing trades, so no history
   accumulates. Correct for live, fatal for a backtest. Resolved with an
   explicit, counted, loudly-reported bootstrap mode enabled only in
   `config/config.backtest.yaml`. See `docs/BACKTESTING.md`.

2. **The shipped defaults trade very rarely.** On synthetic data they produce
   essentially zero trades. The binding constraints are `NO_SIGNAL`, then
   `INSUFFICIENT_CONSENSUS`: the strategies fire in mutually exclusive
   conditions, regime gating permits three or four at a time, and the
   aggregator wants two of those few to agree. Whether that is correct caution
   or over-tuning is a question for real data.

3. **A strategy needs roughly a 62 % realised win rate** to clear the 0.08 %
   minimum edge on a liquid symbol at R ≈ 1.6 with 0.11 % round-trip costs.
   That is a demanding bar, and it is the intended behaviour of the edge filter.

## 9. Gates before any real money

All of these are operator-run, on a machine with Binance access. **None have
been run.**

1. Real historical data downloaded for ≥ 6 months across ≥ 30 symbols.
2. In-sample backtest → parameters chosen on training window only.
3. Out-of-sample backtest on untouched window: positive expectancy, profit
   factor > 1.15, max drawdown within configured limit.
4. Walk-forward analysis: majority of folds positive, no fold catastrophic.
5. Monte Carlo on trade sequence: 5th percentile drawdown within limit.
6. Parameter robustness: performance does not collapse under ±20 % parameter
   perturbation.
7. Paper trading ≥ 14 days against live market data: realised metrics within
   the backtest's confidence band; no unexplained reconciliation mismatch.
8. Security audit passed; API key has **trade only, no withdrawal**, IP-locked.
9. Explicit human confirmation, tiny capital (75 USDT), reduced risk-per-trade
   for the first week.

If a gate fails the answer is to stop and re-examine, not to loosen the gate.

---

## 10. Known risks and honest caveats

- **Scalping is the hardest edge to find.** At 1–5 minute horizons, taker fees
  (0.05 % per side ≈ 0.10 % round trip) plus spread and slippage mean a strategy
  must clear roughly 0.12–0.20 % per trade just to break even. Many technically
  "good" signals will be rejected by the edge filter — that is the filter
  working, not a bug.
- **75 USDT is small.** `MIN_NOTIONAL` on Binance futures (commonly 5 USDT, some
  symbols higher) plus step sizes mean that with 0.5 % risk (0.375 USDT) the
  required stop distance and leverage interact tightly; on some symbols the
  correctly-sized position is not representable and the trade must be skipped.
  The risk engine treats that as a rejection, never as a reason to oversize.
- **Funding** on perpetuals is charged every 8 h; sub-hour trades usually avoid
  it, but positions held across a funding timestamp must account for it.
- **Simulated fills are optimistic by nature.** Paper results will be better
  than live. The paper broker deliberately applies pessimistic slippage.
- **Overfitting is the default outcome** of tuning on one dataset. Walk-forward
  and Monte Carlo exist to catch it; if they fail, the strategy is discarded.
