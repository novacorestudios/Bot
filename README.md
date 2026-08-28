# Dynamic Multi-Strategy Binance USDⓈ-M Futures Trading Engine

A short-term (≤ 60 minute) futures trading system that dynamically selects its
own markets, runs eight independent strategies gated by market regime, and
refuses to trade unless the expected edge survives fees, spread, slippage and
funding.

> ## Status: BUILT AND TESTED — NOT VALIDATED, NOT AUTHORISED FOR LIVE TRADING
>
> **644 automated tests pass.** They verify *behaviour*: that stops are
> mandatory, that risk never exceeds budget, that a timed-out order is never
> blindly re-sent, that the backtester cannot read the future. None of them
> verify *profit*, and no test can.
>
> **No strategy here has been validated on real market data.** No backtest,
> walk-forward or paper-trading result exists — the development environment has
> no network route to Binance, so every number that would appear in a results
> table has to come from your own run. Where such a number would go, this
> repository says **NOT TESTED**, deliberately.
>
> See [`FINAL_AUDIT.md`](FINAL_AUDIT.md) for the current, measured verification
> status — what is proven, what is not, and the order in which the remaining
> steps must happen before this touches real capital.

## What it does

```
Binance WebSocket + REST → MarketState (freshness per symbol)
  → dynamic scanner → top 25 candidates → market regime
  → 8 strategies (regime-gated) → signal consensus → opportunity score
  → EXPECTED NET EDGE → opportunity queue (best-first, expiring)
  → capital preservation mode → correlation → risk engine → position sizing
  → execution (order state machine) → position monitor → exit engine
  → execution-quality feedback → performance database
```

Two properties are worth stating up front, because they are what the design is
actually for:

1. **It does not force trades.** There is no trades-per-hour target. If the
   scanner finds no opportunity whose expected value is positive after all
   costs, the correct output is zero trades, and that is what it produces.
2. **Nothing bypasses the risk engine.** Strategies return data, never orders.
   The AI layer is advisory and has no order path. Exactly one code path turns
   an opportunity into an order, and it is inside `risk/engine.py`.

## Three things worth knowing before you start

**1. The defaults trade rarely.** On synthetic data the shipped configuration
produces essentially no trades. Strategies fire in mutually exclusive
conditions, regime gating permits three or four at a time, and the aggregator
wants two of those few to agree. Whether that is correct caution or over-tuning
is a question only real data can answer. If the bot looks idle, read the
rejection counts on the dashboard before changing anything.

**2. A strategy needs roughly a 62 % realised win rate** to clear the 0.08 %
minimum edge on a liquid symbol at R ≈ 1.6. That is the edge filter working, not
a misconfiguration.

**3. The system cannot bootstrap in live mode, on purpose.** An unproven
strategy's win rate is shrunk toward a prior that makes every trade
negative-edge, so it will not trade at all until it has evidence. Backtesting
supplies that evidence (see [`docs/BACKTESTING.md`](docs/BACKTESTING.md)); live
trading is then seeded from it.

## Why most signals get rejected

At a five-minute horizon the round trip costs roughly:

| Cost | Typical |
|---|---|
| Taker fee in | 0.04 % |
| Taker fee out | 0.04 % |
| Half-spread | 0.01–0.05 % |
| Slippage | 0.01–0.05 % |
| **Round trip** | **≈ 0.11–0.18 %** |

A signal must clear that before it is worth anything. The `min_expected_edge`
gate (default 0.08 % *net*) is the single most consequential setting in the
system, and it is why a technically valid-looking signal is often correctly
refused.

## Quick start

```bash
git clone https://github.com/novacorestudios/Bot.git tradebot && cd tradebot
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env      # then edit — keep BINANCE_TESTNET=true

export PYTHONPATH=src
python -m tradebot.app.cli validate-config
python -m tradebot.app.cli doctor
pytest -q
```

Nothing above touches the network or places an order.

Then, on a host that can reach Binance:

```bash
python scripts/verify_connectivity.py          # testnet first; checks the key
                                               # CANNOT withdraw, and fails if it can
python scripts/download_data.py --top 30 --start 2024-01-01 --end 2024-07-01

CONFIG_FILE=config/config.backtest.yaml \
  python -m tradebot.app.cli backtest --data data/klines --split 2024-05-01
CONFIG_FILE=config/config.backtest.yaml \
  python -m tradebot.app.cli walkforward --data data/klines

python -m tradebot.app.cli run                 # paper, against live prices
```

## Modes

| Mode | Market data | Orders | How to select |
|---|---|---|---|
| `BACKTEST` | historical files | simulated | `TRADING_MODE=BACKTEST` |
| `PAPER` | live WebSocket | simulated | `TRADING_MODE=PAPER` (default) |
| `LIVE` | live WebSocket | **real** | all three of: `TRADING_MODE=LIVE`, `I_UNDERSTAND_LIVE_TRADING_RISK=YES`, `--live` |

Any disagreement among those three aborts startup. A YAML edit alone can never
reach live trading.

## Documentation

| Document | Contents |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | phases, gates, honest status, known risks |
| [`AUDIT_REPORT.md`](AUDIT_REPORT.md) | the V1 deep technical audit — 22 findings, reproduced by execution |
| [`IMPLEMENTATION_PLAN_V2.md`](IMPLEMENTATION_PLAN_V2.md) | the hardening plan for those findings |
| [`BACKTEST_AUDIT.md`](BACKTEST_AUDIT.md) | the backtest engine audited — 21 findings, 4 critical |
| [`DATA_PIPELINE.md`](DATA_PIPELINE.md) | how real Binance history is fetched, validated and stored |
| [`BACKTEST_REPORT.md`](BACKTEST_REPORT.md) | **NOT VERIFIED** — no backtest on real data has been run |
| [`PAPER_TRADING_READINESS.md`](PAPER_TRADING_READINESS.md) | **NOT READY** — 0 of 7 criteria pass, and why |
| [`FINAL_AUDIT.md`](FINAL_AUDIT.md) | **read this before running anything** — what is verified, what is not, and what must happen before real money |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | components, pipeline, concurrency, recovery |
| [`docs/TRADING_ENGINE.md`](docs/TRADING_ENGINE.md) | the decision pipeline in detail |
| [`docs/RISK_MANAGEMENT.md`](docs/RISK_MANAGEMENT.md) | sizing, leverage, limits, kill switches |
| [`docs/STRATEGIES.md`](docs/STRATEGIES.md) | the eight strategies and their parameters |
| [`docs/BACKTESTING.md`](docs/BACKTESTING.md) | backtest, walk-forward, Monte Carlo |
| [`docs/PAPER_TRADING.md`](docs/PAPER_TRADING.md) | paper mode and its validation criteria |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | VPS, Docker, backup, recovery, updates |
| [`docs/SECURITY.md`](docs/SECURITY.md) | API key permissions, secrets, threat model |
| [`docs/API.md`](docs/API.md) | dashboard JSON API |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | symptoms, causes, fixes |

## Configuration

Secrets live in `.env` (never committed). Every threshold lives in
`config/config.yaml` and `config/strategies.yaml` — none in source code. The
shipped values are **starting points chosen for safety, not optimised values**.

## Repository layout

```
src/tradebot/{core,exchange,market,strategies,signals,risk,execution,
              portfolio,backtesting,paper,database,notifications,dashboard,ai,app}
config/  tests/  scripts/  docs/  docker/  .github/
```

## Licence and disclaimer

This software is provided for research and education. Trading leveraged futures
can lose more than your deposit. Nothing here is financial advice, and no
profitability is claimed or implied.
