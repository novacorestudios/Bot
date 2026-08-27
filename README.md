# Dynamic Multi-Strategy Binance USDⓈ-M Futures Trading Engine

A short-term (≤ 60 minute) futures trading system that dynamically selects its
own markets, runs eight independent strategies gated by market regime, and
refuses to trade unless the expected edge survives fees, spread, slippage and
funding.

> ## Status: IN DEVELOPMENT — NOT AUTHORISED FOR LIVE TRADING
>
> No strategy in this repository has been validated on real market data.
> No backtest, walk-forward or paper-trading result exists yet — where numbers
> would normally appear, this repository says **NOT TESTED**, deliberately.
> See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §9 for the gates that
> must pass before real money is involved, and §0 for what has and has not been
> executed.

## What it does

```
Binance market data → dynamic scanner → top 25 candidates → market regime
  → 8 strategies (regime-gated) → signal consensus → opportunity score
  → EXPECTED NET EDGE → correlation → risk engine → position sizing
  → execution → position monitor → exit engine → performance database
```

Two properties are worth stating up front, because they are what the design is
actually for:

1. **It does not force trades.** There is no trades-per-hour target. If the
   scanner finds no opportunity whose expected value is positive after all
   costs, the correct output is zero trades, and that is what it produces.
2. **Nothing bypasses the risk engine.** Strategies return data, never orders.
   The AI layer is advisory and has no order path. Exactly one code path turns
   an opportunity into an order, and it is inside `risk/engine.py`.

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
