# PAPER TRADING

Paper mode runs the **complete** engine against **live** market data, with every
order simulated. The code path is identical to live except for the gateway —
that is the point. A paper run that used a different decision path would
validate the wrong thing.

```bash
cp .env.example .env      # TRADING_MODE=PAPER, BINANCE_TESTNET=true
export PYTHONPATH=src
python -m tradebot.app.cli run
```

No API key is needed for market data. Supplying a read-only testnet key lets the
account endpoints be exercised too, which is worth doing before live.

## The simulator is deliberately harsh

A paper broker that fills at the mid price, instantly, in full, produces results
strictly better than live — and the gap only becomes visible after real money is
committed. So `paper/broker.py` is pessimistic on purpose:

| Behaviour | Default | Why |
|---|---|---|
| Latency | 120 ms ± 80 ms | the price moves between decision and fill |
| Spread crossing | always | marketable orders take the ask / hit the bid |
| Adverse slippage | 65 % of fills | the fills you get are the ones the market was willing to give |
| Partial fills | 10 % of orders | exercises holding less than you asked for |
| Rejections | 0.5 % of orders | the rejection path should be routine, not a surprise |
| Funding | charged | positions crossing a funding timestamp pay |

Individual fills can still land *better* than the mid — favourable slippage is
real — but the average must be worse, and a test asserts it over 40 samples.

Being harsher than reality is the safe direction to be wrong in. **Paper results
will still be better than live**; treat them as an upper bound.

## What paper trading validates

It is good evidence for:

- the engine runs for days without leaking memory, wedging, or losing its state
- reconnection works, and the position monitor survives a WebSocket drop
- reconciliation keeps local state matching the (simulated) exchange
- kill switches trip on the conditions they are meant to
- the decision pipeline behaves on live data the way it did on historical data
- **trade frequency** — whether the filters produce a workable number of
  opportunities in real conditions, which synthetic data cannot answer

## What it does NOT validate

Stated plainly, because paper results are the most over-trusted number in
algorithmic trading:

- **Real fills.** No order ever reaches the book. Real slippage on a thin
  perpetual during a news candle is not modelled by any distribution.
- **Market impact.** Negligible at 75 USDT, but the simulator assumes it is
  exactly zero.
- **Queue position.** Limit orders are not modelled as resting in a real queue.
- **Exchange behaviour under stress.** Rate limiting, order rejections during
  volatility, and outages are simulated only crudely.
- **Your own behaviour.** Watching a real account draw down changes decisions in
  ways paper trading cannot capture.

A profitable paper run is **necessary and not sufficient**. It is one gate of
nine in `IMPLEMENTATION_PLAN.md` §9.

## Validation criteria

Run for **at least 14 days** of continuous operation, then check:

| Criterion | Threshold | Why |
|---|---|---|
| Uptime | > 99 %, no unexplained restarts | a bot that needs babysitting is not ready |
| Reconciliation mismatches | 0 unexplained | any mismatch is a state bug |
| Positions without stops | **0**, ever | non-negotiable |
| Trades | ≥ 100 | below this, nothing is statistically meaningful |
| Win rate | within the backtest's confidence band | a large gap means the backtest was fitted |
| Expectancy | positive, and consistent with backtest | |
| Max drawdown | within `max_drawdown` | |
| Realised vs expected edge | gap explainable | a persistent gap means the cost model is optimistic |
| Kill switches | fired only when they should | false trips waste opportunity; missed trips are dangerous |

The realised-versus-expected comparison is the most informative of these. The
engine records both per strategy (`EdgeCalculator.realised_vs_expected`), and a
persistent negative gap means slippage or the win-probability estimate is wrong
— which invalidates the edge filter that every trade depends on.

## If paper trading produces no trades

That is a real possible outcome and it is diagnosable, not mysterious. Check the
rejection counts on the dashboard or in `SignalPipeline.stats()`:

| Dominant rejection | Meaning |
|---|---|
| `NO_SIGNAL` | strategies are not firing — check regime distribution |
| `INSUFFICIENT_CONSENSUS` | strategies fire but rarely agree |
| `LOW_OPPORTUNITY_SCORE` | setups pass but score below 70 |
| `NEGATIVE_EXPECTED_EDGE` | **working as designed** — costs exceed the expected move |
| `NOTIONAL_BELOW_MINIMUM` | the account is too small for these symbols |
| `COOLDOWN_ACTIVE` | recent losses are suppressing re-entry |

`NEGATIVE_EXPECTED_EDGE` dominating is not a bug. It means the strategies are
finding setups whose expected move does not cover ~0.11 % of round-trip costs.
The fix is better setups, not a lower threshold.

**Never lower `min_expected_edge` to increase trade count.** It is the one gate
standing between the bot and systematically negative-expectancy trading.

## Moving from paper to live

1. Paper for ≥ 14 days; every criterion above met.
2. Export measured strategy statistics from the validated run and seed live with
   them (`EdgeCalculator.seed_from`) — live never bootstraps, so it must start
   from evidence.
3. Run `scripts/verify_connectivity.py` against **production**, confirming the
   API key cannot withdraw.
4. Set `TRADING_MODE=LIVE`, `I_UNDERSTAND_LIVE_TRADING_RISK=YES`,
   `BINANCE_TESTNET=false`, and pass `--live`. All four must agree.
5. Start with **reduced** risk per trade — half the tested value for the first
   week. The first live week is for discovering the difference between paper
   fills and real ones, not for making money.
6. Watch the first ten trades individually. Compare each fill against the price
   at decision time. If realised slippage exceeds the model, stop and re-tune
   the cost model before continuing.
