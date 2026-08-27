# RISK MANAGEMENT

> Every number in this document is a **starting point**, not a validated
> optimum. None has been tested against real trading. See
> `IMPLEMENTATION_PLAN.md` §9.

## The one rule

**Every trade passes through `risk/engine.py`, and nothing bypasses it.**

`RiskEngine.evaluate()` is the only function in the system that constructs an
`OrderIntent`, and `OrderIntent` is the only object the execution engine will
act on. Strategies, the AI layer, the dashboard and the scanner can all
*propose*; only the risk engine can *approve*.

The corollary is equally important: **the risk engine never blocks an exit.**
There is no `evaluate_exit`, no `approve_close`, no code path by which a kill
switch or a portfolio limit can prevent a position from being closed. Being
unable to exit is strictly worse than any condition that would justify not
entering, and a test asserts those methods do not exist.

## Position sizing

```
quantity = (equity × risk_fraction) / |entry − stop|
```

Risk is defined by the **distance to the stop**. Not by notional, not by
leverage. This is what makes "0.5 % per trade" mean the same thing on a
tight-stop scalp and a wide-stop swing.

### Leverage does not change risk

A 600 USDT position with a 0.5 % stop risks 3 USDT at 1× and 3 USDT at 10×. The
difference is that at 10× the liquidation price sits close enough that the
**exchange may close the position before the stop does** — which converts a
planned 3 USDT loss into a much larger one.

So leverage is derived from need, never chosen for ambition:

1. the smallest leverage that makes the risk-correct notional affordable
2. capped by a volatility-adjusted ceiling (halving per doubling of volatility
   past `leverage_volatility_threshold`)
3. capped by the symbol's own maximum
4. **rejected outright** if the estimated liquidation sits closer than
   `min_liquidation_distance_multiple × stop distance` (default 3×)

### The small-account wall

With 75 USDT and 0.5 % risk (0.375 USDT), the sizing arithmetic frequently
produces a position the exchange will not accept:

| Situation | Response |
|---|---|
| quantity rounds to zero at the step size | reject `SIZE_BELOW_MINIMUM` |
| quantity below the symbol's `minQty` | reject `SIZE_BELOW_MINIMUM` |
| notional below the symbol's `MIN_NOTIONAL` | reject `NOTIONAL_BELOW_MINIMUM` |

In every case the answer is to **skip the trade**, never to round up. The
rejection message states exactly what oversizing would have cost:

```
risk-correct notional 75.0000 is below the symbol minimum 100.0.
Meeting it would require 1 units, risking 0.5000 (0.67% of equity)
instead of 0.3750. Skipping rather than oversizing.
```

A consequence worth knowing: with `max_symbol_exposure: 1.0`, any stop tighter
than `risk_per_trade` (0.5 %) produces a position the exposure cap trims, so the
trade risks *less* than budgeted. Safe, but it means very tight stops cannot use
the full allowance. Raising `max_symbol_exposure` is the lever, and it should be
changed knowingly.

## Correlation

Four correlated longs are **one leveraged bet wearing four names**, while every
per-trade check happily reports 0.5 % each.

The engine measures this with the effective number of independent positions:

```
N_eff = (Σw)² / (wᵀ · C · w)
```

which equals N when uncorrelated and collapses toward 1 as correlations rise.
The tests pin both ends: four correlated longs measure as **1.02** independent
bets; four uncorrelated ones measure as **> 3.0**.

Weights are **signed by direction**. A long in A against a short in a correlated
B is a spread whose legs offset, not a doubled bet — using absolute correlation
scored that pair at 0.99 and rejected it, which would have forbidden every hedge
the system could construct. Concentration is about positions that move together
*in PnL*, which is what the signed product measures.

Two limits apply:

| Limit | Meaning |
|---|---|
| `max_pair_correlation` (0.75) | this candidate is too similar to something already held |
| `max_portfolio_correlation` (0.60) / `min_effective_positions_ratio` (0.55) | the book as a whole would be too concentrated |

## Portfolio accounting

"Open risk" is **stop distance × quantity**, summed — not notional, not margin.
A 1000 USDT position stopped 0.2 % away risks 2 USDT; a 100 USDT position
stopped 5 % away risks 5 USDT. The second is the larger risk despite being a
tenth of the notional, and only stop-distance accounting sees that.

A position with **no stop** contributes its entire notional as risk, and is
listed in `unprotected_positions`. That is the honest accounting for a position
whose downside is unknown.

| Limit | Default | Guards against |
|---|---|---|
| `max_concurrent_positions` | 4 | attention and margin spread too thin |
| `max_total_risk` | 2 % | many small risks summing to one large one |
| `max_symbol_exposure` | 1.0× | concentration in one name |
| `max_direction_exposure` | 3.0× | an accidental directional bet |
| `max_total_exposure` | 4.0× | gross leverage |
| `max_margin_usage` | 50 % | no buffer for adverse moves |

## Kill switches

Automatic circuit breakers. **All of them block entries only.**

| Switch | Trigger | Re-arms |
|---|---|---|
| `DAILY_LOSS` | 2 % of the day's starting equity | next trading day |
| `HOURLY_LOSS` | 1 % in an hour | next hour |
| `MAX_DRAWDOWN` | 10 % from the equity peak | next trading day |
| `CONSECUTIVE_LOSSES` | 5 in a row | after `auto_rearm_seconds` |
| `API_ERRORS` | 20 in 5 minutes | after `auto_rearm_seconds` |
| `REJECTED_ORDERS` | 5 in an hour | after `auto_rearm_seconds` |
| `SLIPPAGE` | mean above 0.25 % over recent fills | after `auto_rearm_seconds` |
| `RECONCILIATION` | 2 mismatches with the exchange | manual |
| `STALE_DATA` | no market data for 30 s | automatically when data returns |
| `CONNECTION` | exchange unreachable | automatically on reconnect |

Loss-based switches deliberately do **not** auto-re-arm within the day. If the
day's budget is gone, it is gone.

Each switch exists for a specific failure: `CONSECUTIVE_LOSSES` catches a
strategy that has stopped working before the loss limits notice; `SLIPPAGE`
catches execution quality collapsing, which silently invalidates every edge
estimate; `RECONCILIATION` catches the most dangerous state of all — local
state disagreeing with the exchange.

## Cooldown

After a position closes, the same symbol is not immediately re-entered. The
conditions that produced the exit are usually still present a minute later, so
an immediate re-entry tends to be the same trade again — paying a second round
trip for it.

- longer after a loss (420 s) than a win (90 s)
- multiplied per **consecutive** loss on that symbol. The first loss gets the
  base duration; only the second consecutive loss is multiplied.
- shortened in high volatility, where theses form and invalidate faster
- tracked per strategy as well as per symbol, so one strategy failing on a
  symbol does not silence the others

## Strategy allocation and suspension

Risk budget is distributed by realised expectancy in **R-multiples** (comparable
across position sizes and equity levels), under two constraints:

- **Bounded**: weights clamp to `[0.4×, 2.0×]` of equal weight. A strategy on a
  hot streak is often just a strategy whose conditions happened to persist.
- **Evidence-gated**: parity until `min_trades_for_adjustment` (30) trades.
  Reallocating on ten trades is fitting noise with real money.

The **strategy kill switch** suspends one strategy while the others continue,
when its profit factor or expectancy falls below threshold over enough trades.
Suspension is temporary: a strategy usually stops working because its regime
left, and regimes return.

Allocation never enables or disables a strategy, and **never alters strategy
parameters** — the brief forbids that during live trading, and a parameter
change is a deploy.

## Decision order

Cheapest and most categorical first, so an expensive correlation matrix is never
computed for a trade a kill switch already forbids:

1. account kill switches
2. entries-blocked flag (reconciliation, safe mode)
3. duplicate position / in-flight intent
4. position count
5. cooldown
6. strategy suspension
7. **stop loss present and correctly placed**
8. sizing (equity, risk, stop distance, filters, leverage, liquidation)
9. correlation
10. portfolio limits

Every outcome — approval or rejection — carries a reason code and a `checks`
dictionary, and is written to the audit log.

## What is NOT protected against

Stated plainly, because a risk system that implies completeness is worse than
one that admits its edges:

- **Gaps and liquidation.** A stop is an order, not a guarantee. A move that
  gaps through the stop fills worse, and a violent enough gap can liquidate a
  leveraged position before any stop executes. The 3× liquidation-distance rule
  reduces this; it cannot eliminate it.
- **Exchange failure.** If Binance is down or refuses orders, positions cannot
  be closed. The kill switches stop new entries; they cannot exit for you.
- **Correlation regime change.** Correlations measured over the last 240 bars
  are not the correlations of a crash, when almost everything goes to 1.
- **Model error.** The edge estimate can be wrong. `realised_vs_expected` exists
  to detect that from evidence, but only after trades have happened.
