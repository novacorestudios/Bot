# STRATEGIES

Eight independent strategies. Each can be enabled, disabled, parameterised and
measured on its own, and none of them can place an order — they return data,
which the aggregator, the edge filter and the risk engine then act on.

> **No strategy in this document has been validated on real market data.**
> The tests described below prove each strategy *does what it claims* on
> constructed price paths. They say nothing about profitability. See
> `IMPLEMENTATION_PLAN.md` §9.

## The shared contract

Every strategy receives a `MarketView` — symbol, candles, regime, book
imbalance, funding — and returns a `StrategyOpinion`. The base class converts
that into a validated `Signal`.

Deliberately absent from `MarketView`: account state, positions, equity, and any
exchange handle. A strategy physically cannot act on the account. This is
enforced by a test.

### Levels are derived centrally

Strategies choose their ATR multiple and reward:risk; they do not choose the
*method*. `Strategy.derive_stop` applies, in order:

1. `atr_multiple × ATR`
2. clamped to `[min_stop_pct, max_stop_pct]` of price
3. pushed beyond the nearest swing point within `structure_lookback`, plus a
   buffer
4. re-clamped so structure cannot exceed the maximum stop distance

Step 3 is the one that matters. A stop resting just inside an obvious swing low
is a stop that gets hit: price reaching that low will almost certainly take out
everything above it first.

`derive_target` applies reward:risk, capped by `atr_multiple_cap × ATR` and
floored at `min_rr`. The cap exists because a wide stop times an ambitious R
produces a target price the market will not reach inside 60 minutes, converting
winners into time-based exits.

Centralising this is also what makes R-multiples comparable between strategies —
without it, the risk budget and the performance tracker would be comparing
incompatible units.

## The eight

| Strategy | Timeframe | Thesis | Refuses when |
|---|---|---|---|
| `momentum` | 3m (15m confirm) | A decisive, volume-backed impulse continues briefly | RSI is exhausted, volume is ordinary, EMAs are not aligned, higher timeframe disagrees |
| `trend_following` | 5m (1h confirm) | A pullback to the fast EMA in a confirmed trend is the good entry | ADX below threshold, price extended from the EMA, 1h trend opposes |
| `breakout` | 3m | A compressed range that breaks with volume continues | No prior compression, no volume, already extended past the level |
| `mean_reversion` | 3m | Price stretched from its mean in a range snaps back | ADX says the market is trending; stretch too small; higher timeframe endorses the move |
| `volume_spike` | 1m | A volume surge marks informed flow, price follows briefly | Spike bar is mostly wick (rejection), spike already faded |
| `volatility_expansion` | 3m | Expanded realised range persists for several bars | Expansion has no directional component; slope contradicts range position |
| `vwap` | 3m | Session VWAP is a magnet in balance, support/resistance in trend | Fade mode: ADX too high. Ride mode: ADX too low. Price inside the bands |
| `support_resistance` | 5m | Repeatedly-tested levels produce reactions | No level with enough touches, no rejection candle, strong trend against the level |

### What each strategy refuses is the point

A strategy that always has an opinion has no edge. The refusals encode where
each thesis stops working:

- **momentum** will not chase. Its `rsi_long_max` ceiling means it declines
  precisely the parabolic moves that feel most compelling. This has a real
  consequence, discovered while testing: a move with *no retracement at all*
  drives Wilder RSI above 85 arithmetically, so momentum structurally cannot
  enter a vertical move. That is intended.
- **mean_reversion** and **vwap-fade** both stand down above an ADX ceiling.
  Fading a trend is the single fastest way to lose an account, and neither
  strategy is permitted to do it.
- **breakout** requires prior compression. A "break" from an already-wide range
  is just continuation with nowhere sensible to put a stop.
- **volume_spike** refuses wick-dominated bars rather than trading them
  backwards. A rejection is a different thesis needing a different stop.
- **volatility_expansion** refuses expansion without direction — two-sided
  volatility means paying both sides' costs for no edge.

## Regime gating

Which strategies run is decided by the regime, in `config/config.yaml` under
`regime.strategy_weights`:

| Regime | Strategies permitted |
|---|---|
| `STRONG_TREND` | momentum, trend_following, breakout, vwap(ride) |
| `WEAK_TREND` | momentum, trend_following, vwap, support_resistance |
| `SIDEWAYS` | mean_reversion, vwap, support_resistance |
| `HIGH_VOLATILITY` | breakout, volatility_expansion, momentum |
| `LOW_VOLATILITY` | mean_reversion, vwap, support_resistance |
| `BREAKOUT` | breakout, volatility_expansion, volume_spike, momentum |
| `PANIC` | **none** |

Gating is not advisory. A strategy absent from the regime's map is **not
evaluated at all** — not evaluated and then discounted. In `PANIC` the map is
empty, so nothing runs and no entry is possible.

## Why three trend-ish strategies do not simply duplicate each other

`momentum`, `trend_following` and `breakout` can all be long the same uptrend,
but they enter at different moments: momentum on the impulse, trend_following on
the retracement, breakout on the level break. That is what makes the
aggregator's consensus meaningful — when all three agree, they are agreeing
despite different entry logic, not because they share it.

## Testing approach

Each strategy has tests for:

1. **It fires** in the predicted direction on a path built for its thesis.
2. **It declines** on noise and on each condition it is designed to refuse.
3. **Its output is sound** — stop on the correct side, target beyond entry,
   R-multiple within configured bounds.
4. **It is isolated** — a strategy that raises returns `WAIT`; it cannot abort
   the cycle or emit a malformed signal.

### A note on synthetic price paths

Test paths must be realistic or they test the wrong thing. Three cases were
found while writing these tests where a naive generator exercised only a guard:

| Naive path | What it actually tested | Realistic replacement |
|---|---|---|
| Monotonic trend (`trend_prices`) | RSI saturation and EMA-extension guards, never the entry | `trending_with_pullbacks` — a trend that retraces |
| Impulse with no counter-bars | The exhaustion ceiling only | `impulse_prices` — impulse containing pullback bars |
| Sine wave "range" | The anti-trend guard (half a sine cycle is ~12 consecutive same-direction bars, ADX ≈ 28) | `choppy_prices` — a mean-reverting Ornstein-Uhlenbeck walk |

The instinct when a strategy refuses a test path is to loosen the strategy. In
all three cases the strategy was right and the path was wrong. A rise of 0.15 %
per 5-minute bar is 43 % per day; refusing to buy a pullback in it is correct
behaviour, not a bug.

**These paths prove logic, not profit.** Real data is required for any claim
about performance.

## Adding a strategy

1. Subclass `Strategy`, set `name` and `min_bars`, implement `evaluate`.
2. Register it in `STRATEGY_CLASSES` in `strategies/registry.py`.
3. Add its parameters to `config/strategies.yaml`.
4. Add it to the regimes where it applies in `config/config.yaml`.
5. Write its tests: one path where it fires, one per refusal, plus the shared
   contract tests (which apply automatically via parametrisation).
6. Backtest it in isolation before enabling it alongside the others.
