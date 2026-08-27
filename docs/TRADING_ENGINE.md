# TRADING ENGINE

How a price tick becomes a trade, and — more often — how it does not.

## The pipeline

```
BINANCE MARKET DATA
        ↓  WebSocket klines, book tickers, mark prices
MARKET SCANNER            score every tradable USDT perpetual
        ↓
DYNAMIC TOP 25            re-ranked every cycle; nothing is pinned
        ↓
MARKET REGIME             per candidate
        ↓
MULTI-STRATEGY ENGINE     only the strategies that regime permits
        ↓
SIGNAL AGGREGATOR         consensus, or stand aside on conflict
        ↓
OPPORTUNITY SCORE         0-100; below threshold → reject
        ↓
EXPECTED NET EDGE         after fees, spread, slippage, funding → reject
        ↓
CORRELATION ENGINE        is this a new bet or the same one again?
        ↓
RISK ENGINE               kill switches, budget, sizing, leverage
        ↓
EXECUTION ENGINE          idempotent order, protective stop
        ↓
POSITION MONITOR          TP / SL / trailing / time / regime / edge
        ↓
EXIT ENGINE
        ↓
PERFORMANCE DATABASE      every decision, accepted or not
```

Each stage can only **reduce** the candidate set. There is no path by which a
later stage admits something an earlier one rejected.

## Stage by stage

### 1. Scanner

Every tradable USDT perpetual is scored on thirteen weighted components. Two
market-wide REST calls serve the whole universe; expensive per-symbol kline
requests are limited to a prefiltered shortlist. A test pins the cost: scanning
60 symbols makes at most 16 kline calls.

A high market score means *worth watching*. It is not a signal.

### 2. Regime

Each candidate is classified `STRONG_TREND`, `WEAK_TREND`, `SIDEWAYS`,
`HIGH_VOLATILITY`, `LOW_VOLATILITY`, `BREAKOUT` or `PANIC`, measured against
that symbol's own distribution rather than absolute thresholds.

`PANIC` permits no strategy at all. It is checked first.

### 3. Strategies

Only the strategies the regime permits are **evaluated at all** — not evaluated
and then discounted. Each returns a `Signal` or `WAIT`. `WAIT` results are kept:
"considered and declined" is different from "never ran", and the audit log
records both.

### 4. Aggregation

Signals are weighted by regime appropriateness × the strategy's own confidence ×
its allocation weight. Strong disagreement **stands aside** rather than siding
with the louder half.

Levels are combined as **fractional distances**, not absolute prices — strategies
run on different timeframes and reference different closes, so averaging their
absolute levels inverts the risk:reward. The stop is the weighted-mean distance
(the consensus falsification point); the target is the nearest (a target only the
most optimistic strategy believes in becomes a time exit).

### 5. Opportunity score

Blends market quality, consensus, momentum, volume, trend, liquidity, volatility
fit, execution quality and reward:risk, minus cost and correlation penalties.

It answers a question the earlier stages do not: *is this specific trade, right
now, at this spread, given what we already hold, better than doing nothing?*

### 6. Expected net edge

```
expected_net = p·win − (1−p)·loss − fees − spread − slippage − funding
```

The single most consequential gate. `p` comes from the strategy's realised
history, shrunk toward a prior until enough trades exist.

With the shipped defaults, on a liquid symbol with ~0.11 % round-trip costs and
R ≈ 1.6, **a strategy needs roughly a 62 % realised win rate before a trade
clears the 0.08 % minimum edge.** An unproven strategy cannot clear it at all.
That is intended, and it is worth knowing before wondering why the bot is idle.

### 7-9. Correlation, risk, execution

See [`RISK_MANAGEMENT.md`](RISK_MANAGEMENT.md). Nothing reaches the exchange
without an `OrderIntent`, and only the risk engine can build one.

## Exits

Checked every second, in priority order:

| Priority | Exit | Condition |
|---|---|---|
| 1 | `STOP_LOSS` | protective order filled, or price through the stop |
| 2 | `TAKE_PROFIT` | target reached |
| 3 | `TRAILING_STOP` | trailed level hit after activation at 1 R |
| 4 | `TIME_LIMIT` | 60 minutes — a hard cap |
| 5 | `REGIME_CHANGE` | the regime that justified the trade is gone |
| 6 | `SIGNAL_FLIP` | the strategies now say the opposite |
| 7 | `NEGATIVE_EDGE` | remaining expected edge has turned negative |
| 8 | `KILL_SWITCH` / `EMERGENCY` | risk event requiring immediate flattening |

The 60-minute cap is not a target. Most exits should happen well before it; a
high proportion of `TIME_LIMIT` exits means targets are set beyond what the
holding period can deliver.

## Trade frequency

There is **no** trades-per-hour target, and no code path that relaxes a gate
because too few trades have happened.

| Valid opportunities | Trades taken |
|---|---|
| 20 | as many as risk limits allow |
| 2 | 2 |
| 0 | **0** |

A test polls a dead market 50 times and asserts zero trades. Another asserts
that a market scoring 99 in the scanner still produces none on its own.

If the bot is not trading, the answer is in the rejection counts — never in
lowering `min_expected_edge`.

## Concurrency

Single process, `asyncio`:

| Task | Cadence |
|---|---|
| market WebSocket | continuous, auto-reconnect with jittered backoff |
| user data WebSocket | continuous, listen key renewed every 30 min |
| full scan | 300 s |
| signal loop over candidates | 15 s |
| position monitor | 1 s |
| reconciler | 60 s |
| health | 5 s |

A per-symbol lock serialises everything that mutates that symbol's position, and
an in-flight registry refuses a second intent while one is outstanding. Client
order ids are deterministic, so a retried submission collides on Binance's side
rather than opening a second position.
