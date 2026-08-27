# TROUBLESHOOTING

## The bot is not trading

**This is usually correct behaviour.** The system is opportunity-driven: zero
valid opportunities means zero trades, by design. Diagnose before changing
anything.

```bash
curl -s "localhost:8080/api/status?token=$DASHBOARD_TOKEN" \
  | python -c "import json,sys; [print(f\"{r['count']:>6}  {r['reason']}\") for r in json.load(sys.stdin)['rejections']]"
```

| Dominant rejection | Meaning | What to do |
|---|---|---|
| `NEGATIVE_EXPECTED_EDGE` | expected move does not cover ~0.11 % round-trip costs | **Nothing.** This is the filter working. Never lower `min_expected_edge` to trade more. |
| `NO_SIGNAL` | strategies are silent | check the regime distribution; strategies fire in specific conditions |
| `INSUFFICIENT_CONSENSUS` | strategies fire but rarely agree | see below — this is the most common blocker |
| `LOW_OPPORTUNITY_SCORE` | setups pass but score under 70 | inspect which components are low |
| `NOTIONAL_BELOW_MINIMUM` | account too small for these symbols | expected at 75 USDT on some symbols |
| `COOLDOWN_ACTIVE` | recent losses suppressing re-entry | expected after a losing streak |
| `KILL_SWITCH_ACTIVE` | a circuit breaker is tripped | check Telegram and `/api/risk` |
| `MAX_POSITIONS` | at the concurrent-position limit | expected |

### `INSUFFICIENT_CONSENSUS` dominating

The most common reason for an idle bot, and it is structural rather than a bug.
Three things compound:

1. strategies are built to fire in **mutually exclusive** conditions
2. regime gating permits only **three or four** at a time
3. the aggregator then wants **two of those few** to agree

Backtesting on synthetic data produced essentially zero trades for exactly this
reason (see `docs/BACKTESTING.md`). Whether it is correct caution or over-tuning
can only be settled on real data.

If real-data backtesting confirms it is too strict, the levers in order of how
much evidence should back changing them:

| Change | Effect | Risk |
|---|---|---|
| `aggregator.min_agreeing_strategies` 2 → 1 | admits single-strategy signals | consensus is the main defence against one strategy misfiring |
| `opportunity.min_score` 70 → 65 | admits weaker setups | tune against measured outcomes, never to hit a trade count |
| `regime.strategy_weights` | permits more strategies per regime | gating exists because strategies bleed in the wrong regime |
| `edge.min_expected_edge` | **do not** | the one gate between the bot and negative-expectancy trading |

## Connectivity

### `cannot reach Binance` / repeated timeouts

```bash
python scripts/verify_connectivity.py
```

| Symptom | Cause | Fix |
|---|---|---|
| `CONNECT tunnel failed 403` | region blocked or proxy denial | check whether your region can reach Binance |
| DNS failure | resolver problem | `dig fapi.binance.com` |
| Slow but working | distant region | move the VPS closer to `ap-northeast-1` |

### `-1021 Timestamp outside recvWindow`

Clock drift. Binance rejects requests whose timestamp is outside `recvWindow`.

```bash
timedatectl status                    # check NTP sync
sudo timedatectl set-ntp true
```

The client measures and corrects the offset at startup and re-syncs once on a
`-1021`, but a badly drifting host will keep hitting it.

### `-2015 Invalid API-key, IP, or permissions`

In order of likelihood: the VPS IP is not on the key's allow-list; futures
trading is not enabled for the key; the key was revoked; you are using a mainnet
key against testnet or vice versa.

### WebSocket keeps reconnecting

Some reconnection is normal — Binance closes every connection after 24 hours by
design. Frequent reconnection means an unstable network or being rate limited.

```bash
docker compose logs tradebot | grep ws_reconnecting | tail -20
```

## Orders

### `-4164 Order's notional must be no smaller than 5`

The account is too small for that symbol at the risk-correct size. **This is
handled** — the risk engine rejects the trade rather than oversizing. If you see
the exchange error rather than the local rejection, the symbol's filters are
stale; restart to reload `exchangeInfo`.

### `-2019 Margin is insufficient`

Margin is committed to existing positions. Check `/api/risk` for margin usage.
If it persists with few positions, the leverage the exchange applied may differ
from the one requested — the engine records the applied value, so compare them
in the logs.

### `-1111 Precision is over the maximum`

Should be impossible: quantities and prices are rounded through `Decimal`
against the symbol's filters before transmission. If it happens, the cached
filters are stale — restart — and please report it, because the local validator
should have caught it.

### Orders rejected repeatedly

After `max_rejected_orders_per_hour` the `REJECTED_ORDERS` kill switch trips and
halts entries. That is intended: repeated rejections usually mean a sizing or
filter bug, and continuing would burn the order rate limit.

## Positions

### A position has no stop

**This should be impossible.** The engine closes any position it cannot protect,
and reconciliation re-protects or closes any it finds unprotected.

If you see one:

```bash
docker compose logs tradebot | grep -E "MISSING_STOP_LOSS|position_without_stop"
```

Then close it manually in the Binance UI and report it — this is the most
serious class of bug in the system.

### A position appeared that the bot did not open

Reconciliation adopts it, attaches a protective stop immediately, and marks it
for closure — we have no thesis for a position we did not knowingly open.

```bash
docker compose logs tradebot | grep -E "unexpected_position_found|POSITION_ADOPTED"
```

Common causes: a manual trade in the same account; a previous bot instance still
running. **Do not run two instances against one account** — they will fight over
the same positions.

### Entries are blocked

```bash
curl -s "localhost:8080/api/risk?token=$TOKEN" | python -m json.tool
```

| Cause | Clears when |
|---|---|
| kill switch tripped | its re-arm interval, next day, or manual reset |
| safe mode | the failed component recovers |
| reconciliation in progress | it completes |
| reconciliation failed | it succeeds — deliberately blocked until then |
| indeterminate order state | reconciliation resolves it |

## Safe mode

Safe mode disables **new entries** while leaving open positions fully managed
and closable. It engages when a critical component fails: market data, exchange
REST, the risk engine or execution.

```bash
curl -s "localhost:8080/api/health?token=$TOKEN" | python -m json.tool
```

Non-critical failures — database, Telegram, dashboard — produce warnings only.

## Database

### `database_write_failed`

Trading continues; audit rows are lost. Usually a full disk.

```bash
df -h
docker compose exec tradebot python -c "
import asyncio
from tradebot.database.repository import Repository
async def main():
    r = Repository('sqlite+aiosqlite:///data/tradebot.db')
    await r.connect(); print(await r.prune({'signal_retention_days': 7,
        'market_snapshot_retention_days': 3, 'decision_retention_days': 30},
        __import__('time').time() * 1000)); await r.close()
asyncio.run(main())"
```

### `database_buffer_full`

The database has been unavailable long enough to fill the write buffer. Oldest
audit rows are dropped rather than growing memory — a bot that runs out of
memory with open positions is far worse than one with a gap in its logs.

## Performance

### High memory

Expected: ~500 bars × 5 timeframes × ~75 symbols. If it grows without bound,
check that `candles.retain()` is pruning symbols that left the ranking:

```bash
docker compose logs tradebot | grep candles
```

### Scans taking too long

```bash
docker compose logs tradebot | grep scan_complete | tail -5
```

A scan should take a few seconds. If it takes tens of seconds, the rate limiter
is throttling — check `weight_pct` in `/api/status`.

## Telegram

Silent bot: confirm `TELEGRAM_ENABLED=true`, that you have messaged the bot at
least once (Telegram forbids a bot messaging first), and that the chat id is
your numeric id from @userinfobot rather than a username.

```bash
docker compose logs tradebot | grep telegram
```

Notification failures never affect trading — they are logged and counted only.

## Getting help

Include:

1. `docker compose logs --since 1h tradebot > logs.txt` — **secrets are already
   redacted**, so this is safe to share
2. `curl -s "localhost:8080/api/status?token=$TOKEN"`
3. `config/config.yaml` (contains no secrets)
4. What you expected and what happened

Never share `.env`.
