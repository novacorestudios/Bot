# API

The dashboard's JSON API. **Read-only by design** — no endpoint can open, close
or size a position, or reset a kill switch. A test enumerates the routes and
asserts every one is `GET`.

## Authentication

With `DASHBOARD_TOKEN` set, supply it as a header or query parameter:

```bash
curl -H "x-dashboard-token: $TOKEN" localhost:8080/api/status
curl "localhost:8080/api/status?token=$TOKEN"
```

Without a token, the server binds to loopback and rejects non-local requests.

## Endpoints

### `GET /health` — unauthenticated

Liveness **and** readiness, for Docker's `HEALTHCHECK`. Returns `503` when the
bot is running but not functional — a bot that is up and disconnected from
Binance is not healthy.

```json
{
  "status": "healthy",
  "safe_mode": false,
  "safe_mode_reason": "",
  "uptime_sec": 3821.4,
  "mode": "PAPER"
}
```

### `GET /api/status`

Account, performance and — most usefully — why the bot is not trading.

```json
{
  "mode": "PAPER",
  "testnet": true,
  "equity": 75.42,
  "available_balance": 71.10,
  "today_pnl": 0.42,
  "total_pnl": 0.42,
  "open_positions": 1,
  "total_trades": 12,
  "win_rate": 0.583,
  "profit_factor": 1.34,
  "drawdown": 0.008,
  "total_fees": 0.96,
  "total_funding": 0.02,
  "open_risk_pct": 0.005,
  "exposure_ratio": 0.98,
  "safe_mode": false,
  "entries_allowed": true,
  "uptime_sec": 3821.4,
  "rejections": [
    {"reason": "NEGATIVE_EXPECTED_EDGE", "count": 847},
    {"reason": "INSUFFICIENT_CONSENSUS", "count": 231}
  ]
}
```

`rejections` is the field to read first when diagnosing an idle bot.

### `GET /api/positions`

```json
[{
  "symbol": "SOLUSDT", "direction": "LONG",
  "entry_price": 142.31, "current_price": 143.05,
  "quantity": 0.5, "leverage": 3,
  "unrealized_pnl": 0.37, "unrealized_pct": 0.0052, "r_multiple": 0.74,
  "stop_loss": 141.60, "take_profit": 143.72,
  "duration": "8m 21s", "strategy": "momentum", "score": 84.2,
  "trailing": false, "adopted": false
}]
```

`adopted: true` marks a position found on the exchange that the bot did not
open. It is protected and will be closed.

### `GET /api/opportunities`

The ranked candidates. **Appearing here does not mean a trade will be taken** —
a candidate must still clear consensus, the opportunity score, the edge filter
and every risk limit.

```json
[{
  "rank": 1, "symbol": "SOLUSDT", "market_score": 93.1,
  "regime": "STRONG_TREND", "volatility_pct": 0.482,
  "liquidity_usd": 1840000.0, "spread_bps": 0.8, "funding": 0.0001,
  "best_strategy": "momentum", "direction": "LONG", "confidence": 91.0,
  "expected_net_edge": 0.00142, "opportunity_score": 88.4, "risk_level": "LOW"
}]
```

`best_strategy`, `direction`, `confidence` and `expected_net_edge` are `null`
until the signal loop has evaluated that symbol.

### `GET /api/trades?limit=50`

Completed trades, newest first, with the full cost breakdown — `gross_pnl`,
`fees`, `funding`, `slippage_cost`, `net_pnl`. Net alone hides whether the
strategy actually had an edge.

### `GET /api/strategies`

Per-strategy performance, allocation weight and suspension state.

```json
{
  "strategies": {
    "momentum": {
      "trades": 42, "wins": 25, "win_rate": 0.595,
      "expectancy_r": 0.18, "profit_factor": 1.42,
      "max_drawdown_r": 3.2, "cumulative_r": 7.6, "sharpe_like": 0.21,
      "allocation_weight": 1.3, "suspended": false
    }
  },
  "registry": {"loaded": ["momentum", "..."], "suspended": []}
}
```

### `GET /api/risk`

Kill-switch state, active cooldowns, suspended strategies, allocation weights
and per-strategy performance.

### `GET /api/decisions?limit=100&accepted=false`

**The audit log** — one row per evaluated opportunity, accepted or rejected,
with the full context. This is what makes "why did the bot enter?" and "why
didn't it?" answerable months later.

```json
[{
  "symbol": "BTCUSDT", "timestamp": 1730000000000,
  "accepted": false, "stage": "edge",
  "rejection_reason": "NEGATIVE_EXPECTED_EDGE",
  "detail": "net edge -0.0210% does not clear 0.0800% (gross 0.0890%, costs 0.1100%, p=0.47); needs a 61.0% win rate to break even",
  "market_regime": "STRONG_TREND", "direction": "LONG",
  "strategies": ["momentum", "trend_following"],
  "consensus_score": 78.2, "opportunity_score": 81.0,
  "expected_net_edge": -0.00021, "win_probability": 0.47,
  "entry_price": 67420.5, "stop_loss": 67150.0, "take_profit": 67960.0,
  "context": {"costs": {"entry_fee": 0.0004, "exit_fee": 0.0004,
                        "spread_cost": 0.0001, "slippage": 0.0002,
                        "funding": 0.0, "total": 0.0011}}
}]
```

`accepted=false` is the useful filter: rejections are the majority and are
usually the more informative rows.

### `GET /api/health`

Per-component health, safe-mode state and resource usage.

## Errors

| Status | Meaning |
|---|---|
| `401` | missing or invalid token |
| `403` | no token configured and the request is not from localhost |
| `503` | `/health` only — running but not functional |

## Rate limiting

None. The API is intended for one operator through an SSH tunnel. The dashboard
polls every 5 seconds.
