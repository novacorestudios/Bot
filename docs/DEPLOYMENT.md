# DEPLOYMENT

## Server requirements

Modest. The bot is I/O-bound, not compute-bound.

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| CPU | 1 vCPU | 2 vCPU | indicator maths on ~75 symbols per scan |
| RAM | 1 GB | 2 GB | ~500 bars × 5 timeframes × 75 symbols in memory |
| Disk | 10 GB | 20 GB | SQLite plus logs; pruning keeps it bounded |
| Network | stable | low latency to Binance | latency is a direct cost on a scalper |

**Location matters.** Binance's futures API is served from AWS Tokyo
(`ap-northeast-1`). A VPS there sees round trips of ~5 ms; one in Europe or the
US sees 150–250 ms. On a strategy holding positions for minutes that is not
fatal, but it is a real and permanent cost paid on every entry and exit.

Any 5 USD/month VPS is adequate. Region is worth more than specification here.

## Installation

```bash
# Docker (recommended)
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker "$USER" && newgrp docker

git clone https://github.com/novacorestudios/Bot.git tradebot
cd tradebot
cp .env.example .env
```

## Configuration

Edit `.env`. Start here, and change nothing else until paper trading has run:

```bash
TRADING_MODE=PAPER
BINANCE_TESTNET=true
I_UNDERSTAND_LIVE_TRADING_RISK=NO

BINANCE_API_KEY=<testnet key>
BINANCE_API_SECRET=<testnet secret>

DASHBOARD_TOKEN=<openssl rand -hex 32>
TELEGRAM_BOT_TOKEN=<from @BotFather>
TELEGRAM_CHAT_ID=<from @userinfobot>
TELEGRAM_ENABLED=true
```

```bash
chmod 600 .env      # it contains credentials
```

Tunables live in `config/config.yaml`, mounted **read-only** into the container
so the bot can never rewrite its own risk limits.

## Binance setup

1. Create an API key at <https://www.binance.com/en/my/settings/api-management>
   (or the testnet at <https://testnet.binancefuture.com>).
2. Permissions: **Enable Futures ON. Enable Withdrawals OFF.**
3. Restrict access to your VPS IP.
4. Verify:
   ```bash
   python scripts/verify_connectivity.py
   ```
   The check **fails loudly** if the key can withdraw. Do not proceed past that.

## Telegram setup

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token.
2. Message [@userinfobot](https://t.me/userinfobot) → copy your numeric id.
3. Send your bot any message once, so it is allowed to reply to you.

## Running

```bash
cd docker
docker compose --env-file ../.env up -d
docker compose logs -f tradebot
```

Verify:

```bash
curl -s localhost:8080/health | python -m json.tool
```

`/health` reports whether the bot is **functional**, not merely alive — a bot
that is up but disconnected from Binance returns 503, and Docker restarts it.

### The container command is a bare subcommand

The image's `ENTRYPOINT` is `docker/entrypoint.sh`, which ends with:

```sh
exec python -m tradebot.app.cli "$@"
```

**The interpreter is already supplied.** Whatever you pass to `docker run` after
the image name becomes the CLI's *subcommand*, so it must be one of
`validate-config`, `doctor`, `run`, `scan`, `backtest`, `walkforward`.
`CMD ["run"]` is the default.

```bash
docker run --rm -e TRADING_MODE=PAPER tradebot:ci validate-config   # correct
docker run --rm tradebot:ci                                        # correct: CMD is `run`

docker run --rm tradebot:ci python -m tradebot.app.cli validate-config
# tradebot: error: argument command: invalid choice: 'python'
```

The last form reads as correct — that string *is* a valid command — but not at
this layer, and it exits 2. If you need a raw interpreter (a debug shell, an
ad-hoc script), override the entrypoint explicitly:

```bash
docker run --rm --entrypoint python tradebot:ci -c "import tradebot; print(tradebot.__file__)"
```

`docker compose exec` is the exception: it bypasses the `ENTRYPOINT`, so an
explicit `python -m tradebot.app.cli ...` is correct there and is what the
examples further down use.

Before running anything, the entrypoint validates the configuration and, when
`TRADING_MODE=LIVE`, refuses to start unless every confirmation agrees —
exiting **78** (`EX_CONFIG`) rather than opening a socket to Binance. That
refusal is exercised by CI on every build.

## Dashboard access

The dashboard binds to **loopback only** unless `DASHBOARD_TOKEN` is set. It
exposes balances and open positions, so never publish it on an open port.

The safe way in is an SSH tunnel — no open port, no TLS to manage:

```bash
ssh -L 8080:localhost:8080 user@your-vps
# then open http://localhost:8080?token=<DASHBOARD_TOKEN>
```

If you must expose it, terminate TLS at a reverse proxy (Caddy handles
certificates automatically) and keep the token.

## Database

SQLite by default: one file, no server to also recover after a crash, entirely
adequate for one bot.

```bash
# Postgres, if you prefer
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/tradebot
```

Retention is configured in `config.yaml`; `market_snapshots` and `signals` are
pruned by age. `trades` and `risk_events` are kept forever — they are the record
that matters.

## Backup

```bash
# Daily, via cron
0 3 * * * docker run --rm -v tradebot-data:/data -v /backup:/backup alpine \
  sh -c 'cp /data/tradebot.db /backup/tradebot-$(date +\%F).db'
```

Back up `.env` **separately and encrypted**. It contains credentials, so it must
never sit next to the database backups.

## Recovery

The bot is designed to be killed at any moment. On restart it:

1. connects to Binance
2. fetches account, positions and open orders
3. rebuilds local state
4. reconciles against the database
5. **adopts and protects** any position it did not know about
6. only then permits new entries

Nothing needs to be done by hand:

```bash
docker compose restart tradebot
docker compose logs --since 5m tradebot | grep reconciliation
```

If reconciliation cannot complete, entries stay blocked — deliberately.
Unverified local state must never permit a new position.

## Logs

```bash
docker compose logs -f tradebot                     # follow
docker compose logs tradebot | grep -i risk_event   # risk events
docker compose logs tradebot | grep kill_switch     # kill switches
```

Structured JSON, rotated at 20 MB × 5 files. **Secrets are redacted at two
layers** — registered literal values and sensitive key names — so logs can be
shared in a bug report without scrubbing.

## Monitoring

Check daily, in this order:

1. `/health` — is it functional?
2. Telegram — any risk alerts overnight?
3. Dashboard "Why the bot is not trading" — rejection counts tell you far more
   than the PnL does.
4. Drawdown against `max_drawdown`.

`NEGATIVE_EXPECTED_EDGE` dominating the rejections is **correct behaviour**, not
a fault. It means the strategies are finding setups whose expected move does not
cover costs.

## Updating

```bash
cd tradebot
docker compose down                 # positions stay open, protected by
                                    # their resting stops on the exchange
git pull
docker compose build
docker compose up -d                # reconciliation adopts them on start
```

Positions are deliberately **not** flattened on shutdown: an operator
restarting for a deploy does not want their book closed. They remain protected
by exchange-side stops and are adopted on the next start.

For a change that alters trading behaviour, prefer flattening first:

```bash
docker compose exec tradebot python -m tradebot.app.cli validate-config
```

## Rollback

```bash
git log --oneline -10
git checkout <previous-commit>
docker compose build && docker compose up -d
```

The database schema is additive; an older build reads a newer database. If a
migration ever becomes destructive it will be documented in the release notes.

## Going live

Only after every gate in `IMPLEMENTATION_PLAN.md` §9. Then:

```bash
# .env
TRADING_MODE=LIVE
I_UNDERSTAND_LIVE_TRADING_RISK=YES
BINANCE_TESTNET=false
BINANCE_API_KEY=<production key, no withdrawal, IP-locked>
```

The entrypoint adds `--live` automatically when `TRADING_MODE=LIVE`, and refuses
to start if the acknowledgement is missing or testnet is still set. All three
switches must agree; no single edit can reach live trading.

Start with **half** the tested risk per trade for the first week. The first live
week exists to measure the gap between paper fills and real ones.

## Troubleshooting

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
