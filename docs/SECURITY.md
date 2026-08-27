# SECURITY

## Threat model

What this system is actually protecting against, in order of likelihood:

| Threat | Impact | Mitigation |
|---|---|---|
| API key leaked (git, logs, backup) | funds stolen or drained by trading | no withdrawal permission, IP allow-list, two-layer log redaction, CI secret scan |
| VPS compromised | full control of the bot and key | minimal permissions, non-root container, no inbound ports, key is IP-locked |
| Dashboard exposed | balances and positions disclosed | loopback-only without a token; read-only endpoints |
| Bot bug | account damaged by its own trading | risk engine, kill switches, mandatory stops, reconciliation |
| Dependency compromise | arbitrary code execution | pinned ranges, `pip-audit` in CI, few dependencies |

The **API key is the crown jewel**, and the single most important control is
that it cannot withdraw. A key that can only trade limits a total compromise to
losses from bad trading; a key that can withdraw makes it total.

## API key rules

**Non-negotiable:**

1. **Withdrawals OFF.** `scripts/verify_connectivity.py` checks this against the
   live API and fails loudly if the key can withdraw.
2. **IP allow-list** the VPS address.
3. A **dedicated key** for this bot — never one shared with another tool, so it
   can be revoked without collateral damage.
4. Futures enabled; nothing else.

**If a key is ever exposed — in a commit, a log, a screenshot, a paste — the
only remedy is to delete it on Binance and create a new one.** Removing the file
does not help: it is in the git history, in the CI cache, and possibly already
scraped. Rotate first, investigate second.

## Secrets

Secrets come from environment variables only. There is no code path that reads a
credential from a file the repository ships or writes one to disk.

`.gitignore` excludes `.env` and every `.env.*` except `.env.example`.
`scripts/check_secrets.py` runs in CI and scans tracked files for API-key
assignments, Telegram tokens, private-key blocks and AWS keys, and fails the
build if `.env` is ever tracked.

Run it locally before pushing:

```bash
python scripts/check_secrets.py
```

## Log redaction

Two independent layers, both applied to every log event:

1. **Registered literals.** Every secret is registered at startup; any occurrence
   of its exact value is replaced anywhere in the event, including inside
   nested structures and formatted strings.
2. **Sensitive key names.** Values under keys matching `api_key`, `secret`,
   `signature`, `token`, `password`, `listenKey`, `authorization` and similar
   are replaced regardless of content.

Two layers because either alone fails: a literal scan misses a secret that was
never registered, and a key-name filter misses a token embedded mid-sentence.

Specific care is taken where secrets naturally appear in URLs:

- the Telegram bot token is in the request URL, so Telegram URLs are never
  logged — only status codes
- a user-stream WebSocket URL contains the listen key, so URLs are redacted
  before logging
- a Postgres `DATABASE_URL` can contain a password, so it is redacted at connect

## Container hardening

```dockerfile
USER tradebot          # uid 10001, non-root
```

- multi-stage build: no compiler in the runtime image
- `no-new-privileges:true`
- config mounted **read-only** — the bot cannot rewrite its own risk limits
- dashboard published to `127.0.0.1` only
- memory limit set, so a leak cannot take the host down

## Network exposure

The bot needs **no inbound ports**. Outbound to Binance and Telegram only.

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp        # SSH; consider restricting to your own IP
sudo ufw enable
```

Reach the dashboard through an SSH tunnel rather than opening a port.

## Dashboard

- read-only: **no endpoint can open, close or size a position, or reset a kill
  switch.** A test enumerates the routes and asserts they are all `GET`.
- without `DASHBOARD_TOKEN`, binds to loopback and rejects non-local requests
- with a token, requires it as a header or query parameter
- no `/docs`, `/redoc` or `/openapi.json` — no schema disclosure

Deliberately absent: any control that could move money. A web endpoint is a much
larger attack surface than a file on the VPS, and the convenience is not worth
it.

## What the AI layer may not do

`AI_MUST_NOT_HAVE_DIRECT_AUTHORITY_TO_PLACE_ORDERS` is structural, not a policy
note. The AI layer has no gateway reference, no `OrderIntent` constructor and no
path to the execution engine. It can only produce advisory inputs that the same
scoring and risk gates then evaluate like any other input.

The same applies to strategies: `MarketView` carries no account, no positions,
no equity and no gateway, and a test asserts those attributes are absent.

## Dependency management

Few dependencies, deliberately. No `ccxt` (we want exact control over Binance
semantics), no TA-Lib (build friction and a large C surface), no scraping
libraries.

```bash
pip-audit -r requirements.txt      # runs in CI
bandit -r src -c pyproject.toml    # runs in CI
```

## Incident response

**Suspected key compromise:**

1. Delete the key on Binance — immediately, before anything else.
2. `docker compose down`.
3. Check `/fapi/v1/income` and the Binance UI for trades you did not make.
4. Create a new key with withdrawals off and the VPS IP allow-listed.
5. Only then work out how it leaked.

**Bot behaving unexpectedly:**

```bash
docker compose stop tradebot     # positions stay open, protected by
                                 # exchange-side stops
```

Then close positions manually in the Binance UI if warranted, and read
`risk_events` and `decisions` in the database — every decision, including every
rejection, is recorded with its full context.

**Suspected VPS compromise:** assume the key is compromised. Rotate it first,
then rebuild the host. Do not try to clean it.

## Security checklist before live

- [ ] API key: withdrawals disabled, verified by `verify_connectivity.py`
- [ ] API key: IP-restricted to the VPS
- [ ] API key: dedicated to this bot
- [ ] `.env` is `chmod 600` and not in git
- [ ] `python scripts/check_secrets.py` passes
- [ ] `DASHBOARD_TOKEN` set, or dashboard on loopback only
- [ ] Firewall: inbound denied except SSH
- [ ] SSH: key-based authentication, password login disabled
- [ ] Backups exist, and `.env` is backed up separately and encrypted
- [ ] Telegram alerts confirmed working (trigger one deliberately)
- [ ] `bandit` and `pip-audit` clean
