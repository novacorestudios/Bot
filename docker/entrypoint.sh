#!/bin/sh
# Fail fast and loudly rather than starting in an unknown state.
set -eu

echo "tradebot entrypoint: mode=${TRADING_MODE:-PAPER} testnet=${BINANCE_TESTNET:-true}"

# Refuse to start LIVE unless every confirmation is present. The CLI enforces
# this too; duplicating it here means a mis-set container variable is caught
# before the process ever opens a socket to Binance.
if [ "${TRADING_MODE:-PAPER}" = "LIVE" ]; then
    if [ "${I_UNDERSTAND_LIVE_TRADING_RISK:-NO}" != "YES" ]; then
        echo "REFUSING TO START: TRADING_MODE=LIVE without I_UNDERSTAND_LIVE_TRADING_RISK=YES" >&2
        exit 78
    fi
    if [ "${BINANCE_TESTNET:-true}" = "true" ]; then
        echo "REFUSING TO START: TRADING_MODE=LIVE with BINANCE_TESTNET=true" >&2
        exit 78
    fi
    set -- "$@" --live
fi

python -m tradebot.app.cli validate-config

exec python -m tradebot.app.cli "$@"
