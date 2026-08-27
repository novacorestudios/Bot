#!/usr/bin/env python3
"""Verify Binance connectivity, credentials and permissions.

Run this FIRST on any host that will run the bot, and run it against testnet
before ever pointing it at production:

    BINANCE_TESTNET=true python scripts/verify_connectivity.py

It checks, in order, and stops at the first hard failure:

1. Public REST reachability and clock skew
2. Symbol universe and filters
3. Klines and book tickers
4. Authenticated endpoints (balance, positions, open orders)
5. **That the API key cannot withdraw** — the single most important check
6. WebSocket market stream
7. WebSocket user data stream

It places NO orders.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.core.config import Settings  # noqa: E402
from tradebot.core.errors import ExchangeError  # noqa: E402
from tradebot.core.logging import configure_logging, register_secret  # noqa: E402
from tradebot.exchange.binance.rest import BinanceFuturesREST  # noqa: E402

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    marker = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " WARN ", SKIP: " skip "}[status]
    print(f"[{marker}] {name}" + (f"\n           {detail}" if detail else ""))


async def main() -> int:
    settings = Settings()
    register_secret(settings.binance_api_key)
    register_secret(settings.binance_api_secret)
    configure_logging("WARNING", "console", None)

    print("=" * 72)
    print("Binance USDⓈ-M Futures connectivity check")
    print(f"  endpoint : {settings.rest_url}")
    print(f"  testnet  : {settings.binance_testnet}")
    print(f"  key set  : {bool(settings.binance_api_key)}")
    print("=" * 72)

    if not settings.binance_testnet:
        print("\n  WARNING: this is the PRODUCTION endpoint.\n")

    client = BinanceFuturesREST(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        base_url=settings.rest_url,
        recv_window=settings.binance_recv_window,
    )

    try:
        # -- 1. reachability + clock ---------------------------------------- #
        try:
            started = time.time()
            await client.connect()
            latency_ms = (time.time() - started) * 1000
            record("REST reachable", PASS, f"handshake {latency_ms:.0f} ms")
        except Exception as exc:  # noqa: BLE001
            record("REST reachable", FAIL, f"{type(exc).__name__}: {exc}")
            print("\nCannot reach Binance. Check DNS, firewall, and whether this "
                  "region is blocked. Nothing else can be tested.")
            return 1

        offset = client.clock.offset_ms
        if abs(offset) > 1000:
            record("Clock skew", FAIL, f"offset {offset} ms — install/repair NTP; "
                                       "Binance rejects requests outside recvWindow")
        elif abs(offset) > 300:
            record("Clock skew", WARN, f"offset {offset} ms — consider NTP")
        else:
            record("Clock skew", PASS, f"offset {offset} ms")

        # -- 2. universe ---------------------------------------------------- #
        tradable = [s for s in client.symbols.values() if s.is_tradable
                    and s.quote_asset == "USDT"]
        if len(tradable) < 50:
            record("Symbol universe", WARN, f"only {len(tradable)} USDT perpetuals")
        else:
            record("Symbol universe", PASS, f"{len(tradable)} USDT perpetuals tradable")

        with_notional = [s for s in tradable if s.min_notional > 0]
        if with_notional:
            worst = max(with_notional, key=lambda s: s.min_notional)
            record("Symbol filters", PASS,
                   f"min notional ranges up to {worst.min_notional} USDT "
                   f"({worst.symbol}) — small accounts cannot trade every symbol")

        # -- 3. market data ------------------------------------------------- #
        probe = tradable[0].symbol if tradable else "BTCUSDT"
        try:
            candles = await client.get_klines(probe, "1m", limit=10)
            closed = sum(1 for c in candles if c.closed)
            record("Klines", PASS, f"{probe}: {len(candles)} bars, {closed} closed")
        except Exception as exc:  # noqa: BLE001
            record("Klines", FAIL, str(exc))

        try:
            books = await client.get_book_ticker()
            record("Book tickers", PASS, f"{len(books)} symbols")
        except Exception as exc:  # noqa: BLE001
            record("Book tickers", FAIL, str(exc))

        try:
            marks = await client.get_mark_price()
            record("Mark price / funding", PASS, f"{len(marks)} symbols")
        except Exception as exc:  # noqa: BLE001
            record("Mark price / funding", FAIL, str(exc))

        # -- 4. authenticated ----------------------------------------------- #
        if not client.authenticated:
            record("Authenticated endpoints", SKIP, "no API credentials configured")
            record("Withdrawal permission", SKIP, "cannot check without credentials")
        else:
            try:
                account = await client.get_account()
                record("Account access", PASS,
                       f"equity {account.equity:.2f} USDT, "
                       f"available {account.available_balance:.2f}")
            except Exception as exc:  # noqa: BLE001
                record("Account access", FAIL, f"{type(exc).__name__}: {exc}")
                print("\nThe key was rejected. Check that futures trading is enabled "
                      "for it and that this host's IP is on its allow-list.")
                return 1

            try:
                positions = await client.get_positions()
                record("Position access", PASS, f"{len(positions)} open positions")
                if positions:
                    print("           NOTE: this account already holds positions. "
                          "The bot will adopt and protect them on start.")
            except Exception as exc:  # noqa: BLE001
                record("Position access", FAIL, str(exc))

            try:
                orders = await client.get_open_orders()
                record("Open order access", PASS, f"{len(orders)} open orders")
            except Exception as exc:  # noqa: BLE001
                record("Open order access", FAIL, str(exc))

            # -- 5. THE important one --------------------------------------- #
            await check_withdrawal_permission(settings)

        # -- 6/7. websockets ------------------------------------------------ #
        await check_market_stream(settings, probe)
        if client.authenticated:
            await check_user_stream(client)
        else:
            record("WebSocket user stream", SKIP, "no API credentials configured")

    finally:
        await client.close()

    # -- summary ------------------------------------------------------------ #
    print("=" * 72)
    failures = [r for r in results if r[1] == FAIL]
    warnings = [r for r in results if r[1] == WARN]
    print(f"{len(results)} checks: {len(results) - len(failures) - len(warnings)} pass, "
          f"{len(warnings)} warn, {len(failures)} fail")
    if failures:
        print("\nFailures:")
        for name, _, detail in failures:
            print(f"  - {name}: {detail}")
        return 1
    print("\nConnectivity is healthy. This says nothing about whether any strategy "
          "is profitable — see IMPLEMENTATION_PLAN.md §9.")
    return 0


async def check_withdrawal_permission(settings: Settings) -> None:
    """Confirm the key CANNOT withdraw.

    Futures keys are created on the spot account, and the withdrawal permission
    lives there. We query the spot API's account snapshot for its permission
    flags. A key that can withdraw must never be used by a bot: a compromise
    then costs the balance, not just the open positions.
    """
    import aiohttp

    from tradebot.core.clock import SystemClock

    if settings.binance_testnet:
        record("Withdrawal permission", SKIP,
               "testnet key — re-run this check against production before going live")
        return

    import hashlib
    import hmac
    from urllib.parse import urlencode

    query = urlencode({"timestamp": SystemClock().now_ms(), "recvWindow": 5000})
    signature = hmac.new(settings.binance_api_secret.encode(), query.encode(),
                         hashlib.sha256).hexdigest()
    url = f"https://api.binance.com/sapi/v1/account/apiRestrictions?{query}&signature={signature}"

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session, session.get(
            url, headers={"X-MBX-APIKEY": settings.binance_api_key}
        ) as response:
            if response.status != 200:
                body = await response.text()
                record("Withdrawal permission", WARN,
                       f"could not read API restrictions (HTTP {response.status}: "
                       f"{body[:120]}). VERIFY MANUALLY in the Binance UI that "
                       f"'Enable Withdrawals' is OFF.")
                return
            data = await response.json()
    except Exception as exc:  # noqa: BLE001
        record("Withdrawal permission", WARN,
               f"check failed ({exc}). VERIFY MANUALLY that withdrawals are disabled.")
        return

    can_withdraw = bool(data.get("enableWithdrawals", False))
    can_trade_futures = bool(data.get("enableFutures", False))
    ip_restricted = bool(data.get("ipRestrict", False))

    if can_withdraw:
        record("Withdrawal permission", FAIL,
               "THIS KEY CAN WITHDRAW FUNDS. Disable 'Enable Withdrawals' on the "
               "key immediately, or create a new key without it. Do not run the "
               "bot with this key.")
    else:
        record("Withdrawal permission", PASS, "withdrawals disabled")

    if not can_trade_futures:
        record("Futures permission", FAIL, "'Enable Futures' is off for this key")
    else:
        record("Futures permission", PASS, "futures trading enabled")

    if not ip_restricted:
        record("IP restriction", WARN,
               "the key accepts requests from any IP. Restrict it to this VPS.")
    else:
        record("IP restriction", PASS, "key is IP-restricted")


async def check_market_stream(settings: Settings, symbol: str) -> None:
    from tradebot.exchange.binance.ws import MarketStream

    received: list[str] = []

    async def on_candle(sym: str, _tf: str, _candle) -> None:
        received.append(sym)

    stream = MarketStream(settings.ws_url, on_candle, timeframes=("1m",),
                          include_book=False, include_mark=False)
    await stream.set_symbols([symbol])
    await stream.start()
    try:
        for _ in range(20):
            await asyncio.sleep(1.0)
            if received:
                break
    finally:
        await stream.stop()

    if received:
        record("WebSocket market stream", PASS,
               f"{len(received)} kline messages for {symbol}")
    else:
        record("WebSocket market stream", FAIL,
               "no messages within 20s — check that outbound wss:// is permitted")


async def check_user_stream(client: BinanceFuturesREST) -> None:
    try:
        key = await client.create_listen_key()
        ok = await client.keepalive_listen_key()
        await client.close_listen_key()
        record("WebSocket user stream", PASS,
               f"listen key obtained ({len(key)} chars), keepalive "
               f"{'ok' if ok else 'FAILED'}")
    except ExchangeError as exc:
        record("WebSocket user stream", FAIL, str(exc))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(asyncio.run(main()))
