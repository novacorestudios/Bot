"""Binance USDⓈ-M Futures WebSocket streams.

Two stream managers:

* :class:`MarketStream` — combined public streams (klines, book tickers, mark
  prices) for the current candidate symbols. Resubscribes on reconnect and
  supports changing the symbol set at runtime as the scanner re-ranks.
* :class:`UserStream` — the authenticated user data stream carrying
  ``ORDER_TRADE_UPDATE`` and ``ACCOUNT_UPDATE``. The listen key is renewed every
  30 minutes; if it expires the stream dies *silently*, which is why the
  keepalive task is not optional.

Reconnection uses exponential backoff with jitter. The jitter matters: without
it, every bot on a VPS reconnects at the same instant after a Binance restart
and immediately gets rate limited.

Binance also disconnects every connection after 24 hours by design, so a
reconnect is a normal event, not an error.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from tradebot.core.logging import get_logger
from tradebot.core.types import BookTicker, Candle, MarkPriceInfo
from tradebot.exchange.binance import parsers

log = get_logger(__name__)

MessageHandler = Callable[[str, Any], Awaitable[None]]


class _ReconnectingStream:
    """Shared reconnect/backoff machinery for both stream types."""

    def __init__(self, name: str, max_backoff: float = 60.0, ping_interval: float = 180.0) -> None:
        self.name = name
        self.max_backoff = max_backoff
        self.ping_interval = ping_interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None

        self.connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.message_count = 0
        self.last_message_at: float = 0.0
        self.last_error: str | None = None

    @property
    def is_stale(self) -> bool:
        return self.seconds_since_message > 60.0

    @property
    def seconds_since_message(self) -> float:
        if self.last_message_at == 0.0:
            return float("inf")
        return time.time() - self.last_message_at

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever(), name=f"ws:{self.name}")

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self.connected = False

    async def _run_forever(self) -> None:
        attempt = 0
        while self._running:
            try:
                await self._connect_once()
                attempt = 0  # a clean session resets backoff
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - must never die
                self.last_error = str(exc)
                self.disconnect_count += 1
                log.warning(
                    "ws_connection_failed",
                    stream=self.name,
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            finally:
                self.connected = False

            if not self._running:
                break

            # Exponential backoff with jitter, so a fleet of bots does not
            # reconnect in lockstep and trigger rate limits.
            attempt += 1
            base = min(self.max_backoff, 1.0 * (2 ** min(attempt, 6)))
            # Jitter, not cryptography: without it every bot on a host
            # reconnects in lockstep after a Binance restart and is rate limited.
            delay = base * (0.5 + random.random() * 0.5)  # noqa: S311  # nosec B311
            log.info(
                "ws_reconnecting", stream=self.name, attempt=attempt, delay_sec=round(delay, 2)
            )
            await asyncio.sleep(delay)

    async def _connect_once(self) -> None:
        raise NotImplementedError

    async def _consume(
        self,
        url: str,
        on_message: MessageHandler,
        on_connected: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Open one WebSocket and pump messages until it closes."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        async with self._session.ws_connect(
            url, heartbeat=self.ping_interval, timeout=aiohttp.ClientWSTimeout(ws_close=30.0)
        ) as websocket:
            self._ws = websocket
            self.connected = True
            self.connect_count += 1
            self.last_message_at = time.time()
            log.info("ws_connected", stream=self.name, url=_redact_url(url))

            if on_connected is not None:
                await on_connected()

            async for message in websocket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    self.message_count += 1
                    self.last_message_at = time.time()
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        log.warning("ws_bad_json", stream=self.name)
                        continue
                    # Unwrap the combined-stream envelope, then let the
                    # handler decide — never assume a mapping here, because
                    # array streams are legitimate payloads.
                    stream_name, inner = parsers.unwrap(payload)
                    await on_message(stream_name or parsers.event_type(inner), inner)
                elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    log.warning("ws_closed_by_peer", stream=self.name, type=message.type.name)
                    break

        self._ws = None
        self.connected = False
        self.disconnect_count += 1

    def stats(self) -> dict[str, Any]:
        return {
            "stream": self.name,
            "connected": self.connected,
            "connects": self.connect_count,
            "disconnects": self.disconnect_count,
            "messages": self.message_count,
            "seconds_since_message": round(self.seconds_since_message, 1)
            if self.last_message_at
            else None,
            "last_error": self.last_error,
        }


def _redact_url(url: str) -> str:
    """A user-stream URL contains the listen key. Never log it whole."""
    if "/ws/" in url:
        head, _, _tail = url.partition("/ws/")
        return f"{head}/ws/<listenKey>"
    return url.split("?")[0] + ("?<streams>" if "?" in url else "")


class MarketStream(_ReconnectingStream):
    """Combined public market streams for a dynamic set of symbols."""

    def __init__(
        self,
        base_url: str,
        on_candle: Callable[[str, str, Candle], Awaitable[None]],
        on_book: Callable[[BookTicker], Awaitable[None]] | None = None,
        on_mark: Callable[[MarkPriceInfo], Awaitable[None]] | None = None,
        timeframes: tuple[str, ...] = ("1m",),
        include_book: bool = True,
        include_mark: bool = True,
        on_connect: Callable[[], Awaitable[None]] | None = None,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__("market")
        self.base_url = base_url.rstrip("/")
        self.on_candle = on_candle
        self.on_book = on_book
        self.on_mark = on_mark
        self.timeframes = timeframes
        self.include_book = include_book
        self.include_mark = include_mark
        # A reconnect is not merely a resubscribe: while the socket was down the
        # exchange may have filled or cancelled orders we never saw, so the
        # engine is told and reconciles (AUDIT_REPORT.md C-1).
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self._symbols: tuple[str, ...] = ()

    def build_streams(self, symbols: tuple[str, ...]) -> list[str]:
        """Stream names for the requested symbols, per the official naming scheme."""
        streams: list[str] = []
        for symbol in symbols:
            lower = symbol.lower()
            streams.extend(f"{lower}@kline_{tf}" for tf in self.timeframes)
            if self.include_book:
                streams.append(f"{lower}@bookTicker")
        if self.include_mark and symbols:
            # One array stream for every symbol is far cheaper than one per symbol.
            streams.append("!markPrice@arr@1s")
        return streams

    async def set_symbols(self, symbols: list[str] | tuple[str, ...]) -> None:
        """Change the subscribed set. Triggers a reconnect when it actually differs.

        Binance caps a single connection at 1024 streams and 200 subscription
        messages per 24h, so reconnecting with a fresh combined URL is more
        robust than incremental SUBSCRIBE churn as the top-25 rotates.
        """
        new = tuple(sorted(symbols))
        if new == self._symbols:
            return
        self._symbols = new
        log.info("ws_symbols_changed", count=len(new))
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()  # the run loop reconnects with the new set

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    async def _connect_once(self) -> None:
        if not self._symbols:
            await asyncio.sleep(1.0)
            return
        streams = self.build_streams(self._symbols)
        url = f"{self.base_url}/stream?streams={'/'.join(streams)}"
        try:
            await self._consume(url, self._handle, self.on_connect)
        finally:
            if self.on_disconnect is not None:
                await self.on_disconnect()

    async def _handle(self, stream: str, data: Any) -> None:
        """Route one payload to its handler.

        ``data`` is deliberately typed ``Any``: ``!markPrice@arr@1s`` delivers a
        JSON array, and assuming a mapping here is what caused AUDIT_REPORT.md
        C-2. All shape handling lives in ``parsers``, which resolves it once.
        """
        kind = parsers.event_type(data)

        if kind == "kline":
            parsed = parsers.parse_kline(data)
            if parsed is not None:
                symbol, interval, candle = parsed
                await self.on_candle(symbol, interval, candle)
            return

        if kind == "bookTicker" and self.on_book is not None:
            book = parsers.parse_book_ticker(data)
            if book is not None:
                await self.on_book(book)
            return

        if kind == "markPriceUpdate" and self.on_mark is not None:
            # Handles BOTH shapes: the single-symbol stream and the array
            # stream, which is the one that used to crash.
            for mark in parsers.parse_mark_price(data):
                await self.on_mark(mark)
            return

        if kind:
            log.debug("ws_unhandled_event", stream=stream, event=kind)


class UserStream(_ReconnectingStream):
    """Authenticated user data stream: order and account updates."""

    def __init__(
        self,
        base_url: str,
        rest_client: Any,
        on_event: Callable[[str, Any], Awaitable[None]],
        keepalive_interval: float = 1800.0,
        on_connect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__("user")
        self.base_url = base_url.rstrip("/")
        self.rest = rest_client
        self.on_event = on_event
        self.keepalive_interval = keepalive_interval
        self.on_connect = on_connect
        self._listen_key: str | None = None
        self._keepalive_task: asyncio.Task | None = None

    async def start(self) -> None:
        await super().start()
        if self._keepalive_task is None:
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(), name="ws:user:keepalive"
            )

    async def stop(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None
        if self._listen_key:
            with contextlib.suppress(Exception):
                await self.rest.close_listen_key()
            self._listen_key = None
        await super().stop()

    async def _keepalive_loop(self) -> None:
        """Renew the listen key. Without this the stream dies after 60 minutes."""
        while self._running:
            await asyncio.sleep(self.keepalive_interval)
            if not self._listen_key:
                continue
            ok = await self.rest.keepalive_listen_key()
            if not ok:
                log.warning("listen_key_renewal_failed_forcing_reconnect")
                self._listen_key = None
                if self._ws is not None and not self._ws.closed:
                    await self._ws.close()

    async def _connect_once(self) -> None:
        self._listen_key = await self.rest.create_listen_key()
        url = f"{self.base_url}/ws/{self._listen_key}"
        await self._consume(url, self._handle, self.on_connect)

    async def _handle(self, event: str, data: Any) -> None:
        kind = parsers.event_type(data) or event
        if kind == "listenKeyExpired":
            log.warning("listen_key_expired_reconnecting")
            self._listen_key = None
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
            return
        await self.on_event(kind, data)
