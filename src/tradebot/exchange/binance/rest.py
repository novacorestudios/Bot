"""Binance USDⓈ-M Futures REST client.

Written against the official API documentation. Endpoints, weights and error
codes come from ``https://developers.binance.com/docs/derivatives/usds-margined-futures``.

The safety-critical behaviours implemented here:

* **Idempotent submission.** Every order carries a deterministic
  ``newClientOrderId``. On a timeout the client does NOT re-send — it queries
  the order by that id first. A blind retry is how a bot ends up with two
  positions where it believes it has one.
* **Weight-aware rate limiting** before each call, corrected afterwards from the
  server's own headers.
* **Clock discipline.** The local/server offset is measured at connect and
  re-measured on a ``-1021``, then the request is retried exactly once.
* **No secret ever leaves this module.** The signature is computed here and the
  key header is set here; the logger's redaction covers the rest.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import urlencode

import aiohttp

from tradebot.core.clock import SystemClock
from tradebot.core.errors import (
    AuthenticationError,
    ClockSkewError,
    ExchangeError,
    NetworkError,
    RateLimitError,
    TimeoutError_,
    UnknownOrderError,
)
from tradebot.core.logging import get_logger
from tradebot.core.types import (
    AccountState,
    BookTicker,
    Candle,
    Direction,
    MarkPriceInfo,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SymbolInfo,
    Ticker24h,
    TimeInForce,
)
from tradebot.exchange.binance import errors as berrors
from tradebot.exchange.binance.filters import (
    format_order_params,
    parse_symbol_info,
    validate_order,
)
from tradebot.exchange.binance.ratelimit import RateLimiter

log = get_logger(__name__)

# Endpoint weights from the official documentation. Where an endpoint's weight
# varies with `limit`, the heaviest realistic value is used — over-reserving
# costs throughput, under-reserving costs an IP ban.
WEIGHTS: dict[str, int] = {
    "/fapi/v1/ping": 1,
    "/fapi/v1/time": 1,
    "/fapi/v1/exchangeInfo": 1,
    "/fapi/v1/klines": 10,
    "/fapi/v1/depth": 20,
    "/fapi/v1/ticker/24hr": 40,  # 40 when no symbol is given
    "/fapi/v1/ticker/bookTicker": 5,  # 5 when no symbol is given
    "/fapi/v1/premiumIndex": 10,
    "/fapi/v1/fundingRate": 1,
    "/fapi/v1/leverageBracket": 1,
    "/fapi/v2/balance": 5,
    "/fapi/v2/account": 5,
    "/fapi/v2/positionRisk": 5,
    "/fapi/v1/order": 1,
    "/fapi/v1/openOrders": 40,  # 40 when no symbol is given
    "/fapi/v1/allOpenOrders": 1,
    "/fapi/v1/userTrades": 5,
    "/fapi/v1/income": 30,
    "/fapi/v1/leverage": 1,
    "/fapi/v1/marginType": 1,
    "/fapi/v1/listenKey": 1,
}

ORDER_ENDPOINTS = {"/fapi/v1/order", "/fapi/v1/allOpenOrders"}


class BinanceFuturesREST:
    """Async REST client for Binance USDⓈ-M Futures."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",  # nosec B107 - empty default; public endpoints only
        base_url: str = "https://testnet.binancefuture.com",
        recv_window: int = 5000,
        timeout_sec: float = 10.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 0.5,
        clock: SystemClock | None = None,
        rate_limiter: RateLimiter | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret.encode("utf-8") if api_secret else b""
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.clock = clock or SystemClock()
        self.limiter = rate_limiter or RateLimiter()
        self._session = session
        self._owns_session = session is None
        self._symbols: dict[str, SymbolInfo] = {}
        self._listen_key: str | None = None

        # Observability counters — surfaced on the dashboard and used by the
        # API-error kill switch.
        self.request_count = 0
        self.error_count = 0
        self.last_error: str | None = None
        self.last_request_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def authenticated(self) -> bool:
        return bool(self._api_key and self._api_secret)

    async def connect(self) -> None:
        """Open the HTTP session, sync the clock and load symbol metadata."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                headers={"User-Agent": "tradebot/0.1"},
            )
            self._owns_session = True
        await self.sync_time()
        await self.load_symbols()
        log.info(
            "binance_rest_connected",
            base_url=self.base_url,
            symbols=len(self._symbols),
            authenticated=self.authenticated,
            clock_offset_ms=self.clock.offset_ms,
        )

    async def close(self) -> None:
        if self._session and self._owns_session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> BinanceFuturesREST:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Signing
    # ------------------------------------------------------------------ #
    def _sign(self, query: str) -> str:
        """HMAC-SHA256 over the exact query string, as the documentation requires."""
        return hmac.new(self._api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()

    def _prepare_signed(self, params: dict[str, Any]) -> str:
        """Build the signed query string. Parameter order must not change after signing."""
        if not self.authenticated:
            raise AuthenticationError("this endpoint requires API credentials; none are configured")
        payload = {k: v for k, v in params.items() if v is not None}
        payload["timestamp"] = self.clock.now_ms()
        payload["recvWindow"] = self.recv_window
        query = urlencode(payload, doseq=True)
        return f"{query}&signature={self._sign(query)}"

    # ------------------------------------------------------------------ #
    # Request plumbing
    # ------------------------------------------------------------------ #
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        weight: int | None = None,
        retry_on_skew: bool = True,
    ) -> Any:
        """Issue one request with rate limiting, error mapping and bounded retries."""
        if self._session is None or self._session.closed:
            raise NetworkError("HTTP session is not open; call connect() first")

        params = params or {}
        is_order = endpoint in ORDER_ENDPOINTS and method in {"POST", "DELETE"}
        cost = weight if weight is not None else WEIGHTS.get(endpoint, 1)

        attempt = 0
        while True:
            await self.limiter.acquire(cost, is_order=is_order)

            if signed:
                query = self._prepare_signed(params)
            else:
                query = urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)

            url = f"{self.base_url}{endpoint}"
            if query:
                url = f"{url}?{query}"

            headers = {"X-MBX-APIKEY": self._api_key} if self._api_key else {}

            self.request_count += 1
            self.last_request_at = time.time()

            try:
                async with self._session.request(method, url, headers=headers) as response:
                    self.limiter.update_from_headers(dict(response.headers))
                    text = await response.text()

                    if response.status == 200:
                        return json.loads(text) if text else {}

                    await self._handle_error_status(
                        response.status, text, endpoint, dict(response.headers)
                    )

            except ClockSkewError as exc:
                # The clock drifted. Resync and retry exactly once — retrying
                # repeatedly against a broken clock just burns rate limit.
                if not retry_on_skew:
                    raise
                self.error_count += 1
                log.warning("clock_skew_detected", endpoint=endpoint, error=str(exc))
                await self.sync_time()
                retry_on_skew = False
                continue

            except RateLimitError as exc:
                self.error_count += 1
                self.last_error = str(exc)
                self.limiter.register_ban(exc.retry_after)
                if exc.banned or attempt >= self.max_retries:
                    raise
                attempt += 1
                await asyncio.sleep(exc.retry_after)
                continue

            except (TimeoutError, TimeoutError_, aiohttp.ServerTimeoutError) as exc:
                self.error_count += 1
                self.last_error = f"timeout: {exc}"
                # A timed-out ORDER is indeterminate: it may have been accepted.
                # Never silently retry it — the caller must reconcile by id.
                if is_order:
                    raise TimeoutError_(
                        "order request timed out; state is INDETERMINATE — "
                        "query by clientOrderId before any retry",
                        endpoint=endpoint,
                    ) from exc
                if attempt >= self.max_retries:
                    raise TimeoutError_(f"request timed out: {endpoint}") from exc
                attempt += 1
                await asyncio.sleep(self.retry_backoff_sec * (2**attempt))
                continue

            except aiohttp.ClientError as exc:
                self.error_count += 1
                self.last_error = f"network: {exc}"
                if is_order:
                    raise NetworkError(
                        "order request failed at the transport layer; state is "
                        "INDETERMINATE — query by clientOrderId before any retry",
                        endpoint=endpoint,
                        error=str(exc),
                    ) from exc
                if attempt >= self.max_retries:
                    raise NetworkError(f"network failure: {exc}", endpoint=endpoint) from exc
                attempt += 1
                await asyncio.sleep(self.retry_backoff_sec * (2**attempt))
                continue

            except ExchangeError as exc:
                self.error_count += 1
                self.last_error = str(exc)
                if getattr(exc, "retryable", False) and attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(self.retry_backoff_sec * (2**attempt))
                    continue
                raise

    async def _handle_error_status(
        self, status: int, text: str, endpoint: str, headers: dict[str, str]
    ) -> None:
        """Map a non-200 response onto the right exception. Always raises."""
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {}

        code = int(payload.get("code", 0) or 0)
        message = str(payload.get("msg", text[:200]))

        if status in (418, 429):
            retry_after = float(headers.get("Retry-After", 5) or 5)
            raise RateLimitError(
                f"HTTP {status} from {endpoint}: {message}",
                retry_after=retry_after,
                banned=(status == 418),
                endpoint=endpoint,
            )

        if status == 401 or status == 403:
            raise AuthenticationError(
                f"HTTP {status} from {endpoint}: {message}. Verify the API key, "
                f"its IP allow-list and futures permissions.",
                endpoint=endpoint,
            )

        if status >= 500:
            error = ExchangeError(f"HTTP {status} from {endpoint}: {message}", endpoint=endpoint)
            error.retryable = True
            raise error

        if code:
            berrors.raise_for_code(code, message, endpoint=endpoint, status=status)

        raise ExchangeError(f"HTTP {status} from {endpoint}: {message}", endpoint=endpoint)

    # ------------------------------------------------------------------ #
    # Public market data
    # ------------------------------------------------------------------ #
    async def ping(self) -> bool:
        await self._request("GET", "/fapi/v1/ping")
        return True

    async def server_time(self) -> int:
        data = await self._request("GET", "/fapi/v1/time")
        return int(data["serverTime"])

    async def sync_time(self) -> int:
        """Measure the local/server clock offset, compensating for latency.

        Binance rejects a request whose timestamp is outside ``recvWindow``. On a
        VPS with drifting time this shows up as intermittent -1021 errors that
        look like anything but a clock problem.
        """
        started = time.time()
        server_ms = await self.server_time()
        elapsed_ms = (time.time() - started) * 1000
        # Assume symmetric latency: the server's clock at the moment we received
        # the reply is server_ms + half the round trip.
        local_ms = int(time.time() * 1000)
        offset = int(server_ms + elapsed_ms / 2 - local_ms)
        self.clock.set_offset(offset)
        if abs(offset) > 1000:
            log.warning(
                "large_clock_offset",
                offset_ms=offset,
                hint="run NTP on this host; large drift causes -1021 errors",
            )
        return offset

    async def load_symbols(self) -> dict[str, SymbolInfo]:
        """Fetch ``exchangeInfo`` and build the symbol table.

        Leverage brackets are fetched separately and only when authenticated,
        because ``/fapi/v1/leverageBracket`` is a signed endpoint.
        """
        data = await self._request("GET", "/fapi/v1/exchangeInfo")

        if isinstance(data, dict) and data.get("rateLimits"):
            self.limiter.configure_from_exchange_info(data["rateLimits"])

        brackets: dict[str, list[dict]] = {}
        if self.authenticated:
            try:
                raw = await self._request("GET", "/fapi/v1/leverageBracket", signed=True)
                for entry in raw or []:
                    brackets[entry["symbol"]] = entry.get("brackets", [])
            except ExchangeError as exc:
                log.warning("leverage_brackets_unavailable", error=str(exc))

        symbols: dict[str, SymbolInfo] = {}
        for entry in data.get("symbols", []):
            try:
                info = parse_symbol_info(entry, brackets.get(entry["symbol"]))
            except (KeyError, ValueError, TypeError) as exc:
                log.warning("symbol_parse_failed", symbol=entry.get("symbol"), error=str(exc))
                continue
            symbols[info.symbol] = info

        self._symbols = symbols
        return symbols

    def symbol_info(self, symbol: str) -> SymbolInfo | None:
        return self._symbols.get(symbol)

    @property
    def symbols(self) -> dict[str, SymbolInfo]:
        return self._symbols

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Candle]:
        """Fetch candles. The most recent candle may still be forming."""
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        weight = 10 if limit > 500 else (5 if limit > 100 else 1)
        raw = await self._request("GET", "/fapi/v1/klines", params, weight=weight)

        now_ms = self.clock.now_ms()
        candles = []
        for row in raw:
            close_time = int(row[6])
            candles.append(
                Candle(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time=close_time,
                    quote_volume=float(row[7]),
                    trades=int(row[8]),
                    taker_buy_volume=float(row[9]),
                    # A bar whose close_time is in the future is still forming.
                    closed=close_time < now_ms,
                )
            )
        return candles

    async def get_book_ticker(self, symbol: str | None = None) -> dict[str, BookTicker]:
        params = {"symbol": symbol} if symbol else {}
        raw = await self._request(
            "GET", "/fapi/v1/ticker/bookTicker", params, weight=2 if symbol else 5
        )
        entries = raw if isinstance(raw, list) else [raw]
        out: dict[str, BookTicker] = {}
        for entry in entries:
            try:
                out[entry["symbol"]] = BookTicker(
                    symbol=entry["symbol"],
                    bid_price=float(entry["bidPrice"]),
                    bid_qty=float(entry["bidQty"]),
                    ask_price=float(entry["askPrice"]),
                    ask_qty=float(entry["askQty"]),
                    timestamp=int(entry.get("time", self.clock.now_ms())),
                )
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def get_ticker_24h(self, symbol: str | None = None) -> dict[str, Ticker24h]:
        params = {"symbol": symbol} if symbol else {}
        raw = await self._request("GET", "/fapi/v1/ticker/24hr", params, weight=1 if symbol else 40)
        entries = raw if isinstance(raw, list) else [raw]
        out: dict[str, Ticker24h] = {}
        for entry in entries:
            try:
                out[entry["symbol"]] = Ticker24h(
                    symbol=entry["symbol"],
                    last_price=float(entry["lastPrice"]),
                    price_change_pct=float(entry["priceChangePercent"]) / 100.0,
                    high=float(entry["highPrice"]),
                    low=float(entry["lowPrice"]),
                    volume=float(entry["volume"]),
                    quote_volume=float(entry["quoteVolume"]),
                    trades=int(entry.get("count", 0)),
                    timestamp=int(entry.get("closeTime", self.clock.now_ms())),
                )
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def get_mark_price(self, symbol: str | None = None) -> dict[str, MarkPriceInfo]:
        """Mark price, index price and the current funding rate."""
        params = {"symbol": symbol} if symbol else {}
        raw = await self._request(
            "GET", "/fapi/v1/premiumIndex", params, weight=1 if symbol else 10
        )
        entries = raw if isinstance(raw, list) else [raw]
        out: dict[str, MarkPriceInfo] = {}
        for entry in entries:
            try:
                out[entry["symbol"]] = MarkPriceInfo(
                    symbol=entry["symbol"],
                    mark_price=float(entry["markPrice"]),
                    index_price=float(entry.get("indexPrice", 0) or 0),
                    funding_rate=float(entry.get("lastFundingRate", 0) or 0),
                    next_funding_time=int(entry.get("nextFundingTime", 0) or 0),
                    timestamp=int(entry.get("time", self.clock.now_ms())),
                )
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def get_depth(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        weight = {5: 2, 10: 2, 20: 2, 50: 2, 100: 5, 500: 10, 1000: 20}.get(limit, 5)
        return await self._request(
            "GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit}, weight=weight
        )

    async def get_funding_history(self, symbol: str, limit: int = 100) -> list[dict]:
        return await self._request(
            "GET", "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}
        )

    # ------------------------------------------------------------------ #
    # Account (signed)
    # ------------------------------------------------------------------ #
    async def get_account(self) -> AccountState:
        data = await self._request("GET", "/fapi/v2/account", signed=True)
        equity = float(data.get("totalMarginBalance", 0) or 0)
        return AccountState(
            total_balance=float(data.get("totalWalletBalance", 0) or 0),
            available_balance=float(data.get("availableBalance", 0) or 0),
            equity=equity,
            unrealized_pnl=float(data.get("totalUnrealizedProfit", 0) or 0),
            margin_used=float(data.get("totalPositionInitialMargin", 0) or 0),
            timestamp=self.clock.now_ms(),
        )

    async def get_positions(self) -> dict[str, Position]:
        """Open positions as the exchange sees them. This is the source of truth.

        Binance returns a row for every symbol; only non-zero ``positionAmt``
        rows are real positions.
        """
        raw = await self._request("GET", "/fapi/v2/positionRisk", signed=True)
        out: dict[str, Position] = {}
        for entry in raw or []:
            try:
                amount = float(entry.get("positionAmt", 0) or 0)
            except (TypeError, ValueError):
                continue
            if amount == 0:
                continue
            symbol = entry["symbol"]
            direction = Direction.LONG if amount > 0 else Direction.SHORT
            entry_price = float(entry.get("entryPrice", 0) or 0)
            quantity = abs(amount)
            out[symbol] = Position(
                position_id=f"exchange:{symbol}",
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                entry_price=entry_price,
                leverage=int(float(entry.get("leverage", 1) or 1)),
                # Protective levels are unknown from this endpoint; the
                # reconciler fills them in or attaches a stop immediately.
                stop_loss=0.0,
                take_profit=0.0,
                strategy="unknown",
                regime=__import__(
                    "tradebot.core.types", fromlist=["MarketRegime"]
                ).MarketRegime.SIDEWAYS,
                opened_at=int(entry.get("updateTime", 0) or 0),
                entry_notional=quantity * entry_price,
                metadata={
                    "liquidation_price": float(entry.get("liquidationPrice", 0) or 0),
                    "mark_price": float(entry.get("markPrice", 0) or 0),
                    "unrealized_pnl": float(entry.get("unRealizedProfit", 0) or 0),
                    "margin_type": entry.get("marginType", ""),
                    "source": "exchange",
                },
            )
        return out

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        params = {"symbol": symbol} if symbol else {}
        raw = await self._request(
            "GET", "/fapi/v1/openOrders", params, signed=True, weight=1 if symbol else 40
        )
        return [self._parse_order(entry) for entry in raw or []]

    async def get_user_trades(
        self, symbol: str, limit: int = 100, start_ms: int | None = None
    ) -> list[dict]:
        """Executed fills, including the commission actually charged."""
        params: dict[str, Any] = {"symbol": symbol, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        return await self._request("GET", "/fapi/v1/userTrades", params, signed=True)

    async def get_income(
        self,
        symbol: str | None = None,
        income_type: str | None = None,
        limit: int = 100,
        start_ms: int | None = None,
    ) -> list[dict]:
        """Funding, realised PnL and commission history."""
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if income_type:
            params["incomeType"] = income_type
        if start_ms is not None:
            params["startTime"] = start_ms
        return await self._request("GET", "/fapi/v1/income", params, signed=True)

    # ------------------------------------------------------------------ #
    # Trading (signed)
    # ------------------------------------------------------------------ #
    async def set_leverage(self, symbol: str, leverage: int) -> int:
        """Set leverage, returning the value the exchange actually applied.

        The exchange may cap leverage by notional bracket, so the returned value
        must be used rather than the requested one — sizing computed against a
        leverage the exchange refused would be wrong.
        """
        try:
            data = await self._request(
                "POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True
            )
            return int(data.get("leverage", leverage))
        except ExchangeError as exc:
            code = exc.context.get("code")
            if code in berrors.NO_NEED_TO_CHANGE_LEVERAGE:
                return leverage
            raise

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> bool:
        """Isolated margin caps the loss on one position at its own margin."""
        try:
            await self._request(
                "POST",
                "/fapi/v1/marginType",
                {"symbol": symbol, "marginType": margin_type},
                signed=True,
            )
            return True
        except ExchangeError as exc:
            # -4046: no need to change margin type. Already correct.
            if exc.context.get("code") in berrors.NO_NEED_TO_CHANGE_LEVERAGE:
                return True
            log.warning("set_margin_type_failed", symbol=symbol, error=str(exc))
            return False

    async def place_order(self, intent: OrderIntent) -> Order:
        """Submit an entry order.

        On an indeterminate failure (timeout or transport error) this queries the
        order by its deterministic client id and returns whatever the exchange
        actually has, rather than re-sending.
        """
        info = self._symbols.get(intent.symbol)
        if info is None:
            raise ExchangeError("unknown symbol", symbol=intent.symbol)

        reference = intent.price or intent.metadata.get("reference_price")
        validation = validate_order(
            info, intent.quantity, intent.price, intent.order_type, reference_price=reference
        )
        if not validation.ok:
            from tradebot.core.errors import FilterViolationError

            raise FilterViolationError(
                f"order fails local filter validation: {validation.reason}",
                detail=validation.detail,
                symbol=intent.symbol,
            )

        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": intent.side.value,
            "type": intent.order_type.value,
            "newClientOrderId": intent.client_order_id,
            "newOrderRespType": "RESULT",
        }
        params.update(format_order_params(info, validation.quantity, validation.price))
        if intent.order_type is OrderType.LIMIT:
            params["timeInForce"] = TimeInForce.GTC.value
        if intent.reduce_only:
            params["reduceOnly"] = "true"

        try:
            data = await self._request("POST", "/fapi/v1/order", params, signed=True)
            return self._parse_order(data, intent_id=intent.intent_id)
        except (TimeoutError_, NetworkError) as exc:
            log.error(
                "order_submission_indeterminate",
                symbol=intent.symbol,
                client_order_id=intent.client_order_id,
                error=str(exc),
            )
            existing = await self._resolve_indeterminate(intent.symbol, intent.client_order_id)
            if existing is not None:
                log.warning(
                    "order_recovered_after_indeterminate_submit",
                    symbol=intent.symbol,
                    status=existing.status.value,
                )
                existing.intent_id = intent.intent_id
                return existing
            # Genuinely never reached the exchange.
            order = Order(
                client_order_id=intent.client_order_id,
                symbol=intent.symbol,
                side=intent.side,
                order_type=intent.order_type,
                quantity=validation.quantity,
                price=validation.price,
                status=OrderStatus.UNKNOWN,
                intent_id=intent.intent_id,
                error=str(exc),
            )
            return order

    async def _resolve_indeterminate(self, symbol: str, client_order_id: str) -> Order | None:
        """After a timeout, ask the exchange what actually happened."""
        for attempt in range(3):
            await asyncio.sleep(0.4 * (attempt + 1))
            try:
                found = await self.query_order(symbol, client_order_id)
            except UnknownOrderError:
                return None
            except ExchangeError as exc:
                log.warning(
                    "indeterminate_resolution_failed",
                    symbol=symbol,
                    attempt=attempt,
                    error=str(exc),
                )
                continue
            if found is not None:
                return found
        return None

    async def place_protective_order(
        self,
        symbol: str,
        order_type: OrderType,
        stop_price: float,
        quantity: float,
        direction_sign: int,
        client_order_id: str,
    ) -> Order:
        """Place a reduce-only stop or take-profit for an open position.

        ``closePosition=true`` is used so the protective order always covers the
        whole position even if the position size changes underneath it — a
        partially-covered stop is worse than no stop, because it looks safe.
        """
        info = self._symbols.get(symbol)
        if info is None:
            raise ExchangeError("unknown symbol", symbol=symbol)

        from tradebot.core.mathutil import format_decimal, round_price

        rounded_stop = round_price(stop_price, info.tick_size)
        if rounded_stop <= 0:
            from tradebot.core.errors import FilterViolationError

            raise FilterViolationError("stop price rounds to zero", symbol=symbol)

        side = OrderSide.SELL if direction_sign > 0 else OrderSide.BUY
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "stopPrice": format_decimal(rounded_stop, info.price_precision),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
            "priceProtect": "true",
        }
        _ = quantity  # closePosition covers the whole position by design
        data = await self._request("POST", "/fapi/v1/order", params, signed=True)
        return self._parse_order(data)

    async def query_order(self, symbol: str, client_order_id: str) -> Order | None:
        try:
            data = await self._request(
                "GET",
                "/fapi/v1/order",
                {"symbol": symbol, "origClientOrderId": client_order_id},
                signed=True,
            )
        except UnknownOrderError:
            return None
        return self._parse_order(data)

    async def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        try:
            await self._request(
                "DELETE",
                "/fapi/v1/order",
                {"symbol": symbol, "origClientOrderId": client_order_id},
                signed=True,
            )
            return True
        except UnknownOrderError:
            return True  # already gone: the desired end state
        except ExchangeError as exc:
            log.warning(
                "cancel_order_failed",
                symbol=symbol,
                client_order_id=client_order_id,
                error=str(exc),
            )
            return False

    async def cancel_all_orders(self, symbol: str) -> bool:
        try:
            await self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
            return True
        except ExchangeError as exc:
            log.warning("cancel_all_failed", symbol=symbol, error=str(exc))
            return False

    async def close_position(self, symbol: str, position: Position, client_order_id: str) -> Order:
        """Flatten a position with a reduce-only market order."""
        info = self._symbols.get(symbol)
        if info is None:
            raise ExchangeError("unknown symbol", symbol=symbol)

        from tradebot.core.mathutil import format_decimal, round_quantity

        quantity = round_quantity(position.quantity, info.step_size)
        if quantity <= 0:
            raise ExchangeError("position quantity rounds to zero", symbol=symbol)

        params = {
            "symbol": symbol,
            "side": OrderSide.for_exit(position.direction).value,
            "type": OrderType.MARKET.value,
            "quantity": format_decimal(quantity, info.quantity_precision),
            "reduceOnly": "true",
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        data = await self._request("POST", "/fapi/v1/order", params, signed=True)
        return self._parse_order(data)

    # ------------------------------------------------------------------ #
    # User data stream
    # ------------------------------------------------------------------ #
    async def create_listen_key(self) -> str:
        data = await self._request("POST", "/fapi/v1/listenKey", signed=False)
        self._listen_key = str(data["listenKey"])
        return self._listen_key

    async def keepalive_listen_key(self) -> bool:
        """Must be called at least every 60 minutes or the stream dies silently."""
        try:
            await self._request("PUT", "/fapi/v1/listenKey", signed=False)
            return True
        except ExchangeError as exc:
            log.warning("listen_key_keepalive_failed", error=str(exc))
            return False

    async def close_listen_key(self) -> None:
        with contextlib.suppress(ExchangeError):
            await self._request("DELETE", "/fapi/v1/listenKey", signed=False)
        self._listen_key = None

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_order(data: dict, intent_id: str | None = None) -> Order:
        """Build an Order from a REST order payload."""
        filled = float(data.get("executedQty", 0) or 0)
        avg = float(data.get("avgPrice", 0) or 0)
        try:
            status = OrderStatus(data.get("status", "NEW"))
        except ValueError:
            status = OrderStatus.UNKNOWN
        try:
            order_type = OrderType(data.get("type", "MARKET"))
        except ValueError:
            order_type = OrderType.MARKET

        price_raw = float(data.get("price", 0) or 0)
        stop_raw = float(data.get("stopPrice", 0) or 0)

        return Order(
            client_order_id=str(data.get("clientOrderId", "")),
            symbol=str(data.get("symbol", "")),
            side=OrderSide(data.get("side", "BUY")),
            order_type=order_type,
            quantity=float(data.get("origQty", 0) or 0),
            price=price_raw or None,
            stop_price=stop_raw or None,
            status=status,
            exchange_order_id=str(data.get("orderId", "")) or None,
            filled_quantity=filled,
            average_price=avg,
            reduce_only=bool(data.get("reduceOnly", False)),
            close_position=bool(data.get("closePosition", False)),
            updated_at=int(data.get("updateTime", 0) or 0),
            intent_id=intent_id,
        )

    def stats(self) -> dict[str, Any]:
        """Health/diagnostic counters."""
        return {
            "requests": self.request_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "last_error": self.last_error,
            "clock_offset_ms": self.clock.offset_ms,
            "symbols_loaded": len(self._symbols),
            **self.limiter.usage(),
        }
