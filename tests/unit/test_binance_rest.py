"""Binance REST client, driven by a fake HTTP session.

No network access is used or required. The fake session lets us assert on the
exact URL and headers produced, and to inject the failure modes that matter:
timeouts on order submission, rate limits, clock skew and error codes.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest

from tradebot.core.errors import (
    AuthenticationError,
    ClockSkewError,
    FilterViolationError,
    InsufficientMarginError,
    RateLimitError,
    TimeoutError_,
)
from tradebot.core.types import (
    Direction,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
)
from tradebot.exchange.binance.rest import BinanceFuturesREST

# The official docs' worked example key. Independently verified with
#   echo -n "<query>" | openssl dgst -sha256 -hmac "<secret>"
DOC_SECRET = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
DOC_QUERY = (
    "symbol=BTCUSDT&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=9000"
    "&recvWindow=5000&timestamp=1591702613943"
)
DOC_SIGNATURE = "8ad7e17436d5b04e07c1553b45d56972c46a6bed803c22d8558080e3fb16782c"

EXCHANGE_INFO = {
    "rateLimits": [
        {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
        {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
    ],
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "pricePrecision": 2,
            "quantityPrecision": 3,
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {
                    "filterType": "LOT_SIZE",
                    "stepSize": "0.001",
                    "minQty": "0.001",
                    "maxQty": "1000",
                },
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "stepSize": "0.001",
                    "minQty": "0.001",
                    "maxQty": "120",
                },
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
                {"filterType": "PERCENT_PRICE", "multiplierUp": "1.05", "multiplierDown": "0.95"},
            ],
        },
        {
            "symbol": "OLDCOIN",
            "baseAsset": "OLD",
            "quoteAsset": "USDT",
            "status": "BREAK",
            "contractType": "PERPETUAL",
            "pricePrecision": 2,
            "quantityPrecision": 2,
            "filters": [],
        },
    ],
}


class FakeResponse:
    def __init__(self, status: int, body, headers: dict | None = None):
        self.status = status
        self._body = body if isinstance(body, str) else json.dumps(body)
        self.headers = headers or {}

    async def text(self) -> str:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, routes: dict | None = None):
        self.routes = routes or {}
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def request(self, method: str, url: str, headers=None, **_kwargs):
        self.calls.append((method, url))
        self.last_headers = headers or {}
        path = urlparse(url).path
        entry = self.routes.get(path, {"status": 200, "body": {}})
        if callable(entry):
            entry = entry(method, url)
        if isinstance(entry, list):
            entry = entry.pop(0) if len(entry) > 1 else entry[0]
        if isinstance(entry, BaseException):
            raise entry
        return FakeResponse(entry.get("status", 200), entry.get("body", {}), entry.get("headers"))

    async def close(self):
        self.closed = True


def make_client(routes: dict | None = None, **kwargs) -> BinanceFuturesREST:
    base = {
        "/fapi/v1/time": {"body": {"serverTime": 1_700_000_000_000}},
        "/fapi/v1/exchangeInfo": {"body": EXCHANGE_INFO},
    }
    base.update(routes or {})
    session = FakeSession(base)
    client = BinanceFuturesREST(
        api_key="testkey",
        api_secret=DOC_SECRET,
        base_url="https://testnet.binancefuture.com",
        session=session,
        max_retries=1,
        retry_backoff_sec=0.001,
        **kwargs,
    )
    client._session = session
    return client


def make_intent(**overrides) -> OrderIntent:
    params = {
        "intent_id": "abc123",
        "symbol": "BTCUSDT",
        "direction": Direction.LONG,
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": 0.01,
        "price": None,
        "stop_loss": 59000.0,
        "take_profit": 61000.0,
        "leverage": 3,
        "notional": 600.0,
        "risk_amount": 0.375,
        "strategy": "momentum",
        "regime": MarketRegime.STRONG_TREND,
        "opportunity_score": 82.0,
        "expected_net_edge": 0.0015,
        "metadata": {"reference_price": 60000.0},
    }
    params.update(overrides)
    return OrderIntent(**params)


class TestSigning:
    def test_hmac_matches_openssl(self):
        """Verified independently: openssl dgst -sha256 -hmac <secret>."""
        client = BinanceFuturesREST(api_key="k", api_secret=DOC_SECRET)
        assert client._sign(DOC_QUERY) == DOC_SIGNATURE

    def test_signed_query_appends_timestamp_recvwindow_and_signature(self):
        client = BinanceFuturesREST(api_key="k", api_secret=DOC_SECRET, recv_window=5000)
        query = client._prepare_signed({"symbol": "BTCUSDT"})
        parsed = parse_qs(query)
        assert parsed["symbol"] == ["BTCUSDT"]
        assert "timestamp" in parsed
        assert parsed["recvWindow"] == ["5000"]
        assert len(parsed["signature"][0]) == 64

    def test_signature_covers_every_parameter(self):
        """Changing any parameter must change the signature."""
        client = BinanceFuturesREST(api_key="k", api_secret=DOC_SECRET)
        base = client._sign("symbol=BTCUSDT&quantity=1")
        assert base != client._sign("symbol=BTCUSDT&quantity=2")

    def test_none_valued_parameters_are_dropped(self):
        client = BinanceFuturesREST(api_key="k", api_secret=DOC_SECRET)
        query = client._prepare_signed({"symbol": "BTCUSDT", "price": None})
        assert "price" not in parse_qs(query)

    def test_signing_without_credentials_raises(self):
        client = BinanceFuturesREST()
        with pytest.raises(AuthenticationError):
            client._prepare_signed({"symbol": "BTCUSDT"})


class TestReferenceData:
    async def test_load_symbols_parses_filters(self):
        client = make_client()
        symbols = await client.load_symbols()
        btc = symbols["BTCUSDT"]
        assert btc.tick_size == 0.10
        assert btc.step_size == 0.001
        assert btc.min_notional == 5.0
        assert btc.market_max_qty == 120.0
        assert btc.is_tradable

    async def test_non_trading_symbol_is_not_tradable(self):
        client = make_client()
        symbols = await client.load_symbols()
        assert not symbols["OLDCOIN"].is_tradable

    async def test_rate_limits_are_adopted_from_exchange_info(self):
        client = make_client()
        await client.load_symbols()
        assert client.limiter.usage()["weight_limit"] == 2400

    async def test_klines_mark_the_forming_bar_as_open(self):
        """The newest bar's close_time is in the future while it is still forming."""
        now = 1_700_000_000_000
        rows = [
            [now - 120_000, "1", "2", "0.5", "1.5", "10", now - 60_001, "15", 5, "5", "7", "0"],
            [now - 60_000, "1.5", "2.5", "1", "2", "12", now + 59_999, "24", 6, "6", "9", "0"],
        ]
        client = make_client({"/fapi/v1/klines": {"body": rows}})
        client.clock.set_offset(now - int(__import__("time").time() * 1000))
        candles = await client.get_klines("BTCUSDT", "1m")
        assert candles[0].closed
        assert not candles[1].closed

    async def test_book_ticker_parses_a_list(self):
        body = [
            {
                "symbol": "BTCUSDT",
                "bidPrice": "99.9",
                "bidQty": "1",
                "askPrice": "100.1",
                "askQty": "2",
                "time": 1,
            }
        ]
        client = make_client({"/fapi/v1/ticker/bookTicker": {"body": body}})
        books = await client.get_book_ticker()
        assert books["BTCUSDT"].spread_bps == pytest.approx(20.0)

    async def test_malformed_entries_are_skipped_not_fatal(self):
        body = [
            {
                "symbol": "GOOD",
                "bidPrice": "1",
                "bidQty": "1",
                "askPrice": "2",
                "askQty": "1",
                "time": 1,
            },
            {"symbol": "BAD"},
        ]
        client = make_client({"/fapi/v1/ticker/bookTicker": {"body": body}})
        books = await client.get_book_ticker()
        assert set(books) == {"GOOD"}


class TestClockSync:
    async def test_offset_is_measured_from_server_time(self):
        import time as _time

        server = int(_time.time() * 1000) + 5000
        client = make_client({"/fapi/v1/time": {"body": {"serverTime": server}}})
        offset = await client.sync_time()
        assert 4000 < offset < 6000

    async def test_clock_skew_error_triggers_one_resync_then_retries(self):
        """-1021 means our clock drifted; resync and retry, do not loop."""
        attempts = {"n": 0}

        def account_route(_method, _url):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {
                    "status": 400,
                    "body": {"code": -1021, "msg": "Timestamp outside recvWindow"},
                }
            return {
                "status": 200,
                "body": {
                    "totalWalletBalance": "75",
                    "availableBalance": "75",
                    "totalMarginBalance": "75",
                    "totalUnrealizedProfit": "0",
                    "totalPositionInitialMargin": "0",
                },
            }

        client = make_client({"/fapi/v2/account": account_route})
        account = await client.get_account()
        assert attempts["n"] == 2
        assert account.equity == 75.0

    async def test_repeated_skew_eventually_raises(self):
        client = make_client(
            {"/fapi/v2/account": {"status": 400, "body": {"code": -1021, "msg": "skew"}}}
        )
        with pytest.raises(ClockSkewError):
            await client.get_account()


class TestErrorMapping:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (-2019, InsufficientMarginError),
            (-1111, FilterViolationError),
            (-4164, FilterViolationError),
            (-2015, AuthenticationError),
        ],
    )
    async def test_error_codes_map_to_typed_exceptions(self, code, expected):
        client = make_client(
            {"/fapi/v2/account": {"status": 400, "body": {"code": code, "msg": "x"}}}
        )
        with pytest.raises(expected):
            await client.get_account()

    async def test_http_429_raises_rate_limit_with_retry_after(self):
        client = make_client(
            {
                "/fapi/v2/account": {
                    "status": 429,
                    "body": {"code": -1003, "msg": "too many"},
                    "headers": {"Retry-After": "7"},
                }
            }
        )
        with pytest.raises(RateLimitError) as exc:
            await client.get_account()
        assert exc.value.retry_after == 7.0

    async def test_http_418_marks_the_client_banned(self):
        client = make_client(
            {
                "/fapi/v2/account": {
                    "status": 418,
                    "body": {"msg": "banned"},
                    "headers": {"Retry-After": "2"},
                }
            }
        )
        with pytest.raises(RateLimitError) as exc:
            await client.get_account()
        assert exc.value.banned
        assert client.limiter.is_banned

    async def test_error_counter_increments(self):
        client = make_client(
            {"/fapi/v2/account": {"status": 400, "body": {"code": -2019, "msg": "x"}}}
        )
        with pytest.raises(InsufficientMarginError):
            await client.get_account()
        assert client.error_count >= 1


class TestOrderSubmission:
    async def test_successful_order_is_parsed(self):
        body = {
            "clientOrderId": "tb_abc123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.010",
            "executedQty": "0.010",
            "avgPrice": "60010.0",
            "status": "FILLED",
            "orderId": 999,
            "updateTime": 1,
        }
        client = make_client({"/fapi/v1/order": {"body": body}})
        await client.load_symbols()
        order = await client.place_order(make_intent())
        assert order.status is OrderStatus.FILLED
        assert order.filled_quantity == pytest.approx(0.01)
        assert order.average_price == pytest.approx(60010.0)

    async def test_client_order_id_is_deterministic_from_the_intent(self):
        """This is what makes a retry collide instead of double-filling."""
        assert make_intent(intent_id="xyz").client_order_id == "tb_xyz"
        assert (
            make_intent(intent_id="xyz").client_order_id
            == make_intent(intent_id="xyz").client_order_id
        )

    async def test_order_below_min_notional_is_rejected_locally(self):
        """Caught before transmission, so it never consumes an order-rate slot."""
        client = make_client()
        await client.load_symbols()
        with pytest.raises(FilterViolationError):
            await client.place_order(
                make_intent(quantity=0.00001, metadata={"reference_price": 60000.0})
            )

    async def test_quantity_is_rounded_to_step_before_sending(self):
        captured = {}

        def order_route(_method, url):
            captured["url"] = url
            return {
                "body": {
                    "clientOrderId": "tb_abc123",
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "origQty": "0.010",
                    "executedQty": "0.010",
                    "avgPrice": "60000",
                    "status": "FILLED",
                    "orderId": 1,
                    "updateTime": 1,
                }
            }

        client = make_client({"/fapi/v1/order": order_route})
        await client.load_symbols()
        await client.place_order(make_intent(quantity=0.0109999))
        assert parse_qs(urlparse(captured["url"]).query)["quantity"] == ["0.01"]

    async def test_api_key_header_is_sent(self):
        client = make_client(
            {
                "/fapi/v2/account": {
                    "body": {
                        "totalWalletBalance": "1",
                        "availableBalance": "1",
                        "totalMarginBalance": "1",
                        "totalUnrealizedProfit": "0",
                        "totalPositionInitialMargin": "0",
                    }
                }
            }
        )
        await client.get_account()
        assert client._session.last_headers.get("X-MBX-APIKEY") == "testkey"


class TestIndeterminateSubmission:
    """The single most dangerous failure: a timed-out order submission."""

    async def test_timeout_queries_by_client_id_instead_of_resending(self):
        calls = {"post": 0, "get": 0}
        filled = {
            "clientOrderId": "tb_abc123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.010",
            "executedQty": "0.010",
            "avgPrice": "60000",
            "status": "FILLED",
            "orderId": 5,
            "updateTime": 1,
        }

        def order_route(method, _url):
            if method == "POST":
                calls["post"] += 1
                raise TimeoutError("simulated")
            calls["get"] += 1
            return {"body": filled}

        client = make_client({"/fapi/v1/order": order_route})
        await client.load_symbols()
        order = await client.place_order(make_intent())

        assert calls["post"] == 1, "a timed-out order must never be re-sent blindly"
        assert calls["get"] >= 1, "it must be resolved by querying the client id"
        assert order.status is OrderStatus.FILLED

    async def test_order_that_never_reached_the_exchange_is_reported_unknown(self):
        from tradebot.core.errors import UnknownOrderError

        def order_route(method, _url):
            if method == "POST":
                raise TimeoutError("simulated")
            return {"status": 400, "body": {"code": -2013, "msg": "Order does not exist"}}

        client = make_client({"/fapi/v1/order": order_route})
        await client.load_symbols()
        order = await client.place_order(make_intent())
        assert order.status is OrderStatus.UNKNOWN
        _ = UnknownOrderError

    async def test_network_error_on_order_is_also_treated_as_indeterminate(self):
        posts = {"n": 0}

        def order_route(method, _url):
            if method == "POST":
                posts["n"] += 1
                raise aiohttp.ClientError("connection reset")
            return {"status": 400, "body": {"code": -2013, "msg": "no such order"}}

        client = make_client({"/fapi/v1/order": order_route})
        await client.load_symbols()
        order = await client.place_order(make_intent())
        assert posts["n"] == 1
        assert order.status is OrderStatus.UNKNOWN

    async def test_non_order_requests_may_retry_freely(self):
        """Reads are idempotent, so retrying them is safe and desirable."""
        attempts = {"n": 0}

        def route(_method, _url):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise TimeoutError("transient")
            return {
                "body": [
                    {
                        "symbol": "BTCUSDT",
                        "bidPrice": "1",
                        "bidQty": "1",
                        "askPrice": "2",
                        "askQty": "1",
                        "time": 1,
                    }
                ]
            }

        client = make_client({"/fapi/v1/ticker/bookTicker": route})
        books = await client.get_book_ticker()
        assert attempts["n"] == 2
        assert "BTCUSDT" in books

    async def test_timeout_on_a_read_eventually_raises(self):
        def route(_method, _url):
            raise TimeoutError("always")

        client = make_client({"/fapi/v1/ticker/bookTicker": route})
        with pytest.raises(TimeoutError_):
            await client.get_book_ticker()


class TestPositionParsing:
    async def test_zero_quantity_rows_are_not_positions(self):
        body = [
            {"symbol": "BTCUSDT", "positionAmt": "0", "entryPrice": "0", "leverage": "5"},
            {
                "symbol": "ETHUSDT",
                "positionAmt": "-1.5",
                "entryPrice": "3000",
                "leverage": "3",
                "liquidationPrice": "3500",
                "markPrice": "2990",
                "unRealizedProfit": "15",
                "updateTime": 1,
            },
        ]
        client = make_client({"/fapi/v2/positionRisk": {"body": body}})
        positions = await client.get_positions()
        assert set(positions) == {"ETHUSDT"}

    async def test_negative_amount_is_a_short_with_positive_quantity(self):
        body = [
            {
                "symbol": "ETHUSDT",
                "positionAmt": "-1.5",
                "entryPrice": "3000",
                "leverage": "3",
                "updateTime": 1,
            }
        ]
        client = make_client({"/fapi/v2/positionRisk": {"body": body}})
        position = (await client.get_positions())["ETHUSDT"]
        assert position.direction is Direction.SHORT
        assert position.quantity == pytest.approx(1.5)
        assert position.unrealized_pnl(2900.0) == pytest.approx(150.0)

    async def test_exchange_positions_have_no_assumed_stop(self):
        """An adopted position must be flagged as unprotected, not given a fake stop."""
        body = [
            {
                "symbol": "ETHUSDT",
                "positionAmt": "1",
                "entryPrice": "3000",
                "leverage": "3",
                "updateTime": 1,
            }
        ]
        client = make_client({"/fapi/v2/positionRisk": {"body": body}})
        position = (await client.get_positions())["ETHUSDT"]
        assert position.stop_loss == 0.0
