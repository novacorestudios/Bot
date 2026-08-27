"""Rate limiter.

Exceeding a Binance budget earns a 429 and then an IP ban, which takes the bot
offline with positions open — the worst possible time. These tests pin the
shaping behaviour and the server-header override.
"""

from __future__ import annotations

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.exchange.binance.ratelimit import RateLimiter

# The limiter reads time and sleeps through the same injected clock, so a
# virtual clock makes throttling deterministic and instant to test: an acquire
# that "waits 61 seconds" returns immediately but records the wait.


class TestBudgets:
    async def test_requests_within_budget_do_not_wait(self):
        limiter = RateLimiter(weight_limit=100, safety_factor=1.0, clock=VirtualClock())
        for _ in range(50):
            await limiter.acquire(1)
        assert limiter.waits == 0

    async def test_safety_factor_reserves_headroom(self):
        """Reserving below the true limit leaves room for retries and other clients."""
        clock = VirtualClock()
        limiter = RateLimiter(weight_limit=100, safety_factor=0.8, clock=clock)
        for _ in range(80):
            await limiter.acquire(1)
        assert limiter.usage()["weight_used"] == 80

        # The 81st exceeds 80% of 100 and must be throttled until the window slides.
        await limiter.acquire(1)
        assert limiter.waits == 1
        assert limiter.total_wait_sec > 0

    async def test_weight_is_charged_per_request_not_per_call(self):
        limiter = RateLimiter(weight_limit=100, safety_factor=1.0, clock=VirtualClock())
        await limiter.acquire(40)
        assert limiter.usage()["weight_used"] == 40

    async def test_window_slides_so_old_weight_expires(self):
        clock = VirtualClock()
        limiter = RateLimiter(weight_limit=10, safety_factor=1.0, clock=clock)
        for _ in range(10):
            await limiter.acquire(1)
        clock.advance(61)
        await limiter.acquire(1)
        assert limiter.waits == 0

    async def test_order_budget_is_separate_from_weight_budget(self):
        clock = VirtualClock()
        limiter = RateLimiter(
            weight_limit=10_000,
            order_limit_10s=5,
            order_limit_1m=1000,
            safety_factor=1.0,
            clock=clock,
        )
        for _ in range(5):
            await limiter.acquire(1, is_order=True)

        # The 6th order inside the 10s window must be throttled by roughly the
        # remaining window, not by the (far larger) weight window.
        await limiter.acquire(1, is_order=True)
        assert limiter.waits == 1
        assert 0 < limiter.total_wait_sec <= 10.0

    async def test_non_order_requests_are_unaffected_by_the_order_budget(self):
        clock = VirtualClock()
        limiter = RateLimiter(
            weight_limit=10_000, order_limit_10s=1, safety_factor=1.0, clock=clock
        )
        await limiter.acquire(1, is_order=True)
        await limiter.acquire(1, is_order=False)
        assert limiter.waits == 0


class TestServerTruth:
    async def test_server_headers_override_a_lower_local_count(self):
        """Other processes on the same IP consume budget we cannot see locally."""
        limiter = RateLimiter(weight_limit=1000, safety_factor=1.0, clock=VirtualClock())
        await limiter.acquire(1)
        limiter.update_from_headers({"X-MBX-USED-WEIGHT-1M": "900"})
        assert limiter.usage()["weight_used"] == 900

    async def test_local_count_wins_when_it_is_higher(self):
        limiter = RateLimiter(weight_limit=1000, safety_factor=1.0, clock=VirtualClock())
        await limiter.acquire(500)
        limiter.update_from_headers({"X-MBX-USED-WEIGHT-1M": "10"})
        assert limiter.usage()["weight_used"] == 500

    async def test_unparseable_headers_are_ignored(self):
        limiter = RateLimiter(clock=VirtualClock())
        limiter.update_from_headers({"X-MBX-USED-WEIGHT-1M": "not-a-number"})
        assert limiter.usage()["weight_used"] == 0

    async def test_order_count_headers_are_adopted(self):
        limiter = RateLimiter(clock=VirtualClock())
        limiter.update_from_headers({"X-MBX-ORDER-COUNT-1M": "77"})
        assert limiter.usage()["orders_1m"] == 77


class TestBan:
    async def test_ban_blocks_every_request_until_it_expires(self):
        clock = VirtualClock()
        limiter = RateLimiter(clock=clock)
        limiter.register_ban(30.0)
        assert limiter.is_banned

        await limiter.acquire(1)
        assert limiter.waits == 1
        assert limiter.total_wait_sec >= 30.0
        assert not limiter.is_banned, "the ban must lift once its delay has elapsed"


class TestConfiguration:
    def test_limits_are_adopted_from_exchange_info(self):
        limiter = RateLimiter()
        limiter.configure_from_exchange_info(
            [
                {
                    "rateLimitType": "REQUEST_WEIGHT",
                    "interval": "MINUTE",
                    "intervalNum": 1,
                    "limit": 6000,
                },
                {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 50},
                {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 500},
            ]
        )
        usage = limiter.usage()
        assert usage["weight_limit"] == 6000

    def test_zero_or_missing_limits_are_ignored(self):
        limiter = RateLimiter(weight_limit=2400)
        limiter.configure_from_exchange_info(
            [{"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "limit": 0}]
        )
        assert limiter.usage()["weight_limit"] == 2400


class TestFilters:
    """Symbol-filter validation — the local gate before any order is sent."""

    def test_min_notional_blocks_a_too_small_order(self, symbol_info):
        from tradebot.core.types import OrderType
        from tradebot.exchange.binance.filters import validate_order

        result = validate_order(
            symbol_info, quantity=0.001, price=100.0, order_type=OrderType.LIMIT
        )
        assert not result.ok
        assert result.reason == "BELOW_MIN_NOTIONAL"

    def test_valid_order_returns_adjusted_values(self, symbol_info):
        from tradebot.core.types import OrderType
        from tradebot.exchange.binance.filters import validate_order

        result = validate_order(
            symbol_info, quantity=0.1234567, price=100.123456, order_type=OrderType.LIMIT
        )
        assert result.ok
        assert result.quantity == pytest.approx(0.123)  # rounded DOWN to step
        assert result.price == pytest.approx(100.12)  # rounded to tick

    def test_quantity_that_rounds_to_zero_is_rejected(self, symbol_info):
        from tradebot.core.types import OrderType
        from tradebot.exchange.binance.filters import validate_order

        result = validate_order(
            symbol_info, quantity=0.0001, price=100.0, order_type=OrderType.LIMIT
        )
        assert not result.ok
        assert result.reason in {"QUANTITY_ROUNDS_TO_ZERO", "BELOW_MIN_QTY"}

    def test_market_orders_use_the_market_lot_size_cap(self):
        from tradebot.core.types import OrderType, SymbolInfo
        from tradebot.exchange.binance.filters import validate_order

        info = SymbolInfo(
            symbol="X",
            base_asset="X",
            quote_asset="USDT",
            status="TRADING",
            contract_type="PERPETUAL",
            price_precision=2,
            quantity_precision=3,
            tick_size=0.01,
            step_size=0.001,
            min_qty=0.001,
            max_qty=1000.0,
            min_notional=5.0,
            market_min_qty=0.001,
            market_max_qty=10.0,
        )
        limit_ok = validate_order(info, 50.0, 100.0, OrderType.LIMIT)
        market_bad = validate_order(info, 50.0, None, OrderType.MARKET, reference_price=100.0)
        assert limit_ok.ok
        assert not market_bad.ok
        assert market_bad.reason == "ABOVE_MAX_QTY"

    def test_percent_price_band_is_enforced(self, symbol_info):
        from dataclasses import replace

        from tradebot.core.types import OrderType
        from tradebot.exchange.binance.filters import validate_order

        info = replace(symbol_info, multiplier_up=1.05, multiplier_down=0.95)
        result = validate_order(
            info, 1.0, price=200.0, order_type=OrderType.LIMIT, reference_price=100.0
        )
        assert not result.ok
        assert result.reason == "PERCENT_PRICE"

    def test_min_quantity_for_notional_rounds_up_to_satisfy_the_filter(self, symbol_info):
        from tradebot.exchange.binance.filters import min_quantity_for_notional

        # min_notional 5.0 at price 3000 -> 0.001667 -> must round UP to 0.002
        qty = min_quantity_for_notional(symbol_info, 3000.0)
        assert qty * 3000.0 >= symbol_info.min_notional
        assert qty == pytest.approx(0.002)

    def test_negative_quantity_is_rejected(self, symbol_info):
        from tradebot.core.types import OrderType
        from tradebot.exchange.binance.filters import validate_order

        assert not validate_order(symbol_info, -1.0, 100.0, OrderType.LIMIT).ok
