"""Paper broker.

The broker must be PESSIMISTIC. A simulator that fills at the mid price,
instantly, in full, produces results better than live — and the gap only shows
up after real money is committed. Being harsher than reality is the safe
direction to be wrong in.
"""

from __future__ import annotations

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.config import PaperConfig
from tradebot.core.types import (
    BookTicker,
    Direction,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    SymbolInfo,
)
from tradebot.paper.broker import PaperBroker

from ..fakes import make_symbol_info


class FakeMarket:
    """Minimal market feed for the broker to price against."""

    def __init__(self, prices: dict[str, float], spread_bps: float = 2.0):
        self.prices = dict(prices)
        self.spread_bps = spread_bps
        self.symbols = {s: make_symbol_info(s, min_notional=1.0) for s in prices}

    def price(self, symbol: str) -> float:
        return self.prices.get(symbol, 0.0)

    def book(self, symbol: str) -> BookTicker | None:
        mid = self.prices.get(symbol)
        if mid is None:
            return None
        half = mid * (self.spread_bps / 10_000.0) / 2.0
        return BookTicker(symbol, mid - half, 1000.0, mid + half, 1000.0, 0)

    def symbol_info(self, symbol: str) -> SymbolInfo | None:
        return self.symbols.get(symbol)

    async def load_symbols(self):
        return self.symbols

    def set(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price


def broker(
    config: PaperConfig | None = None,
    balance: float = 1000.0,
    prices: dict[str, float] | None = None,
    seed: int = 11,
) -> tuple[PaperBroker, FakeMarket]:
    market = FakeMarket(prices or {"TESTUSDT": 100.0})
    params = config or PaperConfig(
        latency_ms=0.0, latency_jitter_ms=0.0, partial_fill_probability=0.0, reject_probability=0.0
    )
    return (
        PaperBroker(
            params,
            market,
            initial_balance=balance,
            clock=VirtualClock(1_700_000_000_000),
            seed=seed,
        ),
        market,
    )


def intent(
    symbol: str = "TESTUSDT",
    direction: Direction = Direction.LONG,
    quantity: float = 1.0,
    entry: float = 100.0,
) -> OrderIntent:
    stop = entry * 0.99 if direction is Direction.LONG else entry * 1.01
    target = entry * 1.02 if direction is Direction.LONG else entry * 0.98
    return OrderIntent(
        intent_id="paper1234",
        symbol=symbol,
        direction=direction,
        side=OrderSide.for_entry(direction),
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=None,
        stop_loss=stop,
        take_profit=target,
        leverage=2,
        notional=quantity * entry,
        risk_amount=1.0,
        strategy="momentum",
        regime=MarketRegime.STRONG_TREND,
        opportunity_score=80.0,
        expected_net_edge=0.002,
        metadata={"reference_price": entry},
    )


class TestFills:
    async def test_an_order_fills_and_opens_a_position(self):
        paper, _ = broker()
        order = await paper.place_order(intent())
        assert order.status is OrderStatus.FILLED
        assert "TESTUSDT" in paper.account.positions
        assert paper.account.positions["TESTUSDT"].quantity == pytest.approx(1.0)

    async def test_a_fee_is_charged_on_entry(self):
        paper, _ = broker(balance=1000.0)
        before = paper.account.balance
        await paper.place_order(intent())
        assert paper.account.balance < before
        assert paper.account.fees_paid > 0

    async def test_buys_fill_worse_than_the_mid_on_average(self):
        """Marketable buys cross the spread.

        Asserted as an average rather than per-fill: a favourable slippage draw
        can land a single fill inside the mid, which is realistic. What must
        never happen is that fills are systematically better than the mid.
        """
        fills = []
        for seed in range(40):
            paper, market = broker(seed=seed)
            order = await paper.place_order(intent(direction=Direction.LONG))
            fills.append(order.average_price - market.price("TESTUSDT"))
        assert sum(fills) / len(fills) > 0, "buys must cost more than the mid"

    async def test_sells_fill_worse_than_the_mid_on_average(self):
        fills = []
        for seed in range(40):
            paper, market = broker(seed=seed)
            order = await paper.place_order(intent(direction=Direction.SHORT))
            fills.append(market.price("TESTUSDT") - order.average_price)
        assert sum(fills) / len(fills) > 0, "sells must receive less than the mid"

    async def test_slippage_is_recorded(self):
        paper, _ = broker()
        await paper.place_order(intent())
        assert paper.stats()["mean_slippage_bps"] > 0

    async def test_an_order_below_the_minimum_notional_is_refused(self):
        from tradebot.core.errors import FilterViolationError

        market = FakeMarket({"TESTUSDT": 100.0})
        market.symbols["TESTUSDT"] = make_symbol_info("TESTUSDT", min_notional=10_000.0)
        paper = PaperBroker(
            PaperConfig(latency_ms=0.0, latency_jitter_ms=0.0),
            market,
            1000.0,
            clock=VirtualClock(0),
            seed=1,
        )
        with pytest.raises(FilterViolationError):
            await paper.place_order(intent())

    async def test_insufficient_margin_is_rejected(self):
        paper, _ = broker(balance=5.0)
        order = await paper.place_order(intent(quantity=10.0))
        assert order.status is OrderStatus.REJECTED
        assert "margin" in (order.error or "")


class TestPessimism:
    async def test_rejections_happen_at_the_configured_rate(self):
        """The rejection path must be exercised routinely, not discovered live."""
        config = PaperConfig(
            latency_ms=0.0,
            latency_jitter_ms=0.0,
            reject_probability=1.0,
            partial_fill_probability=0.0,
        )
        paper, _ = broker(config)
        order = await paper.place_order(intent())
        assert order.status is OrderStatus.REJECTED
        assert paper.rejected_count == 1

    async def test_partial_fills_happen_at_the_configured_rate(self):
        config = PaperConfig(
            latency_ms=0.0,
            latency_jitter_ms=0.0,
            reject_probability=0.0,
            partial_fill_probability=1.0,
        )
        paper, _ = broker(config)
        order = await paper.place_order(intent(quantity=1.0))
        assert order.status is OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity < 1.0
        assert paper.account.positions["TESTUSDT"].quantity == order.filled_quantity

    async def test_slippage_is_biased_against_us(self):
        """Over many fills, adverse slippage must dominate."""
        config = PaperConfig(
            latency_ms=0.0,
            latency_jitter_ms=0.0,
            slippage_bps=5.0,
            adverse_slippage_probability=1.0,
            partial_fill_probability=0.0,
            reject_probability=0.0,
        )
        adverse = 0
        for seed in range(30):
            paper, market = broker(config, seed=seed)
            order = await paper.place_order(intent())
            if order.average_price > market.price("TESTUSDT"):
                adverse += 1
        assert adverse == 30

    async def test_latency_is_simulated(self):
        config = PaperConfig(
            latency_ms=5.0,
            latency_jitter_ms=0.0,
            partial_fill_probability=0.0,
            reject_probability=0.0,
        )
        paper, _ = broker(config)
        import time as _time

        started = _time.perf_counter()
        await paper.place_order(intent())
        assert _time.perf_counter() - started >= 0.004


class TestProtectiveOrders:
    async def test_a_stop_triggers_when_price_reaches_it(self):
        paper, market = broker()
        await paper.place_order(intent(entry=100.0))
        position = paper.account.positions["TESTUSDT"]

        await paper.place_protective_order(
            "TESTUSDT",
            OrderType.STOP_MARKET,
            position.stop_loss,
            position.quantity,
            position.direction.sign,
            "sl_1",
        )
        assert await paper.poll() == []

        market.set("TESTUSDT", 98.0)  # through the 99.0 stop
        triggered = await paper.poll()
        assert triggered
        assert triggered[0][1] == "sl_1"
        assert "TESTUSDT" not in paper.account.positions

    async def test_a_take_profit_triggers_at_its_level(self):
        paper, market = broker()
        await paper.place_order(intent(entry=100.0))
        position = paper.account.positions["TESTUSDT"]

        await paper.place_protective_order(
            "TESTUSDT",
            OrderType.TAKE_PROFIT_MARKET,
            position.take_profit,
            position.quantity,
            position.direction.sign,
            "tp_1",
        )
        market.set("TESTUSDT", 103.0)
        triggered = await paper.poll()
        assert triggered
        assert paper.account.realized_pnl > 0

    async def test_a_short_stop_triggers_above_entry(self):
        paper, market = broker()
        await paper.place_order(intent(direction=Direction.SHORT, entry=100.0))
        position = paper.account.positions["TESTUSDT"]
        await paper.place_protective_order(
            "TESTUSDT",
            OrderType.STOP_MARKET,
            position.stop_loss,
            position.quantity,
            position.direction.sign,
            "sl_s",
        )
        market.set("TESTUSDT", 102.0)
        assert await paper.poll()
        assert "TESTUSDT" not in paper.account.positions

    async def test_cancelling_removes_the_resting_order(self):
        paper, market = broker()
        await paper.place_order(intent())
        position = paper.account.positions["TESTUSDT"]
        await paper.place_protective_order(
            "TESTUSDT",
            OrderType.STOP_MARKET,
            position.stop_loss,
            position.quantity,
            position.direction.sign,
            "sl_c",
        )
        await paper.cancel_order("TESTUSDT", "sl_c")
        market.set("TESTUSDT", 90.0)
        assert await paper.poll() == []


class TestAccounting:
    async def test_a_winning_round_trip_increases_the_balance(self):
        paper, market = broker(balance=1000.0)
        start = paper.account.balance
        await paper.place_order(intent(entry=100.0))
        market.set("TESTUSDT", 105.0)
        position = paper.account.positions["TESTUSDT"]
        await paper.close_position("TESTUSDT", position, "x1")
        assert paper.account.balance > start

    async def test_a_losing_round_trip_decreases_the_balance(self):
        paper, market = broker(balance=1000.0)
        start = paper.account.balance
        await paper.place_order(intent(entry=100.0))
        market.set("TESTUSDT", 95.0)
        position = paper.account.positions["TESTUSDT"]
        await paper.close_position("TESTUSDT", position, "x1")
        assert paper.account.balance < start

    async def test_a_short_profits_when_price_falls(self):
        paper, market = broker(balance=1000.0)
        start = paper.account.balance
        await paper.place_order(intent(direction=Direction.SHORT, entry=100.0))
        market.set("TESTUSDT", 95.0)
        position = paper.account.positions["TESTUSDT"]
        await paper.close_position("TESTUSDT", position, "x1")
        assert paper.account.balance > start

    async def test_margin_is_released_on_close(self):
        paper, _ = broker(balance=1000.0)
        await paper.place_order(intent())
        assert paper.account.margin_used > 0
        position = paper.account.positions["TESTUSDT"]
        assert position.allocated_initial_margin == pytest.approx(paper.account.margin_used)
        await paper.close_position("TESTUSDT", position, "x1")
        assert paper.account.margin_used == pytest.approx(0.0, abs=1e-9)

    async def test_equity_includes_unrealised_pnl(self):
        paper, market = broker(balance=1000.0)
        await paper.place_order(intent(entry=100.0, quantity=2.0))
        market.set("TESTUSDT", 110.0)
        account = await paper.get_account()
        assert account.unrealized_pnl > 0
        assert account.equity > account.total_balance

    async def test_funding_is_charged_to_a_long(self):
        paper, _ = broker(balance=1000.0)
        await paper.place_order(intent())
        before = paper.account.balance
        paid = paper.apply_funding("TESTUSDT", 0.0005)
        assert paid > 0
        assert paper.account.balance < before

    async def test_funding_is_received_by_a_short(self):
        paper, _ = broker(balance=1000.0)
        await paper.place_order(intent(direction=Direction.SHORT))
        before = paper.account.balance
        paid = paper.apply_funding("TESTUSDT", 0.0005)
        assert paid < 0
        assert paper.account.balance > before

    async def test_stats_report_the_simulation_quality(self):
        paper, _ = broker()
        await paper.place_order(intent())
        stats = paper.stats()
        assert stats["filled"] == 1
        assert "mean_slippage_bps" in stats
        assert "fees_paid" in stats


class TestGatewayCompatibility:
    async def test_the_broker_satisfies_the_gateway_protocol(self):
        """The engine above must not be able to tell paper from live."""
        paper, _ = broker()
        for method in (
            "connect",
            "close",
            "load_symbols",
            "symbol_info",
            "get_klines",
            "get_book_ticker",
            "get_ticker_24h",
            "get_mark_price",
            "get_account",
            "get_positions",
            "get_open_orders",
            "set_leverage",
            "place_order",
            "place_protective_order",
            "query_order",
            "cancel_order",
            "cancel_all_orders",
            "close_position",
        ):
            assert hasattr(paper, method), f"missing gateway method: {method}"

    async def test_leverage_is_capped_by_the_symbol(self):
        paper, _ = broker()
        assert await paper.set_leverage("TESTUSDT", 500) <= 20
