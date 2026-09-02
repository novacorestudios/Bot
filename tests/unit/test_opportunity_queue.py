"""The opportunity queue.

The bug it fixes: with four slots and twenty-five candidates, V1 traded the
first candidate that passed risk, in scan-rank order. Scan rank measures how
TRADABLE a symbol is, not how good this particular trade is — so an adequate
opportunity on a highly liquid symbol routinely took the slot a far better
trade needed.
"""

from __future__ import annotations

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.types import (
    AggregatedSignal,
    CostEstimate,
    Direction,
    EdgeEstimate,
    MarketRegime,
    MarketScore,
    OpportunityScore,
    Signal,
)
from tradebot.market.microstructure import LiquiditySnapshot
from tradebot.signals.edge import EdgeDecision
from tradebot.signals.pipeline import Opportunity
from tradebot.signals.queue import OpportunityQueue, rank_key

NO_COSTS = CostEstimate(0.0004, 0.0004, 0.0001, 0.0001, 0.0)


def make_opportunity(
    symbol: str,
    score: float,
    edge: float = 0.002,
    confidence: float = 70.0,
    direction: Direction = Direction.LONG,
    strategy: str = "momentum",
) -> Opportunity:
    contributing = Signal(
        symbol=symbol,
        strategy=strategy,
        direction=direction,
        confidence=confidence,
        entry_price=100.0,
        stop_loss=99.5,
        take_profit=101.0,
        timeframe="5m",
        signal_timestamp=0,
    )
    signal = AggregatedSignal(
        symbol=symbol,
        direction=direction,
        consensus_score=score,
        confidence=confidence,
        entry_price=100.0,
        stop_loss=99.5,
        take_profit=101.0,
        contributing=(contributing,),
        opposing=(),
        conflict_ratio=0.0,
        regime=MarketRegime.STRONG_TREND,
        timestamp=0,
    )
    return Opportunity(
        symbol=symbol,
        signal=signal,
        opportunity_score=OpportunityScore(total=score, components={}, penalties={}),
        edge=EdgeDecision(
            estimate=EdgeEstimate(
                win_probability=0.5,
                gross_win=0.01,
                gross_loss=0.005,
                costs=NO_COSTS,
                expected_gross=edge + NO_COSTS.total,
                expected_net=edge,
            ),
            accepted=True,
            threshold=0.0005,
        ),
        market=MarketScore(
            symbol=symbol,
            total=score,
            components={},
            penalties={},
            volatility=0.005,
            liquidity_usd=1e6,
            spread_bps=1.0,
            funding_rate=0.0,
            timestamp=0,
        ),
        liquidity=LiquiditySnapshot(symbol, 1.0, 1e5, 1e5, 0.0, 1e7),
        regime=MarketRegime.STRONG_TREND,
        notional_estimate=100.0,
        timestamp=0,
    )


@pytest.fixture
def queue(clock: VirtualClock) -> OpportunityQueue:
    return OpportunityQueue(ttl_sec=60.0, max_size=10, clock=clock)


class TestOrdering:
    def test_expected_net_dollars_lead_score(self, queue: OpportunityQueue) -> None:
        high_score = make_opportunity("PRETTY", score=95.0, edge=0.001)
        high_value = make_opportunity("VALUE", score=75.0, edge=0.003)
        queue.add(high_score)
        queue.add(high_value)
        assert queue.best().symbol == "VALUE"  # type: ignore[union-attr]

    def test_the_best_score_is_taken_first_regardless_of_insertion_order(
        self, queue: OpportunityQueue
    ) -> None:
        """The whole point: arrival order is scan rank, and scan rank is not
        opportunity quality."""
        queue.add(make_opportunity("FIRSTUSDT", score=72.0))
        queue.add(make_opportunity("BESTUSDT", score=94.0))
        queue.add(make_opportunity("MIDUSDT", score=81.0))

        assert [e.symbol for e in queue.ranked()] == ["BESTUSDT", "MIDUSDT", "FIRSTUSDT"]
        assert queue.best().symbol == "BESTUSDT"  # type: ignore[union-attr]

    def test_expected_net_edge_breaks_a_score_tie(self, queue: OpportunityQueue) -> None:
        """Between two equally scored trades, the one that keeps more after
        costs is strictly better."""
        queue.add(make_opportunity("THINUSDT", score=85.0, edge=0.0008))
        queue.add(make_opportunity("FATUSDT", score=85.0, edge=0.0031))
        assert [e.symbol for e in queue.ranked()] == ["FATUSDT", "THINUSDT"]

    def test_confidence_breaks_the_remaining_tie(self, queue: OpportunityQueue) -> None:
        queue.add(make_opportunity("AUSDT", score=85.0, edge=0.002, confidence=61.0))
        queue.add(make_opportunity("BUSDT", score=85.0, edge=0.002, confidence=88.0))
        assert [e.symbol for e in queue.ranked()] == ["BUSDT", "AUSDT"]

    def test_rank_key_is_ascending_for_better_opportunities(self) -> None:
        from tradebot.signals.queue import QueuedOpportunity

        better = QueuedOpportunity(make_opportunity("A", 90.0), 0, 60.0)
        worse = QueuedOpportunity(make_opportunity("B", 70.0), 0, 60.0)
        assert rank_key(better) < rank_key(worse)


class TestOnePerSymbol:
    def test_a_better_signal_on_the_same_symbol_replaces_the_worse_one(
        self, queue: OpportunityQueue
    ) -> None:
        """Two strategies firing on one symbol are two views of one trade."""
        queue.add(make_opportunity("BTCUSDT", score=74.0, strategy="vwap"))
        queue.add(make_opportunity("BTCUSDT", score=91.0, strategy="momentum"))
        assert len(queue) == 1
        assert queue.best().score == 91.0  # type: ignore[union-attr]
        assert queue.replaced == 1

    def test_a_worse_signal_on_the_same_symbol_is_dropped(self, queue: OpportunityQueue) -> None:
        queue.add(make_opportunity("BTCUSDT", score=91.0))
        assert queue.add(make_opportunity("BTCUSDT", score=74.0)) is False
        assert queue.best().score == 91.0  # type: ignore[union-attr]


class TestExpiry:
    def test_an_opportunity_expires(self, queue: OpportunityQueue, clock: VirtualClock) -> None:
        """A signal computed on a 5m bar is not still valid ten minutes later."""
        queue.add(make_opportunity("BTCUSDT", score=90.0))
        clock.advance(30)
        assert len(queue) == 1
        clock.advance(31)  # 61s > ttl 60s
        assert queue.ranked() == []
        assert queue.expired == 1

    def test_expiry_does_not_take_live_entries_with_it(
        self, queue: OpportunityQueue, clock: VirtualClock
    ) -> None:
        queue.add(make_opportunity("OLDUSDT", score=90.0))
        clock.advance(50)
        queue.add(make_opportunity("NEWUSDT", score=75.0))
        clock.advance(20)  # OLD is 70s, NEW is 20s
        assert [e.symbol for e in queue.ranked()] == ["NEWUSDT"]


class TestTaking:
    def test_take_returns_only_as_many_as_asked_for(self, queue: OpportunityQueue) -> None:
        for index, score in enumerate([95.0, 88.0, 81.0, 74.0]):
            queue.add(make_opportunity(f"S{index}USDT", score=score))
        taken = queue.take(2)
        assert [e.score for e in taken] == [95.0, 88.0]
        assert len(queue) == 2  # the rest stay queued

    def test_taking_zero_slots_takes_nothing(self, queue: OpportunityQueue) -> None:
        """No free slots means no trades — never a forced fill."""
        queue.add(make_opportunity("BTCUSDT", score=95.0))
        assert queue.take(0) == []
        assert queue.take(-3) == []
        assert len(queue) == 1

    def test_taking_more_than_available_returns_what_there_is(
        self, queue: OpportunityQueue
    ) -> None:
        queue.add(make_opportunity("BTCUSDT", score=95.0))
        assert len(queue.take(10)) == 1
        assert queue.is_empty

    def test_an_empty_queue_is_a_legitimate_outcome(self, queue: OpportunityQueue) -> None:
        assert queue.is_empty
        assert queue.best() is None
        assert queue.take(4) == []


class TestCapacity:
    def test_a_full_queue_displaces_its_worst_entry_for_a_better_one(
        self, clock: VirtualClock
    ) -> None:
        small = OpportunityQueue(ttl_sec=60.0, max_size=3, clock=clock)
        for index, score in enumerate([90.0, 85.0, 80.0]):
            small.add(make_opportunity(f"S{index}USDT", score=score))

        assert small.add(make_opportunity("GREATUSDT", score=99.0)) is True
        assert len(small) == 3
        assert [e.symbol for e in small.ranked()][0] == "GREATUSDT"
        assert "S2USDT" not in small  # the 80.0 was displaced

    def test_a_full_queue_rejects_something_worse_than_everything_in_it(
        self, clock: VirtualClock
    ) -> None:
        small = OpportunityQueue(ttl_sec=60.0, max_size=2, clock=clock)
        small.add(make_opportunity("AUSDT", score=90.0))
        small.add(make_opportunity("BUSDT", score=85.0))
        assert small.add(make_opportunity("MEHUSDT", score=71.0)) is False
        assert "MEHUSDT" not in small


class TestBookkeeping:
    def test_attempts_and_rejections_are_recorded(self, queue: OpportunityQueue) -> None:
        queue.add(make_opportunity("BTCUSDT", score=90.0))
        queue.record_attempt("BTCUSDT", "CORRELATION_LIMIT")
        entry = queue.best()
        assert entry is not None
        assert entry.attempts == 1
        assert entry.last_rejection == "CORRELATION_LIMIT"

    def test_discard_removes_one_symbol(self, queue: OpportunityQueue) -> None:
        queue.add(make_opportunity("BTCUSDT", score=90.0))
        queue.discard("BTCUSDT", "risk refused it outright")
        assert queue.is_empty

    def test_report_and_stats(self, queue: OpportunityQueue) -> None:
        queue.add(make_opportunity("BTCUSDT", score=90.0))
        report = queue.report()
        assert report[0]["symbol"] == "BTCUSDT"
        assert report[0]["direction"] == "LONG"
        assert queue.stats()["size"] == 1
        assert queue.stats()["queued_total"] == 1
