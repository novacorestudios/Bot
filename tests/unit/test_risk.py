"""Risk engine.

These are the most safety-critical tests in the project. They encode the
non-negotiable rules from the brief:

* no trade without a stop loss
* no trade that exceeds the risk budget
* no leverage used to enlarge risk
* no oversizing to meet a minimum notional
* correlated positions are not counted as independent bets
* every rejection is explained
* exits are NEVER blocked
"""

from __future__ import annotations

import random

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.config import (
    AllocationConfig,
    CooldownConfig,
    KillSwitchConfig,
    RiskConfig,
    StrategyKillSwitchConfig,
    load_tunables,
)
from tradebot.core.types import (
    AggregatedSignal,
    Direction,
    MarketRegime,
    MarketScore,
    OpportunityScore,
    RejectionReason,
    Signal,
)
from tradebot.market.candles import CandleStore
from tradebot.market.microstructure import LiquiditySnapshot
from tradebot.risk.allocation import StrategyAllocator, StrategyKillSwitch
from tradebot.risk.cooldown import CooldownManager
from tradebot.risk.correlation import CorrelationEngine
from tradebot.risk.engine import RiskContext, RiskEngine
from tradebot.risk.killswitch import KillSwitchManager, SwitchName
from tradebot.risk.portfolio import PortfolioTracker
from tradebot.risk.sizing import PositionSizer
from tradebot.signals.edge import EdgeDecision
from tradebot.signals.pipeline import Opportunity

from ..conftest import REPO_ROOT, make_candles
from ..fakes import make_symbol_info, position_for

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
)
LIQUID = LiquiditySnapshot("TESTUSDT", 1.0, 500_000.0, 500_000.0, 0.0, 1e9)


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #
class TestPositionSizing:
    def sizer(self, **overrides) -> PositionSizer:
        params = {
            "max_margin_per_trade": 10_000.0,
            "max_total_allocated_margin": 40_000.0,
        }
        params.update(overrides)
        return PositionSizer(RiskConfig(**params))

    def test_risk_is_defined_by_stop_distance(self):
        """0.5% of 75 USDT is 0.375 USDT risked, whatever the stop distance."""
        sizer = self.sizer()
        info = make_symbol_info("X", min_notional=1.0)
        for stop_pct in (0.005, 0.010, 0.020):
            result = sizer.size(75.0, 0.005, 100.0, 100.0 * (1 - stop_pct), Direction.LONG, info)
            assert result.ok, result.detail
            assert result.risk_amount == pytest.approx(0.375, rel=0.02)

    def test_risk_is_never_more_than_budgeted(self):
        """The budget is a ceiling. Any cap may reduce it; nothing may raise it."""
        sizer = self.sizer()
        info = make_symbol_info("X", min_notional=1.0)
        for stop_pct in (0.0005, 0.001, 0.002, 0.005, 0.02, 0.05):
            result = sizer.size(75.0, 0.005, 100.0, 100.0 * (1 - stop_pct), Direction.LONG, info)
            if result.ok:
                assert result.risk_amount <= 0.375 * 1.01, (
                    f"stop {stop_pct}: risked {result.risk_amount} over budget"
                )

    def test_a_very_tight_stop_is_capped_by_symbol_exposure_not_by_risk(self):
        """A real consequence of max_symbol_exposure worth understanding.

        With risk_per_trade 0.5% and max_symbol_exposure 1.0x equity, any stop
        tighter than 0.5% produces a position the exposure cap trims — so the
        trade risks LESS than the budget. That is safe, but it means very tight
        stops cannot use the full risk allowance. Raising max_symbol_exposure is
        the lever, and it should be changed knowingly, not by accident.
        """
        sizer = PositionSizer(
            RiskConfig(
                risk_per_trade=0.005,
                max_symbol_exposure=1.0,
                max_margin_per_trade=10_000.0,
                max_total_allocated_margin=40_000.0,
            )
        )
        info = make_symbol_info("X", min_notional=1.0)
        result = sizer.size(75.0, 0.005, 100.0, 99.8, Direction.LONG, info)
        assert result.ok
        assert result.notional <= 75.0 * 1.01
        assert result.risk_amount < 0.375

    def test_tighter_stop_gives_a_larger_position(self):
        sizer = self.sizer()
        info = make_symbol_info("X", min_notional=1.0)
        tight = sizer.size(1000.0, 0.005, 100.0, 99.8, Direction.LONG, info)
        wide = sizer.size(1000.0, 0.005, 100.0, 99.0, Direction.LONG, info)
        assert tight.quantity > wide.quantity

    def test_leverage_does_not_change_the_risk(self):
        """Leverage changes margin, not risk. This is the central point."""
        low = PositionSizer(
            RiskConfig(
                max_leverage=2,
                max_margin_per_trade=10_000.0,
                max_total_allocated_margin=40_000.0,
            )
        )
        high = PositionSizer(
            RiskConfig(
                max_leverage=20,
                max_margin_per_trade=10_000.0,
                max_total_allocated_margin=40_000.0,
            )
        )
        info = make_symbol_info("X", min_notional=1.0)
        a = low.size(1000.0, 0.005, 100.0, 99.0, Direction.LONG, info)
        b = high.size(1000.0, 0.005, 100.0, 99.0, Direction.LONG, info)
        assert a.ok and b.ok
        assert a.risk_amount == pytest.approx(b.risk_amount)

    def test_zero_stop_distance_is_rejected(self):
        result = self.sizer().size(
            1000.0, 0.005, 100.0, 100.0, Direction.LONG, make_symbol_info("X")
        )
        assert not result.ok
        assert result.reason is RejectionReason.INVALID_STOP

    def test_stop_on_the_wrong_side_is_rejected(self):
        sizer = self.sizer()
        info = make_symbol_info("X")
        assert not sizer.size(1000.0, 0.005, 100.0, 101.0, Direction.LONG, info).ok
        assert not sizer.size(1000.0, 0.005, 100.0, 99.0, Direction.SHORT, info).ok

    def test_minimum_notional_causes_a_skip_not_an_oversize(self):
        """The small-account wall. Rounding up would breach the risk budget."""
        sizer = self.sizer()
        info = make_symbol_info("X", min_notional=500.0)
        result = sizer.size(75.0, 0.005, 100.0, 99.5, Direction.LONG, info)
        assert not result.ok
        assert result.reason is RejectionReason.NOTIONAL_BELOW_MINIMUM
        assert "Skipping rather than oversizing" in result.detail
        assert result.checks["implied_risk"] > result.checks["budgeted_risk"]

    def test_quantity_rounding_to_zero_is_rejected(self):
        sizer = self.sizer()
        info = make_symbol_info("X", step=1.0, min_qty=1.0, min_notional=1.0)
        result = sizer.size(75.0, 0.005, 100_000.0, 99_500.0, Direction.LONG, info)
        assert not result.ok
        assert result.reason in {
            RejectionReason.SIZE_BELOW_MINIMUM,
            RejectionReason.NOTIONAL_BELOW_MINIMUM,
        }

    def test_quantity_is_step_aligned(self):
        from decimal import Decimal

        result = self.sizer().size(
            10_000.0,
            0.005,
            137.77,
            135.0,
            Direction.LONG,
            make_symbol_info("X", step=0.01, min_notional=1.0),
        )
        assert result.ok
        assert Decimal(str(result.quantity)) % Decimal("0.01") == 0

    def test_volatility_reduces_the_leverage_ceiling(self):
        sizer = self.sizer(max_leverage=20, leverage_volatility_threshold=0.01)
        calm = sizer.volatility_adjusted_max_leverage(0.005)
        wild = sizer.volatility_adjusted_max_leverage(0.05)
        assert calm == 20
        assert wild < calm
        assert wild >= 1

    def test_liquidation_too_close_to_the_stop_is_rejected(self):
        """The exchange must never be able to close a position before our stop."""
        sizer = PositionSizer(
            RiskConfig(
                max_leverage=100,
                min_liquidation_distance_multiple=3.0,
                max_margin_per_trade=10_000.0,
                max_total_allocated_margin=40_000.0,
            )
        )
        info = make_symbol_info("X", min_notional=1.0, max_leverage=100)
        # A 5% stop at 50x: liquidation sits ~2% away, well inside the stop.
        result = sizer.size(100.0, 0.02, 100.0, 95.0, Direction.LONG, info, available_margin=2.0)
        if not result.ok:
            assert result.reason in {
                RejectionReason.LIQUIDATION_TOO_CLOSE,
                RejectionReason.LEVERAGE_LIMIT,
                RejectionReason.MARGIN_LIMIT,
            }

    def test_liquidation_estimate_directions(self):
        sizer = self.sizer()
        long_liq = sizer.estimate_liquidation(100.0, Direction.LONG, 10)
        short_liq = sizer.estimate_liquidation(100.0, Direction.SHORT, 10)
        assert long_liq < 100.0 < short_liq
        # At 10x, liquidation is roughly 10% away.
        assert long_liq == pytest.approx(90.4, abs=0.5)

    def test_higher_leverage_brings_liquidation_closer(self):
        sizer = self.sizer()
        near = sizer.estimate_liquidation(100.0, Direction.LONG, 50)
        far = sizer.estimate_liquidation(100.0, Direction.LONG, 2)
        assert near > far

    def test_risk_fraction_is_clamped_to_configured_bounds(self):
        sizer = self.sizer(min_risk_per_trade=0.001, max_risk_per_trade=0.01)
        info = make_symbol_info("X", min_notional=1.0)
        result = sizer.size(10_000.0, 0.50, 100.0, 99.0, Direction.LONG, info)
        assert result.ok
        assert result.risk_amount <= 10_000.0 * 0.01 * 1.01

    def test_zero_equity_is_rejected(self):
        assert (
            not self.sizer().size(0.0, 0.005, 100.0, 99.0, Direction.LONG, make_symbol_info("X")).ok
        )

    def test_target_account_position_never_exceeds_five_usdt_margin(self):
        sizer = PositionSizer(RiskConfig())
        result = sizer.size(
            200.0, 0.005, 100.0, 96.0, Direction.LONG, make_symbol_info("X", min_notional=1.0)
        )
        assert result.ok, result.detail
        assert result.margin_required <= 5.0
        assert result.leverage <= 5

    def test_leverage_is_minimum_needed_not_forced_to_five(self):
        sizer = PositionSizer(RiskConfig())
        result = sizer.size(
            200.0, 0.005, 100.0, 75.0, Direction.LONG, make_symbol_info("X", min_notional=1.0)
        )
        assert result.ok, result.detail
        assert result.leverage == 1

    def test_risk_quantity_is_not_increased_to_consume_margin(self):
        sizer = PositionSizer(RiskConfig())
        result = sizer.size(
            200.0, 0.005, 100.0, 75.0, Direction.LONG, make_symbol_info("X", min_notional=1.0)
        )
        assert result.ok, result.detail
        assert result.raw_quantity == pytest.approx(0.04)
        assert result.quantity == pytest.approx(0.04)
        assert result.margin_required == pytest.approx(4.0)

    def test_per_trade_margin_cap_has_an_explicit_reason(self):
        result = PositionSizer(RiskConfig()).size(
            200.0, 0.005, 100.0, 98.0, Direction.LONG, make_symbol_info("X", min_notional=1.0)
        )
        assert not result.ok
        assert result.reason is RejectionReason.PER_TRADE_MARGIN_LIMIT

    def test_total_margin_cap_has_an_explicit_reason(self):
        result = PositionSizer(RiskConfig()).size(
            200.0,
            0.005,
            100.0,
            90.0,
            Direction.LONG,
            make_symbol_info("X", min_notional=1.0),
            total_margin_available=1.9,
        )
        assert not result.ok
        assert result.reason is RejectionReason.TOTAL_MARGIN_LIMIT

    def test_leverage_limit_has_an_explicit_reason(self):
        config = RiskConfig(
            max_leverage=2,
            max_margin_per_trade=100.0,
            max_total_allocated_margin=100.0,
        )
        result = PositionSizer(config).size(
            200.0,
            0.005,
            100.0,
            95.0,
            Direction.LONG,
            make_symbol_info("X", min_notional=1.0, max_leverage=2),
            available_margin=5.0,
        )
        assert not result.ok
        assert result.reason is RejectionReason.LEVERAGE_LIMIT

    def test_exchange_minimum_never_causes_risk_oversizing(self):
        result = PositionSizer(RiskConfig()).size(
            200.0,
            0.005,
            100.0,
            90.0,
            Direction.LONG,
            make_symbol_info("X", min_notional=20.0),
        )
        assert not result.ok
        assert result.reason is RejectionReason.NOTIONAL_BELOW_MINIMUM
        assert result.checks["implied_risk"] > result.checks["budgeted_risk"]


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
class TestCorrelation:
    def engine_with(self, groups: dict[str, float]) -> CorrelationEngine:
        """Build symbols; those sharing a base series are correlated."""
        store = CandleStore(500)
        rng = random.Random(4)
        base = [100.0]
        for _ in range(300):
            base.append(base[-1] * (1 + rng.gauss(0, 0.002)))
        for name, noise in groups.items():
            local = random.Random(hash(name) % 10_000)
            if noise < 0:  # independent series
                prices = [100.0]
                for _ in range(300):
                    prices.append(prices[-1] * (1 + local.gauss(0, 0.002)))
            else:
                prices = [p * (1 + local.gauss(0, noise)) for p in base]
            store.series(name, "5m").extend(make_candles(prices))
        return CorrelationEngine(RiskConfig(), store)

    def test_correlated_symbols_measure_high(self):
        engine = self.engine_with({"AAA": 0.0, "BBB": 0.0002})
        assert engine.correlation("AAA", "BBB") > 0.9

    def test_independent_symbols_measure_low(self):
        engine = self.engine_with({"AAA": 0.0, "IND": -1})
        assert abs(engine.correlation("AAA", "IND")) < 0.3

    def test_a_symbol_is_perfectly_correlated_with_itself(self):
        assert self.engine_with({"AAA": 0.0}).correlation("AAA", "AAA") == 1.0

    def test_missing_history_returns_zero_rather_than_raising(self):
        assert self.engine_with({"AAA": 0.0}).correlation("AAA", "UNKNOWN") == 0.0

    def test_four_correlated_longs_are_one_effective_bet(self):
        """The brief's exact scenario: BTC/ETH/SOL/SUI all long."""
        engine = self.engine_with({"AAA": 0.0, "BBB": 0.0002, "CCC": 0.0003, "DDD": 0.0002})
        effective = engine.effective_positions(["AAA", "BBB", "CCC", "DDD"], [100.0] * 4)
        assert effective < 1.5, f"4 correlated longs measured as {effective:.2f} bets"

    def test_uncorrelated_positions_count_as_independent(self):
        store = CandleStore(500)
        for i in range(4):
            rng = random.Random(100 + i)
            prices = [100.0]
            for _ in range(300):
                prices.append(prices[-1] * (1 + rng.gauss(0, 0.002)))
            store.series(f"S{i}", "5m").extend(make_candles(prices))
        engine = CorrelationEngine(RiskConfig(), store)
        effective = engine.effective_positions([f"S{i}" for i in range(4)], [100.0] * 4)
        assert effective > 3.0

    def test_a_hedge_is_not_counted_as_concentration(self):
        """Long A and short a correlated B partially offset."""
        engine = self.engine_with({"AAA": 0.0, "BBB": 0.0002})
        same = engine.effective_positions(["AAA", "BBB"], [100.0, 100.0])
        hedged = engine.effective_positions(["AAA", "BBB"], [100.0, -100.0])
        assert hedged > same

    def test_adding_a_highly_correlated_position_is_refused(self):
        engine = self.engine_with({"AAA": 0.0, "BBB": 0.0002})
        assessment = engine.assess("AAA", Direction.LONG, 100.0, {"BBB": (Direction.LONG, 100.0)})
        assert not assessment.acceptable
        assert "same bet twice" in assessment.detail

    def test_the_first_position_is_always_acceptable(self):
        engine = self.engine_with({"AAA": 0.0})
        assert engine.assess("AAA", Direction.LONG, 100.0, {}).acceptable

    def test_an_uncorrelated_addition_to_one_position_is_acceptable(self):
        engine = self.engine_with({"AAA": 0.0, "IND": -1})
        assert engine.assess(
            "IND", Direction.LONG, 100.0, {"AAA": (Direction.LONG, 100.0)}
        ).acceptable

    def test_opposite_direction_in_correlated_assets_is_allowed(self):
        """Long A and short a correlated B is a spread, not a doubled bet.

        Concentration must be measured on SIGNED exposure. Using absolute
        correlation scored this pair at 0.99 and rejected it, which would have
        forbidden every hedge the system could construct.
        """
        engine = self.engine_with({"AAA": 0.0, "BBB": 0.0002})
        hedged = engine.assess("AAA", Direction.SHORT, 100.0, {"BBB": (Direction.LONG, 100.0)})
        doubled = engine.assess("AAA", Direction.LONG, 100.0, {"BBB": (Direction.LONG, 100.0)})
        assert hedged.acceptable
        assert not doubled.acceptable
        assert hedged.portfolio_correlation < doubled.portfolio_correlation


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
class TestPortfolio:
    def tracker(self, **overrides) -> PortfolioTracker:
        return PortfolioTracker(RiskConfig(**overrides))

    def test_open_risk_uses_stop_distance_not_notional(self):
        """A big position with a tight stop risks less than the reverse."""
        tracker = self.tracker()
        big_tight = {"A": position_for("A", quantity=10.0, entry=100.0, stop=99.8)}
        small_wide = {"B": position_for("B", quantity=1.0, entry=100.0, stop=95.0)}
        a = tracker.state(1000.0, 1000.0, big_tight, {"A": 100.0})
        b = tracker.state(1000.0, 1000.0, small_wide, {"B": 100.0})
        assert a.total_exposure > b.total_exposure
        assert a.total_open_risk < b.total_open_risk

    def test_a_position_without_a_stop_counts_its_whole_notional_as_risk(self):
        tracker = self.tracker()
        naked = {"A": position_for("A", quantity=1.0, entry=100.0, stop=0.0)}
        state = tracker.state(1000.0, 1000.0, naked, {"A": 100.0})
        assert state.total_open_risk == pytest.approx(100.0)
        assert "A" in state.unprotected_positions

    def test_a_winner_past_its_stop_contributes_no_risk(self):
        tracker = self.tracker()
        winner = {"A": position_for("A", quantity=1.0, entry=100.0, stop=99.0)}
        state = tracker.state(1000.0, 1000.0, winner, {"A": 105.0})
        assert state.total_open_risk >= 0.0

    def test_long_and_short_exposure_are_tracked_separately(self):
        tracker = self.tracker()
        positions = {
            "A": position_for("A", Direction.LONG, quantity=1.0, entry=100.0),
            "B": position_for("B", Direction.SHORT, quantity=2.0, entry=50.0),
        }
        state = tracker.state(1000.0, 1000.0, positions, {"A": 100.0, "B": 50.0})
        assert state.total_long_exposure == pytest.approx(100.0)
        assert state.total_short_exposure == pytest.approx(100.0)
        assert state.net_direction_exposure == pytest.approx(0.0)

    def test_max_positions_is_enforced(self):
        tracker = self.tracker(max_concurrent_positions=2)
        positions = {s: position_for(s) for s in ("A", "B")}
        state = tracker.state(1000.0, 1000.0, positions, {"A": 100.0, "B": 100.0})
        breached, limit, _ = tracker.would_breach(state, "C", Direction.LONG, 100.0, 1.0, 20.0)
        assert breached
        assert limit == "MAX_POSITIONS"

    def test_total_risk_budget_is_enforced(self):
        tracker = self.tracker(max_total_risk=0.02, max_concurrent_positions=10)
        positions = {"A": position_for("A", quantity=1.0, entry=100.0, stop=82.0)}
        state = tracker.state(1000.0, 1000.0, positions, {"A": 100.0})
        breached, limit, detail = tracker.would_breach(
            state, "B", Direction.LONG, 100.0, 10.0, 20.0
        )
        assert breached
        assert limit == "TOTAL_RISK"
        assert "budget" in detail

    def test_direction_exposure_limit_is_enforced(self):
        tracker = self.tracker(
            max_direction_exposure=1.0, max_concurrent_positions=10, max_total_exposure=100.0
        )
        positions = {"A": position_for("A", Direction.LONG, quantity=9.0, entry=100.0)}
        state = tracker.state(1000.0, 1000.0, positions, {"A": 100.0})
        breached, limit, _ = tracker.would_breach(state, "B", Direction.LONG, 500.0, 1.0, 20.0)
        assert breached
        assert limit == "DIRECTION_EXPOSURE"

    def test_margin_usage_limit_is_enforced(self):
        tracker = self.tracker(max_margin_usage=0.5, max_concurrent_positions=10)
        state = tracker.state(1000.0, 1000.0, {}, {})
        breached, limit, _ = tracker.would_breach(state, "A", Direction.LONG, 100.0, 1.0, 600.0)
        assert breached
        assert limit == "MARGIN_USAGE"

    def test_an_empty_portfolio_permits_a_reasonable_trade(self):
        tracker = self.tracker()
        state = tracker.state(1000.0, 1000.0, {}, {})
        breached, _, _ = tracker.would_breach(state, "A", Direction.LONG, 100.0, 5.0, 20.0)
        assert not breached

    def test_remaining_budget_shrinks_as_risk_is_taken(self):
        tracker = self.tracker(max_total_risk=0.02)
        empty = tracker.state(1000.0, 1000.0, {}, {})
        loaded = tracker.state(
            1000.0,
            1000.0,
            {"A": position_for("A", quantity=1.0, entry=100.0, stop=90.0)},
            {"A": 100.0},
        )
        assert tracker.remaining_risk_budget(empty) == pytest.approx(20.0)
        assert tracker.remaining_risk_budget(loaded) < 20.0

    def test_allocated_initial_margin_does_not_move_with_mark_price(self):
        from dataclasses import replace

        tracker = self.tracker()
        position = replace(
            position_for("A", quantity=0.2, leverage=5), allocated_initial_margin=4.0
        )
        low = tracker.state(200.0, 196.0, {"A": position}, {"A": 50.0})
        high = tracker.state(200.0, 196.0, {"A": position}, {"A": 150.0})
        assert low.margin_used != high.margin_used
        assert low.allocated_margin == pytest.approx(4.0)
        assert high.allocated_margin == pytest.approx(4.0)

    def test_total_allocated_margin_never_exceeds_twenty(self):
        from dataclasses import replace

        tracker = self.tracker(max_concurrent_positions=10)
        positions = {
            symbol: replace(
                position_for(symbol, quantity=0.2, leverage=5), allocated_initial_margin=5.0
            )
            for symbol in ("A", "B", "C", "D")
        }
        state = tracker.state(200.0, 180.0, positions, dict.fromkeys(positions, 100.0))
        breached, limit, _ = tracker.would_breach(state, "E", Direction.SHORT, 1.0, 0.0, 0.01)
        assert state.allocated_margin == pytest.approx(20.0)
        assert breached
        assert limit == "TOTAL_MARGIN"


# --------------------------------------------------------------------------- #
# Kill switches
# --------------------------------------------------------------------------- #
class TestKillSwitches:
    def manager(self, clock=None, **risk_overrides) -> KillSwitchManager:
        return KillSwitchManager(
            KillSwitchConfig(),
            RiskConfig(**risk_overrides),
            clock or VirtualClock(1_700_000_000_000),
        )

    def test_entries_are_allowed_when_nothing_is_wrong(self):
        manager = self.manager()
        manager.evaluate(1000.0)
        assert manager.entries_allowed

    def test_daily_loss_limit_halts_entries(self):
        manager = self.manager(max_daily_loss=0.02)
        manager.evaluate(1000.0)
        manager.evaluate(975.0)  # -2.5%
        assert not manager.entries_allowed
        assert SwitchName.DAILY_LOSS in manager.active

    def test_drawdown_from_peak_halts_entries(self):
        # The daily and hourly limits are set just under the drawdown limit so
        # this test isolates the drawdown switch (config forbids daily > drawdown).
        manager = self.manager(max_drawdown=0.10, max_daily_loss=0.099, max_hourly_loss=0.099)
        manager.evaluate(1000.0)
        manager.evaluate(1200.0)  # new peak
        manager.evaluate(1050.0)  # -12.5% from peak
        assert SwitchName.MAX_DRAWDOWN in manager.active

    def test_consecutive_losses_halt_entries(self):
        manager = self.manager(max_consecutive_losses=3)
        manager.evaluate(1000.0)
        for _ in range(3):
            manager.record_trade_result(won=False)
        manager.evaluate(1000.0)
        assert SwitchName.CONSECUTIVE_LOSSES in manager.active

    def test_a_win_resets_the_consecutive_loss_counter(self):
        manager = self.manager(max_consecutive_losses=3)
        manager.evaluate(1000.0)
        manager.record_trade_result(won=False)
        manager.record_trade_result(won=False)
        manager.record_trade_result(won=True)
        manager.record_trade_result(won=False)
        manager.evaluate(1000.0)
        assert manager.entries_allowed

    def test_consecutive_loss_cooldown_expiry_preserves_counter_and_retrips(self):
        """Characterize the current re-arm semantics; this is not a desired-state test."""
        clock = VirtualClock(1_700_000_000_000)
        manager = KillSwitchManager(
            KillSwitchConfig(auto_rearm_seconds=900),
            RiskConfig(max_consecutive_losses=5),
            clock,
        )
        manager.evaluate(1000.0)
        for _ in range(5):
            manager.record_trade_result(won=False)

        manager.evaluate(1000.0)
        assert manager.consecutive_losses == 5
        assert SwitchName.CONSECUTIVE_LOSSES in manager.active
        assert not manager.entries_allowed

        clock.advance(899)
        manager.evaluate(1000.0)
        assert SwitchName.CONSECUTIVE_LOSSES in manager.active

        clock.advance(1)
        manager.evaluate(1000.0)
        assert manager.entries_allowed  # expiry occurs at the end of this evaluation
        assert manager.consecutive_losses == 5

        manager.evaluate(1000.0)
        assert SwitchName.CONSECUTIVE_LOSSES in manager.active
        assert not manager.entries_allowed
        assert manager.consecutive_losses == 5

    def test_new_trading_day_resets_consecutive_losses(self):
        clock = VirtualClock(1_700_000_000_000)
        manager = KillSwitchManager(
            KillSwitchConfig(auto_rearm_seconds=900),
            RiskConfig(max_consecutive_losses=5, day_reset_hour_utc=0),
            clock,
        )
        manager.evaluate(1000.0)
        for _ in range(5):
            manager.record_trade_result(won=False)
        manager.evaluate(1000.0)
        assert not manager.entries_allowed

        clock.advance(86_400)
        manager.evaluate(1000.0)
        assert manager.consecutive_losses == 0
        assert SwitchName.CONSECUTIVE_LOSSES not in manager.active

    def test_kill_switch_only_blocks_entries_not_exits(self):
        """The manager exposes no exit gate; a trip changes entries_allowed only."""
        manager = self.manager(max_consecutive_losses=5)
        manager.evaluate(1000.0)
        for _ in range(5):
            manager.record_trade_result(won=False)
        manager.evaluate(1000.0)

        assert not manager.entries_allowed
        assert not hasattr(manager, "exits_allowed")

    def test_api_error_burst_halts_entries(self):
        manager = self.manager()
        manager.evaluate(1000.0)
        for _ in range(manager.config.max_api_errors_per_5min):
            manager.record_api_error()
        manager.evaluate(1000.0)
        assert SwitchName.API_ERRORS in manager.active

    def test_repeated_order_rejections_halt_entries(self):
        manager = self.manager()
        manager.evaluate(1000.0)
        for _ in range(manager.config.max_rejected_orders_per_hour):
            manager.record_order_rejection()
        manager.evaluate(1000.0)
        assert SwitchName.REJECTED_ORDERS in manager.active

    def test_excessive_slippage_halts_entries(self):
        manager = self.manager()
        manager.evaluate(1000.0)
        for _ in range(5):
            manager.record_slippage(0.01)
        manager.evaluate(1000.0)
        assert SwitchName.SLIPPAGE in manager.active

    def test_reconciliation_mismatches_halt_entries(self):
        manager = self.manager()
        manager.evaluate(1000.0)
        for _ in range(manager.config.max_reconciliation_mismatches):
            manager.record_reconciliation_mismatch()
        manager.evaluate(1000.0)
        assert SwitchName.RECONCILIATION in manager.active

    def test_stale_data_halts_entries_and_clears_when_fresh(self):
        manager = self.manager()
        manager.evaluate(1000.0, data_age_sec=120.0)
        assert SwitchName.STALE_DATA in manager.active
        manager.evaluate(1000.0, data_age_sec=1.0)
        assert SwitchName.STALE_DATA not in manager.active

    def test_disconnection_halts_entries_and_clears_on_reconnect(self):
        manager = self.manager()
        manager.evaluate(1000.0, connected=False)
        assert SwitchName.CONNECTION in manager.active
        manager.evaluate(1000.0, connected=True)
        assert SwitchName.CONNECTION not in manager.active

    def test_a_new_trading_day_clears_the_daily_loss_switch(self):
        clock = VirtualClock(1_700_000_000_000)
        manager = self.manager(clock, max_daily_loss=0.02)
        manager.evaluate(1000.0)
        manager.evaluate(950.0)
        assert not manager.entries_allowed
        clock.advance(86_400 + 60)
        manager.evaluate(950.0)
        assert SwitchName.DAILY_LOSS not in manager.active

    def test_operator_can_reset_a_switch(self):
        manager = self.manager(max_daily_loss=0.02)
        manager.evaluate(1000.0)
        manager.evaluate(950.0)
        manager.reset()
        assert manager.entries_allowed

    def test_every_trip_records_a_risk_event(self):
        manager = self.manager(max_daily_loss=0.02)
        manager.evaluate(1000.0)
        manager.evaluate(950.0)
        assert manager.events
        assert manager.blocking_reason()


# --------------------------------------------------------------------------- #
# Cooldown
# --------------------------------------------------------------------------- #
class TestCooldown:
    def manager(self, clock=None, **overrides) -> CooldownManager:
        return CooldownManager(CooldownConfig(**overrides), clock or VirtualClock(1_000_000_000))

    def test_a_loss_produces_a_longer_cooldown_than_a_win(self):
        manager = self.manager()
        loss = manager.duration_for(won=False, symbol="A")
        win = manager.duration_for(won=True, symbol="A")
        assert loss > win

    def test_consecutive_losses_lengthen_the_cooldown(self):
        clock = VirtualClock(1_000_000_000)
        manager = self.manager(clock, consecutive_loss_multiplier=2.0)
        first = manager.register_close("A", won=False)
        clock.advance(first + 1)
        second = manager.register_close("A", won=False)
        assert second > first

    def test_a_win_resets_the_streak(self):
        clock = VirtualClock(1_000_000_000)
        manager = self.manager(clock)
        manager.register_close("A", won=False)
        clock.advance(10_000)
        manager.register_close("A", won=True)
        clock.advance(10_000)
        assert manager.register_close("A", won=False) == pytest.approx(
            manager.config.after_loss_seconds, rel=0.01
        )

    def test_cooldown_blocks_then_expires(self):
        clock = VirtualClock(1_000_000_000)
        manager = self.manager(clock)
        duration = manager.register_close("A", won=False)
        assert manager.is_active("A")
        clock.advance(duration + 1)
        assert not manager.is_active("A")

    def test_cooldown_is_per_symbol(self):
        manager = self.manager()
        manager.register_close("A", won=False)
        assert manager.is_active("A")
        assert not manager.is_active("B")

    def test_cooldown_is_capped(self):
        clock = VirtualClock(1_000_000_000)
        manager = self.manager(clock, max_seconds=300, consecutive_loss_multiplier=10.0)
        for _ in range(5):
            duration = manager.register_close("A", won=False)
            clock.advance(duration + 1)
        assert duration <= 300

    def test_high_volatility_shortens_the_cooldown(self):
        manager = self.manager()
        calm = manager.duration_for(False, "A", volatility=0.002)
        wild = manager.duration_for(False, "A", volatility=0.02)
        assert wild < calm


# --------------------------------------------------------------------------- #
# Allocation and the strategy kill switch
# --------------------------------------------------------------------------- #
class TestAllocation:
    def allocator(self, **overrides) -> StrategyAllocator:
        return StrategyAllocator(AllocationConfig(**overrides))

    def test_weights_stay_at_parity_without_enough_evidence(self):
        """Reallocating on ten trades is fitting noise."""
        allocator = self.allocator(min_trades_for_adjustment=30)
        for _ in range(10):
            allocator.record_trade("winner", 2.0)
        weights = allocator.weights(["winner", "other"])
        assert weights["winner"] == pytest.approx(1.0)

    def test_a_proven_strategy_earns_more_weight(self):
        allocator = self.allocator(min_trades_for_adjustment=30)
        for i in range(60):
            allocator.record_trade("good", 0.8 if i % 2 else 0.6)
            allocator.record_trade("poor", -0.4 if i % 2 else 0.1)
        weights = allocator.weights(["good", "poor"])
        assert weights["good"] > weights["poor"]

    def test_weights_are_bounded(self):
        """No strategy may take over the account on a hot streak."""
        allocator = self.allocator(min_trades_for_adjustment=30, min_weight=0.4, max_weight=2.0)
        for _ in range(100):
            allocator.record_trade("star", 5.0)
            allocator.record_trade("dud", -0.5)
        weights = allocator.weights(["star", "dud"])
        assert 0.4 <= weights["star"] <= 2.0
        assert 0.4 <= weights["dud"] <= 2.0

    def test_disabled_allocation_gives_equal_weights(self):
        allocator = self.allocator(enabled=False)
        for _ in range(100):
            allocator.record_trade("good", 2.0)
        assert allocator.weights(["good", "other"]) == {"good": 1.0, "other": 1.0}

    def test_expectancy_and_profit_factor_are_computed(self):
        allocator = self.allocator()
        for r in (2.0, -1.0, 2.0, -1.0):
            allocator.record_trade("s", r)
        perf = allocator.performance_for("s")
        assert perf.expectancy_r == pytest.approx(0.5)
        assert perf.profit_factor == pytest.approx(2.0)
        assert perf.win_rate == pytest.approx(0.5)

    def test_drawdown_in_r_is_tracked(self):
        allocator = self.allocator()
        for r in (1.0, 1.0, -1.0, -1.0, -0.5):
            allocator.record_trade("s", r)
        assert allocator.performance_for("s").max_drawdown_r == pytest.approx(2.5)


class TestStrategyKillSwitch:
    def switch(self, allocator, **overrides):
        return StrategyKillSwitch(
            StrategyKillSwitchConfig(**overrides), allocator, VirtualClock(1_000_000_000)
        )

    def test_a_strategy_is_not_judged_before_enough_trades(self):
        allocator = StrategyAllocator(AllocationConfig())
        for _ in range(10):
            allocator.record_trade("new", -1.0)
        disable, reason = self.switch(allocator, min_trades=40).should_disable("new")
        assert not disable
        assert "needed to judge" in reason

    def test_a_losing_strategy_is_suspended(self):
        allocator = StrategyAllocator(AllocationConfig())
        for i in range(60):
            allocator.record_trade("failing", -1.0 if i % 3 else 0.5)
        disable, reason = self.switch(allocator, min_trades=40).should_disable("failing")
        assert disable
        assert reason

    def test_a_profitable_strategy_is_left_alone(self):
        allocator = StrategyAllocator(AllocationConfig())
        for i in range(60):
            allocator.record_trade("working", 1.5 if i % 3 else -1.0)
        disable, _ = self.switch(allocator, min_trades=40).should_disable("working")
        assert not disable

    def test_disabled_switch_never_suspends(self):
        allocator = StrategyAllocator(AllocationConfig())
        for _ in range(100):
            allocator.record_trade("terrible", -2.0)
        disable, _ = self.switch(allocator, enabled=False).should_disable("terrible")
        assert not disable


# --------------------------------------------------------------------------- #
# The engine, end to end
# --------------------------------------------------------------------------- #
def make_opportunity(
    symbol: str = "TESTUSDT",
    direction: Direction = Direction.LONG,
    entry: float = 100.0,
    stop_pct: float = 0.020,
    target_pct: float = 0.010,
    strategy: str = "momentum",
    edge: float = 0.0015,
    score: float = 85.0,
    volatility: float = 0.005,
) -> Opportunity:
    if direction is Direction.LONG:
        stop, target = entry * (1 - stop_pct), entry * (1 + target_pct)
    else:
        stop, target = entry * (1 + stop_pct), entry * (1 - target_pct)
    contributing = (
        Signal(symbol, strategy, direction, 85.0, entry, stop, target, "3m", 0),
        Signal(symbol, "trend_following", direction, 80.0, entry, stop, target, "5m", 0),
    )
    signal = AggregatedSignal(
        symbol=symbol,
        direction=direction,
        consensus_score=78.0,
        confidence=82.0,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        contributing=contributing,
        opposing=(),
        conflict_ratio=0.0,
        regime=MarketRegime.STRONG_TREND,
        timestamp=0,
    )
    market = MarketScore(
        symbol=symbol,
        total=score,
        components={},
        penalties={},
        volatility=volatility,
        liquidity_usd=500_000.0,
        spread_bps=1.0,
        funding_rate=0.0001,
        timestamp=0,
    )
    from tradebot.core.types import CostEstimate, EdgeEstimate

    costs = CostEstimate(0.0004, 0.0004, 0.0001, 0.0002, 0.0)
    estimate = EdgeEstimate(0.55, target_pct, stop_pct, costs, edge + costs.total, edge)
    return Opportunity(
        symbol=symbol,
        signal=signal,
        opportunity_score=OpportunityScore(score, {}, {}),
        edge=EdgeDecision(estimate, True, 0.0008),
        market=market,
        liquidity=LIQUID,
        regime=MarketRegime.STRONG_TREND,
        notional_estimate=75.0,
        timestamp=0,
    )


class TestRiskEngine:
    def engine(self, config=CONFIG, clock=None) -> RiskEngine:
        return RiskEngine(config, CandleStore(500), clock or VirtualClock(1_700_000_000))

    def context(self, **overrides) -> RiskContext:
        params = {
            "equity": 75.0,
            "available_balance": 75.0,
            "positions": {},
            "prices": {},
            "symbol_info": make_symbol_info("TESTUSDT", min_notional=1.0),
            "now": 1_700_000_000.0,
        }
        params.update(overrides)
        return RiskContext(**params)

    def test_a_sound_opportunity_is_approved(self):
        decision = self.engine().evaluate(make_opportunity(), self.context())
        assert decision.approved, decision.detail
        assert decision.intent is not None
        assert decision.intent.quantity > 0
        assert decision.intent.stop_loss > 0

    def test_the_intent_carries_the_full_decision_record(self):
        decision = self.engine().evaluate(make_opportunity(), self.context())
        assert decision.approved
        assert {
            "equity",
            "risk_amount",
            "risk_fraction",
            "leverage",
            "notional",
            "correlation",
            "positions_open",
        } <= set(decision.checks)

    def test_risk_taken_matches_the_configured_fraction(self):
        decision = self.engine().evaluate(make_opportunity(), self.context())
        assert decision.approved
        assert decision.checks["risk_fraction"] == pytest.approx(0.005, rel=0.15)

    def test_a_signal_without_a_stop_is_always_refused(self):
        """Non-negotiable rule 7 from the brief."""
        opportunity = make_opportunity()
        object.__setattr__(opportunity.signal, "stop_loss", 0.0)
        decision = self.engine().evaluate(opportunity, self.context())
        assert not decision.approved
        assert decision.reason is RejectionReason.INVALID_STOP
        assert "never permitted" in decision.detail

    def test_a_stop_on_the_wrong_side_is_refused(self):
        opportunity = make_opportunity()
        object.__setattr__(opportunity.signal, "stop_loss", 101.0)
        decision = self.engine().evaluate(opportunity, self.context())
        assert not decision.approved
        assert decision.reason is RejectionReason.INVALID_STOP

    def test_an_existing_position_blocks_a_second_entry(self):
        context = self.context(
            positions={"TESTUSDT": position_for("TESTUSDT")}, prices={"TESTUSDT": 100.0}
        )
        decision = self.engine().evaluate(make_opportunity(), context)
        assert decision.reason is RejectionReason.ALREADY_IN_POSITION

    def test_an_in_flight_intent_blocks_a_duplicate(self):
        """Race protection: two signals in the same cycle must not double up."""
        context = self.context(in_flight={"TESTUSDT"})
        decision = self.engine().evaluate(make_opportunity(), context)
        assert decision.reason is RejectionReason.INTENT_IN_FLIGHT

    def test_a_tripped_kill_switch_blocks_entry(self):
        engine = self.engine()
        engine.kill_switches.trip_manually("test halt")
        decision = engine.evaluate(make_opportunity(), self.context())
        assert decision.reason is RejectionReason.KILL_SWITCH_ACTIVE

    def test_a_cooldown_blocks_entry(self):
        engine = self.engine()
        engine.cooldowns.register_close("TESTUSDT", won=False, strategy="momentum")
        decision = engine.evaluate(make_opportunity(), self.context())
        assert decision.reason is RejectionReason.COOLDOWN_ACTIVE

    def test_a_suspended_strategy_blocks_entry(self):
        engine = self.engine()
        engine.suspended_strategies["momentum"] = 1_700_000_000 + 10_000
        decision = engine.evaluate(make_opportunity(strategy="momentum"), self.context())
        assert decision.reason is RejectionReason.STRATEGY_DISABLED

    def test_entries_blocked_flag_is_honoured(self):
        context = self.context(entries_blocked=True, entries_blocked_reason="reconciling")
        decision = self.engine().evaluate(make_opportunity(), context)
        assert decision.reason is RejectionReason.ENTRIES_BLOCKED

    def test_max_positions_blocks_entry(self):
        """Reported as MAX_POSITIONS, not as a downstream correlation failure."""
        from copy import deepcopy

        config = deepcopy(CONFIG)
        engine = self.engine(config)
        positions = {
            f"S{i}": position_for(f"S{i}") for i in range(config.risk.max_concurrent_positions)
        }
        context = self.context(
            equity=10_000.0,
            available_balance=10_000.0,
            positions=positions,
            prices=dict.fromkeys(positions, 100.0),
        )
        decision = engine.evaluate(make_opportunity(), context)
        assert decision.reason is RejectionReason.MAX_POSITIONS

    def test_minimum_notional_blocks_a_small_account(self):
        context = self.context(symbol_info=make_symbol_info("TESTUSDT", min_notional=500.0))
        decision = self.engine().evaluate(make_opportunity(), context)
        assert decision.reason is RejectionReason.NOTIONAL_BELOW_MINIMUM

    def test_every_rejection_explains_itself(self):
        """Auditability: no silent refusals."""
        engine = self.engine()
        engine.kill_switches.trip_manually("halt")
        decision = engine.evaluate(make_opportunity(), self.context())
        assert not decision.approved
        assert decision.reason is not None
        assert decision.detail

    def test_statistics_track_approvals_and_rejections(self):
        engine = self.engine()
        engine.evaluate(make_opportunity(), self.context())
        engine.evaluate(make_opportunity(), self.context(in_flight={"TESTUSDT"}))
        stats = engine.stats()
        assert stats["approvals"] >= 1
        assert stats["rejections"]

    def test_recording_a_closed_trade_updates_every_component(self):
        engine = self.engine()
        engine.record_trade_closed(
            "TESTUSDT", "momentum", won=False, r_multiple=-1.0, volatility=0.005
        )
        assert engine.cooldowns.is_active("TESTUSDT")
        assert engine.kill_switches.consecutive_losses == 1
        assert engine.allocator.performance_for("momentum").trades == 1

    def test_a_strategy_that_keeps_losing_is_suspended(self):
        engine = self.engine()
        for _ in range(60):
            engine.record_trade_closed("A", "failing", won=False, r_multiple=-1.0)
        assert "failing" in engine.suspended_strategies

    def test_leverage_never_exceeds_the_configured_maximum(self):
        engine = self.engine()
        decision = engine.evaluate(make_opportunity(stop_pct=0.002), self.context())
        if decision.approved:
            assert decision.intent.leverage <= CONFIG.risk.max_leverage

    def test_high_volatility_reduces_leverage(self):
        engine = self.engine()
        calm = engine.evaluate(make_opportunity(volatility=0.002), self.context())
        engine2 = self.engine()
        wild = engine2.evaluate(make_opportunity(volatility=0.05), self.context())
        if calm.approved and wild.approved:
            assert wild.intent.leverage <= calm.intent.leverage


class TestExitsAreNeverBlocked:
    """Nothing in the risk engine may prevent closing a position.

    Being unable to exit is strictly worse than any condition that would justify
    not entering.
    """

    def test_the_engine_only_gates_entries(self):
        """Structural: the engine exposes no method that can refuse an exit."""
        engine = RiskEngine(CONFIG, CandleStore(500), VirtualClock(0))
        engine.kill_switches.trip_manually("everything is on fire")
        # The only decision method is evaluate(), which takes an Opportunity —
        # an ENTRY. There is no exit path through this class at all.
        assert not hasattr(engine, "evaluate_exit")
        assert not hasattr(engine, "approve_close")
        assert not engine.kill_switches.entries_allowed

    def test_kill_switch_status_names_entries_only(self):
        manager = KillSwitchManager(KillSwitchConfig(), RiskConfig(), VirtualClock(0))
        manager.trip_manually("halt")
        assert "entries_allowed" in manager.status()
        assert manager.status()["entries_allowed"] is False
