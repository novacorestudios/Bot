"""Capital preservation modes.

The arithmetic that motivates the whole module: on 75 USDT, a 10% drawdown is
7.50 USDT — fifteen trades' worth of risk at 0.5% — and recovering it needs an
11% gain. Losses compound against you faster than gains compound for you, so
the correct response to a losing streak is to risk less.

Two invariants are asserted repeatedly because breaking either is dangerous:

* escalation is immediate, de-escalation must be earned;
* no mode, HALTED included, blocks an exit.
"""

from __future__ import annotations

import inspect

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.config import PreservationConfig
from tradebot.risk import preservation as preservation_module
from tradebot.risk.preservation import (
    LIMITS,
    CapitalPreservation,
    PreservationMode,
)

CONFIG = PreservationConfig()


@pytest.fixture
def preservation(clock: VirtualClock) -> CapitalPreservation:
    return CapitalPreservation(CONFIG, clock)


class TestThresholds:
    def test_a_healthy_account_is_normal(self, preservation: CapitalPreservation) -> None:
        state = preservation.evaluate(drawdown=0.0, daily_loss=0.0, consecutive_losses=0)
        assert state.mode is PreservationMode.NORMAL
        assert preservation.risk_multiplier == 1.0
        assert preservation.entries_allowed is True

    @pytest.mark.parametrize(
        ("drawdown", "expected"),
        [
            (0.029, PreservationMode.NORMAL),
            (0.030, PreservationMode.CAUTIOUS),
            (0.059, PreservationMode.CAUTIOUS),
            (0.060, PreservationMode.DEFENSIVE),
            (0.099, PreservationMode.DEFENSIVE),
            (0.100, PreservationMode.HALTED),
        ],
    )
    def test_drawdown_bands(
        self, preservation: CapitalPreservation, drawdown: float, expected: PreservationMode
    ) -> None:
        state = preservation.evaluate(drawdown, daily_loss=0.0, consecutive_losses=0)
        assert state.mode is expected

    def test_the_daily_loss_limit_halts(self, preservation: CapitalPreservation) -> None:
        state = preservation.evaluate(drawdown=0.0, daily_loss=0.02, consecutive_losses=0)
        assert state.mode is PreservationMode.HALTED
        assert preservation.entries_allowed is False
        assert "daily loss" in state.reason

    @pytest.mark.parametrize(
        ("losses", "expected"),
        [
            (2, PreservationMode.NORMAL),
            (3, PreservationMode.CAUTIOUS),
            (4, PreservationMode.DEFENSIVE),
        ],
    )
    def test_a_losing_streak_tightens_without_any_drawdown(
        self, preservation: CapitalPreservation, losses: int, expected: PreservationMode
    ) -> None:
        """Small, frequent losses do not show as drawdown quickly, but they are
        exactly the pattern that says the current conditions do not suit this
        system."""
        state = preservation.evaluate(drawdown=0.0, daily_loss=0.0, consecutive_losses=losses)
        assert state.mode is expected

    def test_the_worst_trigger_wins(self, preservation: CapitalPreservation) -> None:
        state = preservation.evaluate(drawdown=0.04, daily_loss=0.02, consecutive_losses=1)
        assert state.mode is PreservationMode.HALTED


class TestModeLimits:
    def test_each_mode_is_strictly_tighter_than_the_one_before(self) -> None:
        ordered = [
            PreservationMode.NORMAL,
            PreservationMode.CAUTIOUS,
            PreservationMode.DEFENSIVE,
            PreservationMode.HALTED,
        ]
        multipliers = [LIMITS[m].risk_multiplier for m in ordered]
        assert multipliers == sorted(multipliers, reverse=True)
        assert multipliers[0] == 1.0
        assert multipliers[-1] == 0.0

    def test_risk_is_scaled_down_never_up(self) -> None:
        assert all(LIMITS[m].risk_multiplier <= 1.0 for m in PreservationMode)

    def test_defensive_allows_two_positions_and_only_high_quality_ones(
        self, preservation: CapitalPreservation
    ) -> None:
        preservation.evaluate(drawdown=0.07, daily_loss=0.0, consecutive_losses=0)
        assert preservation.mode is PreservationMode.DEFENSIVE
        assert preservation.max_positions(4) == 2
        assert preservation.min_opportunity_score(70.0) == 78.0

    def test_halted_permits_no_position_at_all(self, preservation: CapitalPreservation) -> None:
        preservation.evaluate(drawdown=0.15, daily_loss=0.0, consecutive_losses=0)
        assert preservation.max_positions(4) == 0
        assert preservation.risk_multiplier == 0.0

    def test_the_score_floor_never_lowers_a_stricter_configured_minimum(
        self, preservation: CapitalPreservation
    ) -> None:
        preservation.evaluate(drawdown=0.07, daily_loss=0.0, consecutive_losses=0)
        assert preservation.min_opportunity_score(92.0) == 92.0

    def test_normal_mode_changes_nothing(self, preservation: CapitalPreservation) -> None:
        preservation.evaluate(0.0, 0.0, 0)
        assert preservation.max_positions(4) == 4
        assert preservation.min_opportunity_score(70.0) == 70.0


class TestHysteresisAndDwellTime:
    def test_escalation_is_immediate(
        self, preservation: CapitalPreservation, clock: VirtualClock
    ) -> None:
        preservation.evaluate(0.0, 0.0, 0)
        state = preservation.evaluate(0.035, 0.0, 0)
        assert state.mode is PreservationMode.CAUTIOUS  # no dwell time required

    def test_relaxing_needs_the_minimum_dwell_time(
        self, preservation: CapitalPreservation, clock: VirtualClock
    ) -> None:
        preservation.evaluate(0.035, 0.0, 0)
        assert preservation.mode is PreservationMode.CAUTIOUS

        clock.advance(60)
        preservation.evaluate(0.0, 0.0, 0)
        assert preservation.mode is PreservationMode.CAUTIOUS  # too soon

        clock.advance(CONFIG.min_mode_duration_sec)
        preservation.evaluate(0.0, 0.0, 0)
        assert preservation.mode is PreservationMode.NORMAL

    def test_a_drawdown_hovering_on_the_line_does_not_flip_the_mode(
        self, preservation: CapitalPreservation, clock: VirtualClock
    ) -> None:
        """Without the hysteresis band, an account oscillating around 3.0%
        would toggle modes every cycle and get full size back in exactly the
        conditions that shrank it."""
        preservation.evaluate(0.031, 0.0, 0)
        assert preservation.mode is PreservationMode.CAUTIOUS

        clock.advance(CONFIG.min_mode_duration_sec + 1)
        preservation.evaluate(0.025, 0.0, 0)  # recovered, but inside the band
        assert preservation.mode is PreservationMode.CAUTIOUS

        preservation.evaluate(0.019, 0.0, 0)  # below 0.03 - 0.01
        assert preservation.mode is PreservationMode.NORMAL

    def test_halted_does_not_relax_on_a_recovering_drawdown(
        self, preservation: CapitalPreservation, clock: VirtualClock
    ) -> None:
        """A partly recovered drawdown is not evidence that its cause is gone."""
        preservation.evaluate(0.0, 0.02, 0)
        assert preservation.mode is PreservationMode.HALTED

        clock.advance(CONFIG.min_mode_duration_sec * 10)
        preservation.evaluate(0.0, 0.0, 0)
        assert preservation.mode is PreservationMode.HALTED

    def test_an_explicit_reset_ends_a_halt(
        self, preservation: CapitalPreservation, clock: VirtualClock
    ) -> None:
        preservation.evaluate(0.0, 0.02, 0)
        preservation.reset("new trading day")
        assert preservation.mode is PreservationMode.NORMAL
        assert preservation.entries_allowed is True

    def test_it_tightens_further_even_inside_the_dwell_window(
        self, preservation: CapitalPreservation, clock: VirtualClock
    ) -> None:
        preservation.evaluate(0.035, 0.0, 0)
        clock.advance(5)
        state = preservation.evaluate(0.07, 0.0, 0)
        assert state.mode is PreservationMode.DEFENSIVE


class TestCapitalReserve:
    def test_a_fraction_of_equity_is_never_deployed(
        self, preservation: CapitalPreservation
    ) -> None:
        """The reserve pays funding, fees and adverse margin moves on positions
        already open. An account that deploys every cent has no buffer between
        a normal adverse move and a liquidation."""
        assert preservation.reserve_for(75.0) == pytest.approx(7.5)
        assert preservation.deployable(75.0) == pytest.approx(67.5)

    def test_it_is_reported_in_the_state(self, preservation: CapitalPreservation) -> None:
        state = preservation.evaluate(0.0, 0.0, 0, equity=75.0)
        assert state.capital_reserve == pytest.approx(7.5)

    def test_zero_equity_reserves_nothing(self, preservation: CapitalPreservation) -> None:
        assert preservation.reserve_for(0.0) == 0.0
        assert preservation.deployable(0.0) == 0.0


class TestExitsAreNeverGated:
    def test_no_mode_exposes_anything_that_could_block_an_exit(self) -> None:
        """Structural: the module offers entry-side limits only. A preservation
        mode that could block a close would turn a bad day into a catastrophic
        one."""
        source = inspect.getsource(preservation_module)
        for forbidden in ("block_exit", "allow_exit", "exits_allowed", "close_allowed"):
            assert forbidden not in source

    def test_halted_still_reports_entries_only(self) -> None:
        assert PreservationMode.HALTED.allows_entries is False
        assert not hasattr(PreservationMode.HALTED, "allows_exits")


class TestConfigValidation:
    def test_thresholds_must_increase(self) -> None:
        with pytest.raises(ValueError, match="must increase"):
            PreservationConfig(cautious_drawdown=0.08, defensive_drawdown=0.06)

    def test_the_hysteresis_band_must_leave_room_to_recover(self) -> None:
        """A band wider than the first threshold makes CAUTIOUS a trap."""
        with pytest.raises(ValueError, match="could never relax"):
            PreservationConfig(cautious_drawdown=0.03, recovery_hysteresis=0.03)

    def test_loss_streak_thresholds_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="cautious_consecutive_losses"):
            PreservationConfig(cautious_consecutive_losses=5, defensive_consecutive_losses=3)


class TestBookkeeping:
    def test_transitions_and_time_in_mode_are_tracked(
        self, preservation: CapitalPreservation, clock: VirtualClock
    ) -> None:
        preservation.evaluate(0.0, 0.0, 0)
        clock.advance(100)
        preservation.evaluate(0.035, 0.0, 0)
        stats = preservation.stats()
        assert stats["transitions"] == 1
        assert stats["mode"] == "CAUTIOUS"
        assert stats["time_in_mode"]["NORMAL"] == pytest.approx(100.0)
