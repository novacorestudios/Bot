"""Exchange-facing arithmetic.

These tests exist because getting rounding wrong produces Binance error -1111
intermittently, in production, on a subset of symbols — the worst kind of bug.
"""

from __future__ import annotations

import pytest

from tradebot.core.mathutil import (
    band_score,
    bps,
    clamp,
    decimals_for_step,
    format_decimal,
    from_bps,
    normalise_score,
    pct_change,
    round_price,
    round_quantity,
    round_to_step,
    safe_div,
)


class TestStepRounding:
    @pytest.mark.parametrize(
        ("value", "step", "expected"),
        [
            (0.1 + 0.2, 0.001, 0.3),  # the classic float artefact
            (1.005, 0.01, 1.0),  # rounds DOWN, never up
            (0.0009, 0.001, 0.0),  # below one step -> zero
            (123.456789, 0.001, 123.456),
            (7.0, 1.0, 7.0),
            (2.9999999, 1.0, 2.0),
        ],
    )
    def test_quantity_rounds_down_exactly(self, value, step, expected):
        assert round_quantity(value, step) == pytest.approx(expected, abs=1e-12)

    def test_quantity_never_rounds_up(self):
        """Rounding a quantity up could breach a risk limit; rounding down cannot."""
        for raw in (0.1234, 5.6789, 0.0019, 99.99999):
            assert round_quantity(raw, 0.001) <= raw + 1e-12

    def test_result_is_an_exact_multiple_of_the_step(self):
        from decimal import Decimal

        for step in (0.001, 0.01, 0.1, 1.0, 0.00001):
            out = round_quantity(3.14159265, step)
            remainder = Decimal(str(out)) % Decimal(str(step))
            assert remainder == 0, f"{out} is not a multiple of {step}"

    def test_zero_step_is_a_no_op(self):
        assert round_to_step(1.2345, 0) == 1.2345

    def test_price_rounds_to_nearest_tick(self):
        assert round_price(25436.126, 0.1) == pytest.approx(25436.1)
        assert round_price(25436.16, 0.1) == pytest.approx(25436.2)

    def test_price_rounding_modes(self):
        assert round_price(100.007, 0.01, "down") == pytest.approx(100.00)
        assert round_price(100.001, 0.01, "up") == pytest.approx(100.01)


class TestFormatting:
    def test_never_emits_scientific_notation(self):
        """Binance rejects '1e-05'."""
        text = format_decimal(0.00001, 8)
        assert "e" not in text.lower()
        assert text == "0.00001"

    def test_trailing_zeros_are_stripped(self):
        assert format_decimal(1.5000, 4) == "1.5"
        assert format_decimal(2.0, 3) == "2"

    def test_truncates_rather_than_rounding_up(self):
        assert format_decimal(1.99999, 2) == "1.99"

    @pytest.mark.parametrize(
        ("step", "expected"), [(0.001, 3), (1.0, 0), (0.1, 1), (0.00000001, 8)]
    )
    def test_decimals_for_step(self, step, expected):
        assert decimals_for_step(step) == expected


class TestScoring:
    def test_band_score_is_full_inside_the_band(self):
        assert band_score(0.005, 0.0025, 0.015) == 100.0

    def test_band_score_decays_outside_and_is_clamped(self):
        assert band_score(0.0020, 0.0025, 0.015) < 100.0
        assert band_score(0.5, 0.0025, 0.015) == 0.0
        assert 0.0 <= band_score(0.0001, 0.0025, 0.015) <= 100.0

    def test_normalise_clamps_to_zero_hundred(self):
        assert normalise_score(-5, 0, 10) == 0.0
        assert normalise_score(50, 0, 10) == 100.0

    def test_normalise_inverted_prefers_lower_values(self):
        assert normalise_score(1, 0, 10, invert=True) > normalise_score(9, 0, 10, invert=True)

    def test_degenerate_range_is_neutral(self):
        assert normalise_score(5, 5, 5) == 50.0


class TestGuards:
    def test_safe_div_by_zero_returns_default(self):
        assert safe_div(1, 0, -1) == -1

    def test_safe_div_rejects_non_finite_results(self):
        assert safe_div(float("inf"), float("inf"), 0.0) == 0.0

    def test_clamp_tolerates_reversed_bounds(self):
        assert clamp(5, 10, 0) == 5

    def test_pct_change_handles_zero_base(self):
        assert pct_change(0, 5) == 0.0

    def test_bps_round_trip(self):
        assert from_bps(bps(0.0004)) == pytest.approx(0.0004)
