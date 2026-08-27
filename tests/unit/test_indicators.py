"""Indicator correctness.

Every assertion here is either a hand-computed value (with the arithmetic shown)
or a structural property. An indicator that is subtly wrong produces a strategy
that is subtly wrong, which is the hardest kind of loss to diagnose.
"""

from __future__ import annotations

import numpy as np
import pytest

from tradebot.market.indicators import (
    adx,
    atr,
    atr_percent,
    bollinger,
    bollinger_bandwidth,
    donchian,
    ema,
    last_valid,
    linear_slope,
    macd,
    momentum,
    obv,
    price_levels,
    realised_volatility,
    roc,
    rsi,
    sma,
    structure_score,
    swing_highs,
    swing_lows,
    true_range,
    volume_ratio,
    volume_zscore,
    vwap,
    vwap_bands,
    wma,
)

# Wilder's classic worked example (15 closes -> 14 changes).
WILDER = np.array(
    [
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.42,
        45.84,
        46.08,
        45.89,
        46.03,
        45.61,
        46.28,
        46.28,
    ],
    dtype=float,
)


class TestMovingAverages:
    def test_sma_hand_computed(self):
        values = np.array([1.0, 2, 3, 4, 5, 6])
        # mean(4,5,6) = 5
        assert sma(values, 3)[-1] == pytest.approx(5.0)
        # mean(1,2,3) = 2 at the first valid index
        assert sma(values, 3)[2] == pytest.approx(2.0)

    def test_sma_warmup_is_nan(self):
        out = sma(np.arange(10, dtype=float), 5)
        assert np.isnan(out[:4]).all()
        assert np.isfinite(out[4:]).all()

    def test_sma_of_a_constant_series_is_that_constant(self):
        assert sma(np.full(20, 7.0), 5)[-1] == pytest.approx(7.0)

    def test_ema_seed_is_the_sma(self):
        values = np.array([1.0, 2, 3, 4, 5, 6, 7])
        # First EMA value is the SMA of the first 3: (1+2+3)/3 = 2
        assert ema(values, 3)[2] == pytest.approx(2.0)

    def test_ema_recursion_hand_computed(self):
        values = np.array([1.0, 2, 3, 4])
        # seed 2.0; alpha = 2/4 = 0.5; next = 4*0.5 + 2*0.5 = 3.0
        assert ema(values, 3)[3] == pytest.approx(3.0)

    def test_ema_reacts_faster_than_sma(self):
        values = np.concatenate([np.full(20, 100.0), np.full(5, 110.0)])
        assert ema(values, 10)[-1] > sma(values, 10)[-1]

    def test_wma_weights_recent_bars_more(self):
        values = np.array([1.0, 2, 3])
        # (1*1 + 2*2 + 3*3) / 6 = 14/6
        assert wma(values, 3)[-1] == pytest.approx(14 / 6)

    def test_period_longer_than_series_returns_all_nan(self):
        assert np.isnan(sma(np.arange(3, dtype=float), 10)).all()
        assert np.isnan(ema(np.arange(3, dtype=float), 10)).all()


class TestRSI:
    def test_first_value_matches_hand_arithmetic(self):
        """gains sum 3.34/14 = 0.238571; losses 1.40/14 = 0.10; RS = 2.385714."""
        expected = 100 - 100 / (1 + (3.34 / 14) / (1.40 / 14))
        assert rsi(WILDER, 14)[14] == pytest.approx(expected, abs=1e-6)
        assert rsi(WILDER, 14)[14] == pytest.approx(70.46, abs=0.01)

    def test_monotonic_rise_saturates_at_one_hundred(self):
        assert rsi(np.arange(1, 40, dtype=float), 14)[-1] == pytest.approx(100.0)

    def test_monotonic_fall_approaches_zero(self):
        assert rsi(np.arange(40, 1, -1, dtype=float), 14)[-1] == pytest.approx(0.0)

    def test_flat_series_is_neutral(self):
        assert rsi(np.full(40, 50.0), 14)[-1] == pytest.approx(50.0)

    def test_bounded_zero_to_hundred(self):
        rng = np.random.default_rng(42)
        values = 100 + np.cumsum(rng.normal(0, 1, 200))
        out = rsi(values, 14)
        finite = out[np.isfinite(out)]
        assert finite.min() >= 0.0
        assert finite.max() <= 100.0

    def test_warmup_region_is_nan(self):
        assert np.isnan(rsi(WILDER, 14)[:14]).all()


class TestATR:
    def test_true_range_uses_the_previous_close(self):
        high = np.array([10.0, 12.0])
        low = np.array([9.0, 11.0])
        close = np.array([9.5, 11.5])
        tr = true_range(high, low, close)
        assert tr[0] == pytest.approx(1.0)  # no previous close
        # max(12-11, |12-9.5|, |11-9.5|) = 2.5 — the gap, not the bar range
        assert tr[1] == pytest.approx(2.5)

    def test_atr_of_constant_range_equals_that_range(self):
        n = 40
        close = np.full(n, 100.0)
        assert atr(close + 1.0, close - 1.0, close, 14)[-1] == pytest.approx(2.0)

    def test_atr_percent_is_scale_invariant(self):
        """The whole point: a 100k symbol and a 0.001 symbol become comparable."""
        n = 60
        for price in (100.0, 0.001, 95_000.0):
            close = np.full(n, price)
            pct = atr_percent(close * 1.01, close * 0.99, close, 14)[-1]
            assert pct == pytest.approx(0.02, rel=1e-6)

    def test_atr_rises_with_volatility(self):
        n = 60
        close = np.full(n, 100.0)
        calm = atr(close + 0.5, close - 0.5, close, 14)[-1]
        wild = atr(close + 5.0, close - 5.0, close, 14)[-1]
        assert wild > calm


class TestADX:
    def test_strong_uptrend_scores_high_with_plus_di_dominant(self):
        close = np.array([100 * (1.01**i) for i in range(60)])
        a, pdi, mdi = adx(close * 1.005, close * 0.995, close, 14)
        assert last_valid(a) > 25
        assert last_valid(pdi) > last_valid(mdi)

    def test_strong_downtrend_has_minus_di_dominant(self):
        close = np.array([100 * (0.99**i) for i in range(60)])
        a, pdi, mdi = adx(close * 1.005, close * 0.995, close, 14)
        assert last_valid(a) > 25
        assert last_valid(mdi) > last_valid(pdi)

    def test_choppy_market_scores_low(self):
        close = np.array([100 + (2 if i % 2 else -2) for i in range(80)], dtype=float)
        a, _, _ = adx(close + 0.5, close - 0.5, close, 14)
        assert last_valid(a, 0.0) < 25

    def test_short_series_returns_nan_rather_than_raising(self):
        a, _, _ = adx(np.ones(5), np.ones(5), np.ones(5), 14)
        assert np.isnan(a).all()


class TestBollinger:
    def test_bands_bracket_the_middle(self):
        rng = np.random.default_rng(1)
        values = 100 + np.cumsum(rng.normal(0, 1, 100))
        upper, middle, lower = bollinger(values, 20, 2.0)
        assert (upper[19:] >= middle[19:]).all()
        assert (middle[19:] >= lower[19:]).all()

    def test_constant_series_collapses_the_bands(self):
        upper, middle, lower = bollinger(np.full(40, 100.0), 20, 2.0)
        assert upper[-1] == pytest.approx(lower[-1])
        assert middle[-1] == pytest.approx(100.0)

    def test_bandwidth_detects_a_squeeze(self):
        calm = np.concatenate([np.full(40, 100.0), 100 + np.arange(20) * 2.0])
        bw = bollinger_bandwidth(calm, 20, 2.0)
        assert bw[39] < bw[-1]

    def test_two_std_bands_are_wider_than_one(self):
        rng = np.random.default_rng(3)
        values = 100 + np.cumsum(rng.normal(0, 1, 60))
        wide, _, _ = bollinger(values, 20, 2.0)
        narrow, _, _ = bollinger(values, 20, 1.0)
        assert wide[-1] > narrow[-1]


class TestDonchian:
    def test_excludes_the_current_bar(self):
        """Including the current bar makes every breakout test trivially true."""
        high = np.array([1.0, 2, 3, 4, 100])
        low = np.array([1.0, 2, 3, 4, 5])
        upper, _ = donchian(high, low, 4)
        assert upper[4] == pytest.approx(4.0)  # not 100

    def test_lower_band_tracks_the_minimum(self):
        high = np.array([5.0, 5, 5, 5, 5])
        low = np.array([3.0, 2, 4, 4, 1])
        _, lower = donchian(high, low, 4)
        assert lower[4] == pytest.approx(2.0)


class TestVWAP:
    def test_equals_price_when_price_is_constant(self):
        n = 20
        price = np.full(n, 100.0)
        assert vwap(price, price, price, np.full(n, 10.0))[-1] == pytest.approx(100.0)

    def test_is_pulled_toward_the_high_volume_price(self):
        high = low = close = np.array([100.0, 200.0])
        volume = np.array([1.0, 99.0])
        assert vwap(high, low, close, volume)[-1] > 190.0

    def test_session_reset_discards_earlier_bars(self):
        high = low = close = np.array([100.0, 100.0, 200.0])
        volume = np.array([10.0, 10.0, 10.0])
        resets = np.array([False, False, True])
        assert vwap(high, low, close, volume, resets)[-1] == pytest.approx(200.0)

    def test_bands_bracket_the_vwap(self):
        rng = np.random.default_rng(5)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 60))
        volume = rng.uniform(5, 20, 60)
        upper, centre, lower = vwap_bands(close, close, close, volume, 2.0)
        assert upper[-1] >= centre[-1] >= lower[-1]


class TestVolume:
    def test_ratio_detects_a_spike(self):
        volume = np.concatenate([np.full(30, 100.0), np.array([500.0])])
        assert volume_ratio(volume, 20)[-1] == pytest.approx(5.0)

    def test_ratio_of_steady_volume_is_one(self):
        assert volume_ratio(np.full(40, 100.0), 20)[-1] == pytest.approx(1.0)

    def test_zscore_flags_an_anomaly(self):
        rng = np.random.default_rng(9)
        volume = np.concatenate([rng.normal(100, 5, 60), np.array([300.0])])
        assert volume_zscore(volume, 30)[-1] > 5

    def test_zscore_of_constant_volume_stays_nan(self):
        """Zero variance must not divide by zero."""
        assert np.isnan(volume_zscore(np.full(50, 100.0), 30)[-1])

    def test_obv_accumulates_with_direction(self):
        close = np.array([100.0, 101, 100, 102])
        volume = np.array([10.0, 20, 30, 40])
        # 0, +20, -30, +40
        assert obv(close, volume)[-1] == pytest.approx(30.0)


class TestStructure:
    def test_swing_high_detection(self):
        high = np.array([1.0, 2, 5, 2, 1, 2, 1])
        assert bool(swing_highs(high, 2, 2)[2])

    def test_swing_low_detection(self):
        low = np.array([5.0, 4, 1, 4, 5, 4, 5])
        assert bool(swing_lows(low, 2, 2)[2])

    def test_unconfirmed_pivots_at_the_edge_are_not_marked(self):
        """A pivot needs `right` bars after it; marking it early is look-ahead."""
        high = np.array([1.0, 2, 5])
        assert not swing_highs(high, 2, 2).any()

    def test_price_levels_cluster_repeated_touches(self):
        # A level at ~110 touched three times.
        high = np.array(
            [100, 105, 110, 105, 100, 105, 110.2, 105, 100, 105, 109.9, 105, 100, 102], dtype=float
        )
        low = high - 5
        levels = price_levels(high, low, atr_value=2.0, lookback=60, cluster_atr=0.4, min_touches=2)
        assert levels
        assert any(abs(price - 110) < 1.0 and touches >= 2 for price, touches in levels)

    def test_price_levels_without_atr_returns_empty(self):
        assert price_levels(np.ones(10), np.ones(10), atr_value=0.0) == []

    def test_structure_score_signs(self):
        rising = np.arange(100, 160, dtype=float)
        falling = np.arange(160, 100, -1, dtype=float)
        assert structure_score(rising, 50) == pytest.approx(1.0)
        assert structure_score(falling, 50) == pytest.approx(-1.0)

    def test_structure_score_of_short_series_is_neutral(self):
        assert structure_score(np.arange(10, dtype=float), 50) == 0.0


class TestMomentumIndicators:
    def test_roc_is_a_fraction_not_a_percentage(self):
        values = np.array([100.0] * 10 + [110.0])
        assert roc(values, 10)[-1] == pytest.approx(0.10)

    def test_roc_handles_a_zero_base(self):
        values = np.concatenate([np.zeros(5), np.full(6, 10.0)])
        assert np.isnan(roc(values, 5)[5])

    def test_momentum_is_an_absolute_difference(self):
        values = np.array([100.0] * 10 + [110.0])
        assert momentum(values, 10)[-1] == pytest.approx(10.0)

    def test_macd_histogram_is_line_minus_signal(self):
        rng = np.random.default_rng(11)
        values = 100 + np.cumsum(rng.normal(0, 1, 120))
        line, signal, hist = macd(values, 12, 26, 9)
        valid = np.isfinite(hist)
        assert np.allclose(hist[valid], (line - signal)[valid])

    def test_macd_is_positive_in_an_uptrend(self):
        values = np.array([100 * 1.005**i for i in range(120)])
        line, _, _ = macd(values, 12, 26, 9)
        assert last_valid(line) > 0

    def test_linear_slope_sign_follows_direction(self):
        up = np.arange(100, 160, dtype=float)
        down = np.arange(160, 100, -1, dtype=float)
        assert linear_slope(up, 20)[-1] > 0
        assert linear_slope(down, 20)[-1] < 0

    def test_linear_slope_of_a_flat_series_is_zero(self):
        assert linear_slope(np.full(40, 100.0), 20)[-1] == pytest.approx(0.0)

    def test_realised_volatility_orders_correctly(self):
        rng = np.random.default_rng(13)
        calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 100)))
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 100)))
        assert realised_volatility(wild, 20)[-1] > realised_volatility(calm, 20)[-1]


class TestHelpers:
    def test_last_valid_skips_trailing_nans(self):
        assert last_valid(np.array([1.0, 2.0, np.nan])) == 2.0

    def test_last_valid_of_all_nan_returns_the_default(self):
        assert last_valid(np.array([np.nan, np.nan]), default=-1.0) == -1.0

    def test_empty_input_is_handled(self):
        assert np.isnan(last_valid(np.array([])))

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError):
            sma(np.ones((2, 2)), 2)
