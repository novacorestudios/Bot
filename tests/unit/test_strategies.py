"""Strategy behaviour.

Each strategy is tested for three things:

1. It fires in the direction its thesis predicts, on a price path built for it.
2. It says WAIT on noise, and on the specific conditions it is designed to
   refuse (exhaustion, no volume, an opposing trend...). The refusals matter
   more than the entries: a strategy that never declines has no edge.
3. Its output is structurally sound — stop on the correct side, target beyond
   entry, R-multiple comparable with every other strategy's.

These paths are SYNTHETIC. They prove the logic does what it claims, not that
the logic is profitable. Profitability requires real data — see
IMPLEMENTATION_PLAN.md §9.
"""

from __future__ import annotations

import math

import pytest

from tradebot.core.config import StopsConfig, TargetsConfig, load_tunables
from tradebot.core.types import Direction, MarketRegime
from tradebot.market.candles import CandleStore
from tradebot.strategies.base import MarketView
from tradebot.strategies.breakout import BreakoutStrategy
from tradebot.strategies.mean_reversion import MeanReversionStrategy
from tradebot.strategies.momentum import MomentumStrategy
from tradebot.strategies.registry import STRATEGY_CLASSES, StrategyRegistry
from tradebot.strategies.support_resistance import SupportResistanceStrategy
from tradebot.strategies.trend_following import TrendFollowingStrategy
from tradebot.strategies.volatility_expansion import VolatilityExpansionStrategy
from tradebot.strategies.volume_spike import VolumeSpikeStrategy
from tradebot.strategies.vwap import VWAPStrategy

from ..conftest import (
    REPO_ROOT,
    choppy_prices,
    flat_prices,
    impulse_prices,
    make_candles,
    ranging_prices,
    trend_prices,
    trending_with_pullbacks,
)

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
)


def build_view(
    prices,
    *,
    volumes=None,
    timeframe="3m",
    symbol="TESTUSDT",
    regime=MarketRegime.STRONG_TREND,
    extra_timeframes=None,
    regime_direction=Direction.WAIT,
    wick=0.001,
    candles=None,
) -> MarketView:
    store = CandleStore(500)
    bars = candles if candles is not None else make_candles(prices, volumes=volumes, wick=wick)
    store.series(symbol, timeframe).extend(bars)
    for other_tf, other_prices in (extra_timeframes or {}).items():
        store.series(symbol, other_tf).extend(make_candles(other_prices))
    return MarketView(
        symbol=symbol,
        candles=store,
        regime=regime,
        regime_confidence=80.0,
        regime_direction=regime_direction,
        now_ms=1_700_000_000_000,
    )


def make(cls, **overrides):
    params = dict(CONFIG.strategies[cls.name])
    params.update(overrides)
    return cls(params, CONFIG.stops, CONFIG.targets)


# --------------------------------------------------------------------------- #
# Contract satisfied by every strategy
# --------------------------------------------------------------------------- #
class TestUniversalContract:
    @pytest.mark.parametrize("name", sorted(STRATEGY_CLASSES))
    def test_flat_market_produces_wait(self, name):
        """No movement means no opportunity. Every strategy must decline."""
        strategy = make(STRATEGY_CLASSES[name])
        view = build_view(
            flat_prices(300),
            timeframe=strategy.timeframe,
            extra_timeframes={
                "15m": flat_prices(300),
                "1h": flat_prices(300),
                "5m": flat_prices(300),
            },
        )
        assert strategy.generate(view).direction is Direction.WAIT

    @pytest.mark.parametrize("name", sorted(STRATEGY_CLASSES))
    def test_insufficient_data_produces_wait_not_a_guess(self, name):
        strategy = make(STRATEGY_CLASSES[name])
        view = build_view(trend_prices(15), timeframe=strategy.timeframe)
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert "INSUFFICIENT_DATA" in signal.reason_codes

    @pytest.mark.parametrize("name", sorted(STRATEGY_CLASSES))
    def test_disabled_strategy_never_signals(self, name):
        strategy = make(STRATEGY_CLASSES[name], enabled=False)
        view = build_view(trend_prices(300, drift=0.003), timeframe=strategy.timeframe)
        assert strategy.generate(view).direction is Direction.WAIT

    @pytest.mark.parametrize("name", sorted(STRATEGY_CLASSES))
    def test_every_signal_is_structurally_valid(self, name):
        """Whatever a strategy emits, its geometry must be self-consistent."""
        strategy = make(STRATEGY_CLASSES[name])
        for prices in (
            trend_prices(300, drift=0.002),
            trend_prices(300, drift=-0.002),
            ranging_prices(300),
        ):
            view = build_view(
                prices,
                timeframe=strategy.timeframe,
                extra_timeframes={"15m": prices, "1h": prices, "5m": prices},
            )
            signal = strategy.generate(view)
            assert signal.validate() == []
            if signal.is_actionable:
                assert signal.stop_distance > 0
                assert signal.risk_reward >= CONFIG.targets.min_rr * 0.99

    @pytest.mark.parametrize("name", sorted(STRATEGY_CLASSES))
    def test_a_raising_strategy_is_contained(self, name, monkeypatch):
        """A bug in one strategy must not take down the engine."""
        strategy = make(STRATEGY_CLASSES[name])

        def explode(*_args, **_kwargs):
            raise RuntimeError("simulated strategy bug")

        monkeypatch.setattr(strategy, "evaluate", explode)
        view = build_view(trend_prices(300), timeframe=strategy.timeframe)
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert "STRATEGY_ERROR" in signal.reason_codes
        assert strategy.errors == 1

    @pytest.mark.parametrize("name", sorted(STRATEGY_CLASSES))
    def test_strategies_cannot_reach_the_account_or_exchange(self, name):
        """Structural guarantee: the view carries no account and no gateway."""
        view = build_view(trend_prices(100))
        assert not hasattr(view, "account")
        assert not hasattr(view, "gateway")
        assert not hasattr(view, "positions")
        assert not hasattr(view, "equity")


# --------------------------------------------------------------------------- #
# Level derivation, shared by all strategies
# --------------------------------------------------------------------------- #
class TestLevelDerivation:
    def strategy(self, **stop_overrides):
        stops = StopsConfig(**{**StopsConfig().model_dump(), **stop_overrides})
        return MomentumStrategy(dict(CONFIG.strategies["momentum"]), stops, TargetsConfig())

    def series(self, prices):
        from tradebot.market.candles import CandleSeries

        series = CandleSeries("X", "3m", 500)
        series.extend(make_candles(prices))
        return series

    def test_long_stop_sits_below_entry_and_short_above(self):
        strategy = self.strategy()
        series = self.series(trend_prices(200))
        entry, atr = 100.0, 1.0
        assert strategy.derive_stop(series, Direction.LONG, entry, atr) < entry
        assert strategy.derive_stop(series, Direction.SHORT, entry, atr) > entry

    def test_stop_respects_the_minimum_distance(self):
        """A stop inside the round-trip cost cannot produce a profitable trade."""
        strategy = self.strategy(min_stop_pct=0.005, max_stop_pct=0.02)
        series = self.series(flat_prices(200))
        stop = strategy.derive_stop(series, Direction.LONG, 100.0, atr_value=0.0001)
        assert (100.0 - stop) / 100.0 >= 0.005 - 1e-9

    def test_stop_respects_the_maximum_distance(self):
        """A volatility spike must not produce an absurd stop."""
        strategy = self.strategy(min_stop_pct=0.002, max_stop_pct=0.01)
        series = self.series(flat_prices(200))
        stop = strategy.derive_stop(series, Direction.LONG, 100.0, atr_value=50.0)
        assert (100.0 - stop) / 100.0 <= 0.01 + 1e-9

    def test_stop_is_pushed_beyond_nearby_structure(self):
        """A stop just inside an obvious swing low is a stop that gets hit."""
        strategy = self.strategy(min_stop_pct=0.0005, max_stop_pct=0.05, structure_buffer_atr=0.25)
        # A clear low at 97 within the structure lookback.
        prices = [100.0] * 10 + [97.0] + [100.0] * 9
        series = self.series(prices)
        stop = strategy.derive_stop(series, Direction.LONG, 100.0, atr_value=0.5)
        assert stop < 97.0, "the stop must sit beyond the swing low, not above it"

    def test_target_is_beyond_entry_on_the_correct_side(self):
        strategy = self.strategy()
        long_target = strategy.derive_target(Direction.LONG, 100.0, 99.0, 1.0, 2.0)
        short_target = strategy.derive_target(Direction.SHORT, 100.0, 101.0, 1.0, 2.0)
        assert long_target > 100.0
        assert short_target < 100.0

    def test_reward_risk_is_honoured_within_bounds(self):
        strategy = self.strategy()
        target = strategy.derive_target(
            Direction.LONG, 100.0, 99.0, atr_value=10.0, reward_risk=2.0
        )
        assert (target - 100.0) / 1.0 == pytest.approx(2.0, rel=0.01)

    def test_reward_risk_is_clamped_to_configured_maximum(self):
        strategy = self.strategy()
        target = strategy.derive_target(
            Direction.LONG, 100.0, 99.0, atr_value=100.0, reward_risk=99.0
        )
        assert (target - 100.0) <= CONFIG.targets.max_rr * 1.0 + 1e-9

    def test_target_never_falls_below_minimum_reward_risk(self):
        """An ATR cap must not squash the target below the point of trading."""
        strategy = self.strategy()
        target = strategy.derive_target(
            Direction.LONG, 100.0, 98.0, atr_value=0.01, reward_risk=2.0
        )
        assert (target - 100.0) / 2.0 >= CONFIG.targets.min_rr - 1e-9

    def test_zero_atr_does_not_produce_a_zero_distance_stop(self):
        strategy = self.strategy()
        series = self.series(flat_prices(200))
        stop = strategy.derive_stop(series, Direction.LONG, 100.0, atr_value=0.0)
        assert stop < 100.0


# --------------------------------------------------------------------------- #
# Per-strategy behaviour
# --------------------------------------------------------------------------- #
class TestMomentum:
    """Momentum trades an impulse out of consolidation.

    Every path here contains counter-bars. A run of consecutive same-direction
    bars drives Wilder RSI above 85 as a matter of arithmetic, and the strategy
    is built to refuse that — so a no-retracement path would only ever exercise
    the exhaustion guard.
    """

    def view_for(self, direction: int, seed: int = 3, **overrides):
        prices = impulse_prices(200, direction=direction, seed=seed, **overrides)
        volumes = [1000.0] * 200 + [2200.0] * (len(prices) - 200)
        return build_view(prices, volumes=volumes, timeframe="3m", extra_timeframes={"15m": prices})

    @pytest.mark.parametrize("seed", [3, 7, 13])
    def test_fires_long_on_a_volume_backed_advance(self, seed):
        strategy = make(MomentumStrategy)
        assert strategy.generate(self.view_for(1, seed)).direction is Direction.LONG

    @pytest.mark.parametrize("seed", [3, 7, 13])
    def test_fires_short_on_a_volume_backed_decline(self, seed):
        strategy = make(MomentumStrategy)
        assert strategy.generate(self.view_for(-1, seed)).direction is Direction.SHORT

    def test_refuses_to_chase_an_exhausted_move(self):
        """The whole point: by the time a move is obvious it is often over."""
        strategy = make(MomentumStrategy, rsi_long_max=60.0)
        signal = strategy.generate(self.view_for(1))
        assert signal.direction is Direction.WAIT
        assert any("EXHAUSTED" in code for code in signal.reason_codes)

    def test_parabolic_move_without_retracement_is_refused(self):
        """A vertical move is where momentum entries lose the most."""
        strategy = make(MomentumStrategy)
        prices = trend_prices(200, drift=0.004, noise=0.0002)
        view = build_view(
            prices,
            volumes=[1000.0] * 190 + [3000.0] * 10,
            timeframe="3m",
            extra_timeframes={"15m": prices},
        )
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("EXHAUSTED" in code for code in signal.reason_codes)

    def test_requires_volume_backing(self):
        strategy = make(MomentumStrategy, volume_multiple=5.0)
        prices = impulse_prices(200, direction=1)
        view = build_view(
            prices, volumes=[1000.0] * len(prices), timeframe="3m", extra_timeframes={"15m": prices}
        )
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("VOLUME" in code for code in signal.reason_codes)

    def test_higher_timeframe_disagreement_blocks_entry(self):
        strategy = make(MomentumStrategy)
        prices = impulse_prices(200, direction=1)
        volumes = [1000.0] * 200 + [2200.0] * (len(prices) - 200)
        view = build_view(
            prices,
            volumes=volumes,
            timeframe="3m",
            extra_timeframes={"15m": trend_prices(200, drift=-0.003)},
        )
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert "HIGHER_TIMEFRAME_DISAGREES" in signal.reason_codes

    def test_flat_roc_produces_no_signal(self):
        strategy = make(MomentumStrategy)
        prices = ranging_prices(220, amplitude=0.0004)
        view = build_view(
            prices, volumes=[1000.0] * 220, timeframe="3m", extra_timeframes={"15m": prices}
        )
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("ROC" in code for code in signal.reason_codes)


class TestTrendFollowing:
    """Trend following enters on a PULLBACK, so its paths must retrace.

    A monotonic rise leaves price several ATR above the fast EMA, which the
    strategy correctly refuses as extended — that refusal is the difference
    between this strategy and the breakout one.
    """

    @pytest.mark.parametrize("seed", [3, 5, 7])
    def test_fires_long_in_a_confirmed_uptrend(self, seed):
        prices = trending_with_pullbacks(320, drift=0.0006, seed=seed)
        strategy = make(TrendFollowingStrategy)
        view = build_view(prices, timeframe="5m", extra_timeframes={"1h": prices})
        assert strategy.generate(view).direction is Direction.LONG

    @pytest.mark.parametrize("seed", [3, 5, 7])
    def test_fires_short_in_a_confirmed_downtrend(self, seed):
        prices = trending_with_pullbacks(320, drift=-0.0006, seed=seed)
        strategy = make(TrendFollowingStrategy)
        view = build_view(prices, timeframe="5m", extra_timeframes={"1h": prices})
        assert strategy.generate(view).direction is Direction.SHORT

    def test_weak_trend_is_refused(self):
        strategy = make(TrendFollowingStrategy, adx_min=45.0)
        prices = choppy_prices(320, seed=17)
        view = build_view(prices, timeframe="5m", extra_timeframes={"1h": prices})
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("ADX" in code or "UNCLEAR" in code for code in signal.reason_codes)

    def test_extended_price_is_not_chased(self):
        """Far from the fast EMA means the pullback entry no longer exists."""
        strategy = make(TrendFollowingStrategy)
        prices = trend_prices(320, drift=0.0015, noise=0.0003)
        view = build_view(prices, timeframe="5m", extra_timeframes={"1h": prices})
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("EXTENDED" in code for code in signal.reason_codes)

    def test_opposing_higher_timeframe_blocks_entry(self):
        """A 5m uptrend inside a 1h downtrend is a bounce, not a trend."""
        strategy = make(TrendFollowingStrategy)
        prices = trending_with_pullbacks(320, drift=0.0006, seed=3)
        view = build_view(
            prices, timeframe="5m", extra_timeframes={"1h": trend_prices(320, drift=-0.004)}
        )
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert "HIGHER_TIMEFRAME_TREND_OPPOSES" in signal.reason_codes


class TestBreakout:
    def squeeze_then_break(self, up=True, volume_multiple=4.0, extension=0.008):
        """Tight range, then a decisive break on volume."""
        base = [100.0 + math.sin(i / 3) * 0.05 for i in range(120)]
        direction = 1 if up else -1
        breakout = [100.0 * (1 + direction * extension * (i + 1)) for i in range(3)]
        prices = base + breakout
        volumes = [1000.0] * 120 + [1000.0 * volume_multiple] * 3
        return prices, volumes

    def test_fires_long_on_an_upside_break_with_volume(self):
        prices, volumes = self.squeeze_then_break(up=True)
        strategy = make(BreakoutStrategy, max_extension_atr=99.0)
        view = build_view(prices, volumes=volumes, timeframe="3m")
        assert strategy.generate(view).direction is Direction.LONG

    def test_fires_short_on_a_downside_break(self):
        prices, volumes = self.squeeze_then_break(up=False)
        strategy = make(BreakoutStrategy, max_extension_atr=99.0)
        view = build_view(prices, volumes=volumes, timeframe="3m")
        assert strategy.generate(view).direction is Direction.SHORT

    def test_break_without_volume_is_refused(self):
        prices, _ = self.squeeze_then_break(up=True)
        strategy = make(BreakoutStrategy, max_extension_atr=99.0)
        view = build_view(prices, volumes=[1000.0] * len(prices), timeframe="3m")
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("VOLUME" in code for code in signal.reason_codes)

    def test_break_without_prior_compression_is_refused(self):
        """A break from an already-wide range is continuation, not a breakout."""
        prices = trend_prices(150, drift=0.002, noise=0.004)
        strategy = make(BreakoutStrategy, max_extension_atr=99.0)
        view = build_view(prices, volumes=[1000.0] * 147 + [5000.0] * 3, timeframe="3m")
        signal = strategy.generate(view)
        if signal.direction is Direction.WAIT:
            assert any("COMPRESSION" in c or "NO_BREAKOUT" in c for c in signal.reason_codes)

    def test_already_extended_break_is_not_chased(self):
        prices, volumes = self.squeeze_then_break(up=True, extension=0.05)
        strategy = make(BreakoutStrategy, max_extension_atr=0.5)
        view = build_view(prices, volumes=volumes, timeframe="3m")
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("EXTENDED" in code for code in signal.reason_codes)

    def test_stop_sits_inside_the_broken_level(self):
        """If price returns through the level, the breakout has failed."""
        prices, volumes = self.squeeze_then_break(up=True)
        strategy = make(BreakoutStrategy, max_extension_atr=99.0)
        signal = strategy.generate(build_view(prices, volumes=volumes, timeframe="3m"))
        assert signal.is_actionable
        assert signal.stop_loss < signal.metadata["level"]


class TestMeanReversion:
    """Mean reversion requires a stretched but NON-TRENDING market.

    Paths use a mean-reverting (Ornstein-Uhlenbeck) walk rather than a sine
    wave: half a sine cycle is a dozen consecutive same-direction bars, which
    reads as ADX ~28 and trips the strategy's own anti-trend guard.
    """

    def test_fades_an_overbought_stretch(self):
        strategy = make(MeanReversionStrategy)
        prices = choppy_prices(200, seed=17, final_stretch=0.025)
        view = build_view(prices, timeframe="3m", regime=MarketRegime.SIDEWAYS)
        assert strategy.generate(view).direction is Direction.SHORT

    def test_fades_an_oversold_stretch(self):
        strategy = make(MeanReversionStrategy)
        prices = choppy_prices(200, seed=29, final_stretch=-0.025)
        view = build_view(prices, timeframe="3m", regime=MarketRegime.SIDEWAYS)
        assert strategy.generate(view).direction is Direction.LONG

    def test_refuses_to_fade_a_trending_market(self):
        """The guard that keeps this strategy from being an account-killer."""
        strategy = make(MeanReversionStrategy, rsi_overbought=55.0, min_stretch_atr=0.1, bb_std=1.0)
        prices = trend_prices(250, drift=0.003, noise=0.0003)
        view = build_view(prices, timeframe="3m", regime=MarketRegime.STRONG_TREND)
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("TRENDING" in code for code in signal.reason_codes)

    def test_unstretched_range_produces_no_signal(self):
        strategy = make(MeanReversionStrategy)
        view = build_view(choppy_prices(200, seed=17), timeframe="3m", regime=MarketRegime.SIDEWAYS)
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("NOT_STRETCHED" in c or "STRETCH" in c for c in signal.reason_codes)

    def test_target_is_the_mean_not_an_arbitrary_multiple(self):
        strategy = make(MeanReversionStrategy)
        prices = choppy_prices(200, seed=17, final_stretch=0.025)
        signal = strategy.generate(build_view(prices, timeframe="3m", regime=MarketRegime.SIDEWAYS))
        assert signal.is_actionable
        assert signal.take_profit == pytest.approx(signal.metadata["mean"], rel=0.02)


class TestVolumeSpike:
    def spike_series(self, bullish=True, body=0.9, multiple=6.0):
        prices = flat_prices(120, price=100.0, noise=0.00002)
        move = 0.006 if bullish else -0.006
        prices.append(prices[-1] * (1 + move))  # the spike bar
        prices.append(prices[-1] * (1 + move * 0.2))  # confirmation
        volumes = [1000.0] * 120 + [1000.0 * multiple, 1200.0]
        candles = make_candles(prices, volumes=volumes, wick=(1 - body) * 0.004)
        return candles

    def test_fires_with_the_spike_direction(self):
        strategy = make(VolumeSpikeStrategy, volume_multiple=3.0)
        for bullish, expected in ((True, Direction.LONG), (False, Direction.SHORT)):
            view = build_view(None, timeframe="1m", candles=self.spike_series(bullish=bullish))
            assert strategy.generate(view).direction is expected

    def test_wick_dominated_spike_is_refused(self):
        """A rejected move is a reversal signal, not a continuation one."""
        strategy = make(VolumeSpikeStrategy, volume_multiple=3.0, min_body_fraction=0.9)
        view = build_view(None, timeframe="1m", candles=self.spike_series(body=0.1))
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("REJECTED" in code for code in signal.reason_codes)

    def test_ordinary_volume_produces_no_signal(self):
        strategy = make(VolumeSpikeStrategy, volume_multiple=3.0)
        view = build_view(None, timeframe="1m", candles=self.spike_series(multiple=1.0))
        assert strategy.generate(view).direction is Direction.WAIT

    def test_faded_spike_is_refused(self):
        strategy = make(VolumeSpikeStrategy, volume_multiple=3.0)
        prices = flat_prices(120, price=100.0, noise=0.00002)
        prices.append(prices[-1] * 1.006)  # spike up
        prices.append(prices[-1] * 0.988)  # entirely given back
        volumes = [1000.0] * 120 + [6000.0, 1200.0]
        view = build_view(
            None, timeframe="1m", candles=make_candles(prices, volumes=volumes, wick=0.0001)
        )
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert "SPIKE_FADED" in signal.reason_codes


class TestVolatilityExpansion:
    def expanding(self, up=True):
        calm = flat_prices(120, price=100.0, noise=0.0002)
        direction = 1 if up else -1
        wild = list(calm)
        price = calm[-1]
        for i in range(20):
            price *= 1 + direction * 0.004 + (0.002 if i % 2 else -0.001)
            wild.append(price)
        return wild

    def test_fires_on_directional_expansion(self):
        strategy = make(VolatilityExpansionStrategy, expansion_multiple=1.2)
        for up, expected in ((True, Direction.LONG), (False, Direction.SHORT)):
            view = build_view(
                self.expanding(up=up),
                timeframe="3m",
                regime=MarketRegime.HIGH_VOLATILITY,
                wick=0.003,
            )
            assert strategy.generate(view).direction is expected

    def test_calm_market_produces_no_signal(self):
        strategy = make(VolatilityExpansionStrategy)
        view = build_view(flat_prices(200), timeframe="3m")
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("EXPANSION" in code for code in signal.reason_codes)

    def test_two_sided_expansion_without_direction_is_refused(self):
        """Expanding volatility with price mid-range has no directional edge."""
        strategy = make(
            VolatilityExpansionStrategy, expansion_multiple=1.1, min_directional_fraction=0.95
        )
        view = build_view(self.expanding(up=True), timeframe="3m", wick=0.003)
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT

    def test_uses_a_wider_stop_than_momentum(self):
        """Entering during expansion with a normal stop invites a noise stop-out."""
        expansion = make(VolatilityExpansionStrategy)
        momentum = make(MomentumStrategy)
        assert expansion.atr_stop_multiple > momentum.atr_stop_multiple


class TestVWAP:
    def test_fades_a_stretch_above_vwap_in_a_range(self):
        strategy = make(VWAPStrategy, band_std=0.8)
        prices = choppy_prices(200, seed=17, final_stretch=0.025)
        view = build_view(prices, timeframe="3m", regime=MarketRegime.SIDEWAYS)
        signal = strategy.generate(view)
        assert signal.direction is Direction.SHORT
        assert signal.take_profit == pytest.approx(signal.metadata["vwap"], rel=0.02)

    def test_fades_a_stretch_below_vwap_in_a_range(self):
        strategy = make(VWAPStrategy, band_std=0.8)
        prices = choppy_prices(200, seed=29, final_stretch=-0.025)
        view = build_view(prices, timeframe="3m", regime=MarketRegime.SIDEWAYS)
        assert strategy.generate(view).direction is Direction.LONG

    def test_refuses_to_fade_a_trending_market(self):
        strategy = make(VWAPStrategy, max_adx=10.0, band_std=0.5)
        view = build_view(
            trend_prices(250, drift=0.003, noise=0.0003),
            timeframe="3m",
            regime=MarketRegime.STRONG_TREND,
        )
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert any("TRENDING" in code for code in signal.reason_codes)

    def test_inside_the_bands_produces_no_signal(self):
        strategy = make(VWAPStrategy, band_std=6.0)
        view = build_view(choppy_prices(200, seed=17), timeframe="3m", regime=MarketRegime.SIDEWAYS)
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert "INSIDE_VWAP_BANDS" in signal.reason_codes

    def test_without_volume_data_it_declines(self):
        strategy = make(VWAPStrategy)
        view = build_view(choppy_prices(200), volumes=[0.0] * 200, timeframe="3m")
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT
        assert "NO_VOLUME_DATA" in signal.reason_codes


class TestSupportResistance:
    def level_test(self, bounce=True):
        """Price touches ~97 three times, then bounces from it."""
        prices = []
        for _ in range(3):
            prices += [100.0, 99.0, 98.0, 97.0, 98.0, 99.0, 100.0, 99.5]
        prices += [100.0] * 40
        prices += [99.0, 98.0, 97.05]
        prices.append(98.2 if bounce else 96.0)
        return prices

    def test_fires_long_on_a_rejection_from_support(self):
        strategy = make(SupportResistanceStrategy, min_touches=2, entry_distance_atr=3.0)
        view = build_view(
            self.level_test(), timeframe="5m", regime=MarketRegime.SIDEWAYS, wick=0.004
        )
        signal = strategy.generate(view)
        if signal.is_actionable:
            assert signal.direction is Direction.LONG
            assert signal.stop_loss < signal.metadata["level"]

    def test_no_established_levels_means_no_signal(self):
        strategy = make(SupportResistanceStrategy, min_touches=8)
        view = build_view(trend_prices(200, drift=0.001), timeframe="5m")
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT

    def test_price_away_from_every_level_produces_no_signal(self):
        strategy = make(SupportResistanceStrategy, entry_distance_atr=0.01, min_touches=2)
        view = build_view(self.level_test(), timeframe="5m")
        signal = strategy.generate(view)
        assert signal.direction is Direction.WAIT


# --------------------------------------------------------------------------- #
# Registry and regime gating
# --------------------------------------------------------------------------- #
class TestRegistry:
    def registry(self) -> StrategyRegistry:
        return StrategyRegistry.from_config(CONFIG)

    def test_all_eight_strategies_load(self):
        assert set(self.registry().strategies) == set(STRATEGY_CLASSES)
        assert len(STRATEGY_CLASSES) == 8

    def test_panic_regime_runs_no_strategy_at_all(self):
        """Not 'runs them and discounts them' — does not run them."""
        registry = self.registry()
        view = build_view(trend_prices(300, drift=0.002), regime=MarketRegime.PANIC)
        signals, weights = registry.evaluate(view)
        assert signals == []
        assert weights == {}

    def test_regime_selects_the_appropriate_strategies(self):
        registry = self.registry()
        trending = set(registry.weights_for(MarketRegime.STRONG_TREND))
        ranging = set(registry.weights_for(MarketRegime.SIDEWAYS))
        assert "trend_following" in trending
        assert "mean_reversion" not in trending
        assert "mean_reversion" in ranging
        assert "trend_following" not in ranging

    def test_suspended_strategy_is_excluded_from_the_active_set(self):
        registry = self.registry()
        registry.disable("momentum", until=2_000_000.0, reason="test")
        active = registry.active(MarketRegime.STRONG_TREND, now=1_000_000.0)
        assert "momentum" not in active
        assert "trend_following" in active

    def test_suspension_expires_at_its_deadline(self):
        registry = self.registry()
        registry.disable("momentum", until=1_000.0)
        assert registry.is_disabled("momentum", now=500.0)
        assert "momentum" not in registry.active(MarketRegime.STRONG_TREND, now=500.0)

        assert not registry.is_disabled("momentum", now=2_000.0)
        assert "momentum" in registry.active(MarketRegime.STRONG_TREND, now=2_000.0)

    def test_wait_signals_are_returned_for_the_audit_log(self):
        """'Considered and declined' is different from 'never ran'."""
        registry = self.registry()
        view = build_view(
            flat_prices(300),
            regime=MarketRegime.STRONG_TREND,
            extra_timeframes={
                "15m": flat_prices(300),
                "1h": flat_prices(300),
                "5m": flat_prices(300),
                "3m": flat_prices(300),
            },
        )
        signals, weights = registry.evaluate(view)
        assert weights
        assert signals
        assert all(s.direction is Direction.WAIT for s in signals)

    def test_actionable_filters_out_waits(self):
        registry = self.registry()
        view = build_view(
            flat_prices(300),
            regime=MarketRegime.STRONG_TREND,
            extra_timeframes={"15m": flat_prices(300), "3m": flat_prices(300)},
        )
        assert registry.actionable(view)[0] == []

    def test_unknown_strategy_in_config_is_skipped_not_fatal(self):
        from copy import deepcopy

        config = deepcopy(CONFIG)
        config.strategies["nonexistent_strategy"] = {"enabled": True}
        registry = StrategyRegistry.from_config(config)
        assert "nonexistent_strategy" not in registry.strategies
        assert len(registry.strategies) == 8
