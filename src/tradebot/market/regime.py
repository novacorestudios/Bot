"""Market regime classification.

The same strategy that prints money in a trend bleeds in chop. Rather than
hoping a strategy's own filters catch that, the engine classifies the market
first and only runs the strategies appropriate to it — the mapping lives in
``config.yaml`` under ``regime.strategy_weights``.

Classification order matters and is deliberate:

1. **PANIC** first. An abnormal move or volume explosion overrides everything;
   in this regime no strategy runs and no new entry is permitted. Detecting it
   late is how a bot buys the first leg of a cascade.
2. **BREAKOUT** next — a squeeze that has just released.
3. **HIGH/LOW_VOLATILITY** — extremes of realised volatility relative to the
   symbol's own history, not an absolute threshold.
4. **STRONG/WEAK_TREND** by ADX.
5. **SIDEWAYS** otherwise.

Everything is measured against the symbol's own recent distribution, because an
ATR that is high for BTC is low for a small-cap perpetual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tradebot.core.config import RegimeConfig
from tradebot.core.mathutil import clamp, safe_div
from tradebot.core.types import Direction, MarketRegime
from tradebot.market import indicators as ind
from tradebot.market.candles import CandleSeries


@dataclass(frozen=True, slots=True)
class RegimeState:
    """A classification plus the evidence behind it."""

    regime: MarketRegime
    confidence: float  # 0..100
    direction: Direction  # trend direction, WAIT when undirected
    adx: float
    atr_percent: float
    atr_percentile: float  # where current ATR sits in its own history
    bandwidth: float
    bandwidth_percentile: float
    recent_return: float
    volume_ratio: float
    reasons: tuple[str, ...] = ()
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def blocks_entries(self) -> bool:
        return self.regime.blocks_entries

    @property
    def is_trending(self) -> bool:
        return self.regime in {MarketRegime.STRONG_TREND, MarketRegime.WEAK_TREND}

    def as_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 2),
            "direction": self.direction.value,
            "adx": round(self.adx, 2),
            "atr_percent": round(self.atr_percent, 6),
            "atr_percentile": round(self.atr_percentile, 3),
            "bandwidth_percentile": round(self.bandwidth_percentile, 3),
            "recent_return": round(self.recent_return, 5),
            "volume_ratio": round(self.volume_ratio, 2),
            "reasons": list(self.reasons),
        }


UNKNOWN = RegimeState(
    regime=MarketRegime.SIDEWAYS,
    confidence=0.0,
    direction=Direction.WAIT,
    adx=0.0,
    atr_percent=0.0,
    atr_percentile=0.5,
    bandwidth=0.0,
    bandwidth_percentile=0.5,
    recent_return=0.0,
    volume_ratio=1.0,
    reasons=("INSUFFICIENT_DATA",),
)


class RegimeDetector:
    """Classifies one symbol's current market regime."""

    def __init__(self, config: RegimeConfig) -> None:
        self.config = config

    def min_bars(self) -> int:
        """Bars required before a classification is meaningful."""
        return max(
            self.config.regime_lookback // 2,
            self.config.adx_period * 3,
            self.config.bb_period * 2,
            60,
        )

    def detect(self, series: CandleSeries) -> RegimeState:
        """Classify the regime from closed bars only."""
        if not series.ready(self.min_bars()):
            return UNKNOWN

        cfg = self.config
        highs, lows, closes = series.highs, series.lows, series.closes
        volumes = series.volumes

        # -- measurements ---------------------------------------------------- #
        adx_series, plus_di, minus_di = ind.adx(highs, lows, closes, cfg.adx_period)
        adx_value = ind.last_valid(adx_series, default=0.0)
        pdi = ind.last_valid(plus_di, default=0.0)
        mdi = ind.last_valid(minus_di, default=0.0)

        atr_pct_series = ind.atr_percent(highs, lows, closes, cfg.atr_period)
        atr_pct = ind.last_valid(atr_pct_series, default=0.0)
        atr_percentile = _percentile_rank(atr_pct_series, atr_pct, cfg.regime_lookback)

        bandwidth_series = ind.bollinger_bandwidth(closes, cfg.bb_period, cfg.bb_std)
        bandwidth = ind.last_valid(bandwidth_series, default=0.0)
        bandwidth_percentile = _percentile_rank(bandwidth_series, bandwidth, cfg.regime_lookback)

        window = min(cfg.panic_window_bars, closes.size - 1)
        recent_return = (
            safe_div(closes[-1] - closes[-1 - window], abs(closes[-1 - window]), 0.0)
            if window > 0
            else 0.0
        )

        vol_ratio = ind.last_valid(ind.volume_ratio(volumes, 30), default=1.0)

        direction = Direction.WAIT
        if pdi > mdi and adx_value >= cfg.adx_weak_trend:
            direction = Direction.LONG
        elif mdi > pdi and adx_value >= cfg.adx_weak_trend:
            direction = Direction.SHORT

        base = {
            "adx": adx_value,
            "atr_percent": atr_pct,
            "atr_percentile": atr_percentile,
            "bandwidth": bandwidth,
            "bandwidth_percentile": bandwidth_percentile,
            "recent_return": recent_return,
            "volume_ratio": vol_ratio,
            "direction": direction,
        }

        # -- 1. PANIC — checked first and overrides everything ---------------- #
        # A large move on its own is panic. A merely elevated move is panic only
        # when volume explodes with it; requiring both halves avoids flagging
        # every routine news candle and shutting down entries unnecessarily.
        move = abs(recent_return)
        big_move = move >= cfg.panic_return_threshold
        elevated_move = move >= cfg.panic_return_threshold * 0.5
        volume_explosion = vol_ratio >= cfg.panic_volume_multiple

        if big_move or (elevated_move and volume_explosion):
            panic_reasons = [f"MOVE_{move * 100:.1f}PCT_IN_{window}_BARS"]
            if volume_explosion:
                panic_reasons.append(f"VOLUME_{vol_ratio:.1f}X")
            confidence = clamp(50.0 + 50.0 * move / (cfg.panic_return_threshold * 2), 50.0, 100.0)
            return _build(MarketRegime.PANIC, confidence, tuple(panic_reasons), base)

        # -- 2. BREAKOUT — a squeeze that has just released -------------------- #
        squeeze_before = _was_squeezed(
            bandwidth_series, cfg.squeeze_bandwidth, lookback=cfg.breakout_lookback
        )
        upper, lower = ind.donchian(highs, lows, cfg.breakout_lookback)
        upper_level = ind.last_valid(upper, default=float("inf"))
        lower_level = ind.last_valid(lower, default=0.0)
        broke_up = closes[-1] > upper_level
        broke_down = closes[-1] < lower_level

        if squeeze_before and (broke_up or broke_down):
            confidence = clamp(60.0 + 40.0 * min(vol_ratio / 2.0, 1.0), 60.0, 100.0)
            base["direction"] = Direction.LONG if broke_up else Direction.SHORT
            return _build(
                MarketRegime.BREAKOUT,
                confidence,
                ("SQUEEZE_RELEASE", "BREAK_UP" if broke_up else "BREAK_DOWN"),
                base,
            )

        # -- 3. volatility extremes ------------------------------------------- #
        if atr_percentile >= cfg.high_volatility_percentile:
            confidence = clamp(
                50.0
                + 50.0
                * (atr_percentile - cfg.high_volatility_percentile)
                / max(1e-9, 1.0 - cfg.high_volatility_percentile),
                50.0,
                100.0,
            )
            return _build(
                MarketRegime.HIGH_VOLATILITY,
                confidence,
                (f"ATR_PERCENTILE_{atr_percentile:.2f}",),
                base,
            )

        if atr_percentile <= cfg.low_volatility_percentile:
            confidence = clamp(
                50.0
                + 50.0
                * (cfg.low_volatility_percentile - atr_percentile)
                / max(1e-9, cfg.low_volatility_percentile),
                50.0,
                100.0,
            )
            return _build(
                MarketRegime.LOW_VOLATILITY,
                confidence,
                (f"ATR_PERCENTILE_{atr_percentile:.2f}",),
                base,
            )

        # -- 4. trend ---------------------------------------------------------- #
        if adx_value >= cfg.adx_strong_trend:
            confidence = clamp(60.0 + (adx_value - cfg.adx_strong_trend) * 2.0, 60.0, 100.0)
            return _build(
                MarketRegime.STRONG_TREND,
                confidence,
                (f"ADX_{adx_value:.1f}", f"DI_{'PLUS' if pdi > mdi else 'MINUS'}"),
                base,
            )

        if adx_value >= cfg.adx_weak_trend:
            confidence = clamp(50.0 + (adx_value - cfg.adx_weak_trend) * 2.0, 50.0, 80.0)
            return _build(MarketRegime.WEAK_TREND, confidence, (f"ADX_{adx_value:.1f}",), base)

        # -- 5. default -------------------------------------------------------- #
        confidence = clamp(50.0 + (cfg.adx_weak_trend - adx_value) * 2.0, 50.0, 90.0)
        return _build(
            MarketRegime.SIDEWAYS,
            confidence,
            (f"ADX_{adx_value:.1f}_BELOW_{cfg.adx_weak_trend}",),
            base,
        )

    def strategy_weights(self, regime: MarketRegime) -> dict[str, float]:
        """Which strategies may run in this regime, and at what weight.

        An empty mapping means no strategy runs — which is the whole point of
        the PANIC regime.
        """
        return self.config.weights_for(regime.value)

    def allows(self, regime: MarketRegime, strategy: str) -> bool:
        return strategy in self.strategy_weights(regime)


def _build(
    regime: MarketRegime, confidence: float, reasons: tuple[str, ...], base: dict
) -> RegimeState:
    return RegimeState(
        regime=regime,
        confidence=confidence,
        direction=base["direction"],
        adx=base["adx"],
        atr_percent=base["atr_percent"],
        atr_percentile=base["atr_percentile"],
        bandwidth=base["bandwidth"],
        bandwidth_percentile=base["bandwidth_percentile"],
        recent_return=base["recent_return"],
        volume_ratio=base["volume_ratio"],
        reasons=reasons,
    )


def _percentile_rank(series: np.ndarray, value: float, lookback: int) -> float:
    """Where `value` sits within the series' own recent distribution, in [0, 1]."""
    if not np.isfinite(value):
        return 0.5
    finite = series[np.isfinite(series)]
    if finite.size < 20:
        return 0.5
    window = finite[-lookback:] if finite.size > lookback else finite
    return float((window <= value).mean())


def _was_squeezed(bandwidth: np.ndarray, threshold: float, lookback: int) -> bool:
    """True when the range was compressed in the bars BEFORE the current one.

    Excluding the current bar matters: a breakout bar itself widens the bands, so
    testing the current bandwidth would never see the squeeze that preceded it.
    """
    finite_mask = np.isfinite(bandwidth)
    if finite_mask.sum() < lookback + 2:
        return False
    prior = bandwidth[:-1]
    prior = prior[np.isfinite(prior)]
    if prior.size < lookback:
        return False
    return bool(np.min(prior[-lookback:]) <= threshold)
