"""Shared fixtures.

Every test runs against a virtual clock and an in-memory configuration, so the
suite is deterministic and never touches the network or the wall clock.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from tradebot.core.clock import VirtualClock
from tradebot.core.config import TunableConfig, load_tunables
from tradebot.core.logging import configure_logging
from tradebot.core.types import Candle, SymbolInfo

REPO_ROOT = Path(__file__).resolve().parents[1]

configure_logging("CRITICAL", "json", None)


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(start_ms=1_700_000_000_000)


@pytest.fixture(scope="session")
def tunables() -> TunableConfig:
    """The shipped default configuration, validated."""
    return load_tunables(
        REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
    )


@pytest.fixture
def symbol_info() -> SymbolInfo:
    """A realistic USDⓈ-M symbol with typical Binance filters."""
    return SymbolInfo(
        symbol="TESTUSDT",
        base_asset="TEST",
        quote_asset="USDT",
        status="TRADING",
        contract_type="PERPETUAL",
        price_precision=2,
        quantity_precision=3,
        tick_size=0.01,
        step_size=0.001,
        min_qty=0.001,
        max_qty=10000.0,
        min_notional=5.0,
        max_leverage=20,
    )


# --------------------------------------------------------------------------- #
# Deterministic synthetic price paths
#
# These exist to prove ENGINE correctness, never to estimate profitability.
# --------------------------------------------------------------------------- #
def make_candles(
    prices: list[float],
    start_ms: int = 1_700_000_000_000,
    interval_ms: int = 60_000,
    volume: float = 1000.0,
    volumes: list[float] | None = None,
    wick: float = 0.001,
) -> list[Candle]:
    """Build a candle series from a list of closes.

    Each bar opens at the previous close and carries a symmetric wick, which is
    enough for indicator and strategy tests without pretending to be real data.
    """
    candles: list[Candle] = []
    for i, close in enumerate(prices):
        open_ = prices[i - 1] if i else close
        high = max(open_, close) * (1 + wick)
        low = min(open_, close) * (1 - wick)
        vol = volumes[i] if volumes is not None else volume
        open_time = start_ms + i * interval_ms
        candles.append(
            Candle(
                open_time=open_time,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=vol,
                close_time=open_time + interval_ms - 1,
                quote_volume=vol * close,
                trades=int(vol),
                taker_buy_volume=vol * 0.5,
                closed=True,
            )
        )
    return candles


def trend_prices(
    n: int, start: float = 100.0, drift: float = 0.001, noise: float = 0.0002, seed: int = 7
) -> list[float]:
    """A clean trend with light noise."""
    rng = random.Random(seed)
    price = start
    out = []
    for _ in range(n):
        price *= 1 + drift + rng.gauss(0, noise)
        out.append(price)
    return out


def trending_with_pullbacks(
    n: int,
    start: float = 100.0,
    drift: float = 0.0012,
    pullback_every: int = 12,
    pullback_bars: int = 4,
    pullback_strength: float = 0.6,
    noise: float = 0.0004,
    seed: int = 3,
) -> list[float]:
    """A trend that RETRACES, which is what a real trend looks like.

    ``trend_prices`` rises monotonically, which saturates RSI at 100 and leaves
    price permanently extended from its moving averages — conditions the
    momentum and trend strategies are specifically built to refuse. Any test
    that wants those strategies to fire needs a path with pullbacks in it.
    """
    rng = random.Random(seed)
    price = start
    out: list[float] = []
    for i in range(n):
        in_pullback = (i % pullback_every) >= (pullback_every - pullback_bars)
        step = -drift * pullback_strength if in_pullback else drift
        price *= 1 + step + rng.gauss(0, noise)
        out.append(price)
    return out


def ranging_prices(
    n: int, centre: float = 100.0, amplitude: float = 0.004, period: int = 20, seed: int = 11
) -> list[float]:
    """An oscillating, non-trending series."""
    rng = random.Random(seed)
    return [
        centre * (1 + amplitude * math.sin(2 * math.pi * i / period) + rng.gauss(0, amplitude / 12))
        for i in range(n)
    ]


def impulse_prices(
    n_base: int = 200,
    direction: int = 1,
    size: float = 0.0013,
    seed: int = 3,
    centre: float = 100.0,
) -> list[float]:
    """A consolidation followed by a realistic directional impulse.

    The impulse contains counter-bars. That detail is not cosmetic: a run of
    consecutive same-direction bars with no retracement drives Wilder RSI above
    85 as a matter of arithmetic, and the momentum strategy is built to refuse
    exactly that (its `rsi_long_max` ceiling). Real impulses retrace; a test
    path without retracement tests nothing but the saturation guard.
    """
    prices = ranging_prices(n_base, centre=centre, amplitude=0.0015, period=25, seed=seed)
    # up, up, pullback, up, up, pullback, up
    for step in (1.0, 1.0, -0.45, 0.9, 1.0, -0.35, 0.8):
        prices.append(prices[-1] * (1 + direction * size * step))
    return prices


def choppy_prices(
    n: int,
    centre: float = 100.0,
    sigma: float = 0.0025,
    reversion: float = 0.10,
    seed: int = 17,
    final_stretch: float = 0.0,
) -> list[float]:
    """A mean-reverting (Ornstein-Uhlenbeck) walk — a real ranging market.

    A smooth sine wave is NOT a substitute: half a sine cycle is a dozen
    consecutive same-direction bars, which reads as a strong trend to ADX (~28
    in practice). Mean reversion and VWAP-fade both stand down above their ADX
    ceiling, so a sine-based path can never exercise them.

    ``final_stretch`` appends a sharp move away from the mean, producing the
    stretched-but-not-trending condition those strategies exist to trade.
    """
    rng = random.Random(seed)
    deviation = 0.0
    out: list[float] = []
    for _ in range(n):
        deviation += -reversion * deviation + rng.gauss(0, sigma)
        out.append(centre * (1 + deviation))
    if final_stretch:
        price = out[-1]
        for step in (0.45, 0.3, 0.25):
            price *= 1 + final_stretch * step
            out.append(price)
    return out


def flat_prices(
    n: int, price: float = 100.0, noise: float = 0.00005, seed: int = 13
) -> list[float]:
    """Almost no movement — the case where every strategy should say WAIT."""
    rng = random.Random(seed)
    return [price * (1 + rng.gauss(0, noise)) for i in range(n)]


def realistic_volumes(prices: list[float], base: float = 1000.0, seed: int = 21) -> list[float]:
    """Volume that varies with price movement, as real markets do.

    Constant volume is not a neutral simplification: it pins ``volume_ratio`` at
    1.0 and ``volume_zscore`` at nan, which systematically depresses the volume
    and volume-anomaly components of both the market score and the opportunity
    score. A fixture built that way cannot reach the acceptance threshold no
    matter how good the price action is — the strategy looks broken when the
    data is.
    """
    rng = random.Random(seed)
    out: list[float] = []
    for i, price in enumerate(prices):
        move = abs(price / prices[i - 1] - 1.0) if i else 0.0
        # Volume clusters around movement, plus a lognormal-ish base.
        multiplier = 1.0 + move * 400.0 + abs(rng.gauss(0, 0.45))
        out.append(base * multiplier)
    return out


def resample(candles: list[Candle], factor: int) -> list[Candle]:
    """Aggregate 1-minute candles into a coarser timeframe.

    Reusing one series for every timeframe — which is the lazy way to build
    multi-timeframe fixtures — gives every timeframe identical indicator values,
    so higher-timeframe confirmation becomes a tautology and the strategies
    behave nothing like they would on real data.
    """
    out: list[Candle] = []
    for start in range(0, len(candles) - factor + 1, factor):
        group = candles[start : start + factor]
        out.append(
            Candle(
                open_time=group[0].open_time,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
                close_time=group[-1].close_time,
                quote_volume=sum(c.quote_volume for c in group),
                trades=sum(c.trades for c in group),
                taker_buy_volume=sum(c.taker_buy_volume for c in group),
                closed=True,
            )
        )
    return out


def multi_timeframe(
    prices: list[float],
    volumes: list[float] | None = None,
    start_ms: int = 1_700_000_000_000,
    wick: float = 0.001,
    seed: int = 21,
) -> dict[str, list[Candle]]:
    """Build a realistic 1m/3m/5m/15m/1h set by resampling one 1-minute series.

    Volume defaults to :func:`realistic_volumes` rather than a constant, for the
    reasons documented there.
    """
    if volumes is None:
        volumes = realistic_volumes(prices, seed=seed)
    base = make_candles(prices, start_ms=start_ms, interval_ms=60_000, volumes=volumes, wick=wick)
    return {
        "1m": base,
        "3m": resample(base, 3),
        "5m": resample(base, 5),
        "15m": resample(base, 15),
        "1h": resample(base, 60),
    }


@pytest.fixture
def candle_factory():
    return make_candles
