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


def ranging_prices(
    n: int, centre: float = 100.0, amplitude: float = 0.004, period: int = 20, seed: int = 11
) -> list[float]:
    """An oscillating, non-trending series."""
    rng = random.Random(seed)
    return [
        centre * (1 + amplitude * math.sin(2 * math.pi * i / period) + rng.gauss(0, amplitude / 12))
        for i in range(n)
    ]


def flat_prices(
    n: int, price: float = 100.0, noise: float = 0.00005, seed: int = 13
) -> list[float]:
    """Almost no movement — the case where every strategy should say WAIT."""
    rng = random.Random(seed)
    return [price * (1 + rng.gauss(0, noise)) for i in range(n)]


@pytest.fixture
def candle_factory():
    return make_candles
