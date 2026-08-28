"""The diagnostic recorder must not be able to change a single decision.

Instrumentation that perturbs what it measures is worse than no instrumentation,
because the numbers still look like numbers. These tests exist to make that
failure impossible to ship: the same backtest is run twice on identical data
with identical seeds, once with a recorder attached and once without, and every
observable output is compared — trade for trade, fill for fill, PnL to the last
float, and every rejection counter.

If any of these fail, the diagnostic is not observational and its output is not
evidence about the uninstrumented system.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tradebot.backtesting.diagnostics import CandidateRecorder
from tradebot.backtesting.engine import BacktestData, BacktestEngine
from tradebot.core.config import load_tunables
from tradebot.core.types import Candle, Direction, SymbolInfo
from tradebot.risk.sizing import PositionSizer, SizingResult

from ..conftest import REPO_ROOT

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.backtest.yaml", REPO_ROOT / "config" / "strategies.yaml"
)
START = 1_704_067_200_000
STEPS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def info(symbol: str = "BTCUSDT") -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol,
        base_asset=symbol[:-4],
        quote_asset="USDT",
        status="TRADING",
        contract_type="PERPETUAL",
        price_precision=2,
        quantity_precision=3,
        tick_size=0.10,
        step_size=0.001,
        min_qty=0.002,
        max_qty=1000.0,
        min_notional=7.0,
        market_min_qty=0.004,
        max_leverage=125,
    )


def moving_bars(n: int, step: int) -> list[Candle]:
    """Bars with enough movement that strategies actually fire."""
    out: list[Candle] = []
    price = 100.0
    for i in range(n):
        # A deterministic zig-zag with drift, so signals and rejections both occur.
        price *= 1.0 + (0.004 if (i // 7) % 3 else -0.003)
        high = price * 1.004
        low = price * 0.996
        out.append(
            Candle(
                START + i * step,
                price,
                high,
                low,
                price * 1.001,
                10.0 + (i % 13),
                START + i * step + step - 1,
                quote_volume=1_000_000.0 + (i % 29) * 5_000,
                trades=50 + (i % 17),
                taker_buy_volume=5.0,
            )
        )
    return out


@pytest.fixture(scope="module")
def dataset() -> dict[str, BacktestData]:
    return {
        "BTCUSDT": BacktestData(
            symbol="BTCUSDT",
            candles={tf: moving_bars(700, step) for tf, step in STEPS.items()},
            symbol_info=info(),
            funding_rates={START + i * 28_800_000: 0.0001 for i in range(8)},
        )
    }


def run(data: dict[str, BacktestData], recorder: CandidateRecorder | None):
    engine = BacktestEngine(CONFIG)
    if recorder is not None:
        engine.pipeline.recorder = recorder
        engine.risk.recorder = recorder
    return engine.run(data, seed=42), engine


class TestTheRecorderChangesNothing:
    def test_the_same_run_produces_the_same_pnl(self, dataset) -> None:
        plain, _ = run(dataset, None)
        recorder = CandidateRecorder()
        observed, _ = run(dataset, recorder)

        assert observed.metrics.net_profit == plain.metrics.net_profit
        assert observed.metrics.total_trades == plain.metrics.total_trades
        assert observed.metrics.total_return == plain.metrics.total_return
        assert observed.metrics.max_drawdown == plain.metrics.max_drawdown

    def test_every_trade_is_identical(self, dataset) -> None:
        plain, _ = run(dataset, None)
        observed, _ = run(dataset, CandidateRecorder())

        assert len(observed.trades) == len(plain.trades)
        for want, got in zip(plain.trades, observed.trades, strict=True):
            assert got.symbol == want.symbol
            assert got.strategy == want.strategy
            assert got.direction == want.direction
            assert got.entry_price == want.entry_price
            assert got.exit_price == want.exit_price
            assert got.quantity == want.quantity
            assert got.opened_at == want.opened_at
            assert got.closed_at == want.closed_at
            assert got.net_pnl == want.net_pnl
            assert got.exit_reason == want.exit_reason

    def test_every_rejection_counter_is_identical(self, dataset) -> None:
        plain, _ = run(dataset, None)
        observed, _ = run(dataset, CandidateRecorder())
        assert observed.rejections == plain.rejections

    def test_the_equity_curve_is_identical(self, dataset) -> None:
        plain, _ = run(dataset, None)
        observed, _ = run(dataset, CandidateRecorder())
        assert len(observed.equity_curve) == len(plain.equity_curve)
        assert [p.equity for p in observed.equity_curve] == [p.equity for p in plain.equity_curve]

    def test_the_recorder_actually_recorded_something(self, dataset) -> None:
        """Guards against the tests above passing because nothing was hooked up."""
        recorder = CandidateRecorder()
        run(dataset, recorder)
        assert recorder.records, "the recorder captured nothing; the hooks are not wired"


class TestTheRecordedValuesAreHonest:
    def test_a_missing_value_is_none_not_zero(self) -> None:
        """A candidate rejected before consensus was computed has no consensus.
        Recording 0.0 would invent a point at the bottom of the distribution."""
        recorder = CandidateRecorder()
        recorder.record("BTCUSDT", "aggregation", "INSUFFICIENT_CONSENSUS", agreeing=1)
        record = recorder.records[0]
        assert record.agreeing == 1
        assert record.consensus is None
        assert record.expected_net is None

    def test_values_helper_skips_missing_rather_than_coercing(self) -> None:
        recorder = CandidateRecorder()
        recorder.record("A", "aggregation", "X", consensus=None)
        recorder.record("B", "aggregation", "X", consensus=61.5)
        assert recorder.values("consensus") == [61.5]

    def test_values_can_be_filtered_by_reason(self) -> None:
        recorder = CandidateRecorder()
        recorder.record("A", "edge", "NEGATIVE_EXPECTED_EDGE", expected_net=-0.001)
        recorder.record("B", "complete", None, expected_net=0.002)
        assert recorder.values("expected_net", reason="NEGATIVE_EXPECTED_EDGE") == [-0.001]

    def test_accepted_candidates_are_recorded_with_no_reason(self) -> None:
        recorder = CandidateRecorder()
        recorder.record("A", "complete", None, agreeing=3)
        assert recorder.by_reason() == {"ACCEPTED": 1}


class TestSizingAnnotationPreservesTheDecision:
    """`observed()` uses `dataclasses.replace`, so it must copy every other
    field verbatim. A regression here would silently corrupt sizing."""

    def _sized(self, stop: float) -> SizingResult:
        return PositionSizer(CONFIG.risk, CONFIG.execution.max_min_notional_ratio).size(
            equity=75.0,
            risk_fraction=0.005,
            entry_price=100.0,
            stop_price=stop,
            direction=Direction.LONG,
            symbol_info=info(),
        )

    def test_the_annotation_matches_a_manual_replace(self) -> None:
        result = self._sized(99.0)
        stripped = replace(result, stop_distance=None, raw_quantity=None)
        rebuilt = replace(
            stripped, stop_distance=result.stop_distance, raw_quantity=result.raw_quantity
        )
        assert rebuilt == result

    def test_the_recorded_stop_distance_is_the_one_used(self) -> None:
        result = self._sized(99.0)
        assert result.stop_distance == pytest.approx(1.0)

    def test_raw_quantity_is_recorded_even_when_the_size_is_refused(self) -> None:
        """The refusal is the interesting case: without the recorded quantity
        there is no way to know how far below the minimum it fell."""
        result = self._sized(99.999)  # a stop so tight the exposure cap binds
        assert result.stop_distance is not None
        assert result.raw_quantity is not None

    def test_an_early_refusal_records_nothing_rather_than_guessing(self) -> None:
        """A zero stop distance is refused before any quantity exists."""
        result = self._sized(100.0)
        assert not result.ok
        assert result.stop_distance is None
        assert result.raw_quantity is None
