"""V3.1 correctness patch: one regression test per fixed issue.

Every test here is named for the defect it prevents. Several of the defects
were **silent** — they produced plausible numbers rather than errors — so where
the bug was structural the test is structural too: checking only the output
would have passed with the bug present.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from tradebot.backtesting.data import load_dataset
from tradebot.backtesting.engine import BacktestData, BacktestEngine
from tradebot.backtesting.runner import EdgeMode, OOSMode, run_strict_oos
from tradebot.backtesting.trust import TrustLevel, evaluate_trust
from tradebot.core.config import load_tunables
from tradebot.core.types import (
    AggregatedSignal,
    Candle,
    Direction,
    ExitReason,
    MarketRegime,
    Position,
    Signal,
    SymbolInfo,
    Trade,
)
from tradebot.data.store import DataStore

from ..conftest import REPO_ROOT

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.backtest.yaml", REPO_ROOT / "config" / "strategies.yaml"
)
START = 1_704_067_200_000
FUNDING_STEP = 28_800_000  # 8h


def bars(n: int, step: int, start: int = START, price: float = 100.0) -> list[Candle]:
    return [
        Candle(
            start + i * step,
            price,
            price * 1.002,
            price * 0.998,
            price,
            10.0,
            start + i * step + step - 1,
            quote_volume=1000.0,
            trades=5,
        )
        for i in range(n)
    ]


def info(symbol: str = "BTCUSDT", **kwargs) -> SymbolInfo:
    defaults = {
        "symbol": symbol,
        "base_asset": symbol[:-4],
        "quote_asset": "USDT",
        "status": "TRADING",
        "contract_type": "PERPETUAL",
        "price_precision": 2,
        "quantity_precision": 3,
        "tick_size": 0.10,
        "step_size": 0.001,
        "min_qty": 0.002,
        "max_qty": 1000.0,
        "min_notional": 7.0,
        "market_min_qty": 0.004,
        "max_leverage": 125,
    }
    defaults.update(kwargs)
    return SymbolInfo(**defaults)  # type: ignore[arg-type]


def dataset(
    tmp_path,
    *,
    funding: dict[int, float] | None = None,
    exchange_info: bool = True,
    timeframes=("1m", "3m", "5m", "15m", "1h"),
):
    """A complete on-disk dataset, written by the real DataStore."""
    store = DataStore(tmp_path)
    steps = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
    for timeframe in timeframes:
        step = steps[timeframe]
        store.write_klines("BTCUSDT", timeframe, bars(400, step), source="test")
    if funding is not None:
        store.write_funding("BTCUSDT", funding, source="test")
    if exchange_info:
        store.write_exchange_info({"BTCUSDT": info()}, source="test")
    return store


# --------------------------------------------------------------------------- #
class TestIssue1FundingLoadsFromParquet:
    """The store wrote <SYMBOL>.parquet; the loader read <SYMBOL>.csv. Nothing
    raised — funding was silently zero in every backtest."""

    def test_funding_written_by_the_store_is_read_by_the_loader(self, tmp_path) -> None:
        rates = {START: 0.0001, START + FUNDING_STEP: -0.00005}
        dataset(tmp_path, funding=rates)

        data, _ = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        assert data["BTCUSDT"].funding_rates == rates, (
            "funding did not survive the store -> loader round trip"
        )

    def test_funding_is_not_silently_empty(self, tmp_path) -> None:
        """The assertion the original bug would fail."""
        dataset(tmp_path, funding={START: 0.0001})
        data, _ = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        assert data["BTCUSDT"].funding_rates, "funding silently became empty"

    def test_legacy_csv_still_loads(self, tmp_path) -> None:
        import pandas as pd

        dataset(tmp_path, funding=None)
        path = tmp_path / "funding"
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"fundingTime": [START], "fundingRate": [0.0002]}).to_csv(
            path / "BTCUSDT.csv", index=False
        )

        data, _ = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        assert data["BTCUSDT"].funding_rates == {START: 0.0002}

    def test_absent_funding_is_empty_not_an_error(self, tmp_path) -> None:
        dataset(tmp_path, funding=None)
        data, _ = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        assert data["BTCUSDT"].funding_rates == {}


class TestIssue11FundingUsesRealEvents:
    """Not "every 8 hours since entry"."""

    def _engine(self) -> BacktestEngine:
        engine = BacktestEngine(CONFIG)
        return engine

    def _position(self, opened_at: int, direction=Direction.LONG) -> Position:
        return Position(
            position_id="p",
            symbol="BTCUSDT",
            direction=direction,
            quantity=1.0,
            entry_price=100.0,
            leverage=2,
            stop_loss=99.0,
            take_profit=102.0,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opened_at=opened_at,
        )

    def _data(self, rates: dict[int, float]) -> BacktestData:
        return BacktestData("BTCUSDT", {"5m": bars(10, 300_000)}, info(), rates)

    def test_a_position_opened_after_an_event_does_not_pay_it(self) -> None:
        engine = self._engine()
        position = self._position(opened_at=START + 1)  # just after 00:00 funding
        data = self._data({START: 0.001})
        engine._apply_funding(position, data, START + 3_600_000)
        assert position.funding_paid == 0.0, "charged for an event it missed"

    def test_a_position_open_across_an_event_pays_it(self) -> None:
        engine = self._engine()
        position = self._position(opened_at=START - 1000)
        data = self._data({START: 0.001})
        engine._apply_funding(position, data, START + 1000)
        assert position.funding_paid == pytest.approx(100.0 * 0.001)

    def test_a_position_opened_exactly_at_the_event_does_not_pay_it(self) -> None:
        """The exchange's snapshot is taken at that instant; we were not in it."""
        engine = self._engine()
        position = self._position(opened_at=START)
        data = self._data({START: 0.001})
        engine._apply_funding(position, data, START + 1000)
        assert position.funding_paid == 0.0

    def test_just_before_the_event_pays_nothing_yet(self) -> None:
        engine = self._engine()
        position = self._position(opened_at=START - 10_000)
        data = self._data({START: 0.001})
        engine._apply_funding(position, data, START - 1)
        assert position.funding_paid == 0.0

    def test_a_short_receives_positive_funding(self) -> None:
        engine = self._engine()
        position = self._position(opened_at=START - 1000, direction=Direction.SHORT)
        data = self._data({START: 0.001})
        engine._apply_funding(position, data, START + 1000)
        assert position.funding_paid < 0, "a short pays negative funding, i.e. receives"

    def test_multiple_events_are_all_charged_exactly_once(self) -> None:
        engine = self._engine()
        position = self._position(opened_at=START - 1000)
        rates = {START: 0.001, START + FUNDING_STEP: 0.001, START + 2 * FUNDING_STEP: 0.001}
        data = self._data(rates)

        engine._apply_funding(position, data, START + 2 * FUNDING_STEP + 1)
        assert position.funding_paid == pytest.approx(3 * 100.0 * 0.001)

        # A second sweep over the same window must not double-charge.
        engine._apply_funding(position, data, START + 2 * FUNDING_STEP + 2)
        assert position.funding_paid == pytest.approx(3 * 100.0 * 0.001)

    def test_no_funding_data_means_no_charge_rather_than_a_guess(self) -> None:
        engine = self._engine()
        position = self._position(opened_at=START - 1000)
        engine._apply_funding(position, self._data({}), START + 10 * FUNDING_STEP)
        assert position.funding_paid == 0.0

    def test_the_model_is_not_interval_since_entry(self) -> None:
        """The defect: 8h after entry rather than at the exchange's timestamps."""
        source = inspect.getsource(BacktestEngine._apply_funding)
        assert "funding_interval_hours" not in source
        assert "data.funding_rates" in source


class TestIssue3RealExchangeInfo:
    def test_stored_filters_reach_the_backtest(self, tmp_path) -> None:
        dataset(tmp_path)
        data, _ = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        loaded = data["BTCUSDT"].symbol_info
        assert loaded.tick_size == 0.10, "placeholder filters were used instead"
        assert loaded.step_size == 0.001
        assert loaded.min_qty == 0.002
        assert loaded.min_notional == 7.0
        assert loaded.market_min_qty == 0.004
        assert loaded.max_leverage == 125

    def test_placeholders_are_used_only_when_nothing_was_stored(self, tmp_path) -> None:
        dataset(tmp_path, exchange_info=False)
        data, _ = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        assert data["BTCUSDT"].symbol_info.tick_size == 1e-8  # the placeholder

    def test_missing_exchange_info_downgrades_trust(self, tmp_path) -> None:
        """It must not silently proceed as if the filters were real."""
        dataset(tmp_path, exchange_info=False)
        data, quality = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        report = evaluate_trust(
            data, quality, ["5m"], funding_enabled=False, have_exchange_info=False
        )
        assert report.level is TrustLevel.UNTRUSTED
        assert any("exchangeInfo" in d for d in report.downgrades)


class TestIssue4EquityMarkingTiming:
    """A newly opened position must not be marked at a price from before it
    existed — that books the entry gap as instant PnL."""

    def test_a_position_opened_this_cycle_is_not_marked(self) -> None:
        engine = BacktestEngine(CONFIG, initial_capital=100.0)
        engine.decision_interval = "5m"
        engine.candles.append("BTCUSDT", "5m", bars(1, 300_000)[0])
        engine.positions["BTCUSDT"] = Position(
            position_id="p",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=110.0,  # a big gap above the previous close of 100
            leverage=1,
            stop_loss=100.0,
            take_profit=120.0,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opened_at=START + 300_000,
        )
        engine._record_equity({}, START + 300_000)
        assert engine.equity == pytest.approx(100.0), (
            "the entry gap was booked as instant PnL before any bar closed"
        )

    def test_a_gap_does_not_distort_equity(self) -> None:
        """previous_close != next_open is the whole point."""
        engine = BacktestEngine(CONFIG, initial_capital=100.0)
        engine.decision_interval = "5m"
        engine.candles.append("BTCUSDT", "5m", bars(1, 300_000)[0])  # closes at 100
        for entry_price in (90.0, 100.0, 110.0):
            engine.positions = {
                "BTCUSDT": Position(
                    position_id="p",
                    symbol="BTCUSDT",
                    direction=Direction.LONG,
                    quantity=1.0,
                    entry_price=entry_price,
                    leverage=1,
                    stop_loss=1.0,
                    take_profit=999.0,
                    strategy="momentum",
                    regime=MarketRegime.STRONG_TREND,
                    opened_at=START + 300_000,
                )
            }
            engine.equity_curve.clear()
            engine._record_equity({}, START + 300_000)
            assert engine.equity == pytest.approx(100.0), (
                f"entry at {entry_price} distorted equity via the gap"
            )

    def test_a_position_from_an_earlier_cycle_is_marked_normally(self) -> None:
        engine = BacktestEngine(CONFIG, initial_capital=100.0)
        engine.decision_interval = "5m"
        engine.candles.append("BTCUSDT", "5m", bars(1, 300_000)[0])  # close 100
        engine.positions["BTCUSDT"] = Position(
            position_id="p",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=2.0,
            entry_price=95.0,
            leverage=1,
            stop_loss=90.0,
            take_profit=120.0,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opened_at=START - 1_000_000,
        )
        engine._record_equity({}, START + 300_000)
        assert engine.equity == pytest.approx(100.0 + (100.0 - 95.0) * 2.0)


class TestIssue5SamplingInterval:
    """Sharpe scales with the square root of samples per year; a factor-of-50
    error in the interval misstates it by ~7x."""

    @pytest.mark.parametrize(
        ("timeframe", "step_ms"), [("1m", 60_000), ("5m", 300_000), ("15m", 900_000)]
    )
    def test_the_interval_is_measured_from_the_curve(self, timeframe, step_ms) -> None:
        from tradebot.backtesting.metrics import EquityPoint

        engine = BacktestEngine(CONFIG)
        engine.equity_curve = [EquityPoint(START + i * step_ms, 100.0) for i in range(50)]
        assert engine._equity_sampling_sec() == pytest.approx(step_ms / 1000.0)

    def test_one_long_gap_does_not_distort_it(self) -> None:
        """Median, not mean: a data hole would drag every derived ratio."""
        from tradebot.backtesting.metrics import EquityPoint

        engine = BacktestEngine(CONFIG)
        stamps = [START + i * 60_000 for i in range(40)]
        stamps.append(stamps[-1] + 30 * 86_400_000)  # a month-long hole
        engine.equity_curve = [EquityPoint(t, 100.0) for t in stamps]
        assert engine._equity_sampling_sec() == pytest.approx(60.0)

    def test_the_hardcoded_multiplier_is_gone(self) -> None:
        assert "* 50" not in inspect.getsource(BacktestEngine._result)


class TestIssue6Attribution:
    """The same trade could be attributed to two different strategies."""

    def _signal(self, contributing: tuple[Signal, ...]) -> AggregatedSignal:
        return AggregatedSignal(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            consensus_score=80.0,
            confidence=80.0,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=102.0,
            contributing=contributing,
            opposing=(),
            conflict_ratio=0.0,
            regime=MarketRegime.STRONG_TREND,
            timestamp=0,
        )

    def _contributor(self, name: str, confidence: float) -> Signal:
        return Signal(
            symbol="BTCUSDT",
            strategy=name,
            direction=Direction.LONG,
            confidence=confidence,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=102.0,
            timeframe="5m",
            signal_timestamp=0,
        )

    def test_attribution_does_not_depend_on_tuple_order(self) -> None:
        """The regression. contributing[0] vs highest-confidence disagreed."""
        low = self._contributor("vwap", 60.0)
        high = self._contributor("momentum", 90.0)

        forward = self._signal((low, high))
        reversed_ = self._signal((high, low))
        assert forward.primary_strategy == reversed_.primary_strategy == "momentum"

    def test_ties_break_deterministically_on_name(self) -> None:
        a = self._contributor("breakout", 75.0)
        b = self._contributor("momentum", 75.0)
        assert self._signal((a, b)).primary_strategy == self._signal((b, a)).primary_strategy

    def test_the_edge_calculator_uses_the_same_primary(self) -> None:
        source = inspect.getsource(
            __import__("tradebot.signals.edge", fromlist=["x"]).EdgeCalculator.estimate
        )
        # Comments stripped: the fix's own explanation mentions the old code.
        code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
        assert "signal.primary_strategy" in code
        assert "contributing[0]" not in code

    def test_the_opportunity_uses_the_same_primary(self) -> None:
        from tradebot.signals.pipeline import Opportunity

        assert "primary_strategy" in inspect.getsource(Opportunity.strategy.fget)  # type: ignore[union-attr]

    def test_supporting_strategies_are_recorded_separately(self) -> None:
        """A single name cannot express a multi-strategy consensus."""
        signal = self._signal(
            (self._contributor("momentum", 90.0), self._contributor("vwap", 60.0))
        )
        assert signal.contributing_strategies == ("momentum", "vwap")
        weights = signal.contribution_weights
        assert weights["momentum"] > weights["vwap"]
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_no_contributors_is_unknown_not_a_crash(self) -> None:
        assert self._signal(()).primary_strategy == "unknown"
        assert self._signal(()).contribution_weights == {}


class TestIssue10CostLedger:
    """One coherent accounting, so a report cannot double-count."""

    def test_the_identity_holds(self) -> None:
        trade = Trade(
            trade_id="t",
            symbol="BTCUSDT",
            strategy="momentum",
            direction=Direction.LONG,
            entry_price=100.5,
            exit_price=101.4,
            quantity=1.0,
            leverage=2,
            stop_loss=99.0,
            take_profit=102.0,
            opened_at=0,
            closed_at=1000,
            gross_pnl=0.9,
            fees=0.08,
            funding=0.02,
            slippage_cost=0.06,
            net_pnl=0.82,
            exit_reason=ExitReason.TAKE_PROFIT,
            regime=MarketRegime.STRONG_TREND,
            reference_gross_pnl=1.0,
            entry_fee=0.04,
            exit_fee=0.04,
            spread_cost=0.02,
            entry_slippage=0.03,
            exit_slippage=0.03,
            latency_cost=0.0,
        )
        assert trade.execution_costs == pytest.approx(0.08)
        assert trade.total_cost == pytest.approx(0.18)
        assert trade.cost_identity_error() == pytest.approx(0.0, abs=1e-9)

    def test_a_broken_ledger_is_detectable(self) -> None:
        """The check must actually catch a missing component."""
        trade = Trade(
            trade_id="t",
            symbol="BTCUSDT",
            strategy="momentum",
            direction=Direction.LONG,
            entry_price=100.0,
            exit_price=101.0,
            quantity=1.0,
            leverage=1,
            stop_loss=99.0,
            take_profit=102.0,
            opened_at=0,
            closed_at=1,
            gross_pnl=1.0,
            fees=0.1,
            funding=0.0,
            slippage_cost=0.0,
            net_pnl=1.0,
            exit_reason=ExitReason.TAKE_PROFIT,
            regime=MarketRegime.SIDEWAYS,
            reference_gross_pnl=1.0,
            entry_fee=0.05,
            exit_fee=0.05,
        )
        assert trade.cost_identity_error() != 0.0

    def test_a_legacy_trade_without_a_ledger_reports_no_error(self) -> None:
        trade = Trade(
            trade_id="t",
            symbol="X",
            strategy="s",
            direction=Direction.LONG,
            entry_price=1.0,
            exit_price=1.0,
            quantity=1.0,
            leverage=1,
            stop_loss=0.9,
            take_profit=1.1,
            opened_at=0,
            closed_at=1,
            gross_pnl=0.0,
            fees=0.0,
            funding=0.0,
            slippage_cost=0.0,
            net_pnl=0.0,
            exit_reason=ExitReason.MANUAL,
            regime=MarketRegime.SIDEWAYS,
        )
        assert trade.cost_identity_error() == 0.0

    def test_the_engine_checks_the_identity(self) -> None:
        source = inspect.getsource(BacktestEngine._close_position)
        assert "identity_error" in source
        assert "cost_ledger_does_not_balance" in source


class TestIssue7DecisionCadence:
    def test_decisions_run_at_the_finest_available_timeframe(self) -> None:
        engine = BacktestEngine(CONFIG)
        data = {
            "BTCUSDT": BacktestData(
                "BTCUSDT",
                {"1m": bars(10, 60_000), "5m": bars(10, 300_000)},
                info(),
                {},
            )
        }
        assert engine.decision_timeframe(data) == "1m"

    def test_it_falls_back_to_what_exists(self) -> None:
        engine = BacktestEngine(CONFIG)
        data = {"BTCUSDT": BacktestData("BTCUSDT", {"15m": bars(10, 900_000)}, info(), {})}
        assert engine.decision_timeframe(data) == "15m"

    def test_the_limitation_is_documented_not_faked(self) -> None:
        """15s decisions cannot be reconstructed from 1m OHLCV."""
        source = inspect.getsource(BacktestEngine.decision_timeframe)
        assert "15-second" in source or "15s" in source
        assert "cannot be reconstructed" in source


class TestIssue8And12ModesAreDeclared:
    def test_strict_oos_freezes_learned_state(self) -> None:
        source = inspect.getsource(run_strict_oos)
        assert "export_stats" in source
        assert "seed_from" in source

    def test_a_continuous_run_is_never_called_a_holdout(self) -> None:
        from tradebot.backtesting.runner import split_continuous

        note = split_continuous(START).as_dict()["note"]
        assert "NOT a clean holdout" in note

    def test_both_modes_exist_and_are_distinct(self) -> None:
        assert OOSMode.STRICT_OOS != OOSMode.LIVE_LIKE_FORWARD
        assert EdgeMode.RESEARCH_STRICT != EdgeMode.LIVE_FAITHFUL

    def test_the_context_records_which_modes_produced_the_run(self) -> None:
        from tradebot.backtesting.runner import build_context

        payload = build_context(CONFIG, {}, seed=1).as_dict()
        for field in ("oos_mode", "edge_mode", "universe_provenance", "data_trust"):
            assert field in payload


class TestIssue9WalkForwardHonesty:
    def test_the_unused_window_is_called_an_embargo_not_validation(self) -> None:
        from tradebot.backtesting import walkforward

        source = inspect.getsource(walkforward)
        assert "embargo" in source.lower()
        assert "60/20/20" in source, "the doc must say what this is NOT"

    def test_no_parameter_optimisation_is_claimed(self) -> None:
        from tradebot.backtesting import walkforward

        assert "No parameter optimisation is performed" in inspect.getsource(walkforward)

    def test_the_fold_records_that_the_embargo_is_not_evaluated(self) -> None:
        from tradebot.backtesting.walkforward import Fold

        fold = Fold(
            index=0,
            train_start=0,
            train_end=1,
            embargo_start=1,
            embargo_end=2,
            test_start=2,
            test_end=3,
        )
        assert fold.as_dict()["embargo_is_evaluated"] is False


class TestIssue14DataQualityGate:
    def _data(self, tmp_path, **kwargs):
        dataset(tmp_path, **kwargs)
        return load_dataset(tmp_path, ["BTCUSDT"], ["1m", "3m", "5m", "15m", "1h"], strict=False)

    def test_complete_data_is_trusted(self, tmp_path) -> None:
        data, quality = self._data(tmp_path, funding={START: 0.0001})
        report = evaluate_trust(
            data,
            quality,
            ["1m", "3m", "5m", "15m", "1h"],
            funding_enabled=True,
            have_exchange_info=True,
        )
        assert report.level is TrustLevel.TRUSTED

    def test_missing_timeframes_are_refused(self, tmp_path) -> None:
        """A low trade count would mean NO DATA, not NO EDGE."""
        dataset(tmp_path, timeframes=("5m",))
        data, quality = load_dataset(tmp_path, ["BTCUSDT"], ["5m"], strict=False)
        report = evaluate_trust(
            data, quality, ["1m", "5m"], funding_enabled=False, have_exchange_info=True
        )
        assert report.level is TrustLevel.REFUSED
        assert not report.may_run

    def test_missing_funding_downgrades_when_funding_is_enabled(self, tmp_path) -> None:
        data, quality = self._data(tmp_path, funding=None)
        report = evaluate_trust(
            data,
            quality,
            ["1m", "3m", "5m", "15m", "1h"],
            funding_enabled=True,
            have_exchange_info=True,
        )
        assert report.level is TrustLevel.UNTRUSTED
        assert any("funding" in d for d in report.downgrades)

    def test_no_symbols_is_refused(self) -> None:
        assert evaluate_trust({}, [], ["5m"], False, True).level is TrustLevel.REFUSED

    def test_untrusted_runs_carry_a_banner(self, tmp_path) -> None:
        data, quality = self._data(tmp_path, funding=None)
        report = evaluate_trust(
            data,
            quality,
            ["1m", "3m", "5m", "15m", "1h"],
            funding_enabled=True,
            have_exchange_info=False,
        )
        assert "not evidence" in report.banner()


class TestIssue2CLIUsesTheScenarioRunner:
    """Not merely that run_scenarios() works — that the CLI path calls it."""

    def _calls(self, source: str) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(ast.parse(textwrap.dedent(source))):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    names.add(func.id)
                elif isinstance(func, ast.Attribute):
                    names.add(func.attr)
        return names

    def test_the_cli_calls_run_scenarios(self) -> None:
        from tradebot.app.commands import run_backtest

        calls = self._calls(inspect.getsource(run_backtest))
        assert "run_scenarios" in calls
        assert "BacktestEngine" not in calls, "the CLI still runs a single engine"

    def test_the_cli_evaluates_trust_before_running(self) -> None:
        """V3.2 moved the gate into the shared `_load_and_trust` helper so that
        `walkforward` goes through the same rules. The ordering this test was
        written for is unchanged: nothing runs before trust is decided."""
        from tradebot.app.commands import _load_and_trust, run_backtest

        source = inspect.getsource(run_backtest)
        assert "_load_and_trust(" in source
        # Compare CALL sites, not the import block at the top.
        assert source.index("_load_and_trust(") < source.index("run_scenarios(")
        assert "evaluate_trust(" in inspect.getsource(_load_and_trust)

    def test_the_cli_refuses_when_trust_says_so(self) -> None:
        from tradebot.app.commands import _load_and_trust

        assert "trust.may_run" in inspect.getsource(_load_and_trust)

    def test_the_cli_loads_stored_exchange_info(self) -> None:
        from tradebot.app.commands import _load_and_trust

        assert "load_exchange_info" in inspect.getsource(_load_and_trust)

    def test_the_cli_offers_strict_oos(self) -> None:
        from tradebot.app.cli import build_parser

        action = next(
            a
            for a in build_parser()._subparsers._group_actions  # type: ignore[union-attr]
        )
        backtest = action.choices["backtest"]
        flags = {o for a in backtest._actions for o in a.option_strings}
        assert "--strict-oos" in flags
        assert "--allow-degraded" in flags
        assert "--edge-mode" in flags
        assert "--universe" in flags


class TestArgumentsAreValidatedBeforeWork:
    """Found while verifying V3.1: `--split` was parsed AFTER the scenarios
    ran, so a mistyped date threw away 36 seconds of completed work. A CLI that
    does expensive work and then rejects an argument is reporting a typo in the
    most expensive way available."""

    def test_every_date_is_parsed_before_the_data_is_loaded(self) -> None:
        from tradebot.app.commands import run_backtest

        source = inspect.getsource(run_backtest)
        # The load itself moved into `_load_and_trust` in V3.2; the property
        # under test is the same one — no date is parsed after work has begun.
        assert source.index("_parse_date(args.split)") < source.index("_load_and_trust(")
        assert source.index("_parse_date(args.start)") < source.index("_load_and_trust(")

    def test_the_iso_t_separator_is_accepted(self) -> None:
        """It is what every other tool prints."""
        from tradebot.app.commands import _parse_date

        assert _parse_date("2024-01-01T18:00:00") == _parse_date("2024-01-01 18:00:00")

    @pytest.mark.parametrize(
        "text",
        ["2024-01-01", "2024-01-01 18:00", "2024-01-01T18:00", "2024-01-01T18:00:00"],
    )
    def test_accepted_forms(self, text: str) -> None:
        from tradebot.app.commands import _parse_date

        assert isinstance(_parse_date(text), int)

    def test_a_bad_date_names_the_accepted_forms(self) -> None:
        from tradebot.app.commands import _parse_date

        with pytest.raises(SystemExit, match="accepted forms"):
            _parse_date("not-a-date")

    def test_an_empty_date_is_none_not_an_error(self) -> None:
        from tradebot.app.commands import _parse_date

        assert _parse_date("") is None

    def test_incoherent_ranges_are_rejected_up_front(self) -> None:
        from tradebot.app.commands import run_backtest

        source = inspect.getsource(run_backtest)
        assert "--end must be after --start" in source
        assert "--split must be after --start" in source
        assert "--split must be before --end" in source
