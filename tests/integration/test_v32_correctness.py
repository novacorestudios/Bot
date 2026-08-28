"""V3.2 trust-and-timing patch: one regression test per fixed defect.

Two of the V3.2 defects were **completely silent**. The trust gate read three
attributes the loader's quality object did not have, through
``getattr(q, "status", None)``, so every check evaluated to ``False`` and a
dataset with impossible OHLC came back ``TRUSTED``. The backtest filled at the
next bar's open while stamping the position with the *signal* timestamp, so
every trade appeared to have been opened one decision interval before it was.

Neither produced an error, a warning, or an implausible number. So several of
the tests below are structural — they assert on what the code *is*, not only on
what it returns — because an output-only test would have passed with the bug in
place.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tradebot.backtesting.data import DataQuality, load_candles, load_dataset
from tradebot.backtesting.engine import BacktestData, BacktestEngine
from tradebot.backtesting.execution import Scenario
from tradebot.backtesting.trust import DatasetQuality, TrustLevel, evaluate_trust
from tradebot.backtesting.walkforward import WalkForwardAnalyzer
from tradebot.core.config import load_tunables
from tradebot.core.types import Candle, Direction, MarketRegime, Position, SymbolInfo, Trade
from tradebot.data.store import DataStore
from tradebot.data.validation import QualityStatus, ValidationReport

from ..conftest import REPO_ROOT

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.backtest.yaml", REPO_ROOT / "config" / "strategies.yaml"
)
START = 1_704_067_200_000
MINUTE = 60_000


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


TIMEFRAMES = ("1m", "3m", "5m", "15m", "1h")
STEPS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def dataset(tmp_path, *, timeframes=TIMEFRAMES, funding=None, exchange_info=True) -> DataStore:
    """A clean on-disk dataset, written by the real DataStore."""
    store = DataStore(tmp_path)
    for timeframe in timeframes:
        store.write_klines("BTCUSDT", timeframe, bars(400, STEPS[timeframe]), source="test")
    store.write_funding("BTCUSDT", funding if funding is not None else {START: 0.0001}, "test")
    if exchange_info:
        store.write_exchange_info({"BTCUSDT": info()}, source="test")
    return store


def damage(tmp_path, timeframe: str, mutate) -> None:
    """Rewrite one kline file through pandas, applying `mutate` to the frame."""
    import pandas as pd

    path = tmp_path / "klines" / timeframe / "BTCUSDT.parquet"
    frame = pd.read_parquet(path)
    mutate(frame).to_parquet(path, index=False)


def run_cli(tmp_path, command: str, *extra: str) -> subprocess.CompletedProcess[str]:
    """Drive the REAL command-line entry point in a subprocess.

    Not the helper classes: the defects this file guards were in the wiring
    between them, which is exactly what a unit test of either side misses.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tradebot.app.cli",
            "--log-level",
            "ERROR",
            command,
            "--data",
            str(tmp_path),
            "--symbols",
            "BTCUSDT",
            "--report",
            str(tmp_path / "report.json"),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "CONFIG_FILE": "config/config.backtest.yaml",
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "TRADING_MODE": "PAPER",
        },
        timeout=600,
    )


# --------------------------------------------------------------------------- #
class TestP0TheQualityContractIsShared:
    """`evaluate_trust` read `status`, `interval` and `missing_bars` off an
    object that had none of them. `getattr(q, "status", None)` made every guard
    falsy, so no dataset was ever refused or downgraded for its own contents."""

    def test_both_quality_classes_satisfy_one_contract(self) -> None:
        loader_side = DataQuality(symbol="BTCUSDT", timeframe="1m", bars=10)
        pipeline_side = ValidationReport(symbol="BTCUSDT", interval="1m")
        assert isinstance(loader_side, DatasetQuality)
        assert isinstance(pipeline_side, DatasetQuality)

    def test_the_loader_quality_reports_the_shared_status_vocabulary(self) -> None:
        """Not a private enum of its own: the same QualityStatus the pipeline
        uses, so the gate cannot be handed two incompatible notions of OK."""
        clean = DataQuality(symbol="BTCUSDT", timeframe="1m", bars=10)
        gapped = DataQuality(symbol="BTCUSDT", timeframe="1m", bars=10, missing_bars=4)
        broken = DataQuality(symbol="BTCUSDT", timeframe="1m", bars=10, problems=["bad"])
        assert clean.status is QualityStatus.OK
        assert gapped.status is QualityStatus.DEGRADED
        assert broken.status is QualityStatus.UNUSABLE

    def test_an_empty_series_is_unusable_on_its_own(self) -> None:
        assert (
            DataQuality(symbol="BTCUSDT", timeframe="1m", bars=0).status is QualityStatus.UNUSABLE
        )

    def test_interval_and_timeframe_are_the_same_string(self) -> None:
        q = DataQuality(symbol="BTCUSDT", timeframe="15m", bars=1)
        assert q.interval == q.timeframe == "15m"

    def test_the_gate_no_longer_reads_attributes_defensively(self) -> None:
        """The literal mechanism of the bug: a `getattr` default that turned a
        schema mismatch into `False` instead of an error."""
        import inspect

        from tradebot.backtesting import trust

        source = inspect.getsource(trust.evaluate_trust)
        assert 'getattr(q, "status"' not in source
        assert 'getattr(q, "missing_bars"' not in source


class TestP0DamagedDataIsRefused:
    """The four deliberate corruptions the brief names, each driven through the
    real `tradebot backtest` command."""

    def _quality(self, tmp_path, timeframe: str = "1m") -> DataQuality:
        _, quality = load_candles(tmp_path / "klines" / timeframe / "BTCUSDT.parquet", timeframe)
        return quality

    def test_impossible_ohlc_is_detected(self, tmp_path) -> None:
        dataset(tmp_path)

        def break_ohlc(frame):
            frame.loc[5, "high"] = frame.loc[5, "low"] - 10.0
            return frame

        damage(tmp_path, "1m", break_ohlc)
        quality = self._quality(tmp_path)
        assert quality.inconsistent_ohlc >= 1
        assert quality.status is QualityStatus.UNUSABLE

    def test_an_open_outside_the_bar_range_is_detected(self, tmp_path) -> None:
        """Checking only against `close` let a bar whose OPEN sat outside its
        own high/low pass as consistent."""
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.assign(open=f["open"].where(f.index != 7, 1e6)))
        assert self._quality(tmp_path).status is QualityStatus.UNUSABLE

    def test_a_non_positive_price_is_detected(self, tmp_path) -> None:
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.assign(low=f["low"].where(f.index != 9, 0.0)))
        quality = self._quality(tmp_path)
        assert quality.non_positive_prices >= 1
        assert quality.status is QualityStatus.UNUSABLE

    def test_a_gap_beyond_tolerance_is_fatal(self, tmp_path) -> None:
        """100 missing 1-minute bars is an hour and a half of invented
        continuity. Indicators computed across it are wrong, so the dataset
        fails rather than degrading."""
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.drop(index=range(20, 120)).reset_index(drop=True))
        quality = self._quality(tmp_path)
        assert quality.missing_bars >= 100
        assert quality.largest_gap_bars > 10
        assert quality.status is QualityStatus.UNUSABLE

    def test_a_small_gap_degrades_rather_than_fails(self, tmp_path) -> None:
        """Exchanges have brief outages. Refusing every one of them would
        reject all real history, so a gap inside tolerance is measured and
        reported instead."""
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.drop(index=range(20, 25)).reset_index(drop=True))
        quality = self._quality(tmp_path)
        assert quality.missing_bars == 5
        assert quality.status is QualityStatus.DEGRADED

    def test_duplicate_timestamps_are_reported_not_silently_dropped(self, tmp_path) -> None:
        """The duplicates were always removed. Nothing ever said so, and a file
        that needed repairing was indistinguishable from a clean one.

        The copies are inserted next to their originals so the file stays in
        chronological order — this isolates duplication from the separate,
        fatal condition below."""
        import pandas as pd

        dataset(tmp_path)
        damage(
            tmp_path,
            "1m",
            lambda f: (
                pd.concat([f, f.iloc[10:15]])
                .sort_values("open_time", kind="stable")
                .reset_index(drop=True)
            ),
        )
        quality = self._quality(tmp_path)
        assert quality.duplicates_removed == 5
        assert quality.out_of_order == 0
        assert quality.status is QualityStatus.DEGRADED

    def test_shuffled_rows_are_fatal_not_merely_degraded(self, tmp_path) -> None:
        """Sorting would hide it. A file that arrives out of order came from a
        broken writer, and nothing else it contains can be trusted either."""
        import pandas as pd

        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: pd.concat([f.iloc[200:], f.iloc[:200]], ignore_index=True))
        quality = self._quality(tmp_path)
        assert quality.out_of_order >= 1
        assert quality.status is QualityStatus.UNUSABLE

    def test_the_cli_refuses_impossible_ohlc(self, tmp_path) -> None:
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.assign(high=f["high"].where(f.index != 5, 0.001)))
        result = run_cli(tmp_path, "backtest")
        assert result.returncode == 1, result.stdout[-3000:]
        assert "REFUSED" in result.stdout
        assert "structurally corrupt" in result.stdout

    def test_the_cli_refuses_a_missing_timeframe(self, tmp_path) -> None:
        dataset(tmp_path, timeframes=("1m", "3m", "5m", "15m"))
        result = run_cli(tmp_path, "backtest")
        assert result.returncode == 1, result.stdout[-3000:]
        assert "REFUSED" in result.stdout

    def test_the_cli_refuses_a_gap_until_it_is_acknowledged(self, tmp_path) -> None:
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.drop(index=range(30, 35)).reset_index(drop=True))
        refused = run_cli(tmp_path, "backtest")
        assert refused.returncode == 1, refused.stdout[-3000:]
        assert "DEGRADED" in refused.stdout

    def test_allow_degraded_runs_a_gap_but_never_says_trusted(self, tmp_path) -> None:
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.drop(index=range(30, 35)).reset_index(drop=True))
        allowed = run_cli(tmp_path, "backtest", "--allow-degraded")
        assert allowed.returncode == 0, allowed.stdout[-3000:]
        assert "UNTRUSTED" in allowed.stdout
        assert "Data trust          TRUSTED" not in allowed.stdout

    def test_allow_degraded_cannot_rescue_structural_corruption(self, tmp_path) -> None:
        """Before V3.2 it could: one flag turned a refusal into a running
        backtest over bars with impossible OHLC."""
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.assign(high=f["high"].where(f.index != 5, 0.001)))
        result = run_cli(tmp_path, "backtest", "--allow-degraded")
        assert result.returncode == 1, result.stdout[-3000:]
        assert "REFUSED" in result.stdout

    def test_nothing_reaches_trusted_except_clean_data(self) -> None:
        """The property that matters: an override may move REFUSED to
        UNTRUSTED, and nothing whatsoever moves anything to TRUSTED."""

        class Entry:
            candles = {"1m": [1]}
            funding_rates = {START: 0.0001}

        data = {"BTCUSDT": Entry()}
        dirty = [
            DataQuality(symbol="BTCUSDT", timeframe="1m", bars=5, problems=["impossible OHLC"]),
            DataQuality(symbol="BTCUSDT", timeframe="1m", bars=5, missing_bars=50),
            DataQuality(symbol="BTCUSDT", timeframe="1m", bars=0),
            DataQuality(symbol="BTCUSDT", timeframe="1m", bars=5, duplicates_removed=3),
        ]
        for quality in dirty:
            for flag in (False, True):
                report = evaluate_trust(
                    data=data,
                    quality=[quality],
                    required_timeframes=["1m"],
                    funding_enabled=True,
                    have_exchange_info=True,
                    allow_degraded=flag,
                )
                assert report.level is not TrustLevel.TRUSTED, (quality.status, flag)


class TestP0WalkForwardUsesTheSameGate:
    """`walkforward` loaded with `strict=False` and never evaluated trust, so
    the dataset `backtest` refused ran to a printed verdict here."""

    def test_the_command_shares_one_trust_implementation(self) -> None:
        import inspect

        from tradebot.app import commands

        source = inspect.getsource(commands.run_walkforward)
        assert "_load_and_trust" in source
        # Not a second copy of the rules.
        assert "evaluate_trust(" not in source

    def test_the_cli_refuses_corrupt_data(self, tmp_path) -> None:
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.assign(close=f["close"].where(f.index != 3, -5.0)))
        result = run_cli(tmp_path, "walkforward")
        assert result.returncode == 1, result.stdout[-3000:]
        assert "REFUSED" in result.stdout

    def test_the_report_carries_the_trust_level(self, tmp_path) -> None:
        dataset(tmp_path)
        result = run_cli(tmp_path, "walkforward")
        assert result.returncode == 0, result.stdout[-3000:]
        assert "Data trust" in result.stdout
        payload = json.loads((tmp_path / "report.json").read_text())
        assert payload["data_trust"] in {"TRUSTED", "UNTRUSTED"}

    def test_a_report_with_no_trust_evaluation_says_so(self) -> None:
        """A direct caller that skips the gate must not read as trusted."""
        report = WalkForwardAnalyzer(CONFIG).run({})
        assert report.trust_level == "NOT EVALUATED"


class TestP1WalkForwardMatchesTheBacktest:
    """It called `BacktestEngine(config).run(data, start, end)` — no capital, no
    scenario, no seed — so it silently ran seed 0 while the headline backtest
    ran the configured seed."""

    def test_execution_inputs_are_explicit_and_recorded(self) -> None:
        analyzer = WalkForwardAnalyzer(CONFIG, seed=42, scenario=Scenario.CONSERVATIVE)
        assert analyzer.seed == 42
        assert analyzer.assumptions.name is Scenario.CONSERVATIVE
        assert analyzer.initial_capital == CONFIG.account.initial_capital

    def test_the_seed_reaches_the_engine(self) -> None:
        """Structural: the engine must be handed the seed and the assumptions,
        not left to its own defaults."""
        import inspect

        source = inspect.getsource(WalkForwardAnalyzer.run)
        assert "seed=self.seed" in source
        assert "self.assumptions" in source

    def test_capital_matches_the_headline_run(self) -> None:
        assert WalkForwardAnalyzer(CONFIG)._engine().initial_capital == (
            CONFIG.account.initial_capital
        )

    def test_the_one_intentional_difference_is_documented(self) -> None:
        """One scenario per fold, not three — stated in the docstring so the
        comparison with `backtest` is never implicit."""
        doc = WalkForwardAnalyzer.__init__.__doc__ or ""
        assert "intentionally differs" in doc
        assert "identical" in doc


class TestP0OpenedAtIsTheFillTime:
    """`opened_at` held the signal timestamp while the fill happened at the next
    decision bar's open — so duration, the maximum-hold cap and funding
    eligibility all measured from a moment the position did not yet exist."""

    def _engine_and_data(self) -> tuple[BacktestEngine, BacktestData]:
        engine = BacktestEngine(CONFIG)
        engine.decision_interval = "1m"
        candles = bars(10, MINUTE)
        data = BacktestData(
            symbol="BTCUSDT",
            candles={"1m": candles},
            symbol_info=info(),
            funding_rates={},
        )
        return engine, data

    def test_the_fill_is_the_next_bar_and_carries_its_timestamp(self) -> None:
        engine, data = self._engine_and_data()
        signal_at = START + 3 * MINUTE
        filled_at, price = engine._next_fill(data, signal_at)
        assert filled_at == signal_at + MINUTE
        assert price == data.candles["1m"][4].open

    def test_signal_at_noon_fills_at_one_minute_past(self) -> None:
        """The brief's own example: signal 12:00, fill 12:01, opened_at 12:01."""
        engine, data = self._engine_and_data()
        noon = START + 2 * MINUTE
        filled_at, _ = engine._next_fill(data, noon)
        assert filled_at == noon + MINUTE
        assert filled_at != noon

    def test_the_engine_stamps_the_fill_not_the_signal(self) -> None:
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(BacktestEngine._open_position))
        tree = ast.parse(source)
        assigned = {
            keyword.arg: keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Position"
            for keyword in node.keywords
        }
        assert isinstance(assigned["opened_at"], ast.Name)
        assert assigned["opened_at"].id == "filled_at"
        assert assigned["signal_at"].id == "timestamp"

    def test_the_fill_timestamp_is_deterministic(self) -> None:
        engine, data = self._engine_and_data()
        at = START + MINUTE
        assert engine._next_fill(data, at) == engine._next_fill(data, at)

    def test_no_bar_after_the_signal_means_no_fill(self) -> None:
        engine, data = self._engine_and_data()
        assert engine._next_fill(data, START + 100 * MINUTE) is None

    def test_duration_is_measured_from_the_fill(self) -> None:
        position = Position(
            position_id="p",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            leverage=5,
            stop_loss=99.0,
            take_profit=102.0,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opened_at=START + MINUTE,
            signal_at=START,
        )
        # 60s after the fill, not 120s after the signal.
        assert position.duration_sec(START + 2 * MINUTE) == 60.0

    def test_the_max_hold_boundary_uses_the_fill(self) -> None:
        """The cap is 3600s. Measured from the signal, a position filled a
        minute later would be force-closed 60 seconds early — every time."""
        cap = CONFIG.trade.max_duration_sec
        position = Position(
            position_id="p",
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            leverage=5,
            stop_loss=99.0,
            take_profit=102.0,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opened_at=START + MINUTE,
            signal_at=START,
        )
        just_inside = START + MINUTE + (cap - 1) * 1000
        exactly_at = START + MINUTE + cap * 1000
        assert position.duration_sec(just_inside) < cap
        assert position.duration_sec(exactly_at) == pytest.approx(float(cap))

    def test_a_trade_records_the_whole_timeline(self) -> None:
        trade = Trade(
            trade_id="t",
            symbol="BTCUSDT",
            strategy="momentum",
            direction=Direction.LONG,
            entry_price=100.0,
            exit_price=101.0,
            quantity=1.0,
            leverage=5,
            stop_loss=99.0,
            take_profit=102.0,
            opened_at=START + MINUTE,
            closed_at=START + 5 * MINUTE,
            gross_pnl=1.0,
            fees=0.0,
            funding=0.0,
            slippage_cost=0.0,
            net_pnl=1.0,
            exit_reason=__import__(
                "tradebot.core.types", fromlist=["ExitReason"]
            ).ExitReason.TAKE_PROFIT,
            regime=MarketRegime.STRONG_TREND,
            signal_at=START,
        )
        assert trade.signal_to_fill_sec == 60.0
        assert trade.duration_sec == 240.0

    def test_an_unrecorded_timeline_is_zero_not_a_guess(self) -> None:
        trade = Trade(
            trade_id="t",
            symbol="BTCUSDT",
            strategy="momentum",
            direction=Direction.LONG,
            entry_price=100.0,
            exit_price=101.0,
            quantity=1.0,
            leverage=5,
            stop_loss=99.0,
            take_profit=102.0,
            opened_at=START + MINUTE,
            closed_at=START + 5 * MINUTE,
            gross_pnl=1.0,
            fees=0.0,
            funding=0.0,
            slippage_cost=0.0,
            net_pnl=1.0,
            exit_reason=__import__(
                "tradebot.core.types", fromlist=["ExitReason"]
            ).ExitReason.TAKE_PROFIT,
            regime=MarketRegime.STRONG_TREND,
        )
        assert trade.signal_to_fill_sec == 0.0


class TestP1FundingComesFromTheRealSchedule:
    """`_funding_rate` snapped the timestamp onto an assumed 8-hour grid and
    looked that bucket up. Real event timestamps do not land on the grid, so the
    lookup missed and returned 0.0 — a symbol with a full funding history was
    priced as if funding did not exist."""

    def _data(self, funding: dict[int, float]) -> BacktestData:
        return BacktestData(
            symbol="BTCUSDT",
            candles={"1m": bars(10, MINUTE)},
            symbol_info=info(),
            funding_rates=funding,
        )

    def test_an_off_grid_event_is_found(self) -> None:
        off_grid = START + 3_600_000 + 137  # nowhere near 00:00/08:00/16:00
        engine = BacktestEngine(CONFIG)
        data = self._data({off_grid: 0.00042})
        assert engine._funding_rate(data, off_grid + 60_000) == pytest.approx(0.00042)

    def test_the_rate_is_the_most_recent_settled_one(self) -> None:
        engine = BacktestEngine(CONFIG)
        data = self._data({START: 0.0001, START + 4 * 3_600_000: 0.0009})
        assert engine._funding_rate(data, START + 2 * 3_600_000) == pytest.approx(0.0001)
        assert engine._funding_rate(data, START + 5 * 3_600_000) == pytest.approx(0.0009)

    def test_before_the_first_event_there_is_no_rate(self) -> None:
        engine = BacktestEngine(CONFIG)
        data = self._data({START + 10 * 3_600_000: 0.0009})
        assert engine._funding_rate(data, START) == 0.0

    def test_seconds_to_funding_counts_to_the_actual_next_event(self) -> None:
        engine = BacktestEngine(CONFIG)
        next_event = START + 5_000_000
        data = self._data({START: 0.0001, next_event: 0.0002})
        assert engine._seconds_to_funding(data, START + 1_000_000) == pytest.approx(4000.0)

    def test_no_funding_history_is_infinity_not_a_fabricated_boundary(self) -> None:
        """The old code always returned a number, computed from a schedule it
        had invented. Infinity is the honest answer, and the trust gate
        separately downgrades the run for the missing history."""
        engine = BacktestEngine(CONFIG)
        assert engine._seconds_to_funding(self._data({}), START) == float("inf")

    def test_past_the_last_event_is_also_infinity(self) -> None:
        engine = BacktestEngine(CONFIG)
        data = self._data({START: 0.0001})
        assert engine._seconds_to_funding(data, START + 10 * 3_600_000) == float("inf")

    def test_the_grid_assumption_is_gone(self) -> None:
        import inspect

        for method in (BacktestEngine._funding_rate, BacktestEngine._seconds_to_funding):
            source = inspect.getsource(method)
            assert "funding_interval_hours" not in source, method.__name__
            assert "data.funding_rates" in source or "_funding_schedule" in source


class TestP1TheQualityArtifact:
    """Every run leaves a machine-readable record of the data behind it."""

    def test_the_artifact_is_written_beside_the_report(self, tmp_path) -> None:
        dataset(tmp_path)
        result = run_cli(tmp_path, "backtest")
        assert result.returncode == 0, result.stdout[-3000:]
        artifact = tmp_path / "report.data_quality.json"
        assert artifact.is_file()

    def test_it_carries_every_documented_column(self, tmp_path) -> None:
        dataset(tmp_path)
        run_cli(tmp_path, "backtest")
        payload = json.loads((tmp_path / "report.data_quality.json").read_text())
        expected = {
            "SYMBOL",
            "INTERVAL",
            "START",
            "END",
            "ROWS",
            "MISSING",
            "DUPLICATES",
            "GAPS",
            "COVERAGE",
            "QUALITY_STATUS",
        }
        assert payload["rows"]
        for row in payload["rows"]:
            assert set(row) == expected

    def test_it_records_the_trust_verdict_too(self, tmp_path) -> None:
        dataset(tmp_path)
        run_cli(tmp_path, "backtest")
        payload = json.loads((tmp_path / "report.data_quality.json").read_text())
        assert payload["trust"]["level"] in {"TRUSTED", "UNTRUSTED"}

    def test_a_refused_run_still_leaves_the_evidence(self, tmp_path) -> None:
        """The artifact explains WHY the run was refused, so it must survive
        the refusal."""
        dataset(tmp_path)
        damage(tmp_path, "1m", lambda f: f.assign(high=f["high"].where(f.index != 5, 0.001)))
        result = run_cli(tmp_path, "backtest")
        assert result.returncode == 1
        payload = json.loads((tmp_path / "report.data_quality.json").read_text())
        assert payload["trust"]["level"] == "REFUSED"
        assert any(row["QUALITY_STATUS"] == "UNUSABLE" for row in payload["rows"])


class TestTheRealCommandsRunEndToEnd:
    """Driving the documented commands, not the classes behind them."""

    def test_backtest_reports_trust_and_all_three_scenarios(self, tmp_path) -> None:
        dataset(tmp_path)
        result = run_cli(tmp_path, "backtest")
        assert result.returncode == 0, result.stdout[-3000:]
        assert "Data trust          TRUSTED" in result.stdout
        for scenario in ("BASE", "CONSERVATIVE", "STRESS"):
            assert scenario in result.stdout

    def test_strict_oos_labels_its_windows(self, tmp_path) -> None:
        dataset(tmp_path)
        split = "2024-01-01T03:00:00"
        result = run_cli(tmp_path, "backtest", "--split", split, "--strict-oos")
        assert result.returncode == 0, result.stdout[-3000:]
        assert "STRICT_OOS" in result.stdout
        assert "TRAIN" in result.stdout and "TEST" in result.stdout

    def test_no_bar_after_the_window_end_is_ever_used(self, tmp_path) -> None:
        """A cheap look-ahead check on the real loader: every bar the engine
        can see for a window lies inside it."""
        dataset(tmp_path)
        data, _ = load_dataset(tmp_path, ["BTCUSDT"], ["1m"], strict=False)
        end = START + 100 * MINUTE
        visible = [c for c in data["BTCUSDT"].candles["1m"] if c.close_time <= end]
        assert visible
        assert max(c.close_time for c in visible) <= end
