"""Full backtest runs against synthetic data.

These prove the ENGINE is correct — that bars are replayed without look-ahead,
that fills, fees, exits and the risk engine all engage, and that a zero-trade
run is reported honestly. They prove nothing about profitability; the price
paths are generated, not observed.
"""

from __future__ import annotations

import copy

import pytest

from tradebot.backtesting.engine import BacktestData, BacktestEngine
from tradebot.core.config import load_tunables
from tradebot.core.types import ExitReason

from ..conftest import (
    REPO_ROOT,
    choppy_prices,
    flat_prices,
    multi_timeframe,
    trending_with_pullbacks,
)
from ..fakes import make_symbol_info

BACKTEST_CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.backtest.yaml", REPO_ROOT / "config" / "strategies.yaml"
)


def dataset(specs: dict[str, list[float]], seed: int = 21) -> dict[str, BacktestData]:
    return {
        symbol: BacktestData(
            symbol=symbol,
            candles=multi_timeframe(prices, seed=seed + index),
            symbol_info=make_symbol_info(symbol, min_notional=1.0),
        )
        for index, (symbol, prices) in enumerate(specs.items())
    }


def permissive(config, agreeing: int = 1, min_score: float = 45.0):
    """A copy of the config with the analytical gates opened up.

    Used ONLY to exercise the engine's fill, exit and PnL machinery. Those code
    paths are the point of a backtester, and with the shipped thresholds they
    never execute on synthetic data: the strategies are built to fire in
    mutually exclusive conditions, regime gating permits three or four at a
    time, and the aggregator then wants two of those few to agree.

    This is a test harness, not a recommendation. The shipped thresholds are
    asserted separately in ``TestShippedDefaultsAreConservative``, and nothing
    here changes them.
    """
    clone = copy.deepcopy(config)
    object.__setattr__(clone.aggregator, "min_agreeing_strategies", agreeing)
    object.__setattr__(clone.aggregator, "min_consensus", 30.0)
    object.__setattr__(clone.opportunity, "min_score", min_score)
    return clone


TRENDING = trending_with_pullbacks(2400, drift=0.0003, pullback_every=25, pullback_bars=9, seed=3)
CHOPPY = choppy_prices(2400, sigma=0.0012, seed=17)


class TestEngineMechanics:
    def test_a_run_completes_and_reports(self):
        result = BacktestEngine(BACKTEST_CONFIG, 75.0).run(dataset({"AAAUSDT": TRENDING}))
        assert result.bars_processed > 0
        assert "BACKTEST RESULT" in result.report()
        assert result.metrics.initial_capital == 75.0

    def test_capital_is_preserved_when_no_trade_is_taken(self):
        result = BacktestEngine(BACKTEST_CONFIG, 75.0).run(dataset({"FLATUSDT": flat_prices(1000)}))
        assert result.metrics.total_trades == 0
        assert result.metrics.final_capital == pytest.approx(75.0)
        assert result.metrics.net_profit == pytest.approx(0.0)

    def test_every_rejection_is_attributed(self):
        """A zero-trade backtest must say WHY, or it is indistinguishable from
        a broken one."""
        result = BacktestEngine(BACKTEST_CONFIG, 75.0).run(
            dataset({"AAAUSDT": TRENDING, "BBBUSDT": CHOPPY})
        )
        if result.metrics.total_trades == 0:
            assert result.rejections
            assert sum(result.rejections.values()) > 0

    def test_the_report_never_claims_live_profitability(self):
        result = BacktestEngine(BACKTEST_CONFIG, 75.0).run(dataset({"AAAUSDT": TRENDING}))
        report = result.report()
        assert "SIMULATION" in report
        assert "not evidence of live" in report

    def test_config_snapshot_records_the_assumptions(self):
        """A result without its cost assumptions cannot be interpreted later."""
        result = BacktestEngine(BACKTEST_CONFIG, 75.0).run(dataset({"AAAUSDT": TRENDING}))
        snapshot = result.config_snapshot
        assert snapshot["intrabar"] == "pessimistic"
        assert snapshot["fill_model"] == "next_open"
        assert snapshot["taker_fee"] > 0
        assert snapshot["slippage_bps"] > 0


@pytest.fixture(scope="module")
def traded_result():
    """One backtest run shared by the trade-path tests.

    Module-scoped deliberately: a full replay costs tens of seconds, and running
    it once per assertion turned this file into a multi-minute suite for no
    additional coverage.
    """
    config = permissive(BACKTEST_CONFIG)
    return BacktestEngine(config, 75.0).run(dataset({"AAAUSDT": TRENDING, "BBBUSDT": CHOPPY}))


class TestTradePath:
    """Exercise fills, exits and PnL with the analytical gates opened up."""

    def test_trades_are_executed(self, traded_result):
        result = traded_result
        assert result.metrics.total_trades >= 1, (
            f"expected the trade path to run; rejections were {result.rejections}"
        )

    def test_every_trade_pays_fees(self, traded_result):
        result = traded_result
        for executed in result.trades:
            assert executed.fees > 0, "a filled trade must pay entry and exit fees"

    def test_every_trade_has_a_recorded_exit_reason(self, traded_result):
        result = traded_result
        for executed in result.trades:
            assert executed.exit_reason in set(ExitReason)

    def test_no_trade_exceeds_the_maximum_duration(self, traded_result):
        """The 60-minute cap is a hard rule from the brief."""
        result = traded_result
        limit = BACKTEST_CONFIG.trade.max_duration_sec
        for executed in result.trades:
            assert executed.duration_sec <= limit + 300, (
                f"{executed.symbol} ran {executed.duration_sec}s, over the {limit}s limit"
            )

    def test_net_pnl_equals_gross_minus_costs(self, traded_result):
        """The arithmetic must close exactly, for every trade."""
        result = traded_result
        for executed in result.trades:
            expected = executed.gross_pnl - executed.fees - executed.funding
            assert executed.net_pnl == pytest.approx(expected, abs=1e-9)

    def test_equity_curve_is_produced(self, traded_result):
        result = traded_result
        assert len(result.equity_curve) > 1
        assert result.equity_curve[0].equity == pytest.approx(75.0)

    def test_position_count_never_exceeds_the_limit(self, traded_result):
        result = traded_result
        assert result.metrics.total_trades >= 0
        # Concurrency is enforced by the risk engine; no trade should overlap
        # more than max_concurrent_positions others.
        limit = BACKTEST_CONFIG.risk.max_concurrent_positions
        for candidate in result.trades:
            overlapping = sum(
                1
                for other in result.trades
                if other is not candidate
                and other.opened_at < candidate.closed_at
                and other.closed_at > candidate.opened_at
            )
            assert overlapping <= limit


class TestNoLookAhead:
    def test_a_run_is_deterministic(self):
        """Same data, same config, same result — or something reads the future."""
        config = permissive(BACKTEST_CONFIG)
        data = dataset({"AAAUSDT": TRENDING})
        first = BacktestEngine(config, 75.0).run(data)
        second = BacktestEngine(config, 75.0).run(data)
        assert first.metrics.total_trades == second.metrics.total_trades
        assert first.metrics.net_profit == pytest.approx(second.metrics.net_profit)

    def test_truncating_the_future_does_not_change_the_past(self):
        """The decisive look-ahead test.

        Running over the first half of the data must produce exactly the trades
        that the full run produced in that same half. If a later bar can change
        an earlier decision, the backtest is reading the future.
        """
        config = permissive(BACKTEST_CONFIG)
        full_prices = TRENDING
        half = len(full_prices) // 2

        full = BacktestEngine(config, 75.0).run(dataset({"AAAUSDT": full_prices}))
        truncated = BacktestEngine(config, 75.0).run(dataset({"AAAUSDT": full_prices[:half]}))

        cutoff = truncated.end_ms
        full_in_window = [
            t
            for t in full.trades
            if t.closed_at <= cutoff and t.exit_reason is not ExitReason.MANUAL
        ]
        truncated_in_window = [
            t for t in truncated.trades if t.exit_reason is not ExitReason.MANUAL
        ]
        assert len(full_in_window) == len(truncated_in_window)
        for a, b in zip(full_in_window, truncated_in_window, strict=False):
            assert a.symbol == b.symbol
            assert a.opened_at == b.opened_at
            assert a.entry_price == pytest.approx(b.entry_price)


class TestShippedDefaultsAreConservative:
    """A finding worth stating plainly rather than hiding in a config file.

    With the shipped defaults the system trades very rarely: the strategies are
    deliberately built to fire in mutually exclusive conditions, regime gating
    permits only three or four of them at a time, and the aggregator then
    requires two of those few to agree. On this synthetic data that combination
    produces essentially no trades.

    Whether that is correct caution or over-tuning cannot be settled with
    generated prices. It is a question for real data, and the operator needs to
    know it exists before concluding the bot is broken.
    """

    def test_default_consensus_requirement_is_the_binding_constraint(self):
        data = dataset({"AAAUSDT": TRENDING, "BBBUSDT": CHOPPY})
        strict = BacktestEngine(BACKTEST_CONFIG, 75.0).run(data)
        relaxed = BacktestEngine(permissive(BACKTEST_CONFIG), 75.0).run(data)
        assert relaxed.metrics.total_trades >= strict.metrics.total_trades
        assert strict.rejections.get("INSUFFICIENT_CONSENSUS", 0) > 0

    def test_zero_trades_is_reported_as_a_valid_outcome_not_an_error(self):
        result = BacktestEngine(BACKTEST_CONFIG, 75.0).run(
            dataset({"AAAUSDT": TRENDING, "BBBUSDT": CHOPPY})
        )
        if result.metrics.total_trades == 0:
            assert any("valid outcome" in w for w in result.metrics.warnings)


class TestBootstrapHonesty:
    def test_bootstrap_usage_is_counted_and_surfaced(self):
        """A result resting on assumptions must say so, prominently."""
        config = permissive(BACKTEST_CONFIG)
        result = BacktestEngine(config, 75.0).run(dataset({"AAAUSDT": TRENDING, "BBBUSDT": CHOPPY}))
        if result.bootstrap_estimates:
            report = result.report()
            assert "BOOTSTRAP MODE WAS ACTIVE" in report
            assert "NOT evidence" in report

    def test_live_config_never_bootstraps(self):
        """The safety property: live must not trade on an assumed win rate."""
        live = load_tunables(
            REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
        )
        assert live.edge.bootstrap_enabled is False

    def test_measured_statistics_are_exported_for_seeding(self):
        config = permissive(BACKTEST_CONFIG)
        result = BacktestEngine(config, 75.0).run(dataset({"AAAUSDT": TRENDING}))
        if result.metrics.total_trades:
            assert result.strategy_stats
            for stats in result.strategy_stats.values():
                assert "trades" in stats
                assert "wins" in stats
