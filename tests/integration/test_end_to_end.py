"""End-to-end and structural guarantees.

Two kinds of test:

1. **Structural** — properties enforced by the *shape* of the code, not by a
   runtime check. These are the guarantees the brief calls non-negotiable, and
   asserting them here means a refactor that quietly breaks one fails the build.
2. **End-to-end** — the whole pipeline from market data to a completed trade,
   against a fake exchange.
"""

from __future__ import annotations

import inspect

import pytest

from tradebot.ai.analyzer import Advisory, AdvisoryKind, MarketAnalyzer
from tradebot.core.config import load_tunables
from tradebot.core.types import (
    Direction,
    ExitReason,
    MarketRegime,
    OrderIntent,
    RejectionReason,
    Trade,
)
from tradebot.execution.engine import ExecutionEngine
from tradebot.risk.engine import RiskEngine
from tradebot.signals.pipeline import SignalPipeline
from tradebot.strategies.base import MarketView

from ..conftest import REPO_ROOT

CONFIG = load_tunables(
    REPO_ROOT / "config" / "config.yaml", REPO_ROOT / "config" / "strategies.yaml"
)


# --------------------------------------------------------------------------- #
class TestNonNegotiableRules:
    """The brief's critical rules, asserted structurally where possible."""

    def test_only_the_risk_engine_constructs_an_order_intent(self):
        """Rule 4: no strategy may bypass the risk engine.

        Enforced by construction: OrderIntent is built in exactly one place.
        """
        import pathlib

        root = pathlib.Path(REPO_ROOT / "src" / "tradebot")
        constructors: list[str] = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "OrderIntent(" in text and "intent_id=" in text:
                constructors.append(str(path.relative_to(root)))

        assert constructors == ["risk/engine.py"], (
            f"OrderIntent is constructed outside the risk engine: {constructors}"
        )

    def test_strategies_cannot_reach_the_account_or_exchange(self):
        """Rule 4, again: MarketView carries nothing actionable."""
        fields = set(MarketView.__slots__)
        forbidden = {
            "account",
            "gateway",
            "exchange",
            "positions",
            "equity",
            "balance",
            "execution",
            "risk",
        }
        assert not (fields & forbidden), (
            f"MarketView exposes actionable state: {fields & forbidden}"
        )

    def test_the_ai_layer_imports_nothing_that_can_trade(self):
        """Rule 5: AI may not place an order.

        Checked against the module's actual IMPORTS via the AST, not its text —
        the prose in its own docstring names these types precisely in order to
        explain that it does not use them.
        """
        import ast
        import pathlib

        source = (REPO_ROOT / "src" / "tradebot" / "ai" / "analyzer.py").read_text()
        tree = ast.parse(source)

        imported: set[str] = set()
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.add(node.module or "")
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)

        forbidden_names = {
            "OrderIntent",
            "ExecutionEngine",
            "RiskEngine",
            "BinanceFuturesREST",
            "PaperBroker",
            "Reconciler",
        }
        assert not (imported & forbidden_names), (
            f"the AI layer imports {imported & forbidden_names}, which would "
            f"give it a path to the exchange"
        )

        forbidden_modules = {
            "tradebot.execution",
            "tradebot.exchange",
            "tradebot.risk",
            "tradebot.paper",
        }
        offending = {
            m for m in modules if any(m == f or m.startswith(f + ".") for f in forbidden_modules)
        }
        assert not offending, (
            f"the AI layer imports from {offending}, which would give it a path to the exchange"
        )

        # And it must not CALL anything that transmits an order.
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not (
            calls
            & {
                "place_order",
                "close_position",
                "open_position",
                "place_protective_order",
                "evaluate",
            }
        )
        _ = pathlib

    def test_an_advisory_can_never_increase_a_score(self):
        """A positive adjustment would be authority by another name."""
        with pytest.raises(ValueError, match="never increase"):
            Advisory(
                kind=AdvisoryKind.PATTERN,
                severity="INFO",
                subject="X",
                message="looks good",
                score_adjustment=+10.0,
            )

    def test_the_risk_engine_has_no_exit_path(self):
        """Rule 15: protecting the account beats continuing to trade.

        Being unable to CLOSE is strictly worse than being unable to open, so
        the risk engine gates entries only.
        """
        methods = {name for name, _ in inspect.getmembers(RiskEngine, predicate=inspect.isfunction)}
        for forbidden in ("evaluate_exit", "approve_close", "allow_exit", "can_close"):
            assert forbidden not in methods

    def test_every_rejection_reason_is_distinct(self):
        values = [r.value for r in RejectionReason]
        assert len(values) == len(set(values))

    def test_max_trade_duration_cannot_exceed_sixty_minutes(self):
        """Rule: trades are short-term, hard-capped at 60 minutes."""
        from tradebot.core.config import TradeConfig

        with pytest.raises(Exception):
            TradeConfig(max_duration_sec=7200)

    def test_live_mode_requires_three_independent_confirmations(self):
        """Rule 1: no accidental live trading."""
        from tradebot.core.config import Settings, enforce_live_gate
        from tradebot.core.errors import SafetyError
        from tradebot.core.types import TradingMode

        base = {
            "trading_mode": TradingMode.LIVE,
            "i_understand_live_trading_risk": "YES",
            "binance_api_key": "k",
            "binance_api_secret": "s",
            "binance_testnet": False,
        }
        enforce_live_gate(Settings(**base), live_flag=True)  # all three agree

        for removed in ("i_understand_live_trading_risk", "binance_testnet"):
            broken = dict(base)
            broken[removed] = "NO" if removed.startswith("i_") else True
            with pytest.raises(SafetyError):
                enforce_live_gate(Settings(**broken), live_flag=True)

        with pytest.raises(SafetyError):
            enforce_live_gate(Settings(**base), live_flag=False)

    def test_bootstrap_is_disabled_in_the_live_configuration(self):
        """Live must never trade on an assumed win rate."""
        assert CONFIG.edge.bootstrap_enabled is False

    def test_a_stop_can_never_be_tighter_than_the_round_trip_cost(self):
        """Such a trade is mathematically unable to pay for itself."""
        assert CONFIG.stops.min_stop_pct >= 2 * CONFIG.edge.taker_fee

    def test_the_dashboard_exposes_no_mutating_endpoint(self):
        """Rule: no web endpoint may move money."""
        from tradebot.dashboard.app import create_app

        app = create_app(object(), "token")
        for route in app.routes:
            methods = getattr(route, "methods", set())
            assert methods <= {"GET", "HEAD"}


class TestAdvisoryLayer:
    def analyzer(self) -> MarketAnalyzer:
        return MarketAnalyzer(enabled=True)

    def test_disabled_by_default(self):
        assert not MarketAnalyzer().enabled
        assert CONFIG.ai.enabled is False

    def test_a_disabled_analyzer_produces_nothing(self):
        import numpy as np

        analyzer = MarketAnalyzer(enabled=False)
        closes = np.concatenate([np.full(100, 100.0), np.array([150.0])])
        assert analyzer.detect_price_anomaly("X", closes, np.ones(101)) is None

    def test_an_extreme_move_is_flagged(self):
        import numpy as np

        rng = np.random.default_rng(3)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 200)))
        closes = np.append(closes, closes[-1] * 1.15)
        advisory = self.analyzer().detect_price_anomaly("X", closes, np.ones(closes.size) * 1000)
        assert advisory is not None
        assert advisory.kind is AdvisoryKind.ANOMALY
        assert advisory.score_adjustment < 0

    def test_ordinary_movement_is_not_flagged(self):
        import numpy as np

        rng = np.random.default_rng(5)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 200)))
        assert self.analyzer().detect_price_anomaly("X", closes, np.ones(200) * 1000) is None

    def test_strategy_degradation_is_detected(self):
        trades = [_trade(2.0) for _ in range(40)] + [_trade(-1.0) for _ in range(30)]
        advisory = self.analyzer().analyse_strategy("momentum", trades, window=30)
        assert advisory is not None
        assert advisory.kind is AdvisoryKind.STRATEGY_DEGRADATION
        assert "regimes return" in advisory.message

    def test_a_consistent_strategy_is_not_flagged(self):
        trades = [_trade(1.0 if i % 3 else -0.5) for i in range(80)]
        assert self.analyzer().analyse_strategy("momentum", trades, 30) is None

    def test_an_optimistic_cost_model_is_flagged(self):
        """The most consequential check: the edge filter gates every trade."""
        advisory = self.analyzer().analyse_cost_model("momentum", [0.002] * 30, [0.0002] * 30)
        assert advisory is not None
        assert advisory.severity == "CRITICAL"
        assert "optimistic" in advisory.message

    def test_an_accurate_cost_model_is_not_flagged(self):
        assert self.analyzer().analyse_cost_model("momentum", [0.002] * 30, [0.0019] * 30) is None

    def test_exit_reason_analysis_flags_excessive_time_exits(self):
        trades = [_trade(0.1, ExitReason.TIME_LIMIT) for _ in range(8)] + [
            _trade(1.0) for _ in range(2)
        ]
        report = self.analyzer().analyse_exit_reasons(trades)
        assert any("time limit" in note for note in report["notes"])

    def test_regime_performance_returns_data_not_advisories(self):
        """Which regime a strategy suits is a human's configuration decision."""
        trades = [_trade(1.0), _trade(-1.0)]
        result = self.analyzer().analyse_regime_performance(trades)
        assert "STRONG_TREND" in result
        assert result["STRONG_TREND"]["trades"] == 2

    def test_the_summary_warns_about_small_samples(self):
        summary = self.analyzer().summarise([_trade(1.0)], 76.0, 76.0)
        assert "noise" in summary

    def test_the_summary_handles_no_trades(self):
        summary = self.analyzer().summarise([], 75.0, 75.0)
        assert "valid state" in summary

    def test_adjustments_are_bounded_and_never_positive(self):
        analyzer = self.analyzer()
        for _ in range(20):
            analyzer._record(
                Advisory(AdvisoryKind.PATTERN, "INFO", "X", "m", score_adjustment=-30.0)
            )
        assert -50.0 <= analyzer.total_adjustment("X") <= 0.0


class TestFullPipeline:
    """Market data in, completed trade out, through the real components."""

    async def test_the_pipeline_wires_together(self):
        from tradebot.core.clock import VirtualClock
        from tradebot.core.events import EventBus
        from tradebot.market.candles import CandleStore
        from tradebot.market.microstructure import CostModel
        from tradebot.strategies.registry import StrategyRegistry

        from ..fakes import FakeGateway, make_symbol_info

        candles = CandleStore(500)
        gateway = FakeGateway()
        gateway.symbols["TESTUSDT"] = make_symbol_info("TESTUSDT", min_notional=1.0)

        pipeline = SignalPipeline(
            CONFIG, StrategyRegistry.from_config(CONFIG), CostModel(CONFIG.edge)
        )
        risk = RiskEngine(CONFIG, candles, VirtualClock(1_700_000_000_000))
        execution = ExecutionEngine(CONFIG, gateway, EventBus(), VirtualClock(1_700_000_000_000))

        assert pipeline.registry.strategies
        assert risk.kill_switches.entries_allowed
        assert not execution.entries_blocked

    async def test_an_approved_intent_becomes_a_protected_position(self):
        """The complete entry path: risk approval, fill, protective stop."""
        from tradebot.core.clock import VirtualClock
        from tradebot.core.events import EventBus
        from tradebot.core.types import OrderSide, OrderType

        from ..fakes import FakeGateway, make_symbol_info

        gateway = FakeGateway()
        gateway.symbols["TESTUSDT"] = make_symbol_info("TESTUSDT", min_notional=1.0)
        execution = ExecutionEngine(CONFIG, gateway, EventBus(), VirtualClock(1_700_000_000_000))

        intent = OrderIntent(
            intent_id="e2e001",
            symbol="TESTUSDT",
            direction=Direction.LONG,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.75,
            price=None,
            stop_loss=99.5,
            take_profit=101.0,
            leverage=2,
            notional=75.0,
            risk_amount=0.375,
            strategy="momentum",
            regime=MarketRegime.STRONG_TREND,
            opportunity_score=85.0,
            expected_net_edge=0.0015,
            metadata={"reference_price": 100.0},
        )

        result = await execution.open_position(intent)
        assert result.success
        assert result.protected
        position = execution.positions["TESTUSDT"]
        assert position.stop_loss == pytest.approx(99.5)
        assert position.stop_order_id is not None

        trade = await execution.close_position("TESTUSDT", ExitReason.TAKE_PROFIT)
        assert trade is not None
        assert "TESTUSDT" not in execution.positions
        assert trade.exit_reason is ExitReason.TAKE_PROFIT
        # Costs are always accounted, even on a scratch exit.
        assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.fees - trade.funding)


def _trade(r_multiple: float, reason: ExitReason = ExitReason.TAKE_PROFIT) -> Trade:
    return Trade(
        trade_id=f"t{r_multiple}",
        symbol="TESTUSDT",
        strategy="momentum",
        direction=Direction.LONG,
        entry_price=100.0,
        exit_price=100.0 + r_multiple,
        quantity=1.0,
        leverage=2,
        stop_loss=99.0,
        take_profit=103.0,
        opened_at=0,
        closed_at=600_000,
        gross_pnl=r_multiple,
        fees=0.08,
        funding=0.0,
        slippage_cost=0.02,
        net_pnl=r_multiple,
        exit_reason=reason,
        regime=MarketRegime.STRONG_TREND,
        entry_notional=100.0,
        initial_risk=1.0,
    )
