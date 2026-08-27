"""Database, notifications, health monitoring and the dashboard.

The theme: **support systems must never be able to break trading.** A database
outage, a Telegram failure or a dashboard error should degrade observability,
never the ability to exit a position.
"""

from __future__ import annotations

import pytest

from tradebot.app.health import ComponentState, HealthMonitor
from tradebot.core.config import HealthConfig
from tradebot.core.types import (
    Direction,
    ExitReason,
    MarketRegime,
    RiskEvent,
    RiskEventType,
    Trade,
    TradingMode,
)
from tradebot.database.repository import Repository, _redact_url
from tradebot.notifications.telegram import Priority, TelegramNotifier

from ..fakes import position_for


def sample_trade(gross: float = 1.5) -> Trade:
    """A trade whose arithmetic actually closes: net = gross - fees - funding."""
    fees, funding = 0.08, 0.01
    return Trade(
        trade_id="t1",
        symbol="TESTUSDT",
        strategy="momentum",
        direction=Direction.LONG,
        entry_price=100.0,
        exit_price=101.5,
        quantity=1.0,
        leverage=2,
        stop_loss=99.0,
        take_profit=103.0,
        opened_at=1_700_000_000_000,
        closed_at=1_700_000_600_000,
        gross_pnl=gross,
        fees=fees,
        funding=funding,
        slippage_cost=0.02,
        net_pnl=gross - fees - funding,
        exit_reason=ExitReason.TAKE_PROFIT,
        regime=MarketRegime.STRONG_TREND,
        entry_notional=100.0,
        initial_risk=1.0,
    )


# --------------------------------------------------------------------------- #
class TestRepository:
    async def repo(self) -> Repository:
        repository = Repository(
            "sqlite+aiosqlite:///:memory:", TradingMode.PAPER, flush_interval_sec=0.05
        )
        await repository.connect()
        return repository

    async def test_a_trade_round_trips(self):
        repository = await self.repo()
        repository.record_trade(sample_trade())
        await repository.flush()
        rows = await repository.recent_trades()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "TESTUSDT"
        assert rows[0]["net_pnl"] == pytest.approx(1.41)
        await repository.close()

    async def test_costs_are_stored_separately_from_pnl(self):
        """'Net was positive' and 'the strategy had an edge' differ."""
        repository = await self.repo()
        repository.record_trade(sample_trade())
        await repository.flush()
        row = (await repository.recent_trades())[0]
        assert row["fees"] == pytest.approx(0.08)
        assert row["funding"] == pytest.approx(0.01)
        assert row["slippage_cost"] == pytest.approx(0.02)
        assert row["gross_pnl"] == pytest.approx(1.5)
        assert row["net_pnl"] == pytest.approx(1.41)
        assert row["gross_pnl"] - row["fees"] - row["funding"] == pytest.approx(row["net_pnl"])
        await repository.close()

    async def test_rejections_are_audited_as_well_as_acceptances(self):
        """'Why didn't it trade?' must be answerable months later."""
        repository = await self.repo()
        repository.record_decision(
            "BTCUSDT",
            False,
            "edge",
            "costs exceed the expected move",
            1_700_000_000_000,
            rejection_reason="NEGATIVE_EXPECTED_EDGE",
            context={
                "regime": "STRONG_TREND",
                "expected_net_edge": -0.0004,
                "agreeing_strategies": ["momentum", "breakout"],
            },
        )
        await repository.flush()
        rows = await repository.recent_decisions()
        assert rows[0]["rejection_reason"] == "NEGATIVE_EXPECTED_EDGE"
        assert rows[0]["accepted"] is False
        assert rows[0]["strategies"] == ["momentum", "breakout"]
        await repository.close()

    async def test_decisions_can_be_filtered_by_outcome(self):
        repository = await self.repo()
        repository.record_decision("A", True, "complete", "opened", 1)
        repository.record_decision("B", False, "risk", "rejected", 2)
        await repository.flush()
        assert len(await repository.recent_decisions(accepted=True)) == 1
        assert len(await repository.recent_decisions(accepted=False)) == 1
        await repository.close()

    async def test_enum_values_survive_json_serialisation(self):
        """An enum reaching the driver raises mid-flush and loses the batch."""
        repository = await self.repo()
        trade = sample_trade()
        trade.metadata = {
            "regime": MarketRegime.BREAKOUT,
            "direction": Direction.LONG,
            "codes": ("A", "B"),
        }
        repository.record_trade(trade)
        assert await repository.flush() == 1
        row = (await repository.recent_trades())[0]
        assert row["extra"]["regime"] == "BREAKOUT"
        assert row["extra"]["codes"] == ["A", "B"]
        await repository.close()

    async def test_a_write_failure_does_not_raise_into_the_caller(self):
        """Trading must continue when the database does not."""
        repository = await self.repo()
        await repository._engine.dispose()
        repository._engine = None

        class Broken:
            def __call__(self):
                raise RuntimeError("database is gone")

        repository._session_factory = Broken()
        repository.record_trade(sample_trade())
        assert await repository.flush() == 0  # no exception
        assert repository.health.failures == 1
        assert not repository.health.available

    async def test_the_buffer_is_bounded(self):
        """A long outage must not grow memory until the process dies."""
        repository = Repository("sqlite+aiosqlite:///:memory:", TradingMode.PAPER, buffer_size=10)
        for i in range(50):
            repository.record_system_event("TEST", f"event {i}", i)
        assert len(repository._buffer) == 10
        assert repository.health.dropped == 40

    async def test_pruning_respects_retention(self):
        repository = await self.repo()
        now = 1_700_000_000_000
        repository.record_signal(
            __import__("tradebot.core.types", fromlist=["Signal"]).Signal(
                "OLD",
                "momentum",
                Direction.WAIT,
                0.0,
                0,
                0,
                0,
                "3m",
                now - 40 * 86_400_000,
            )
        )
        repository.record_signal(
            __import__("tradebot.core.types", fromlist=["Signal"]).Signal(
                "NEW",
                "momentum",
                Direction.WAIT,
                0.0,
                0,
                0,
                0,
                "3m",
                now,
            )
        )
        await repository.flush()
        removed = await repository.prune({"signal_retention_days": 30}, now)
        assert removed.get("signals") == 1
        await repository.close()

    async def test_zero_retention_keeps_everything(self):
        repository = await self.repo()
        repository.record_decision("A", True, "s", "d", 0)
        await repository.flush()
        assert await repository.prune({"decision_retention_days": 0}, 1_700_000_000_000) == {}
        await repository.close()

    def test_a_database_password_is_never_logged(self):
        redacted = _redact_url("postgresql+asyncpg://user:hunter2@host/db")
        assert "hunter2" not in redacted
        assert "user" not in redacted
        assert "host/db" in redacted

    def test_a_sqlite_url_is_unchanged(self):
        url = "sqlite+aiosqlite:///data/tradebot.db"
        assert _redact_url(url) == url


class TestTelegram:
    def notifier(self, **kwargs) -> TelegramNotifier:
        params = {"bot_token": "123:abc", "chat_id": "42", "enabled": True}
        params.update(kwargs)
        return TelegramNotifier(**params)

    def test_disabled_without_credentials(self):
        assert not TelegramNotifier("", "", True).enabled
        assert not TelegramNotifier("t", "", True).enabled

    def test_a_disabled_notifier_silently_ignores_sends(self):
        notifier = TelegramNotifier("", "", False)
        notifier.send("hello")
        assert not notifier._queue

    def test_messages_are_queued(self):
        notifier = self.notifier()
        notifier.send("hello")
        assert len(notifier._queue) == 1

    def test_a_full_queue_sheds_the_lowest_priority(self):
        """A CRITICAL alert must never be lost behind scan summaries."""
        notifier = self.notifier(queue_size=3)
        for i in range(3):
            notifier.send(f"low {i}", Priority.LOW)
        notifier.send("KILL SWITCH", Priority.CRITICAL)

        texts = [m.text for m in notifier._queue]
        assert "KILL SWITCH" in texts
        assert len(notifier._queue) == 3

    def test_a_full_queue_of_critical_messages_drops_the_new_low_one(self):
        notifier = self.notifier(queue_size=2)
        notifier.send("crit 1", Priority.CRITICAL)
        notifier.send("crit 2", Priority.CRITICAL)
        notifier.send("low", Priority.LOW)
        assert notifier.dropped == 1
        assert all("crit" in m.text for m in notifier._queue)

    def test_rate_limiting_blocks_beyond_the_cap(self):
        import time as _time

        notifier = self.notifier(max_per_minute=2)
        notifier._sent_times.extend([_time.time(), _time.time()])
        assert not notifier._can_send()

    def test_a_trade_message_contains_the_cost_breakdown(self):
        """Net PnL alone hides whether the strategy actually had an edge."""
        notifier = self.notifier()
        notifier.notify_trade_closed(sample_trade())
        text = notifier._queue[0].text
        assert "Fees" in text
        assert "Funding" in text
        assert "Net PnL" in text
        assert "0.0800" in text

    def test_a_new_trade_message_contains_the_levels(self):
        notifier = self.notifier()
        notifier.notify_trade_opened(position_for("TESTUSDT"), 0.005, 0.0015, 85.0)
        text = notifier._queue[0].text
        assert "SL" in text and "TP" in text
        assert "Expected Net Edge" in text

    def test_a_kill_switch_message_says_exits_still_work(self):
        """An operator seeing 'suspended' must know exits are unaffected."""
        notifier = self.notifier()
        notifier.notify_kill_switch("DAILY_LOSS", "2% limit reached", 73.0)
        text = notifier._queue[0].text
        assert "can still be closed" in text
        assert notifier._queue[0].priority is Priority.CRITICAL

    def test_a_live_startup_message_is_marked(self):
        notifier = self.notifier()
        notifier.notify_startup("LIVE", 75.0, ["momentum"], testnet=False)
        assert "REAL MONEY" in notifier._queue[0].text

    def test_a_paper_startup_message_is_not_marked(self):
        notifier = self.notifier()
        notifier.notify_startup("PAPER", 75.0, ["momentum"], testnet=True)
        assert "REAL MONEY" not in notifier._queue[0].text

    async def test_a_send_failure_does_not_raise(self):
        class BrokenSession:
            async def post(self, *_args, **_kwargs):
                raise ConnectionError("telegram unreachable")

        notifier = self.notifier(session=BrokenSession())
        assert await notifier._deliver("hello") is False
        assert notifier.failed == 1

    async def test_a_non_200_response_is_counted_without_logging_the_token(self):
        class Response:
            status_code = 401

        class Session:
            async def post(self, *_args, **_kwargs):
                return Response()

        notifier = self.notifier(session=Session())
        assert await notifier._deliver("hello") is False
        assert notifier.last_error == "HTTP 401"
        assert "123:abc" not in str(notifier.last_error)

    def test_risk_events_map_to_priorities(self):
        notifier = self.notifier()
        notifier.notify_risk_event(
            RiskEvent(RiskEventType.KILL_SWITCH_TRIGGERED, "CRITICAL", "halted", 0)
        )
        assert notifier._queue[-1].priority is Priority.CRITICAL


class TestHealthMonitor:
    def monitor(self, **kwargs) -> HealthMonitor:
        return HealthMonitor(HealthConfig(**kwargs))

    def beat_all(self, monitor: HealthMonitor) -> None:
        for name in monitor.components:
            monitor.beat(name)

    def test_all_healthy_means_no_safe_mode(self):
        monitor = self.monitor()
        self.beat_all(monitor)
        report = monitor.check()
        assert report.healthy
        assert not report.safe_mode

    def test_a_critical_failure_forces_safe_mode(self):
        monitor = self.monitor()
        self.beat_all(monitor)
        monitor.fail("market_data", "WebSocket disconnected")
        report = monitor.check()
        assert report.safe_mode
        assert "market_data" in report.safe_mode_reason

    def test_a_non_critical_failure_only_warns(self):
        """Losing Telegram degrades observability, not safety."""
        monitor = self.monitor()
        self.beat_all(monitor)
        monitor.fail("telegram", "token rejected")
        report = monitor.check()
        assert not report.safe_mode
        assert any("telegram" in w for w in report.warnings)

    def test_recovery_leaves_safe_mode(self):
        monitor = self.monitor()
        self.beat_all(monitor)
        monitor.fail("exchange_rest", "unreachable")
        assert monitor.check().safe_mode
        monitor.beat("exchange_rest")
        assert not monitor.check().safe_mode

    def test_a_silent_component_is_not_healthy(self):
        """'It last said it was fine' is not the same as 'it is fine'."""
        monitor = self.monitor(component_timeout_sec=0.001)
        monitor.beat("market_data")
        import time as _time

        _time.sleep(0.01)
        assert monitor.components["market_data"].evaluate() is ComponentState.FAILED

    def test_a_component_that_never_reported_is_unknown(self):
        monitor = self.monitor()
        assert monitor.components["market_data"].evaluate() is ComponentState.UNKNOWN

    def test_safe_mode_callbacks_fire(self):
        entered: list[str] = []
        recovered: list[bool] = []
        monitor = HealthMonitor(
            HealthConfig(), on_safe_mode=entered.append, on_recovered=lambda: recovered.append(True)
        )
        self.beat_all(monitor)
        monitor.fail("risk_engine", "stalled")
        monitor.check()
        assert entered
        monitor.beat("risk_engine")
        monitor.check()
        assert recovered

    def test_the_report_is_serialisable(self):
        monitor = self.monitor()
        self.beat_all(monitor)
        payload = monitor.check().as_dict()
        assert "components" in payload
        assert "safe_mode" in payload
        assert isinstance(payload["components"], list)


class TestDashboard:
    def client(self, token: str = "secret"):
        from fastapi.testclient import TestClient

        from tradebot.dashboard.app import create_app

        class FakeEngine:
            def __init__(self):
                self.health = HealthMonitor(HealthConfig())
                for name in self.health.components:
                    self.health.beat(name)
                self.config = type("C", (), {"mode": TradingMode.PAPER})()

            async def status_snapshot(self):
                return {"mode": "PAPER", "equity": 75.0, "rejections": []}

            def open_positions_view(self):
                return []

            def opportunities_view(self):
                return []

            def strategy_view(self):
                return {"strategies": {}}

            def risk_view(self):
                return {}

            async def recent_trades(self, limit=50):
                return []

            async def recent_decisions(self, limit=100, accepted=None):
                return []

        return TestClient(create_app(FakeEngine(), token=token))

    def test_health_is_unauthenticated_for_docker(self):
        response = self.client().get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_reports_unhealthy_when_a_component_failed(self):
        """A bot that is up but disconnected must fail its health check."""
        from fastapi.testclient import TestClient

        from tradebot.dashboard.app import create_app

        class FailingEngine:
            def __init__(self):
                self.health = HealthMonitor(HealthConfig())
                for name in self.health.components:
                    self.health.beat(name)
                self.health.fail("market_data", "disconnected")
                self.config = type("C", (), {"mode": TradingMode.PAPER})()

        response = TestClient(create_app(FailingEngine(), "t")).get("/health")
        assert response.status_code == 503
        assert response.json()["safe_mode"] is True

    def test_api_requires_the_token(self):
        client = self.client()
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/status?token=wrong").status_code == 401
        assert client.get("/api/status?token=secret").status_code == 200

    def test_the_token_may_be_supplied_as_a_header(self):
        response = self.client().get("/api/status", headers={"x-dashboard-token": "secret"})
        assert response.status_code == 200

    def test_the_dashboard_is_read_only(self):
        """No endpoint may move money. A web surface is far larger than a file."""
        from tradebot.dashboard.app import create_app

        app = create_app(object(), "t")
        for route in app.routes:
            methods = getattr(route, "methods", set())
            assert methods <= {"GET", "HEAD"}, f"{getattr(route, 'path', route)} exposes {methods}"

    def test_the_page_is_mobile_friendly(self):
        response = self.client().get("/?token=secret")
        assert response.status_code == 200
        assert "viewport" in response.text
        assert "prefers-color-scheme" in response.text

    def test_the_opportunity_page_states_it_is_not_a_signal(self):
        response = self.client().get("/?token=secret")
        assert "does not mean a trade will be taken" in response.text
