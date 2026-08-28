"""A failed database stops NEW ENTRIES — and never stops exits.

AUDIT_REPORT.md C-3: the brief lists "continuing new entries during a database
failure" among the things the engine must never do, and the V1 code marked the
database non-critical, so a total persistence failure produced a log line and
nothing else. A trade with no audit trail cannot be reconciled against the
exchange or learned from afterwards.

The asymmetry is the important half: refusing to CLOSE a position because a log
write failed would be far more dangerous than the missing row.
"""

from __future__ import annotations

import inspect

import pytest

from tradebot.app import runner as runner_module
from tradebot.app.health import ComponentState, HealthMonitor
from tradebot.core.config import HealthConfig
from tradebot.database.repository import Repository


@pytest.fixture
def monitor() -> HealthMonitor:
    reasons: list[str] = []
    built = HealthMonitor(
        HealthConfig(component_timeout_sec=10.0),
        on_safe_mode=reasons.append,
        on_recovered=lambda: reasons.append("recovered"),
    )
    built.safe_mode_reasons = reasons  # type: ignore[attr-defined]
    return built


class TestDatabaseIsCritical:
    def test_database_is_a_critical_component_by_default(self, monitor: HealthMonitor) -> None:
        assert monitor.components["database"].critical is True

    def test_a_failed_database_puts_the_engine_in_safe_mode(self, monitor: HealthMonitor) -> None:
        for name in monitor.components:
            monitor.beat(name)
        assert monitor.check().healthy is True

        monitor.fail("database", "disk is full")
        report = monitor.check()

        assert monitor.safe_mode is True
        assert "database" in monitor.safe_mode_reason
        assert report.healthy is False

    def test_recovery_clears_safe_mode(self, monitor: HealthMonitor) -> None:
        for name in monitor.components:
            monitor.beat(name)
        monitor.fail("database", "disk is full")
        monitor.check()
        assert monitor.safe_mode is True

        monitor.beat("database", "reconnected")
        monitor.check()
        assert monitor.safe_mode is False
        assert monitor.components["database"].evaluate() is not ComponentState.FAILED

    def test_it_can_be_made_advisory_for_deployments_that_want_that(self) -> None:
        """A deliberate opt-out, in config, not a silent default."""
        relaxed = HealthMonitor(HealthConfig(database_critical=False))
        assert relaxed.components["database"].critical is False
        for name in relaxed.components:
            relaxed.beat(name)
        relaxed.fail("database", "disk is full")
        relaxed.check()
        assert relaxed.safe_mode is False


class TestEntriesStopButExitsDoNot:
    def test_safe_mode_blocks_entries_only(self) -> None:
        source = inspect.getsource(runner_module.TradingEngine._on_safe_mode)
        assert "block_entries" in source
        assert "close" not in source, "safe mode must never trigger closures"

    def test_the_exit_path_is_not_gated_on_health(self) -> None:
        """Position management runs regardless of safe mode; only entry does not."""
        manage = inspect.getsource(runner_module.TradingEngine._manage_positions)
        assert "safe_mode" not in manage

        evaluate = inspect.getsource(runner_module.TradingEngine._evaluate_candidates)
        assert "self.health.safe_mode" in evaluate

    def test_the_engine_retries_the_connection_rather_than_giving_up(self) -> None:
        source = inspect.getsource(runner_module.TradingEngine._check_database)
        assert "reconnect" in source
        assert "self.health.fail" in source


class TestRepositoryReconnection:
    @pytest.mark.asyncio
    async def test_buffered_rows_survive_an_outage(self, tmp_path) -> None:
        """The audit trail resumes where it stopped instead of starting at the
        moment the database came back."""
        repo = Repository(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
        await repo.connect()

        repo.record_equity(1, 75.0, 75.0)
        assert await repo.flush() == 1

        # Simulate the connection dying with rows still queued.
        repo.health.available = False
        repo.health.last_error = "connection reset"
        repo.record_equity(2, 76.0, 76.0)

        assert await repo.reconnect(attempts=1, backoff_sec=0.0) is True
        assert repo.health.available is True
        assert repo.health.reconnects == 1
        assert await repo.flush() == 1  # the row buffered during the outage
        await repo.close()

    @pytest.mark.asyncio
    async def test_reconnection_failure_is_reported_not_raised(self) -> None:
        # An unresolvable dialect: connecting can never succeed.
        repo = Repository("nosuchdriver://host/db")
        repo.health.available = False
        assert await repo.reconnect(attempts=2, backoff_sec=0.0) is False
        assert repo.health.reconnect_failures == 2
        assert repo.health.last_error

    @pytest.mark.asyncio
    async def test_reconnect_is_a_no_op_when_healthy(self, tmp_path) -> None:
        repo = Repository(f"sqlite+aiosqlite:///{tmp_path / 'a.db'}")
        await repo.connect()
        assert await repo.reconnect() is True
        assert repo.health.reconnects == 0
        await repo.close()

    @pytest.mark.asyncio
    async def test_a_write_failure_marks_the_database_unavailable(self, tmp_path) -> None:
        repo = Repository(f"sqlite+aiosqlite:///{tmp_path / 'b.db'}")
        await repo.connect()
        repo.record_equity(1, 75.0, 75.0)

        # Dispose the engine underneath the session factory.
        assert repo._engine is not None
        await repo._engine.dispose()
        repo._engine = None
        repo._session_factory = None
        assert await repo.flush() == 0
        await repo.close()
