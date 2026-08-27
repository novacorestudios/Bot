"""The trading engine orchestrator.

Assembles every component and runs the concurrent loops. This is the only place
that knows how the pieces fit together; each component below it is independently
testable and none of them import this module.

Loops:

| Task | Cadence | Responsibility |
|---|---|---|
| `scan` | `scan_interval_sec` | re-rank the whole universe |
| `signals` | `signal_interval_sec` | evaluate candidates, place trades |
| `monitor` | `monitor_interval_sec` | exits, trailing stops, the 60-minute cap |
| `reconcile` | `reconcile_interval_sec` | exchange truth vs local state |
| `health` | `heartbeat_interval_sec` | component heartbeats, safe mode |
| `persist` | continuous | buffered database writes |

Ordering inside a cycle matters: **positions are managed before new entries are
considered.** An exit frees risk budget that a new entry may need, and a stale
position left unexamined can block a better opportunity.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal as os_signal
import time
from typing import Any

from tradebot.core.clock import SystemClock, format_duration
from tradebot.core.config import AppConfig
from tradebot.core.events import Event, EventBus, EventType
from tradebot.core.logging import get_logger
from tradebot.core.types import (
    Direction,
    ExitReason,
    MarketRegime,
    RiskEvent,
    Trade,
    TradingMode,
)
from tradebot.market.candles import CandleStore
from tradebot.market.microstructure import CostModel, snapshot_from_book
from tradebot.market.regime import RegimeDetector
from tradebot.market.scanner import MarketScanner, enrich_candidates
from tradebot.market.scoring import MarketScorer
from tradebot.market.universe import UniverseBuilder
from tradebot.risk.engine import RiskContext, RiskEngine
from tradebot.signals.pipeline import SignalPipeline
from tradebot.strategies.base import MarketView
from tradebot.strategies.registry import StrategyRegistry

log = get_logger(__name__)


class TradingEngine:
    """Owns every component and drives the concurrent loops."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tunables = config.tunables
        self.clock = SystemClock()
        self.events = EventBus()

        self.candles = CandleStore(self.tunables.timeframes.history_bars)
        self.cost_model = CostModel(self.tunables.edge)
        self.registry = StrategyRegistry.from_config(self.tunables)
        self.pipeline = SignalPipeline(self.tunables, self.registry, self.cost_model)
        self.risk = RiskEngine(self.tunables, self.candles, self.clock)

        self.gateway: Any = None
        self.execution: Any = None
        self.reconciler: Any = None
        self.scanner: MarketScanner | None = None
        self.repository: Any = None
        self.telegram: Any = None
        self.market_stream: Any = None

        from tradebot.app.health import HealthMonitor

        self.health = HealthMonitor(
            self.tunables.health,
            on_safe_mode=self._on_safe_mode,
            on_recovered=self._on_recovered,
        )

        self.equity = self.tunables.account.initial_capital
        self.available_balance = self.equity
        self.peak_equity = self.equity
        self.day_start_equity = self.equity
        self.realized_pnl_today = 0.0
        self.total_pnl = 0.0
        self.trades: list[Trade] = []
        self.book_tickers: dict[str, Any] = {}
        self.mark_prices: dict[str, Any] = {}

        self.started_at = time.time()
        self.running = False
        self._tasks: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()
        self._last_market_data = 0.0
        self._connected = True

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def run(self, duration_sec: float = 0.0) -> int:
        """Start every component and run until stopped."""
        try:
            await self._build()
        except Exception as exc:  # noqa: BLE001
            log.critical("engine_startup_failed", error=str(exc), error_type=type(exc).__name__)
            return 1

        self.running = True
        self._install_signal_handlers()

        self._tasks = [
            asyncio.create_task(self._scan_loop(), name="scan"),
            asyncio.create_task(self._signal_loop(), name="signals"),
            asyncio.create_task(self._monitor_loop(), name="monitor"),
            asyncio.create_task(self._reconcile_loop(), name="reconcile"),
            asyncio.create_task(self._health_loop(), name="health"),
        ]

        if duration_sec > 0:
            self._tasks.append(asyncio.create_task(self._stop_after(duration_sec), name="timer"))

        log.info(
            "engine_running",
            mode=self.config.mode.value,
            equity=self.equity,
            strategies=sorted(self.registry.strategies),
        )

        try:
            await self._shutdown.wait()
        finally:
            await self._teardown()
        return 0

    async def _build(self) -> None:
        """Construct the gateway, persistence, notifications and dashboard."""
        settings = self.config.settings

        # -- persistence (non-fatal if it fails) ---------------------------- #
        from tradebot.database.repository import Repository

        self.repository = Repository(settings.database_url, self.config.mode)
        try:
            await self.repository.connect()
            self.health.beat("database")
        except Exception as exc:  # noqa: BLE001
            self.health.fail("database", str(exc)[:200])
            log.error(
                "database_unavailable",
                error=str(exc)[:200],
                message="trading continues without an audit trail",
            )

        # -- notifications (never fatal) ------------------------------------ #
        from tradebot.notifications.telegram import TelegramNotifier

        self.telegram = TelegramNotifier(
            settings.telegram_bot_token,
            settings.telegram_chat_id,
            settings.telegram_enabled,
        )
        await self.telegram.start()
        self.health.beat("telegram")

        # -- exchange gateway ------------------------------------------------ #
        from tradebot.exchange.binance.rest import BinanceFuturesREST

        rest = BinanceFuturesREST(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            base_url=settings.rest_url,
            recv_window=settings.binance_recv_window,
            max_retries=self.tunables.execution.max_retries,
            retry_backoff_sec=self.tunables.execution.retry_backoff_sec,
            clock=self.clock,
        )
        await rest.connect()
        self.health.beat("exchange_rest")

        if self.config.mode is TradingMode.PAPER:
            from tradebot.paper.broker import PaperBroker

            self.gateway = PaperBroker(
                self.tunables.paper,
                _MarketAdapter(self, rest),
                initial_balance=self.tunables.account.initial_capital,
                taker_fee=self.tunables.edge.taker_fee,
                maker_fee=self.tunables.edge.maker_fee,
                clock=self.clock,
            )
        else:
            self.gateway = rest

        self._rest = rest

        # -- scanner ---------------------------------------------------------- #
        scanner_config = self.tunables.scanner
        self.scanner = MarketScanner(
            config=scanner_config,
            gateway=rest,
            candles=self.candles,
            scorer=MarketScorer(scanner_config, self.cost_model),
            regime_detector=RegimeDetector(self.tunables.regime),
            universe_builder=UniverseBuilder(
                scanner_config,
                self.equity,
                self.tunables.execution.max_min_notional_ratio,
            ),
            cost_model=self.cost_model,
            primary_timeframe=self.tunables.timeframes.primary,
        )

        # -- execution and reconciliation -------------------------------------- #
        from tradebot.execution.engine import ExecutionEngine
        from tradebot.execution.reconciliation import Reconciler

        self.execution = ExecutionEngine(self.tunables, self.gateway, self.events, self.clock)
        self.reconciler = Reconciler(
            self.gateway, self.execution, self.tunables, self.clock, self.risk
        )

        self.events.subscribe(EventType.TRADE_COMPLETED, self._on_trade_completed)
        self.events.subscribe(EventType.RISK_EVENT, self._on_risk_event)

        # -- account state ------------------------------------------------------ #
        await self._refresh_account()

        # -- reconcile BEFORE any entry is permitted ---------------------------- #
        await self.reconciler.startup()

        # -- dashboard ---------------------------------------------------------- #
        if settings.dashboard_enabled:
            await self._start_dashboard()

        self.telegram.notify_startup(
            self.config.mode.value,
            self.equity,
            sorted(self.registry.strategies),
            settings.binance_testnet,
        )
        for name in ("market_data", "risk_engine", "execution"):
            self.health.beat(name)

    async def _start_dashboard(self) -> None:
        import uvicorn

        from tradebot.dashboard.app import create_app

        settings = self.config.settings
        host = settings.dashboard_host if settings.dashboard_token else "127.0.0.1"
        if not settings.dashboard_token:
            log.warning(
                "dashboard_localhost_only",
                message="no DASHBOARD_TOKEN set; binding to loopback only",
            )

        server = uvicorn.Server(
            uvicorn.Config(
                create_app(self, settings.dashboard_token),
                host=host,
                port=settings.dashboard_port,
                log_level="warning",
                access_log=False,
            )
        )
        self._tasks.append(asyncio.create_task(server.serve(), name="dashboard"))
        self.health.beat("dashboard")
        log.info("dashboard_started", host=host, port=settings.dashboard_port)

    async def _teardown(self) -> None:
        log.info("engine_stopping")
        self.running = False

        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Positions are NOT auto-closed on shutdown: an operator restarting the
        # bot for a deploy does not want their book flattened. Reconciliation on
        # the next start adopts and protects whatever is still open.
        if self.execution is not None and self.execution.positions:
            log.warning(
                "shutdown_with_open_positions",
                symbols=sorted(self.execution.positions),
                message="positions remain open and protected by their "
                "resting stops; they will be adopted on restart",
            )

        if self.telegram is not None:
            self.telegram.notify_shutdown(
                "operator stop"
                if not self.health.safe_mode
                else f"safe mode: {self.health.safe_mode_reason}",
                self.equity,
                len(self.trades),
            )
            await asyncio.sleep(1.0)  # let the queue drain
            await self.telegram.stop()

        if self.market_stream is not None:
            await self.market_stream.stop()
        if self.repository is not None:
            await self.repository.close()
        if getattr(self, "_rest", None) is not None:
            await self._rest.close()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.stop)

    def stop(self) -> None:
        self._shutdown.set()

    async def _stop_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        log.info("duration_elapsed", seconds=seconds)
        self.stop()

    # ------------------------------------------------------------------ #
    # Loops
    # ------------------------------------------------------------------ #
    async def _scan_loop(self) -> None:
        while self.running:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a loop must never die
                log.error("scan_loop_error", error=str(exc), error_type=type(exc).__name__)
                self.risk.record_api_error()
            await asyncio.sleep(self.tunables.scanner.scan_interval_sec)

    async def _scan_once(self) -> None:
        if self.scanner is None:
            return
        protected = set(self.execution.positions) if self.execution else set()
        self.scanner.set_correlation_penalties(
            self.risk.correlation_penalties(
                list(self.candles.symbols()),
                self.execution.positions if self.execution else {},
                self._prices(),
            )
        )
        result = await self.scanner.scan(protected)
        self._last_market_data = time.time()
        self.health.beat("market_data", f"{len(result.candidates)} candidates")

        if self.repository is not None:
            self.repository.record_scan(result.candidates, result.timestamp)

        self.book_tickers = await self.gateway.get_book_ticker()
        with contextlib.suppress(Exception):
            self.mark_prices = await self.gateway.get_mark_price()

        # Retain candles for candidates plus anything we hold.
        self.candles.retain({c.symbol for c in result.candidates}, protected)

    async def _signal_loop(self) -> None:
        while self.running:
            try:
                await self._evaluate_candidates()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("signal_loop_error", error=str(exc), error_type=type(exc).__name__)
            await asyncio.sleep(self.tunables.scanner.signal_interval_sec)

    async def _evaluate_candidates(self) -> None:
        if self.scanner is None or self.scanner.last_result is None:
            return
        if self.health.safe_mode:
            return

        result = self.scanner.last_result
        best: dict[str, tuple] = {}

        for candidate in result.candidates:
            symbol = candidate.symbol
            series = self.candles.get(symbol, self.tunables.timeframes.primary)
            if series is None or series.is_empty:
                continue

            regime_state = result.regimes.get(symbol)
            regime = regime_state.regime if regime_state else MarketRegime.SIDEWAYS
            book = self.book_tickers.get(symbol)
            mark = self.mark_prices.get(symbol)
            liquidity = snapshot_from_book(
                symbol, book, quote_volume_24h=candidate.market_score.liquidity_usd
            )

            view = MarketView(
                symbol=symbol,
                candles=self.candles,
                regime=regime,
                regime_confidence=regime_state.confidence if regime_state else 0.0,
                regime_direction=regime_state.direction if regime_state else Direction.WAIT,
                book_imbalance=book.imbalance if book else 0.0,
                spread_bps=book.spread_bps if book else 0.0,
                funding_rate=mark.funding_rate if mark else 0.0,
                now_ms=self.clock.now_ms(),
            )

            pipeline_result = self.pipeline.evaluate(
                view,
                candidate.market_score,
                liquidity,
                notional_estimate=self.equity * self.tunables.risk.max_symbol_exposure,
                correlation=self.scanner.correlation_penalties.get(symbol, 0.0),
                strategy_allocation=self.risk.strategy_weights(list(self.registry.strategies)),
                seconds_to_funding=self._seconds_to_funding(mark),
                now=time.time(),
            )

            if self.repository is not None:
                for produced in pipeline_result.signals:
                    self.repository.record_signal(produced, regime.value)

            if not pipeline_result.accepted or pipeline_result.opportunity is None:
                if self.repository is not None:
                    self.repository.record_decision(
                        symbol,
                        False,
                        pipeline_result.stage,
                        pipeline_result.detail,
                        self.clock.now_ms(),
                        pipeline_result.rejection.value if pipeline_result.rejection else None,
                        pipeline_result.audit,
                    )
                continue

            opportunity = pipeline_result.opportunity
            best[symbol] = (
                opportunity.strategy,
                opportunity.direction,
                opportunity.signal.confidence,
                opportunity.expected_net_edge,
                opportunity.opportunity_score.total,
            )
            await self._attempt_trade(opportunity, pipeline_result.audit)

        if best:
            self.scanner.last_result.candidates = enrich_candidates(result.candidates, best)

    async def _attempt_trade(self, opportunity: Any, audit: dict[str, Any]) -> None:
        info = self.gateway.symbol_info(opportunity.symbol)
        if info is None:
            return

        context = RiskContext(
            equity=self.equity,
            available_balance=self.available_balance,
            positions=dict(self.execution.positions),
            prices=self._prices(),
            symbol_info=info,
            realized_pnl_today=self.realized_pnl_today,
            data_age_sec=time.time() - self._last_market_data if self._last_market_data else 0.0,
            connected=self._connected,
            entries_blocked=self.execution.entries_blocked or self.health.safe_mode,
            entries_blocked_reason=self.execution.entries_blocked_reason
            or self.health.safe_mode_reason,
            in_flight=self.execution.in_flight,
            now=time.time(),
        )
        decision = self.risk.evaluate(opportunity, context)
        self.health.beat("risk_engine")

        merged = {**audit, **decision.checks}
        if not decision.approved or decision.intent is None:
            if self.repository is not None:
                self.repository.record_decision(
                    opportunity.symbol,
                    False,
                    "risk",
                    decision.detail,
                    self.clock.now_ms(),
                    decision.reason.value if decision.reason else None,
                    merged,
                )
            return

        result = await self.execution.open_position(decision.intent)
        self.health.beat("execution")

        if self.repository is not None:
            self.repository.record_decision(
                opportunity.symbol,
                result.success,
                "execution" if not result.success else "complete",
                result.reason or "position opened",
                self.clock.now_ms(),
                None if result.success else "EXECUTION_FAILED",
                merged,
            )
            if result.order is not None:
                self.repository.record_order(result.order)

        if result.success and result.position is not None:
            if self.repository is not None:
                self.repository.record_position(result.position)
            self.telegram.notify_trade_opened(
                result.position,
                decision.checks.get("risk_fraction", 0.0),
                opportunity.expected_net_edge,
                opportunity.opportunity_score.total,
            )
            self.risk.record_slippage(result.slippage)
        elif not result.success:
            self.risk.record_order_rejected(opportunity.symbol)

    # ------------------------------------------------------------------ #
    async def _monitor_loop(self) -> None:
        while self.running:
            try:
                await self._manage_positions()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("monitor_loop_error", error=str(exc), error_type=type(exc).__name__)
            await asyncio.sleep(self.tunables.execution.monitor_interval_sec)

    async def _manage_positions(self) -> None:
        if self.execution is None or not self.execution.positions:
            return

        # In paper mode the simulator's resting orders must be polled.
        if hasattr(self.gateway, "poll"):
            for symbol, _order_id, _price in await self.gateway.poll():
                await self._finalise_simulated_exit(symbol)

        now_ms = self.clock.now_ms()
        for symbol in list(self.execution.positions):
            position = self.execution.positions.get(symbol)
            if position is None:
                continue
            price = self.candles.price(symbol) or position.entry_price
            if price <= 0:
                continue
            position.update_extremes(price)

            reason = self._exit_reason(position, price, now_ms)
            if reason is not None:
                await self.execution.close_position(symbol, reason)
                continue

            await self._update_trailing_stop(position, price)

    def _exit_reason(self, position: Any, price: float, now_ms: int) -> ExitReason | None:
        """Decide whether this position should be closed now.

        Stop and target are handled by resting exchange orders; this covers the
        conditions the exchange cannot know about.
        """
        if position.duration_sec(now_ms) >= self.tunables.trade.max_duration_sec:
            return ExitReason.TIME_LIMIT

        if self.tunables.trade.exit_on_regime_change and self.scanner is not None:
            result = self.scanner.last_result
            if result is not None:
                state = result.regimes.get(position.symbol)
                if state is not None and state.regime.blocks_entries:
                    return ExitReason.REGIME_CHANGE

        _ = price
        return None

    async def _update_trailing_stop(self, position: Any, price: float) -> None:
        cfg = self.tunables.trailing_stop
        if not cfg.enabled or position.initial_risk <= 0:
            return
        if position.r_multiple(price) < cfg.activation_r:
            return

        atr = position.metadata.get("atr", 0.0)
        if atr <= 0:
            atr = abs(position.entry_price - position.initial_stop) / 1.5
        distance = atr * cfg.atr_multiple

        if position.direction is Direction.LONG:
            candidate = position.highest_price - distance
            if cfg.never_below_breakeven:
                candidate = max(candidate, position.entry_price)
            improved = candidate > position.stop_loss
        else:
            candidate = position.lowest_price + distance
            if cfg.never_below_breakeven:
                candidate = min(candidate, position.entry_price)
            improved = candidate < position.stop_loss

        # Only move the stop when the improvement is material: every move is a
        # cancel/replace with a brief unprotected window, so churning it on
        # every tick trades safety for nothing.
        if improved and abs(candidate - position.stop_loss) > atr * 0.1:
            await self.execution.update_stop(position.symbol, candidate)

    async def _finalise_simulated_exit(self, symbol: str) -> None:
        """A simulated protective order filled; record the trade."""
        position = self.execution.positions.get(symbol)
        if position is None:
            return
        await self.execution.close_position(
            symbol, ExitReason.STOP_LOSS, "simulated protective order filled"
        )

    # ------------------------------------------------------------------ #
    async def _reconcile_loop(self) -> None:
        while self.running:
            await asyncio.sleep(self.tunables.execution.reconcile_interval_sec)
            try:
                await self._refresh_account()
                await self.reconciler.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("reconcile_loop_error", error=str(exc), error_type=type(exc).__name__)
                self.risk.record_api_error()

    async def _refresh_account(self) -> None:
        try:
            account = await self.gateway.get_account()
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            self.health.fail("exchange_rest", str(exc)[:200])
            self.risk.record_api_error()
            return

        self._connected = True
        self.health.beat("exchange_rest")
        self.equity = account.equity or self.equity
        self.available_balance = account.available_balance
        self.peak_equity = max(self.peak_equity, self.equity)

        if self.scanner is not None:
            self.scanner.universe_builder.update_equity(self.equity)

        if self.repository is not None:
            state = self.risk.portfolio_state(
                RiskContext(
                    equity=self.equity,
                    available_balance=self.available_balance,
                    positions=dict(self.execution.positions) if self.execution else {},
                    prices=self._prices(),
                    symbol_info=None,  # type: ignore[arg-type]
                    realized_pnl_today=self.realized_pnl_today,
                )
            )
            self.repository.record_equity(
                self.clock.now_ms(),
                self.equity,
                account.total_balance,
                unrealized_pnl=account.unrealized_pnl,
                realized_pnl_today=self.realized_pnl_today,
                open_positions=state.position_count,
                total_exposure=state.total_exposure,
                open_risk=state.total_open_risk,
                margin_used=state.margin_used,
                drawdown=self._drawdown(),
            )

    # ------------------------------------------------------------------ #
    async def _health_loop(self) -> None:
        while self.running:
            await asyncio.sleep(self.tunables.health.heartbeat_interval_sec)
            try:
                report = self.health.check()
                if self.repository is not None:
                    self.health.beat("database" if self.repository.health.available else "database")
                    if not self.repository.health.available:
                        self.health.degrade("database", self.repository.health.last_error or "")

                stale = time.time() - self._last_market_data if self._last_market_data else 0.0
                switches = self.risk.kill_switches.evaluate(self.equity, stale, self._connected)
                for switch in switches:
                    self.telegram.notify_kill_switch(switch.name.value, switch.reason, self.equity)
                _ = report
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("health_loop_error", error=str(exc))

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    async def _on_trade_completed(self, event: Event) -> None:
        trade: Trade = event.payload
        self.trades.append(trade)
        self.realized_pnl_today += trade.net_pnl
        self.total_pnl += trade.net_pnl

        realised_edge = trade.net_pnl / trade.entry_notional if trade.entry_notional > 0 else 0.0
        self.risk.record_trade_closed(
            trade.symbol,
            trade.strategy,
            won=trade.net_pnl > 0,
            r_multiple=trade.r_multiple,
            volatility=trade.metadata.get("volatility", 0.0),
            reason=trade.exit_reason.value,
        )
        self.pipeline.edge_calculator.record_result(
            trade.strategy,
            won=trade.net_pnl > 0,
            gross_return=trade.gross_pnl / trade.entry_notional
            if trade.entry_notional > 0
            else 0.0,
            expected_edge=trade.expected_net_edge,
            realised_edge=realised_edge,
        )

        if self.repository is not None:
            self.repository.record_trade(trade, realised_edge)
        self.telegram.notify_trade_closed(trade)

    async def _on_risk_event(self, event: Event) -> None:
        risk_event: RiskEvent = event.payload
        if self.repository is not None:
            self.repository.record_risk_event(risk_event)
        if risk_event.severity in {"CRITICAL", "ERROR"}:
            self.telegram.notify_risk_event(risk_event)

    def _on_safe_mode(self, reason: str) -> None:
        if self.execution is not None:
            self.execution.block_entries(f"safe mode: {reason}")
        if self.telegram is not None:
            self.telegram.notify_system_alert("Safe mode entered — new entries disabled", reason)

    def _on_recovered(self) -> None:
        if self.execution is not None:
            self.execution.unblock_entries()
        if self.telegram is not None:
            self.telegram.notify_system_alert("Safe mode cleared — entries re-enabled")

    # ------------------------------------------------------------------ #
    # Views for the dashboard
    # ------------------------------------------------------------------ #
    def _prices(self) -> dict[str, float]:
        if self.execution is None:
            return {}
        return {
            symbol: self.candles.price(symbol) or position.entry_price
            for symbol, position in self.execution.positions.items()
        }

    def _drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def _seconds_to_funding(self, mark: Any) -> float:
        if mark is None or not getattr(mark, "next_funding_time", 0):
            return float("inf")
        return max(0.0, (mark.next_funding_time - self.clock.now_ms()) / 1000.0)

    async def status_snapshot(self) -> dict[str, Any]:
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        gross_profit = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl <= 0))
        state = self.risk.portfolio_state(
            RiskContext(
                equity=self.equity,
                available_balance=self.available_balance,
                positions=dict(self.execution.positions) if self.execution else {},
                prices=self._prices(),
                symbol_info=None,  # type: ignore[arg-type]
            )
        )
        rejections = {**self.pipeline.rejections, **self.risk.rejections}

        return {
            "mode": self.config.mode.value,
            "testnet": self.config.settings.binance_testnet,
            "equity": self.equity,
            "available_balance": self.available_balance,
            "today_pnl": self.realized_pnl_today,
            "total_pnl": self.total_pnl,
            "open_positions": len(self.execution.positions) if self.execution else 0,
            "total_trades": len(self.trades),
            "win_rate": wins / len(self.trades) if self.trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
            "drawdown": self._drawdown(),
            "total_fees": sum(t.fees for t in self.trades),
            "total_funding": sum(t.funding for t in self.trades),
            "open_risk_pct": state.open_risk_fraction,
            "exposure_ratio": state.exposure_ratio,
            "safe_mode": self.health.safe_mode,
            "entries_allowed": (
                self.risk.kill_switches.entries_allowed
                and not self.health.safe_mode
                and not (self.execution.entries_blocked if self.execution else False)
            ),
            "uptime_sec": time.time() - self.started_at,
            "rejections": [
                {"reason": reason, "count": count}
                for reason, count in sorted(rejections.items(), key=lambda kv: -kv[1])[:12]
            ],
        }

    def open_positions_view(self) -> list[dict[str, Any]]:
        if self.execution is None:
            return []
        now_ms = self.clock.now_ms()
        out = []
        for symbol, position in self.execution.positions.items():
            price = self.candles.price(symbol) or position.entry_price
            out.append(
                {
                    "symbol": symbol,
                    "direction": position.direction.value,
                    "entry_price": position.entry_price,
                    "current_price": price,
                    "quantity": position.quantity,
                    "leverage": position.leverage,
                    "unrealized_pnl": position.unrealized_pnl(price),
                    "unrealized_pct": position.unrealized_pnl_pct(price),
                    "r_multiple": position.r_multiple(price),
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "duration": format_duration(position.duration_sec(now_ms)),
                    "strategy": position.strategy,
                    "score": position.opportunity_score,
                    "trailing": position.trailing_active,
                    "adopted": position.adopted,
                }
            )
        return out

    def opportunities_view(self) -> list[dict[str, Any]]:
        if self.scanner is None or self.scanner.last_result is None:
            return []
        return self.scanner.last_result.table()

    def strategy_view(self) -> dict[str, Any]:
        weights = self.risk.strategy_weights(list(self.registry.strategies))
        out: dict[str, Any] = {}
        for name in self.registry.strategies:
            performance = self.risk.allocator.performance_for(name).as_dict()
            performance["allocation_weight"] = weights.get(name, 1.0)
            performance["suspended"] = name in self.risk.suspended_strategies
            out[name] = performance
        return {"strategies": out, "registry": self.registry.stats()}

    def risk_view(self) -> dict[str, Any]:
        return self.risk.stats()

    async def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.repository is not None:
            rows = await self.repository.recent_trades(limit)
            if rows:
                return rows
        return [
            {
                "symbol": t.symbol,
                "strategy": t.strategy,
                "direction": t.direction.value,
                "net_pnl": t.net_pnl,
                "r_multiple": t.r_multiple,
                "fees": t.fees,
                "exit_reason": t.exit_reason.value,
                "closed_at": t.closed_at,
            }
            for t in reversed(self.trades[-limit:])
        ]

    async def recent_decisions(
        self, limit: int = 100, accepted: bool | None = None
    ) -> list[dict[str, Any]]:
        if self.repository is None:
            return []
        return await self.repository.recent_decisions(limit, accepted)


class _MarketAdapter:
    """Gives the paper broker read-only market access via the REST client."""

    def __init__(self, engine: TradingEngine, rest: Any) -> None:
        self._engine = engine
        self._rest = rest

    @property
    def symbols(self) -> dict[str, Any]:
        return self._rest.symbols

    def symbol_info(self, symbol: str) -> Any:
        return self._rest.symbol_info(symbol)

    async def load_symbols(self) -> dict[str, Any]:
        return await self._rest.load_symbols()

    async def get_klines(self, *args: Any, **kwargs: Any) -> Any:
        return await self._rest.get_klines(*args, **kwargs)

    async def get_book_ticker(self, symbol: str | None = None) -> Any:
        return await self._rest.get_book_ticker(symbol)

    async def get_ticker_24h(self, symbol: str | None = None) -> Any:
        return await self._rest.get_ticker_24h(symbol)

    async def get_mark_price(self, symbol: str | None = None) -> Any:
        return await self._rest.get_mark_price(symbol)

    def price(self, symbol: str) -> float:
        return self._engine.candles.price(symbol)

    def book(self, symbol: str) -> Any:
        return self._engine.book_tickers.get(symbol)
