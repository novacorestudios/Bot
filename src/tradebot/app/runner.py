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
from tradebot.core.diagnostics import aggregate_rejections
from tradebot.core.events import Event, EventBus, EventType
from tradebot.core.logging import get_logger
from tradebot.core.types import (
    Direction,
    ExitReason,
    MarketRegime,
    RejectionReason,
    RiskEvent,
    Trade,
    TradingMode,
)
from tradebot.execution.exits import ExitContext, ExitEvaluator, holding_edge
from tradebot.execution.quality import ExecutionQuality, ExecutionRecord
from tradebot.execution.state import OrderState
from tradebot.market.candles import CandleStore
from tradebot.market.microstructure import CostModel, snapshot_from_book
from tradebot.market.regime import RegimeDetector
from tradebot.market.scanner import MarketScanner, enrich_candidates
from tradebot.market.scoring import MarketScorer
from tradebot.market.state import DataSource, MarketState
from tradebot.market.universe import UniverseBuilder
from tradebot.risk.engine import RiskContext, RiskEngine
from tradebot.signals.pipeline import SignalPipeline
from tradebot.signals.queue import OpportunityQueue
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
        # Every consumer of market data reads through this object, and only
        # through it. Both WebSocket and REST write into it, so "how old is my
        # view of this symbol?" always has an answer (AUDIT_REPORT.md C-1).
        self.market = MarketState(
            self.candles,
            stale_after_sec=self.tunables.stream.stale_after_sec,
            lagging_after_sec=self.tunables.stream.lagging_after_sec,
            clock=self.clock,
        )
        self.cost_model = CostModel(self.tunables.edge)
        self.registry = StrategyRegistry.from_config(self.tunables)
        self.pipeline = SignalPipeline(self.tunables, self.registry, self.cost_model)
        self.risk = RiskEngine(self.tunables, self.candles, self.clock)
        # Exit conditions the exchange cannot see: the clock, the regime, a
        # flipped signal, an evaporated edge, and a stop that is no longer there.
        self.exits = ExitEvaluator(self.tunables)
        # Measures the gap between the cost the edge filter assumed and the
        # cost actually paid, and feeds the difference back into the model.
        self.execution_quality = ExecutionQuality(
            min_samples=self.tunables.execution.quality_min_samples,
            max_adjustment=self.tunables.execution.quality_max_adjustment,
        )
        #: Latest consensus per symbol, kept so a HELD symbol's exit can be
        #: judged even after it drops out of the top-25 ranking.
        self._consensus: dict[str, tuple[Direction, float, int]] = {}
        # Candidates are scored first and spent best-first, so a merely
        # adequate opportunity high in the MARKET ranking cannot take the slot
        # a better trade needed (AUDIT_REPORT.md M-7).
        self.opportunities = OpportunityQueue(
            ttl_sec=self.tunables.opportunity.queue_ttl_sec,
            max_size=self.tunables.opportunity.queue_max_size,
            clock=self.clock,
        )

        self.gateway: Any = None
        self.execution: Any = None
        self.reconciler: Any = None
        self.scanner: MarketScanner | None = None
        self.repository: Any = None
        self.telegram: Any = None
        self.market_stream: Any = None
        self.user_stream: Any = None

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

        self.started_at = time.time()
        self.running = False
        self._tasks: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()
        self._last_market_data = 0.0
        self._connected = True
        self._reconcile_requested = asyncio.Event()

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
            self.repository.health.available = False
            self.repository.health.last_error = str(exc)[:200]
            self.health.fail("database", str(exc)[:200])
            log.error(
                "database_unavailable",
                error=str(exc)[:200],
                message="NEW ENTRIES ARE BLOCKED until the audit trail is "
                "writable; exits and position management continue",
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

        # -- live data streams --------------------------------------------------- #
        await self._start_streams(rest)

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

    async def _start_streams(self, rest: Any) -> None:
        """Start the WebSocket feeds and make them the primary data path.

        Before this, every component polled REST on the scan interval, so the
        engine was acting on a picture of the market up to 15 seconds old — for
        a scalper that is the difference between an edge and a loss. REST is now
        the fallback and the reconciliation path, not the heartbeat.
        """
        stream_config = self.tunables.stream
        if not stream_config.enabled:
            log.warning(
                "market_stream_disabled",
                message="running on REST polling only; data will lag",
            )
            return

        from tradebot.exchange.binance.ws import MarketStream, UserStream

        ws_url = self.config.settings.ws_url
        self.market_stream = MarketStream(
            base_url=ws_url,
            on_candle=self._on_stream_candle,
            on_book=self._on_stream_book,
            on_mark=self._on_stream_mark,
            timeframes=self.tunables.timeframes.all(),
            include_book=stream_config.include_book,
            include_mark=stream_config.include_mark,
            on_connect=self._on_market_stream_connected,
            on_disconnect=self._on_market_stream_disconnected,
        )
        await self.market_stream.start()

        # The user stream carries real fills. In paper mode the broker is the
        # source of truth for fills, so there is nothing for it to deliver.
        if stream_config.user_stream_enabled and self.config.mode is not TradingMode.PAPER:
            self.user_stream = UserStream(
                base_url=ws_url,
                rest_client=rest,
                on_event=self._on_user_event,
                keepalive_interval=stream_config.keepalive_interval_sec,
                on_connect=self._on_user_stream_connected,
            )
            await self.user_stream.start()

        self.health.beat("market_data", "streams starting")

    # ------------------------------------------------------------------ #
    # Stream callbacks — the write side of MarketState
    # ------------------------------------------------------------------ #
    async def _on_stream_candle(self, symbol: str, timeframe: str, candle: Any) -> None:
        self.market.apply_candle(symbol, timeframe, candle, DataSource.WEBSOCKET)
        self._last_market_data = time.time()

    async def _on_stream_book(self, book: Any) -> None:
        self.market.apply_book(book, DataSource.WEBSOCKET)
        self._last_market_data = time.time()

    async def _on_stream_mark(self, mark: Any) -> None:
        # The mark-price array stream carries every symbol on the exchange;
        # keeping only what we follow stops the state growing without bound.
        if mark.symbol in self.market.subscribed or (
            self.execution is not None and mark.symbol in self.execution.positions
        ):
            self.market.apply_mark(mark, DataSource.WEBSOCKET)

    def _apply_account_update(self, account: dict[str, Any]) -> None:
        """Keep equity current between REST polls.

        ACCOUNT_UPDATE carries wallet balance and per-position unrealised PnL,
        which together are the equity. It does NOT carry available balance, so
        that stays whatever the last REST poll said rather than being guessed:
        available balance gates position sizing, and a wrong value there sizes
        wrongly.
        """
        quote = self.tunables.account.quote_asset
        wallet = next(
            (b["wallet_balance"] for b in account.get("balances", []) if b["asset"] == quote),
            None,
        )
        if wallet is None:
            return
        unrealized = sum(p["unrealized_pnl"] for p in account.get("positions", []))
        self.equity = float(wallet) + float(unrealized)
        self.peak_equity = max(self.peak_equity, self.equity)

        # A position appearing or vanishing that we did not initiate means our
        # local book and the exchange's disagree.
        if account.get("positions"):
            self._reconcile_requested.set()

    async def _on_market_stream_connected(self) -> None:
        self.market.set_stream_connected(True)
        self.health.beat("market_data", "stream connected")
        # A gap in the feed is a gap in our knowledge of our own orders, so a
        # reconnect asks for reconciliation rather than assuming continuity.
        self._reconcile_requested.set()

    async def _on_market_stream_disconnected(self) -> None:
        self.market.set_stream_connected(False)

    async def _on_user_stream_connected(self) -> None:
        self._reconcile_requested.set()

    async def _on_user_event(self, event: str, data: Any) -> None:
        """Route an authenticated user-stream event.

        Order events are informational here; V2-5 gives them authority over the
        order state machine. Account events refresh equity between REST polls.
        """
        from tradebot.exchange.binance import parsers

        if event == "ORDER_TRADE_UPDATE":
            update = parsers.parse_order_update(data)
            if update is not None:
                log.debug(
                    "user_order_update",
                    symbol=update.get("symbol"),
                    status=update.get("status"),
                    client_order_id=update.get("client_order_id"),
                )
                if self.execution is not None and hasattr(self.execution, "on_order_update"):
                    await self.execution.on_order_update(update)
        elif event == "ACCOUNT_UPDATE":
            account = parsers.parse_account_update(data)
            if account is not None:
                self._apply_account_update(account)
        elif event == "MARGIN_CALL":
            log.critical(
                "margin_call_received", data_keys=sorted(data) if isinstance(data, dict) else None
            )
            self.health.fail("execution", "margin call from exchange")

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

        if self.user_stream is not None:
            await self.user_stream.stop()
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

        # The subscribed set follows the ranking: candidates we might enter,
        # plus everything we hold — a position must never lose its data feed
        # because its symbol dropped out of the top 25.
        follow = {c.symbol for c in result.candidates} | protected
        await self._resubscribe(follow)

        # Retain candles for candidates plus anything we hold.
        self.candles.retain({c.symbol for c in result.candidates}, protected)

        await self._rest_backfill()

    async def _resubscribe(self, symbols: set[str]) -> None:
        """Point the stream at the current symbol set."""
        self.market.set_subscribed(symbols)
        if self.market_stream is not None:
            await self.market_stream.set_symbols(sorted(symbols))

    async def _rest_backfill(self) -> None:
        """Fill in over REST whatever the stream has not delivered.

        Two cases are covered by the same call: the seconds after a
        resubscription, before the first kline arrives for a newly ranked
        symbol, and a symbol whose stream has genuinely gone quiet. Neither is
        treated as an error — but both are recorded, because a rising fallback
        count means the stream is not doing its job.
        """
        if not self.tunables.stream.rest_fallback_enabled:
            return

        stale = self.market.stale_symbols()
        if not stale and self.market_stream is not None:
            return

        try:
            books = await self.gateway.get_book_ticker()
        except Exception as exc:  # noqa: BLE001
            log.warning("rest_book_fallback_failed", error=str(exc)[:200])
            self.risk.record_api_error()
            return

        for symbol, book in books.items():
            if symbol in self.market.subscribed:
                self.market.apply_book(book, DataSource.REST)
        for symbol in stale:
            self.market.record_rest_fallback(symbol)

        with contextlib.suppress(Exception):
            for symbol, mark in (await self.gateway.get_mark_price()).items():
                if symbol in self.market.subscribed:
                    self.market.apply_mark(mark, DataSource.REST)

        if stale:
            log.warning(
                "market_data_stale_symbols",
                count=len(stale),
                symbols=stale[:10],
                message="backfilled over REST; entries stay blocked until live",
            )

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

            # Acting on stale data is how a scalper turns a spread into a loss:
            # the price the decision was made at is no longer available. No
            # entry may be opened on a symbol whose feed has gone quiet.
            if not self.market.is_tradable(symbol):
                reason = RejectionReason.STALE_DATA.value
                self.pipeline.rejections[reason] = self.pipeline.rejections.get(reason, 0) + 1
                if self.repository is not None:
                    self.repository.record_decision(
                        symbol,
                        False,
                        "market_data",
                        f"data {self.market.age_sec(symbol):.0f}s old",
                        self.clock.now_ms(),
                        reason,
                        {"freshness": self.market.freshness(symbol).value},
                    )
                continue

            regime_state = result.regimes.get(symbol)
            regime = regime_state.regime if regime_state else MarketRegime.SIDEWAYS
            book = self.market.book(symbol)
            mark = self.market.mark(symbol)
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

            self._record_consensus(symbol, view)

            pipeline_result = self.pipeline.evaluate(
                view,
                candidate.market_score,
                liquidity,
                notional_estimate=self.tunables.risk.expected_edge_notional(self.equity),
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
            self.opportunities.add(opportunity, pipeline_result.audit)

        if best:
            self.scanner.last_result.candidates = enrich_candidates(result.candidates, best)

        await self._spend_slots()

    async def _spend_slots(self) -> None:
        """Offer the best queued opportunities to the risk engine, best first.

        Only as many as there are free position slots: asking risk to judge
        twenty-five candidates when one slot is free wastes the cycle and, worse,
        biases the outcome toward whichever symbol the scanner happened to rank
        first. Zero free slots means zero offers — never a forced trade.
        """
        if self.execution is None:
            return

        configured = self.tunables.risk.max_concurrent_positions
        allowed = (
            self.risk.preservation.max_positions(configured)
            if self.tunables.preservation.enabled
            else configured
        )
        free = allowed - len(self.execution.positions) - len(self.execution.in_flight)
        if free <= 0:
            return

        for entry in self.opportunities.take(free):
            await self._attempt_trade(entry.opportunity, entry.audit)

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
            data_age_sec=self._data_age_sec(opportunity.symbol),
            connected=self._connected,
            entries_blocked=self.execution.entries_blocked or self.health.safe_mode,
            entries_blocked_reason=self.execution.entries_blocked_reason
            or self.health.safe_mode_reason,
            in_flight=self.execution.in_flight,
            drawdown=self._drawdown(),
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
            self._record_execution(opportunity, decision.intent, result)
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

    def _record_execution(self, opportunity: Any, intent: Any, result: Any) -> None:
        """Compare the fill we got against the cost the edge filter assumed.

        Recorded whether the trade turns out well or badly: this measures the
        MODEL, not the trade. A strategy can be right about direction and still
        lose money to an execution cost nobody measured.
        """
        order = result.order
        position = result.position
        if order is None or position is None:
            return

        reference = intent.metadata.get("reference_price", position.entry_price)
        # What the edge filter assumed a single leg would cost.
        costs = opportunity.edge.estimate.costs
        expected_leg_cost = costs.slippage / 2.0 + costs.spread_cost / 2.0

        self.execution_quality.record(
            ExecutionRecord(
                symbol=position.symbol,
                direction=position.direction,
                order_type=order.order_type.value,
                is_entry=True,
                reference_price=reference,
                fill_price=position.entry_price,
                quantity=position.quantity,
                expected_cost=expected_leg_cost,
                at_ms=self.clock.now_ms(),
            )
        )
        self._recalibrate_costs(position.symbol)

    def _recalibrate_costs(self, symbol: str) -> None:
        """Push the measured bias back into the cost model.

        This is the loop that keeps the edge filter honest. Without it, a
        systematically optimistic slippage assumption approves trades that were
        never profitable, and the losses read as strategy failure rather than as
        the measurement error they are.
        """
        if not self.execution_quality.is_calibrated():
            return
        self.cost_model.set_slippage_adjustment(self.execution_quality.slippage_adjustment())
        if self.execution_quality.is_calibrated(symbol):
            self.cost_model.set_slippage_adjustment(
                self.execution_quality.slippage_adjustment(symbol), symbol
            )

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
            price = self.market.price(symbol) or position.entry_price
            if price <= 0:
                continue
            position.update_extremes(price)

            decision = self.exits.evaluate(position, self._exit_context(position, price, now_ms))
            if decision.should_exit and decision.reason is not None:
                await self.execution.close_position(symbol, decision.reason, decision.detail)
                continue

            await self._update_trailing_stop(position, price)

    def _exit_context(self, position: Any, price: float, now_ms: int) -> ExitContext:
        """Gather everything the exit rules read, without deciding anything."""
        symbol = position.symbol

        regime_blocks = False
        if self.scanner is not None and self.scanner.last_result is not None:
            state = self.scanner.last_result.regimes.get(symbol)
            regime_blocks = state is not None and state.regime.blocks_entries

        direction: Direction | None = None
        confidence = 0.0
        latest = self._consensus.get(symbol)
        if latest is not None:
            seen_direction, seen_confidence, at_ms = latest
            # A consensus older than one signal cycle is not evidence of a flip;
            # it is evidence that we have not looked recently.
            if (now_ms - at_ms) / 1000.0 <= self.tunables.scanner.signal_interval_sec * 3:
                direction, confidence = seen_direction, seen_confidence

        return ExitContext(
            price=price,
            now_ms=now_ms,
            signal_direction=direction,
            signal_confidence=confidence,
            regime_blocks=regime_blocks,
            holding_edge=self._holding_edge(position, price),
            stop_order_missing=self._stop_order_missing(position),
        )

    def _record_consensus(self, symbol: str, view: MarketView) -> None:
        """Note the current consensus, whether or not it becomes a trade.

        Recorded from the aggregator rather than from accepted opportunities:
        an opportunity rejected for cost or correlation still tells us which way
        the strategies are leaning, and that is exactly what a flip needs.
        """
        signals, weights = self.registry.evaluate(view, time.time())
        if not weights:
            return
        aggregation = self.pipeline.aggregator.aggregate(
            symbol, signals, weights, view.regime, view.now_ms
        )
        signal = aggregation.signal
        if signal is not None:
            self._consensus[symbol] = (signal.direction, signal.confidence, view.now_ms)

    def _holding_edge(self, position: Any, price: float) -> float | None:
        """Expected net edge of CONTINUING to hold, per unit of notional."""
        if not self.tunables.trade.exit_on_negative_edge:
            return None

        book = self.market.book(position.symbol)
        liquidity = snapshot_from_book(position.symbol, book)
        notional = position.quantity * price
        remaining_sec = max(
            0.0,
            self.tunables.trade.max_duration_sec - position.duration_sec(self.clock.now_ms()),
        )
        costs = self.cost_model.estimate(
            direction=position.direction,
            notional=notional,
            liquidity=liquidity,
            funding_rate=self.market.funding_rate(position.symbol),
            expected_duration_sec=remaining_sec,
            seconds_to_funding=self.market.seconds_to_funding(position.symbol),
        )
        # The entry fee is already paid; only the exit leg is still ahead.
        forward_cost = costs.total - costs.entry_fee

        return holding_edge(
            position,
            price,
            self.pipeline.edge_calculator.win_probability(position.strategy),
            forward_cost,
        )

    def _stop_order_missing(self, position: Any) -> bool:
        """True when the tracker says no protective stop is resting.

        Only trusted once the order has been through the state machine: an
        unknown id means we never tracked it, which is not the same as knowing
        it is gone, and closing a healthy position on that guess would be worse
        than the risk it is meant to avert.
        """
        if self.execution is None or not position.stop_order_id:
            return False
        tracked = self.execution.tracker.get(position.stop_order_id)
        if tracked is None:
            return False
        return tracked.is_terminal and tracked.state is not OrderState.FILLED

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
            # Wake early when a stream reconnect says our view may have a hole
            # in it; otherwise run on the normal interval.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._reconcile_requested.wait(),
                    timeout=self.tunables.execution.reconcile_interval_sec,
                )
            self._reconcile_requested.clear()
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
                await self._check_database()
                report = self.health.check()

                # The kill switch watches the FEED, not the scan loop: a scan
                # that succeeds every five minutes says nothing about whether
                # prices are still arriving.
                stale = self.market.stream_age_sec()
                if stale == float("inf"):
                    stale = time.time() - self._last_market_data if self._last_market_data else 0.0
                if self.market_stream is not None and self.market.stream_is_stale():
                    self.health.degrade("market_data", f"no stream message for {stale:.0f}s")
                else:
                    self.health.beat("market_data")
                switches = self.risk.kill_switches.evaluate(self.equity, stale, self._connected)
                for switch in switches:
                    self.telegram.notify_kill_switch(switch.name.value, switch.reason, self.equity)
                _ = report
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("health_loop_error", error=str(exc))

    async def _check_database(self) -> None:
        """Keep the health monitor's view of the database honest, and retry.

        The audit trail is not optional bookkeeping: without it a trade cannot
        be reconciled against the exchange or learned from afterwards. So a
        failed database fails a CRITICAL component, which stops new entries —
        while exits, trailing stops and the 60-minute cap keep running, because
        refusing to close a position because a log write failed would be far
        more dangerous than the missing row.
        """
        if self.repository is None:
            return

        if self.repository.health.available:
            self.health.beat("database", f"{self.repository.health.writes} rows written")
            return

        recovered = await self.repository.reconnect(
            self.tunables.health.database_reconnect_attempts,
            self.tunables.health.database_reconnect_backoff_sec,
        )
        if recovered:
            self.health.beat("database", "reconnected")
            return

        self.health.fail("database", self.repository.health.last_error or "unavailable")

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    async def _on_trade_completed(self, event: Event) -> None:
        trade: Trade = event.payload
        self.trades.append(trade)
        self.realized_pnl_today += trade.net_pnl
        self.total_pnl += trade.net_pnl

        realised_edge = trade.net_pnl / trade.entry_notional if trade.entry_notional > 0 else 0.0
        self._record_exit_execution(trade)
        self.risk.record_trade_closed(
            trade.symbol,
            trade.strategy,
            won=trade.net_pnl > 0,
            r_multiple=trade.r_multiple,
            volatility=trade.metadata.get("volatility", 0.0),
            reason=trade.exit_reason.value,
            regime=trade.regime,
            pnl=trade.net_pnl,
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

    def _record_exit_execution(self, trade: Trade) -> None:
        """The exit leg's execution quality.

        The reference is the level we were aiming for — the stop or the target,
        whichever this exit was — so the number answers "how much worse than the
        level did we actually get out at?", which is the cost the edge model
        needs to predict.
        """
        reference = {
            ExitReason.STOP_LOSS: trade.stop_loss,
            ExitReason.TRAILING_STOP: trade.stop_loss,
            ExitReason.TAKE_PROFIT: trade.take_profit,
        }.get(trade.exit_reason, trade.exit_price)
        if reference <= 0:
            return

        self.execution_quality.record(
            ExecutionRecord(
                symbol=trade.symbol,
                direction=trade.direction,
                order_type="MARKET",
                is_entry=False,
                reference_price=reference,
                fill_price=trade.exit_price,
                quantity=trade.quantity,
                expected_cost=self.tunables.edge.taker_fee,
                at_ms=self.clock.now_ms(),
            )
        )
        self._recalibrate_costs(trade.symbol)

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
            symbol: self.market.price(symbol) or position.entry_price
            for symbol, position in self.execution.positions.items()
        }

    def _data_age_sec(self, symbol: str) -> float:
        """Age of THIS symbol's data, not of the last scan.

        The risk engine's staleness check used to see the scan-loop timestamp,
        which is a global number: one dead symbol was invisible behind
        twenty-four healthy ones.
        """
        age = self.market.age_sec(symbol)
        if age == float("inf"):
            return time.time() - self._last_market_data if self._last_market_data else 0.0
        return age

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
        rejections, rejections_by_stage = aggregate_rejections(
            self.pipeline.rejections, self.risk.rejections
        )

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
                and self.risk.preservation.entries_allowed
                and not self.health.safe_mode
                and not (self.execution.entries_blocked if self.execution else False)
            ),
            "uptime_sec": time.time() - self.started_at,
            "market_data": self.market.stats(),
            "preservation": self.risk.preservation.state.as_dict(),
            "queued_opportunities": len(self.opportunities),
            "exits": self.exits.stats(),
            "execution_quality": self.execution_quality.stats(),
            "rejections": [
                {"reason": reason, "count": count}
                for reason, count in sorted(rejections.items(), key=lambda kv: -kv[1])[:12]
            ],
            "rejections_by_stage": rejections_by_stage,
        }

    def open_positions_view(self) -> list[dict[str, Any]]:
        if self.execution is None:
            return []
        now_ms = self.clock.now_ms()
        out = []
        for symbol, position in self.execution.positions.items():
            price = self.market.price(symbol) or position.entry_price
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

    def queue_view(self) -> dict[str, Any]:
        """What is waiting, in the order it will be offered to risk."""
        return {"stats": self.opportunities.stats(), "queue": self.opportunities.report()}

    def matrices_view(self) -> dict[str, Any]:
        """Which strategy works in which regime, and on which symbol."""
        return self.risk.matrices.report()

    def execution_quality_view(self) -> dict[str, Any]:
        """Expected versus actual execution cost, per symbol."""
        return {
            "stats": self.execution_quality.stats(),
            "symbols": self.execution_quality.report(),
            "worst": self.execution_quality.worst_symbols(),
            "edge_calibration": {
                name: self.pipeline.edge_calculator.realised_vs_expected(name)
                for name in sorted(self.registry.strategies)
            },
        }

    def market_data_view(self) -> dict[str, Any]:
        """Feed health: what is live, what is lagging, what is stale."""
        return {
            "state": self.market.stats(),
            "symbols": self.market.symbol_report(),
            "streams": [
                stream.stats()
                for stream in (self.market_stream, self.user_stream)
                if stream is not None
            ],
        }

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
        return self._engine.market.price(symbol)

    def book(self, symbol: str) -> Any:
        return self._engine.market.book(symbol)
