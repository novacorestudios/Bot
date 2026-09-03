"""Event-driven backtester.

Replays historical bars through the *same* pipeline the live engine uses:
scanner scoring, regime detection, strategies, aggregation, edge filter and risk
engine. Only the exchange is simulated. A backtester that reimplements the
trading logic tests the reimplementation, not the system.

### The look-ahead rules

These are what separate a backtest worth reading from one that flatters itself:

1. **Decisions use closed bars only.** At bar *i*, the strategies see bars
   `0..i`, all of them closed. The forming bar does not exist.
2. **Fills happen at the NEXT bar's open** (`fill_model: next_open`). A signal
   generated from bar *i*'s close cannot be filled at that same close — in live
   trading the close is already history by the time the signal is computed.
3. **Intrabar resolution is pessimistic.** When a bar's range touches both stop
   and target, the STOP is assumed first. Bar data cannot say which came first,
   and assuming the favourable one inflates every result. This single choice
   often separates a "profitable" strategy from a losing one.
4. **Stops fill at the stop price or worse.** A gap through the stop fills at
   the open, not at the stop.

### What is simulated

Fees (maker/taker), spread, size-aware slippage, funding across funding
timestamps, position sizing, leverage, exchange filters, stop/target/trailing/
time exits, concurrent positions, correlation limits and the full risk budget.

### What is NOT simulated

Stated plainly because these are the gaps between a backtest and reality:

* **Order-book depth.** Slippage is modelled parametrically, not by walking a
  historical book (which is not in kline data at all).
* **Partial fills on entry.** Entries fill completely or not at all.
* **Latency.** Live signal-to-fill delay is not modelled beyond next-bar fills.
* **Exchange outages, rate limiting and rejections.**
* **Market impact.** Assumed negligible, which is true for 75 USDT and false at
  scale.

Every one of these makes a backtest more optimistic than reality.
"""

from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from tradebot.backtesting.execution import (
    CostBreakdown,
    ExecutionAssumptions,
    ExecutionSimulator,
    Scenario,
    scenarios,
)
from tradebot.backtesting.metrics import BacktestMetrics, EquityPoint, compute_metrics
from tradebot.core.clock import SystemClock
from tradebot.core.config import TunableConfig
from tradebot.core.diagnostics import aggregate_rejections
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import from_bps, round_quantity, safe_div
from tradebot.core.types import (
    Candle,
    Direction,
    ExitReason,
    OrderIntent,
    Position,
    SymbolInfo,
    Timeframe,
    Trade,
    new_id,
)
from tradebot.execution.quality import ExecutionQuality, ExecutionRecord
from tradebot.market.candles import CandleStore
from tradebot.market.microstructure import CostModel, LiquiditySnapshot
from tradebot.market.regime import RegimeDetector
from tradebot.market.scoring import MarketScorer, ScoringInputs
from tradebot.risk.engine import RiskContext, RiskEngine
from tradebot.signals.pipeline import Opportunity, SignalPipeline
from tradebot.signals.queue import OpportunityQueue
from tradebot.strategies.base import MarketView
from tradebot.strategies.registry import StrategyRegistry

log = get_logger(__name__)


class ExchangeFilterProvenance(StrEnum):
    """Whether a symbol's trading filters came from stored exchangeInfo."""

    GENUINE = "GENUINE_EXCHANGE_INFO"
    PLACEHOLDER = "PLACEHOLDER"


@dataclass(slots=True)
class BacktestData:
    """Historical bars for one symbol across the timeframes the engine needs."""

    symbol: str
    candles: dict[str, list[Candle]]
    symbol_info: SymbolInfo
    funding_rates: dict[int, float] = field(default_factory=dict)
    exchange_filter_provenance: ExchangeFilterProvenance = ExchangeFilterProvenance.GENUINE

    def primary(self, timeframe: str) -> list[Candle]:
        return self.candles.get(timeframe, [])


@dataclass(slots=True)
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[Trade]
    equity_curve: list[EquityPoint]
    rejections: dict[str, int]
    bars_processed: int
    duration_sec: float
    rejections_by_stage: dict[str, dict[str, int]] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    start_ms: int = 0
    end_ms: int = 0
    #: Symbols whose dataset lacked a timeframe the strategies need. Non-empty
    #: means a low trade count is a DATA problem, not a strategy result.
    missing_timeframes: dict[str, list[str]] = field(default_factory=dict)
    liquidations: int = 0
    #: Edge estimates that used an ASSUMED win rate rather than a measured one.
    bootstrap_estimates: int = 0
    bootstrap_strategies: tuple[str, ...] = ()
    strategy_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST RESULT",
            "=" * 60,
            *self.metrics.summary_lines(),
            "",
            f"Bars processed         {self.bars_processed}",
            f"Wall time              {self.duration_sec:.1f}s",
        ]
        if self.rejections:
            lines += ["", "Why trades were not taken:"]
            lines += [
                f"  {reason:<32} {count}"
                for reason, count in sorted(self.rejections.items(), key=lambda kv: -kv[1])
            ]
        if self.bootstrap_estimates:
            lines += [
                "",
                "*** BOOTSTRAP MODE WAS ACTIVE ***",
                f"  {self.bootstrap_estimates} edge estimates used an ASSUMED win",
                "  rate (break-even + margin) rather than a measured one, for:",
                f"  {', '.join(self.bootstrap_strategies)}",
                "  This result shows what WOULD happen if that assumption held.",
                "  It is NOT evidence that it does. The measured win rates below",
                "  are the actual output of this run; feed them into a second",
                "  run with bootstrap disabled to see whether the edge survives.",
            ]
        if self.strategy_stats:
            lines += ["", "Measured per-strategy win rates:"]
            for name, stats in sorted(self.strategy_stats.items()):
                trades = int(stats.get("trades", 0))
                if trades:
                    rate = stats.get("wins", 0) / trades
                    lines.append(f"  {name:<24} {trades:>4} trades  {rate * 100:.1f}% win rate")
        if self.metrics.warnings:
            lines += ["", "WARNINGS:"]
            lines += [f"  ! {w}" for w in self.metrics.warnings]
        lines += [
            "",
            "This result is from SIMULATION. It is not evidence of live",
            "profitability. See IMPLEMENTATION_PLAN.md section 9 for the",
            "out-of-sample, walk-forward and paper-trading gates that are.",
            "=" * 60,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Bar-by-bar replay through the live decision pipeline."""

    def __init__(self, config: TunableConfig, initial_capital: float | None = None) -> None:
        self.config = config
        self.initial_capital = initial_capital or config.account.initial_capital

        self.candles = CandleStore(config.timeframes.history_bars)
        self.cost_model = CostModel(config.edge)
        self.scorer = MarketScorer(config.scanner, self.cost_model)
        self.regime_detector = RegimeDetector(config.regime)
        self.registry = StrategyRegistry.from_config(config)
        self.pipeline = SignalPipeline(config, self.registry, self.cost_model)
        self.risk = RiskEngine(config, self.candles)

        # Simulated account
        self.equity = self.initial_capital
        self.balance = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []

        self.bars_processed = 0
        self._now_ms = 0

        # -- V3 additions, each fixing a BACKTEST_AUDIT.md finding ------------
        # B-1: the universe is re-ranked every cycle from point-in-time data.
        self.universe_log: list[dict[str, Any]] = []
        # B-2: opportunities are scored first and the free slots spent
        # best-first, exactly as the live engine does.
        self.queue = OpportunityQueue(
            ttl_sec=config.opportunity.queue_ttl_sec,
            max_size=config.opportunity.queue_max_size,
            clock=SystemClock(),
        )
        # B-3: preservation needs a drawdown to act on.
        self.peak_equity = self.initial_capital
        self.day_start_equity = self.initial_capital
        self.realized_pnl_today = 0.0
        self._day_index = -1
        # B-5: leverage risk is measured, not assumed away.
        self.liquidations = 0
        # B-8 / §35: expected versus actual execution.
        self.execution_quality = ExecutionQuality(min_samples=10)
        self.simulator: ExecutionSimulator | None = None
        self.cost_breakdown = CostBreakdown()
        self._universe: list[tuple[str, float]] = []
        self._last_scan_ms = 0
        self._funding_times: dict[str, list[int]] = {}
        #: Symbols whose dataset lacks a timeframe the strategies need. A
        #: non-empty mapping means "few trades" is a data problem, not a result.
        self.missing_timeframes: dict[str, list[str]] = {}
        #: The timeframe decisions were actually evaluated on.
        self.decision_interval: str = config.timeframes.primary

    # ------------------------------------------------------------------ #
    def run(
        self,
        data: dict[str, BacktestData],
        start_ms: int | None = None,
        end_ms: int | None = None,
        assumptions: ExecutionAssumptions | None = None,
        seed: int = 0,
    ) -> BacktestResult:
        """Replay every bar in chronological order across all symbols.

        The loop is organised by **timestamp**, not by (timestamp, symbol): the
        universe has to be ranked and the opportunities compared across symbols
        before any of them is acted on. Iterating per symbol — as this did
        before V3 — meant alphabetical order decided who got the last free slot.
        """
        started = time.time()
        # Decisions run at the finest resolution the data holds; the strategies
        # still read their own closed series. See decision_timeframe().
        primary = self.decision_timeframe(data)
        self.decision_interval = primary

        self.simulator = ExecutionSimulator(
            assumptions
            or scenarios(
                self.config.backtest.spread_bps,
                self.config.backtest.slippage_bps,
                self.config.backtest.taker_fee,
                self.config.backtest.maker_fee,
            )[Scenario.BASE],
            seed=seed,
        )

        scan_interval_ms = self.config.scanner.scan_interval_sec * 1000
        self._universe = []
        self._last_scan_ms = -scan_interval_ms

        self._check_timeframe_coverage(data)

        cycles = self._build_timeline(data, primary, start_ms, end_ms)
        if not cycles:
            log.warning("backtest_no_data")
            return self._result(started, 0, 0)

        warmup = self.config.backtest.warmup_bars
        log.info(
            "backtest_starting",
            symbols=len(data),
            cycles=len(cycles),
            warmup=warmup,
            initial_capital=self.initial_capital,
            scenario=self.simulator.assumptions.name.value,
            seed=seed,
            decision_timeframe=primary,
            strategy_primary=self.config.timeframes.primary,
        )

        for index, (timestamp, symbols) in enumerate(cycles):
            self._now_ms = timestamp

            # Feed every timeframe of every symbol up to this moment.
            for symbol in symbols:
                self._advance(data[symbol], timestamp)
                self.bars_processed += 1

            # Manage open positions FIRST: an exit frees budget an entry may
            # need, and a stale position left unexamined can block a better one.
            self._manage_positions(data, timestamp)
            self._roll_day(timestamp)

            if index >= warmup:
                # The universe is re-ranked on the SCAN interval, not every bar
                # — matching the live engine, where the scanner runs every
                # scan_interval_sec and the signal loop works the candidates in
                # between. Re-ranking every bar would be both slower and a
                # different system from the one that would be deployed.
                if timestamp - self._last_scan_ms >= scan_interval_ms:
                    self._universe = self._rank_universe(data, symbols, timestamp)
                    self._last_scan_ms = timestamp
                self._fill_queue(data, self._universe, timestamp)
                self._spend_slots(data, timestamp)

            # Every cycle, not every 50 bars: drawdown measured on a sparse
            # curve misses any drawdown that opens and recovers between samples.
            self._record_equity(data, timestamp)

        self._flatten_all(data, cycles[-1][0])
        self._record_equity(data, cycles[-1][0])

        return self._result(started, cycles[0][0], cycles[-1][0])

    def _check_timeframe_coverage(self, data: dict[str, BacktestData]) -> None:
        """Refuse to run quietly on a dataset the strategies cannot read.

        The strategies are multi-timeframe: momentum, breakout and VWAP read the
        fast/entry/context series, not just the primary. Given a dataset with
        only the primary timeframe they all return INSUFFICIENT_DATA, the
        pipeline rejects every symbol as NO_SIGNAL, and the backtest completes
        successfully with **zero trades** — a result that looks like "no edge"
        and is actually "no data". That is the most expensive kind of silent
        failure a backtest can have, so it is made loud.
        """
        required = set(self.config.timeframes.all())
        missing: dict[str, list[str]] = {}
        for symbol, entry in data.items():
            absent = sorted(required - {tf for tf, bars in entry.candles.items() if bars})
            if absent:
                missing[symbol] = absent

        if not missing:
            return

        sample = dict(sorted(missing.items())[:5])
        log.error(
            "backtest_dataset_missing_timeframes",
            symbols_affected=len(missing),
            required=sorted(required),
            example=sample,
            message="strategies that read these timeframes will return "
            "INSUFFICIENT_DATA and the run will produce few or no trades — "
            "this is a DATA problem, not a strategy result",
        )
        self.missing_timeframes = missing

    # ------------------------------------------------------------------ #
    def decision_timeframe(self, data: dict[str, BacktestData]) -> str:
        """The finest timeframe present, which is how often decisions are made.

        The live engine evaluates every ``signal_interval_sec`` — 15 seconds by
        default. Driving the backtest off the 5-minute primary instead means
        each decision point stands in for twenty live ones, which
        **systematically undercounts short-lived opportunities**: a setup that
        appears and resolves inside one 5m bar is invisible.

        So decisions run at the finest resolution the dataset actually holds,
        normally 1m, while the strategies keep reading properly closed 3m/5m/
        15m/1h series. Nothing is interpolated and no bar is invented.

        **15-second decisions cannot be reconstructed from 1m OHLCV.** Four
        sub-intervals of a minute are not recoverable from its open, high, low
        and close, and pretending otherwise would be fabricating data. 1m is
        therefore a floor on the discrepancy, not a removal of it.
        """
        present = {tf for entry in data.values() for tf, bars in entry.candles.items() if bars}
        candidates = [tf for tf in self.config.timeframes.all() if tf in present]
        if not candidates:
            return self.config.timeframes.primary

        def seconds(timeframe: str) -> int:
            try:
                return Timeframe(timeframe).seconds
            except ValueError:
                return 10**9

        return min(candidates, key=seconds)

    def _build_timeline(
        self, data: dict[str, BacktestData], primary: str, start_ms: int | None, end_ms: int | None
    ) -> list[tuple[int, list[str]]]:
        """Decision points grouped by timestamp, chronologically.

        Grouping is what makes cross-symbol ranking possible at all: at a given
        moment the engine must see every symbol that printed a bar, rank them,
        and only then decide.

        ``primary`` here is the DECISION timeframe, not the strategies' primary
        series — see :meth:`decision_timeframe`.
        """
        grouped: dict[int, list[str]] = {}
        for symbol, entry in data.items():
            for candle in entry.primary(primary):
                if start_ms and candle.open_time < start_ms:
                    continue
                if end_ms and candle.open_time >= end_ms:
                    continue
                grouped.setdefault(candle.open_time, []).append(symbol)
        return [(ts, sorted(symbols)) for ts, symbols in sorted(grouped.items())]

    def _advance(self, data: BacktestData, timestamp: int) -> None:
        """Append every bar that has CLOSED by `timestamp`.

        A bar is available only once its close_time has passed; this is what
        prevents the strategies from seeing the bar they are about to trade.
        """
        for timeframe, candles in data.candles.items():
            series = self.candles.series(data.symbol, timeframe)
            last_open = series.last.open_time if series.last else -1
            for candle in candles:
                if candle.open_time <= last_open:
                    continue
                if candle.close_time > timestamp:
                    break
                series.append(candle)

    # ------------------------------------------------------------------ #
    # Universe -> queue -> slots. The live engine's ordering, reproduced.
    # ------------------------------------------------------------------ #
    def _rank_universe(
        self, data: dict[str, BacktestData], symbols: list[str], timestamp: int
    ) -> list[tuple[str, float]]:
        """Rebuild the tradable universe from POINT-IN-TIME information only.

        Every input here comes from bars that have already closed. Nothing reads
        a future bar, a future ranking or a future volatility — which is the
        whole reason this is computed per cycle rather than once up front from
        the full dataset.

        Symbols already held are excluded: they cannot be entered again, and
        leaving them in would displace a candidate that could be.
        """
        scored: list[tuple[str, float]] = []
        for symbol in symbols:
            if symbol in self.positions:
                continue
            series = self.candles.get(symbol, self.config.timeframes.primary)
            if series is None or not series.ready(self.regime_detector.min_bars()):
                continue
            price = series.last_price
            if price <= 0:
                continue

            liquidity = self._liquidity(data[symbol], price)
            market = self.scorer.score(
                ScoringInputs(
                    symbol=symbol,
                    series=series,
                    liquidity=liquidity,
                    funding_rate=self._funding_rate(data[symbol], timestamp),
                    quote_volume_24h=liquidity.quote_volume_24h,
                    correlation_penalty=self._correlation_penalty(symbol),
                    timestamp=timestamp,
                )
            )
            if liquidity.quote_volume_24h < self.config.scanner.min_24h_quote_volume:
                continue
            if liquidity.spread_bps > self.config.scanner.max_spread_bps:
                continue
            scored.append((symbol, market.total))

        scored.sort(key=lambda item: -item[1])
        top = scored[: self.config.scanner.top_markets]

        # §8: record WHY each symbol entered the universe, so the ranking can be
        # audited after the fact rather than taken on trust.
        for rank, (symbol, score) in enumerate(top, start=1):
            self.universe_log.append(
                {
                    "timestamp": timestamp,
                    "rank": rank,
                    "symbol": symbol,
                    "score": round(score, 3),
                    "reason": "top_by_market_score",
                }
            )
        return top

    def _correlation_penalty(self, symbol: str) -> float:
        """How correlated this symbol is with what is already held."""
        if not self.positions:
            return 0.0
        held = {
            name: (position.direction, position.quantity * position.entry_price)
            for name, position in self.positions.items()
        }
        assessment = self.risk.correlation.assess(symbol, Direction.LONG, 0.0, held)
        return max(0.0, min(1.0, assessment.portfolio_correlation))

    def _fill_queue(
        self, data: dict[str, BacktestData], ranked: list[tuple[str, float]], timestamp: int
    ) -> None:
        """Evaluate every ranked symbol and queue whatever the pipeline accepts."""
        self.queue.clear()
        for symbol, _score in ranked:
            if symbol not in data:
                continue
            opportunity = self._evaluate(data[symbol], timestamp)
            if opportunity is not None:
                self.queue.add(opportunity)

    def _evaluate(self, data: BacktestData, timestamp: int) -> Opportunity | None:
        """Run one symbol through the live analytical pipeline."""
        symbol = data.symbol
        if symbol in self.positions:
            return None

        primary = self.config.timeframes.primary
        series = self.candles.get(symbol, primary)
        if series is None or not series.ready(self.regime_detector.min_bars()):
            return None

        regime_state = self.regime_detector.detect(series)
        price = series.last_price
        if price <= 0:
            return None

        liquidity = self._liquidity(data, price)
        view = MarketView(
            symbol=symbol,
            candles=self.candles,
            regime=regime_state.regime,
            regime_confidence=regime_state.confidence,
            regime_direction=regime_state.direction,
            book_imbalance=0.0,
            spread_bps=liquidity.spread_bps,
            funding_rate=self._funding_rate(data, timestamp),
            now_ms=timestamp,
        )

        market = self.scorer.score(
            ScoringInputs(
                symbol=symbol,
                series=series,
                liquidity=liquidity,
                funding_rate=view.funding_rate,
                quote_volume_24h=liquidity.quote_volume_24h,
                correlation_penalty=self._correlation_penalty(symbol),
                timestamp=timestamp,
            )
        )

        result = self.pipeline.evaluate(
            view,
            market,
            liquidity,
            self.config.risk.expected_edge_notional(self.equity),
            correlation=self._correlation_penalty(symbol),
            strategy_allocation=self.risk.strategy_weights(list(self.registry.strategies)),
            seconds_to_funding=self._seconds_to_funding(data, timestamp),
            now=timestamp / 1000.0,
        )
        return result.opportunity if result.accepted else None

    def _spend_slots(self, data: dict[str, BacktestData], timestamp: int) -> None:
        """Offer the best queued opportunities to risk, best first.

        Only as many as there are free slots. Zero free slots means zero offers —
        never a forced trade.
        """
        configured = self.config.risk.max_concurrent_positions
        allowed = (
            self.risk.preservation.max_positions(configured)
            if self.config.preservation.enabled
            else configured
        )
        free = allowed - len(self.positions)
        if free <= 0:
            return

        for entry in self.queue.take(free):
            opportunity = entry.opportunity
            symbol = opportunity.symbol
            if symbol in self.positions or symbol not in data:
                continue

            context = RiskContext(
                equity=self.equity,
                available_balance=self.balance - self._margin_used(),
                positions=dict(self.positions),
                prices=self._prices(),
                symbol_info=data[symbol].symbol_info,
                realized_pnl_today=self.realized_pnl_today,
                # B-3: without this the preservation ladder never engages and
                # the backtest measures a system with the brakes disconnected.
                drawdown=self._drawdown(),
                now=timestamp / 1000.0,
            )
            decision = self.risk.evaluate(opportunity, context)
            if not decision.approved or decision.intent is None:
                continue

            self._open_position(
                decision.intent, data[symbol], timestamp, opportunity.market.volatility
            )

    def _drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def _roll_day(self, timestamp: int) -> None:
        """Reset the daily loss counter on the UTC day boundary."""
        day = timestamp // 86_400_000
        if day != self._day_index:
            self._day_index = day
            self.day_start_equity = self.equity
            self.realized_pnl_today = 0.0

    # ------------------------------------------------------------------ #
    def _open_position(
        self, intent: OrderIntent, data: BacktestData, timestamp: int, volatility: float
    ) -> None:
        """Fill at the NEXT bar's open — never at the close that produced the signal.

        Three distinct moments, and they are not interchangeable:

        ``signal_at``
            The decision point. Every bar the strategies read had already closed
            by then, which is what keeps the decision free of look-ahead.
        ``order_at``
            When the order was submitted. In the backtest this is the same
            instant as the signal — there is no queueing delay to model beyond
            the simulator's own latency assumption, which is priced into the
            fill rather than the clock.
        ``filled_at``
            When the fill actually happened: the open of the next decision bar.
            This is what ``Position.opened_at`` carries, so duration, the
            maximum-hold cap and funding eligibility all measure from the fill.
        """
        fill_target = self._next_fill(data, timestamp)
        if fill_target is None:
            return
        filled_at, fill_price = fill_target

        quantity = round_quantity(intent.quantity, data.symbol_info.step_size)
        if quantity <= 0:
            return

        series = self.candles.get(intent.symbol, self.decision_interval)
        bar = series.last if series else None
        liquidity = self._liquidity(data, fill_price)

        simulator = self._require_simulator()
        fill = simulator.execute(
            reference_price=fill_price,
            quantity=quantity,
            direction=intent.direction,
            is_entry=True,
            bar=bar,
            depth_notional=liquidity.depth_notional,
        )
        if not fill.filled:
            # Rejected outright, or the scenario refused it. The opportunity is
            # lost — which is exactly what happens live.
            return

        entry_price = fill.price
        quantity = fill.quantity
        notional = quantity * entry_price
        fee = fill.fee
        margin = notional / max(1, intent.leverage)

        # Sizing uses the decision price, while a market fill can be slightly
        # worse.  The hard cap applies to the filled position, not merely the
        # intent, so never admit a position whose realised initial margin is
        # above it.
        if margin > self.config.risk.max_margin_per_trade + 1e-12:
            return

        if margin + fee > self.balance:
            return

        self.balance -= fee
        slippage_cost = fill.slippage_cost
        # §35: what the edge model assumed this leg would cost, versus what it did.
        self.execution_quality.record(
            ExecutionRecord(
                symbol=intent.symbol,
                direction=intent.direction,
                order_type="MARKET",
                is_entry=True,
                reference_price=fill_price,
                fill_price=entry_price,
                quantity=quantity,
                expected_cost=from_bps(
                    self.config.backtest.slippage_bps + self.config.backtest.spread_bps / 2.0
                ),
                latency_ms=simulator.assumptions.latency_ms,
                at_ms=timestamp,
            )
        )

        position = Position(
            position_id=new_id("bt_"),
            symbol=intent.symbol,
            direction=intent.direction,
            quantity=quantity,
            entry_price=entry_price,
            leverage=intent.leverage,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            strategy=intent.strategy,
            regime=intent.regime,
            # opened_at IS filled_at. Never the signal timestamp.
            opened_at=filled_at,
            signal_at=timestamp,
            order_at=timestamp,
            entry_notional=notional,
            allocated_initial_margin=margin,
            entry_fee=fee,
            entry_slippage=slippage_cost,
            initial_stop=intent.stop_loss,
            initial_risk=abs(entry_price - intent.stop_loss) * quantity,
            highest_price=entry_price,
            lowest_price=entry_price,
            opportunity_score=intent.opportunity_score,
            expected_net_edge=intent.expected_net_edge,
            metadata={
                "volatility": volatility,
                "consensus_score": intent.metadata.get("consensus_score", 0.0),
                "entry_reference_price": fill_price,
                "entry_spread_cost": fill.spread_cost,
                "entry_latency_cost": fill.latency_cost,
                "primary_strategy": intent.metadata.get("primary_strategy", intent.strategy),
                "contributing_strategies": intent.metadata.get("contributing_strategies", []),
                "contribution_weights": intent.metadata.get("contribution_weights", {}),
                "edge_context_key": intent.metadata.get("edge_context_key", ""),
            },
        )
        self.positions[intent.symbol] = position

    # ------------------------------------------------------------------ #
    def _manage_positions(self, data: dict[str, BacktestData], timestamp: int) -> None:
        for symbol in list(self.positions):
            position = self.positions[symbol]
            entry = data.get(symbol)
            if entry is None:
                continue
            series = self.candles.get(symbol, self.decision_interval)
            bar = series.last if series else None
            if bar is None:
                continue

            exit_reason, exit_price = self._exit_for(position, bar, timestamp)
            if exit_reason is not None:
                self._close_position(position, exit_price, exit_reason, timestamp, entry)
                continue

            self._update_trailing(position, bar)
            self._apply_funding(position, entry, timestamp)

    def _exit_for(
        self, position: Position, bar: Candle, timestamp: int
    ) -> tuple[ExitReason | None, float]:
        """Decide whether this bar closes the position, pessimistically."""
        # Liquidation first: the exchange does not wait for our stop.
        liquidation = self._liquidation_price(position)
        if liquidation > 0 and (
            (position.direction is Direction.LONG and bar.low <= liquidation)
            or (position.direction is Direction.SHORT and bar.high >= liquidation)
        ):
            self.liquidations += 1
            log.warning(
                "backtest_liquidation",
                symbol=position.symbol,
                entry=position.entry_price,
                liquidation=liquidation,
                leverage=position.leverage,
            )
            return ExitReason.LIQUIDATION, liquidation

        stop_hit = position.is_stop_hit(bar.low, bar.high)
        target_hit = position.is_target_hit(bar.low, bar.high)

        if stop_hit and target_hit:
            # Both touched inside one bar. Bar data cannot say which came first,
            # so assume the stop. Assuming the target inflates every result.
            if self.config.backtest.intrabar == "optimistic":
                return ExitReason.TAKE_PROFIT, position.take_profit
            return ExitReason.STOP_LOSS, self._stop_fill(position, bar)

        if stop_hit:
            return ExitReason.STOP_LOSS, self._stop_fill(position, bar)
        if target_hit:
            return ExitReason.TAKE_PROFIT, position.take_profit

        if position.duration_sec(timestamp) >= self.config.trade.max_duration_sec:
            return ExitReason.TIME_LIMIT, bar.close

        return None, 0.0

    def _liquidation_price(self, position: Position) -> float:
        """Where this position is force-closed.

        A simplified isolated-margin model: the position is liquidated once the
        loss consumes the posted margin less the maintenance requirement. It is
        approximate — real maintenance margin is tiered by notional — but the
        point is that leverage risk is MEASURED rather than assumed away. The
        risk engine refuses entries whose liquidation sits too close; whether it
        succeeds is exactly what a backtest is supposed to test.
        """
        maintenance = 0.004  # 0.4%, Binance's lowest tier
        leverage = max(1, position.leverage)
        move = (1.0 / leverage) - maintenance
        return position.entry_price * (1.0 - move * position.direction.sign)

    @staticmethod
    def _stop_fill(position: Position, bar: Candle) -> float:
        """A gap through the stop fills at the open, not at the stop price."""
        if position.direction is Direction.LONG:
            return (
                min(position.stop_loss, bar.open)
                if bar.open < position.stop_loss
                else position.stop_loss
            )
        return (
            max(position.stop_loss, bar.open)
            if bar.open > position.stop_loss
            else position.stop_loss
        )

    def _update_trailing(self, position: Position, bar: Candle) -> None:
        cfg = self.config.trailing_stop
        if not cfg.enabled or position.initial_risk <= 0:
            return
        position.update_extremes(bar.close)

        r_multiple = position.r_multiple(bar.close)
        if r_multiple < cfg.activation_r:
            return

        atr = position.metadata.get("volatility", 0.0) * position.entry_price
        if atr <= 0:
            atr = abs(position.entry_price - position.initial_stop) / 1.5
        distance = atr * cfg.atr_multiple

        if position.direction is Direction.LONG:
            candidate = position.highest_price - distance
            if cfg.never_below_breakeven:
                candidate = max(candidate, position.entry_price)
            if candidate > position.stop_loss:
                position.stop_loss = candidate
                position.trailing_active = True
        else:
            candidate = position.lowest_price + distance
            if cfg.never_below_breakeven:
                candidate = min(candidate, position.entry_price)
            if candidate < position.stop_loss:
                position.stop_loss = candidate
                position.trailing_active = True

    def _apply_funding(self, position: Position, data: BacktestData, timestamp: int) -> None:
        """Charge every ACTUAL funding event the position has lived through.

        Not "every eight hours since entry", which is what this did before.
        That model charges a position opened at 07:59 as if it had paid the
        08:00 event only at 15:59, and charges one opened at 00:01 for an event
        it missed — both wrong, in opposite directions, and neither visible in
        the output.

        The exchange charges at its own published timestamps. A position pays an
        event if and only if it was open when that timestamp passed:

            entry ─────┬──────────┬───────── exit
                    08:00      16:00          <- both charged
            entry ──────────────────┬──────── exit
                                 16:00        <- only this one

        Rate sign follows the direction: a long pays positive funding, a short
        receives it.
        """
        if not self.config.backtest.apply_funding or not data.funding_rates:
            return

        last_charged = int(position.metadata.get("last_funding_ms", position.opened_at))
        # Events strictly after what we have already charged, up to now. Strict
        # on the lower bound so an event exactly at entry is not charged — the
        # position did not exist when the exchange took its snapshot.
        due = [
            (event_ms, rate)
            for event_ms, rate in data.funding_rates.items()
            if last_charged < event_ms <= timestamp and event_ms > position.opened_at
        ]
        if not due:
            return

        for event_ms, rate in sorted(due):
            if not rate:
                continue
            notional = position.quantity * position.entry_price
            payment = notional * rate * position.direction.sign
            position.funding_paid += payment
            self.balance -= payment
            self.cost_breakdown.funding += payment
            log.debug(
                "funding_charged",
                symbol=position.symbol,
                at_ms=event_ms,
                rate=rate,
                payment=round(payment, 8),
            )

        position.metadata["last_funding_ms"] = max(event for event, _ in due)

    # ------------------------------------------------------------------ #
    def _close_position(
        self,
        position: Position,
        exit_price: float,
        reason: ExitReason,
        timestamp: int,
        data: BacktestData,
    ) -> None:
        series = self.candles.get(position.symbol, self.decision_interval)
        bar = series.last if series else None
        liquidity = self._liquidity(data, exit_price)

        simulator = self._require_simulator()
        # A stop, a liquidation or the time cap is not optional: a reduce-only
        # market order in a liquid perpetual gets done. Modelling an un-exitable
        # position would understate risk in the one direction that matters.
        urgent = reason in {
            ExitReason.STOP_LOSS,
            ExitReason.LIQUIDATION,
            ExitReason.TIME_LIMIT,
            ExitReason.RISK_EVENT,
        }
        fill = simulator.execute(
            reference_price=exit_price,
            quantity=position.quantity,
            direction=position.direction,
            is_entry=False,
            bar=bar,
            depth_notional=liquidity.depth_notional,
            is_exit_urgent=urgent,
        )
        filled = fill.price if fill.filled else exit_price
        slippage_cost = fill.slippage_cost
        # -- the cost ledger -------------------------------------------------
        # One accounting, computed once, so nothing downstream can double-count.
        #
        # `gross` uses the FILLED prices, which already contain spread, slippage
        # and latency. `reference_gross` uses the prices the decision was made
        # at. The difference between them IS the execution cost, which is why
        # subtracting execution costs from `gross` again would be wrong — and is
        # exactly the trap a report walking the fields naively would fall into.
        sign = position.direction.sign
        entry_reference = float(
            position.metadata.get("entry_reference_price", position.entry_price)
        )
        gross = (filled - position.entry_price) * position.quantity * sign
        reference_gross = (exit_price - entry_reference) * position.quantity * sign

        exit_fee = (
            fill.fee if fill.filled else filled * position.quantity * self.config.backtest.taker_fee
        )
        fees = position.entry_fee + exit_fee
        net = gross - fees - position.funding_paid

        entry_spread = float(position.metadata.get("entry_spread_cost", 0.0))
        entry_latency = float(position.metadata.get("entry_latency_cost", 0.0))
        spread_cost = entry_spread + fill.spread_cost
        latency_cost = entry_latency + fill.latency_cost
        execution_costs = spread_cost + position.entry_slippage + slippage_cost + latency_cost

        # The identity must hold to floating-point tolerance. A drift here means
        # a cost is counted twice or not at all, and every derived figure —
        # expectancy, cost ratio, edge calibration — is wrong by that amount.
        identity_error = (reference_gross - execution_costs - fees - position.funding_paid) - net
        if abs(identity_error) > max(1e-9, abs(net) * 1e-6):
            log.error(
                "cost_ledger_does_not_balance",
                symbol=position.symbol,
                error=identity_error,
                reference_gross=reference_gross,
                execution_costs=execution_costs,
                net=net,
                message="a cost is double-counted or missing",
            )

        self.balance += net
        self.equity = self.balance
        self.realized_pnl_today += net
        self.peak_equity = max(self.peak_equity, self.equity)

        trade = Trade(
            trade_id=new_id("t_"),
            symbol=position.symbol,
            strategy=position.strategy,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=filled,
            quantity=position.quantity,
            leverage=position.leverage,
            stop_loss=position.initial_stop,
            take_profit=position.take_profit,
            opened_at=position.opened_at,
            signal_at=position.signal_at,
            order_at=position.order_at,
            closed_at=timestamp,
            gross_pnl=gross,
            fees=fees,
            funding=position.funding_paid,
            slippage_cost=position.entry_slippage + slippage_cost,
            net_pnl=net,
            reference_gross_pnl=reference_gross,
            entry_fee=position.entry_fee,
            exit_fee=exit_fee,
            spread_cost=spread_cost,
            entry_slippage=position.entry_slippage,
            exit_slippage=slippage_cost,
            latency_cost=latency_cost,
            exit_reason=reason,
            regime=position.regime,
            opportunity_score=position.opportunity_score,
            expected_net_edge=position.expected_net_edge,
            consensus_score=position.metadata.get("consensus_score", 0.0),
            entry_notional=position.entry_notional,
            initial_risk=position.initial_risk,
            metadata={
                "primary_strategy": position.metadata.get("primary_strategy", position.strategy),
                "contributing_strategies": position.metadata.get("contributing_strategies", []),
                "contribution_weights": position.metadata.get("contribution_weights", {}),
                "edge_context_key": position.metadata.get("edge_context_key", ""),
            },
        )
        self.trades.append(trade)
        self.cost_breakdown.add_trade(trade)
        del self.positions[position.symbol]

        self.execution_quality.record(
            ExecutionRecord(
                symbol=position.symbol,
                direction=position.direction,
                order_type="MARKET",
                is_entry=False,
                reference_price=exit_price,
                fill_price=filled,
                quantity=position.quantity,
                expected_cost=self.config.backtest.taker_fee,
                latency_ms=simulator.assumptions.latency_ms,
                at_ms=timestamp,
            )
        )

        # Feed the result back so cooldowns, allocation and the strategy kill
        # switch behave exactly as they would live.
        self.risk.record_trade_closed(
            position.symbol,
            position.strategy,
            won=net > 0,
            r_multiple=trade.r_multiple,
            volatility=position.metadata.get("volatility", 0.0),
            reason=reason.value,
            # B-4: without these the strategy x regime matrix records every
            # trade as SIDEWAYS with zero PnL, and looks like a real result.
            regime=position.regime,
            pnl=net,
        )
        self.pipeline.edge_calculator.record_result(
            position.strategy,
            won=net > 0,
            gross_return=safe_div(gross, position.entry_notional, 0.0),
            expected_edge=position.expected_net_edge,
            realised_edge=safe_div(net, position.entry_notional, 0.0),
            target_before_stop=reason is ExitReason.TAKE_PROFIT,
            context_key=str(position.metadata.get("edge_context_key", "")),
        )
        _ = data

    def _flatten_all(self, data: dict[str, BacktestData], timestamp: int) -> None:
        for symbol in list(self.positions):
            position = self.positions[symbol]
            series = self.candles.get(symbol, self.config.timeframes.primary)
            price = series.last_price if series else position.entry_price
            entry = data.get(symbol)
            if entry is not None:
                self._close_position(position, price, ExitReason.MANUAL, timestamp, entry)

    # ------------------------------------------------------------------ #
    def _next_fill(self, data: BacktestData, timestamp: int) -> tuple[int, float] | None:
        """``(filled_at, fill_price)`` for the first DECISION bar after `timestamp`.

        Uses the decision interval, not the strategies' primary: filling a
        1-minute decision at the next 5-minute open would impose four minutes of
        delay the live system does not have.

        Returning the bar's ``open_time`` alongside its open is the whole point.
        The fill happens at a different moment from the signal, and until V3.2
        the engine filled at this bar's price while stamping the position with
        the *signal* timestamp — so every position appeared to have been opened
        one decision interval before it actually was.
        """
        for candle in data.primary(self.decision_interval):
            if candle.open_time > timestamp:
                return candle.open_time, candle.open
        return None

    def _liquidity(self, data: BacktestData, price: float) -> LiquiditySnapshot:
        """Approximate liquidity from bar volume; the book is not in kline data."""
        series = self.candles.get(data.symbol, self.config.timeframes.primary)
        quote_volume = 0.0
        if series is not None and len(series) >= 20:
            # B-9: bars per day is derived from the timeframe, not hardcoded to
            # 288. The old constant was right only for 5m bars and understated
            # volume 5x on 1m data, which feeds both the market score and the
            # slippage model.
            quote_volume = float(series.quote_volumes[-20:].mean()) * self._bars_per_day()
        depth = max(quote_volume * 0.001, 1000.0)
        return LiquiditySnapshot(
            symbol=data.symbol,
            spread_bps=self.config.backtest.spread_bps,
            bid_notional=depth,
            ask_notional=depth,
            book_imbalance=0.0,
            quote_volume_24h=quote_volume,
        )

    def _require_simulator(self) -> ExecutionSimulator:
        """The simulator, or a clear error rather than an AttributeError.

        `run()` always creates one, so this cannot fire in normal use — but an
        `assert` here would be stripped by `python -O` and reappear as an
        AttributeError in production and nowhere else.
        """
        if self.simulator is None:
            raise RuntimeError(
                "the execution simulator is not initialised; call run() rather "
                "than driving the engine's internals directly"
            )
        return self.simulator

    def _bars_per_day(self) -> float:
        from tradebot.core.types import Timeframe

        try:
            seconds = Timeframe(self.config.timeframes.primary).seconds
        except ValueError:
            seconds = 300
        return 86_400.0 / max(1, seconds)

    def _funding_schedule(self, data: BacktestData) -> list[int]:
        """The symbol's actual funding event times, ascending. Cached.

        ``data.funding_rates`` is the single source of truth for funding
        timing — display, seconds-to-funding, expected cost and the charge
        itself all read this same schedule.
        """
        cached = self._funding_times.get(data.symbol)
        if cached is None:
            cached = sorted(data.funding_rates)
            self._funding_times[data.symbol] = cached
        return cached

    def _funding_rate(self, data: BacktestData, timestamp: int) -> float:
        """The most recently settled funding rate as of `timestamp`.

        Live, this is ``premiumIndex.lastFundingRate``: the rate that actually
        settled, used as the estimate of the next one.

        Until V3.2 this snapped the timestamp onto an assumed 8-hour grid and
        looked the bucket up directly. Real funding timestamps do not land on
        that grid — the bulk archive's are calculation times, and Binance has
        moved symbols to 4-hour funding — so the lookup missed and returned
        0.0. A symbol with a full funding history was priced as if funding
        did not exist.
        """
        times = self._funding_schedule(data)
        if not times:
            return 0.0
        index = bisect_right(times, timestamp)
        if index == 0:
            return 0.0
        return data.funding_rates[times[index - 1]]

    def _seconds_to_funding(self, data: BacktestData, timestamp: int) -> float:
        """Seconds until this symbol's next ACTUAL funding event.

        ``inf`` when the schedule has no event after `timestamp` — either
        because no funding history was loaded, or because the data ends first.
        Infinity is the honest answer: the edge model reads it as ''no funding
        falls inside the expected hold'', which is what not knowing should
        mean. It is never replaced with a guessed 00:00/08:00/16:00 boundary.
        The trust gate separately downgrades a run whose funding history is
        missing, so this silence is always reported.
        """
        times = self._funding_schedule(data)
        if not times:
            return float("inf")
        index = bisect_right(times, timestamp)
        if index >= len(times):
            return float("inf")
        return (times[index] - timestamp) / 1000.0

    def _prices(self) -> dict[str, float]:
        return {
            symbol: self.candles.price(symbol, self.config.timeframes.primary)
            or position.entry_price
            for symbol, position in self.positions.items()
        }

    def _margin_used(self) -> float:
        return sum(
            position.allocated_initial_margin
            if position.allocated_initial_margin > 0
            else position.entry_notional / max(1, position.leverage)
            for position in self.positions.values()
        )

    def _record_equity(self, data: dict[str, BacktestData], timestamp: int) -> None:
        """Mark open positions at a price that EXISTED at ``timestamp``.

        The subtlety this fixes: a signal is computed from bars closed at
        ``timestamp`` and filled at the *next* bar's open. At the moment of
        entry the newest closed bar is the one BEFORE the fill — so marking the
        new position against `series.last_price` values it at a price from
        before it existed, and books the entire open-to-previous-close gap as
        instant PnL.

        On a gappy market that is not a rounding error: a 1% overnight gap on a
        5x position is 5% of margin appearing in the equity curve at the instant
        of entry, inflating the curve and understating the drawdown that follows.

        A position opened at this very timestamp is therefore marked at its own
        entry price — its true mark-to-market is zero until a bar actually
        closes after it.
        """
        unrealized = 0.0
        for symbol, position in self.positions.items():
            if position.opened_at >= timestamp:
                # Opened this cycle: no bar has closed since the fill, so the
                # only honest mark is the price we paid.
                continue

            series = self.candles.get(symbol, self.decision_interval)
            last = series.last if series is not None else None
            if last is None or last.close_time > timestamp:
                # No closed bar at or before now: nothing observable to mark at.
                continue
            if last.close_time <= position.opened_at:
                # The newest closed bar predates the fill. Marking against it
                # would price the position before it existed.
                continue

            unrealized += position.unrealized_pnl(last.close)

        _ = data
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(EquityPoint(timestamp, self.equity))

    def _equity_sampling_sec(self) -> float:
        """Seconds between equity samples, measured rather than assumed.

        Sharpe and annualised volatility scale by the square root of the number
        of samples per year, so this number is not cosmetic: getting it wrong by
        a factor of 50 misstates Sharpe by a factor of ~7.

        It used to be hardcoded as `primary_timeframe x 50`, from when equity
        was sampled every 50 bars. Equity is now recorded every cycle, so that
        constant was silently wrong.

        The **median** consecutive gap, not the mean: a single long gap — a data
        hole, or a symbol that stops printing — would drag a mean and distort
        every derived ratio.
        """
        if len(self.equity_curve) < 3:
            try:
                return float(Timeframe(self.config.timeframes.primary).seconds)
            except ValueError:
                return 300.0

        gaps = [
            (later.timestamp - earlier.timestamp) / 1000.0
            for earlier, later in zip(self.equity_curve, self.equity_curve[1:], strict=False)
            if later.timestamp > earlier.timestamp
        ]
        if not gaps:
            return 300.0
        return float(np.median(gaps))

    def _result(self, started: float, start_ms: int, end_ms: int) -> BacktestResult:
        period = self._equity_sampling_sec()

        metrics = compute_metrics(self.trades, self.equity_curve, self.initial_capital, period)
        rejections, rejections_by_stage = aggregate_rejections(
            self.pipeline.rejections, self.risk.rejections
        )
        return BacktestResult(
            metrics=metrics,
            trades=self.trades,
            equity_curve=self.equity_curve,
            rejections=rejections,
            bars_processed=self.bars_processed,
            duration_sec=time.time() - started,
            rejections_by_stage=rejections_by_stage,
            config_snapshot={
                "risk_per_trade": self.config.risk.risk_per_trade,
                "min_expected_edge": self.config.edge.min_expected_edge,
                "min_opportunity_score": self.config.opportunity.min_score,
                "max_concurrent_positions": self.config.risk.max_concurrent_positions,
                "taker_fee": self.config.backtest.taker_fee,
                "slippage_bps": self.config.backtest.slippage_bps,
                "intrabar": self.config.backtest.intrabar,
                "fill_model": self.config.backtest.fill_model,
            },
            start_ms=start_ms,
            end_ms=end_ms,
            missing_timeframes=dict(self.missing_timeframes),
            liquidations=self.liquidations,
            bootstrap_estimates=self.pipeline.edge_calculator.bootstrap_estimates,
            bootstrap_strategies=tuple(sorted(self.pipeline.edge_calculator.bootstrap_strategies)),
            strategy_stats=self.pipeline.edge_calculator.export_stats(),
        )
