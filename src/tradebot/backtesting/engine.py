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
from dataclasses import dataclass, field
from typing import Any

from tradebot.backtesting.metrics import BacktestMetrics, EquityPoint, compute_metrics
from tradebot.core.config import TunableConfig
from tradebot.core.logging import get_logger
from tradebot.core.mathutil import from_bps, round_quantity, safe_div
from tradebot.core.types import (
    Candle,
    Direction,
    ExitReason,
    OrderIntent,
    Position,
    SymbolInfo,
    Trade,
    new_id,
)
from tradebot.market.candles import CandleStore
from tradebot.market.microstructure import CostModel, LiquiditySnapshot
from tradebot.market.regime import RegimeDetector
from tradebot.market.scoring import MarketScorer, ScoringInputs
from tradebot.risk.engine import RiskContext, RiskEngine
from tradebot.signals.pipeline import SignalPipeline
from tradebot.strategies.base import MarketView
from tradebot.strategies.registry import StrategyRegistry

log = get_logger(__name__)


@dataclass(slots=True)
class BacktestData:
    """Historical bars for one symbol across the timeframes the engine needs."""

    symbol: str
    candles: dict[str, list[Candle]]
    symbol_info: SymbolInfo
    funding_rates: dict[int, float] = field(default_factory=dict)

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
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    start_ms: int = 0
    end_ms: int = 0
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

    # ------------------------------------------------------------------ #
    def run(
        self, data: dict[str, BacktestData], start_ms: int | None = None, end_ms: int | None = None
    ) -> BacktestResult:
        """Replay every bar in chronological order across all symbols."""
        started = time.time()
        primary = self.config.timeframes.primary

        timeline = self._build_timeline(data, primary, start_ms, end_ms)
        if not timeline:
            log.warning("backtest_no_data")
            return self._result(started, 0, 0)

        warmup = self.config.backtest.warmup_bars
        log.info(
            "backtest_starting",
            symbols=len(data),
            bars=len(timeline),
            warmup=warmup,
            initial_capital=self.initial_capital,
        )

        for index, (timestamp, symbol) in enumerate(timeline):
            self._now_ms = timestamp
            entry = data[symbol]

            # Feed every timeframe up to this moment.
            self._advance(entry, timestamp)
            self.bars_processed += 1

            # Manage open positions FIRST: an exit frees budget for an entry,
            # and processing entries first would let a stale position block one.
            self._manage_positions(data, timestamp)

            if index >= warmup:
                self._consider_entry(entry, timestamp)

            if index % 50 == 0 or index == len(timeline) - 1:
                self._record_equity(data, timestamp)

        # Close anything still open at the end, at the last known price.
        self._flatten_all(data, timeline[-1][0])
        self._record_equity(data, timeline[-1][0])

        return self._result(started, timeline[0][0], timeline[-1][0])

    # ------------------------------------------------------------------ #
    def _build_timeline(
        self, data: dict[str, BacktestData], primary: str, start_ms: int | None, end_ms: int | None
    ) -> list[tuple[int, str]]:
        events: list[tuple[int, str]] = []
        for symbol, entry in data.items():
            for candle in entry.primary(primary):
                if start_ms and candle.open_time < start_ms:
                    continue
                if end_ms and candle.open_time >= end_ms:
                    continue
                events.append((candle.open_time, symbol))
        events.sort()
        return events

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
    def _consider_entry(self, data: BacktestData, timestamp: int) -> None:
        symbol = data.symbol
        if symbol in self.positions:
            return

        primary = self.config.timeframes.primary
        series = self.candles.get(symbol, primary)
        if series is None or not series.ready(self.regime_detector.min_bars()):
            return

        regime_state = self.regime_detector.detect(series)
        price = series.last_price
        if price <= 0:
            return

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
                correlation_penalty=0.0,
                timestamp=timestamp,
            )
        )

        notional_estimate = self.equity * self.config.risk.max_symbol_exposure
        result = self.pipeline.evaluate(
            view,
            market,
            liquidity,
            notional_estimate,
            seconds_to_funding=self._seconds_to_funding(timestamp),
            now=timestamp / 1000.0,
        )
        if not result.accepted or result.opportunity is None:
            return

        context = RiskContext(
            equity=self.equity,
            available_balance=self.balance - self._margin_used(),
            positions=dict(self.positions),
            prices=self._prices(),
            symbol_info=data.symbol_info,
            now=timestamp / 1000.0,
        )
        decision = self.risk.evaluate(result.opportunity, context)
        if not decision.approved or decision.intent is None:
            return

        self._open_position(decision.intent, data, timestamp, market.volatility)

    # ------------------------------------------------------------------ #
    def _open_position(
        self, intent: OrderIntent, data: BacktestData, timestamp: int, volatility: float
    ) -> None:
        """Fill at the NEXT bar's open — never at the close that produced the signal."""
        fill_price = self._next_open(data, timestamp)
        if fill_price is None:
            return

        cfg = self.config.backtest
        # Slippage is adverse by construction: a buy fills higher, a sell lower.
        slippage_fraction = from_bps(cfg.slippage_bps + cfg.spread_bps / 2.0)
        entry_price = fill_price * (1 + slippage_fraction * intent.direction.sign)

        quantity = round_quantity(intent.quantity, data.symbol_info.step_size)
        if quantity <= 0:
            return
        notional = quantity * entry_price
        fee = notional * cfg.taker_fee
        margin = notional / max(1, intent.leverage)

        if margin + fee > self.balance:
            return

        self.balance -= fee
        slippage_cost = abs(entry_price - fill_price) * quantity

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
            opened_at=timestamp,
            entry_notional=notional,
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
            series = self.candles.get(symbol, self.config.timeframes.primary)
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
        """Charge funding when the position crosses a funding timestamp."""
        if not self.config.backtest.apply_funding:
            return
        interval_ms = self.config.edge.funding_interval_hours * 3_600_000
        last_charge = position.metadata.get("last_funding_ms", position.opened_at)
        if timestamp - last_charge < interval_ms:
            return
        rate = self._funding_rate(data, timestamp)
        if rate:
            notional = position.quantity * position.entry_price
            payment = notional * rate * position.direction.sign
            position.funding_paid += payment
            self.balance -= payment
        position.metadata["last_funding_ms"] = timestamp

    # ------------------------------------------------------------------ #
    def _close_position(
        self,
        position: Position,
        exit_price: float,
        reason: ExitReason,
        timestamp: int,
        data: BacktestData,
    ) -> None:
        cfg = self.config.backtest
        # Exits are marketable too, so they pay slippage in the adverse direction.
        slippage_fraction = from_bps(cfg.slippage_bps + cfg.spread_bps / 2.0)
        filled = exit_price * (1 - slippage_fraction * position.direction.sign)
        slippage_cost = abs(filled - exit_price) * position.quantity

        gross = (filled - position.entry_price) * position.quantity * position.direction.sign
        exit_fee = filled * position.quantity * cfg.taker_fee
        fees = position.entry_fee + exit_fee
        net = gross - fees - position.funding_paid

        self.balance += net
        self.equity = self.balance

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
            closed_at=timestamp,
            gross_pnl=gross,
            fees=fees,
            funding=position.funding_paid,
            slippage_cost=position.entry_slippage + slippage_cost,
            net_pnl=net,
            exit_reason=reason,
            regime=position.regime,
            opportunity_score=position.opportunity_score,
            expected_net_edge=position.expected_net_edge,
            consensus_score=position.metadata.get("consensus_score", 0.0),
            entry_notional=position.entry_notional,
            initial_risk=position.initial_risk,
        )
        self.trades.append(trade)
        del self.positions[position.symbol]

        # Feed the result back so cooldowns, allocation and the strategy kill
        # switch behave exactly as they would live.
        self.risk.record_trade_closed(
            position.symbol,
            position.strategy,
            won=net > 0,
            r_multiple=trade.r_multiple,
            volatility=position.metadata.get("volatility", 0.0),
            reason=reason.value,
        )
        self.pipeline.edge_calculator.record_result(
            position.strategy,
            won=net > 0,
            gross_return=safe_div(gross, position.entry_notional, 0.0),
            expected_edge=position.expected_net_edge,
            realised_edge=safe_div(net, position.entry_notional, 0.0),
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
    def _next_open(self, data: BacktestData, timestamp: int) -> float | None:
        """The open of the first bar starting after `timestamp`."""
        primary = self.config.timeframes.primary
        for candle in data.primary(primary):
            if candle.open_time > timestamp:
                return candle.open
        return None

    def _liquidity(self, data: BacktestData, price: float) -> LiquiditySnapshot:
        """Approximate liquidity from bar volume; the book is not in kline data."""
        series = self.candles.get(data.symbol, self.config.timeframes.primary)
        quote_volume = 0.0
        if series is not None and len(series) >= 20:
            quote_volume = float(series.quote_volumes[-20:].mean()) * 288
        depth = max(quote_volume * 0.001, 1000.0)
        return LiquiditySnapshot(
            symbol=data.symbol,
            spread_bps=self.config.backtest.spread_bps,
            bid_notional=depth,
            ask_notional=depth,
            book_imbalance=0.0,
            quote_volume_24h=quote_volume,
        )

    def _funding_rate(self, data: BacktestData, timestamp: int) -> float:
        if not data.funding_rates:
            return 0.0
        interval = self.config.edge.funding_interval_hours * 3_600_000
        bucket = (timestamp // interval) * interval
        return data.funding_rates.get(bucket, 0.0)

    def _seconds_to_funding(self, timestamp: int) -> float:
        interval_ms = self.config.edge.funding_interval_hours * 3_600_000
        return (interval_ms - (timestamp % interval_ms)) / 1000.0

    def _prices(self) -> dict[str, float]:
        return {
            symbol: self.candles.price(symbol, self.config.timeframes.primary)
            or position.entry_price
            for symbol, position in self.positions.items()
        }

    def _margin_used(self) -> float:
        prices = self._prices()
        return sum(
            position.margin(prices.get(symbol, position.entry_price))
            for symbol, position in self.positions.items()
        )

    def _record_equity(self, data: dict[str, BacktestData], timestamp: int) -> None:
        unrealized = 0.0
        for symbol, position in self.positions.items():
            series = self.candles.get(symbol, self.config.timeframes.primary)
            price = series.last_price if series else position.entry_price
            unrealized += position.unrealized_pnl(price)
            _ = data
        self.equity = self.balance + unrealized
        self.equity_curve.append(EquityPoint(timestamp, self.equity))

    def _result(self, started: float, start_ms: int, end_ms: int) -> BacktestResult:
        from tradebot.core.types import Timeframe

        try:
            period = float(Timeframe(self.config.timeframes.primary).seconds) * 50
        except ValueError:
            period = 300.0 * 50

        metrics = compute_metrics(self.trades, self.equity_curve, self.initial_capital, period)
        return BacktestResult(
            metrics=metrics,
            trades=self.trades,
            equity_curve=self.equity_curve,
            rejections={**self.pipeline.rejections, **self.risk.rejections},
            bars_processed=self.bars_processed,
            duration_sec=time.time() - started,
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
            bootstrap_estimates=self.pipeline.edge_calculator.bootstrap_estimates,
            bootstrap_strategies=tuple(sorted(self.pipeline.edge_calculator.bootstrap_strategies)),
            strategy_stats=self.pipeline.edge_calculator.export_stats(),
        )
