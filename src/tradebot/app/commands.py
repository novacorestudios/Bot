"""CLI command implementations.

Kept out of ``cli.py`` so the argument parsing stays readable and so each
command can be exercised directly from tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tradebot.core.config import AppConfig
from tradebot.core.errors import DataError
from tradebot.core.logging import get_logger

log = get_logger(__name__)


def _write_report(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nreport written to {target}")


def _parse_date(text: str) -> int | None:
    if not text:
        return None
    from datetime import UTC, datetime

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            continue
    raise SystemExit(f"cannot parse date: {text!r} (use YYYY-MM-DD)")


async def run_backtest(config: AppConfig, args: argparse.Namespace) -> int:
    """Run a historical backtest and report."""
    from tradebot.backtesting.data import load_dataset
    from tradebot.backtesting.engine import BacktestEngine
    from tradebot.backtesting.metrics import compute_metrics

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        data, quality = load_dataset(
            args.data,
            symbols or None,
            list(config.tunables.timeframes.all()),
            strict=False,
        )
    except DataError as exc:
        print(f"DATA ERROR: {exc}")
        print("\nDownload history first:")
        print("  python scripts/download_data.py --top 30 --start 2024-01-01")
        return 1

    if not data:
        print(f"no usable data found in {args.data}")
        return 1

    problems = [q for q in quality if q.problems]
    if problems:
        print(f"\nWARNING: {len(problems)} data-quality problems were found:")
        for report in problems[:10]:
            print(f"  {report.symbol} {report.timeframe}: {'; '.join(report.problems)}")
        print("  A backtest on damaged data produces confident wrong numbers.\n")

    if config.tunables.edge.bootstrap_enabled:
        print("\nNOTE: bootstrap mode is ON. Unproven strategies are assumed to")
        print("win at break-even plus a margin, so this run measures what WOULD")
        print("happen if that held — it is not evidence that it does.\n")

    start_ms = _parse_date(args.start)
    end_ms = _parse_date(args.end)

    engine = BacktestEngine(config.tunables, config.tunables.account.initial_capital)
    result = engine.run(data, start_ms, end_ms)
    print(result.report())

    payload: dict[str, Any] = {
        "metrics": result.metrics.as_dict(),
        "config": result.config_snapshot,
        "rejections": result.rejections,
        "bars_processed": result.bars_processed,
        "bootstrap_estimates": result.bootstrap_estimates,
        "bootstrap_strategies": list(result.bootstrap_strategies),
        "strategy_stats": result.strategy_stats,
        "symbols": sorted(data),
        "trades": [
            {
                "symbol": t.symbol,
                "strategy": t.strategy,
                "direction": t.direction.value,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "net_pnl": t.net_pnl,
                "fees": t.fees,
                "funding": t.funding,
                "duration_sec": t.duration_sec,
                "exit_reason": t.exit_reason.value,
                "regime": t.regime.value,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
            }
            for t in result.trades
        ],
    }

    # -- out-of-sample split ------------------------------------------------ #
    split_ms = _parse_date(args.split) if args.split else None
    if split_ms is not None:
        in_sample = [t for t in result.trades if t.closed_at < split_ms]
        out_sample = [t for t in result.trades if t.closed_at >= split_ms]
        in_curve = [p for p in result.equity_curve if p.timestamp < split_ms]
        out_curve = [p for p in result.equity_curve if p.timestamp >= split_ms]

        print("\n" + "=" * 60)
        print(f"IN-SAMPLE / OUT-OF-SAMPLE SPLIT AT {args.split}")
        print("=" * 60)
        for label, trades, points, capital in (
            ("IN-SAMPLE", in_sample, in_curve, config.tunables.account.initial_capital),
            (
                "OUT-OF-SAMPLE",
                out_sample,
                out_curve,
                in_curve[-1].equity if in_curve else config.tunables.account.initial_capital,
            ),
        ):
            metrics = compute_metrics(trades, points, capital)
            print(
                f"\n{label}: {metrics.total_trades} trades, "
                f"return {metrics.total_return * 100:+.2f}%, "
                f"win rate {metrics.win_rate * 100:.1f}%, "
                f"max drawdown {metrics.max_drawdown * 100:.1f}%"
            )
            payload[label.lower().replace("-", "_")] = metrics.as_dict()

        if out_sample:
            out_metrics = compute_metrics(out_sample, out_curve, 1.0)
            if out_metrics.total_return <= 0:
                print(
                    "\n  OUT-OF-SAMPLE IS NEGATIVE. The in-sample result is not "
                    "evidence of anything."
                )
        else:
            print("\n  No out-of-sample trades: the split proves nothing.")

    # -- Monte Carlo -------------------------------------------------------- #
    if len(result.trades) >= 30:
        from tradebot.backtesting.montecarlo import MonteCarloAnalyzer

        analyzer = MonteCarloAnalyzer(config.tunables.monte_carlo, seed=42)
        monte = analyzer.run(
            result.trades,
            config.tunables.account.initial_capital,
            config.tunables.risk.max_drawdown,
        )
        print("\n" + monte.summary())
        payload["monte_carlo"] = {
            "verdict": monte.verdict,
            "median_return": monte.median_return,
            "percentile_5_return": monte.percentile_5_return,
            "percentile_95_drawdown": monte.percentile_95_drawdown,
            "probability_of_loss": monte.probability_of_loss,
            "probability_of_ruin": monte.probability_of_ruin,
            "warnings": monte.warnings,
        }
    elif result.trades:
        print(
            f"\nMonte Carlo skipped: {len(result.trades)} trades is too few "
            f"to resample meaningfully."
        )

    _write_report(args.report, payload)
    return 0


async def run_walkforward(config: AppConfig, args: argparse.Namespace) -> int:
    """Run a walk-forward analysis."""
    from tradebot.backtesting.data import load_dataset
    from tradebot.backtesting.walkforward import WalkForwardAnalyzer

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        data, _quality = load_dataset(
            args.data,
            symbols or None,
            list(config.tunables.timeframes.all()),
            strict=False,
        )
    except DataError as exc:
        print(f"DATA ERROR: {exc}")
        return 1

    if not data:
        print(f"no usable data found in {args.data}")
        return 1

    report = WalkForwardAnalyzer(config.tunables).run(data)
    print(report.summary())

    _write_report(
        args.report,
        {
            "verdict": report.verdict,
            "consistency": report.consistency,
            "mean_test_return": report.mean_test_return,
            "median_test_return": report.median_test_return,
            "mean_efficiency": report.mean_efficiency,
            "total_test_trades": report.total_test_trades,
            "warnings": report.warnings,
            "folds": [
                {
                    **result.fold.as_dict(),
                    "train_return": result.train_return,
                    "test_return": result.test_return,
                    "test_trades": result.test_trades,
                    "efficiency": result.efficiency,
                }
                for result in report.folds
            ],
        },
    )
    return 0


async def run_scan_once(config: AppConfig, top: int | None = None) -> int:
    """Run a single scanner cycle and print the ranked candidates."""
    from tradebot.exchange.binance.rest import BinanceFuturesREST
    from tradebot.market.candles import CandleStore
    from tradebot.market.microstructure import CostModel
    from tradebot.market.regime import RegimeDetector
    from tradebot.market.scanner import MarketScanner
    from tradebot.market.scoring import MarketScorer
    from tradebot.market.universe import UniverseBuilder

    settings = config.settings
    tunables = config.tunables

    client = BinanceFuturesREST(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        base_url=settings.rest_url,
        recv_window=settings.binance_recv_window,
    )
    try:
        await client.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"cannot reach Binance: {exc}")
        print("Run scripts/verify_connectivity.py to diagnose.")
        return 1

    try:
        cost_model = CostModel(tunables.edge)
        scanner_config = tunables.scanner
        scanner = MarketScanner(
            config=scanner_config,
            gateway=client,
            candles=CandleStore(tunables.timeframes.history_bars),
            scorer=MarketScorer(scanner_config, cost_model),
            regime_detector=RegimeDetector(tunables.regime),
            universe_builder=UniverseBuilder(
                scanner_config,
                tunables.account.initial_capital,
                tunables.execution.max_min_notional_ratio,
            ),
            cost_model=cost_model,
            primary_timeframe=tunables.timeframes.primary,
        )
        result = await scanner.scan()
    finally:
        await client.close()

    rows = result.table()[: top or scanner_config.top_markets]
    if not rows:
        print("no candidates passed the universe filters")
        print(f"exclusions: {result.universe.exclusion_counts()}")
        return 0

    header = (
        f"{'#':>3} {'SYMBOL':<14} {'SCORE':>6} {'REGIME':<16} {'VOL%':>7} {'SPREAD':>7} {'RISK':<7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rank']:>3} {row['symbol']:<14} {row['market_score']:>6.1f} "
            f"{row['regime']:<16} {row['volatility_pct']:>7.3f} "
            f"{row['spread_bps']:>7.2f} {row['risk_level']:<7}"
        )

    print(
        f"\nscanned {result.scanned} of {len(result.universe.entries)} tradable "
        f"symbols in {result.duration_sec:.1f}s"
    )
    print("A high market score means the market is worth WATCHING. It is not a")
    print("signal: entry additionally requires strategy consensus, an")
    print("opportunity score above threshold, and positive expected net edge.")
    return 0
