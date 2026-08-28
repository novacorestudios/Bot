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
    """Run a historical backtest across all three execution scenarios.

    This is the documented entry point, so it must exercise the real V3 path:
    the scenario runner, the stored exchange filters and the data-trust gate.
    Before V3.1 it called `BacktestEngine.run()` directly with `strict=False`
    and no `symbol_infos`, which meant the documented command silently produced
    a single-scenario run on placeholder filters and possibly damaged data.
    """
    from tradebot.backtesting.data import load_dataset
    from tradebot.backtesting.report import BacktestReport
    from tradebot.backtesting.runner import (
        EdgeMode,
        OOSMode,
        run_scenarios,
        run_strict_oos,
        split_continuous,
    )
    from tradebot.backtesting.trust import TrustLevel, evaluate_trust
    from tradebot.data.store import DataStore

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    required = list(config.tunables.timeframes.all())

    try:
        data, quality = load_dataset(args.data, symbols or None, required, strict=False)
    except DataError as exc:
        print(f"DATA ERROR: {exc}")
        print("\nDownload history first:")
        print("  python scripts/fetch_data.py --top 30 --intervals 1m,3m,5m,15m,1h \\")
        print("      --start 2024-01-01 --end 2025-01-01 --out data")
        return 1

    if not data:
        print(f"no usable data found in {args.data}")
        return 1

    # -- the trust gate: decided BEFORE the run, and it travels with it ----- #
    store = DataStore(args.data)
    exchange_info = store.load_exchange_info()
    trust = evaluate_trust(
        data=data,
        quality=quality,
        required_timeframes=required,
        funding_enabled=config.tunables.backtest.apply_funding,
        have_exchange_info=bool(exchange_info),
        allow_degraded=bool(getattr(args, "allow_degraded", False)),
    )

    print("\n" + "=" * 68)
    print("DATA TRUST")
    print("=" * 68)
    for line in trust.lines():
        print(line)

    if not trust.may_run:
        print(
            "\nREFUSED. These inputs cannot produce a result worth reading.\n"
            "Fix the data, or pass --allow-degraded to inspect a run whose "
            "output will be marked UNTRUSTED."
        )
        return 1

    # -- edge mode: never mixed, always declared --------------------------- #
    tunables = config.tunables
    requested_edge_mode = getattr(args, "edge_mode", "") or ""
    if requested_edge_mode == EdgeMode.RESEARCH_STRICT.value and tunables.edge.bootstrap_enabled:
        tunables = tunables.model_copy(
            update={"edge": tunables.edge.model_copy(update={"bootstrap_enabled": False})}
        )
        print(
            "\nRESEARCH_STRICT: bootstrap disabled. A strategy with no measured\n"
            "evidence will not trade, so this run cannot manufacture an edge."
        )
    elif (
        requested_edge_mode == EdgeMode.LIVE_FAITHFUL.value and not tunables.edge.bootstrap_enabled
    ):
        print(
            "\nNOTE: --edge-mode LIVE_FAITHFUL was requested but the config has\n"
            "bootstrap disabled. Running RESEARCH_STRICT; the config wins."
        )

    if tunables.edge.bootstrap_enabled:
        print("\nNOTE: bootstrap mode is ON (edge mode LIVE_FAITHFUL). Unproven")
        print("strategies are assumed to win at break-even plus a margin, so this")
        print("run measures what WOULD happen if that held — not that it does.")

    start_ms = _parse_date(args.start)
    end_ms = _parse_date(args.end)
    seed = int(getattr(args, "seed", 42))

    # -- all three scenarios, never one ------------------------------------- #
    manifests = store.manifests()
    scenarios = run_scenarios(
        tunables,
        data,
        start_ms,
        end_ms,
        seed=seed,
        manifests=manifests,
        initial_capital=tunables.account.initial_capital,
    )
    scenarios.context.data_trust = trust.level.value
    scenarios.context.edge_mode = (
        EdgeMode.LIVE_FAITHFUL.value
        if tunables.edge.bootstrap_enabled
        else EdgeMode.RESEARCH_STRICT.value
    )
    scenarios.context.universe_provenance = str(getattr(args, "universe", "PRESENT_DAY_UNIVERSE"))

    print("\n" + "=" * 68)
    print("RUN")
    print("=" * 68)
    for line in scenarios.context.lines():
        print(line)

    print("\n" + "=" * 68)
    print("EXECUTION SCENARIOS")
    print("=" * 68)
    header = f"  {'scenario':<14}{'trades':>8}{'net PnL':>12}{'return %':>10}{'win':>8}{'maxDD':>8}"
    print(header)
    for row in scenarios.comparison():
        print(
            f"  {row['scenario']:<14}{row['trades']:>8}{row['net_pnl']:>12.4f}"
            f"{row['return_pct']:>10.3f}{row['win_rate']:>8.3f}{row['max_drawdown']:>8.3f}"
        )
    print(f"\n  Survives STRESS: {scenarios.survives_stress}")

    report = BacktestReport(scenarios)
    payload: dict[str, Any] = report.as_dict()
    payload["trust"] = trust.as_dict()

    # -- out-of-sample ------------------------------------------------------ #
    split_ms = _parse_date(args.split) if args.split else None
    if split_ms:
        strict = bool(getattr(args, "strict_oos", False))
        print("\n" + "=" * 68)
        print(f"OUT-OF-SAMPLE — {'STRICT_OOS' if strict else 'LIVE_LIKE_FORWARD'}")
        print("=" * 68)

        if strict:
            split = run_strict_oos(
                tunables,
                data,
                split_ms,
                start_ms,
                end_ms,
                seed=seed,
                initial_capital=tunables.account.initial_capital,
            )
            scenarios.context.oos_mode = OOSMode.STRICT_OOS.value
            for label, part in (("TRAIN", split.train), ("TEST", split.test)):
                if part is None:
                    continue
                metrics = part.metrics
                print(
                    f"  {label:<6} {metrics.total_trades:>5} trades  "
                    f"return {metrics.total_return * 100:+7.2f}%  "
                    f"win {metrics.win_rate * 100:5.1f}%  "
                    f"maxDD {metrics.max_drawdown * 100:5.1f}%"
                )
            print(f"\n  {split.as_dict()['note']}")
            if split.test and split.test.metrics.total_return <= 0:
                print("\n  OUT-OF-SAMPLE IS NEGATIVE. The train result is not evidence.")
            payload["out_of_sample"] = split.as_dict()
        else:
            print(
                "  ONE CONTINUOUS ADAPTIVE RUN split at the date. The second half\n"
                "  was traded by a system that had already learned from the first,\n"
                "  so this is NOT a clean holdout. Use --strict-oos for that."
            )
            payload["out_of_sample"] = split_continuous(split_ms).as_dict()

    # -- Monte Carlo on the BASE scenario's real trades --------------------- #
    from tradebot.backtesting.execution import Scenario

    base = scenarios.results.get(Scenario.BASE)
    if base and len(base.trades) >= 30:
        from tradebot.backtesting.montecarlo import MonteCarloAnalyzer

        analyzer = MonteCarloAnalyzer(config.tunables.monte_carlo, seed=seed)
        monte = analyzer.run(
            base.trades,
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
    elif base and base.trades:
        print(
            f"\nMonte Carlo skipped: {len(base.trades)} trades is too few to resample meaningfully."
        )

    if trust.level is not TrustLevel.TRUSTED:
        print("\n" + "!" * 68)
        print(trust.banner())
        print("!" * 68)

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nreport written to {path}")
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
