"""Command-line entry point.

Subcommands:

    validate-config   parse and validate configuration, print a summary, exit
    run               start the trading engine (PAPER unless --live is given)
    scan              run one scanner cycle and print the ranked candidates
    backtest          run a historical backtest
    walkforward       run a walk-forward analysis
    doctor            check the environment and report what is missing

``--live`` is required, in addition to the two environment variables, before the
engine will place a real order.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from tradebot.core.config import AppConfig, load_config
from tradebot.core.errors import ConfigError, SafetyError, TradeBotError
from tradebot.core.logging import configure_logging, get_logger, register_secret
from tradebot.core.types import TradingMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradebot",
        description="Dynamic Multi-Strategy Binance USDⓈ-M Futures Trading Engine",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Acknowledge live trading. Also requires TRADING_MODE=LIVE and "
        "I_UNDERSTAND_LIVE_TRADING_RISK=YES. Without all three, refuses to start.",
    )
    parser.add_argument("--config", help="Override CONFIG_FILE")
    parser.add_argument("--log-level", help="Override LOG_LEVEL")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-config", help="Validate configuration and exit")
    sub.add_parser("doctor", help="Report environment readiness")

    run = sub.add_parser("run", help="Run the trading engine")
    run.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds (0 = run until interrupted)",
    )

    scan = sub.add_parser("scan", help="Run one scanner cycle and print the ranking")
    scan.add_argument("--top", type=int, default=0, help="Override top_markets")

    bt = sub.add_parser("backtest", help="Run a historical backtest")
    bt.add_argument("--data", required=True, help="Directory of historical klines")
    bt.add_argument("--symbols", default="", help="Comma-separated symbols (default: all)")
    bt.add_argument("--start", default="", help="ISO date, inclusive")
    bt.add_argument("--end", default="", help="ISO date, exclusive")
    bt.add_argument("--split", default="", help="Split into train/test at this ISO date")
    bt.add_argument(
        "--strict-oos",
        action="store_true",
        help="Freeze learned state at --split so the test period is a real "
        "holdout. Without this the split is ONE continuous adaptive run and is "
        "labelled LIVE_LIKE_FORWARD, not a clean holdout.",
    )
    bt.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Run on data with gaps. The result is marked UNTRUSTED.",
    )
    bt.add_argument(
        "--universe",
        choices=[
            "POINT_IN_TIME_UNIVERSE",
            "PRESENT_DAY_UNIVERSE",
            "MANUAL_SMOKE_UNIVERSE",
        ],
        default="PRESENT_DAY_UNIVERSE",
        help="How the symbol list was chosen. POINT_IN_TIME came from a listing "
        "snapshot taken at the start of the period — the only one free of "
        "survivorship bias. PRESENT_DAY was ranked by today's liquidity, so "
        "symbols delisted since are missing. MANUAL_SMOKE was hand-picked to "
        "exercise the pipeline and is not a research universe at all: it is "
        "neither point-in-time nor ranked, and no result from it generalises. "
        "Say which one it was rather than letting a report imply otherwise.",
    )
    bt.add_argument(
        "--edge-mode",
        choices=["LIVE_FAITHFUL", "RESEARCH_STRICT"],
        default="",
        help="LIVE_FAITHFUL keeps bootstrap on, so unproven strategies trade on "
        "an ASSUMED win rate — faithful to the live system, and partly a "
        "measurement of the assumption. RESEARCH_STRICT turns it off: a strategy "
        "with no evidence does not trade. Default: whatever the config says.",
    )
    bt.add_argument("--seed", type=int, default=42, help="Execution RNG seed")
    bt.add_argument("--report", default="reports/backtest.json")

    wf = sub.add_parser("walkforward", help="Run a walk-forward analysis")
    wf.add_argument("--data", required=True)
    wf.add_argument("--symbols", default="")
    wf.add_argument("--report", default="reports/walkforward.json")
    # The same two flags the `backtest` command takes, because walk-forward now
    # goes through the same trust gate and the same execution inputs.
    wf.add_argument(
        "--allow-degraded",
        action="store_true",
        help="run on gapped data; the result is marked UNTRUSTED",
    )
    wf.add_argument("--seed", type=int, default=42, help="execution simulator seed")

    return parser


def _configure(args: argparse.Namespace) -> AppConfig:
    import os

    if args.config:
        os.environ["CONFIG_FILE"] = args.config
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    config = load_config(live_flag=args.live)
    settings = config.settings

    # Register secrets BEFORE any logging happens, so nothing can leak.
    register_secret(settings.binance_api_key)
    register_secret(settings.binance_api_secret)
    register_secret(settings.telegram_bot_token)
    register_secret(settings.dashboard_token)

    configure_logging(settings.log_level, settings.log_format, settings.log_file)
    return config


def _summarise(config: AppConfig) -> dict[str, Any]:
    t = config.tunables
    return {
        "mode": config.mode.value,
        "testnet": config.settings.binance_testnet,
        "credentials_present": config.settings.has_credentials,
        "initial_capital": t.account.initial_capital,
        "risk_per_trade": t.risk.risk_per_trade,
        "max_concurrent_positions": t.risk.max_concurrent_positions,
        "max_daily_loss": t.risk.max_daily_loss,
        "max_drawdown": t.risk.max_drawdown,
        "max_leverage": t.risk.max_leverage,
        "top_markets": t.scanner.top_markets,
        "min_opportunity_score": t.opportunity.min_score,
        "min_expected_edge": t.edge.min_expected_edge,
        "max_trade_duration_sec": t.trade.max_duration_sec,
        "strategies_configured": sorted(t.strategies),
        "strategies_enabled": sorted(
            name for name, params in t.strategies.items() if params.get("enabled", True)
        ),
    }


def cmd_validate_config(config: AppConfig) -> int:
    log = get_logger("cli")
    summary = _summarise(config)
    log.info("configuration_valid", **summary)

    print("configuration OK")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if config.mode is TradingMode.LIVE:
        print("\n  *** LIVE MODE — real orders will be placed with real money ***")
    return 0


def cmd_doctor(config: AppConfig) -> int:
    """Report what is ready and what is missing, without touching the network."""
    s = config.settings
    checks: list[tuple[str, bool, str]] = [
        ("configuration parses", True, ""),
        (
            "API credentials present",
            s.has_credentials,
            "set BINANCE_API_KEY and BINANCE_API_SECRET in .env",
        ),
        (
            "testnet selected",
            s.binance_testnet,
            "BINANCE_TESTNET=false — this is a REAL money endpoint",
        ),
        (
            "telegram configured",
            bool(s.telegram_bot_token and s.telegram_chat_id),
            "alerts are disabled; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
        ),
        (
            "dashboard token set",
            bool(s.dashboard_token),
            "dashboard will refuse non-local connections without DASHBOARD_TOKEN",
        ),
        ("mode is not LIVE", config.mode is not TradingMode.LIVE, "LIVE mode is armed"),
    ]
    failures = 0
    for name, ok, hint in checks:
        mark = "ok  " if ok else "WARN"
        print(f"[{mark}] {name}" + ("" if ok else f"  -> {hint}"))
        if not ok:
            failures += 1
    print(f"\n{len(checks) - failures}/{len(checks)} checks clean")
    print(
        "Note: no network call was made. Run `scripts/verify_connectivity.py` "
        "on a host with access to Binance to test connectivity."
    )
    return 0


async def _run_async(config: AppConfig, duration: float) -> int:
    from tradebot.app.runner import TradingEngine

    engine = TradingEngine(config)
    return await engine.run(duration_sec=duration)


def cmd_run(config: AppConfig, args: argparse.Namespace) -> int:
    log = get_logger("cli")
    log.info("engine_starting", **_summarise(config))
    if config.mode is TradingMode.LIVE:
        log.critical(
            "live_trading_armed",
            message="Real orders will be placed. Kill switches are the only "
            "thing between a bug and your balance.",
        )
    try:
        return asyncio.run(_run_async(config, args.duration))
    except KeyboardInterrupt:
        log.info("engine_interrupted")
        return 0


def cmd_scan(config: AppConfig, args: argparse.Namespace) -> int:
    from tradebot.app.commands import run_scan_once

    return asyncio.run(run_scan_once(config, top=args.top or None))


def cmd_backtest(config: AppConfig, args: argparse.Namespace) -> int:
    from tradebot.app.commands import run_backtest

    return asyncio.run(run_backtest(config, args))


def cmd_walkforward(config: AppConfig, args: argparse.Namespace) -> int:
    from tradebot.app.commands import run_walkforward

    return asyncio.run(run_walkforward(config, args))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = _configure(args)
    except SafetyError as exc:
        print(f"SAFETY CHECK FAILED: {exc}", file=sys.stderr)
        return 78  # EX_CONFIG
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 78

    try:
        if args.command == "validate-config":
            return cmd_validate_config(config)
        if args.command == "doctor":
            return cmd_doctor(config)
        if args.command == "run":
            return cmd_run(config, args)
        if args.command == "scan":
            return cmd_scan(config, args)
        if args.command == "backtest":
            return cmd_backtest(config, args)
        if args.command == "walkforward":
            return cmd_walkforward(config, args)
    except TradeBotError as exc:
        get_logger("cli").critical("fatal", error=str(exc), error_type=type(exc).__name__)
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
