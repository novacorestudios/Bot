#!/usr/bin/env python3
"""Download real Binance USDⓈ-M Futures history for backtesting.

Run this on a host with a route to Binance. The sandbox this repository was
written in has none, so **no historical data is committed and no backtest result
on real data exists** — see BACKTEST_REPORT.md.

    # A year of 5m bars for the 40 most liquid perpetuals, from the bulk archive
    python scripts/fetch_data.py --top 40 --intervals 5m,15m \
        --start 2024-01-01 --end 2025-01-01

    # Specific symbols, and the REST API instead of the archive
    python scripts/fetch_data.py --symbols BTCUSDT,ETHUSDT --source rest \
        --start 2024-06-01 --end 2024-07-01

The bulk archive (data.binance.vision) is the default and needs no credentials.
It is roughly 30x fewer requests than REST for a long range, but lags real time
by about a day; use `--source rest` to top up the tail.

`exchangeInfo` — tick size, step size, minimum quantity and minimum notional —
is fetched over REST regardless of `--source`, because the archive does not
carry it and the backtester otherwise guesses permissive filters.

Everything lands under --out as Parquet with a manifest per dataset, plus a
data-quality report. Read the report before trusting a backtest built on it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.core.config import Settings  # noqa: E402
from tradebot.core.logging import configure_logging, get_logger  # noqa: E402
from tradebot.data.download import acquire  # noqa: E402
from tradebot.data.sources import RestSource, VisionSource  # noqa: E402
from tradebot.data.store import DataStore  # noqa: E402

log = get_logger("fetch")


def parse_date(text: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            continue
    raise SystemExit(f"cannot parse date {text!r} — use YYYY-MM-DD")


async def pick_symbols(rest: object, top: int, quote: str) -> list[str]:
    """The most liquid perpetuals by 24h quote volume, right now.

    NOTE: this is a *present-day* liquidity ranking applied to a *historical*
    range, which is survivorship bias — symbols that were liquid in the period
    but have since been delisted will not appear. It is documented rather than
    hidden; see DATA_PIPELINE.md. Pass --symbols explicitly to avoid it.
    """
    tickers = await rest.get_ticker_24h()  # type: ignore[attr-defined]
    symbols = await rest.load_symbols()  # type: ignore[attr-defined]
    ranked = sorted(
        (
            t
            for name, t in tickers.items()
            if name.endswith(quote) and name in symbols and symbols[name].is_tradable
        ),
        key=lambda t: t.quote_volume,
        reverse=True,
    )
    return [t.symbol for t in ranked[:top]]


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbols", default="", help="Comma-separated. Omit to use --top.")
    parser.add_argument("--top", type=int, default=0, help="Most liquid N perpetuals.")
    parser.add_argument("--intervals", default="5m,15m")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD, UTC, inclusive")
    parser.add_argument("--end", default="", help="YYYY-MM-DD, UTC, exclusive. Default: now.")
    parser.add_argument("--out", default="data", help="Dataset root directory")
    parser.add_argument("--source", choices=["archive", "rest"], default="archive")
    parser.add_argument("--quote", default="USDT")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level, "console", None)

    start_ms = parse_date(args.start)
    end_ms = parse_date(args.end) if args.end else int(datetime.now(tz=UTC).timestamp() * 1000)
    if end_ms <= start_ms:
        raise SystemExit("--end must be after --start")

    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    settings = Settings()
    from tradebot.exchange.binance.rest import BinanceFuturesREST

    rest = BinanceFuturesREST(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        base_url=settings.rest_url,
    )
    try:
        await rest.connect()
    except Exception as exc:  # noqa: BLE001 - report the reason, do not stack-trace it
        await rest.close()
        print(f"Cannot reach Binance at {settings.rest_url}:\n  {exc}", file=sys.stderr)
        print(
            "\nThis script needs a host with a route to Binance. Nothing was "
            "downloaded and nothing was written.",
            file=sys.stderr,
        )
        return 2

    try:
        if not symbols:
            if not args.top:
                raise SystemExit("pass --symbols or --top")
            symbols = await pick_symbols(rest, args.top, args.quote)
            log.warning(
                "symbols_chosen_by_present_day_liquidity",
                count=len(symbols),
                message="this is survivorship bias; see DATA_PIPELINE.md",
            )

        klines_source = VisionSource() if args.source == "archive" else RestSource(rest)
        result = await acquire(
            klines_source,
            DataStore(args.out),
            symbols,
            intervals,
            start_ms,
            end_ms,
            metadata_source=RestSource(rest),
        )
    finally:
        await rest.close()

    print(result.describe())
    if result.unusable:
        print("\nSome datasets are UNUSABLE. Fix or exclude them before backtesting.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
