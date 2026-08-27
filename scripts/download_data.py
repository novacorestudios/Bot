#!/usr/bin/env python3
"""Download historical klines for backtesting.

Run this on a host with access to Binance. The development sandbox used to write
this repository has none, so **no historical data is committed and no backtest
result exists yet**.

    python scripts/download_data.py --symbols BTCUSDT,ETHUSDT --timeframes 1m,5m \
        --start 2024-01-01 --end 2024-07-01 --out data/klines

    # or let it pick the most liquid symbols itself
    python scripts/download_data.py --top 40 --start 2024-01-01 --end 2024-07-01

Output is one Parquet (or CSV) file per symbol/timeframe. Funding history is
downloaded alongside, because a backtest that ignores funding overstates the
result on any position held across a funding timestamp.

Binance also publishes bulk archives at https://data.binance.vision which are
far faster for multi-year downloads; this script exists for convenience and for
topping up recent data.
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
from tradebot.core.types import Timeframe  # noqa: E402
from tradebot.exchange.binance.rest import BinanceFuturesREST  # noqa: E402

log = get_logger("download")

MAX_PER_REQUEST = 1500


def parse_date(text: str) -> int:
    """ISO date or datetime -> epoch milliseconds (UTC)."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            continue
    raise SystemExit(f"cannot parse date: {text!r} (use YYYY-MM-DD)")


async def fetch_range(client: BinanceFuturesREST, symbol: str, interval: str,
                      start_ms: int, end_ms: int) -> list:
    """Page through klines, respecting the 1500-per-request cap."""
    step = Timeframe(interval).milliseconds
    out: list = []
    cursor = start_ms
    while cursor < end_ms:
        batch = await client.get_klines(symbol, interval, limit=MAX_PER_REQUEST,
                                        start_ms=cursor, end_ms=end_ms)
        if not batch:
            break
        # Only keep fully closed bars; a forming bar would poison the backtest.
        batch = [c for c in batch if c.closed and c.open_time < end_ms]
        if not batch:
            break
        out.extend(batch)
        advanced = batch[-1].open_time + step
        if advanced <= cursor:
            break                       # no progress; avoid an infinite loop
        cursor = advanced
        if len(out) % 15000 < MAX_PER_REQUEST:
            log.info("downloading", symbol=symbol, interval=interval, bars=len(out))
    # De-duplicate defensively: overlapping pages are possible at boundaries.
    seen: dict[int, object] = {}
    for candle in out:
        seen[candle.open_time] = candle
    return [seen[k] for k in sorted(seen)]


def write_frame(candles: list, path: Path, fmt: str) -> int:
    import pandas as pd

    if not candles:
        return 0
    frame = pd.DataFrame(
        {
            "open_time": [c.open_time for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
            "close_time": [c.close_time for c in candles],
            "quote_volume": [c.quote_volume for c in candles],
            "trades": [c.trades for c in candles],
            "taker_buy_volume": [c.taker_buy_volume for c in candles],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            frame.to_parquet(path, index=False)
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
            frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return len(frame)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="",
                        help="Comma-separated symbols. Omit to use --top.")
    parser.add_argument("--top", type=int, default=0,
                        help="Instead of --symbols, take the N highest 24h-volume "
                             "USDT perpetuals.")
    parser.add_argument("--timeframes", default="1m,3m,5m,15m,1h")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", default="", help="YYYY-MM-DD (UTC, exclusive). "
                                                  "Defaults to now.")
    parser.add_argument("--out", default="data/klines")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--funding", action="store_true", default=True,
                        help="Also download funding-rate history (default on)")
    args = parser.parse_args()

    configure_logging("INFO", "console", None)
    settings = Settings()

    start_ms = parse_date(args.start)
    end_ms = parse_date(args.end) if args.end else int(
        datetime.now(tz=UTC).timestamp() * 1000
    )
    if end_ms <= start_ms:
        raise SystemExit("--end must be after --start")

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    for tf in timeframes:
        try:
            Timeframe(tf)
        except ValueError:
            raise SystemExit(f"unsupported timeframe: {tf}") from None

    client = BinanceFuturesREST(
        api_key=settings.binance_api_key, api_secret=settings.binance_api_secret,
        base_url=settings.rest_url, recv_window=settings.binance_recv_window,
    )
    await client.connect()

    try:
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            unknown = [s for s in symbols if s not in client.symbols]
            if unknown:
                raise SystemExit(f"unknown symbols: {unknown}")
        elif args.top:
            tickers = await client.get_ticker_24h()
            eligible = [
                (t.quote_volume, t.symbol) for t in tickers.values()
                if (info := client.symbol_info(t.symbol)) is not None
                and info.is_tradable and info.quote_asset == "USDT"
            ]
            eligible.sort(reverse=True)
            symbols = [symbol for _, symbol in eligible[: args.top]]
            log.info("selected_by_volume", count=len(symbols),
                     symbols=symbols[:10])
        else:
            raise SystemExit("pass either --symbols or --top")

        out_dir = Path(args.out)
        total_bars = 0
        for symbol in symbols:
            for interval in timeframes:
                candles = await fetch_range(client, symbol, interval, start_ms, end_ms)
                suffix = "parquet" if args.format == "parquet" else "csv"
                path = out_dir / interval / f"{symbol}.{suffix}"
                written = write_frame(candles, path, args.format)
                total_bars += written
                log.info("written", symbol=symbol, interval=interval,
                         bars=written, path=str(path))

            if args.funding:
                try:
                    history = await client.get_funding_history(symbol, limit=1000)
                    if history:
                        import pandas as pd

                        funding_path = out_dir / "funding" / f"{symbol}.csv"
                        funding_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(history).to_csv(funding_path, index=False)
                except Exception as exc:  # noqa: BLE001
                    log.warning("funding_download_failed", symbol=symbol, error=str(exc))

        print(f"\ndownloaded {total_bars} bars for {len(symbols)} symbols into {out_dir}")
        print("Next: run a backtest, then out-of-sample and walk-forward validation.")
        print("A single in-sample backtest proves nothing (IMPLEMENTATION_PLAN.md §9).")
    finally:
        await client.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
