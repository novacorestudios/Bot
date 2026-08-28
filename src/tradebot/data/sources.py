"""Where historical data comes from.

Two sources, because they answer different questions.

**The bulk archive** (``data.binance.vision``) publishes daily and monthly ZIPs
of klines and funding, needs no credentials, and is the only practical way to
obtain the multi-year 1-minute history a scalping study needs. Pulling a year of
1m bars over REST is roughly 350 requests per symbol; over the archive it is 12
files. This is the primary source.

**The REST API** is for topping up the days the archive has not published yet —
it lags real time by about a day — and for ``exchangeInfo``, which the archive
does not carry at all.

Both produce the same `Candle` objects and both write through `DataStore`, so
the backtester cannot tell them apart and nothing downstream has to care.

### On what is NOT here

Historical **order-book** snapshots and **bookTicker** history are not fetched.
Binance does publish some of it, but at a volume (tens of GB per symbol-month)
and a completeness that does not justify the claim it would let us make. The
spread model in `execution.py` is parametric and says so. Pretending to have
historical spreads we do not have would be the single most flattering lie a
backtest of a scalping system could tell, since spread is most of the cost.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from tradebot.core.errors import DataError
from tradebot.core.logging import get_logger
from tradebot.core.types import Candle

log = get_logger(__name__)

VISION_BASE = "https://data.binance.vision/data/futures/um"

#: The archive's kline CSV column order, which has no header row.
#: https://github.com/binance/binance-public-data
VISION_KLINE_FIELDS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


class CandleSource(Protocol):
    """Anything that can produce historical bars for a symbol."""

    name: str

    async def fetch_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Candle]: ...


# --------------------------------------------------------------------------- #
# Bulk archive
# --------------------------------------------------------------------------- #
def vision_kline_url(symbol: str, interval: str, day: date, monthly: bool = False) -> str:
    """The archive URL for one symbol/interval/period.

    Monthly files exist for months that have completed and are far cheaper;
    daily files cover the recent tail.
    """
    if monthly:
        stem = f"{symbol}-{interval}-{day:%Y-%m}"
        return f"{VISION_BASE}/monthly/klines/{symbol}/{interval}/{stem}.zip"
    stem = f"{symbol}-{interval}-{day:%Y-%m-%d}"
    return f"{VISION_BASE}/daily/klines/{symbol}/{interval}/{stem}.zip"


def vision_funding_url(symbol: str, day: date) -> str:
    stem = f"{symbol}-fundingRate-{day:%Y-%m}"
    return f"{VISION_BASE}/monthly/fundingRate/{symbol}/{stem}.zip"


def parse_vision_klines(payload: bytes, symbol: str) -> list[Candle]:
    """Parse one archive ZIP into candles.

    The archive occasionally ships a header row and occasionally does not, and
    has shipped timestamps in microseconds since 2025. Both are handled by
    inspection rather than by assumption, because a silently misparsed timestamp
    shifts every bar and produces a backtest that looks fine.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DataError(f"{symbol}: archive is not a valid zip ({exc})") from exc

    names = archive.namelist()
    if not names:
        raise DataError(f"{symbol}: archive is empty")

    text = archive.read(names[0]).decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []

    # Drop a header row if present: its first cell will not parse as a number.
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        rows = rows[1:]

    return [candle for candle in (_vision_row(row) for row in rows) if candle is not None]


def _vision_row(row: list[str]) -> Candle | None:
    if len(row) < 7:
        return None
    try:
        open_time = _to_ms(int(float(row[0])))
        close_time = _to_ms(int(float(row[6])))
        return Candle(
            open_time=open_time,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=close_time,
            quote_volume=float(row[7]) if len(row) > 7 else 0.0,
            trades=int(float(row[8])) if len(row) > 8 else 0,
            taker_buy_volume=float(row[9]) if len(row) > 9 else 0.0,
            closed=True,
        )
    except (ValueError, IndexError) as exc:
        log.warning("vision_row_unparseable", error=str(exc), row=row[:3])
        return None


def _to_ms(value: int) -> int:
    """Normalise a timestamp to milliseconds.

    Binance switched the archive to MICROsecond timestamps partway through 2025.
    A microsecond value is ~1000x larger, and the boundary is unambiguous for any
    date this century, so the magnitude decides it. Guessing wrong here shifts
    every bar by three orders of magnitude, which is why this is explicit.
    """
    # 1e14 ms is the year 5138; 1e14 us is 1973. Anything above is microseconds.
    return value // 1000 if value > 100_000_000_000_000 else value


def parse_vision_funding(payload: bytes, symbol: str) -> dict[int, float]:
    """Parse a monthly fundingRate archive into {funding_time_ms: rate}."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DataError(f"{symbol}: funding archive is not a valid zip ({exc})") from exc

    names = archive.namelist()
    if not names:
        return {}

    text = archive.read(names[0]).decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return {}

    header = [cell.strip().lower() for cell in rows[0]]
    if "calc_time" in header or "fundingtime" in header or "funding_time" in header:
        time_index = next(
            i for i, h in enumerate(header) if h in {"calc_time", "fundingtime", "funding_time"}
        )
        rate_index = next((i for i, h in enumerate(header) if "rate" in h), len(header) - 1)
        rows = rows[1:]
    else:
        time_index, rate_index = 0, 1

    out: dict[int, float] = {}
    for row in rows:
        if len(row) <= max(time_index, rate_index):
            continue
        try:
            out[_to_ms(int(float(row[time_index])))] = float(row[rate_index])
        except ValueError:
            continue
    return out


def days_between(start_ms: int, end_ms: int) -> Iterator[date]:
    """Every UTC day the requested range touches."""
    current = datetime.fromtimestamp(start_ms / 1000, tz=UTC).date()
    last = datetime.fromtimestamp(max(end_ms - 1, start_ms) / 1000, tz=UTC).date()
    while current <= last:
        yield current
        current += timedelta(days=1)


def months_between(start_ms: int, end_ms: int) -> Iterator[date]:
    """The first day of every month the range touches."""
    current = datetime.fromtimestamp(start_ms / 1000, tz=UTC).date().replace(day=1)
    last = datetime.fromtimestamp(max(end_ms - 1, start_ms) / 1000, tz=UTC).date().replace(day=1)
    while current <= last:
        yield current
        current = (current + timedelta(days=32)).replace(day=1)


class VisionSource:
    """Bulk archive downloader. No credentials, no rate limit worth modelling."""

    name = "data.binance.vision"

    def __init__(self, session_factory: Any = None, timeout_sec: float = 120.0) -> None:
        self._session_factory = session_factory
        self.timeout_sec = timeout_sec
        self.downloaded = 0
        self.missing: list[str] = []

    async def _get(self, url: str) -> bytes | None:
        """Fetch one archive. A 404 means "not published", not an error.

        The archive genuinely does not have every day for every symbol — a
        symbol listed on the 5th has nothing for the 4th — so a missing file is
        expected traffic and is recorded rather than raised.
        """
        import aiohttp

        factory = self._session_factory or aiohttp.ClientSession
        async with (
            factory() as session,
            session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout_sec)) as response,
        ):
            if response.status == 404:
                self.missing.append(url)
                return None
            if response.status != 200:
                raise DataError(f"archive request failed with HTTP {response.status}", path=url)
            self.downloaded += 1
            return await response.read()

    async def fetch_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Candle]:
        """Monthly files where possible, daily for the remainder."""
        collected: dict[int, Candle] = {}

        for month in months_between(start_ms, end_ms):
            payload = await self._get(vision_kline_url(symbol, interval, month, monthly=True))
            if payload is None:
                continue
            for candle in parse_vision_klines(payload, symbol):
                collected[candle.open_time] = candle

        # Days the monthly files did not cover (the current month, mostly).
        for day in days_between(start_ms, end_ms):
            if any(c.open_time // 86_400_000 == _day_index(day) for c in collected.values()):
                continue
            payload = await self._get(vision_kline_url(symbol, interval, day))
            if payload is None:
                continue
            for candle in parse_vision_klines(payload, symbol):
                collected[candle.open_time] = candle

        return sorted(
            (c for c in collected.values() if start_ms <= c.open_time < end_ms),
            key=lambda c: c.open_time,
        )

    async def fetch_funding(self, symbol: str, start_ms: int, end_ms: int) -> dict[int, float]:
        out: dict[int, float] = {}
        for month in months_between(start_ms, end_ms):
            payload = await self._get(vision_funding_url(symbol, month))
            if payload is None:
                continue
            out.update(parse_vision_funding(payload, symbol))
        return {t: r for t, r in out.items() if start_ms <= t < end_ms}


def _day_index(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) // 86_400


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #
class RestSource:
    """Paged REST downloader, for topping up the archive's tail.

    The archive lags real time by roughly a day; this fills that in, and is the
    only way to get `exchangeInfo`.
    """

    name = "fapi.binance.com"

    def __init__(self, client: Any, page_limit: int = 1500) -> None:
        self.client = client
        self.page_limit = page_limit

    async def fetch_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[Candle]:
        from tradebot.core.types import Timeframe

        step = Timeframe(interval).seconds * 1000
        collected: dict[int, Candle] = {}
        cursor = start_ms

        while cursor < end_ms:
            page = await self.client.get_klines(
                symbol, interval, limit=self.page_limit, start_ms=cursor, end_ms=end_ms
            )
            if not page:
                break
            for candle in page:
                if start_ms <= candle.open_time < end_ms:
                    collected[candle.open_time] = candle
            advanced = page[-1].open_time + step
            if advanced <= cursor:
                break  # the exchange is not advancing; stop rather than spin
            cursor = advanced

        return sorted(collected.values(), key=lambda c: c.open_time)

    async def fetch_exchange_info(self) -> dict[str, Any]:
        """Real tick size, step size, min qty and min notional."""
        return await self.client.load_symbols()
