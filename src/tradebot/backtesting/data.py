"""Historical data loading.

Reads the Parquet/CSV files written by ``scripts/download_data.py`` into
:class:`BacktestData`. The loader is deliberately strict: silently accepting
malformed history produces a backtest whose result is meaningless in ways that
are very hard to spot afterwards.

It refuses, rather than repairs, data that is:

* **out of order** — indicators computed across shuffled bars are nonsense
* **duplicated** — the same bar counted twice inflates volume and distorts ATR
* **gapped beyond tolerance** — a missing hour silently becomes a price jump the
  strategies read as a real move

Small gaps are reported rather than refused, because exchanges do have brief
outages and requiring perfection would reject all real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tradebot.backtesting.engine import BacktestData, ExchangeFilterProvenance
from tradebot.core.errors import DataError
from tradebot.core.logging import get_logger
from tradebot.core.types import Candle, SymbolInfo, Timeframe
from tradebot.data.validation import QualityStatus

log = get_logger(__name__)


@dataclass(slots=True)
class DataQuality:
    """What was found while loading. Surfaced with every backtest.

    This is one of the two implementations of the
    :class:`~tradebot.backtesting.trust.DatasetQuality` contract — the other is
    :class:`~tradebot.data.validation.ValidationReport`, produced at
    acquisition time. Both report the **same** ``QualityStatus`` vocabulary, so
    the trust gate does not care which one it is handed.

    Before V3.2 this class had no ``status``, ``interval`` or ``missing_bars``,
    while the trust gate read exactly those three through ``getattr(..., None)``.
    The result was silent: every check that mattered evaluated to ``False`` and
    structurally corrupt data was reported ``TRUSTED``.
    """

    symbol: str
    timeframe: str
    bars: int
    duplicates_removed: int = 0
    gaps: int = 0
    largest_gap_bars: int = 0
    coverage: float = 1.0
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_bars: int = 0
    start_ms: int = 0
    end_ms: int = 0
    out_of_order: int = 0
    non_positive_prices: int = 0
    inconsistent_ohlc: int = 0

    @property
    def interval(self) -> str:
        """The contract's name for :attr:`timeframe`. Same string."""
        return self.timeframe

    @property
    def status(self) -> QualityStatus:
        """UNUSABLE beats DEGRADED beats OK.

        ``bars <= 0`` is UNUSABLE on its own: an empty series cannot be
        backtested, whether or not anything else was noticed about it.
        """
        if self.problems or self.bars <= 0:
            return QualityStatus.UNUSABLE
        if self.warnings or self.missing_bars or self.gaps or self.duplicates_removed:
            return QualityStatus.DEGRADED
        return QualityStatus.OK

    @property
    def usable(self) -> bool:
        return self.status is not QualityStatus.UNUSABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "bars": self.bars,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duplicates_removed": self.duplicates_removed,
            "missing_bars": self.missing_bars,
            "gaps": self.gaps,
            "largest_gap_bars": self.largest_gap_bars,
            "coverage": round(self.coverage, 6),
            "status": self.status.value,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
        }

    def row(self) -> dict[str, object]:
        """One line of the machine-readable quality artifact."""
        return {
            "SYMBOL": self.symbol,
            "INTERVAL": self.interval,
            "START": self.start_ms,
            "END": self.end_ms,
            "ROWS": self.bars,
            "MISSING": self.missing_bars,
            "DUPLICATES": self.duplicates_removed,
            "GAPS": self.gaps,
            "COVERAGE": round(self.coverage, 4),
            "QUALITY_STATUS": self.status.value,
        }


def load_candles(
    path: Path, timeframe: str, max_gap_bars: int = 10
) -> tuple[list[Candle], DataQuality]:
    """Load one symbol/timeframe file, validating as it goes."""
    import pandas as pd

    if not path.is_file():
        raise DataError("historical data file not found", path=str(path))

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    symbol = path.stem
    quality = DataQuality(symbol=symbol, timeframe=timeframe, bars=0)

    required = {"open_time", "open", "high", "low", "close", "volume", "close_time"}
    missing = required - set(frame.columns)
    if missing:
        raise DataError(f"data file is missing columns: {sorted(missing)}", path=str(path))

    before = len(frame)
    # Out-of-order rows are counted BEFORE sorting: sorting hides the fact that
    # the file arrived shuffled, which usually means the writer was broken.
    quality.out_of_order = int((frame["open_time"].diff().dropna() < 0).sum())
    frame = frame.drop_duplicates(subset="open_time").sort_values("open_time")
    quality.duplicates_removed = before - len(frame)

    if frame.empty:
        quality.problems.append("file contains no rows")
        return [], quality

    quality.start_ms = int(frame["open_time"].iloc[0])
    quality.end_ms = int(frame["close_time"].iloc[-1])

    # Any price at or below zero is not a price. Checked on all four legs:
    # a non-positive high or low is just as impossible as a non-positive close,
    # and before V3.2 neither was looked at.
    non_positive = frame[
        (frame["open"] <= 0) | (frame["high"] <= 0) | (frame["low"] <= 0) | (frame["close"] <= 0)
    ]
    if not non_positive.empty:
        quality.non_positive_prices = int(len(non_positive))
        quality.problems.append(f"{len(non_positive)} bars have a price at or below zero")

    # The high is the highest price of the bar and the low is the lowest, so
    # both open and close must lie between them. Checking only against close
    # let a bar whose OPEN sits outside its range pass.
    inconsistent = frame[
        (frame["high"] < frame["low"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
    ]
    if not inconsistent.empty:
        quality.inconsistent_ohlc = int(len(inconsistent))
        quality.problems.append(f"{len(inconsistent)} bars have impossible OHLC relationships")

    if "volume" in frame.columns:
        negative_volume = int((frame["volume"] < 0).sum())
        if negative_volume:
            quality.problems.append(f"{negative_volume} bars have negative volume")

    if quality.out_of_order:
        quality.problems.append(f"{quality.out_of_order} rows arrived out of chronological order")

    if quality.duplicates_removed:
        # Not fatal — the duplicate rows were dropped — but the file was wrong,
        # and a dataset that needed repairing is not a clean dataset.
        quality.warnings.append(f"{quality.duplicates_removed} duplicate timestamp(s) were dropped")

    try:
        interval_ms = Timeframe(timeframe).milliseconds
    except ValueError as exc:
        raise DataError(f"unsupported timeframe: {timeframe}") from exc

    deltas = frame["open_time"].diff().dropna()
    gap_rows = deltas[deltas > interval_ms]
    quality.gaps = int(len(gap_rows))
    if quality.gaps:
        largest = int(gap_rows.max() // interval_ms)
        quality.largest_gap_bars = largest
        if largest > max_gap_bars:
            quality.problems.append(
                f"a gap of {largest} bars is larger than the {max_gap_bars}-bar "
                f"tolerance; indicators computed across it would be wrong"
            )

    expected = int((frame["open_time"].iloc[-1] - frame["open_time"].iloc[0]) // interval_ms) + 1
    quality.coverage = len(frame) / max(1, expected)
    quality.missing_bars = max(0, expected - len(frame))

    candles = [
        Candle(
            open_time=int(row.open_time),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            close_time=int(row.close_time),
            quote_volume=float(getattr(row, "quote_volume", 0.0) or 0.0),
            trades=int(getattr(row, "trades", 0) or 0),
            taker_buy_volume=float(getattr(row, "taker_buy_volume", 0.0) or 0.0),
            closed=True,
        )
        for row in frame.itertuples(index=False)
    ]
    quality.bars = len(candles)
    return candles, quality


def load_dataset(
    directory: str | Path,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    symbol_infos: dict[str, SymbolInfo] | None = None,
    strict: bool = True,
) -> tuple[dict[str, BacktestData], list[DataQuality]]:
    """Load every symbol/timeframe under ``directory``.

    Layout produced by ``scripts/download_data.py``::

        data/klines/<timeframe>/<SYMBOL>.parquet
        data/funding/<SYMBOL>.parquet   (or .csv, legacy)
    """
    dataset_root = Path(directory)
    if not dataset_root.is_dir():
        raise DataError("data directory not found", path=str(dataset_root))

    # Two layouts are accepted, so `--data data` and `--data data/klines` both
    # work and neither silently finds nothing:
    #   <root>/klines/<interval>/<SYMBOL>.parquet   (DataStore, current)
    #   <root>/<interval>/<SYMBOL>.parquet          (legacy)
    root = dataset_root / "klines" if (dataset_root / "klines").is_dir() else dataset_root

    timeframes = timeframes or ["1m", "3m", "5m", "15m", "1h"]
    available = {p.name for p in root.iterdir() if p.is_dir()}
    usable_timeframes = [tf for tf in timeframes if tf in available]
    if not usable_timeframes:
        raise DataError(
            f"no timeframe directories found under {root}; expected one of "
            f"{timeframes}, found {sorted(available)}"
        )

    primary_dir = root / usable_timeframes[0]
    discovered = sorted(p.stem for p in primary_dir.iterdir() if p.suffix in {".parquet", ".csv"})
    chosen = symbols or discovered
    unknown = set(chosen) - set(discovered)
    if unknown:
        raise DataError(f"symbols not present in the dataset: {sorted(unknown)}")

    # Real exchange filters, if the acquisition pipeline stored them. Without
    # these the loader invents PERMISSIVE placeholders (tick 1e-8, min notional
    # 5.0), which lets a backtest take positions the exchange would reject —
    # most often the small ones a 75 USDT account depends on.
    resolved_infos = dict(symbol_infos or {})
    if not resolved_infos:
        from tradebot.data.store import DataStore

        stored = DataStore(dataset_root)
        resolved_infos = stored.load_exchange_info()
        if resolved_infos:
            log.info("exchange_info_loaded", symbols=len(resolved_infos))

    out: dict[str, BacktestData] = {}
    reports: list[DataQuality] = []

    for symbol in chosen:
        candles: dict[str, list[Candle]] = {}
        for timeframe in usable_timeframes:
            for suffix in (".parquet", ".csv"):
                path = root / timeframe / f"{symbol}{suffix}"
                if path.is_file():
                    bars, quality = load_candles(path, timeframe)
                    reports.append(quality)
                    if quality.problems:
                        message = f"{symbol} {timeframe}: " + "; ".join(quality.problems)
                        if strict:
                            raise DataError(message, path=str(path))
                        log.warning(
                            "data_quality_problem",
                            symbol=symbol,
                            timeframe=timeframe,
                            problems=quality.problems,
                        )
                    candles[timeframe] = bars
                    break

        if not candles:
            log.warning("symbol_has_no_data", symbol=symbol)
            continue

        stored_info = resolved_infos.get(symbol)
        info = stored_info or _default_symbol_info(symbol)
        out[symbol] = BacktestData(
            symbol=symbol,
            candles=candles,
            symbol_info=info,
            funding_rates=_load_funding(dataset_root, symbol),
            exchange_filter_provenance=(
                ExchangeFilterProvenance.GENUINE
                if stored_info is not None
                else ExchangeFilterProvenance.PLACEHOLDER
            ),
        )

    total_bars = sum(q.bars for q in reports)
    log.info(
        "dataset_loaded",
        symbols=len(out),
        timeframes=usable_timeframes,
        bars=total_bars,
        gaps=sum(q.gaps for q in reports),
        duplicates=sum(q.duplicates_removed for q in reports),
    )
    return out, reports


#: Column pairs each supported funding layout uses, newest first.
_FUNDING_COLUMNS = (
    ("funding_time", "funding_rate"),  # DataStore.write_funding (current)
    ("fundingTime", "fundingRate"),  # legacy CSV from the old download script
    ("calc_time", "last_funding_rate"),  # the bulk archive's own header
)


def _load_funding(root: Path, symbol: str) -> dict[int, float]:
    """Funding history keyed by the exchange's funding timestamp.

    Reads Parquet first, because that is what :class:`DataStore` writes; CSV is
    kept for datasets produced by the older download script.

    This function used to look only for ``<SYMBOL>.csv`` with ``fundingTime`` /
    ``fundingRate`` columns, while the store wrote ``<SYMBOL>.parquet`` with
    ``funding_time`` / ``funding_rate``. Nothing raised — the mismatch simply
    produced an empty mapping, so every backtest silently ran with **zero
    funding**, which flatters any position held across a funding timestamp.
    A missing file is legitimate; a file that exists and cannot be read is not,
    so the two are logged differently.
    """
    import pandas as pd

    candidates = [root / "funding" / f"{symbol}{ext}" for ext in (".parquet", ".csv")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}

    try:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except (OSError, ValueError) as exc:
        log.error(
            "funding_file_unreadable",
            path=str(path),
            error=str(exc)[:200],
            message="the backtest will run with NO funding costs",
        )
        return {}

    columns = set(frame.columns)
    pair = next((c for c in _FUNDING_COLUMNS if set(c) <= columns), None)
    if pair is None:
        log.error(
            "funding_file_has_no_recognised_columns",
            path=str(path),
            found=sorted(columns),
            expected=[list(c) for c in _FUNDING_COLUMNS],
            message="the backtest will run with NO funding costs",
        )
        return {}

    time_column, rate_column = pair
    out: dict[int, float] = {}
    for row in frame.itertuples(index=False):
        try:
            out[int(getattr(row, time_column))] = float(getattr(row, rate_column))
        except (TypeError, ValueError):
            continue
    log.info("funding_loaded", symbol=symbol, events=len(out), path=str(path))
    return out


def _default_symbol_info(symbol: str) -> SymbolInfo:
    """Placeholder filters when exchangeInfo was not saved alongside the data.

    Deliberately permissive: using tight guessed filters would reject trades the
    real exchange would have accepted, which is a subtler distortion than
    accepting a few it would have rejected. The operator should supply real
    filters for any result they intend to act on.
    """
    log.warning(
        "using_placeholder_symbol_filters",
        symbol=symbol,
        hint="supply real exchangeInfo filters for an accurate backtest",
    )
    return SymbolInfo(
        symbol=symbol,
        base_asset=symbol.replace("USDT", ""),
        quote_asset="USDT",
        status="TRADING",
        contract_type="PERPETUAL",
        price_precision=8,
        quantity_precision=8,
        tick_size=1e-8,
        step_size=1e-8,
        min_qty=1e-8,
        max_qty=1e9,
        min_notional=5.0,
        max_leverage=20,
    )
