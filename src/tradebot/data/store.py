"""Dataset storage: ingestion -> validation -> normalization -> Parquet.

The layout is deliberately boring, because a clever layout is one you cannot
re-derive in six months::

    <root>/
      klines/<interval>/<SYMBOL>.parquet          + .manifest.json
      funding/<SYMBOL>.parquet                    + .manifest.json
      symbols/exchange_info.json                  real tick/step/minNotional
      reports/data_quality.json                   the §5 quality report

One file per symbol/interval keeps a re-download of one symbol from rewriting
everything, and keeps a partial download from being indistinguishable from a
complete one.

**Normalization** here means exactly two things and no more: sort by open time,
and drop exact duplicate timestamps. Both are lossless. Anything further —
filling gaps, adjusting prices, synthesising bars — would be inventing data, and
this module will not do it. Bars that fail validation are reported, not
repaired.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tradebot.core.errors import DataError
from tradebot.core.logging import get_logger
from tradebot.core.types import Candle, LeverageBracket, SymbolInfo
from tradebot.data.manifest import (
    KLINE_COLUMNS,
    SCHEMA_VERSION,
    DatasetManifest,
    content_hash,
    read_manifest,
    utc_now_iso,
    write_manifest,
)
from tradebot.data.validation import ValidationReport, validate_candles

log = get_logger(__name__)


class DataStore:
    """Reads and writes the on-disk dataset tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    def klines_path(self, symbol: str, interval: str) -> Path:
        return self.root / "klines" / interval / f"{symbol}.parquet"

    def funding_path(self, symbol: str) -> Path:
        return self.root / "funding" / f"{symbol}.parquet"

    @property
    def exchange_info_path(self) -> Path:
        return self.root / "symbols" / "exchange_info.json"

    @property
    def quality_report_path(self) -> Path:
        return self.root / "reports" / "data_quality.json"

    def intervals(self) -> list[str]:
        base = self.root / "klines"
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def symbols(self, interval: str) -> list[str]:
        directory = self.root / "klines" / interval
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.parquet"))

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def write_klines(
        self,
        symbol: str,
        interval: str,
        candles: list[Candle],
        source: str,
        notes: str = "",
    ) -> tuple[Path, DatasetManifest, ValidationReport]:
        """Normalise, validate and write one symbol/interval.

        Returns the path, the manifest and the validation report. The data is
        written even when validation fails, so the operator can inspect what
        went wrong — but the manifest records it, and :meth:`load_klines`
        refuses it by default.
        """
        import pandas as pd

        normalised, transformations = normalise(candles)
        report = validate_candles(symbol, interval, normalised)

        rows = [
            (
                c.open_time,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.close_time,
                c.quote_volume,
                c.trades,
                c.taker_buy_volume,
            )
            for c in normalised
        ]
        frame = pd.DataFrame(rows, columns=list(KLINE_COLUMNS))

        path = self.klines_path(symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

        manifest = DatasetManifest(
            symbol=symbol,
            interval=interval,
            start_ms=normalised[0].open_time if normalised else 0,
            end_ms=normalised[-1].close_time if normalised else 0,
            rows=len(normalised),
            source=source,
            downloaded_at=utc_now_iso(),
            schema_version=SCHEMA_VERSION,
            content_hash=content_hash(rows),
            transformations=transformations,
            notes=notes or ("; ".join(report.errors) if report.errors else ""),
        )
        write_manifest(path, manifest)
        log.info("klines_written", path=str(path), status=report.status.value)
        return path, manifest, report

    def write_funding(
        self, symbol: str, rates: dict[int, float], source: str
    ) -> tuple[Path, DatasetManifest]:
        """Funding history, keyed by the exchange's own funding timestamp."""
        import pandas as pd

        ordered = sorted(rates.items())
        frame = pd.DataFrame(ordered, columns=["funding_time", "funding_rate"])

        path = self.funding_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

        manifest = DatasetManifest(
            symbol=symbol,
            interval="funding",
            start_ms=ordered[0][0] if ordered else 0,
            end_ms=ordered[-1][0] if ordered else 0,
            rows=len(ordered),
            source=source,
            downloaded_at=utc_now_iso(),
            content_hash=content_hash(ordered),
        )
        write_manifest(path, manifest)
        return path, manifest

    def write_exchange_info(self, infos: dict[str, SymbolInfo], source: str) -> Path:
        """Real tick size, step size, min qty and min notional.

        Without this the backtester invents permissive filters, which lets it
        take positions the exchange would have rejected outright — most often
        the small ones a 75 USDT account depends on.
        """
        path = self.exchange_info_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": source,
            "downloaded_at": utc_now_iso(),
            "schema_version": SCHEMA_VERSION,
            "symbols": {name: _symbol_to_dict(info) for name, info in sorted(infos.items())},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        log.info("exchange_info_written", path=str(path), symbols=len(infos))
        return path

    def write_quality_report(self, reports: list[ValidationReport]) -> Path:
        from tradebot.data.validation import quality_table, summarise

        path = self.quality_report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "generated_at": utc_now_iso(),
                    "summary": summarise(reports),
                    "table": quality_table(reports),
                    "detail": [r.as_dict() for r in reports],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return path

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def load_klines(
        self, symbol: str, interval: str, strict: bool = True
    ) -> tuple[list[Candle], DatasetManifest | None, ValidationReport]:
        """Read one symbol/interval back, re-validating as it goes.

        Re-validation on read is deliberate. The manifest records what was true
        at download time; a file can be edited, truncated or partially written
        afterwards, and trusting the manifest alone would let that through.
        """
        import pandas as pd

        path = self.klines_path(symbol, interval)
        if not path.is_file():
            raise DataError("dataset not found", path=str(path))

        frame = pd.read_parquet(path)
        missing = set(KLINE_COLUMNS[:7]) - set(frame.columns)
        if missing:
            raise DataError(f"dataset is missing columns: {sorted(missing)}", path=str(path))

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

        manifest = read_manifest(path)
        report = validate_candles(symbol, interval, candles)

        if manifest is not None and manifest.rows != len(candles):
            report.errors.append(f"manifest says {manifest.rows} rows, file has {len(candles)}")
        if strict and not report.usable:
            raise DataError(
                f"{symbol} {interval} failed validation: {'; '.join(report.errors)}",
                path=str(path),
            )
        return candles, manifest, report

    def load_funding(self, symbol: str) -> dict[int, float]:
        import pandas as pd

        path = self.funding_path(symbol)
        if not path.is_file():
            return {}
        frame = pd.read_parquet(path)
        if not {"funding_time", "funding_rate"} <= set(frame.columns):
            log.warning("funding_file_missing_columns", path=str(path))
            return {}
        return {
            int(row.funding_time): float(row.funding_rate) for row in frame.itertuples(index=False)
        }

    def load_exchange_info(self) -> dict[str, SymbolInfo]:
        """Real filters, or an empty mapping when they were never downloaded."""
        path = self.exchange_info_path
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text())
        return {name: _symbol_from_dict(raw) for name, raw in payload.get("symbols", {}).items()}

    def manifests(self) -> list[DatasetManifest]:
        """Every manifest under this root, for the run fingerprint.

        A manifest sits at ``<data file>.manifest.json``, so the data path is
        recovered by stripping that exact suffix — not by juggling
        ``with_suffix``, which for ``BTCUSDT.parquet.manifest.json`` produced a
        path that existed nowhere and silently returned no manifests at all.
        That left every run quoting the fingerprint of an empty set, which looks
        like a valid hash and identifies nothing.
        """
        found: list[DatasetManifest] = []
        for path in sorted(self.root.rglob("*.manifest.json")):
            data_path = Path(str(path).removesuffix(".manifest.json"))
            entry = read_manifest(data_path)
            if entry is not None:
                found.append(entry)
        return found

    def stats(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "intervals": self.intervals(),
            "symbols": {i: len(self.symbols(i)) for i in self.intervals()},
            "has_exchange_info": self.exchange_info_path.is_file(),
        }


def _symbol_to_dict(info: SymbolInfo) -> dict[str, Any]:
    """Every field, not a hand-picked subset.

    Serialising a subset is how `market_min_qty` — the minimum for MARKET
    orders, which is the order type this engine uses for entries — silently
    became 0.0 on the way back in, letting the backtest take positions the
    exchange would have rejected.
    """
    payload = asdict(info)
    payload["brackets"] = [asdict(b) for b in info.brackets]
    return payload


def _symbol_from_dict(raw: dict[str, Any]) -> SymbolInfo:
    fields = dict(raw)
    fields["brackets"] = tuple(
        LeverageBracket(**bracket) for bracket in fields.get("brackets", ()) or ()
    )
    known = set(SymbolInfo.__dataclass_fields__)
    unknown = set(fields) - known
    if unknown:
        log.warning("exchange_info_has_unknown_fields", fields=sorted(unknown))
        fields = {k: v for k, v in fields.items() if k in known}
    return SymbolInfo(**fields)


def normalise(candles: list[Candle]) -> tuple[list[Candle], list[str]]:
    """Sort by open time and drop exact duplicates. Nothing else.

    Both operations are lossless and reversible in meaning: sorting fixes an
    ordering the exchange never intended, and a duplicate timestamp carries no
    information the first copy did not. Filling gaps is NOT done here, and must
    not be: a synthesised bar is a fabricated price.
    """
    transformations: list[str] = []
    if not candles:
        return [], transformations

    ordered = sorted(candles, key=lambda c: c.open_time)
    if [c.open_time for c in ordered] != [c.open_time for c in candles]:
        transformations.append("sorted by open_time")

    deduped: list[Candle] = []
    seen: set[int] = set()
    for candle in ordered:
        if candle.open_time in seen:
            continue
        seen.add(candle.open_time)
        deduped.append(candle)
    removed = len(ordered) - len(deduped)
    if removed:
        transformations.append(f"dropped {removed} duplicate timestamps")

    return deduped, transformations
