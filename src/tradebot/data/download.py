"""The acquisition pipeline, as one callable flow.

    ingest -> normalise -> validate -> store (+ manifest) -> quality report

Written as a library rather than only a script so the tests can drive it with a
fake source, which is how it is exercised in an environment with no route to
Binance.

The ordering matters: data is validated *before* it is announced as usable, and
the quality report is written whether or not everything passed. A pipeline that
only reports on success tells you nothing on the day it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tradebot.core.logging import get_logger
from tradebot.data.manifest import DatasetManifest, dataset_fingerprint
from tradebot.data.store import DataStore
from tradebot.data.validation import QualityStatus, ValidationReport, summarise

log = get_logger(__name__)


@dataclass(slots=True)
class DownloadResult:
    """What one acquisition run produced."""

    manifests: list[DatasetManifest] = field(default_factory=list)
    reports: list[ValidationReport] = field(default_factory=list)
    funding_symbols: list[str] = field(default_factory=list)
    exchange_info_symbols: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return dataset_fingerprint(self.manifests)

    @property
    def unusable(self) -> list[ValidationReport]:
        return [r for r in self.reports if r.status is QualityStatus.UNUSABLE]

    def summary(self) -> dict[str, Any]:
        return {
            "datasets": len(self.manifests),
            "fingerprint": self.fingerprint[:16],
            "funding_symbols": len(self.funding_symbols),
            "exchange_info_symbols": self.exchange_info_symbols,
            "failures": len(self.failures),
            **summarise(self.reports),
        }

    def describe(self) -> str:
        lines = ["=" * 68, "DATA ACQUISITION", "=" * 68]
        for key, value in self.summary().items():
            if key == "symbols":
                value = f"{len(value)} symbols"
            lines.append(f"  {key:<24} {value}")
        if self.unusable:
            lines += ["", "UNUSABLE datasets (do NOT backtest on these):"]
            lines += [f"  {r.symbol} {r.interval}: {'; '.join(r.errors)}" for r in self.unusable]
        if self.failures:
            lines += ["", "Downloads that failed:"]
            lines += [f"  {k}: {v}" for k, v in sorted(self.failures.items())]
        return "\n".join(lines)


async def download_klines(
    source: Any,
    store: DataStore,
    symbols: list[str],
    intervals: list[str],
    start_ms: int,
    end_ms: int,
) -> DownloadResult:
    """Fetch, validate and store every symbol/interval combination.

    One symbol failing does not abort the run: a partial dataset an operator can
    inspect is more useful than an exception halfway through a multi-hour
    download.
    """
    result = DownloadResult()

    for symbol in symbols:
        for interval in intervals:
            key = f"{symbol}:{interval}"
            try:
                candles = await source.fetch_klines(symbol, interval, start_ms, end_ms)
            except Exception as exc:  # noqa: BLE001 - one symbol must not kill the run
                result.failures[key] = f"{type(exc).__name__}: {exc}"
                log.error("download_failed", symbol=symbol, interval=interval, error=str(exc))
                continue

            if not candles:
                result.failures[key] = "no data returned for the requested range"
                continue

            _, manifest, report = store.write_klines(
                symbol, interval, candles, source=getattr(source, "name", "unknown")
            )
            result.manifests.append(manifest)
            result.reports.append(report)

    store.write_quality_report(result.reports)
    return result


async def download_funding(
    source: Any, store: DataStore, symbols: list[str], start_ms: int, end_ms: int
) -> list[str]:
    """Funding history. Absent funding overstates every held position."""
    stored: list[str] = []
    for symbol in symbols:
        fetch = getattr(source, "fetch_funding", None)
        if fetch is None:
            break
        try:
            rates = await fetch(symbol, start_ms, end_ms)
        except Exception as exc:  # noqa: BLE001
            log.warning("funding_download_failed", symbol=symbol, error=str(exc))
            continue
        if rates:
            store.write_funding(symbol, rates, source=getattr(source, "name", "unknown"))
            stored.append(symbol)
    return stored


async def download_exchange_info(source: Any, store: DataStore) -> int:
    """Real trading rules. Without these the backtester guesses (B-15)."""
    fetch = getattr(source, "fetch_exchange_info", None)
    if fetch is None:
        return 0
    try:
        infos = await fetch()
    except Exception as exc:  # noqa: BLE001
        log.warning("exchange_info_download_failed", error=str(exc))
        return 0
    if not infos:
        return 0
    store.write_exchange_info(infos, source=getattr(source, "name", "unknown"))
    return len(infos)


async def acquire(
    klines_source: Any,
    store: DataStore | str | Path,
    symbols: list[str],
    intervals: list[str],
    start_ms: int,
    end_ms: int,
    metadata_source: Any = None,
) -> DownloadResult:
    """The whole pipeline, end to end.

    ``metadata_source`` is separate because ``exchangeInfo`` only exists on the
    REST API — the bulk archive does not carry it — so a run using the archive
    for bars still needs a REST client for the trading rules.
    """
    resolved = store if isinstance(store, DataStore) else DataStore(store)

    result = await download_klines(klines_source, resolved, symbols, intervals, start_ms, end_ms)
    result.funding_symbols = await download_funding(
        klines_source, resolved, symbols, start_ms, end_ms
    )
    result.exchange_info_symbols = await download_exchange_info(
        metadata_source or klines_source, resolved
    )

    log.info(
        "acquisition_complete", **{k: v for k, v in result.summary().items() if k != "symbols"}
    )
    return result
