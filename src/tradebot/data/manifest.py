"""Dataset manifests.

A backtest result is worth exactly as much as your ability to say what data
produced it. A number in a report with no record of its inputs cannot be
reproduced, cannot be audited, and cannot be compared against a later run — so
it is not evidence, it is an anecdote.

Every dataset written by this package therefore carries a manifest recording
what the brief's §4 requires: symbol, interval, the range actually covered, the
source it came from, when it was downloaded, and the schema version.

The **content hash** is the part that matters most. It is computed over the bar
data itself, not the file bytes, so it is stable across Parquet versions,
compression settings and column ordering. Two runs quoting the same dataset hash
saw the same bars; two quoting different hashes did not, however similar the
file sizes look.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradebot.core.errors import DataError
from tradebot.core.logging import get_logger

log = get_logger(__name__)

#: Bumped whenever the on-disk column set changes in a way that would make an
#: older file load incorrectly rather than merely incompletely.
SCHEMA_VERSION = 1

#: Columns every kline dataset carries, in this order.
KLINE_COLUMNS = (
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
)


@dataclass(slots=True)
class DatasetManifest:
    """What one symbol/interval file contains and where it came from."""

    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    rows: int
    source: str
    downloaded_at: str
    schema_version: int = SCHEMA_VERSION
    content_hash: str = ""
    #: Set when the rows were repaired or filtered on the way in, so a reader
    #: knows the file is not a byte-for-byte copy of what the exchange served.
    transformations: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def start_iso(self) -> str:
        return _iso(self.start_ms)

    @property
    def end_iso(self) -> str:
        return _iso(self.end_ms)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"{self.symbol} {self.interval}: {self.rows} rows "
            f"{self.start_iso} -> {self.end_iso} "
            f"[{self.source}, schema v{self.schema_version}, {self.content_hash[:12]}]"
        )


def _iso(ms: int) -> str:
    if ms <= 0:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def content_hash(rows: list[tuple[Any, ...]]) -> str:
    """A stable digest of the bar data itself.

    Deliberately hashes VALUES, not file bytes: the same bars written by two
    Parquet versions must produce the same hash, or the hash answers "did the
    library change?" instead of "did the data change?".

    Floats are rendered with `repr`, which round-trips exactly in Python, so two
    identical datasets cannot disagree because of formatting.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update("|".join(repr(value) for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_path(data_path: Path) -> Path:
    """Manifests sit beside their data file, one per file."""
    return data_path.with_suffix(data_path.suffix + ".manifest.json")


def write_manifest(data_path: Path, manifest: DatasetManifest) -> Path:
    path = manifest_path(data_path)
    path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(data_path: Path) -> DatasetManifest | None:
    """Load a manifest, or None when the file predates manifests."""
    path = manifest_path(data_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"manifest is unreadable: {exc}", path=str(path)) from exc

    known = set(DatasetManifest.__dataclass_fields__)
    unknown = set(payload) - known
    if unknown:
        # Forward compatibility: a newer writer may have added fields. Warn and
        # drop them rather than refusing to read data that is probably fine.
        log.warning("manifest_has_unknown_fields", path=str(path), fields=sorted(unknown))
        payload = {k: v for k, v in payload.items() if k in known}
    return DatasetManifest(**payload)


def dataset_fingerprint(manifests: list[DatasetManifest]) -> str:
    """One hash covering every dataset a run consumed.

    This is what a backtest report quotes. Order-independent, so the same
    datasets loaded in a different order still fingerprint identically.
    """
    digest = hashlib.sha256()
    for entry in sorted(manifests, key=lambda m: (m.symbol, m.interval)):
        digest.update(f"{entry.symbol}:{entry.interval}:{entry.content_hash}\n".encode())
    return digest.hexdigest()
