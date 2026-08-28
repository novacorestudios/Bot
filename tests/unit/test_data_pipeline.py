"""The historical data pipeline: ingest -> normalise -> validate -> store.

The property under test throughout is that **bad data is refused or reported,
never repaired**. A backtest cannot tell you its input was wrong; it will
happily compute a Sharpe ratio from a series with a six-hour hole in it, and the
hole reads as a price jump the strategies interpret as a real move.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from tradebot.core.errors import DataError
from tradebot.core.types import Candle, SymbolInfo
from tradebot.data.manifest import (
    DatasetManifest,
    content_hash,
    dataset_fingerprint,
    read_manifest,
    write_manifest,
)
from tradebot.data.sources import (
    days_between,
    months_between,
    parse_vision_funding,
    parse_vision_klines,
    vision_kline_url,
)
from tradebot.data.store import DataStore, normalise
from tradebot.data.validation import QualityStatus, validate_candles

START = 1_704_067_200_000  # 2024-01-01T00:00:00Z
STEP = 300_000  # 5m


def bar(index: int, *, price: float = 100.0, volume: float = 10.0, step: int = STEP) -> Candle:
    open_time = START + index * step
    return Candle(
        open_time=open_time,
        open=price,
        high=price * 1.005,
        low=price * 0.995,
        close=price * 1.001,
        volume=volume,
        close_time=open_time + step - 1,
        quote_volume=volume * price,
        trades=25,
    )


def clean(n: int = 50) -> list[Candle]:
    return [bar(i) for i in range(n)]


# --------------------------------------------------------------------------- #
class TestValidation:
    def test_clean_data_passes(self) -> None:
        report = validate_candles("BTCUSDT", "5m", clean())
        assert report.status is QualityStatus.OK
        assert report.coverage == pytest.approx(1.0, abs=0.02)
        assert report.usable

    def test_no_rows_is_unusable(self) -> None:
        assert validate_candles("BTCUSDT", "5m", []).status is QualityStatus.UNUSABLE

    @pytest.mark.parametrize(
        ("name", "candles"),
        [
            ("duplicate", [*clean(10), bar(5)]),
            ("out of order", [*clean(5), bar(9), *[bar(i) for i in range(5, 9)]]),
            ("non-positive price", [*clean(5), bar(5, price=-1.0), *clean(10)[6:]]),
            ("negative volume", [*clean(5), bar(5, volume=-3.0), *clean(10)[6:]]),
        ],
    )
    def test_corruption_is_unusable(self, name: str, candles: list[Candle]) -> None:
        """These corrupt indicators silently and cannot be repaired without
        inventing data, so they fail the dataset rather than warn."""
        report = validate_candles("BTCUSDT", "5m", candles)
        assert report.status is QualityStatus.UNUSABLE, name
        assert report.errors

    def test_inconsistent_ohlc_is_unusable(self) -> None:
        """A bar whose high is not the highest is a transcription error."""
        broken = Candle(START, 100.0, 90.0, 110.0, 100.0, 5.0, START + STEP - 1)
        report = validate_candles("BTCUSDT", "5m", [broken])
        assert report.inconsistent_ohlc == 1
        assert report.status is QualityStatus.UNUSABLE

    def test_a_gap_degrades_rather_than_fails(self) -> None:
        """Exchanges have outages. Requiring perfection rejects all real data."""
        gapped = [*clean(10), *[bar(i) for i in range(20, 30)]]
        report = validate_candles("BTCUSDT", "5m", gapped)
        assert report.status is QualityStatus.DEGRADED
        assert report.usable is True
        assert report.missing_bars == 10
        assert len(report.gaps) == 1
        assert report.gaps[0].missing_bars == 10

    def test_gaps_are_measured_not_filled(self) -> None:
        gapped = [*clean(10), *[bar(i) for i in range(20, 30)]]
        report = validate_candles("BTCUSDT", "5m", gapped)
        assert report.rows == 20, "validation must not synthesise the missing bars"

    def test_coverage_reflects_the_hole(self) -> None:
        gapped = [*clean(10), *[bar(i) for i in range(20, 30)]]
        report = validate_candles("BTCUSDT", "5m", gapped)
        assert report.coverage < 0.8

    def test_a_dead_symbol_is_flagged(self) -> None:
        """Zero-volume bars score as tradable and are not."""
        dead = [bar(i, volume=0.0) for i in range(50)]
        report = validate_candles("DEADUSDT", "5m", dead)
        assert any("zero volume" in w for w in report.warnings)

    def test_the_quality_row_has_the_columns_the_brief_asks_for(self) -> None:
        row = validate_candles("BTCUSDT", "5m", clean()).row()
        assert set(row) == {
            "SYMBOL",
            "INTERVAL",
            "DATA_START",
            "DATA_END",
            "ROWS",
            "MISSING",
            "DUPLICATES",
            "GAPS",
            "COVERAGE",
            "QUALITY_STATUS",
        }


class TestNormalisation:
    def test_out_of_order_input_is_sorted(self) -> None:
        shuffled = [bar(3), bar(1), bar(2)]
        out, transformations = normalise(shuffled)
        assert [c.open_time for c in out] == sorted(c.open_time for c in shuffled)
        assert any("sorted" in t for t in transformations)

    def test_duplicates_are_dropped_and_recorded(self) -> None:
        out, transformations = normalise([bar(1), bar(1), bar(2)])
        assert len(out) == 2
        assert any("duplicate" in t for t in transformations)

    def test_normalisation_never_invents_a_bar(self) -> None:
        """The line this module will not cross: a synthesised bar is a
        fabricated price, and a backtest trading on it carries false authority."""
        gapped = [bar(0), bar(10)]
        out, _ = normalise(gapped)
        assert len(out) == 2

    def test_clean_input_is_untouched(self) -> None:
        out, transformations = normalise(clean(10))
        assert transformations == []
        assert out == clean(10)


class TestManifest:
    def test_the_hash_covers_values_not_file_bytes(self) -> None:
        """Two writers producing the same bars must agree, or the hash answers
        'did the library change?' instead of 'did the data change?'."""
        rows = [(1, 2.0, 3.0), (4, 5.0, 6.0)]
        assert content_hash(rows) == content_hash(list(rows))

    def test_different_data_hashes_differently(self) -> None:
        assert content_hash([(1, 2.0)]) != content_hash([(1, 2.000001)])

    def test_the_fingerprint_is_order_independent(self) -> None:
        a = DatasetManifest("BTCUSDT", "5m", 0, 1, 1, "s", "t", content_hash="aa")
        b = DatasetManifest("ETHUSDT", "5m", 0, 1, 1, "s", "t", content_hash="bb")
        assert dataset_fingerprint([a, b]) == dataset_fingerprint([b, a])

    def test_a_changed_dataset_changes_the_fingerprint(self) -> None:
        a = DatasetManifest("BTCUSDT", "5m", 0, 1, 1, "s", "t", content_hash="aa")
        c = DatasetManifest("BTCUSDT", "5m", 0, 1, 1, "s", "t", content_hash="cc")
        assert dataset_fingerprint([a]) != dataset_fingerprint([c])

    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "x.parquet"
        path.touch()
        manifest = DatasetManifest("BTCUSDT", "5m", START, START + 1000, 42, "src", "now")
        write_manifest(path, manifest)
        assert read_manifest(path) == manifest

    def test_a_missing_manifest_is_none_not_an_error(self, tmp_path) -> None:
        assert read_manifest(tmp_path / "absent.parquet") is None

    def test_unknown_future_fields_are_dropped_not_fatal(self, tmp_path) -> None:
        """Forward compatibility: a newer writer must not brick an older reader."""
        path = tmp_path / "x.parquet"
        payload = {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "start_ms": 0,
            "end_ms": 1,
            "rows": 1,
            "source": "s",
            "downloaded_at": "t",
            "a_field_from_the_future": True,
        }
        (tmp_path / "x.parquet.manifest.json").write_text(json.dumps(payload))
        assert read_manifest(path).symbol == "BTCUSDT"  # type: ignore[union-attr]


class TestStore:
    def test_write_then_read_is_lossless(self, tmp_path) -> None:
        store = DataStore(tmp_path)
        bars = clean(100)
        _, manifest, report = store.write_klines("BTCUSDT", "5m", bars, source="test")
        assert report.status is QualityStatus.OK
        assert manifest.rows == 100

        back, read_manifest_, read_report = store.load_klines("BTCUSDT", "5m")
        assert back == bars
        assert read_manifest_ is not None
        assert read_manifest_.content_hash == manifest.content_hash
        assert read_report.status is QualityStatus.OK

    def test_reading_revalidates_rather_than_trusting_the_manifest(self, tmp_path) -> None:
        """A file can be truncated after download; the manifest would not know."""
        store = DataStore(tmp_path)
        store.write_klines("BTCUSDT", "5m", clean(100), source="test")

        import pandas as pd

        path = store.klines_path("BTCUSDT", "5m")
        pd.read_parquet(path).head(40).to_parquet(path, index=False)

        with pytest.raises(DataError, match="manifest says 100 rows"):
            store.load_klines("BTCUSDT", "5m")

    def test_a_missing_dataset_is_a_clear_error(self, tmp_path) -> None:
        with pytest.raises(DataError, match="dataset not found"):
            DataStore(tmp_path).load_klines("NOPEUSDT", "5m")

    def test_duplicates_are_repaired_losslessly_and_recorded(self, tmp_path) -> None:
        """A duplicate timestamp carries no information the first copy did not,
        so dropping it is lossless — but the manifest must say it happened, or
        the file silently differs from what the exchange served."""
        store = DataStore(tmp_path)
        _, manifest, report = store.write_klines(
            "DUPUSDT", "5m", [*clean(10), bar(5)], source="test"
        )
        assert report.status is QualityStatus.OK
        assert manifest.rows == 10
        assert any("duplicate" in t for t in manifest.transformations)

    def test_unrepairable_data_is_written_but_refused_on_load(self, tmp_path) -> None:
        """Written so the operator can inspect it; refused so nobody backtests it.

        A negative price cannot be repaired without inventing a number, so
        unlike a duplicate it survives normalisation and fails validation.
        """
        store = DataStore(tmp_path)
        corrupt = [*clean(5), bar(5, price=-1.0), *clean(10)[6:]]
        _, _, report = store.write_klines("BADUSDT", "5m", corrupt, source="test")
        assert store.klines_path("BADUSDT", "5m").is_file()
        assert report.status is QualityStatus.UNUSABLE

        with pytest.raises(DataError, match="failed validation"):
            store.load_klines("BADUSDT", "5m")

        bars, _, _ = store.load_klines("BADUSDT", "5m", strict=False)
        assert bars, "non-strict load must still return the rows for inspection"

    def test_exchange_info_round_trip_keeps_every_field(self, tmp_path) -> None:
        """Real filters, so the backtester stops guessing (BACKTEST_AUDIT B-15).

        Every field, not a hand-picked subset: `market_min_qty` is the minimum
        for MARKET orders, which is the order type this engine uses for entries,
        and losing it lets the backtest take positions the exchange would have
        rejected.
        """
        from tradebot.core.types import LeverageBracket

        store = DataStore(tmp_path)
        info = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            contract_type="PERPETUAL",
            price_precision=2,
            quantity_precision=3,
            tick_size=0.10,
            step_size=0.001,
            min_qty=0.001,
            max_qty=1000.0,
            min_notional=5.0,
            market_min_qty=0.002,
            market_max_qty=120.0,
            multiplier_up=5.0,
            multiplier_down=0.2,
            max_leverage=125,
            brackets=(LeverageBracket(1, 125, 50_000.0, 0.0, 0.004, 0.0),),
            onboard_date=1_569_398_400_000,
        )
        store.write_exchange_info({"BTCUSDT": info}, source="test")
        assert store.load_exchange_info()["BTCUSDT"] == info

    def test_exchange_info_keeps_the_market_order_minimum(self, tmp_path) -> None:
        """Named separately because it is the field that was silently dropped."""
        store = DataStore(tmp_path)
        info = SymbolInfo(
            "BTCUSDT",
            "BTC",
            "USDT",
            "TRADING",
            "PERPETUAL",
            2,
            3,
            0.10,
            0.001,
            0.001,
            1000.0,
            5.0,
            market_min_qty=0.004,
        )
        store.write_exchange_info({"BTCUSDT": info}, source="test")
        assert store.load_exchange_info()["BTCUSDT"].market_min_qty == 0.004

    def test_absent_exchange_info_is_empty_not_an_error(self, tmp_path) -> None:
        assert DataStore(tmp_path).load_exchange_info() == {}

    def test_funding_round_trip(self, tmp_path) -> None:
        store = DataStore(tmp_path)
        rates = {START: 0.0001, START + 28_800_000: -0.00005}
        store.write_funding("BTCUSDT", rates, source="test")
        assert store.load_funding("BTCUSDT") == rates

    def test_absent_funding_is_empty(self, tmp_path) -> None:
        assert DataStore(tmp_path).load_funding("BTCUSDT") == {}

    def test_discovery(self, tmp_path) -> None:
        store = DataStore(tmp_path)
        store.write_klines("BTCUSDT", "5m", clean(10), source="t")
        store.write_klines("ETHUSDT", "5m", clean(10), source="t")
        store.write_klines(
            "BTCUSDT",
            "1m",
            clean(
                10,
            ),
            source="t",
        )
        assert store.intervals() == ["1m", "5m"]
        assert store.symbols("5m") == ["BTCUSDT", "ETHUSDT"]


# --------------------------------------------------------------------------- #
def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", text)
    return buffer.getvalue()


HEADERLESS = (
    "1704067200000,42000.1,42100.5,41950.0,42050.2,120.5,"
    "1704067259999,5065000.0,900,60.2,2530000.0,0\n"
)


class TestBulkArchiveParsing:
    """Every format Binance has actually shipped."""

    def test_headerless_rows(self) -> None:
        bars = parse_vision_klines(_zip(HEADERLESS), "BTCUSDT")
        assert len(bars) == 1
        assert bars[0].open_time == 1_704_067_200_000
        assert bars[0].close == 42050.2
        assert bars[0].trades == 900

    def test_a_header_row_is_skipped(self) -> None:
        header = "open_time,open,high,low,close,volume,close_time,q,n,tb,tbq,ig\n"
        assert len(parse_vision_klines(_zip(header + HEADERLESS), "BTCUSDT")) == 1

    def test_microsecond_timestamps_are_normalised(self) -> None:
        """Binance switched the archive to microseconds partway through 2025.
        Guessing wrong shifts every bar by three orders of magnitude."""
        micro = HEADERLESS.replace("1704067200000,", "1704067200000000,").replace(
            "1704067259999,", "1704067259999000,"
        )
        bars = parse_vision_klines(_zip(micro), "BTCUSDT")
        assert bars[0].open_time == 1_704_067_200_000

    def test_a_bad_zip_raises_a_clear_error(self) -> None:
        with pytest.raises(DataError, match="not a valid zip"):
            parse_vision_klines(b"definitely not a zip", "BTCUSDT")

    def test_unparseable_rows_are_skipped_not_fatal(self) -> None:
        assert parse_vision_klines(_zip("garbage,,,\n" + HEADERLESS), "BTCUSDT") != []

    @pytest.mark.parametrize(
        "text",
        [
            "calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.0001\n",
            "1704067200000,0.0001\n",
        ],
    )
    def test_funding_layouts(self, text: str) -> None:
        assert parse_vision_funding(_zip(text), "BTCUSDT") == {1_704_067_200_000: 0.0001}

    def test_urls_match_the_documented_layout(self) -> None:
        import datetime

        day = datetime.date(2024, 3, 5)
        assert vision_kline_url("BTCUSDT", "1m", day).endswith(
            "daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03-05.zip"
        )
        assert vision_kline_url("BTCUSDT", "1m", day, monthly=True).endswith(
            "monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03.zip"
        )

    def test_day_and_month_ranges(self) -> None:
        start, end = 1_704_067_200_000, 1_704_067_200_000 + 3 * 86_400_000
        assert len(list(days_between(start, end))) == 3
        # A range inside one month yields that one month.
        assert len(list(months_between(start, end))) == 1
