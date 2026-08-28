# DATA_PIPELINE.md

How historical Binance USDⓈ-M Futures data gets from the exchange to a backtest,
and what is known to be wrong with it.

```
   ingest            normalise           validate            store
 ┌──────────┐      ┌───────────┐      ┌──────────┐      ┌────────────┐
 │ archive  │─────▶│  sort by  │─────▶│ quality  │─────▶│  Parquet   │
 │   REST   │      │ open_time │      │  checks  │      │ + manifest │
 └──────────┘      │  dedupe   │      └────┬─────┘      └─────┬──────┘
                   └───────────┘           │                  │
                                     data_quality.json    backtest
```

Each stage is separate and each writes an artefact, because a pipeline whose
stages are fused can only tell you *that* something is wrong, never *where*.

---

## Sources

| Source | What it gives | Credentials | Use it for |
|---|---|---|---|
| `data.binance.vision` | klines, funding, monthly + daily ZIPs | none | the bulk of any range |
| `fapi.binance.com` (REST) | klines, funding, **`exchangeInfo`** | public endpoints only | the recent tail, and trading rules |

The bulk archive is the default. A year of 1m bars for one symbol is 12 monthly
files instead of roughly 350 REST pages, and it needs no API key at all. It lags
real time by about a day, which is what `--source rest` is for.

**`exchangeInfo` always comes from REST**, whatever `--source` says, because the
archive does not carry it. This matters more than it sounds: without real tick
size, step size, minimum quantity and minimum **notional**, the backtester falls
back to permissive placeholder filters and takes positions the exchange would
have rejected — most often the small ones a 75 USDT account depends on.

### Formats the archive parser handles

Binance has changed the archive layout more than once, and each change is a
silent data corruption if unhandled:

* **headerless CSV** (the original) and **with a header row** (added later) —
  distinguished by whether the first cell parses as a number;
* **millisecond** and **microsecond** timestamps — Binance switched partway
  through 2025. A microsecond value is ~1000× larger and the boundary is
  unambiguous for any date this century, so magnitude decides it. Guessing wrong
  shifts every bar by three orders of magnitude, and the backtest still runs.

### What is deliberately not fetched

**Historical order books and bookTicker.** Binance publishes some of it, at tens
of GB per symbol-month and with gaps. The spread model in the backtester is
therefore parametric and says so.

This is the most important limitation in this document. Spread is a large
fraction of a scalper's cost, so claiming historical spreads we do not have
would be the single most flattering lie a backtest of this system could tell.
See the BASE/CONSERVATIVE/STRESS scenarios: they exist precisely because the
spread is assumed rather than measured.

---

## Storage layout

```
data/
  klines/<interval>/<SYMBOL>.parquet        + .manifest.json
  funding/<SYMBOL>.parquet                  + .manifest.json
  symbols/exchange_info.json
  reports/data_quality.json
```

One file per symbol/interval. Re-downloading one symbol rewrites one file, and a
partial download is distinguishable from a complete one.

### Manifests

Every dataset carries the fields brief §4 requires — symbol, interval, start,
end, source, download timestamp, schema version — plus two more that matter:

* **`content_hash`** — sha256 over the bar *values*, not the file bytes, so it
  is stable across Parquet versions, compression settings and column order. Two
  runs quoting the same hash saw the same bars. Two quoting different hashes did
  not, however similar the files look.
* **`transformations`** — what normalisation changed on the way in, so the file
  is never silently different from what the exchange served.

`dataset_fingerprint()` folds every manifest in a run into one hash, which is
what a backtest report quotes. It is order-independent.

---

## Normalisation: exactly two operations

1. **Sort by open time.** Indicators computed over shuffled bars are nonsense.
2. **Drop exact duplicate timestamps.** A duplicate carries no information the
   first copy did not.

Both are lossless and both are recorded in the manifest.

**Nothing else happens here, and in particular gaps are never filled.** A
synthesised bar is a fabricated price, and a backtest that trades on fabricated
prices is worse than no backtest, because it carries the same authority.

---

## Validation

Run as its own stage, with its own artefact, before anything downstream believes
the data. The distinction is between *unusable* and *imperfect*:

| Problem | Verdict | Why |
|---|---|---|
| duplicate timestamps | repaired losslessly, recorded | the second copy adds nothing |
| out-of-order bars | **UNUSABLE** | indicators over shuffled bars are meaningless |
| non-positive price | **UNUSABLE** | cannot be repaired without inventing a number |
| `high < low`, OHLC inconsistent | **UNUSABLE** | not a bar; a transcription error |
| negative volume | **UNUSABLE** | as above |
| invalid timestamps | **UNUSABLE** | as above |
| gaps | **DEGRADED** | exchanges have outages; measured and reported |
| >10% zero-volume bars | **DEGRADED** | a dead symbol scores as tradable and is not |

`DEGRADED` data is usable with the gap documented. `UNUSABLE` data is still
*written* — so the operator can look at it — but `load_klines()` refuses it
unless `strict=False`.

**Validation runs again on read.** The manifest records what was true at
download time; a file can be truncated or partially rewritten afterwards, and
trusting the manifest alone would let that through.

### The quality report

`data/reports/data_quality.json` carries the table brief §5 asks for, worst rows
first:

```
SYMBOL  INTERVAL  DATA_START  DATA_END  ROWS  MISSING  DUPLICATES  GAPS  COVERAGE  QUALITY_STATUS
```

---

## Known biases, stated rather than hidden

### Survivorship bias

**`--top` now refuses to run without `--i-understand-survivorship-bias`.**
Ranking by present-day volume and applying it to a historical range excludes
symbols that were liquid then and have since been delisted — whose outcomes are
usually bad. It can make a losing system look profitable, and nothing downstream
removes it.

Use `--symbols-file` with a listing snapshot from the **start** of the period to
avoid it. The provenance is written to `data/symbols/universe_provenance.json`
and carried in every run context as `POINT_IN_TIME_UNIVERSE` or
`PRESENT_DAY_UNIVERSE`, so a report cannot imply a clean universe it never had.

### The original note

`--top N` ranks symbols by **present-day** 24h volume and applies that ranking
to a **historical** range. A symbol that was liquid during the period but has
since been delisted will not appear, and its (often bad) outcome is silently
excluded. The script logs a warning when you use `--top`.

To avoid it, pass `--symbols` explicitly from a listing snapshot taken at the
start of the period. The pipeline cannot do this for you: Binance does not
publish a historical universe endpoint.

### Listing boundaries

A symbol listed mid-period simply has no archive files before its listing date.
The parser records those as missing rather than failing, so a newly-listed
symbol shows as `DEGRADED` with a large leading gap. That is the correct signal —
but it means **coverage figures for newly listed symbols are not comparable** to
those of symbols present throughout.

### exchangeInfo is a present-day snapshot

Tick size, step size, minimum quantity and minimum notional are fetched **now**
and applied to a **historical** backtest. Binance changes these over time and
publishes no historical filter endpoint, so a 2024 backtest runs under 2026's
filters. Bounded, real, and unavoidable — stated so it is not mistaken for
precision.

### Funding alignment

Funding timestamps are the exchange's own (00:00 / 08:00 / 16:00 UTC). These
happen to be exact multiples of 8h from the Unix epoch, which is why the
backtester's bucket arithmetic lines up. Correct today; it would break silently
if `funding_interval_hours` were ever changed.

---

## Reproducing a dataset

```bash
python scripts/fetch_data.py --top 40 --intervals 5m,15m \
    --start 2024-01-01 --end 2025-01-01 --out data
cat data/reports/data_quality.json | python -m json.tool | head -40
```

Then quote the fingerprint the run prints in any result derived from it.

---

## Status in this environment

`fapi.binance.com`, `api.binance.com` and `data.binance.vision` are **all
refused by this sandbox's egress policy** (403 to CONNECT). The pipeline is
exercised against fixtures and a fake source in
`tests/unit/test_data_pipeline.py`; running it here fails cleanly and writes
nothing:

```
$ python scripts/fetch_data.py --symbols BTCUSDT --intervals 5m \
    --start 2024-01-01 --end 2024-01-03 --out /tmp/realtry
Cannot reach Binance at https://testnet.binancefuture.com:
  HTTP 403 ... Host not in allowlist
This script needs a host with a route to Binance. Nothing was downloaded
and nothing was written.
```

**No real Binance data has been downloaded, and no backtest on real data has
been run, from this environment.**

---

## V3.2 — the quality artifact and the shared status vocabulary

### One vocabulary, two producers

`QualityStatus` (`OK` / `DEGRADED` / `UNUSABLE`) is now the *only* grading in
the project. Both quality producers speak it:

* `tradebot.data.validation.ValidationReport` — acquisition time, from
  `scripts/fetch_data.py`.
* `tradebot.backtesting.data.DataQuality` — load time, from `load_dataset()`.

Both satisfy the `DatasetQuality` protocol in `backtesting/trust.py`, which is
what the trust gate consumes. Before V3.2 the gate read three attributes
`DataQuality` did not have, so the checks silently did nothing; the protocol
makes the next such drift a type error.

### What makes a series UNUSABLE rather than DEGRADED

| condition | status |
| --- | --- |
| impossible OHLC — `high < low`, or `open`/`close` outside the bar's range | UNUSABLE |
| a price at or below zero on any of the four legs | UNUSABLE |
| negative volume | UNUSABLE |
| rows out of chronological order | UNUSABLE |
| a gap larger than the tolerance (`max_gap_bars`, default 10) | UNUSABLE |
| no rows | UNUSABLE |
| duplicate timestamps — dropped, but the file was wrong | DEGRADED |
| a gap within tolerance | DEGRADED |
| none of the above | OK |

Nothing is repaired by interpolation. A synthesised bar is a fabricated price.

### The per-run quality artifact

Every `backtest` and `walkforward` run writes `<report>.data_quality.json`
beside its report:

```json
{
  "dataset": "data",
  "trust": {"level": "TRUSTED", "blockers": [], "downgrades": [], "overrides": []},
  "rows": [
    {"SYMBOL": "BTCUSDT", "INTERVAL": "1m", "START": 1704067200000,
     "END": 1704153599999, "ROWS": 1440, "MISSING": 0, "DUPLICATES": 0,
     "GAPS": 0, "COVERAGE": 1.0, "QUALITY_STATUS": "OK"}
  ],
  "detail": []
}
```

It is written **before** the refusal check, so a refused run still leaves the
evidence of why it was refused.
