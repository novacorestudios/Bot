# DATA ACQUISITION REPORT — first real Binance dataset

**Status: COMPLETE.** A real Binance USDⓈ-M Futures dataset was imported,
validated, cleared the trust gate, and carried a full three-scenario backtest
end to end.

This is a **pipeline validation experiment**. It is not a profitability
experiment, and nothing in it should be read as evidence about the strategies.
Two trades were taken in total.

| | |
| --- | --- |
| Code | `e97a38d` (V3.2) at run time |
| Run ID | `f62cb17a9a37d2f8` |
| Config hash | `7f80d232738031d7` |
| Dataset fingerprint | `4149ae10aba2c977af359c513c76c443212d552bb7bea72006df8be4bb9a41a3` |
| Seed | 42 |
| Started | 2026-08-28T20:24:08Z |

## Provenance — read this before any other number

**MANUAL_SMOKE_UNIVERSE.** Two symbols, chosen by hand to exercise the
pipeline. This is **not** a research universe:

* it is **not point-in-time** — the symbols were not taken from a listing
  snapshot dated 2024-06-01;
* it is **not liquidity-ranked** — no scan selected them;
* it is **not representative** — two of the most liquid perpetuals in existence
  are not a sample of the market.

Any result from this universe describes these two symbols in this one month and
generalises to nothing. The label is carried in the run context and printed in
every report, and `--universe MANUAL_SMOKE_UNIVERSE` was added to the CLI for
exactly this reason: the two pre-existing labels would both have been false
here.

## Acquisition

Binance is unreachable from this environment — `data.binance.vision`,
`fapi.binance.com` and `api.binance.com` are all refused with 403 at CONNECT by
the egress policy. The first attempt at a direct download therefore failed and
is recorded in git history; `fetch_data.py` exited 2 at its preflight and wrote
nothing, which is the correct fail-safe.

The data was instead downloaded manually (Termux, from `data.binance.vision`),
packaged, and published as a GitHub release asset on the repository this session
can already reach.

| | |
| --- | --- |
| Release | `smoke-data-2024-06` |
| Asset | `binance_smoke_complete_2024-06.tar.gz` |
| Size | 6,120,367 bytes — matched the release metadata exactly |
| SHA-256 | `295986ff132a08363733fc4707f18f260621288f425d3831192eb9b3a49cd0a7` — matched GitHub's published digest byte for byte |

**Integrity:** `gzip -t` PASS. All 12 inner ZIPs pass `unzip -t` independently.
15 archive entries, matching the stated manifest.

**Nothing was fetched from Binance by this session, and no synthetic data was
generated at any point.**

## Import

Performed by the project's own public parsers — no parsing logic was added:

```
parse_vision_klines(zip_bytes, symbol)   ->  list[Candle]
parse_vision_funding(zip_bytes, symbol)  ->  {funding_time_ms: rate}
parse_symbol_info(entry, brackets)       ->  SymbolInfo
DataStore.write_klines / write_funding / write_exchange_info
```

The only substitution was the byte source: a local file instead of an HTTP GET.
How the bytes are interpreted is unchanged, which is what makes the stored
dataset equivalent to one `fetch_data.py` would have produced.

### Exact contents

* **Symbols:** `BTCUSDT`, `ETHUSDT`
* **Range:** 2024-06-01 00:00:00 UTC → 2024-06-30 23:59:59 UTC (30 days)
* **Timeframes:** 1m, 3m, 5m, 15m, 1h
* **Stored:** 12 Parquet files, 9.0 MB, DataStore layout with a manifest each

| SYMBOL | INTERVAL | ROWS | MISSING | DUPLICATES | GAPS | COVERAGE | QUALITY_STATUS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BTCUSDT | 1m | 43200 | 0 | 0 | 0 | 1.0000 | OK |
| BTCUSDT | 3m | 14400 | 0 | 0 | 0 | 1.0000 | OK |
| BTCUSDT | 5m | 8640 | 0 | 0 | 0 | 1.0000 | OK |
| BTCUSDT | 15m | 2880 | 0 | 0 | 0 | 1.0000 | OK |
| BTCUSDT | 1h | 720 | 0 | 0 | 0 | 1.0000 | OK |
| ETHUSDT | 1m | 43200 | 0 | 0 | 0 | 1.0000 | OK |
| ETHUSDT | 3m | 14400 | 0 | 0 | 0 | 1.0000 | OK |
| ETHUSDT | 5m | 8640 | 0 | 0 | 0 | 1.0000 | OK |
| ETHUSDT | 15m | 2880 | 0 | 0 | 0 | 1.0000 | OK |
| ETHUSDT | 1h | 720 | 0 | 0 | 0 | 1.0000 | OK |

**139,680 rows. Zero missing bars, zero duplicates, zero gaps, coverage 1.0000
on every series.** The counts are arithmetically exact for a 30-day month
(30 × 1440 = 43,200 one-minute bars, and so on down), which is itself a check:
a truncated or double-counted file could not produce these numbers.

Validation ran twice and agreed both times — once inside `write_klines` at
import, and again inside `load_klines`, which re-validates on read rather than
trusting the manifest. Summary: 10 datasets, 10 usable, 0 unusable, 0 degraded.

### Funding coverage

90 events per symbol — exactly 30 days × 3 at eight-hour intervals.

| Symbol | Events | Range | Rate min | Rate max |
| --- | ---: | --- | ---: | ---: |
| BTCUSDT | 90 | 1717200000000 → 1719763200000 | 0.000023 | 0.000164 |
| ETHUSDT | 90 | 1717200000000 → 1719763200000 | 0.000007 | 0.000203 |

**13 of the 90 events per symbol are not on the eight-hour grid** —
`1717257600001`, `1717660800008` and others carry +1 ms and +8 ms offsets — and
the observed spacing includes seven-hour intervals, not only eight.

This matters. Before V3.2 the engine found a funding rate by snapping the
timestamp onto an assumed eight-hour grid and looking that bucket up, so it
would have **missed all thirteen and priced them at zero**. That fix was written
from reasoning about the archive format, with no real data to test against; this
dataset is the first confirmation that the failure mode was real rather than
theoretical.

### exchangeInfo

Real snapshot from `fapi.binance.com`, supplied in the archive. **882 symbols
present, 882 parsed, zero failures.** No placeholder filters were used anywhere
in the run.

| Symbol | tick | step | minQty | marketMinQty | minNotional | maxLeverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 0.1 | 0.001 | 0.001 | 0.001 | 50.0 | 20 |
| ETHUSDT | 0.01 | 0.001 | 0.001 | 0.001 | 20.0 | 20 |

## Trust gate

```
Data trust          TRUSTED
  blockers:   none
  downgrades: none
  overrides:  none
```

**TRUSTED** — the first time the gate has returned it on real Binance data.
Every requirement was met without an override: all five required timeframes
present, no structural corruption, no gaps, real exchangeInfo found, and funding
history present for both symbols. `--allow-degraded` was not used and was not
needed.

## Backtest

```
tradebot backtest --data <dataset> \
  --edge-mode RESEARCH_STRICT \
  --universe MANUAL_SMOKE_UNIVERSE \
  --seed 42
```

`RESEARCH_STRICT` confirmed active in the run header: *"bootstrap disabled. A
strategy with no measured evidence will not trade, so this run cannot
manufacture an edge."*

**Runtime: 95 min 21 s**, exit 0. Code version stamped `v3.2`.

| Scenario | Trades | Net PnL | Return | Win rate | Max DD | Costs | Liquidations | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 2 | −0.0940 | −0.153% | 0.50 | 0.39% | 0.0591 | 0 | 0 |
| CONSERVATIVE | 2 | −0.0852 | −0.136% | 0.50 | 0.34% | 0.0605 | 0 | 0 |
| STRESS | 2 | −0.2686 | −0.378% | 0.00 | 0.42% | 0.0761 | 0 | 0 |

Survives STRESS: **False**. Capital 75.00 → 74.8849. Status: `MEASURED`.
Monte Carlo skipped — two trades is too few to resample meaningfully.

## What this run does and does not establish

**Established.** The pipeline works end to end on real exchange data: a real
archive is decoded correctly, real bars validate clean, real off-grid funding
events are found, real exchange filters are applied instead of placeholders, the
trust gate reaches TRUSTED on merit, three scenarios execute, the cost ledger
balances (no `cost_ledger_does_not_balance` errors), and a reproducible report
is written with a real dataset fingerprint. Costs scale correctly across
scenarios — 0.0591 → 0.0605 → 0.0761 — and STRESS converts the single winner
into a loser, which is the pessimistic model behaving as designed.

**Not established — and not even weakly suggested.** Anything about
profitability. **Two trades is the entire sample.** It cannot distinguish a
strategy from a coin flip. The negative PnL is not evidence of a losing system,
the 50% win rate is not a win rate, and `Survives STRESS: False` is not a
verdict on robustness — all three are one or two observations wearing the
formatting of a statistic. The report generator declined to run Monte Carlo on
this sample for precisely that reason, and that refusal should be read as the
most informative line in the output.

The low trade count is expected rather than anomalous: under `RESEARCH_STRICT`
a strategy with no measured evidence does not trade, and one month across two
symbols supplies almost no evidence to accumulate. That is the mode working.

## Limitations

1. **Two symbols, one month.** No statistical power whatsoever.
2. **Universe is hand-picked** — MANUAL_SMOKE_UNIVERSE, neither point-in-time
   nor ranked. Survivorship bias is not merely present; the universe was never
   sampled at all.
3. **exchangeInfo is a present-day snapshot** (fetched 2026-08-28) applied to
   June 2024 bars. Filters change over time — `minNotional` for BTCUSDT is 50.0
   in this snapshot and may not be what was in force during the period. The
   archive carries no historical exchangeInfo, so this mismatch is unavoidable
   with this data source rather than a defect.
4. **Zero rejected orders** shows `minNotional=50` did not block either attempt.
   With two attempts that is weak evidence, not a clearance for a 75 USDT
   account.
5. **`LIVE_LIKE_FORWARD`, not a holdout.** No `--split` was passed, so there is
   no out-of-sample separation in this run at all.
6. **Runtime scales badly.** 95 minutes for two symbols over one month at
   one-minute decision cadence. A year across thirty symbols is not a
   proportional extrapolation from this — budget for it before requesting one.

## Reproducing

The dataset is not committed — 9.0 MB of Parquet does not belong in the
repository. To rebuild it: download the release asset `smoke-data-2024-06`,
verify the SHA-256 above, extract, and import with the project's public parsers.
The dataset fingerprint `4149ae10aba2c977…` identifies the exact bytes; a run
quoting a different fingerprint is a different experiment.

---

```
DATA_DOWNLOAD:     PASS
ARCHIVE_INTEGRITY: PASS
DATA_IMPORT:       PASS
DATA_QUALITY:      TRUSTED
BACKTEST_PIPELINE: PASS
PROFITABILITY:     NOT_MEASURED
LIVE_TRADING:      BLOCKED
```

`BACKTEST_PIPELINE: PASS` means the pipeline executed end to end and exited 0.
It says nothing about the strategies.
