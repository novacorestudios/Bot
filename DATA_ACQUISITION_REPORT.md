# DATA ACQUISITION REPORT — first real Binance dataset

**Attempted:** 2026-08-28 · **Code:** `70a51ee` (V3.2, CI green 6/6) · **Result: FAILED — no data acquired**

This experiment did not run. Binance is not reachable from this environment, and
the brief's rule 11 applies: stop, substitute nothing, report exactly what
failed. **No synthetic data was generated, no dataset was written, and no
backtest was run.** Every field below that would describe real data is marked
NOT ACQUIRED rather than filled with a placeholder.

## What was attempted

Universe type: **POINT_IN_TIME_UNIVERSE** (`--symbols-file`; `--top` deliberately
not used, per rule 1).

```
python scripts/fetch_data.py \
  --symbols-file /tmp/real_attempt/universe.txt \
  --intervals 1m,3m,5m,15m,1h \
  --start 2024-06-01 --end 2024-06-08 \
  --out /tmp/real_attempt/data
```

Proposed universe (4 symbols, one per line): `BTCUSDT`, `ETHUSDT`, `SOLUSDT`,
`BNBUSDT`. Range: 2024-06-01 → 2024-06-08 (7 days). Intervals: 1m, 3m, 5m, 15m,
1h, plus funding history and exchangeInfo.

**Caveat on the universe:** these four symbols were written from general
knowledge of what was listed in mid-2024, **not** from a verified listing
snapshot taken at the start of the period. A genuine POINT_IN_TIME_UNIVERSE
requires an actual snapshot. Since nothing downloaded, this list only exercised
the code path; it must be replaced before a real run.

## What failed

`fetch_data.py` exited **2** at its connectivity preflight, before any download:

```
Cannot reach Binance at https://testnet.binancefuture.com:
  HTTP 403 from /fapi/v1/time: Host not in allowlist: testnet.binancefuture.com.
  Add this host to your network egress settings to allow access.

This script needs a host with a route to Binance. Nothing was downloaded and
nothing was written.
```

`/tmp/real_attempt/data` **does not exist** — the directory was never created.

Direct reachability checks, outside the tool:

| Host | Purpose | Result |
| --- | --- | --- |
| `data.binance.vision` | bulk archive: klines + funding | **403 at CONNECT** |
| `fapi.binance.com` | exchangeInfo, REST klines | **403 at CONNECT** |
| `api.binance.com` | spot API | **403 at CONNECT** |
| `testnet.binancefuture.com` | preflight endpoint in PAPER mode | **403 at CONNECT** |

Confirmed independently by the egress proxy's own log
(`$HTTPS_PROXY/__agentproxy/status`), which recorded each rejection:

```
{"kind":"connect_rejected",
 "detail":"gateway answered 403 to CONNECT (policy denial or upstream failure)",
 "host":"data.binance.vision:443"}
```

This is an **organization egress policy for this session**, not a transient
network fault and not a defect in the acquisition pipeline. The proxy
documentation (`/root/.ccr/README.md`) is explicit that 403 at CONNECT means the
host is not permitted, and that it must be reported rather than routed around.

## Report fields

| Field | Value |
| --- | --- |
| Exact symbols | NOT ACQUIRED (4 proposed, unverified — see caveat) |
| Exact date range | NOT ACQUIRED (2024-06-01 → 2024-06-08 requested) |
| Exact timeframes | NOT ACQUIRED (1m, 3m, 5m, 15m, 1h requested) |
| Number of rows | NOT ACQUIRED — 0 files written |
| Data quality | NOT MEASURED — `validate_data.py` not run, nothing to validate |
| Funding coverage | NOT ACQUIRED |
| exchangeInfo status | NOT ACQUIRED |
| Universe type | POINT_IN_TIME_UNIVERSE (intended); nothing stamped, no dataset exists |
| Trust status | NOT EVALUATED — the gate never ran, because no data reached it |
| Runtime | ~1s to fail the preflight; 0s of download |

## What this does and does not tell us

It tells us the acquisition pipeline **fails safely**: it refused at the
preflight, reported the blocked host, and wrote nothing — no empty dataset, no
partial dataset, no fabricated bars. That is the behaviour V3.2's trust work was
built to guarantee, and it held on first contact with a real failure.

It tells us **nothing whatsoever** about the strategies, the data, or
profitability. No backtest was run on real data. None of BASE, CONSERVATIVE or
STRESS was executed for this experiment.

## To run this experiment for real

The environment needs these hosts on its egress allowlist:

1. `data.binance.vision` — klines and funding history (bulk archive, no credentials)
2. `fapi.binance.com` — exchangeInfo, and REST klines if `--source rest` is used

`testnet.binancefuture.com` is reached only because the preflight probes the
endpoint configured for the current mode. Allowlisting the two hosts above is
what a data download actually requires.

Alternatively, run `scripts/fetch_data.py` on a host that has a route to
Binance and copy the resulting `data/` tree here — the DataStore format is
self-contained, and `validate_data.py` plus the trust gate will check it on
arrival exactly as they would have checked a local download.

Before the real run, replace the proposed universe with a genuine listing
snapshot from 2024-06-01, or the dataset carries survivorship bias regardless of
which flag produced it.

---

```
DATA_ACQUISITION: FAIL
DATA_QUALITY:     NOT EVALUATED (no data)
BACKTEST_PIPELINE: NOT RUN (no data)
PROFITABILITY:    NOT_MEASURED
LIVE_TRADING:     BLOCKED
```
