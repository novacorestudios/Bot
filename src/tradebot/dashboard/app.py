"""Mobile-friendly dashboard and JSON API.

Read-only by design. There is no endpoint that opens a position, changes a
limit, or resets a kill switch. The dashboard is for *seeing* what the bot is
doing; anything that could move money is deliberately absent, because a web
endpoint is a much larger attack surface than a config file on the VPS.

Access control is minimal but not absent: with no `DASHBOARD_TOKEN` the server
binds to loopback only. Exposing account balances and open positions on an open
port would be an obvious mistake, and defaulting to "reachable from anywhere"
is how it happens.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from tradebot.core.logging import get_logger

log = get_logger(__name__)


def create_app(engine: Any, token: str = "") -> FastAPI:  # nosec B107
    """Build the dashboard app around a running engine."""
    app = FastAPI(title="tradebot", docs_url=None, redoc_url=None, openapi_url=None)

    def authorise(request: Request) -> None:
        """Reject unauthorised requests when a token is configured."""
        if not token:
            client = request.client.host if request.client else ""
            if client not in {"127.0.0.1", "::1", "testclient", "localhost"}:
                raise HTTPException(
                    status_code=403,
                    detail="no DASHBOARD_TOKEN is set, so access is restricted to localhost",
                )
            return
        supplied = request.headers.get("x-dashboard-token") or request.query_params.get("token", "")
        if supplied != token:
            raise HTTPException(status_code=401, detail="invalid dashboard token")

    # ------------------------------------------------------------------ #
    @app.get("/health")
    async def health() -> JSONResponse:
        """Unauthenticated liveness+readiness, for Docker's HEALTHCHECK.

        Deliberately reports whether the bot is actually FUNCTIONAL, not merely
        whether the process is alive: a bot that is up but disconnected from
        Binance should fail its health check, because it is not doing its job.
        """
        report = engine.health.check()
        return JSONResponse(
            status_code=200 if report.healthy else 503,
            content={
                "status": "healthy" if report.healthy else "unhealthy",
                "safe_mode": report.safe_mode,
                "safe_mode_reason": report.safe_mode_reason,
                "uptime_sec": round(report.uptime_sec, 1),
                "mode": engine.config.mode.value,
            },
        )

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        authorise(request)
        return await engine.status_snapshot()

    @app.get("/api/positions")
    async def positions(request: Request) -> list[dict[str, Any]]:
        authorise(request)
        return engine.open_positions_view()

    @app.get("/api/trades")
    async def trades(
        request: Request, limit: int = Query(50, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        authorise(request)
        return await engine.recent_trades(limit)

    @app.get("/api/opportunities")
    async def opportunities(request: Request) -> list[dict[str, Any]]:
        """The Opportunity Dashboard.

        Appearing here does NOT mean a trade will be taken — these are ranked
        candidates that still have to clear consensus, the opportunity score,
        the edge filter and the risk engine.
        """
        authorise(request)
        return engine.opportunities_view()

    @app.get("/api/strategies")
    async def strategies(request: Request) -> dict[str, Any]:
        authorise(request)
        return engine.strategy_view()

    @app.get("/api/risk")
    async def risk(request: Request) -> dict[str, Any]:
        authorise(request)
        return engine.risk_view()

    @app.get("/api/matrices")
    async def matrices(request: Request) -> dict[str, Any]:
        """Strategy x regime and symbol x strategy performance."""
        authorise(request)
        return engine.matrices_view()

    @app.get("/api/execution-quality")
    async def execution_quality(request: Request) -> dict[str, Any]:
        """Expected versus actual execution cost, and edge calibration."""
        authorise(request)
        return engine.execution_quality_view()

    @app.get("/api/queue")
    async def queue(request: Request) -> dict[str, Any]:
        """Opportunities waiting, best first."""
        authorise(request)
        return engine.queue_view()

    @app.get("/api/market-data")
    async def market_data(request: Request) -> dict[str, Any]:
        """Feed health — which symbols are live, lagging or stale."""
        authorise(request)
        return engine.market_data_view()

    @app.get("/api/decisions")
    async def decisions(
        request: Request, limit: int = Query(100, ge=1, le=500), accepted: bool | None = None
    ) -> list[dict[str, Any]]:
        """The audit log: why the bot did, or did not, trade."""
        authorise(request)
        return await engine.recent_decisions(limit, accepted)

    @app.get("/api/health")
    async def health_detail(request: Request) -> dict[str, Any]:
        authorise(request)
        return engine.health.check().as_dict()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        authorise(request)
        return HTMLResponse(DASHBOARD_HTML)

    return app


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>tradebot</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
    --muted:#8b949e; --up:#3fb950; --down:#f85149; --warn:#d29922;
    --accent:#58a6ff;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#ffffff; --panel:#f6f8fa; --border:#d0d7de; --text:#1f2328;
            --muted:#656d76; }
  }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5
         -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom); }
  header { position:sticky; top:0; background:var(--panel);
           border-bottom:1px solid var(--border); padding:12px 16px; z-index:10; }
  h1 { margin:0; font-size:16px; font-weight:600; display:flex;
       justify-content:space-between; align-items:center; gap:8px; }
  .badge { font-size:11px; padding:2px 8px; border-radius:10px;
           border:1px solid var(--border); color:var(--muted); font-weight:500; }
  .badge.live { background:var(--down); color:#fff; border-color:var(--down); }
  .badge.safe { background:var(--warn); color:#000; border-color:var(--warn); }
  main { padding:12px; max-width:900px; margin:0 auto; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
          gap:8px; margin-bottom:16px; }
  .card { background:var(--panel); border:1px solid var(--border);
          border-radius:8px; padding:12px; }
  .card .label { font-size:11px; color:var(--muted); text-transform:uppercase;
                 letter-spacing:.04em; }
  .card .value { font-size:20px; font-weight:600; margin-top:4px;
                 font-variant-numeric:tabular-nums; }
  .up { color:var(--up); } .down { color:var(--down); } .warn { color:var(--warn); }
  section { margin-bottom:20px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em;
       color:var(--muted); margin:0 0 8px; }
  .scroll { overflow-x:auto; -webkit-overflow-scrolling:touch;
            border:1px solid var(--border); border-radius:8px; }
  table { width:100%; border-collapse:collapse; font-size:13px;
          font-variant-numeric:tabular-nums; }
  th { text-align:left; padding:8px 10px; background:var(--panel);
       color:var(--muted); font-weight:500; font-size:11px;
       text-transform:uppercase; white-space:nowrap; }
  td { padding:8px 10px; border-top:1px solid var(--border); white-space:nowrap; }
  .empty { padding:20px; text-align:center; color:var(--muted); font-size:13px; }
  .note { font-size:12px; color:var(--muted); margin-top:6px; line-height:1.5; }
  footer { padding:16px; text-align:center; color:var(--muted); font-size:11px; }
</style>
</head>
<body>
<header>
  <h1><span>tradebot</span><span id="badges"></span></h1>
</header>
<main>
  <div class="grid" id="stats"></div>

  <section>
    <h2>Open positions</h2>
    <div class="scroll"><table id="positions"></table></div>
  </section>

  <section>
    <h2>Top opportunities</h2>
    <div class="scroll"><table id="opportunities"></table></div>
    <p class="note">Appearing here does not mean a trade will be taken. A
    candidate must still clear strategy consensus, the opportunity score, the
    expected-net-edge filter and every risk limit.</p>
  </section>

  <section>
    <h2>Recent trades</h2>
    <div class="scroll"><table id="trades"></table></div>
  </section>

  <section>
    <h2>Strategies</h2>
    <div class="scroll"><table id="strategies"></table></div>
  </section>

  <section>
    <h2>Why the bot is not trading</h2>
    <div class="scroll"><table id="rejections"></table></div>
    <p class="note">Rejections are the normal case. NEGATIVE_EXPECTED_EDGE
    dominating means setups are being found whose expected move does not cover
    costs — that is the filter working, not a fault.</p>
  </section>
</main>
<footer id="footer"></footer>

<script>
const token = new URLSearchParams(location.search).get('token') || '';
const q = token ? ('?token=' + encodeURIComponent(token)) : '';

const num = (v, d = 2) => (v === null || v === undefined || isNaN(v))
  ? '-' : Number(v).toFixed(d);
const pct = (v, d = 2) => (v === null || v === undefined || isNaN(v))
  ? '-' : (Number(v) * 100).toFixed(d) + '%';
const cls = v => Number(v) > 0 ? 'up' : (Number(v) < 0 ? 'down' : '');

function table(el, cols, rows, empty) {
  if (!rows || !rows.length) {
    el.outerHTML = '<table id="' + el.id + '"><tr><td class="empty">'
      + empty + '</td></tr></table>';
    return;
  }
  el.innerHTML = '<thead><tr>' + cols.map(c => '<th>' + c[0] + '</th>').join('')
    + '</tr></thead><tbody>' + rows.map(r => '<tr>'
    + cols.map(c => '<td class="' + (c[2] ? c[2](r) : '') + '">'
    + c[1](r) + '</td>').join('') + '</tr>').join('') + '</tbody>';
}

async function get(path) {
  const res = await fetch('/api/' + path + q);
  if (!res.ok) throw new Error(path + ': ' + res.status);
  return res.json();
}

async function refresh() {
  try {
    const [s, positions, opportunities, trades, strategies] = await Promise.all([
      get('status'), get('positions'), get('opportunities'),
      get('trades?limit=15'), get('strategies'),
    ]);

    const badges = [];
    badges.push('<span class="badge' + (s.mode === 'LIVE' ? ' live' : '') + '">'
      + s.mode + '</span>');
    if (s.safe_mode) badges.push('<span class="badge safe">SAFE MODE</span>');
    if (!s.entries_allowed) badges.push('<span class="badge safe">NO ENTRIES</span>');
    document.getElementById('badges').innerHTML = badges.join(' ');

    document.getElementById('stats').innerHTML = [
      ['Equity', num(s.equity, 2), ''],
      ['Available', num(s.available_balance, 2), ''],
      ['Today PnL', num(s.today_pnl, 4), cls(s.today_pnl)],
      ['Total PnL', num(s.total_pnl, 4), cls(s.total_pnl)],
      ['Open', String(s.open_positions), ''],
      ['Trades', String(s.total_trades), ''],
      ['Win rate', pct(s.win_rate, 1), ''],
      ['Profit factor', num(s.profit_factor, 2), ''],
      ['Drawdown', pct(s.drawdown, 2), s.drawdown > 0.05 ? 'down' : ''],
      ['Fees', num(s.total_fees, 4), 'down'],
      ['Funding', num(s.total_funding, 4), ''],
      ['Open risk', pct(s.open_risk_pct, 2), ''],
    ].map(c => '<div class="card"><div class="label">' + c[0]
      + '</div><div class="value ' + c[2] + '">' + c[1] + '</div></div>').join('');

    table(document.getElementById('positions'), [
      ['Symbol', r => r.symbol],
      ['Dir', r => r.direction],
      ['Entry', r => num(r.entry_price, 6)],
      ['Price', r => num(r.current_price, 6)],
      ['PnL', r => num(r.unrealized_pnl, 4), r => cls(r.unrealized_pnl)],
      ['R', r => num(r.r_multiple, 2), r => cls(r.r_multiple)],
      ['SL', r => num(r.stop_loss, 6)],
      ['TP', r => num(r.take_profit, 6)],
      ['Age', r => r.duration],
      ['Strategy', r => r.strategy],
    ], positions, 'No open positions.');

    table(document.getElementById('opportunities'), [
      ['#', r => r.rank],
      ['Symbol', r => r.symbol],
      ['Score', r => num(r.market_score, 1)],
      ['Regime', r => r.regime],
      ['Strategy', r => r.best_strategy || '-'],
      ['Dir', r => r.direction],
      ['Conf', r => num(r.confidence, 0)],
      ['Edge', r => r.expected_net_edge === null ? '-'
        : pct(r.expected_net_edge, 3), r => cls(r.expected_net_edge)],
      ['Vol', r => pct(r.volatility_pct / 100, 2)],
      ['Risk', r => r.risk_level],
    ], opportunities, 'No candidates ranked yet.');

    table(document.getElementById('trades'), [
      ['Symbol', r => r.symbol],
      ['Dir', r => r.direction],
      ['Net', r => num(r.net_pnl, 4), r => cls(r.net_pnl)],
      ['R', r => num(r.r_multiple, 2), r => cls(r.r_multiple)],
      ['Fees', r => num(r.fees, 4)],
      ['Exit', r => r.exit_reason],
      ['Strategy', r => r.strategy],
    ], trades, 'No completed trades.');

    table(document.getElementById('strategies'), [
      ['Strategy', r => r.name],
      ['Trades', r => r.trades],
      ['Win', r => pct(r.win_rate, 1)],
      ['Expectancy', r => num(r.expectancy_r, 3) + 'R',
        r => cls(r.expectancy_r)],
      ['PF', r => num(r.profit_factor, 2)],
      ['Weight', r => num(r.allocation_weight, 2)],
      ['State', r => r.suspended ? 'SUSPENDED' : 'active',
        r => r.suspended ? 'down' : ''],
    ], Object.entries(strategies.strategies || {}).map(
      ([name, v]) => Object.assign({ name }, v)), 'No strategy data yet.');

    table(document.getElementById('rejections'), [
      ['Reason', r => r.reason],
      ['Count', r => r.count],
    ], (s.rejections || []), 'No rejections recorded.');

    document.getElementById('footer').textContent =
      'updated ' + new Date().toLocaleTimeString()
      + '  |  uptime ' + Math.floor(s.uptime_sec / 60) + 'm';
  } catch (e) {
    document.getElementById('footer').textContent = 'error: ' + e.message;
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
