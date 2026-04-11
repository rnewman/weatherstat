"""Embedded HTTP server for remote control of the Weatherstat TUI.

The server runs in a daemon thread inside the TUI process and shares
in-memory state via `TUIController`. Designed for phone use on a trusted
LAN: small page, no JavaScript, 30s meta-refresh, mobile viewport. There
is no authentication — bind to a private interface and rely on network
isolation.

Routes:
    GET  /             — HTML status page
    GET  /status.json  — JSON snapshot of the same data
    GET  /style.css    — inline stylesheet
    POST /action/run   — start a control cycle
    POST /action/sysid — start sysid
    POST /action/force — force-execute past overrides
    POST /action/profile [name=<profile>] — toggle or set comfort profile
    POST /action/live    [mode=on|off]    — toggle or set live mode
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from weatherstat.tui.controller import (
    ActionResult,
    StatusSnapshot,
    TUIController,
)

# ── Public API ──────────────────────────────────────────────────────────────


def start_web_server(
    controller: TUIController,
    host: str,
    port: int,
    *,
    log: Callable[[str], None] | None = None,
) -> ThreadingHTTPServer:
    """Start the embedded HTTP server in a daemon thread.

    Returns the server handle so callers can `shutdown()` it. The thread
    exits with the process.
    """
    handler_cls = _make_handler(controller, log or (lambda _msg: None))
    server = ThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="weatherstat-web")
    thread.start()
    return server


# ── Request handler factory ─────────────────────────────────────────────────


def _make_handler(
    controller: TUIController,
    log: Callable[[str], None],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # Silence the default request log; route through our `log` callback
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            log(f"[web] {format % args}")

        # ── GET ──
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._render_status()
            elif parsed.path == "/status.json":
                self._render_json()
            elif parsed.path == "/style.css":
                self._render_css()
            else:
                self._send_text(HTTPStatus.NOT_FOUND, "Not Found\n")

        # ── POST ──
        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/action/"):
                self._send_text(HTTPStatus.NOT_FOUND, "Not Found\n")
                return

            action = parsed.path[len("/action/"):]
            form = self._read_form()

            try:
                result = self._dispatch_action(action, form)
            except _UnknownAction:
                self._send_text(HTTPStatus.NOT_FOUND, f"Unknown action: {action}\n")
                return

            # Respond: form posts get a 303 back to /, JSON posts get JSON.
            wants_json = self.headers.get("Accept", "").startswith("application/json")
            if wants_json:
                self._send_json(
                    HTTPStatus.OK if result.ok else HTTPStatus.BAD_REQUEST,
                    {
                        "status": result.status,
                        "message": result.message,
                        "detail": result.detail,
                    },
                )
            else:
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", "/")
                self.end_headers()

        # Disallow other methods explicitly so 405 is unambiguous.
        def do_PUT(self) -> None:  # noqa: N802
            self._send_text(HTTPStatus.METHOD_NOT_ALLOWED, "Method Not Allowed\n")

        def do_DELETE(self) -> None:  # noqa: N802
            self._send_text(HTTPStatus.METHOD_NOT_ALLOWED, "Method Not Allowed\n")

        # ── Helpers ──
        def _read_form(self) -> dict[str, str]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length == 0:
                return {}
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            parsed = parse_qs(body)
            return {k: v[0] for k, v in parsed.items() if v}

        def _dispatch_action(self, action: str, form: dict[str, str]) -> ActionResult:
            if action == "run":
                return controller.run_cycle()
            if action == "sysid":
                return controller.run_sysid()
            if action == "force":
                return controller.force_execute()
            if action == "profile":
                name = form.get("name")
                if name:
                    return controller.set_profile(name)
                return controller.toggle_profile()
            if action == "live":
                mode = form.get("mode")
                if mode == "on":
                    return controller.set_live(True)
                if mode == "off":
                    return controller.set_live(False)
                return controller.toggle_live()
            raise _UnknownAction(action)

        def _render_status(self) -> None:
            snap = controller.latest_snapshot() or controller.publish_snapshot()
            html = _render_html(snap, controller)
            self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")

        def _render_json(self) -> None:
            snap = controller.latest_snapshot() or controller.publish_snapshot()
            self._send_json(HTTPStatus.OK, _snapshot_to_dict(snap))

        def _render_css(self) -> None:
            self._send(HTTPStatus.OK, _CSS.encode("utf-8"), "text/css; charset=utf-8")

        def _send_text(self, status: HTTPStatus, body: str) -> None:
            self._send(status, body.encode("utf-8"), "text/plain; charset=utf-8")

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
            self._send(status, body, "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


class _UnknownAction(Exception):
    pass


# ── HTML rendering ──────────────────────────────────────────────────────────


def _render_html(snap: StatusSnapshot, controller: TUIController) -> str:
    """Mobile-first single-page status view. No JavaScript."""
    profile = snap.profile or "?"
    mode_class = "live" if snap.live else "dry"
    mode_label = "LIVE" if snap.live else "DRY-RUN"
    outdoor = f"{snap.outdoor_temp:.0f}°" if snap.outdoor_temp is not None else "?"
    now_local = datetime.now().strftime("%H:%M")

    cycle_pill = _pill(
        "running" if snap.cycle_running else "idle",
        "running" if snap.cycle_running else "idle",
    )
    sysid_pill = _pill(
        "fitting" if snap.sysid_running else "idle",
        "running" if snap.sysid_running else "idle",
    )

    next_cycle_str = "—"
    if snap.next_cycle_at is not None:
        remaining = snap.next_cycle_at - datetime.now(UTC)
        secs = max(0, int(remaining.total_seconds()))
        next_cycle_str = f"{secs // 60}m {secs % 60:02d}s"

    sysid_age = _format_age(snap.sysid_age)
    collector_age = _format_age(snap.collector_age)

    # ── Action buttons ──
    has_overrides = bool(controller.overrides())
    cycle_disabled = " disabled" if snap.cycle_running else ""
    sysid_disabled = " disabled" if snap.sysid_running else ""
    force_disabled = "" if has_overrides else " disabled"

    actions = f"""
    <section class="actions">
      <form method="post" action="/action/run">
        <button type="submit"{cycle_disabled}>Run cycle</button>
      </form>
      <form method="post" action="/action/force">
        <button type="submit"{force_disabled}>Force execute</button>
      </form>
      <form method="post" action="/action/profile">
        <button type="submit">Profile →</button>
      </form>
      <form method="post" action="/action/live">
        <button type="submit" class="{mode_class}">{mode_label} ↔</button>
      </form>
      <form method="post" action="/action/sysid">
        <button type="submit"{sysid_disabled}>Run sysid</button>
      </form>
    </section>
    """

    # ── Temperatures ──
    temps_rows: list[str] = []
    for sensor_col, value in snap.temps.items():
        label = snap.sensor_labels.get(sensor_col, sensor_col.removesuffix("_temp"))
        bounds = snap.comfort.get(label)
        cell_class = "temp"
        bound_str = ""
        if bounds:
            cell_class = _temp_class(value, bounds)
            bound_str = (
                f"{bounds.acceptable_lo:.0f}°"
                f"<span class='band'> · {bounds.preferred_lo:.0f}–{bounds.preferred_hi:.0f}° · </span>"
                f"{bounds.acceptable_hi:.0f}°"
            )
        mrt = snap.mrt_offsets.get(sensor_col)
        mrt_str = ""
        if mrt is not None and abs(mrt) >= 0.05:
            mrt_str = f"<span class='mrt'>{mrt:+.1f} mrt</span>"
        temps_rows.append(
            f"<tr><td>{escape(label)}</td>"
            f"<td class='{cell_class}'>{value:.1f}°</td>"
            f"<td class='bounds'>{bound_str}</td>"
            f"<td>{mrt_str}</td></tr>"
        )

    temps_html = (
        "<section><h2>Temperatures</h2><table>"
        + "".join(temps_rows)
        + "</table></section>"
    ) if temps_rows else ""

    # ── Effectors ──
    effector_rows: list[str] = []
    for e in snap.effectors:
        mode_str = e.mode.upper() if e.mode else "OFF"
        target_str = f"{e.target:.0f}°" if e.target is not None else ""
        timing_parts = []
        if e.delay_steps:
            timing_parts.append(f"delay {e.delay_steps * 5}m")
        if e.duration_steps is not None:
            timing_parts.append(f"dur {e.duration_steps * 5}m")
        timing = " · ".join(timing_parts)
        override = (
            f"<span class='override'>OVERRIDE: {escape(e.override)}</span>"
            if e.override
            else ""
        )
        rationale = snap.rationale.get(e.name, "")
        rationale_html = (
            f"<div class='rationale'>{escape(rationale)}</div>" if rationale else ""
        )
        mode_class = "off" if mode_str == "OFF" else "on"
        effector_rows.append(
            f"<tr><td>{escape(e.name)}</td>"
            f"<td class='mode {mode_class}'>{mode_str}</td>"
            f"<td>{target_str}</td>"
            f"<td>{escape(timing)}</td>"
            f"<td>{override}</td></tr>"
            + (f"<tr class='detail'><td colspan='5'>{rationale_html}</td></tr>" if rationale else "")
        )
    if effector_rows:
        cost_str = ""
        if snap.costs is not None:
            cost_str = (
                f"<p class='cost'>Cost: {snap.costs.total:.1f} "
                f"(comfort {snap.costs.comfort:.1f} + energy {snap.costs.energy:.2f})"
            )
            if snap.costs.baseline is not None:
                cost_str += f" · all-off {snap.costs.baseline:.1f}"
            cost_str += "</p>"
        effectors_html = (
            "<section><h2>Effectors</h2><table>"
            + "".join(effector_rows)
            + "</table>"
            + cost_str
            + "</section>"
        )
    else:
        effectors_html = "<section><h2>Effectors</h2><p>(no decision yet)</p></section>"

    # ── Forecast ──
    forecast_html = ""
    if snap.forecast:
        cells = " ".join(
            f"<span class='fc'><b>{escape(h)}</b> {t:.0f}°</span>"
            for h, t in snap.forecast.items()
        )
        cond = f" · {escape(snap.weather_condition)}" if snap.weather_condition else ""
        forecast_html = f"<section><h2>Forecast</h2><p>{cells}{cond}</p></section>"

    # ── Environment ──
    env_html = ""
    if snap.environment:
        active = [e for e in snap.environment if e.is_active]
        if active:
            items = " · ".join(
                f"<span class='env-active'>{escape(e.label)} {escape(e.kind)}</span>"
                for e in active
            )
            env_html = f"<section><h2>Environment</h2><p>{items}</p></section>"
        else:
            env_html = "<section><h2>Environment</h2><p class='dim'>All at default</p></section>"

    # ── Opportunities ──
    opp_items: list[str] = []
    for w in snap.warnings:
        opp_items.append(f"<li class='warn'>{escape(str(w.get('message', '?')))}</li>")
    for opp in snap.opportunities:
        device = escape(str(opp.get("device", "?")))
        action = escape(str(opp.get("action", "?")))
        mins = opp.get("in_minutes", 0)
        delta = opp.get("cost_delta", 0)
        timing = "now" if mins == 0 else f"in {mins}m"
        active_class = "active" if opp.get("current_state") else "proactive"
        opp_items.append(
            f"<li class='{active_class}'>{action} {device} {timing} ({delta:+.2f})</li>"
        )
    if opp_items:
        opps_html = "<section><h2>Opportunities</h2><ul>" + "".join(opp_items) + "</ul></section>"
    else:
        opps_html = "<section><h2>Opportunities</h2><p class='dim'>None</p></section>"

    # ── Header bar ──
    header = f"""
    <header>
      <div class="row1">
        <span class="profile">{escape(profile)}</span>
        <span class="mode {mode_class}">{mode_label}</span>
        <span class="outdoor">{outdoor}</span>
        <span class="time">{now_local}</span>
      </div>
      <div class="row2">
        <span>Cycle {cycle_pill} · next {escape(next_cycle_str)}</span>
        <span>Sysid {sysid_pill} · {escape(sysid_age)}</span>
        <span>Collector {escape(collector_age)} · {snap.collector_rows} rows</span>
      </div>
    </header>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <title>Weatherstat</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
{header}
{actions}
{temps_html}
{effectors_html}
{forecast_html}
{env_html}
{opps_html}
<footer><a href="/">refresh</a> · <a href="/status.json">json</a></footer>
</body>
</html>
"""


def _temp_class(value: float, bounds) -> str:
    if value < bounds.acceptable_lo or value > bounds.acceptable_hi:
        return "temp red"
    if value < bounds.preferred_lo or value > bounds.preferred_hi:
        return "temp yellow"
    return "temp green"


def _pill(text: str, kind: str) -> str:
    return f"<span class='pill {kind}'>{escape(text)}</span>"


def _format_age(td) -> str:
    if td is None:
        return "?"
    total = int(td.total_seconds())
    if total < 0:
        return "<1s"
    if total < 60:
        return f"{total}s"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


# ── JSON serialization ──────────────────────────────────────────────────────


def _snapshot_to_dict(snap: StatusSnapshot) -> dict:
    return {
        "timestamp": snap.timestamp.isoformat(),
        "profile": snap.profile,
        "live": snap.live,
        "cycle_running": snap.cycle_running,
        "sysid_running": snap.sysid_running,
        "next_cycle_at": snap.next_cycle_at.isoformat() if snap.next_cycle_at else None,
        "sysid_age_seconds": snap.sysid_age.total_seconds() if snap.sysid_age else None,
        "collector_age_seconds": snap.collector_age.total_seconds() if snap.collector_age else None,
        "collector_rows": snap.collector_rows,
        "outdoor_temp": snap.outdoor_temp,
        "weather_condition": snap.weather_condition,
        "temps": snap.temps,
        "comfort": {
            label: {
                "acceptable_lo": cb.acceptable_lo,
                "preferred_lo": cb.preferred_lo,
                "preferred_hi": cb.preferred_hi,
                "acceptable_hi": cb.acceptable_hi,
            }
            for label, cb in snap.comfort.items()
        },
        "mrt_offsets": snap.mrt_offsets,
        "environment": [
            {"label": e.label, "kind": e.kind, "active": e.is_active}
            for e in snap.environment
        ],
        "forecast": snap.forecast,
        "effectors": [
            {
                "name": e.name,
                "mode": e.mode,
                "target": e.target,
                "delay_steps": e.delay_steps,
                "duration_steps": e.duration_steps,
                "override": e.override,
            }
            for e in snap.effectors
        ],
        "command_targets": snap.command_targets,
        "rationale": snap.rationale,
        "costs": (
            {
                "total": snap.costs.total,
                "comfort": snap.costs.comfort,
                "energy": snap.costs.energy,
                "baseline": snap.costs.baseline,
            }
            if snap.costs
            else None
        ),
        "sensor_costs": snap.sensor_costs,
        "baseline_sensor_costs": snap.baseline_sensor_costs,
        "predictions": snap.predictions,
        "opportunities": snap.opportunities,
        "warnings": snap.warnings,
    }


def _json_default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "total_seconds"):
        return obj.total_seconds()
    raise TypeError(f"not serializable: {type(obj).__name__}")


# ── Inline stylesheet ───────────────────────────────────────────────────────


_CSS = """
:root {
  color-scheme: light dark;
  --bg: #0e1116;
  --fg: #e6edf3;
  --dim: #7d8590;
  --accent: #2f81f7;
  --green: #3fb950;
  --yellow: #d29922;
  --red: #f85149;
  --panel: #161b22;
  --border: #30363d;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #ffffff;
    --fg: #1f2328;
    --dim: #57606a;
    --accent: #0969da;
    --green: #1a7f37;
    --yellow: #9a6700;
    --red: #cf222e;
    --panel: #f6f8fa;
    --border: #d0d7de;
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--fg);
  margin: 0;
  padding: 0 12px 24px;
  font-size: 16px;
  line-height: 1.4;
}
header {
  position: sticky;
  top: 0;
  background: var(--bg);
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  z-index: 5;
}
header .row1 {
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 18px;
  font-weight: 600;
}
header .row2 {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 4px;
  font-size: 13px;
  color: var(--dim);
}
.profile { padding: 2px 8px; border: 1px solid var(--border); border-radius: 6px; }
.mode { padding: 2px 8px; border-radius: 6px; font-weight: 700; }
.mode.live { background: var(--red); color: #fff; }
.mode.dry { background: var(--green); color: #fff; }
.outdoor { margin-left: auto; }
.time { color: var(--dim); }
.pill {
  display: inline-block;
  padding: 0 6px;
  border-radius: 4px;
  font-size: 12px;
  background: var(--panel);
  border: 1px solid var(--border);
}
.pill.running { background: var(--yellow); color: #fff; border-color: var(--yellow); }
.actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 16px 0;
}
.actions form { margin: 0; }
.actions button {
  width: 100%;
  min-height: 56px;
  font-size: 16px;
  font-weight: 600;
  background: var(--panel);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 8px;
  cursor: pointer;
  -webkit-appearance: none;
  appearance: none;
}
.actions button:active { background: var(--accent); color: #fff; }
.actions button:disabled { opacity: 0.4; cursor: not-allowed; }
.actions button.live { background: var(--red); color: #fff; border-color: var(--red); }
.actions button.dry { background: var(--green); color: #fff; border-color: var(--green); }
section {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 12px 0;
  padding: 10px 12px;
}
section h2 {
  margin: 0 0 6px;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--dim);
}
table { width: 100%; border-collapse: collapse; }
td { padding: 4px 0; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: 0; }
.temp { text-align: right; font-weight: 600; padding-right: 8px; }
.temp.green { color: var(--green); }
.temp.yellow { color: var(--yellow); }
.temp.red { color: var(--red); }
.bounds { color: var(--dim); font-size: 13px; }
.bounds .band { color: var(--dim); }
.mrt { color: var(--dim); font-size: 12px; }
.mode.on { color: var(--green); font-weight: 700; }
.mode.off { color: var(--dim); }
.override { color: var(--red); font-weight: 700; font-size: 12px; }
tr.detail td { border-bottom: 0; padding-top: 0; }
.rationale { color: var(--dim); font-size: 12px; padding-left: 8px; }
.cost { color: var(--dim); font-size: 13px; margin: 6px 0 0; }
.fc { display: inline-block; margin-right: 10px; }
.fc b { color: var(--accent); font-weight: 600; }
.env-active { color: var(--yellow); }
.dim { color: var(--dim); }
ul { margin: 0; padding-left: 18px; }
ul li.warn { color: var(--red); font-weight: 700; }
ul li.active { color: var(--accent); }
ul li.proactive { color: var(--dim); }
footer {
  margin-top: 24px;
  text-align: center;
  font-size: 12px;
  color: var(--dim);
}
footer a { color: var(--dim); }
"""
