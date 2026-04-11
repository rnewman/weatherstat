"""Tests for the embedded web frontend.

Spins up a real `ThreadingHTTPServer` against a `FakeController` and
exercises the routes via the standard library's `http.client`.
"""

from __future__ import annotations

import http.client
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import pytest

from weatherstat.tui.controller import (
    ActionResult,
    ComfortBounds,
    CycleCosts,
    EffectorView,
    EnvEntry,
    StatusSnapshot,
)
from weatherstat.web import start_web_server

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_snapshot(**overrides) -> StatusSnapshot:
    base = {
        "timestamp": datetime(2026, 4, 11, 12, 0, tzinfo=UTC),
        "profile": "Home",
        "live": False,
        "cycle_running": False,
        "sysid_running": False,
        "next_cycle_at": datetime(2026, 4, 11, 12, 5, tzinfo=UTC),
        "sysid_age": timedelta(hours=2),
        "collector_age": timedelta(seconds=30),
        "collector_rows": 12345,
        "local_tz": "America/Los_Angeles",
        "outdoor_temp": 52.0,
        "weather_condition": "partly_cloudy",
        "temps": {"thermostat_upstairs_temp": 68.5, "kitchen_temp": 70.1},
        "sensor_labels": {
            "thermostat_upstairs_temp": "Upstairs",
            "kitchen_temp": "Kitchen",
        },
        "comfort": {
            "Upstairs": ComfortBounds(60.0, 65.0, 72.0, 80.0),
            "Kitchen": ComfortBounds(60.0, 67.0, 73.0, 80.0),
        },
        "mrt_offsets": {"thermostat_upstairs_temp": -1.5},
        "environment": (
            EnvEntry(label="Bedroom", kind="window", is_active=False),
            EnvEntry(label="Kitchen", kind="window", is_active=True),
        ),
        "forecast": {"1h": 53.0, "6h": 48.0},
        "effectors": (
            EffectorView(
                name="thermostat_upstairs",
                mode="heating",
                target=68.0,
                delay_steps=0,
                duration_steps=12,
                override=None,
            ),
        ),
        "command_targets": {"thermostat_upstairs": 68.0},
        "rationale": {"thermostat_upstairs": "fights cold bedroom"},
        "costs": CycleCosts(total=4.2, comfort=3.8, energy=0.4, baseline=8.1),
        "sensor_costs": {"bedroom_temp": 1.2},
        "baseline_sensor_costs": {"bedroom_temp": 3.0},
        "predictions": {"bedroom_temp": {"1h": 67.0, "2h": 67.5}},
        "opportunities": [
            {
                "device": "kitchen_window",
                "action": "open",
                "in_minutes": 0,
                "cost_delta": -0.8,
                "current_state": False,
            }
        ],
        "warnings": [{"message": "bedroom predicted below floor in 4h"}],
    }
    base.update(overrides)
    return StatusSnapshot(**base)


class FakeController:
    """Stand-in for TUIController exposing the surface the web server uses."""

    def __init__(self, snapshot: StatusSnapshot | None = None) -> None:
        self._snapshot = snapshot or _make_snapshot()
        self._overrides: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []

    # Used by web.py
    def latest_snapshot(self) -> StatusSnapshot:
        return self._snapshot

    def publish_snapshot(self) -> StatusSnapshot:
        return self._snapshot

    def overrides(self) -> dict[str, str]:
        return dict(self._overrides)

    def run_cycle(self) -> ActionResult:
        self.calls.append(("run_cycle", {}))
        return ActionResult("started", "Cycle started")

    def run_sysid(self) -> ActionResult:
        self.calls.append(("run_sysid", {}))
        return ActionResult("started", "Sysid started")

    def force_execute(self) -> ActionResult:
        self.calls.append(("force_execute", {}))
        if not self._overrides:
            return ActionResult("noop", "No overrides")
        return ActionResult("started", "Forcing")

    def toggle_profile(self) -> ActionResult:
        self.calls.append(("toggle_profile", {}))
        return ActionResult("ok", "Switched to Away", {"profile": "Away"})

    def set_profile(self, name: str) -> ActionResult:
        self.calls.append(("set_profile", {"name": name}))
        return ActionResult("ok", f"Switched to {name}", {"profile": name})

    def toggle_live(self) -> ActionResult:
        self.calls.append(("toggle_live", {}))
        return ActionResult("ok", "Switched to LIVE", {"live": True})

    def set_live(self, live: bool) -> ActionResult:
        self.calls.append(("set_live", {"live": live}))
        return ActionResult("ok", f"Switched to {'LIVE' if live else 'DRY-RUN'}", {"live": live})


@pytest.fixture
def server_and_ctrl():
    ctrl = FakeController()
    server = start_web_server(ctrl, host="127.0.0.1", port=0)
    try:
        yield server, ctrl
    finally:
        server.shutdown()
        server.server_close()


def _client(server) -> http.client.HTTPConnection:
    host, port = server.server_address
    return http.client.HTTPConnection(host, port, timeout=5)


def _get(server, path: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    conn = _client(server)
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body
    finally:
        conn.close()


def _post(
    server,
    path: str,
    form: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, dict[str, str]]:
    conn = _client(server)
    try:
        body = urlencode(form or {})
        h = {"Content-Type": "application/x-www-form-urlencoded", **(headers or {})}
        conn.request("POST", path, body=body, headers=h)
        resp = conn.getresponse()
        out = resp.read().decode("utf-8", errors="replace")
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, out, resp_headers
    finally:
        conn.close()


# ── GET tests ───────────────────────────────────────────────────────────────


class TestGetRoutes:
    def test_status_html_renders(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        status, body = _get(server, "/")
        assert status == 200
        assert "<title>Weatherstat</title>" in body
        assert "Upstairs" in body
        assert "Kitchen" in body
        assert 'class="actions"' in body
        assert "thermostat_upstairs" in body
        assert "DRY-RUN" in body

    def test_status_json_renders(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        status, body = _get(server, "/status.json")
        assert status == 200
        data = json.loads(body)
        assert data["profile"] == "Home"
        assert data["live"] is False
        assert data["temps"]["kitchen_temp"] == 70.1
        assert data["effectors"][0]["name"] == "thermostat_upstairs"
        assert data["costs"]["total"] == 4.2
        assert data["opportunities"][0]["device"] == "kitchen_window"

    def test_style_css(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        status, body = _get(server, "/style.css")
        assert status == 200
        assert "body" in body

    def test_unknown_get_404(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        status, _ = _get(server, "/nope")
        assert status == 404


# ── POST tests ──────────────────────────────────────────────────────────────


class TestPostRoutes:
    def test_post_run_calls_controller_and_redirects(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        status, _, headers = _post(server, "/action/run")
        assert status == 303
        assert headers.get("location") == "/"
        assert ("run_cycle", {}) in ctrl.calls

    def test_post_sysid(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        status, _, _ = _post(server, "/action/sysid")
        assert status == 303
        assert ("run_sysid", {}) in ctrl.calls

    def test_post_force(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        status, _, _ = _post(server, "/action/force")
        assert status == 303
        assert ("force_execute", {}) in ctrl.calls

    def test_post_profile_toggle(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        status, _, _ = _post(server, "/action/profile")
        assert status == 303
        assert ("toggle_profile", {}) in ctrl.calls

    def test_post_profile_with_name(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        status, _, _ = _post(server, "/action/profile", {"name": "Away"})
        assert status == 303
        assert ("set_profile", {"name": "Away"}) in ctrl.calls

    def test_post_live_toggle(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        status, _, _ = _post(server, "/action/live")
        assert status == 303
        assert ("toggle_live", {}) in ctrl.calls

    def test_post_live_explicit(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        _post(server, "/action/live", {"mode": "on"})
        _post(server, "/action/live", {"mode": "off"})
        assert ("set_live", {"live": True}) in ctrl.calls
        assert ("set_live", {"live": False}) in ctrl.calls

    def test_post_unknown_action_404(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        status, _, _ = _post(server, "/action/teleport")
        assert status == 404

    def test_json_accept_returns_json(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        status, body, headers = _post(
            server, "/action/run", headers={"Accept": "application/json"}
        )
        assert status == 200
        assert headers.get("content-type", "").startswith("application/json")
        data = json.loads(body)
        assert data["status"] == "started"
        assert data["message"] == "Cycle started"


# ── Method handling ────────────────────────────────────────────────────────


class TestMethodHandling:
    def test_put_405(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        conn = _client(server)
        try:
            conn.request("PUT", "/")
            resp = conn.getresponse()
            assert resp.status == 405
        finally:
            conn.close()

    def test_delete_405(self, server_and_ctrl) -> None:
        server, _ = server_and_ctrl
        conn = _client(server)
        try:
            conn.request("DELETE", "/")
            resp = conn.getresponse()
            assert resp.status == 405
        finally:
            conn.close()


# ── Edge case: button-disabled rendering ───────────────────────────────────


class TestButtonStates:
    def test_force_button_disabled_without_overrides(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        ctrl._overrides = {}
        _, body = _get(server, "/")
        # The Force button form section should contain a disabled attribute
        force_idx = body.find("Force execute")
        assert force_idx > 0
        # Look backwards for the button tag
        button_start = body.rfind("<button", 0, force_idx)
        button_tag = body[button_start:force_idx + len("Force execute")]
        assert "disabled" in button_tag

    def test_force_button_enabled_with_overrides(self, server_and_ctrl) -> None:
        server, ctrl = server_and_ctrl
        ctrl._overrides = {"thermostat_upstairs": "manual"}
        _, body = _get(server, "/")
        force_idx = body.find("Force execute")
        button_start = body.rfind("<button", 0, force_idx)
        button_tag = body[button_start:force_idx + len("Force execute")]
        assert "disabled" not in button_tag
