"""Tests for TUIController.

The controller is exercised in isolation against a fake `app` (no Textual
required). Worker callables run synchronously so we can observe state
transitions deterministically.
"""

from __future__ import annotations

import dataclasses
import threading
from unittest.mock import patch

import pytest

from weatherstat.tui.controller import (
    ActionResult,
    StatusSnapshot,
    TUIController,
)
from weatherstat.yaml_config import load_config


def _config_with_comfort_entity(entity_id: str | None):
    """Return a copy of the test sandbox config with a different comfort_entity."""
    return dataclasses.replace(load_config(), comfort_entity=entity_id)


# ── Test scaffolding ────────────────────────────────────────────────────────


class _Capture:
    """Collects log lines and pushed snapshots for assertions."""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.snapshots: list[StatusSnapshot] = []

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def on_snapshot(self, snap: StatusSnapshot) -> None:
        self.snapshots.append(snap)


def _sync_worker(fn):
    """Run worker callables on the calling thread."""
    fn()


def _make_controller(*, live: bool = False, worker=_sync_worker) -> tuple[TUIController, _Capture]:
    cap = _Capture()
    ctrl = TUIController(
        live=live,
        log=cap.log,
        worker=worker,
        on_snapshot=cap.on_snapshot,
        overlay_triggers_cycle=False,
    )
    return ctrl, cap


# ── Snapshot ────────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_shape(self) -> None:
        ctrl, _ = _make_controller()
        snap = ctrl.build_snapshot()
        assert isinstance(snap, StatusSnapshot)
        assert snap.live is False
        assert snap.cycle_running is False
        assert snap.sysid_running is False
        assert isinstance(snap.temps, dict)
        assert isinstance(snap.environment, tuple)
        assert isinstance(snap.opportunities, list)
        assert isinstance(snap.warnings, list)
        # Snapshot is cached
        assert ctrl.latest_snapshot() is snap

    def test_publish_snapshot_invokes_callback(self) -> None:
        ctrl, cap = _make_controller()
        ctrl.publish_snapshot()
        assert len(cap.snapshots) == 1


# ── Live mode toggling ──────────────────────────────────────────────────────


class TestToggleLive:
    def test_toggle_no_confirmation(self) -> None:
        ctrl, _ = _make_controller(live=False)
        result = ctrl.toggle_live()
        assert result.status == "ok"
        assert ctrl.live_mode is True
        result2 = ctrl.toggle_live()
        assert ctrl.live_mode is False
        assert result2.detail["live"] is False

    def test_set_live_idempotent(self) -> None:
        ctrl, _ = _make_controller(live=True)
        result = ctrl.set_live(True)
        assert result.status == "noop"

    def test_set_live_publishes_snapshot(self) -> None:
        ctrl, cap = _make_controller(live=False)
        cap.snapshots.clear()
        ctrl.set_live(True)
        assert any(s.live for s in cap.snapshots)


# ── Run cycle ───────────────────────────────────────────────────────────────


class TestRunCycle:
    def test_run_cycle_sets_running_flag(self) -> None:
        ctrl, cap = _make_controller()
        with patch("weatherstat.control.run_control_cycle") as mock_run:
            mock_run.return_value = None  # decision-less cycle
            result = ctrl.run_cycle()
        assert result.status == "started"
        # After sync worker completes the flag should be back to False
        assert ctrl.is_cycle_running() is False
        assert mock_run.call_count == 1

    def test_run_cycle_idempotent(self) -> None:
        # Use a worker that defers execution so we can observe mid-flight state
        deferred: list = []

        def deferred_worker(fn):
            deferred.append(fn)

        ctrl, _ = _make_controller(worker=deferred_worker)
        first = ctrl.run_cycle()
        assert first.status == "started"
        assert ctrl.is_cycle_running() is True

        second = ctrl.run_cycle()
        assert second.status == "already_running"
        assert len(deferred) == 1

        # Drain the deferred worker (mocked control cycle)
        with patch("weatherstat.control.run_control_cycle", return_value=None):
            deferred[0]()
        assert ctrl.is_cycle_running() is False

    def test_run_cycle_handles_exception(self) -> None:
        ctrl, cap = _make_controller()
        with patch("weatherstat.control.run_control_cycle", side_effect=RuntimeError("boom")):
            ctrl.run_cycle()
        assert ctrl.is_cycle_running() is False
        assert any("boom" in line for line in cap.logs)


# ── Sysid ───────────────────────────────────────────────────────────────────


class TestRunSysid:
    def test_run_sysid_idempotent(self) -> None:
        deferred: list = []

        def deferred_worker(fn):
            deferred.append(fn)

        ctrl, _ = _make_controller(worker=deferred_worker)
        assert ctrl.run_sysid().status == "started"
        assert ctrl.run_sysid().status == "already_running"


# ── Force execute ───────────────────────────────────────────────────────────


class TestForceExecute:
    def test_force_execute_no_overrides(self) -> None:
        ctrl, _ = _make_controller()
        result = ctrl.force_execute()
        assert result.status == "noop"
        assert "overrides" in result.message.lower()

    def test_force_execute_with_overrides(self) -> None:
        ctrl, _ = _make_controller()
        # Inject overrides as if a previous executor run had detected them
        with ctrl._lock:
            ctrl._overrides = {"thermostat_upstairs": "manual setpoint"}

        with patch("weatherstat.executor.execute") as mock_exec:
            mock_exec.return_value = type("R", (), {"overrides": {}})()
            result = ctrl.force_execute()
        assert result.status == "started"
        assert result.detail["effectors"] == ["thermostat_upstairs"]
        assert mock_exec.called
        # Overrides cleared after the executor reports none remaining
        assert ctrl.overrides() == {}


# ── Profile toggle (regression for the comfort_entity hardcode bug) ────────


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class TestToggleProfile:
    def test_no_comfort_entity_returns_error(self) -> None:
        ctrl, _ = _make_controller()
        cfg = _config_with_comfort_entity(None)
        with patch("weatherstat.yaml_config.load_config", return_value=cfg):
            result = ctrl.toggle_profile()
        assert result.status == "error"
        assert "comfort_entity" in result.message

    def test_uses_configured_comfort_entity(self) -> None:
        """Regression: hardcode of input_select.thermostat_mode is gone."""
        ctrl, cap = _make_controller()
        cfg = _config_with_comfort_entity("input_select.house_mode")

        get_calls: list[str] = []
        post_calls: list[dict] = []

        def fake_get(url, **_kwargs):
            get_calls.append(url)
            return _Resp({
                "state": "Home",
                "attributes": {"options": ["Home", "Away"]},
            })

        def fake_post(url, json=None, **_kwargs):
            post_calls.append({"url": url, "json": json})
            return _Resp({})

        with (
            patch("weatherstat.yaml_config.load_config", return_value=cfg),
            patch.dict("os.environ", {"HA_URL": "http://ha.test", "HA_TOKEN": "tok"}),
            patch("requests.get", side_effect=fake_get),
            patch("requests.post", side_effect=fake_post),
        ):
            result = ctrl.toggle_profile()

        assert result.status == "ok", f"got {result.status}: {result.message} | logs: {cap.logs}"
        assert any("input_select.house_mode" in url for url in get_calls)
        assert post_calls
        assert post_calls[0]["json"]["entity_id"] == "input_select.house_mode"
        assert post_calls[0]["json"]["option"] == "Away"  # cycled from Home

    def test_set_profile_explicit_name(self) -> None:
        ctrl, _ = _make_controller()
        cfg = _config_with_comfort_entity("input_select.house_mode")
        post_calls: list[dict] = []

        def fake_get(_url, **_kwargs):
            return _Resp({"state": "Home", "attributes": {"options": ["Home", "Away"]}})

        def fake_post(url, json=None, **_kwargs):
            post_calls.append({"url": url, "json": json})
            return _Resp({})

        with (
            patch("weatherstat.yaml_config.load_config", return_value=cfg),
            patch.dict("os.environ", {"HA_URL": "http://ha.test", "HA_TOKEN": "tok"}),
            patch("requests.get", side_effect=fake_get),
            patch("requests.post", side_effect=fake_post),
        ):
            result = ctrl.set_profile("Away")

        assert result.status == "ok"
        assert post_calls[0]["json"]["option"] == "Away"

    def test_set_profile_unknown_returns_error(self) -> None:
        ctrl, _ = _make_controller()
        cfg = _config_with_comfort_entity("input_select.house_mode")

        with (
            patch("weatherstat.yaml_config.load_config", return_value=cfg),
            patch.dict("os.environ", {"HA_URL": "http://ha.test", "HA_TOKEN": "tok"}),
            patch("requests.get", return_value=_Resp({
                "state": "Home", "attributes": {"options": ["Home", "Away"]},
            })),
            patch("requests.post", return_value=_Resp({})),
        ):
            result = ctrl.set_profile("Vacation")
        assert result.status == "error"
        assert "Vacation" in result.message


# ── Threading sanity ────────────────────────────────────────────────────────


class TestThreading:
    def test_concurrent_run_cycle_only_one_starts(self) -> None:
        """Two threads racing to start a cycle: only one wins."""
        ctrl, _ = _make_controller(worker=lambda fn: None)  # never run worker
        results: list[ActionResult] = []
        barrier = threading.Barrier(2)

        def run() -> None:
            barrier.wait()
            results.append(ctrl.run_cycle())

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        statuses = sorted(r.status for r in results)
        assert statuses == ["already_running", "started"]


# ── Comfort overlays ────────────────────────────────────────────────────────


@pytest.fixture
def overlay_sandbox(tmp_path, monkeypatch):
    """Point the overlay state file and event log at a per-test directory."""
    import weatherstat.decision_log as dl
    import weatherstat.overlay as ov

    monkeypatch.setattr(ov, "OVERLAY_STATE_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(dl, "DECISION_LOG_DB", tmp_path / "decision_log.db")
    return tmp_path


class TestOverlays:
    def test_bump_then_snapshot_shifts_bounds(self, overlay_sandbox) -> None:
        ctrl, cap = _make_controller()
        before = ctrl.build_snapshot().comfort["bedroom"]

        result = ctrl.bump_overlay("bedroom", 2.0, source="web")
        assert result.status == "ok", result.message
        assert result.detail["delta"] == 2.0
        assert any("[overlay]" in line and "bedroom +2.0" in line for line in cap.logs)

        snap = ctrl.build_snapshot()
        after = snap.comfort["bedroom"]
        assert after.preferred_lo == before.preferred_lo + 2.0
        assert after.acceptable_hi == before.acceptable_hi + 2.0
        assert snap.overlays["bedroom_temp"].delta == 2.0
        assert snap.overlays["bedroom_temp"].source == "web"
        # Unrelated rooms untouched
        assert snap.comfort["office"] == ctrl.build_snapshot().comfort["office"]

    def test_overlay_command_and_clear(self, overlay_sandbox) -> None:
        ctrl, _ = _make_controller()
        assert ctrl.overlay_command("office -1 2h").status == "ok"
        assert ctrl.build_snapshot().overlays["office_temp"].delta == -1.0
        result = ctrl.clear_overlay("office")
        assert result.status == "ok"
        assert "office_temp" not in ctrl.build_snapshot().overlays

    def test_bad_command_is_error(self, overlay_sandbox) -> None:
        ctrl, cap = _make_controller()
        result = ctrl.overlay_command("attic +1")
        assert result.status == "error"
        assert "unknown sensor" in result.message
        assert any("[overlay]" in line and "attic" in line for line in cap.logs)

    def test_bump_triggers_cycle_when_enabled(self, overlay_sandbox) -> None:
        cap = _Capture()
        deferred: list = []
        ctrl = TUIController(
            live=False, log=cap.log, worker=deferred.append, on_snapshot=cap.on_snapshot,
            overlay_triggers_cycle=True,
        )
        ctrl.bump_overlay("bedroom", 1.0)
        assert ctrl.is_cycle_running() is True
        assert len(deferred) == 1


class TestOverlayListeners:
    def test_listener_called_after_change(self, overlay_sandbox) -> None:
        from weatherstat.overlay import OverlayCommand

        ctrl, _ = _make_controller()
        calls: list[int] = []
        ctrl.add_overlay_listener(lambda: calls.append(1))
        result = ctrl.apply_overlay_command(OverlayCommand("bedroom_temp", "set", 2.0), source="ha")
        assert result.status == "ok", result.message
        assert calls == [1]
        assert ctrl.build_snapshot().overlays["bedroom_temp"].source == "ha"

    def test_listener_error_is_logged(self, overlay_sandbox) -> None:
        ctrl, cap = _make_controller()

        def boom() -> None:
            raise RuntimeError("broker gone")

        ctrl.add_overlay_listener(boom)
        assert ctrl.bump_overlay("bedroom", 1.0).status == "ok"
        assert any("broker gone" in line for line in cap.logs)
