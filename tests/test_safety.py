"""Tests for infrastructure safety checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from weatherstat.safety import (
    SafetyAlert,
    _check_sustained,
    check_device_health,
    check_thermostat_modes,
    process_safety_alerts,
)
from weatherstat.yaml_config import HealthCheck

# ── Mock decision and latest row ──────────────────────────────────────────


class FakeDecision:
    def __init__(self, upstairs_heating: bool = False, downstairs_heating: bool = False):
        from weatherstat.types import EffectorDecision

        effs: list[EffectorDecision] = []
        if upstairs_heating:
            effs.append(EffectorDecision("thermostat_upstairs", mode="heating"))
        else:
            effs.append(EffectorDecision("thermostat_upstairs", mode="off"))
        if downstairs_heating:
            effs.append(EffectorDecision("thermostat_downstairs", mode="heating"))
        else:
            effs.append(EffectorDecision("thermostat_downstairs", mode="off"))
        self.effectors = tuple(effs)


class FakeRow:
    """Minimal dict-like object mimicking a pandas Series row."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def get(self, key, default=""):
        return self._data.get(key, default)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name, "")


# ── check_thermostat_modes ────────────────────────────────────────────────


class TestCheckThermostatModes:

    def test_no_alert_when_heating_and_thermostat_active(self) -> None:
        latest = FakeRow(thermostat_upstairs_action="heating", thermostat_downstairs_action="idle")
        decision = FakeDecision(upstairs_heating=True, downstairs_heating=False)
        assert check_thermostat_modes(latest, decision) == []

    def test_alert_when_heating_but_thermostat_off(self) -> None:
        latest = FakeRow(thermostat_upstairs_action="off", thermostat_downstairs_action="idle")
        decision = FakeDecision(upstairs_heating=True, downstairs_heating=False)
        alerts = check_thermostat_modes(latest, decision)
        assert len(alerts) == 1
        assert alerts[0].key == "thermostat_upstairs_off"
        assert alerts[0].severity == "critical"

    def test_alert_both_zones_off(self) -> None:
        latest = FakeRow(thermostat_upstairs_action="off", thermostat_downstairs_action="off")
        decision = FakeDecision(upstairs_heating=True, downstairs_heating=True)
        alerts = check_thermostat_modes(latest, decision)
        assert len(alerts) == 2
        keys = {a.key for a in alerts}
        assert keys == {"thermostat_upstairs_off", "thermostat_downstairs_off"}

    def test_no_alert_when_not_heating(self) -> None:
        latest = FakeRow(thermostat_upstairs_action="off", thermostat_downstairs_action="off")
        decision = FakeDecision(upstairs_heating=False, downstairs_heating=False)
        assert check_thermostat_modes(latest, decision) == []

    def test_no_alert_when_thermostat_idle(self) -> None:
        """Idle means on but not calling for heat — not a problem."""
        latest = FakeRow(thermostat_upstairs_action="idle", thermostat_downstairs_action="idle")
        decision = FakeDecision(upstairs_heating=True, downstairs_heating=True)
        assert check_thermostat_modes(latest, decision) == []


# ── check_device_health ──────────────────────────────────────────────────


class TestCheckDeviceHealth:

    @staticmethod
    def _mock_ha_response(state_val: str, status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"state": state_val}
        return resp

    @staticmethod
    def _mock_by_entity(entity_responses: dict[str, dict], *, history_readings: dict[str, list[tuple[float, str]]] | None = None) -> callable:
        """Return a side_effect function that returns different responses per entity.

        Values can be a plain string (state only) or a dict with ``state`` and
        optional ``last_changed`` keys.

        ``history_readings``, if provided, maps entity IDs to lists of
        ``(minutes_ago, state_value)`` pairs for the HA history API
        (used by sustained threshold checks).
        """
        def _get(url: str, **_kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200

            # History API: /api/history/period/<timestamp>?filter_entity_id=...
            if "/api/history/period/" in url:
                params = _kwargs.get("params", {})
                eid = params.get("filter_entity_id", "")
                raw = (history_readings or {}).get(eid, [])
                now = datetime.now(UTC)
                entries = [
                    {"state": val, "last_changed": (now - timedelta(minutes=m)).isoformat()}
                    for m, val in raw
                ]
                resp.json.return_value = [entries]
                return resp

            # State API: /api/states/<entity_id>
            entity_id = url.rsplit("/", 1)[-1]
            info = entity_responses.get(entity_id)
            if info is None:
                resp.json.return_value = {"state": "unknown"}
            elif isinstance(info, str):
                resp.json.return_value = {"state": info}
            else:
                resp.json.return_value = info
            return resp
        return _get

    @staticmethod
    def _heating_since(minutes_ago: float) -> str:
        """Return an ISO timestamp *minutes_ago* minutes in the past."""
        return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()

    def test_no_alert_when_healthy(self) -> None:
        """Connection OK, outlet healthy while heating long enough → no alert."""
        states = {
            "binary_sensor.combi_connection_status": "on",
            "sensor.combi_heating_mode": {"state": "Space Heating", "last_changed": self._heating_since(10)},
            "sensor.combi_outlet_temp": "150.0",
        }
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
            alerts = check_device_health()
        assert len(alerts) == 0

    def test_alert_when_outlet_low_and_heating(self) -> None:
        """Outlet temp below threshold while heating for 5 min → critical alert."""
        states = {
            "binary_sensor.combi_connection_status": "on",
            "sensor.combi_heating_mode": {"state": "Space Heating", "last_changed": self._heating_since(5)},
            "sensor.combi_outlet_temp": "100.0",
        }
        # Sustained check: low since start of window (single transition held 15 min).
        history = {"sensor.combi_outlet_temp": [(15, "100")]}
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states, history_readings=history)):
            alerts = check_device_health()
        assert len(alerts) == 1
        assert alerts[0].key == "combi_outlet_fault"
        assert alerts[0].severity == "critical"

    def test_no_alert_when_not_heating(self) -> None:
        """Outlet temp low but boiler is idle → when condition fails, no alert."""
        states = {
            "binary_sensor.combi_connection_status": "on",
            "sensor.combi_heating_mode": {"state": "Idle", "last_changed": self._heating_since(30)},
            "sensor.combi_outlet_temp": "80.0",
        }
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
            alerts = check_device_health()
        assert len(alerts) == 0

    def test_no_alert_when_heating_too_briefly(self) -> None:
        """Outlet low but heating started <3 min ago → within for_minutes grace, no alert."""
        states = {
            "binary_sensor.combi_connection_status": "on",
            "sensor.combi_heating_mode": {"state": "Space Heating", "last_changed": self._heating_since(1)},
            "sensor.combi_outlet_temp": "80.0",
        }
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
            alerts = check_device_health()
        assert len(alerts) == 0

    def test_alert_when_connection_lost(self) -> None:
        """Connection status off → critical alert."""
        states = {
            "binary_sensor.combi_connection_status": "off",
            "sensor.combi_heating_mode": {"state": "Idle", "last_changed": self._heating_since(30)},
            "sensor.combi_outlet_temp": "120.0",
        }
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
            alerts = check_device_health()
        assert len(alerts) == 1
        assert alerts[0].key == "combi_connection_fault"
        assert alerts[0].severity == "critical"
        assert "connection" in alerts[0].message.lower()

    def test_alert_when_unavailable(self) -> None:
        """Sensor state 'unavailable' → warning alert."""
        with patch("weatherstat.safety.requests.get", return_value=self._mock_ha_response("unavailable")):
            alerts = check_device_health()
        # Both checks see 'unavailable' (when condition entity also returns unavailable,
        # which doesn't match "Space Heating", so the outlet check is skipped — only
        # connection check produces an unavailable warning).
        assert len(alerts) >= 1
        assert all(a.severity == "warning" for a in alerts)

    def test_ha_failure_does_not_crash(self) -> None:
        """HTTP error → no alert, no crash."""
        with patch("weatherstat.safety.requests.get", side_effect=Exception("timeout")):
            alerts = check_device_health()
        assert len(alerts) == 0

    def test_ha_404_does_not_crash(self) -> None:
        """Non-200 response → no alert."""
        with patch("weatherstat.safety.requests.get", return_value=self._mock_ha_response("", 404)):
            alerts = check_device_health()
        assert len(alerts) == 0


# ── _check_sustained ────────────────────────────────────────────────────


class TestCheckSustained:
    """Tests for the M-of-N sustained threshold check with 1-minute resampling."""

    @staticmethod
    def _make_check(
        min_value: float | None = None,
        max_value: float | None = None,
        sustain_minutes: float = 15,
        sustain_samples: int = 12,
    ) -> HealthCheck:
        return HealthCheck(
            name="test_outlet",
            entity_id="sensor.test_outlet_temp",
            min_value=min_value,
            max_value=max_value,
            sustain_minutes=sustain_minutes,
            sustain_samples=sustain_samples,
        )

    @staticmethod
    def _mock_history_resp(
        entries: list[tuple[float, str]],
        *,
        sustain_minutes: float = 15,
        status_code: int = 200,
    ) -> MagicMock:
        """Build a mock HA history response with timestamped entries.

        ``entries`` is a list of (minutes_ago, state_value) pairs.  Timestamps
        are computed relative to "now" which aligns with ``_check_sustained``'s
        own ``datetime.now(UTC)`` call.
        """
        now = datetime.now(UTC)
        resp = MagicMock()
        resp.status_code = status_code
        if status_code != 200:
            return resp
        history = []
        for mins_ago, val in entries:
            ts = (now - timedelta(minutes=mins_ago)).isoformat()
            history.append({"state": val, "last_changed": ts})
        resp.json.return_value = [history]
        return resp

    def test_sustained_low_triggers(self) -> None:
        """Single low reading at start of window held for 15 min → 15 samples, all violating."""
        check = self._make_check(min_value=110, sustain_samples=12, sustain_minutes=15)
        # One state change at the start: 100°F held for the entire window.
        entries = [(15, "100")]
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp(entries)):
            assert _check_sustained(check) is True

    def test_transient_dip_does_not_trigger(self) -> None:
        """30-second purge dip then recovery → only ~1 minute sample violates."""
        check = self._make_check(min_value=110, sustain_samples=12, sustain_minutes=15)
        # Healthy at start, dip at 3 min ago for 30 seconds, then recovery.
        entries = [(15, "150"), (3.0, "80"), (2.5, "150")]
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp(entries)):
            assert _check_sustained(check) is False

    def test_sawtooth_recovery_does_not_trigger(self) -> None:
        """10-minute sawtooth with recovery → not enough sustained violation."""
        check = self._make_check(min_value=110, sustain_samples=12, sustain_minutes=15)
        # Healthy first 5 min, dip for 5 min, recover for last 5 min.
        entries = [(15, "150"), (10, "90"), (5, "150")]
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp(entries)):
            assert _check_sustained(check) is False

    def test_12_of_15_minutes_low(self) -> None:
        """Low for 12+ minutes out of 15 → triggers."""
        check = self._make_check(min_value=110, sustain_samples=12, sustain_minutes=15)
        # Healthy for first 2 min, then low for remaining 13 min.
        entries = [(15, "150"), (13, "90")]
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp(entries)):
            assert _check_sustained(check) is True

    def test_11_of_15_minutes_low(self) -> None:
        """Low for 11 minutes out of 15 → does not trigger (need 12)."""
        check = self._make_check(min_value=110, sustain_samples=12, sustain_minutes=15)
        # Healthy for first 4 min, then low for remaining 11 min.
        entries = [(15, "150"), (11, "90")]
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp(entries)):
            assert _check_sustained(check) is False

    def test_max_value_sustained(self) -> None:
        """max_value threshold sustained for full window → True."""
        check = self._make_check(max_value=200, sustain_samples=3, sustain_minutes=5)
        entries = [(5, "210")]
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp(entries, sustain_minutes=5)):
            assert _check_sustained(check) is True

    def test_no_data_before_first_sample(self) -> None:
        """State change only in the last minute → most samples have no value."""
        check = self._make_check(min_value=110, sustain_samples=12, sustain_minutes=15)
        entries = [(0.5, "90")]  # only 0.5 min ago
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp(entries)):
            assert _check_sustained(check) is False

    def test_ha_error_returns_false(self) -> None:
        """HA history API failure → don't alert."""
        check = self._make_check(min_value=110, sustain_samples=5)
        with patch("weatherstat.safety.requests.get", return_value=self._mock_history_resp([], status_code=500)):
            assert _check_sustained(check) is False

    def test_empty_history_returns_false(self) -> None:
        """No history data → don't alert."""
        check = self._make_check(min_value=110, sustain_samples=5)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [[]]
        with patch("weatherstat.safety.requests.get", return_value=resp):
            assert _check_sustained(check) is False


# ── process_safety_alerts ────────────────────────────────────────────────


class TestProcessSafetyAlerts:

    def test_empty_alerts(self) -> None:
        assert process_safety_alerts([], live=False) == []

    def test_alert_dispatched_dry_run(self, capsys) -> None:
        alert = SafetyAlert(
            key="thermostat_upstairs_off",
            title="Upstairs thermostat is off",
            message="Turn it on.",
            severity="critical",
        )
        result = process_safety_alerts([alert], live=False)
        assert len(result) == 1
        output = capsys.readouterr().out
        assert "CRITICAL" in output
        assert "Upstairs thermostat is off" in output

    def test_multiple_alerts(self) -> None:
        alerts = [
            SafetyAlert(key="thermostat_upstairs_off", title="T1", message="m1"),
            SafetyAlert(key="combi_fault", title="T2", message="m2"),
        ]
        result = process_safety_alerts(alerts, live=False)
        assert len(result) == 2
