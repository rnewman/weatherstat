"""Tests for infrastructure safety checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from weatherstat.safety import (
    SafetyAlert,
    check_device_health,
    check_thermostat_modes,
    process_safety_alerts,
)

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
    def _mock_by_entity(entity_responses: dict[str, dict]) -> callable:
        """Return a side_effect function that returns different responses per entity.

        Values can be a plain string (state only) or a dict with ``state`` and
        optional ``last_changed`` keys.
        """
        def _get(url: str, **_kwargs: object) -> MagicMock:
            entity_id = url.rsplit("/", 1)[-1]
            info = entity_responses.get(entity_id)
            resp = MagicMock()
            resp.status_code = 200
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
        return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()

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
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
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
