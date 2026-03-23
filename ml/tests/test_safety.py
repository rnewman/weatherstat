"""Tests for infrastructure safety checks."""

from __future__ import annotations

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
    def _mock_by_entity(entity_states: dict[str, str]) -> callable:
        """Return a side_effect function that returns different states per entity."""
        def _get(url: str, **_kwargs: object) -> MagicMock:
            entity_id = url.rsplit("/", 1)[-1]
            state = entity_states.get(entity_id, "unknown")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"state": state}
            return resp
        return _get

    def test_no_alert_when_healthy(self) -> None:
        """All checks pass → no alert."""
        states = {
            "binary_sensor.navien_navien_connection_status": "on",
            "sensor.navien_navien_sh_return_temp": "120.0",
        }
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
            alerts = check_device_health()
        assert len(alerts) == 0

    def test_alert_when_below_min(self) -> None:
        """Sensor value at or below min_value → critical alert."""
        states = {
            "binary_sensor.navien_navien_connection_status": "on",
            "sensor.navien_navien_sh_return_temp": "32.0",
        }
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
            alerts = check_device_health()
        assert len(alerts) == 1
        assert alerts[0].key == "navien_return_fault"
        assert alerts[0].severity == "critical"

    def test_alert_when_connection_lost(self) -> None:
        """Connection status off → critical alert."""
        states = {
            "binary_sensor.navien_navien_connection_status": "off",
            "sensor.navien_navien_sh_return_temp": "120.0",
        }
        with patch("weatherstat.safety.requests.get", side_effect=self._mock_by_entity(states)):
            alerts = check_device_health()
        assert len(alerts) == 1
        assert alerts[0].key == "navien_connection_fault"
        assert alerts[0].severity == "critical"
        assert "connection" in alerts[0].message.lower()

    def test_alert_when_unavailable(self) -> None:
        """Sensor state 'unavailable' → warning alert."""
        with patch("weatherstat.safety.requests.get", return_value=self._mock_ha_response("unavailable")):
            alerts = check_device_health()
        # Both checks see 'unavailable'
        assert len(alerts) == 2
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
            SafetyAlert(key="navien_fault", title="T2", message="m2"),
        ]
        result = process_safety_alerts(alerts, live=False)
        assert len(result) == 2
