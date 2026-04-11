"""Tests for physics-based window opportunities.

Tests opportunity evaluation (physics simulator-backed), opportunity
lifecycle (persist, expire, dismiss), and notification dispatch.
"""

from __future__ import annotations

import json

import pytest

from weatherstat.simulator import SimParams, TauModel, load_sim_params
from weatherstat.types import (
    ComfortSchedule,
    ComfortScheduleEntry,
    EffectorDecision,
    Scenario,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sim_params():
    """Load sim params with synthetic environment betas for advisory tests.

    Sysid may not learn environment betas if there's insufficient data.
    Advisory tests need environment effects to exist so toggling a device
    changes simulation output.
    """
    params = load_sim_params()
    # Inject environment betas: each sensor with a matching environment entry gets a beta
    from weatherstat.yaml_config import load_config
    cfg = load_config()
    env_col_to_name = {entry.column: name for name, entry in cfg.environment.items()}
    augmented_taus: dict[str, TauModel] = {}
    for sensor, tau_model in params.taus.items():
        env_cols = cfg.environment_columns_for_sensor(sensor)
        if env_cols and not tau_model.environment_tau_betas:
            # Inject a reasonable beta: 1/tau_eff ≈ 1/tau_base + beta
            # beta = 0.02 → effective tau ≈ 25h (from 45h base)
            env_name = env_col_to_name.get(env_cols[0], env_cols[0])
            augmented_taus[sensor] = TauModel(
                tau_base=tau_model.tau_base,
                environment_tau_betas={env_name: 0.02},
            )
        else:
            augmented_taus[sensor] = tau_model
    return SimParams(
        taus=augmented_taus,
        gains=params.gains,
        solar=params.solar,
        sensors=params.sensors,
        effectors=params.effectors,
        solar_elevation_gains=params.solar_elevation_gains,
    )


_CURRENT_TEMPS = {
    "upstairs": 70.0, "downstairs": 69.0, "bedroom": 68.5,
    "office": 67.0, "family_room": 69.5, "kitchen": 68.0,
    "piano": 67.5, "bathroom": 68.0, "living_room": 69.0,
}


def _all_off() -> Scenario:
    return Scenario(effectors={})


def _both_on() -> Scenario:
    return Scenario(effectors={
        "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating"),
        "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating"),
    })


def _make_schedules(**overrides: list[ComfortScheduleEntry]) -> list[ComfortSchedule]:
    """Comfort schedules with optional room overrides."""
    from weatherstat.control import default_comfort_schedules

    defaults = {s.label: s for s in default_comfort_schedules()}
    for label, entries in overrides.items():
        defaults[label] = ComfortSchedule(sensor=f"{label}_temp", label=label, entries=tuple(entries))
    return list(defaults.values())


# ── Solar elevation regression test ──────────────────────────────────────
#
# Regression test for the bug where the (now-deleted) per-window re-sweep
# dropped solar_elevations when constructing the toggled HouseState, causing
# the simulator to fall back from elevation-based solar (significant gains)
# to the empty legacy per-hour model. Every window toggle appeared to cool
# rooms down, producing spurious opportunities.


class TestSolarElevationNotDropped:
    """Toggling a no-beta window must not change predictions when solar is active."""

    def test_no_beta_window_zero_benefit(self) -> None:
        """Window with no sysid beta on any sensor produces zero comfort benefit.

        Minimal unit test: two sensors, one window (no betas), solar_elevation_gains
        present. Toggling the window must not change predictions.
        """
        from weatherstat.control import CONTROL_HORIZONS
        from weatherstat.simulator import HouseState, SimParams, TauModel, predict

        # Two sensors, no window betas, significant solar elevation gains
        sensors = ["room_a_temp", "room_b_temp"]
        params = SimParams(
            taus={s: TauModel(tau_base=45.0) for s in sensors},
            gains={},
            solar={},
            sensors=sensors,
            effectors=[],
            solar_elevation_gains={"room_a_temp": 3.0, "room_b_temp": 2.0},
        )

        # Daytime solar elevations (simulating sunny afternoon)
        n_steps = max(CONTROL_HORIZONS)
        solar_elevs = [0.6] * n_steps  # constant for simplicity

        temps = {"room_a_temp": 74.0, "room_b_temp": 73.0}
        state = HouseState(
            current_temps=temps,
            outdoor_temp=60.0,
            forecast_temps=[60.0] * 12,
            environment_states={"fake_window": False},
            hour_of_day=14.0,
            solar_fractions=[1.0] * 12,
            solar_elevations=solar_elevs,
        )

        scenario = Scenario(effectors={})

        # Baseline prediction
        target_names, baseline_preds = predict(state, [scenario], params, CONTROL_HORIZONS)
        baseline_dict = {t: float(baseline_preds[0, j]) for j, t in enumerate(target_names)}

        # Toggled prediction — same state except window open
        toggled_state = HouseState(
            current_temps=temps,
            outdoor_temp=60.0,
            forecast_temps=[60.0] * 12,
            environment_states={"fake_window": True},
            hour_of_day=14.0,
            solar_fractions=[1.0] * 12,
            solar_elevations=solar_elevs,  # critical: must be preserved
        )
        _, toggled_preds = predict(toggled_state, [scenario], params, CONTROL_HORIZONS)
        toggled_dict = {t: float(toggled_preds[0, j]) for j, t in enumerate(target_names)}

        # With no window betas, predictions must be identical
        for key in baseline_dict:
            assert baseline_dict[key] == pytest.approx(toggled_dict[key], abs=1e-6), (
                f"Prediction for {key} changed when toggling no-beta window: "
                f"{baseline_dict[key]:.4f} → {toggled_dict[key]:.4f}"
            )

    def test_house_state_requires_solar_fractions(self) -> None:
        """HouseState raises ValueError if solar_fractions is empty."""
        from weatherstat.simulator import HouseState

        with pytest.raises(ValueError, match="solar_fractions"):
            HouseState(
                current_temps={"x": 70.0},
                outdoor_temp=60.0,
                forecast_temps=[60.0],
                environment_states={},
                hour_of_day=12.0,
            )

    def test_house_state_requires_solar_elevations(self) -> None:
        """HouseState raises ValueError if solar_elevations is empty."""
        from weatherstat.simulator import HouseState

        with pytest.raises(ValueError, match="solar_elevations"):
            HouseState(
                current_temps={"x": 70.0},
                outdoor_temp=60.0,
                forecast_temps=[60.0],
                environment_states={},
                hour_of_day=12.0,
                solar_fractions=[1.0],
            )


# ── Opportunity state tests ──────────────────────────────────────────────


class TestAdvisoryState:
    """Test persistent advisory state management."""

    def test_load_empty(self, tmp_path, monkeypatch) -> None:
        """Missing state file → empty cooldowns/opportunities/warnings."""
        from weatherstat.advisory import AdvisoryState

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        state = AdvisoryState.load()
        assert state.cooldowns == {}
        assert state.opportunities == []
        assert state.warnings == []

    def test_load_existing_format(self, tmp_path, monkeypatch) -> None:
        """State file with cooldowns + opportunities + warnings round-trips."""
        from weatherstat.advisory import AdvisoryState

        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "cooldowns": {"opportunity_bedroom": 1000.0},
            "opportunities": [{"device": "bedroom", "action": "open", "cost_delta": -2.0}],
            "warnings": [{"message": "test"}],
        }))
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        state = AdvisoryState.load()
        assert state.cooldowns["opportunity_bedroom"] == 1000.0
        assert state.opportunities[0]["device"] == "bedroom"
        assert state.warnings[0]["message"] == "test"

    def test_save_and_reload(self, tmp_path, monkeypatch) -> None:
        """Round-trip save and load."""
        from weatherstat.advisory import AdvisoryState

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        state = AdvisoryState(
            cooldowns={"opportunity_piano": 5000.0},
            opportunities=[{"device": "piano", "cost_delta": -1.5}],
            warnings=[{"message": "breach"}],
        )
        state.save()
        loaded = AdvisoryState.load()
        assert loaded.cooldowns == state.cooldowns
        assert loaded.opportunities == state.opportunities
        assert loaded.warnings == state.warnings


# ── Opportunity lifecycle tests ──────────────────────────────────────────


class TestProcessAdvisoryPlan:
    """Test the unified advisory pipeline: persistence + notifications."""

    @staticmethod
    def _plan(*opps, breaches: tuple[str, ...] = ()):
        from weatherstat.types import AdvisoryPlan

        return AdvisoryPlan(
            baseline_idx=0,
            baseline_cost=5.0,
            opportunities=tuple(opps),
            backup_breaches=breaches,
        )

    @staticmethod
    def _opp(device: str, cost_delta: float, action: str = "open", **adv_kwargs):
        from weatherstat.types import AdvisoryDecision, DeviceOpportunity

        return DeviceOpportunity(
            device=device,
            current_state=False,
            advisory=AdvisoryDecision(device, action=action, **adv_kwargs),
            idx=1,
            cost_delta=cost_delta,
        )

    def test_beneficial_opportunity_persisted(self, tmp_path, monkeypatch) -> None:
        """Beneficial opportunity is written to the state file in live mode."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        plan = self._plan(self._opp("bedroom", -1.0))

        process_advisory_plan(plan, live=True, current_hour=12)

        data = json.loads(state_file.read_text())
        assert len(data["opportunities"]) == 1
        assert data["opportunities"][0]["device"] == "bedroom"
        assert data["opportunities"][0]["cost_delta"] == -1.0

    def test_non_beneficial_dropped(self, tmp_path, monkeypatch) -> None:
        """Opportunities with cost_delta >= 0 are not persisted."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        plan = self._plan(self._opp("bedroom", 0.5))  # worse than baseline

        process_advisory_plan(plan, live=True, current_hour=12)

        data = json.loads(state_file.read_text())
        assert data["opportunities"] == []

    def test_dry_run_no_save(self, tmp_path, monkeypatch) -> None:
        """Dry run does not write the state file."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        plan = self._plan(self._opp("bedroom", -2.0))

        process_advisory_plan(plan, live=False, current_hour=12)
        assert not state_file.exists()

    def test_below_notification_threshold_not_notified(self, tmp_path, monkeypatch) -> None:
        """Opportunity below |cost_delta| threshold is tracked but does not notify."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        # |cost_delta| = 0.5 < default notification_threshold of 1.5
        plan = self._plan(self._opp("bedroom", -0.5))
        sent: list[tuple] = []
        monkeypatch.setattr(
            "weatherstat.advisory.send_ha_notification",
            lambda *a, **kw: sent.append((a, kw)) or True,
        )

        process_advisory_plan(plan, live=True, current_hour=12)

        # Tracked
        data = json.loads(state_file.read_text())
        assert len(data["opportunities"]) == 1
        # Not notified
        assert sent == []
        assert "opportunity_bedroom" not in data["cooldowns"]

    def test_above_notification_threshold_notified(self, tmp_path, monkeypatch) -> None:
        """Opportunity above threshold sends a single rolled-up notification."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        plan = self._plan(self._opp("bedroom", -2.0))
        sent: list[tuple] = []
        monkeypatch.setattr(
            "weatherstat.advisory.send_ha_notification",
            lambda *a, **kw: sent.append((a, kw)) or True,
        )

        process_advisory_plan(plan, live=True, current_hour=12)

        assert len(sent) == 1
        data = json.loads(state_file.read_text())
        assert "opportunity_bedroom" in data["cooldowns"]

    def test_quiet_hours_suppress_notification(self, tmp_path, monkeypatch) -> None:
        """Quiet hours suppress notifications even when above threshold."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        plan = self._plan(self._opp("bedroom", -5.0))
        sent: list[tuple] = []
        monkeypatch.setattr(
            "weatherstat.advisory.send_ha_notification",
            lambda *a, **kw: sent.append((a, kw)) or True,
        )

        process_advisory_plan(plan, live=True, current_hour=23)

        # Tracked but not notified
        data = json.loads(state_file.read_text())
        assert len(data["opportunities"]) == 1
        assert sent == []

    def test_cooldown_suppresses_repeat_notification(self, tmp_path, monkeypatch) -> None:
        """Recently-notified opportunity does not re-notify within cooldown."""
        import time as _time

        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        # Pre-populate cooldown timestamp from now
        state_file.write_text(json.dumps({
            "cooldowns": {"opportunity_bedroom": _time.time()},
            "opportunities": [],
            "warnings": [],
        }))
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        plan = self._plan(self._opp("bedroom", -3.0))
        sent: list[tuple] = []
        monkeypatch.setattr(
            "weatherstat.advisory.send_ha_notification",
            lambda *a, **kw: sent.append((a, kw)) or True,
        )

        process_advisory_plan(plan, live=True, current_hour=12)

        # Tracked but not re-notified
        data = json.loads(state_file.read_text())
        assert len(data["opportunities"]) == 1
        assert sent == []

    def test_warnings_persisted(self, tmp_path, monkeypatch) -> None:
        """Backup-breach warnings are persisted alongside opportunities."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        plan = self._plan(breaches=("room_temp projected 63 at 6h",))

        process_advisory_plan(plan, live=True, current_hour=12)

        data = json.loads(state_file.read_text())
        assert len(data["warnings"]) == 1
        assert "63" in data["warnings"][0]["message"]

    def test_none_plan_clears_state_and_dismisses_rollup(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Passing advisory_plan=None clears opportunities and dismisses the rollup."""
        from weatherstat.advisory import process_advisory_plan

        state_file = tmp_path / "state.json"
        # Pre-populate with stale opportunities so the dismiss path triggers
        state_file.write_text(json.dumps({
            "cooldowns": {},
            "opportunities": [{"device": "old"}],
            "warnings": [{"message": "old"}],
        }))
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        dismissed: list[tuple] = []
        monkeypatch.setattr(
            "weatherstat.advisory.dismiss_ha_notification",
            lambda *a, **kw: dismissed.append((a, kw)) or True,
        )

        process_advisory_plan(None, live=True, current_hour=12)

        data = json.loads(state_file.read_text())
        assert data["opportunities"] == []
        assert data["warnings"] == []
        assert len(dismissed) == 1

    def test_empty_opportunities_console_output(self, tmp_path, monkeypatch, capsys) -> None:
        """Empty plan logs the no-opportunities line."""
        from weatherstat.advisory import process_advisory_plan

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        plan = self._plan()

        process_advisory_plan(plan, live=False, current_hour=12)

        assert "No beneficial opportunities" in capsys.readouterr().out


class TestEnvironmentEntryConfig:
    """Test EnvironmentEntryConfig type and YAML parsing."""

    def test_config_type(self) -> None:
        """EnvironmentEntryConfig stores name, entity_id, column, kind, etc."""
        from weatherstat.types import EnvironmentEntryConfig

        cfg = EnvironmentEntryConfig(
            name="piano", entity_id="binary_sensor.window_piano_is_open",
            column="piano_open", kind="window", default_state="closed",
            active_state="on", advisory=True,
        )
        assert cfg.name == "piano"
        assert cfg.column == "piano_open"
        assert cfg.kind == "window"
        assert cfg.advisory is True
        assert cfg.label == "piano"
        assert cfg.close_action == "close"
        assert cfg.open_action == "open"

    def test_shade_action_verbs(self) -> None:
        """Shade kind uses raise/lower verbs.

        close_action returns the shade to its default state (open=raised),
        open_action moves it toward the active state (closed=lowered).
        """
        from weatherstat.types import EnvironmentEntryConfig

        cfg = EnvironmentEntryConfig(
            name="moss_garden_shade", entity_id="cover.moss_garden_shade",
            column="moss_garden_shade_active", kind="shade", default_state="open",
            active_state="closed", advisory=True,
        )
        assert cfg.close_action == "raise"
        assert cfg.open_action == "lower"
        assert cfg.label == "moss garden shade"

    def test_yaml_parsing(self) -> None:
        """Config environment section parsed from example YAML."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        assert isinstance(cfg.environment, dict)
        assert len(cfg.environment) > 0
        for _name, entry in cfg.environment.items():
            assert entry.entity_id
            assert entry.column
            assert entry.kind

    def test_advisory_environment(self) -> None:
        """advisory_environment returns only entries with advisory=True."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        for entry in cfg.advisory_environment.values():
            assert entry.advisory is True

    def test_config_is_frozen(self) -> None:
        """EnvironmentEntryConfig is immutable."""
        from weatherstat.types import EnvironmentEntryConfig

        cfg = EnvironmentEntryConfig(
            name="test", entity_id="binary_sensor.test",
            column="test_open", kind="window", default_state="closed",
            active_state="on",
        )
        with pytest.raises(AttributeError):
            cfg.name = "changed"  # type: ignore[misc]


# TestSaveAdvisoryRecommendations was merged into TestProcessAdvisoryPlan above.
# (save_advisory_recommendations was folded into process_advisory_plan during the
# advisory pipeline cleanup.)
