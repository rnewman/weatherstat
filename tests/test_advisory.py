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
# Regression test for the bug where evaluate_environment_opportunities dropped
# solar_elevations when constructing the toggled HouseState, causing the
# simulator to fall back from elevation-based solar (significant gains) to
# the empty legacy per-hour model.  Every window toggle appeared to cool
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


class TestOpportunityState:
    """Test persistent opportunity state management."""

    def test_load_empty(self, tmp_path, monkeypatch) -> None:
        """Empty state file → empty active and cooldowns."""
        from weatherstat.advisory import OpportunityState

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        state = OpportunityState.load()
        assert state.active == {}
        assert state.cooldowns == {}

    def test_load_new_format(self, tmp_path, monkeypatch) -> None:
        """New format with active + cooldowns loads correctly."""
        from weatherstat.advisory import OpportunityState

        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "active": {"bedroom": {"action": "open", "total_benefit": 2.0}},
            "cooldowns": {"opportunity_bedroom": 1000.0},
        }))
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        state = OpportunityState.load()
        assert "bedroom" in state.active
        assert state.cooldowns["opportunity_bedroom"] == 1000.0

    def test_save_and_reload(self, tmp_path, monkeypatch) -> None:
        """Round-trip save and load."""
        from weatherstat.advisory import OpportunityState

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        state = OpportunityState(
            active={"piano": {"action": "open", "total_benefit": 1.5}},
            cooldowns={"opportunity_piano": 5000.0},
        )
        state.save()
        loaded = OpportunityState.load()
        assert loaded.active == state.active
        assert loaded.cooldowns == state.cooldowns


# ── Opportunity lifecycle tests ──────────────────────────────────────────


class TestProcessOpportunities:
    """Test opportunity lifecycle: new, persist, expire."""

    @staticmethod
    def _make_opp(entry: str, benefit: float = 2.0):
        from weatherstat.types import EnvironmentOpportunity

        return EnvironmentOpportunity(
            entry=entry,
            action="open",
            comfort_improvement=benefit * 0.7,
            energy_saving=benefit * 0.3,
            total_benefit=benefit,
            message=f"Open {entry} — test",
        )

    def test_new_opportunity_tracked(self, tmp_path, monkeypatch, capsys) -> None:
        """New opportunity is added to active set."""
        from weatherstat.advisory import process_opportunities

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        opp = self._make_opp("bedroom", 1.0)
        active, dismissed = process_opportunities([opp], live=False, current_hour=12)
        assert len(active) == 1
        assert active[0].entry == "bedroom"
        assert len(dismissed) == 0
        assert "new" in capsys.readouterr().out.lower()

    def test_opportunity_persists_across_cycles(self, tmp_path, monkeypatch) -> None:
        """Opportunity stays active across multiple cycles."""
        from weatherstat.advisory import process_opportunities

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        opp = self._make_opp("bedroom", 1.0)

        # Cycle 1: new
        active1, _ = process_opportunities([opp], live=True, current_hour=12)
        assert len(active1) == 1

        # Cycle 2: still valid
        active2, _ = process_opportunities([opp], live=True, current_hour=12)
        assert len(active2) == 1
        assert active2[0].first_seen == active1[0].first_seen  # preserved

    def test_expired_opportunity_dismissed(self, tmp_path, monkeypatch, capsys) -> None:
        """Opportunity removed when no longer in candidates."""
        from weatherstat.advisory import process_opportunities

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)
        opp = self._make_opp("bedroom", 1.0)

        # Cycle 1: active
        process_opportunities([opp], live=True, current_hour=12)

        # Cycle 2: no longer a candidate → dismissed
        active, dismissed = process_opportunities([], live=True, current_hour=12)
        assert len(active) == 0
        assert "bedroom" in dismissed

    def test_below_notification_threshold_not_notified(self, tmp_path, monkeypatch) -> None:
        """Opportunity below notification threshold is tracked but not notified."""
        from weatherstat.advisory import process_opportunities

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        # benefit=0.5, notification_threshold=1.5 → tracked, not notified
        opp = self._make_opp("bedroom", 0.5)
        active, _ = process_opportunities([opp], live=False, current_hour=12)
        assert len(active) == 1
        assert not active[0].notified

    def test_above_notification_threshold_notified(self, tmp_path, monkeypatch) -> None:
        """Opportunity above notification threshold is marked as notified."""
        from weatherstat.advisory import process_opportunities

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        # benefit=2.0 > notification_threshold=1.5 → notified
        opp = self._make_opp("bedroom", 2.0)
        active, _ = process_opportunities([opp], live=False, current_hour=12)
        assert len(active) == 1
        assert active[0].notified

    def test_quiet_hours_suppress_notification_not_tracking(self, tmp_path, monkeypatch) -> None:
        """During quiet hours: tracked but not notified, even above threshold."""
        from weatherstat.advisory import process_opportunities

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        opp = self._make_opp("bedroom", 5.0)
        active, _ = process_opportunities([opp], live=False, current_hour=23)
        assert len(active) == 1
        assert not active[0].notified  # quiet hours suppress notification

    def test_empty_opportunities(self, tmp_path, monkeypatch, capsys) -> None:
        """No opportunities → empty result."""
        from weatherstat.advisory import process_opportunities

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        active, dismissed = process_opportunities([], live=False, current_hour=12)
        assert active == []
        assert dismissed == []
        assert "No opportunities" in capsys.readouterr().out


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
        """Shade kind uses lower/raise action verbs."""
        from weatherstat.types import EnvironmentEntryConfig

        cfg = EnvironmentEntryConfig(
            name="moss_garden_shade", entity_id="cover.moss_garden_shade",
            column="moss_garden_shade_active", kind="shade", default_state="open",
            active_state="closed", advisory=True,
        )
        assert cfg.close_action == "lower"
        assert cfg.open_action == "raise"
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


class TestSaveAdvisoryRecommendations:
    """Test advisory recommendation persistence."""

    def test_save_with_layers(self, tmp_path, monkeypatch) -> None:
        """Advisory layers are saved to state file."""
        from weatherstat.advisory import save_advisory_recommendations
        from weatherstat.types import AdvisoryDecision

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)

        layers = {
            "reasonable_advisories": {
                "win_a": AdvisoryDecision("win_a", action="close", transition_step=12),
            },
            "reasonable_cost": 3.0,
            "baseline_cost": 5.0,
            "backup_breaches": ["room_temp projected 63 at 6h"],
            "proactive": {
                "win_b": {"idx": 0, "cost_delta": -0.5},
            },
            "proactive_advisories": {
                "win_b": AdvisoryDecision("win_b", action="open", transition_step=0, return_step=24),
            },
        }
        save_advisory_recommendations(layers, live=True)

        data = json.loads(state_file.read_text())
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["device"] == "win_a"
        assert data["recommendations"][0]["action"] == "close"
        assert data["recommendations"][0]["in_minutes"] == 60
        assert len(data["warnings"]) == 1
        assert "63" in data["warnings"][0]["message"]
        assert len(data["proactive"]) == 1
        assert data["proactive"][0]["device"] == "win_b"
        assert data["proactive"][0]["duration_minutes"] == 120

    def test_save_clears_when_no_layers(self, tmp_path, monkeypatch) -> None:
        """None advisory_layers clears recommendations."""
        from weatherstat.advisory import save_advisory_recommendations

        state_file = tmp_path / "state.json"
        # Pre-populate with stale data
        state_file.write_text(json.dumps({
            "active": {},
            "cooldowns": {},
            "recommendations": [{"device": "old"}],
            "warnings": [{"message": "old"}],
            "proactive": [{"device": "old"}],
        }))
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)

        save_advisory_recommendations(None, live=True)

        data = json.loads(state_file.read_text())
        assert data["recommendations"] == []
        assert data["warnings"] == []
        assert data["proactive"] == []

    def test_no_save_in_dry_run(self, tmp_path, monkeypatch) -> None:
        """Dry run does not persist anything."""
        from weatherstat.advisory import save_advisory_recommendations
        from weatherstat.types import AdvisoryDecision

        state_file = tmp_path / "state.json"
        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", state_file)

        layers = {
            "reasonable_advisories": {
                "win": AdvisoryDecision("win", action="close"),
            },
            "reasonable_cost": 3.0,
            "baseline_cost": 5.0,
            "backup_breaches": [],
            "proactive": {},
            "proactive_advisories": {},
        }
        save_advisory_recommendations(layers, live=False)
        assert not state_file.exists()
