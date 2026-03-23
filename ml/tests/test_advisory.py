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
    """Load sim params with synthetic window betas for advisory tests.

    Sysid may not learn window betas if there's insufficient sealed/ventilated
    data. Advisory tests need window effects to exist so toggling a window
    changes simulation output.
    """
    params = load_sim_params()
    # Inject window betas: each sensor with a matching window gets a beta
    from weatherstat.yaml_config import load_config
    cfg = load_config()
    augmented_taus: dict[str, TauModel] = {}
    for sensor, tau_model in params.taus.items():
        win_cols = cfg.window_columns_for_sensor(sensor)
        if win_cols and not tau_model.window_betas:
            # Inject a reasonable beta: 1/tau_eff ≈ 1/tau_base + beta
            # beta = 0.02 → effective tau ≈ 25h (from 45h base)
            win_name = win_cols[0].removeprefix("window_").removesuffix("_open")
            augmented_taus[sensor] = TauModel(
                tau_base=tau_model.tau_base,
                window_betas={win_name: 0.02},
                interaction_betas=tau_model.interaction_betas,
            )
        else:
            augmented_taus[sensor] = tau_model
    return SimParams(
        taus=augmented_taus,
        gains=params.gains,
        solar=params.solar,
        sensors=params.sensors,
        effectors=params.effectors,
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
        defaults[label] = ComfortSchedule(label=label, entries=tuple(entries))
    return list(defaults.values())


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
    def _make_opp(window: str, benefit: float = 2.0):
        from weatherstat.types import WindowOpportunity

        return WindowOpportunity(
            window=window,
            action="open",
            comfort_improvement=benefit * 0.7,
            energy_saving=benefit * 0.3,
            total_benefit=benefit,
            message=f"Open {window} — test",
        )

    def test_new_opportunity_tracked(self, tmp_path, monkeypatch, capsys) -> None:
        """New opportunity is added to active set."""
        from weatherstat.advisory import process_opportunities

        monkeypatch.setattr("weatherstat.advisory.ADVISORY_STATE_FILE", tmp_path / "state.json")
        opp = self._make_opp("bedroom", 1.0)
        active, dismissed = process_opportunities([opp], live=False, current_hour=12)
        assert len(active) == 1
        assert active[0].window == "bedroom"
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
