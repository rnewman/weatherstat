"""Tests for physics-based window advisories.

Tests advisory evaluation (physics simulator-backed) and notification
dispatch (cooldowns, quiet hours).
"""

from __future__ import annotations

import time

import pytest

from weatherstat.advisory import (
    Advisory,
    AdvisoryType,
    _cooldown_key,
    _is_on_cooldown,
    evaluate_window_advisories,
    process_advisories,
)
from weatherstat.simulator import HouseState, SimParams, TauModel, load_sim_params
from weatherstat.types import (
    BlowerDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    MiniSplitDecision,
    RoomComfort,
    ThermostatTrajectory,
    TrajectoryScenario,
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


def _all_off() -> TrajectoryScenario:
    return TrajectoryScenario(
        ThermostatTrajectory(heating=False),
        ThermostatTrajectory(heating=False),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def _both_on() -> TrajectoryScenario:
    return TrajectoryScenario(
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=None),
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=None),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def _make_schedules(**overrides: list[ComfortScheduleEntry]) -> list[ComfortSchedule]:
    """Comfort schedules with optional room overrides."""
    from weatherstat.control import default_comfort_schedules

    defaults = {s.room: s for s in default_comfort_schedules()}
    for room, entries in overrides.items():
        defaults[room] = ComfortSchedule(room=room, entries=tuple(entries))
    return list(defaults.values())


# ── evaluate_window_advisories tests ──────────────────────────────────────


class TestEvaluateWindowAdvisories:
    """Test physics-based window advisory evaluation."""

    def test_close_window_when_heating_active(self, sim_params) -> None:
        """Open window during heating should generate close advisory."""
        state = HouseState(
            current_temps=_CURRENT_TEMPS,
            outdoor_temp=35.0,
            forecast_temps=[35.0] * 12,
            window_states={"bedroom": True},
            hour_of_day=6.0,
        )
        schedules = _make_schedules()
        advisories = evaluate_window_advisories(
            state, _both_on(), sim_params, schedules, base_hour=6,
        )
        bedroom_advs = [a for a in advisories if a.window == "bedroom"]
        assert len(bedroom_advs) > 0, "Should recommend closing bedroom window when heating"
        assert bedroom_advs[0].advisory_type == AdvisoryType.CLOSE_WINDOWS
        assert bedroom_advs[0].improvement > 0

    def test_open_window_when_room_too_warm(self, sim_params) -> None:
        """Warm room with cooler outdoor temp should generate open advisory."""
        warm_temps = dict(_CURRENT_TEMPS)
        warm_temps["bedroom"] = 77.0  # well above comfort max

        state = HouseState(
            current_temps=warm_temps,
            outdoor_temp=62.0,
            forecast_temps=[62.0] * 12,
            window_states={},
            hour_of_day=14.0,
        )
        # Tight comfort band that the warm room violates
        schedules = _make_schedules(
            bedroom=[ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 73.0, 2.0, 2.0))],
        )
        advisories = evaluate_window_advisories(
            state, _all_off(), sim_params, schedules, base_hour=14,
        )
        bedroom_advs = [a for a in advisories if a.window == "bedroom"]
        assert len(bedroom_advs) > 0, "Should recommend opening bedroom window for cooling"
        assert bedroom_advs[0].advisory_type == AdvisoryType.FREE_COOLING

    def test_no_advisory_when_outdoor_colder_and_closed(self, sim_params) -> None:
        """Opening window to colder outdoor should not be recommended when room is in comfort."""
        state = HouseState(
            current_temps=_CURRENT_TEMPS,
            outdoor_temp=35.0,
            forecast_temps=[35.0] * 12,
            window_states={},
            hour_of_day=2.0,
        )
        # Wide comfort band that current temps satisfy
        schedules = _make_schedules(
            bedroom=[ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 60.0, 80.0, 1.0, 1.0))],
        )
        advisories = evaluate_window_advisories(
            state, _all_off(), sim_params, schedules, base_hour=2,
        )
        bedroom_opens = [
            a for a in advisories
            if a.window == "bedroom" and a.advisory_type == AdvisoryType.FREE_COOLING
        ]
        assert len(bedroom_opens) == 0, "Should not recommend opening window to colder air when in comfort"

    def test_windows_without_rooms_ignored(self, sim_params) -> None:
        """Windows with rooms=[] (basement, balcony) should not generate advisories."""
        state = HouseState(
            current_temps=_CURRENT_TEMPS,
            outdoor_temp=55.0,
            forecast_temps=[55.0] * 12,
            window_states={"basement": True},
            hour_of_day=14.0,
        )
        schedules = _make_schedules()
        advisories = evaluate_window_advisories(
            state, _all_off(), sim_params, schedules, base_hour=14,
        )
        basement_advs = [a for a in advisories if a.window == "basement"]
        assert len(basement_advs) == 0

    def test_advisories_sorted_by_improvement(self, sim_params) -> None:
        """Multiple advisories should be sorted by improvement (best first)."""
        state = HouseState(
            current_temps=_CURRENT_TEMPS,
            outdoor_temp=35.0,
            forecast_temps=[35.0] * 12,
            window_states={"bedroom": True, "kitchen": True},
            hour_of_day=6.0,
        )
        schedules = _make_schedules()
        advisories = evaluate_window_advisories(
            state, _both_on(), sim_params, schedules, base_hour=6,
        )
        if len(advisories) >= 2:
            assert advisories[0].improvement >= advisories[1].improvement


# ── Cooldown and dispatch tests ──────────────────────────────────────────


class TestCooldownKey:
    """Test per-window cooldown key generation."""

    def test_key_with_window(self) -> None:
        a = Advisory(AdvisoryType.FREE_COOLING, "t", "m", window="bedroom")
        assert _cooldown_key(a) == "free_cooling_bedroom"

    def test_key_without_window(self) -> None:
        a = Advisory(AdvisoryType.FREE_COOLING, "t", "m")
        assert _cooldown_key(a) == "free_cooling"


class TestCooldown:
    """Test cooldown logic."""

    def test_not_on_cooldown_initially(self) -> None:
        assert not _is_on_cooldown({}, "free_cooling_bedroom", AdvisoryType.FREE_COOLING)

    def test_on_cooldown_when_recent(self) -> None:
        state = {"free_cooling_bedroom": time.time() - 60}
        assert _is_on_cooldown(state, "free_cooling_bedroom", AdvisoryType.FREE_COOLING)

    def test_off_cooldown_after_expiry(self) -> None:
        state = {"free_cooling_bedroom": time.time() - 20000}
        assert not _is_on_cooldown(state, "free_cooling_bedroom", AdvisoryType.FREE_COOLING)

    def test_per_window_isolation(self) -> None:
        """Cooldown for bedroom should not affect kitchen."""
        state = {"free_cooling_bedroom": time.time() - 60}
        assert _is_on_cooldown(state, "free_cooling_bedroom", AdvisoryType.FREE_COOLING)
        assert not _is_on_cooldown(state, "free_cooling_kitchen", AdvisoryType.FREE_COOLING)


class TestProcessAdvisories:
    """Test advisory dispatch with cooldowns and quiet hours."""

    def _make_advisory(self, window: str = "bedroom", adv_type: AdvisoryType = AdvisoryType.FREE_COOLING) -> Advisory:
        return Advisory(adv_type, f"Open {window}", f"msg for {window}", window=window, improvement=5.0)

    def test_empty_advisories(self, capsys) -> None:
        result = process_advisories([], live=False)
        assert result == []
        assert "No advisories triggered" in capsys.readouterr().out

    def test_advisory_dispatched_when_not_on_cooldown(self, capsys) -> None:
        adv = self._make_advisory()
        result = process_advisories([adv], live=False, current_hour=12)
        assert len(result) == 1
        assert result[0].window == "bedroom"

    def test_quiet_hours_suppress_dispatch(self, capsys) -> None:
        adv = self._make_advisory()
        result = process_advisories([adv], live=False, current_hour=23)
        assert len(result) == 0
        assert "quiet hours" in capsys.readouterr().out

    def test_outside_quiet_hours(self) -> None:
        adv = self._make_advisory()
        result = process_advisories([adv], live=False, current_hour=12)
        assert len(result) == 1

    def test_multiple_advisories_dispatched(self) -> None:
        advs = [self._make_advisory("bedroom"), self._make_advisory("kitchen")]
        result = process_advisories(advs, live=False, current_hour=12)
        assert len(result) == 2
