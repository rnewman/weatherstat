"""Tests for the grey-box forward simulator."""

from __future__ import annotations

import numpy as np
import pytest

from weatherstat.simulator import (
    SimParams,
    _outdoor_at,
    _scenario_to_activities,
    batch_simulate,
    build_activity_timeline,
    load_sim_params,
    simulate_sensor,
)
from weatherstat.types import BlowerDecision, HVACScenario, MiniSplitDecision


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sim_params() -> SimParams:
    return load_sim_params()


def _all_off() -> HVACScenario:
    return HVACScenario(
        False, False,
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def _both_on() -> HVACScenario:
    return HVACScenario(
        True, True,
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def _bedroom_heat() -> HVACScenario:
    return HVACScenario(
        False, False,
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "heat", 72), MiniSplitDecision("living_room", "off", 72)),
    )


_CURRENT_TEMPS = {
    "upstairs": 70.0, "downstairs": 69.0, "bedroom": 68.5,
    "office": 67.0, "family_room": 69.5, "kitchen": 68.0,
    "piano": 67.5, "bathroom": 68.0, "living_room": 69.0,
}


# ── load_sim_params ─────────────────────────────────────────────────────


def test_load_sim_params(sim_params: SimParams) -> None:
    assert len(sim_params.sensors) > 0
    assert len(sim_params.effectors) > 0
    assert len(sim_params.taus) > 0
    assert len(sim_params.gains) > 0
    assert len(sim_params.solar) > 0


def test_tau_values_reasonable(sim_params: SimParams) -> None:
    for sensor, (tau_s, tau_v) in sim_params.taus.items():
        assert tau_s > 0, f"{sensor} tau_sealed must be positive"
        assert tau_v > 0, f"{sensor} tau_vent must be positive"
        assert tau_v <= tau_s, f"{sensor} ventilated tau should be <= sealed"


# ── _outdoor_at ──────────────────────────────────────────────────────────


def test_outdoor_at_no_forecast() -> None:
    assert _outdoor_at(42.0, [], 3.0) == 42.0


def test_outdoor_at_within_first_hour() -> None:
    # 0.5 hours ahead: interpolate between 42 and 40
    result = _outdoor_at(42.0, [40.0, 38.0], 0.5)
    assert result == pytest.approx(41.0)


def test_outdoor_at_exact_hour() -> None:
    # 1.0 hours ahead -> forecast[0]
    assert _outdoor_at(42.0, [40.0, 38.0], 1.0) == 40.0


def test_outdoor_at_beyond_forecast() -> None:
    # 5 hours ahead with only 2 forecast entries -> clamp to last
    assert _outdoor_at(42.0, [40.0, 38.0], 5.0) == 38.0


# ── build_activity_timeline ──────────────────────────────────────────────


def test_timeline_pads_short_history() -> None:
    tl = build_activity_timeline(1.0, [0.5, 0.5], n_future_steps=3)
    # Should be padded to 18 history steps + 3 future
    assert len(tl) == 21
    assert tl[0] == 0.0  # padding
    assert tl[16] == 0.5  # actual history
    assert tl[17] == 0.5
    assert tl[18] == 1.0  # future
    assert tl[20] == 1.0


def test_timeline_truncates_long_history() -> None:
    tl = build_activity_timeline(0.0, [1.0] * 30, n_future_steps=2)
    assert len(tl) == 20  # 18 history + 2 future
    assert all(v == 1.0 for v in tl[:18])
    assert all(v == 0.0 for v in tl[18:])


# ── _scenario_to_activities ──────────────────────────────────────────────


def test_scenario_all_off_activities(sim_params: SimParams) -> None:
    acts = _scenario_to_activities(_all_off(), sim_params)
    for eff_name, val in acts.items():
        assert val == 0.0, f"{eff_name} should be 0 when all off"


def test_scenario_both_on_activities(sim_params: SimParams) -> None:
    acts = _scenario_to_activities(_both_on(), sim_params)
    assert acts["thermostat_upstairs"] == 1.0
    assert acts["thermostat_downstairs"] == 1.0
    assert acts["navien"] == 1.0


def test_scenario_bedroom_heat(sim_params: SimParams) -> None:
    acts = _scenario_to_activities(_bedroom_heat(), sim_params)
    assert acts["mini_split_bedroom"] == 1.0
    assert acts["thermostat_upstairs"] == 0.0
    assert acts["navien"] == 0.0  # no thermostat on


# ── simulate_sensor ──────────────────────────────────────────────────────


def test_passive_cooling_toward_outdoor() -> None:
    """With no effectors, temperature should decay toward outdoor."""
    temps = simulate_sensor(
        sensor="test",
        current_temp=70.0,
        outdoor_temp=42.0,
        forecast_temps=[42.0] * 12,
        tau_sealed=40.0,
        tau_vent=17.0,
        is_ventilated=False,
        effector_timelines={},
        gains={},
        solar_profile={},
        start_hour=2.0,  # nighttime, no solar
        n_steps=72,
    )
    # Should be monotonically decreasing
    assert all(temps[i] > temps[i + 1] for i in range(len(temps) - 1))
    # Should approach but not reach outdoor temp
    assert temps[-1] > 42.0
    assert temps[-1] < 70.0


def test_ventilated_cools_faster() -> None:
    """Ventilated (open windows) should cool faster."""
    kwargs = dict(
        sensor="test", current_temp=70.0, outdoor_temp=42.0,
        forecast_temps=[42.0] * 12, tau_sealed=40.0, tau_vent=17.0,
        effector_timelines={}, gains={}, solar_profile={},
        start_hour=2.0, n_steps=72,
    )
    sealed = simulate_sensor(**kwargs, is_ventilated=False)
    vent = simulate_sensor(**kwargs, is_ventilated=True)
    # At every step, ventilated should be cooler
    assert all(v < s for v, s in zip(vent, sealed))


def test_heating_warms_sensor() -> None:
    """Positive effector gain should cause warming."""
    timeline = [0.0] * 18 + [1.0] * 73  # history=off, future=on
    temps = simulate_sensor(
        sensor="test",
        current_temp=68.0,
        outdoor_temp=42.0,
        forecast_temps=[42.0] * 12,
        tau_sealed=40.0,
        tau_vent=17.0,
        is_ventilated=False,
        effector_timelines={"heater": timeline},
        gains={"heater": (2.0, 0.0)},  # 2°F/hr, no delay
        solar_profile={},
        start_hour=2.0,
        n_steps=72,
    )
    # At 6h with 2°F/hr gain, should be significantly warmer
    assert temps[-1] > 68.0


# ── batch_simulate ───────────────────────────────────────────────────────


def test_batch_simulate_shape(sim_params: SimParams) -> None:
    targets, preds = batch_simulate(
        _CURRENT_TEMPS, 42.0, [42.0] * 12, {},
        [_all_off(), _both_on()], sim_params, 14.5, [12, 24, 48, 72],
    )
    assert preds.shape[0] == 2
    assert preds.shape[1] == len(targets)
    assert all("_temp_t+" in t for t in targets)


def test_batch_simulate_no_nan(sim_params: SimParams) -> None:
    targets, preds = batch_simulate(
        _CURRENT_TEMPS, 42.0, [42.0] * 12, {},
        [_all_off()], sim_params, 14.5, [12],
    )
    assert not np.any(np.isnan(preds))


def test_heating_warms_vs_all_off(sim_params: SimParams) -> None:
    """Both-on should predict warmer temps than all-off at all horizons."""
    targets, preds = batch_simulate(
        _CURRENT_TEMPS, 42.0, [42.0] * 12, {},
        [_all_off(), _both_on()], sim_params, 14.5, [12, 24, 48, 72],
    )
    # Check upstairs (directly heated)
    for j, t in enumerate(targets):
        if "upstairs" in t:
            assert preds[1, j] >= preds[0, j], f"{t}: heating should be >= all-off"


def test_mini_split_heats_bedroom_only(sim_params: SimParams) -> None:
    """Mini split bedroom heat should warm bedroom much more than office."""
    targets, preds = batch_simulate(
        _CURRENT_TEMPS, 42.0, [42.0] * 12, {},
        [_all_off(), _bedroom_heat()], sim_params, 14.5, [72],
    )
    # Find bedroom and office 6h predictions
    bed_idx = next(j for j, t in enumerate(targets) if t == "bedroom_temp_t+72")
    off_idx = next(j for j, t in enumerate(targets) if t == "office_temp_t+72")

    bed_delta = preds[1, bed_idx] - preds[0, bed_idx]  # bedroom: heat vs off
    off_delta = preds[1, off_idx] - preds[0, off_idx]  # office: heat vs off

    assert bed_delta > off_delta, "Bedroom should warm more than office from bedroom split"
    assert bed_delta > 1.0, "Bedroom should warm significantly from mini split"


def test_all_off_cooling(sim_params: SimParams) -> None:
    """All-off should cool below starting temperature when outdoor is cold."""
    targets, preds = batch_simulate(
        _CURRENT_TEMPS, 42.0, [42.0] * 12, {},
        [_all_off()], sim_params, 2.0, [72],  # nighttime, no solar
    )
    for j, t in enumerate(targets):
        if "t+72" in t:
            room = t.split("_temp_t+")[0]
            start = _CURRENT_TEMPS.get(room, 70.0)
            assert preds[0, j] < start, f"{room} should cool from {start:.1f} (got {preds[0, j]:.1f})"


def test_performance(sim_params: SimParams) -> None:
    """Full sweep should complete in under 200ms."""
    import time

    from weatherstat.control import generate_scenarios

    scenarios = generate_scenarios()
    t0 = time.monotonic()
    batch_simulate(
        _CURRENT_TEMPS, 42.0, [42.0] * 12, {},
        scenarios, sim_params, 14.5, [12, 24, 48, 72],
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert elapsed_ms < 200, f"Full sweep took {elapsed_ms:.0f}ms (should be <200ms)"
