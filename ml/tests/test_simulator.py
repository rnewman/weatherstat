"""Tests for the grey-box forward simulator."""

from __future__ import annotations

import numpy as np
import pytest

from weatherstat.simulator import (
    HouseState,
    SimParams,
    TauModel,
    _outdoor_at,
    build_activity_timeline,
    load_sim_params,
    predict,
    simulate_sensor,
)
from weatherstat.types import BlowerDecision, MiniSplitDecision, ThermostatTrajectory, TrajectoryScenario

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sim_params() -> SimParams:
    return load_sim_params()


def _all_off() -> TrajectoryScenario:
    return TrajectoryScenario(
        ThermostatTrajectory(heating=False),
        ThermostatTrajectory(heating=False),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def _both_on() -> TrajectoryScenario:
    """Both thermostats on for full horizon."""
    return TrajectoryScenario(
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=None),
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=None),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def _bedroom_heat() -> TrajectoryScenario:
    """Mini split bedroom heat, thermostats off."""
    return TrajectoryScenario(
        ThermostatTrajectory(heating=False),
        ThermostatTrajectory(heating=False),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "heat", 72), MiniSplitDecision("living_room", "off", 72)),
    )


_CURRENT_TEMPS = {
    "upstairs": 70.0, "downstairs": 69.0, "bedroom": 68.5,
    "office": 67.0, "family_room": 69.5, "kitchen": 68.0,
    "piano": 67.5, "bathroom": 68.0, "living_room": 69.0,
}


def _make_state(
    outdoor: float = 42.0,
    hour: float = 14.5,
    temps: dict[str, float] | None = None,
) -> HouseState:
    return HouseState(
        current_temps=temps or _CURRENT_TEMPS,
        outdoor_temp=outdoor,
        forecast_temps=[outdoor] * 12,
        window_states={},
        hour_of_day=hour,
    )


# ── load_sim_params ─────────────────────────────────────────────────────


def test_load_sim_params(sim_params: SimParams) -> None:
    assert len(sim_params.sensors) > 0
    assert len(sim_params.effectors) > 0
    assert len(sim_params.taus) > 0
    assert len(sim_params.gains) > 0
    assert len(sim_params.solar) > 0


def test_tau_values_reasonable(sim_params: SimParams) -> None:
    for sensor, tau_model in sim_params.taus.items():
        assert tau_model.tau_base > 0, f"{sensor} tau_base must be positive"
        # Window betas should be non-negative (physical constraint)
        for win, beta in tau_model.window_betas.items():
            assert beta >= 0, f"{sensor} window {win} beta must be >= 0"


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


def test_timeline_with_delay() -> None:
    """Activity starts at switch_on_step, not at step 0."""
    tl = build_activity_timeline(1.0, [], n_future_steps=6, switch_on_step=3)
    assert len(tl) == 18 + 6  # padded history + future
    # Future: [0, 0, 0, 1, 1, 1]
    assert tl[18:] == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_timeline_with_switch_off() -> None:
    """Activity ends at switch_off_step."""
    tl = build_activity_timeline(1.0, [], n_future_steps=6, switch_off_step=3)
    # Future: [1, 1, 1, 0, 0, 0]
    assert tl[18:] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_timeline_with_delay_and_switch_off() -> None:
    """Activity only during [switch_on_step, switch_off_step)."""
    tl = build_activity_timeline(1.0, [], n_future_steps=8, switch_on_step=2, switch_off_step=5)
    # Future: [0, 0, 1, 1, 1, 0, 0, 0]
    assert tl[18:] == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


# ── simulate_sensor ──────────────────────────────────────────────────────


def test_passive_cooling_toward_outdoor() -> None:
    """With no effectors, temperature should decay toward outdoor."""
    temps = simulate_sensor(
        sensor="test",
        current_temp=70.0,
        outdoor_temp=42.0,
        forecast_temps=[42.0] * 12,
        tau=40.0,
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
    """Open windows (lower effective tau) should cool faster."""
    tau_model = TauModel(tau_base=40.0, window_betas={"test_window": 0.04})
    kwargs = dict(
        sensor="test", current_temp=70.0, outdoor_temp=42.0,
        forecast_temps=[42.0] * 12,
        effector_timelines={}, gains={}, solar_profile={},
        start_hour=2.0, n_steps=72,
    )
    sealed = simulate_sensor(**kwargs, tau=tau_model.effective_tau({}))
    vent = simulate_sensor(**kwargs, tau=tau_model.effective_tau({"test_window": True}))
    # At every step, ventilated should be cooler
    assert all(v < s for v, s in zip(vent, sealed, strict=True))


def test_heating_warms_sensor() -> None:
    """Positive effector gain should cause warming."""
    timeline = [0.0] * 18 + [1.0] * 73  # history=off, future=on
    temps = simulate_sensor(
        sensor="test",
        current_temp=68.0,
        outdoor_temp=42.0,
        forecast_temps=[42.0] * 12,
        tau=40.0,
        effector_timelines={"heater": timeline},
        gains={"heater": (2.0, 0.0)},  # 2°F/hr, no delay
        solar_profile={},
        start_hour=2.0,
        n_steps=72,
    )
    # At 6h with 2°F/hr gain, should be significantly warmer
    assert temps[-1] > 68.0


# ── predict ──────────────────────────────────────────────────────────────


def test_predict_shape(sim_params: SimParams) -> None:
    targets, preds = predict(
        _make_state(), [_all_off(), _both_on()], sim_params, [12, 24, 48, 72],
    )
    assert preds.shape[0] == 2
    assert preds.shape[1] == len(targets)
    assert all("_temp_t+" in t for t in targets)


def test_predict_no_nan(sim_params: SimParams) -> None:
    targets, preds = predict(
        _make_state(), [_all_off()], sim_params, [12],
    )
    assert not np.any(np.isnan(preds))


def test_heating_warms_vs_all_off(sim_params: SimParams) -> None:
    """Both-on should predict warmer temps than all-off at all horizons."""
    targets, preds = predict(
        _make_state(), [_all_off(), _both_on()], sim_params, [12, 24, 48, 72],
    )
    # Check upstairs (directly heated)
    for j, t in enumerate(targets):
        if "upstairs" in t:
            assert preds[1, j] >= preds[0, j], f"{t}: heating should be >= all-off"


def test_mini_split_heats_bedroom_only(sim_params: SimParams) -> None:
    """Mini split bedroom heat should warm bedroom much more than office."""
    targets, preds = predict(
        _make_state(), [_all_off(), _bedroom_heat()], sim_params, [72],
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
    targets, preds = predict(
        _make_state(hour=2.0), [_all_off()], sim_params, [72],  # nighttime, no solar
    )
    for j, t in enumerate(targets):
        if "t+72" in t:
            room = t.split("_temp_t+")[0]
            start = _CURRENT_TEMPS.get(room, 70.0)
            assert preds[0, j] < start, f"{room} should cool from {start:.1f} (got {preds[0, j]:.1f})"


def test_performance(sim_params: SimParams) -> None:
    """Full trajectory sweep should complete in reasonable time."""
    import time

    from weatherstat.control import generate_trajectory_scenarios

    scenarios = generate_trajectory_scenarios()
    state = _make_state()
    t0 = time.monotonic()
    predict(state, scenarios, sim_params, [12, 24, 48, 72])
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert elapsed_ms < 5000, f"Full sweep took {elapsed_ms:.0f}ms (should be <5000ms)"


# ── Trajectory scenario tests ──────────────────────────────────────────


def _traj_heat_2h() -> TrajectoryScenario:
    """Both thermostats heat for 2h starting now."""
    return TrajectoryScenario(
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=24),
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=24),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def _traj_delayed_heat() -> TrajectoryScenario:
    """Upstairs heats after 1h delay for 2h."""
    return TrajectoryScenario(
        ThermostatTrajectory(heating=True, delay_steps=12, duration_steps=24),
        ThermostatTrajectory(heating=False),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )


def test_trajectory_2h_warmer_than_off(sim_params: SimParams) -> None:
    """2h heating trajectory should produce warmer temps than all-off."""
    targets, preds = predict(
        _make_state(), [_all_off(), _traj_heat_2h()], sim_params, [12, 24, 48, 72],
    )
    # At 2h horizon (step 24), heating should be warmer
    for j, t in enumerate(targets):
        if "upstairs" in t and "t+24" in t:
            assert preds[1, j] > preds[0, j], f"{t}: 2h heating should be warmer than off"


def test_trajectory_2h_cooler_than_6h(sim_params: SimParams) -> None:
    """2h heating should produce cooler 6h temps than 6h continuous heating."""
    traj_6h = TrajectoryScenario(
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=72),
        ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=72),
        (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
        (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
    )
    targets, preds = predict(
        _make_state(), [_traj_heat_2h(), traj_6h], sim_params, [72],
    )
    # At 6h, 2h-heating should be cooler than 6h-heating (it turned off at 2h)
    for j, t in enumerate(targets):
        if "upstairs" in t:
            assert preds[0, j] < preds[1, j], f"{t}: 2h should be cooler than 6h at 6h horizon"


def test_trajectory_delayed_heat_delayed_effect(sim_params: SimParams) -> None:
    """Delayed heating should have minimal effect at 1h but warm up by 4h."""
    targets, preds = predict(
        _make_state(hour=2.0), [_all_off(), _traj_delayed_heat()], sim_params, [12, 48],
    )
    for j, t in enumerate(targets):
        if "upstairs" in t and "t+12" in t:
            # At 1h, delayed heat hasn't started yet — should be similar to all-off
            assert abs(preds[0, j] - preds[1, j]) < 0.5, f"{t}: delayed heat shouldn't affect 1h prediction"
        if "upstairs" in t and "t+48" in t:
            # At 4h, delayed heat has been running for 2h — should be warmer
            assert preds[1, j] > preds[0, j], f"{t}: delayed heat should warm by 4h"


def test_trajectory_performance(sim_params: SimParams) -> None:
    """Trajectory sweep should complete in under 5 seconds.

    ~7K scenarios with vectorized numpy integration. Budget 5s for CI variability.
    """
    import time

    from weatherstat.control import generate_trajectory_scenarios

    scenarios = generate_trajectory_scenarios()
    assert len(scenarios) > 1000, f"Expected >1000 trajectory scenarios, got {len(scenarios)}"
    state = _make_state()
    t0 = time.monotonic()
    predict(state, scenarios, sim_params, [12, 24, 48, 72])
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert elapsed_ms < 5000, f"Trajectory sweep took {elapsed_ms:.0f}ms (should be <5000ms)"
