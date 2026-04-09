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
from weatherstat.simulator import _build_environment_timelines as build_advisory_timelines
from weatherstat.types import AdvisoryDecision, EffectorDecision, Scenario

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sim_params() -> SimParams:
    """Load from the test sandbox's synthetic thermal_params.json (see conftest.py)."""
    return load_sim_params()


def _all_off() -> Scenario:
    return Scenario(effectors={})


def _both_on() -> Scenario:
    """Both thermostats on for full horizon."""
    return Scenario(effectors={
        "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating"),
        "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating"),
    })


def _bedroom_heat() -> Scenario:
    """Mini split bedroom heat, thermostats off."""
    return Scenario(effectors={
        "mini_split_bedroom": EffectorDecision("mini_split_bedroom", mode="heat", target=72.0),
    })


_CURRENT_TEMPS = {
    "thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5,
    "office_temp": 67.0, "family_room_temp": 69.5,
    "kitchen_temp": 68.0, "piano_temp": 67.5, "bathroom_temp": 68.0, "living_room_temp": 69.0,
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
        environment_states={},
        hour_of_day=hour,
        solar_fractions=[0.0] * 12,
        solar_elevations=[0.0] * 72,
    )


# ── load_sim_params ─────────────────────────────────────────────────────


def test_synthetic_params_well_formed(sim_params: SimParams) -> None:
    assert len(sim_params.sensors) > 0
    assert len(sim_params.effectors) > 0
    assert len(sim_params.taus) > 0
    assert len(sim_params.gains) > 0
    # Elevation-based solar gains (new format) or legacy per-hour solar
    assert len(sim_params.solar_elevation_gains) > 0 or len(sim_params.solar) > 0


def test_tau_values_reasonable(sim_params: SimParams) -> None:
    for sensor, tau_model in sim_params.taus.items():
        assert tau_model.tau_base > 0, f"{sensor} tau_base must be positive"
        # Advisory tau betas should be non-negative (physical constraint)
        for dev, beta in tau_model.environment_tau_betas.items():
            assert beta >= 0, f"{sensor} advisory {dev} beta must be >= 0"


# ── Integration tests (require live thermal_params.json) ─────────────


def test_load_sim_params_from_sandbox() -> None:
    """Verify load_sim_params reads from the test sandbox data dir."""
    sp = load_sim_params()
    assert len(sp.sensors) > 0
    assert len(sp.gains) > 0
    # Gains should be filtered by t-statistic threshold
    for (eff, _), _ in sp.gains.items():
        if "mini_split" in eff:
            # No confounded cross-coupling should survive
            assert any(
                eff.removeprefix("mini_split_") in sen
                for (e, sen), _ in sp.gains.items()
                if e == eff
            )


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


def test_empty_forecast_uses_constant_outdoor(sim_params: SimParams) -> None:
    """Empty forecast_temps degrades to constant outdoor temp, not crash or wrong results.

    With empty forecast, _outdoor_at returns current outdoor for all horizons.
    Predictions should still converge toward outdoor temp.
    """
    outdoor = 42.0
    state_with_forecast = _make_state(outdoor=outdoor)
    state_no_forecast = HouseState(
        current_temps=state_with_forecast.current_temps,
        outdoor_temp=outdoor,
        forecast_temps=[],  # empty forecast
        environment_states={},
        hour_of_day=14.5,
        solar_fractions=[0.0] * 12,
        solar_elevations=[0.0] * 72,
    )

    _, preds_with = predict(state_with_forecast, [_all_off()], sim_params, [12, 72])
    _, preds_no = predict(state_no_forecast, [_all_off()], sim_params, [12, 72])

    # Both should produce valid predictions (no NaN, no crash)
    assert not np.any(np.isnan(preds_with))
    assert not np.any(np.isnan(preds_no))

    # With constant-outdoor forecast (42°F), both should give same results
    # since _make_state uses [outdoor] * 12 which is already constant
    np.testing.assert_allclose(preds_with, preds_no, atol=0.01)


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
    tau_model = TauModel(tau_base=40.0, environment_tau_betas={"test_window": 0.04})
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
    """Mini split bedroom heat should warm bedroom, not office (no cross-coupling gain)."""
    targets, preds = predict(
        _make_state(), [_all_off(), _bedroom_heat()], sim_params, [72],
    )
    bed_idx = next(j for j, t in enumerate(targets) if t == "bedroom_temp_t+72")
    off_idx = next(j for j, t in enumerate(targets) if t == "office_temp_t+72")

    bed_delta = preds[1, bed_idx] - preds[0, bed_idx]  # bedroom: heat vs off
    off_delta = preds[1, off_idx] - preds[0, off_idx]  # office: heat vs off

    assert bed_delta > off_delta, "Bedroom should warm more than office from bedroom split"
    assert off_delta == 0.0, "Office should not be affected (no cross-coupling gain)"
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


def _traj_heat_2h() -> Scenario:
    """Both thermostats heat for 2h starting now."""
    return Scenario(effectors={
        "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating", duration_steps=24),
        "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating", duration_steps=24),
    })


def _traj_delayed_heat() -> Scenario:
    """Upstairs heats after 1h delay for 2h."""
    up = EffectorDecision("thermostat_upstairs", mode="heating", delay_steps=12, duration_steps=24)
    return Scenario(effectors={"thermostat_upstairs": up})


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
    traj_6h = Scenario(effectors={
        "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating", duration_steps=72),
        "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating", duration_steps=72),
    })
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


# ── Regulating effector tests ────────────────────────────────────────


def _bedroom_heat_target(target: float) -> Scenario:
    """Mini split bedroom heat at a specific target temperature."""
    return Scenario(effectors={
        "mini_split_bedroom": EffectorDecision("mini_split_bedroom", mode="heat", target=target),
    })


def test_regulating_different_targets_different_predictions(sim_params: SimParams) -> None:
    """Different target temperatures should produce different predictions."""
    scenarios = [_all_off(), _bedroom_heat_target(68.0), _bedroom_heat_target(72.0)]
    targets, preds = predict(_make_state(), scenarios, sim_params, [72])

    bed_idx = next(j for j, t in enumerate(targets) if t == "bedroom_temp_t+72")
    # Higher target should produce warmer prediction
    assert preds[2, bed_idx] > preds[1, bed_idx], "Higher target should produce warmer bedroom"
    # Both should be warmer than all-off
    assert preds[1, bed_idx] > preds[0, bed_idx], "Heat@68 should be warmer than off"


def test_regulating_temperature_approaches_target(sim_params: SimParams) -> None:
    """With a regulating effector, temperature should approach but not wildly overshoot target."""
    target = 72.0
    scenarios = [_bedroom_heat_target(target)]
    # Start bedroom cold so there's room to heat
    cold_temps = dict(_CURRENT_TEMPS, bedroom=65.0)
    state = _make_state(outdoor=42.0, hour=2.0, temps=cold_temps)
    targets, preds = predict(state, scenarios, sim_params, [72])

    bed_idx = next(j for j, t in enumerate(targets) if t == "bedroom_temp_t+72")
    pred_6h = preds[0, bed_idx]
    # Should be closer to target than starting temp (heated toward it)
    assert pred_6h > 65.0, f"Should have warmed from 65, got {pred_6h:.1f}"
    # Should not overshoot by more than a few degrees (proportional control stabilizes)
    assert pred_6h < target + 3.0, f"Should not overshoot {target} by >3°F, got {pred_6h:.1f}"


def test_regulating_off_same_as_all_off(sim_params: SimParams) -> None:
    """Mini split off with target 0 should produce same results as all-off baseline."""
    off_split = Scenario(effectors={
        "mini_split_bedroom": EffectorDecision("mini_split_bedroom", mode="off"),
        "mini_split_living_room": EffectorDecision("mini_split_living_room", mode="off"),
    })
    targets, preds = predict(_make_state(), [_all_off(), off_split], sim_params, [12, 72])
    # All predictions should be identical
    np.testing.assert_allclose(preds[0], preds[1], atol=0.01)


# ── Solar elevation tests ──────────────────────────────────────────────────


def test_solar_elevation_basic() -> None:
    """Solar elevation computes reasonable values for Seattle."""
    from datetime import UTC, datetime

    from weatherstat.weather import solar_elevation, solar_sin_elevation

    lat, lon = 47.66, -122.40
    # April noon (solar noon ~ 20:10 UTC for Seattle)
    noon_apr = datetime(2026, 4, 5, 20, 10, tzinfo=UTC)
    elev = solar_elevation(lat, lon, noon_apr)
    assert 45 < elev < 52, f"April noon should be ~48°, got {elev:.1f}°"

    # Night
    night = datetime(2026, 4, 5, 10, 0, tzinfo=UTC)
    assert solar_sin_elevation(lat, lon, night) == 0.0

    # Seasonal: summer noon > spring noon > winter noon
    noon_feb = datetime(2026, 2, 15, 20, 10, tzinfo=UTC)
    noon_jun = datetime(2026, 6, 21, 20, 10, tzinfo=UTC)
    assert solar_elevation(lat, lon, noon_feb) < solar_elevation(lat, lon, noon_apr)
    assert solar_elevation(lat, lon, noon_apr) < solar_elevation(lat, lon, noon_jun)


def test_elevation_solar_warms_during_day(sim_params: SimParams) -> None:
    """Elevation-based solar gains should produce warmer daytime predictions."""
    # State at noon with solar elevations (daytime)
    n_steps = 72  # 6 hours
    sin_elevations_day = [0.7] * n_steps  # high sun
    sin_elevations_night = [0.0] * n_steps  # no sun

    state_day = _make_state(hour=12.0)
    state_day = HouseState(
        current_temps=state_day.current_temps,
        outdoor_temp=state_day.outdoor_temp,
        forecast_temps=state_day.forecast_temps,
        environment_states=state_day.environment_states,
        hour_of_day=12.0,
        solar_fractions=[1.0] * 13,  # sunny
        solar_elevations=sin_elevations_day,
    )
    state_night = HouseState(
        current_temps=state_day.current_temps,
        outdoor_temp=state_day.outdoor_temp,
        forecast_temps=state_day.forecast_temps,
        environment_states=state_day.environment_states,
        hour_of_day=0.0,
        solar_fractions=[0.0] * 13,
        solar_elevations=sin_elevations_night,
    )

    targets_d, preds_d = predict(state_day, [_all_off()], sim_params, [72])
    targets_n, preds_n = predict(state_night, [_all_off()], sim_params, [72])

    # Day predictions should be warmer than night for sensors with solar gain
    for i, name in enumerate(targets_d):
        if "piano_temp" in name:
            assert preds_d[0, i] > preds_n[0, i], (
                f"Piano should be warmer during day: day={preds_d[0, i]:.1f} night={preds_n[0, i]:.1f}"
            )


# ── Advisory timeline tests ──────────────────────────────────────────────


class TestBuildAdvisoryTimelines:
    """Tests for _build_environment_timelines()."""

    def test_empty_advisories_returns_empty(self) -> None:
        """No advisory decisions → empty dict."""
        scenarios = [Scenario(effectors={}), Scenario(effectors={})]
        result = build_advisory_timelines(scenarios, {"win": True}, n_future=73)
        assert result == {}

    def test_hold_keeps_current_state(self) -> None:
        """Hold decision keeps device at current state for all steps."""
        scenarios = [Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="hold")},
        )]
        result = build_advisory_timelines(scenarios, {"win": True}, n_future=10)
        assert "win" in result
        # Current state is True → 1.0 everywhere
        np.testing.assert_array_equal(result["win"][0], 1.0)

    def test_close_transitions_at_step(self) -> None:
        """Close decision transitions from 1.0 to 0.0 at transition_step."""
        scenarios = [Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="close", transition_step=5)},
        )]
        tl = build_advisory_timelines(scenarios, {"win": True}, n_future=10)
        arr = tl["win"][0]
        # History (18 steps) + first 5 future steps = 1.0
        assert all(arr[i] == 1.0 for i in range(18 + 5))
        # From step 5 onward = 0.0
        assert all(arr[i] == 0.0 for i in range(18 + 5, 18 + 10))

    def test_open_transitions_at_step(self) -> None:
        """Open decision transitions from 0.0 to 1.0 at transition_step."""
        scenarios = [Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="open", transition_step=3)},
        )]
        tl = build_advisory_timelines(scenarios, {"win": False}, n_future=10)
        arr = tl["win"][0]
        # History + first 3 future steps = 0.0
        assert all(arr[i] == 0.0 for i in range(18 + 3))
        # From step 3 onward = 1.0
        assert all(arr[i] == 1.0 for i in range(18 + 3, 18 + 10))

    def test_multiple_scenarios_independent(self) -> None:
        """Each scenario gets its own timeline row."""
        scenarios = [
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="close", transition_step=0)}),
        ]
        tl = build_advisory_timelines(scenarios, {"win": True}, n_future=10)
        arr = tl["win"]
        assert arr.shape == (2, 18 + 10)
        # Scenario 0: hold at open (1.0 everywhere)
        np.testing.assert_array_equal(arr[0], 1.0)
        # Scenario 1: close at step 0 (history=1.0, future=0.0)
        assert all(arr[1, i] == 1.0 for i in range(18))
        assert all(arr[1, i] == 0.0 for i in range(18, 28))

    def test_return_step_reverts_to_original(self) -> None:
        """return_step causes device to revert to original state."""
        scenarios = [Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="open", transition_step=0, return_step=5)},
        )]
        tl = build_advisory_timelines(scenarios, {"win": False}, n_future=10)
        arr = tl["win"][0]
        # History: 0.0 (closed)
        assert all(arr[i] == 0.0 for i in range(18))
        # Steps 0-4: 1.0 (open)
        assert all(arr[i] == 1.0 for i in range(18, 18 + 5))
        # Steps 5+: 0.0 (reverted to original closed state)
        assert all(arr[i] == 0.0 for i in range(18 + 5, 18 + 10))

    def test_unknown_device_defaults_to_inactive(self) -> None:
        """Device not in environment_states defaults to 0.0."""
        scenarios = [Scenario(
            effectors={},
            advisories={"heater": AdvisoryDecision("heater", action="turn_on", transition_step=2)},
        )]
        tl = build_advisory_timelines(scenarios, {}, n_future=10)
        arr = tl["heater"][0]
        # Default is 0.0, turns on at step 2
        assert all(arr[i] == 0.0 for i in range(18 + 2))
        assert all(arr[i] == 1.0 for i in range(18 + 2, 18 + 10))


# ── Advisory effects in predict() ────────────────────────────────────────


class TestAdvisoryPredict:
    """Tests for advisory effects in the vectorized predict() path."""

    def test_advisory_close_changes_trajectory(self, sim_params: SimParams) -> None:
        """Advisory close at step 12 produces different predictions than hold.

        With an open window (lower tau → faster cooling toward outdoor),
        closing it mid-horizon should produce warmer predictions at later
        horizons compared to holding it open.
        """
        # Inject window betas so opening/closing has an effect
        augmented_taus: dict[str, TauModel] = {}
        for sensor, tau_model in sim_params.taus.items():
            augmented_taus[sensor] = TauModel(
                tau_base=tau_model.tau_base,
                environment_tau_betas={"test_window": 0.03, **tau_model.environment_tau_betas},
            )
        augmented = SimParams(
            taus=augmented_taus,
            gains=sim_params.gains,
            solar=sim_params.solar,
            sensors=sim_params.sensors,
            effectors=sim_params.effectors,
            solar_elevation_gains=sim_params.solar_elevation_gains,
            environment_solar_betas=sim_params.environment_solar_betas,
        )

        state = HouseState(
            current_temps=_CURRENT_TEMPS,
            outdoor_temp=42.0,
            forecast_temps=[42.0] * 12,
            environment_states={"test_window": True},  # currently open
            hour_of_day=2.0,
            solar_fractions=[0.0] * 12,
            solar_elevations=[0.0] * 72,
        )

        hold_open = Scenario(
            effectors={},
            advisories={"test_window": AdvisoryDecision("test_window", action="hold")},
        )
        close_at_12 = Scenario(
            effectors={},
            advisories={"test_window": AdvisoryDecision("test_window", action="close", transition_step=12)},
        )

        targets, preds = predict(state, [hold_open, close_at_12], augmented, [12, 48, 72])

        # At step 12 (1 hour): closing just happened, predictions should be very similar
        # At step 48 (4 hours) and 72 (6 hours): closed window retains heat → warmer
        for j, name in enumerate(targets):
            if "t+72" in name:
                assert preds[1, j] > preds[0, j], (
                    f"{name}: closing window should produce warmer 6h prediction "
                    f"(hold={preds[0, j]:.2f} close={preds[1, j]:.2f})"
                )

    def test_no_advisory_fast_path_unchanged(self, sim_params: SimParams) -> None:
        """Scenarios without advisories produce identical results to pre-advisory code.

        This verifies the fast path: when no scenario has advisories, the
        code uses scalar tau (no per-step matrix overhead).
        """
        state = _make_state(hour=2.0)
        scenarios = [_all_off(), _both_on()]
        targets, preds = predict(state, scenarios, sim_params, [12, 24, 48, 72])

        # Run again — deterministic, should be identical
        _, preds2 = predict(state, scenarios, sim_params, [12, 24, 48, 72])
        np.testing.assert_array_equal(preds, preds2)

        # All-off should cool, both-on should warm
        for j, name in enumerate(targets):
            if "upstairs" in name and "t+72" in name:
                assert preds[1, j] > preds[0, j]

    def test_advisory_solar_modulation(self, sim_params: SimParams) -> None:
        """Advisory solar betas modulate solar forcing per-scenario.

        A device with negative solar beta (like blinds closing reduces solar)
        should produce cooler daytime predictions when active.
        """
        # Augment sim_params with advisory solar betas for piano_temp
        augmented = SimParams(
            taus=sim_params.taus,
            gains=sim_params.gains,
            solar=sim_params.solar,
            sensors=sim_params.sensors,
            effectors=sim_params.effectors,
            solar_elevation_gains=sim_params.solar_elevation_gains,
            environment_solar_betas={"blinds": {"piano_temp": -0.5}},
        )

        state = HouseState(
            current_temps=_CURRENT_TEMPS,
            outdoor_temp=60.0,
            forecast_temps=[60.0] * 12,
            environment_states={"blinds": False},
            hour_of_day=12.0,
            solar_fractions=[1.0] * 12,
            solar_elevations=[0.7] * 72,
        )

        no_blinds = Scenario(effectors={})  # no advisories → blinds stay off
        close_blinds = Scenario(
            effectors={},
            advisories={"blinds": AdvisoryDecision("blinds", action="turn_on", transition_step=0)},
        )

        targets, preds = predict(state, [no_blinds, close_blinds], augmented, [72])
        piano_idx = next(j for j, t in enumerate(targets) if "piano_temp_t+72" in t)

        # Blinds active with beta=-0.5 → solar reduced by 50% → cooler
        assert preds[0, piano_idx] > preds[1, piano_idx], (
            f"Active blinds should reduce solar warming: "
            f"no_blinds={preds[0, piano_idx]:.2f} blinds={preds[1, piano_idx]:.2f}"
        )
