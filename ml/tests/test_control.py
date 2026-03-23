"""Tests for control optimizer constraints.

Tests the sweep constraints without HA or real models. Verifies the
**constraint logic**: comfort cost, energy cost, cautious setpoints,
trajectory generation, window schedule adjustment, and quiet hours.
"""

from __future__ import annotations

from weatherstat.config import EFFECTOR_MAP, EFFECTORS
from weatherstat.control import (
    ABSOLUTE_MAX,
    ABSOLUTE_MIN,
    CONTROL_HORIZONS,
    HORIZON_WEIGHTS,
    _cautious_setpoint,
    _in_hold_window,
    _in_quiet_hours,
    _regulating_sweep_options,
    adjust_schedules_for_windows,
    apply_mrt_correction,
    compute_comfort_cost,
    compute_energy_cost,
    generate_trajectory_scenarios,
)
from weatherstat.types import (
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlState,
    EffectorDecision,
    RoomComfort,
    Scenario,
)

# ── Test helpers ──────────────────────────────────────────────────────────


def make_schedules(**overrides: list[ComfortScheduleEntry]) -> list[ComfortSchedule]:
    """Return default_comfort_schedules() with optional room overrides.

    Usage:
        make_schedules(upstairs=[ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 72, 70, 74))])
    """
    from weatherstat.control import default_comfort_schedules

    defaults = {s.label: s for s in default_comfort_schedules()}
    for label, entries in overrides.items():
        defaults[label] = ComfortSchedule(label=label, entries=tuple(entries))
    return list(defaults.values())


# ── Comfort cost tests ────────────────────────────────────────────────────


class TestComfortCost:
    """Test comfort cost computation."""

    def test_comfort_cost_continuous_plus_rail(self) -> None:
        """Room at 76F, preferred 72F, max 74F, hot_penalty 2.0.

        Cost = continuous(76-72)^2*2.0*weight + rail(76-74)^2*2.0*10*weight.
        """
        schedules = [
            ComfortSchedule(
                label="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 72.0, 70.0, 74.0, hot_penalty=2.0)),),
            ),
        ]
        predictions = {"upstairs_temp_t+12": 76.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)

        w = HORIZON_WEIGHTS[12]
        continuous = (76.0 - 72.0) ** 2 * 2.0 * w  # deviation from preferred
        rail = (76.0 - 74.0) ** 2 * 2.0 * 10.0 * w  # exceeds hard max
        expected = continuous + rail
        assert abs(cost - expected) < 0.001

    def test_comfort_cost_respects_schedule_hour(self) -> None:
        """Bedroom at hour 22 (night: preferred 69F) vs hour 10 (day: preferred 70F).

        At 74F: night schedule has higher cold_penalty AND 74 exceeds night max (72),
        so night cost should be much higher than day cost. Day cost is non-zero
        (74 > preferred 70) but has no hard rail penalty (74 < max 75).
        """
        from weatherstat.control import default_comfort_schedules

        schedules = default_comfort_schedules()
        bedroom_schedules = [s for s in schedules if s.label == "bedroom"]

        predictions = {"bedroom_temp_t+12": 74.0}

        # Hour 22 (night: preferred 69, max 72) -> 74 > max, big penalty
        cost_night = compute_comfort_cost(predictions, bedroom_schedules, base_hour=22)

        # Hour 10 (day: preferred 70, max 75) -> 74 > preferred but < max, small penalty
        cost_day = compute_comfort_cost(predictions, bedroom_schedules, base_hour=10)

        assert cost_night > cost_day
        assert cost_day > 0.0  # continuous cost: 74 is above preferred (70)

    def test_comfort_cost_zero_at_preferred(self) -> None:
        """Room exactly at preferred temperature -> zero cost."""
        schedules = [
            ComfortSchedule(
                label="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 71.0, 69.0, 74.0)),),
            ),
        ]
        predictions = {"upstairs_temp_t+12": 71.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)
        assert cost == 0.0

    def test_comfort_cost_within_band_nonzero(self) -> None:
        """Room within [min, max] but not at preferred -> non-zero continuous cost."""
        schedules = [
            ComfortSchedule(
                label="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 71.0, 69.0, 74.0)),),
            ),
        ]
        predictions = {"upstairs_temp_t+12": 73.0}  # in band but above preferred
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)
        assert cost > 0.0  # continuous deviation cost

    def test_energy_cost_tiebreaker(self) -> None:
        """Two scenarios with identical comfort cost -> lower energy wins."""
        scenario_both = Scenario(effectors={
            "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating"),
            "thermostat_downstairs": EffectorDecision("thermostat_downstairs", mode="heating"),
        })
        scenario_one = Scenario(effectors={
            "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating"),
        })

        cost_both = compute_energy_cost(scenario_both)
        cost_one = compute_energy_cost(scenario_one)

        assert cost_both > cost_one


# ── Cautious setpoint tests ──────────────────────────────────────────────


class TestCautiousSetpoint:
    """Test cautious setpoint computation and clamping."""

    def test_cautious_setpoint_clamped(self) -> None:
        """Current temp 77F + offset 2 = 79F -> clamped to ABSOLUTE_MAX (78F)."""
        sp = _cautious_setpoint(77.0, heating=True)
        assert sp == ABSOLUTE_MAX

    def test_cautious_setpoint_heating(self) -> None:
        """Normal heating: current + offset."""
        sp = _cautious_setpoint(70.0, heating=True)
        assert sp == 72.0

    def test_cautious_setpoint_off_uses_comfort_min(self) -> None:
        """Heating off: setpoint = comfort_min - 1 (safety net just below comfort floor)."""
        sp = _cautious_setpoint(70.0, heating=False, comfort_min=70.0)
        assert sp == 69.0

    def test_cautious_setpoint_off_defaults_to_absolute_min(self) -> None:
        """Heating off without comfort_min: defaults to ABSOLUTE_MIN."""
        sp = _cautious_setpoint(70.0, heating=False)
        assert sp == ABSOLUTE_MIN

    def test_cautious_setpoint_heating_respects_comfort_min(self) -> None:
        """Heating on: setpoint = max(current + offset, comfort_min + offset)."""
        # Current temp below comfort min — setpoint should use comfort_min + offset
        sp = _cautious_setpoint(67.0, heating=True, comfort_min=70.0)
        assert sp == 72.0


# ── Window comfort adjustment tests ─────────────────────────────────────


class TestAdjustSchedulesForWindows:
    """Test comfort schedule adjustment when windows are open."""

    def test_no_open_windows_returns_unchanged(self) -> None:
        """All windows closed -> schedules unchanged."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules()
        window_states = {name: False for name in cfg.windows}

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, {c.label for c in cfg.constraints}, -3.0, 2.0,
        )
        # Same number of schedules, same entries
        assert len(adjusted) == len(schedules)
        for orig, adj in zip(schedules, adjusted, strict=True):
            assert orig is adj  # should be the exact same object

    def test_open_window_widens_comfort_bounds(self) -> None:
        """Bedroom window open -> bedroom schedule shifted by offset."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules(
            bedroom=[
                ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 70.0, 74.0, 2.0, 1.0)),
            ],
        )
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, {c.label for c in cfg.constraints}, -3.0, 2.0,
        )

        bedroom_sched = next(s for s in adjusted if s.label == "bedroom")
        entry = bedroom_sched.entries[0]
        assert entry.comfort.min_temp == 67.0  # 70 + (-3)
        assert entry.comfort.max_temp == 76.0  # 74 + 2

    def test_unrelated_room_not_affected(self) -> None:
        """Bedroom window open -> upstairs schedule untouched."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules()
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, {c.label for c in cfg.constraints}, -3.0, 2.0,
        )

        upstairs_orig = next(s for s in schedules if s.label == "upstairs")
        upstairs_adj = next(s for s in adjusted if s.label == "upstairs")
        assert upstairs_orig is upstairs_adj  # unchanged

    def test_penalties_preserved(self) -> None:
        """Window adjustment preserves cold_penalty and hot_penalty."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules(
            bedroom=[
                ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 70.0, 74.0, 3.0, 0.5)),
            ],
        )
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, {c.label for c in cfg.constraints}, -3.0, 2.0,
        )

        bedroom_sched = next(s for s in adjusted if s.label == "bedroom")
        entry = bedroom_sched.entries[0]
        assert entry.comfort.cold_penalty == 3.0
        assert entry.comfort.hot_penalty == 0.5

    def test_window_without_constraint_no_effect(self) -> None:
        """Basement window (no matching constraint) open -> no schedules affected."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules()
        window_states = {name: False for name in cfg.windows}
        window_states["basement"] = True  # basement has rooms=[]

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, {c.label for c in cfg.constraints}, -3.0, 2.0,
        )

        for orig, adj in zip(schedules, adjusted, strict=True):
            assert orig is adj


# ── MRT correction tests ──────────────────────────────────────────────────


class TestMrtCorrection:
    """Test mean radiant temperature correction for comfort targets."""

    @staticmethod
    def _cfg(alpha: float = 0.1, reference_temp: float = 50.0, max_offset: float = 3.0):
        from weatherstat.yaml_config import MrtCorrectionConfig

        return MrtCorrectionConfig(alpha=alpha, reference_temp=reference_temp, max_offset=max_offset)

    def test_cold_day_raises_targets(self) -> None:
        """35°F outside, ref=50, alpha=0.1 → +1.5°F offset."""
        schedules = [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 70.0, 74.0)),),
        )]
        adjusted, offset = apply_mrt_correction(schedules, 35.0, self._cfg())
        assert abs(offset - 1.5) < 0.01
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred - 73.5) < 0.01
        assert abs(entry.comfort.min_temp - 71.5) < 0.01
        assert abs(entry.comfort.max_temp - 75.5) < 0.01

    def test_warm_day_lowers_targets(self) -> None:
        """80°F outside, ref=50, alpha=0.1 → clamped to -3.0°F."""
        schedules = [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 70.0, 74.0)),),
        )]
        adjusted, offset = apply_mrt_correction(schedules, 80.0, self._cfg())
        assert abs(offset - (-3.0)) < 0.01  # clamped
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred - 69.0) < 0.01

    def test_at_reference_no_change(self) -> None:
        """Outdoor = reference → no correction."""
        schedules = make_schedules()
        adjusted, offset = apply_mrt_correction(schedules, 50.0, self._cfg())
        assert offset == 0.0
        assert adjusted is schedules  # exact same object

    def test_clamped_at_max(self) -> None:
        """0°F outside → raw offset 5.0, clamped to 3.0."""
        schedules = [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 70.0, 74.0)),),
        )]
        adjusted, offset = apply_mrt_correction(schedules, 0.0, self._cfg())
        assert abs(offset - 3.0) < 0.01
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred - 75.0) < 0.01

    def test_none_config_no_change(self) -> None:
        """mrt_config=None → schedules returned unchanged."""
        schedules = make_schedules()
        adjusted, offset = apply_mrt_correction(schedules, 35.0, None)
        assert offset == 0.0
        assert adjusted is schedules

    def test_penalties_preserved(self) -> None:
        """MRT correction shifts temps but preserves penalty weights."""
        schedules = [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 70.0, 74.0, 3.0, 0.5)),),
        )]
        adjusted, _ = apply_mrt_correction(schedules, 35.0, self._cfg())
        entry = adjusted[0].entries[0]
        assert entry.comfort.cold_penalty == 3.0
        assert entry.comfort.hot_penalty == 0.5

    def test_mrt_weight_scales_offset(self) -> None:
        """Per-sensor weight=0.5 produces half the offset."""
        schedules = [ComfortSchedule(
            label="piano",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 70.0, 74.0)),),
        )]
        # Base offset: 0.1 * (50 - 35) = 1.5°F; weighted: 1.5 * 0.5 = 0.75°F
        adjusted, base_offset = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"piano": 0.5},
        )
        assert abs(base_offset - 1.5) < 0.01
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred - 72.75) < 0.01
        assert abs(entry.comfort.min_temp - 70.75) < 0.01

    def test_mrt_weight_zero_no_adjustment(self) -> None:
        """Per-sensor weight=0.0 suppresses MRT correction for that sensor."""
        schedules = [ComfortSchedule(
            label="piano",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 70.0, 74.0)),),
        )]
        adjusted, base_offset = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"piano": 0.0},
        )
        assert abs(base_offset - 1.5) < 0.01
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred - 72.0) < 0.01  # unchanged

    def test_mrt_different_weights_per_sensor(self) -> None:
        """Two sensors with different weights get different offsets."""
        schedules = [
            ComfortSchedule(
                label="piano",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 70.0, 74.0)),),
            ),
            ComfortSchedule(
                label="bathroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bathroom", 72.0, 70.0, 74.0)),),
            ),
        ]
        # Base offset: 1.5°F; piano×0.5=0.75, bathroom×1.5=2.25
        adjusted, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"piano": 0.5, "bathroom": 1.5},
        )
        piano_pref = adjusted[0].entries[0].comfort.preferred
        bath_pref = adjusted[1].entries[0].comfort.preferred
        assert abs(piano_pref - 72.75) < 0.01
        assert abs(bath_pref - 74.25) < 0.01

    def test_mrt_weight_default_backward_compat(self) -> None:
        """No weights dict (None) produces same behavior as uniform weight=1.0."""
        schedules = [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 70.0, 74.0)),),
        )]
        adj_none, off_none = apply_mrt_correction(schedules, 35.0, self._cfg())
        adj_ones, off_ones = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"bedroom": 1.0},
        )
        assert off_none == off_ones
        assert adj_none[0].entries[0].comfort.preferred == adj_ones[0].entries[0].comfort.preferred


# ── Derived MRT weight tests ─────────────────────────────────────────────


class TestComputeMrtWeights:
    """Test MRT weight derivation from solar gain profiles."""

    @staticmethod
    def _make_solar(sensor: str, total_gain: float, n_hours: int = 10, t_stat: float = 3.0):
        from weatherstat.sysid import SolarGainProfile

        gain_per_hour = total_gain / n_hours if n_hours > 0 else 0.0
        return [
            SolarGainProfile(
                sensor=sensor,
                hour_of_day=7 + i,
                gain_f_per_hour=gain_per_hour,
                std_error=0.05,
                t_statistic=t_stat,
            )
            for i in range(n_hours)
        ]

    def test_high_solar_low_weight(self) -> None:
        """Sensor with 2x mean solar gain gets low weight."""
        from weatherstat.sysid import _compute_mrt_weights

        gains = self._make_solar("piano_temp", 5.0) + self._make_solar("bedroom_temp", 1.0)
        # mean of nonzero = (5+1)/2 = 3.0; piano ratio=5/3=1.67; weight=2-1.67=0.33
        weights = _compute_mrt_weights(gains, ["piano_temp", "bedroom_temp"])
        assert weights["piano_temp"] < 1.0
        assert weights["piano_temp"] >= 0.3

    def test_zero_solar_high_weight(self) -> None:
        """Sensor with zero solar gain gets weight 2.0."""
        from weatherstat.sysid import _compute_mrt_weights

        gains = (
            self._make_solar("piano_temp", 3.0)
            + self._make_solar("bathroom_temp", 0.0, t_stat=0.5)  # below t-stat threshold
        )
        weights = _compute_mrt_weights(gains, ["piano_temp", "bathroom_temp"])
        assert weights["bathroom_temp"] == 2.0
        assert weights["piano_temp"] == 1.0  # only nonzero sensor → ratio=1 → weight=1

    def test_average_sensor_gets_one(self) -> None:
        """Sensor at mean solar gain gets weight 1.0."""
        from weatherstat.sysid import _compute_mrt_weights

        # All three sensors with same solar gain → mean equals each → ratio=1 → weight=1
        gains = (
            self._make_solar("a_temp", 2.0)
            + self._make_solar("b_temp", 2.0)
            + self._make_solar("c_temp", 2.0)
        )
        weights = _compute_mrt_weights(gains, ["a_temp", "b_temp", "c_temp"])
        assert abs(weights["a_temp"] - 1.0) < 0.01

    def test_weight_clamped_low(self) -> None:
        """Very high solar gain clamped to 0.3 minimum."""
        from weatherstat.sysid import _compute_mrt_weights

        gains = self._make_solar("piano_temp", 10.0) + self._make_solar("bedroom_temp", 0.5)
        weights = _compute_mrt_weights(gains, ["piano_temp", "bedroom_temp"])
        assert weights["piano_temp"] >= 0.3

    def test_no_solar_data_returns_empty(self) -> None:
        """No significant solar gains → empty dict (no derived weights)."""
        from weatherstat.sysid import _compute_mrt_weights

        gains = self._make_solar("piano_temp", 0.0, t_stat=0.5)
        weights = _compute_mrt_weights(gains, ["piano_temp"])
        assert weights == {}

    def test_unconstrained_sensor_ignored(self) -> None:
        """Sensors not in constrained list are excluded."""
        from weatherstat.sysid import _compute_mrt_weights

        gains = self._make_solar("outdoor_temp", 5.0) + self._make_solar("piano_temp", 2.0)
        weights = _compute_mrt_weights(gains, ["piano_temp"])
        assert "outdoor_temp" not in weights
        assert "piano_temp" in weights


# ── Quiet hours tests ─────────────────────────────────────────────────────


class TestQuietHours:
    """Test quiet hours helper."""

    def test_within_quiet_hours_wrapping(self) -> None:
        """22:00-07:00 range wraps midnight. Hour 23, 0, 6 are quiet."""
        assert _in_quiet_hours(23, (22, 7)) is True
        assert _in_quiet_hours(0, (22, 7)) is True
        assert _in_quiet_hours(6, (22, 7)) is True

    def test_outside_quiet_hours_wrapping(self) -> None:
        """22:00-07:00 range. Hour 7, 12, 21 are not quiet."""
        assert _in_quiet_hours(7, (22, 7)) is False
        assert _in_quiet_hours(12, (22, 7)) is False
        assert _in_quiet_hours(21, (22, 7)) is False

    def test_non_wrapping_range(self) -> None:
        """8:00-18:00 non-wrapping range."""
        assert _in_quiet_hours(8, (8, 18)) is True
        assert _in_quiet_hours(12, (8, 18)) is True
        assert _in_quiet_hours(17, (8, 18)) is True
        assert _in_quiet_hours(18, (8, 18)) is False
        assert _in_quiet_hours(7, (8, 18)) is False

    def test_boundary_start(self) -> None:
        """Start hour is inclusive."""
        assert _in_quiet_hours(22, (22, 7)) is True

    def test_boundary_end(self) -> None:
        """End hour is exclusive."""
        assert _in_quiet_hours(7, (22, 7)) is False


# ── Trajectory scenario generation tests ─────────────────────────────


class TestTrajectoryScenarioGeneration:
    """Test trajectory scenario generation for physics sweep."""

    @staticmethod
    def _eff(s: Scenario, name: str) -> EffectorDecision | None:
        return s.effectors.get(name)

    def test_trajectory_scenarios_include_all_off(self) -> None:
        """All-off should be in the trajectory scenario set."""
        scenarios = generate_trajectory_scenarios()
        all_off = [
            s for s in scenarios
            if all(e.mode == "off" for e in s.effectors.values())
            or len(s.effectors) == 0
        ]
        assert len(all_off) == 1

    def test_trajectory_scenarios_include_delayed_start(self) -> None:
        """Trajectories with delay > 0 should exist."""
        scenarios = generate_trajectory_scenarios()
        delayed = [
            s for s in scenarios
            if (up := self._eff(s, "thermostat_upstairs")) is not None
            and up.mode != "off" and up.delay_steps > 0
        ]
        assert len(delayed) > 0

    def test_trajectory_scenarios_include_short_duration(self) -> None:
        """Trajectories with finite duration should exist."""
        scenarios = generate_trajectory_scenarios()
        short = [
            s for s in scenarios
            if (up := self._eff(s, "thermostat_upstairs")) is not None
            and up.mode != "off" and up.duration_steps is not None and up.duration_steps < 72
        ]
        assert len(short) > 0

    def test_trajectory_scenarios_nontrivial(self) -> None:
        """Trajectory search space should be substantial (more than just all-off)."""
        trajectory_count = len(generate_trajectory_scenarios())
        # With 2 zones × (1 off + delay×duration grid) × blowers × mini-splits,
        # expect hundreds of scenarios
        assert trajectory_count > 100

    def test_trajectory_no_delay_past_horizon(self) -> None:
        """No trajectory should have delay >= max horizon (would be equivalent to OFF)."""
        max_horizon = max(CONTROL_HORIZONS)
        scenarios = generate_trajectory_scenarios()
        for s in scenarios:
            for eff in EFFECTORS:
                if eff.control_type != "trajectory":
                    continue
                ed = self._eff(s, eff.name)
                if ed is not None and ed.mode != "off":
                    assert ed.delay_steps < max_horizon

    def test_trajectory_duration_capped_at_horizon(self) -> None:
        """delay + duration should not exceed max horizon."""
        max_horizon = max(CONTROL_HORIZONS)
        scenarios = generate_trajectory_scenarios()
        for s in scenarios:
            for eff in EFFECTORS:
                if eff.control_type != "trajectory":
                    continue
                ed = self._eff(s, eff.name)
                if ed is not None and ed.mode != "off" and ed.duration_steps is not None:
                    assert ed.delay_steps + ed.duration_steps <= max_horizon


class TestTrajectoryEnergyCost:
    """Test that trajectory energy cost scales with duration."""

    def test_shorter_duration_lower_cost(self) -> None:
        """2h trajectory should cost less than 6h trajectory."""
        short = Scenario(effectors={
            "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating", duration_steps=24),
        })
        long = Scenario(effectors={
            "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating", duration_steps=72),
        })
        assert compute_energy_cost(short) < compute_energy_cost(long)

    def test_off_trajectory_no_gas_cost(self) -> None:
        """All-off trajectory should have no gas zone cost."""
        off = Scenario(effectors={})
        assert compute_energy_cost(off) == 0.0

    def test_full_horizon_trajectory_energy_cost(self) -> None:
        """Full-horizon trajectory should cost one gas zone unit."""
        traj = Scenario(effectors={
            "thermostat_upstairs": EffectorDecision("thermostat_upstairs", mode="heating"),
        })
        from weatherstat.config import ENERGY_COST_GAS_ZONE
        assert compute_energy_cost(traj) == ENERGY_COST_GAS_ZONE


# ── Target grid sweep tests ──────────────────────────────────────────


class TestRegulatingSweepOptions:
    """Test target-based regulating effector sweep option generation."""

    @staticmethod
    def _bedroom_eff():
        return EFFECTOR_MAP["mini_split_bedroom"]

    def _bedroom_schedule(self, min_t: float = 68.0, max_t: float = 72.0) -> list[ComfortSchedule]:
        preferred = (min_t + max_t) / 2.0
        return [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", preferred, min_t, max_t)),),
        )]

    def test_sweep_options_include_off(self) -> None:
        """Off should always be an option."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(eff, self._bedroom_schedule(), 12, current_temps={"bedroom": 68.0})
        assert any(o.mode == "off" for o in options)

    def test_sweep_options_preferred_target(self) -> None:
        """Should generate off + preferred target."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(eff, self._bedroom_schedule(), 12, current_temps={"bedroom": 68.0})
        assert len(options) == 2  # off + preferred
        active = [o for o in options if o.mode != "off"]
        assert len(active) == 1
        assert active[0].target == 70.0  # preferred = midpoint of 68-72

    def test_sweep_mode_heat_when_cold(self) -> None:
        """Mode should be 'heat' when room temp is below preferred."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(eff, self._bedroom_schedule(), 12, current_temps={"bedroom": 68.0})
        active = [o for o in options if o.mode != "off"]
        assert all(o.mode == "heat" for o in active)

    def test_sweep_mode_cool_when_hot(self) -> None:
        """Mode should be 'cool' when room temp is above preferred."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(eff, self._bedroom_schedule(), 12, current_temps={"bedroom": 72.0})
        active = [o for o in options if o.mode != "off"]
        assert all(o.mode == "cool" for o in active)

    def test_sweep_no_schedule_returns_off_only(self) -> None:
        """No matching schedule -> only off option."""
        options = _regulating_sweep_options(self._bedroom_eff(), [], 12)
        assert len(options) == 1
        assert options[0].mode == "off"


# ── Mode hold window tests ──────────────────────────────────────────


class TestModeHoldWindow:
    """Test mode hold window constraints (quiet hours only)."""

    def _bedroom_schedule(self) -> list[ComfortSchedule]:
        return [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 68.0, 72.0)),),
        )]

    def test_in_hold_window_wraps_midnight(self) -> None:
        """22:00-07:00 window wraps midnight correctly."""
        assert _in_hold_window(23, (22, 7)) is True
        assert _in_hold_window(0, (22, 7)) is True
        assert _in_hold_window(6, (22, 7)) is True
        assert _in_hold_window(7, (22, 7)) is False
        assert _in_hold_window(12, (22, 7)) is False

    def test_mode_locked_during_hold_window(self) -> None:
        """During hold window, options are filtered to current mode."""
        from datetime import UTC, datetime

        prev_state = ControlState(
            last_decision_time=datetime.now(UTC).isoformat(),
            modes={"mini_split_bedroom": "heat"},
            setpoints={"mini_split_bedroom": 70.0},
            mode_times={"mini_split_bedroom": datetime.now(UTC).isoformat()},
        )
        eff = EFFECTOR_MAP["mini_split_bedroom"]
        # Hour 23 is inside bedroom's hold window [22, 7]
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 23, prev_state, current_temps={"bedroom": 68.0},
        )
        # All options should be "heat" (current mode)
        assert all(o.mode == "heat" for o in options)
        assert len(options) >= 1

    def test_mode_not_locked_outside_hold_window(self) -> None:
        """Outside hold window, mode can change freely."""
        from datetime import UTC, datetime

        prev_state = ControlState(
            last_decision_time=datetime.now(UTC).isoformat(),
            modes={"mini_split_bedroom": "heat"},
            setpoints={"mini_split_bedroom": 70.0},
            mode_times={"mini_split_bedroom": datetime.now(UTC).isoformat()},
        )
        eff = EFFECTOR_MAP["mini_split_bedroom"]
        # Hour 12 is outside hold window — mode should be unlocked
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12, prev_state, current_temps={"bedroom": 68.0},
        )
        assert any(o.mode == "off" for o in options)

    def test_cool_offered_when_room_above_target(self) -> None:
        """Cool option offered when room is above preferred."""
        eff = EFFECTOR_MAP["mini_split_bedroom"]
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            current_temps={"bedroom": 73.0},  # 3 deg F above preferred 70
        )
        # Room is above preferred — cool mode should be offered
        assert any(o.mode == "cool" for o in options)

    def test_split_offered_when_room_near_target(self) -> None:
        """Cool option available when room is slightly above preferred."""
        eff = EFFECTOR_MAP["mini_split_bedroom"]
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            current_temps={"bedroom": 70.5},  # within proportional_band of preferred 70
        )
        # Room is 0.5 deg F above preferred — should get cool option
        assert any(o.mode == "cool" for o in options)


# ── Proportional energy cost tests ──────────────────────────────────


class TestProportionalEnergyCost:
    """Test proportional energy cost for mini splits."""

    def test_higher_target_more_energy_when_heating(self) -> None:
        """Higher target vs room temp -> more expected activity -> higher cost.

        With proportional_band=1.0 and room at 68F:
        target 68.5 -> delta=0.5 -> activity=0.5,
        target 69.0 -> delta=1.0 -> activity=1.0.
        """
        low_target = Scenario(effectors={
            "mini_split_bedroom": EffectorDecision("mini_split_bedroom", mode="heat", target=68.5),
        })
        high_target = Scenario(effectors={
            "mini_split_bedroom": EffectorDecision("mini_split_bedroom", mode="heat", target=69.0),
        })
        temps = {"bedroom": 68.0}
        assert compute_energy_cost(low_target, temps) < compute_energy_cost(high_target, temps)

    def test_off_mini_split_no_energy_cost(self) -> None:
        """Off mini split should have zero energy cost."""
        off = Scenario(effectors={})
        assert compute_energy_cost(off) == 0.0


# ── Trajectory scenario generation with schedules ──────────────────


class TestTrajectoryWithSchedules:
    """Test trajectory generation with comfort schedule integration."""

    def test_scenario_count_with_schedules(self) -> None:
        """With schedules, should have 2 options per split (off + preferred)."""
        schedules = [
            ComfortSchedule(
                label="bedroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 68.0, 72.0)),),
            ),
            ComfortSchedule(
                label="living_room",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("living_room", 71.0, 69.0, 74.0)),),
            ),
        ]
        # Room temps below preferred -> heat mode for both splits
        current_temps = {"bedroom": 68.0, "living_room": 69.0}
        scenarios = generate_trajectory_scenarios(schedules, base_hour=12, current_temps=current_temps)
        # 4 mini split combos (2x2): off + preferred per split
        split_combos = set()
        for s in scenarios:
            combo = tuple(
                (name, ed.mode, ed.target)
                for name, ed in sorted(s.effectors.items())
                if EFFECTOR_MAP.get(name) and EFFECTOR_MAP[name].control_type == "regulating"
            )
            split_combos.add(combo)
        assert len(split_combos) == 4

    def test_all_off_still_present(self) -> None:
        """All-off baseline should still be in scenario set."""
        schedules = [
            ComfortSchedule(
                label="bedroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 68.0, 72.0)),),
            ),
        ]
        current_temps = {"bedroom": 68.0}
        scenarios = generate_trajectory_scenarios(schedules, base_hour=12, current_temps=current_temps)
        all_off = [
            s for s in scenarios
            if all(e.mode == "off" for e in s.effectors.values())
            or len(s.effectors) == 0
        ]
        assert len(all_off) == 1


# ── Effector eligibility tests ───────────────────────────────────────────


class TestEffectorEligibility:
    """Test the effector eligibility gate."""

    def test_thermostat_off_ineligible(self) -> None:
        """Thermostat with hvac_mode off is ineligible."""
        from unittest.mock import patch

        from weatherstat.control import check_effector_eligibility

        # _fetch_entity_state returns hvac_mode for thermostat, then state_device state
        def mock_fetch(entity_id: str) -> str:
            if "upstairs" in entity_id:
                return "off"
            if "downstairs" in entity_id:
                return "heat"
            return "Idle"  # state_device

        with patch("weatherstat.control._fetch_entity_state", side_effect=mock_fetch):
            result = check_effector_eligibility()
        assert "thermostat_upstairs" in result
        assert "thermostat_downstairs" not in result
        assert "off" in result["thermostat_upstairs"]

    def test_thermostat_heat_eligible(self) -> None:
        """Thermostat with hvac_mode=heat is eligible."""
        from unittest.mock import patch

        from weatherstat.control import check_effector_eligibility

        def mock_fetch(entity_id: str) -> str:
            if "thermostat" in entity_id:
                return "heat"
            return "Idle"  # state_device

        with patch("weatherstat.control._fetch_entity_state", side_effect=mock_fetch):
            result = check_effector_eligibility()
        assert result == {}

    def test_both_off_both_ineligible(self) -> None:
        """Both thermostats off -> both ineligible."""
        from unittest.mock import patch

        from weatherstat.control import check_effector_eligibility

        with patch("weatherstat.control._fetch_entity_state", return_value="off"):
            result = check_effector_eligibility()
        assert "thermostat_upstairs" in result
        assert "thermostat_downstairs" in result

    def test_state_device_unavailable_ineligible(self) -> None:
        """State device reporting unavailable -> effector ineligible."""
        from unittest.mock import patch

        from weatherstat.control import check_effector_eligibility

        def mock_fetch(entity_id: str) -> str:
            if "thermostat" in entity_id:
                return "heat"
            return "unavailable"  # state_device

        with patch("weatherstat.control._fetch_entity_state", side_effect=mock_fetch):
            result = check_effector_eligibility()
        # Both thermostats share the same state_device, both ineligible
        assert "thermostat_upstairs" in result
        assert "thermostat_downstairs" in result
        assert "unavailable" in result["thermostat_upstairs"]

    def test_state_device_functional_eligible(self) -> None:
        """State device reporting a valid state → zone eligible."""
        from unittest.mock import patch

        from weatherstat.control import check_effector_eligibility

        def mock_fetch(entity_id: str) -> str:
            if "thermostat" in entity_id:
                return "heat"
            return "Idle"

        with patch("weatherstat.control._fetch_entity_state", side_effect=mock_fetch):
            result = check_effector_eligibility()
        assert result == {}

    def test_scenarios_reduced_when_ineligible(self) -> None:
        """Ineligible effector produces fewer scenarios with no heating for that effector."""
        schedules = make_schedules()
        all_scenarios = generate_trajectory_scenarios(schedules, base_hour=12)
        reduced = generate_trajectory_scenarios(schedules, base_hour=12, ineligible_effectors={"thermostat_upstairs"})

        assert len(reduced) < len(all_scenarios)
        # No upstairs heating in any scenario
        up_heating = [
            s for s in reduced
            if (ed := s.effectors.get("thermostat_upstairs")) is not None and ed.mode != "off"
        ]
        assert len(up_heating) == 0
        # Downstairs still has heating options
        dn_heating = [
            s for s in reduced
            if (ed := s.effectors.get("thermostat_downstairs")) is not None and ed.mode != "off"
        ]
        assert len(dn_heating) > 0

    def test_scenarios_both_effectors_ineligible(self) -> None:
        """Both trajectory effectors ineligible -- only regulating/binary variations remain."""
        schedules = make_schedules()
        reduced = generate_trajectory_scenarios(
            schedules, base_hour=12, ineligible_effectors={"thermostat_upstairs", "thermostat_downstairs"},
        )
        # No thermostat heating in any scenario
        for s in reduced:
            for name in ("thermostat_upstairs", "thermostat_downstairs"):
                ed = s.effectors.get(name)
                assert ed is None or ed.mode == "off"
        # Should still have mini-split variations
        assert len(reduced) >= 1
        # All blowers should be off (no zone heating -> blowers forced off)
        for s in reduced:
            for name, ed in s.effectors.items():
                if EFFECTOR_MAP.get(name) and EFFECTOR_MAP[name].control_type == "binary":
                    assert ed.mode == "off"
