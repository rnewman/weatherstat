"""Tests for control optimizer constraints.

Tests the sweep constraints without HA or real models. Verifies the
**constraint logic**: comfort cost, energy cost, cautious setpoints,
trajectory generation, window schedule adjustment, and quiet hours.
"""

from __future__ import annotations

import numpy as np

from weatherstat.config import EFFECTOR_MAP, EFFECTORS
from weatherstat.control import (
    _ADVISORY_HARD_CAP,
    ABSOLUTE_MAX,
    ABSOLUTE_MIN,
    CONTROL_HORIZONS,
    HORIZON_WEIGHTS,
    _advisory_combo_count,
    _advisory_has_effect,
    _advisory_sweep_options,
    _cautious_setpoint,
    _cross_with_advisory,
    _in_hold_window,
    _in_quiet_hours,
    _regulating_sweep_options,
    apply_mrt_correction,
    compute_comfort_cost,
    compute_energy_cost,
    extract_advisory_plan,
    generate_trajectory_scenarios,
    write_command_json,
)
from weatherstat.types import (
    AdvisoryDecision,
    AdvisoryPlan,
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlDecision,
    ControlState,
    DeviceOpportunity,
    EffectorDecision,
    RoomComfort,
    Scenario,
)
from weatherstat.yaml_config import MrtCorrectionConfig

# ── Test helpers ──────────────────────────────────────────────────────────


def make_schedules(**overrides: list[ComfortScheduleEntry]) -> list[ComfortSchedule]:
    """Return default_comfort_schedules() with optional room overrides.

    Usage:
        make_schedules(upstairs=[ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 72, 72, 70, 74, 67, 77))])
    """
    from weatherstat.control import default_comfort_schedules

    defaults = {s.label: s for s in default_comfort_schedules()}
    for label, entries in overrides.items():
        defaults[label] = ComfortSchedule(sensor=f"{label}_temp", label=label, entries=tuple(entries))
    return list(defaults.values())


# ── Comfort cost tests ────────────────────────────────────────────────────


class TestComfortCost:
    """Test comfort cost computation."""

    def test_comfort_cost_continuous_plus_rail(self) -> None:
        """Room at 76F, preferred 72F, max 74F, hot_penalty 2.0.

        Cost = continuous(76-72)^2*2.0*weight + rail(76-74)^2*2.0*multiplier*weight.
        """
        schedules = [
            ComfortSchedule(
                sensor="upstairs_temp",
                label="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0, hot_penalty=2.0)),),
            ),
        ]
        predictions = {"upstairs_temp_t+12": 76.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)

        w = HORIZON_WEIGHTS[12]
        from weatherstat.control import _HARD_RAIL_MULTIPLIER
        continuous = (76.0 - 72.0) ** 2 * 2.0 * w  # deviation from preferred
        rail = (76.0 - 74.0) ** 2 * 2.0 * _HARD_RAIL_MULTIPLIER * w  # exceeds hard max
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
                sensor="upstairs_temp",
                label="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 71.0, 71.0, 69.0, 74.0, 66.0, 77.0)),),
            ),
        ]
        predictions = {"upstairs_temp_t+12": 71.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)
        assert cost == 0.0

    def test_comfort_cost_within_band_nonzero(self) -> None:
        """Room within [min, max] but not at preferred -> non-zero continuous cost."""
        schedules = [
            ComfortSchedule(
                sensor="upstairs_temp",
                label="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 71.0, 71.0, 69.0, 74.0, 66.0, 77.0)),),
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


# ── Three-tier comfort bounds tests ──────────────────────────────────────


class TestThreeTierComfort:
    """Test three-tier comfort bounds: preferred, acceptable, backup."""

    def test_config_parses_three_tiers(self) -> None:
        """Example YAML with min/max produces three-tier ComfortEntry."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        # Bedroom schedule: min=70, max=73 → acceptable_lo=70, acceptable_hi=73, backup = ±3
        bedroom = next(c for c in cfg.constraints if c.label == "bedroom")
        entry = bedroom.entries[0]
        assert entry.acceptable_lo == 70.0
        assert entry.acceptable_hi == 73.0
        # Backup defaults from acceptable ± 3
        assert entry.backup_lo == 67.0
        assert entry.backup_hi == 76.0

    def test_backup_defaults_from_margin(self) -> None:
        """When backup is not specified, it defaults to acceptable ± backup_margin."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        assert cfg.backup_margin == 3.0
        for constraint in cfg.constraints:
            for entry in constraint.entries:
                assert entry.backup_lo == entry.acceptable_lo - cfg.backup_margin
                assert entry.backup_hi == entry.acceptable_hi + cfg.backup_margin

    def test_default_comfort_schedules_three_tiers(self) -> None:
        """default_comfort_schedules produces RoomComfort with all three tiers."""
        from weatherstat.control import default_comfort_schedules

        schedules = default_comfort_schedules()
        for sched in schedules:
            for entry in sched.entries:
                c = entry.comfort
                assert c.acceptable_lo <= c.preferred_lo
                assert c.preferred_hi <= c.acceptable_hi
                assert c.backup_lo <= c.acceptable_lo
                assert c.acceptable_hi <= c.backup_hi

    def test_acceptable_cost_matches_old_hard_rail(self) -> None:
        """Temperature outside acceptable produces same cost as old min/max hard rail."""
        schedules = [
            ComfortSchedule(
                sensor="test_temp",
                label="test",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("test", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
            ),
        ]
        # 68°F is below acceptable_lo (70) by 2°F
        predictions = {"test_temp_t+12": 68.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)
        w = HORIZON_WEIGHTS[12]
        continuous = (72.0 - 68.0) ** 2 * 2.0 * w  # below preferred
        rail = (70.0 - 68.0) ** 2 * 2.0 * 3.0 * w  # below acceptable (default multiplier)
        expected = continuous + rail
        assert abs(cost - expected) < 0.001

    def test_between_acceptable_and_backup_no_extra_penalty(self) -> None:
        """Temperature between acceptable and backup has rail penalty but not backup penalty.

        Currently backup is data-only — not used in scoring yet (future advisory sweep).
        """
        schedules = [
            ComfortSchedule(
                sensor="test_temp",
                label="test",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("test", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
            ),
        ]
        # 68°F: below acceptable (70) but above backup (67)
        predictions = {"test_temp_t+12": 68.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)
        # Cost should include continuous + acceptable rail only
        w = HORIZON_WEIGHTS[12]
        continuous = (72.0 - 68.0) ** 2 * 2.0 * w
        rail = (70.0 - 68.0) ** 2 * 2.0 * 3.0 * w
        expected = continuous + rail
        assert abs(cost - expected) < 0.001

    def test_profile_applies_to_all_tiers(self) -> None:
        """Comfort profile offsets shift acceptable and backup bounds."""
        from weatherstat.control import apply_comfort_profile
        from weatherstat.yaml_config import ComfortProfile

        schedules = [ComfortSchedule(
            sensor="test_temp",
            label="test",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("test", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        profile = ComfortProfile(name="Away", min_offset=-2.0, max_offset=1.0)
        adjusted = apply_comfort_profile(schedules, profile)
        c = adjusted[0].entries[0].comfort
        assert c.acceptable_lo == 68.0  # 70 + (-2)
        assert c.acceptable_hi == 75.0  # 74 + 1
        assert c.backup_lo == 65.0  # 67 + (-2)
        assert c.backup_hi == 78.0  # 77 + 1

    def test_mrt_correction_shifts_all_tiers(self) -> None:
        """MRT correction shifts all three tiers equally."""
        schedules = [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        cfg = MrtCorrectionConfig(alpha=0.1, reference_temp=50.0, max_offset=3.0)
        adjusted, offset, _ = apply_mrt_correction(schedules, 35.0, cfg)
        # Offset = 0.1 × (50 - 35) = 1.5
        c = adjusted[0].entries[0].comfort
        assert abs(c.preferred_lo - 73.5) < 0.01
        assert abs(c.acceptable_lo - 71.5) < 0.01
        assert abs(c.backup_lo - 68.5) < 0.01
        assert abs(c.backup_hi - 78.5) < 0.01


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
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        adjusted, offset, _ = apply_mrt_correction(schedules, 35.0, self._cfg())
        assert abs(offset - 1.5) < 0.01
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred_lo - 73.5) < 0.01
        assert abs(entry.comfort.acceptable_lo - 71.5) < 0.01
        assert abs(entry.comfort.acceptable_hi - 75.5) < 0.01

    def test_warm_day_lowers_targets(self) -> None:
        """80°F outside, ref=50, alpha=0.1 → clamped to -3.0°F."""
        schedules = [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        adjusted, offset, _ = apply_mrt_correction(schedules, 80.0, self._cfg())
        assert abs(offset - (-3.0)) < 0.01  # clamped
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred_lo - 69.0) < 0.01

    def test_at_reference_no_change(self) -> None:
        """Outdoor = reference → no correction."""
        schedules = make_schedules()
        adjusted, offset, _ = apply_mrt_correction(schedules, 50.0, self._cfg())
        assert offset == 0.0
        assert adjusted == schedules

    def test_clamped_at_max(self) -> None:
        """0°F outside → raw offset 5.0, clamped to 3.0."""
        schedules = [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        adjusted, offset, _ = apply_mrt_correction(schedules, 0.0, self._cfg())
        assert abs(offset - 3.0) < 0.01
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred_lo - 75.0) < 0.01

    def test_none_config_no_change(self) -> None:
        """mrt_config=None → schedules returned unchanged."""
        schedules = make_schedules()
        adjusted, offset, _ = apply_mrt_correction(schedules, 35.0, None)
        assert offset == 0.0
        assert adjusted is schedules

    def test_penalties_preserved(self) -> None:
        """MRT correction shifts temps but preserves penalty weights."""
        schedules = [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0, 3.0, 0.5)),),
        )]
        adjusted, _, _ = apply_mrt_correction(schedules, 35.0, self._cfg())
        entry = adjusted[0].entries[0]
        assert entry.comfort.cold_penalty == 3.0
        assert entry.comfort.hot_penalty == 0.5

    def test_mrt_weight_scales_offset(self) -> None:
        """Per-sensor weight=0.5 produces half the offset."""
        schedules = [ComfortSchedule(
            sensor="piano_temp",
            label="piano",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        # Base offset: 0.1 * (50 - 35) = 1.5°F; weighted: 1.5 * 0.5 = 0.75°F
        adjusted, base_offset, per_sensor = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"piano_temp": 0.5},
        )
        assert abs(base_offset - 1.5) < 0.01
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred_lo - 72.75) < 0.01
        assert abs(entry.comfort.acceptable_lo - 70.75) < 0.01
        assert abs(per_sensor["piano_temp"] - 0.75) < 0.01

    def test_mrt_weight_zero_no_adjustment(self) -> None:
        """Per-sensor weight=0.0 suppresses MRT correction for that sensor."""
        schedules = [ComfortSchedule(
            sensor="piano_temp",
            label="piano",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        adjusted, base_offset, per_sensor = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"piano_temp": 0.0},
        )
        assert abs(base_offset - 1.5) < 0.01
        assert abs(per_sensor["piano_temp"]) < 0.01  # weight=0 → zero offset
        entry = adjusted[0].entries[0]
        assert abs(entry.comfort.preferred_lo - 72.0) < 0.01  # unchanged

    def test_mrt_different_weights_per_sensor(self) -> None:
        """Two sensors with different weights get different offsets."""
        schedules = [
            ComfortSchedule(
                sensor="piano_temp",
                label="piano",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
            ),
            ComfortSchedule(
                sensor="bathroom_temp",
                label="bathroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bathroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
            ),
        ]
        # Base offset: 1.5°F; piano×0.5=0.75, bathroom×1.5=2.25
        adjusted, _, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"piano_temp": 0.5, "bathroom_temp": 1.5},
        )
        piano_pref = adjusted[0].entries[0].comfort.preferred_lo
        bath_pref = adjusted[1].entries[0].comfort.preferred_lo
        assert abs(piano_pref - 72.75) < 0.01
        assert abs(bath_pref - 74.25) < 0.01

    def test_mrt_weight_default_backward_compat(self) -> None:
        """No weights dict (None) produces same behavior as uniform weight=1.0."""
        schedules = [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        adj_none, off_none, _ = apply_mrt_correction(schedules, 35.0, self._cfg())
        adj_ones, off_ones, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(), mrt_weights={"bedroom": 1.0},
        )
        assert off_none == off_ones
        assert adj_none[0].entries[0].comfort.preferred_lo == adj_ones[0].entries[0].comfort.preferred_lo


# ── Derived MRT weight tests ─────────────────────────────────────────────


class TestSunAwareMrt:
    """Test dynamic sun-aware MRT correction."""

    def _cfg(self) -> MrtCorrectionConfig:
        return MrtCorrectionConfig(alpha=0.1, reference_temp=50.0, max_offset=3.0, solar_response=2.0)

    def test_sunny_day_reduces_cold_correction(self) -> None:
        """High solar gain sensor gets less MRT correction on sunny cold day."""
        schedules = [ComfortSchedule(
            sensor="piano_temp",
            label="piano",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        # Cloudy: no solar → full cold correction
        adj_cloudy, _, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(),
            solar_elevation_gains={"piano_temp": 3.0},
            current_solar_elev=0.7, current_solar_fraction=0.0,
        )
        # Sunny: solar warms walls → reduced correction
        adj_sunny, _, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(),
            solar_elevation_gains={"piano_temp": 3.0},
            current_solar_elev=0.7, current_solar_fraction=1.0,
        )
        cloudy_pref = adj_cloudy[0].entries[0].comfort.preferred_lo
        sunny_pref = adj_sunny[0].entries[0].comfort.preferred_lo
        # Sunny day needs less heating → lower preferred target
        assert sunny_pref < cloudy_pref

    def test_no_solar_gains_same_as_base(self) -> None:
        """Without solar elevation gains, sun state doesn't matter."""
        schedules = [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        adj_no_gains, offset, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(),
            solar_elevation_gains=None,
            current_solar_elev=0.7, current_solar_fraction=1.0,
        )
        entry = adj_no_gains[0].entries[0]
        # Should be base offset: 0.1 * (50 - 35) = 1.5°F
        assert abs(entry.comfort.preferred_lo - 73.5) < 0.01

    def test_night_no_solar_effect(self) -> None:
        """At night (elev=0), solar gains don't change MRT correction."""
        schedules = [ComfortSchedule(
            sensor="piano_temp",
            label="piano",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
        )]
        adj_night, _, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(),
            solar_elevation_gains={"piano_temp": 5.0},
            current_solar_elev=0.0, current_solar_fraction=1.0,
        )
        entry = adj_night[0].entries[0]
        # sin⁺=0 → no solar wall warming → full base offset
        assert abs(entry.comfort.preferred_lo - 73.5) < 0.01

    def test_per_sensor_solar_differentiation(self) -> None:
        """Sensors with different solar gains get different MRT offsets."""
        schedules = [
            ComfortSchedule(
                sensor="piano_temp",
                label="piano",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("piano", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
            ),
            ComfortSchedule(
                sensor="bedroom_temp",
                label="bedroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 72.0, 72.0, 70.0, 74.0, 67.0, 77.0)),),
            ),
        ]
        adjusted, _, _ = apply_mrt_correction(
            schedules, 35.0, self._cfg(),
            solar_elevation_gains={"piano_temp": 5.0, "bedroom_temp": 0.5},
            current_solar_elev=0.7, current_solar_fraction=1.0,
        )
        piano_pref = adjusted[0].entries[0].comfort.preferred_lo
        bedroom_pref = adjusted[1].entries[0].comfort.preferred_lo
        # Piano has more solar → more wall warming → less cold correction → lower target
        assert piano_pref < bedroom_pref


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
        from weatherstat.config import EFFECTOR_MAP
        expected = EFFECTOR_MAP["thermostat_upstairs"].energy_cost
        assert compute_energy_cost(traj) == expected


# ── Target grid sweep tests ──────────────────────────────────────────


class TestRegulatingSweepOptions:
    """Test target-based regulating effector sweep option generation."""

    @staticmethod
    def _bedroom_eff():
        return EFFECTOR_MAP["mini_split_bedroom"]

    def _bedroom_schedule(self, min_t: float = 68.0, max_t: float = 72.0) -> list[ComfortSchedule]:
        preferred = (min_t + max_t) / 2.0
        return [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", preferred, preferred, min_t, max_t, min_t - 3.0, max_t + 3.0)),),
        )]

    @staticmethod
    def _bedroom_gains() -> dict[tuple[str, str], tuple[float, float]]:
        return {("mini_split_bedroom", "bedroom_temp"): (0.732, 10.0)}

    def test_sweep_options_include_off(self) -> None:
        """Off should always be an option."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(), current_temps={"bedroom_temp": 68.0},
        )
        assert any(o.mode == "off" for o in options)

    def test_sweep_options_preferred_target(self) -> None:
        """Should generate heat and/or cool options targeting affected sensor prefs."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(), current_temps={"bedroom_temp": 68.0},
        )
        active = [o for o in options if o.mode != "off"]
        assert len(active) >= 1
        # Room is cold — heat@pref_lo should be offered
        assert any(o.mode == "heat" and o.target == 70.0 for o in active)

    def test_sweep_mode_heat_when_cold(self) -> None:
        """Heat should be offered when room temp is below preferred."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(), current_temps={"bedroom_temp": 68.0},
        )
        assert any(o.mode == "heat" for o in options)

    def test_sweep_mode_cool_when_hot(self) -> None:
        """Cool should be offered when room temp is above preferred."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(), current_temps={"bedroom_temp": 72.0},
        )
        assert any(o.mode == "cool" for o in options)

    def test_sweep_idle_suppression_prunes_heat_when_hot(self) -> None:
        """Heat option should be pruned when room is well above target."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(), current_temps={"bedroom_temp": 72.0},
        )
        # Room is 2°F above pref_lo=70 and p_band=1.0, so heat@70 is idle-suppressed
        assert not any(o.mode == "heat" for o in options)

    def test_sweep_no_schedule_returns_off_only(self) -> None:
        """No matching schedule -> only off option."""
        options = _regulating_sweep_options(self._bedroom_eff(), [], 12)
        assert len(options) == 1
        assert options[0].mode == "off"

    def test_sweep_without_gains_returns_off_only(self) -> None:
        """Without gains, no sensors are affected -> only off option."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            current_temps={"bedroom_temp": 68.0},
        )
        assert len(options) == 1
        assert options[0].mode == "off"


# ── Mode hold window tests ──────────────────────────────────────────


class TestModeHoldWindow:
    """Test mode hold window constraints (quiet hours only)."""

    def _bedroom_schedule(self) -> list[ComfortSchedule]:
        return [ComfortSchedule(
            sensor="bedroom_temp",
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 70.0, 68.0, 72.0, 65.0, 75.0)),),
        )]

    def test_in_hold_window_wraps_midnight(self) -> None:
        """22:00-07:00 window wraps midnight correctly."""
        assert _in_hold_window(23, (22, 7)) is True
        assert _in_hold_window(0, (22, 7)) is True
        assert _in_hold_window(6, (22, 7)) is True
        assert _in_hold_window(7, (22, 7)) is False
        assert _in_hold_window(12, (22, 7)) is False

    @staticmethod
    def _bedroom_gains() -> dict[tuple[str, str], tuple[float, float]]:
        return {("mini_split_bedroom", "bedroom_temp"): (0.732, 10.0)}

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
            eff, self._bedroom_schedule(), 23,
            gains=self._bedroom_gains(), prev_state=prev_state,
            current_temps={"bedroom_temp": 68.0},
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
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(), prev_state=prev_state,
            current_temps={"bedroom_temp": 68.0},
        )
        assert any(o.mode == "off" for o in options)

    def test_cool_offered_when_room_above_target(self) -> None:
        """Cool option offered when room is above preferred."""
        eff = EFFECTOR_MAP["mini_split_bedroom"]
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(),
            current_temps={"bedroom_temp": 73.0},  # 3 deg F above preferred 70
        )
        # Room is above preferred — cool mode should be offered
        assert any(o.mode == "cool" for o in options)

    def test_split_offered_when_room_near_target(self) -> None:
        """Cool option available when room is slightly above preferred."""
        eff = EFFECTOR_MAP["mini_split_bedroom"]
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            gains=self._bedroom_gains(),
            current_temps={"bedroom_temp": 70.5},  # within proportional_band of preferred 70
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
        temps = {"mini_split_bedroom_temp": 68.0}
        assert compute_energy_cost(low_target, temps) < compute_energy_cost(high_target, temps)

    def test_off_mini_split_no_energy_cost(self) -> None:
        """Off mini split should have zero energy cost."""
        off = Scenario(effectors={})
        assert compute_energy_cost(off) == 0.0


# ── Trajectory scenario generation with schedules ──────────────────


class TestTrajectoryWithSchedules:
    """Test trajectory generation with comfort schedule integration."""

    def test_scenario_count_with_schedules(self) -> None:
        """With schedules and gains, regulating effectors get delay×duration options."""
        schedules = [
            ComfortSchedule(
                sensor="bedroom_temp",
                label="bedroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 70.0, 68.0, 72.0, 65.0, 75.0)),),
            ),
            ComfortSchedule(
                sensor="living_room_temp",
                label="living_room",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("living_room", 71.0, 71.0, 69.0, 74.0, 66.0, 77.0)),),
            ),
        ]
        gains = {
            ("mini_split_bedroom", "bedroom_temp"): (0.732, 10.0),
            ("mini_split_living_room", "living_room_temp"): (0.65, 10.0),
        }
        # Room temps below preferred -> heat mode for both splits
        current_temps = {"bedroom_temp": 68.0, "living_room_temp": 69.0}
        scenarios = generate_trajectory_scenarios(
            schedules, base_hour=12, current_temps=current_temps, gains=gains,
        )
        # Each split: off + heat@preferred × delay×duration combos
        # Should have more than 4 combos now (off + multiple delay/duration variants)
        split_combos = set()
        for s in scenarios:
            combo = tuple(
                (name, ed.mode, ed.target)
                for name, ed in sorted(s.effectors.items())
                if EFFECTOR_MAP.get(name) and EFFECTOR_MAP[name].control_type == "regulating"
            )
            split_combos.add(combo)
        # At minimum: off×off, off×heat, heat×off, heat×heat (mode/target combos)
        assert len(split_combos) >= 4

    def test_all_off_still_present(self) -> None:
        """All-off baseline should still be in scenario set."""
        schedules = [
            ComfortSchedule(
                sensor="bedroom_temp",
                label="bedroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 70.0, 68.0, 72.0, 65.0, 75.0)),),
            ),
        ]
        current_temps = {"bedroom_temp": 68.0}
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


# ── Advisory sweep tests ─────────────────────────────────────────────────


class TestAdvisorySweepOptions:
    """Test _advisory_sweep_options() and advisory scenario generation."""

    def _make_sim_params_with_advisory(self, device: str):
        """Create SimParams with a window beta for the given device."""
        from weatherstat.simulator import SimParams, TauModel, load_sim_params

        base = load_sim_params()
        augmented_taus = {}
        for sensor, tau_model in base.taus.items():
            augmented_taus[sensor] = TauModel(
                tau_base=tau_model.tau_base,
                environment_tau_betas={device: 0.02, **tau_model.environment_tau_betas},
            )
        return SimParams(
            taus=augmented_taus,
            gains=base.gains,
            solar=base.solar,
            sensors=base.sensors,
            effectors=base.effectors,
            solar_elevation_gains=base.solar_elevation_gains,
            environment_solar_betas=base.environment_solar_betas,
        )

    def test_relevance_filter_skips_unknown_device(self) -> None:
        """Device with no effects in thermal_params is not relevant."""
        from weatherstat.simulator import load_sim_params

        sp = load_sim_params()
        assert not _advisory_has_effect("nonexistent_device", sp)

    def test_relevance_filter_finds_tau_beta(self) -> None:
        """Device with tau beta is relevant."""
        sp = self._make_sim_params_with_advisory("test_window")
        assert _advisory_has_effect("test_window", sp)

    def test_active_device_gets_return_options(self) -> None:
        """Non-default (active) device gets hold + return-at-various-steps."""
        sp = self._make_sim_params_with_advisory("piano_window")
        options = _advisory_sweep_options({"piano_window": True}, sp)
        assert "piano_window" in options
        opts = options["piano_window"]
        # Should have hold + 4 return timings
        assert len(opts) == 5
        assert opts[0].action == "hold"
        assert all(o.action == "close" for o in opts[1:])
        # Return timings at 0, 12, 24, 36 steps
        steps = [o.transition_step for o in opts[1:]]
        assert steps == [0, 12, 24, 36]

    def test_default_device_gets_proactive_options(self) -> None:
        """Default (inactive) device gets hold + proactive activate-with-return."""
        sp = self._make_sim_params_with_advisory("piano_window")
        options = _advisory_sweep_options({"piano_window": False}, sp)
        assert "piano_window" in options
        opts = options["piano_window"]
        # Should have hold + 3 proactive options (skip instant return)
        assert len(opts) == 4
        assert opts[0].action == "hold"
        assert all(o.action == "open" for o in opts[1:])
        # All activate at step 0, return at 12, 24, 36
        assert all(o.transition_step == 0 for o in opts[1:])
        returns = [o.return_step for o in opts[1:]]
        assert returns == [12, 24, 36]

    def test_irrelevant_device_omitted(self) -> None:
        """Device with no effects in thermal_params is omitted from options."""
        from weatherstat.simulator import load_sim_params

        sp = load_sim_params()
        options = _advisory_sweep_options({"fake_window": True}, sp)
        assert "fake_window" not in options

    def test_scenario_count_with_advisory(self) -> None:
        """Advisory options multiply HVAC scenario count."""
        schedules = make_schedules()
        sp = self._make_sim_params_with_advisory("test_window")

        base = generate_trajectory_scenarios(schedules, base_hour=12)
        n_base = len(base)

        advisory_opts = _advisory_sweep_options({"test_window": True}, sp)
        with_adv = generate_trajectory_scenarios(
            schedules, base_hour=12, advisory_options=advisory_opts,
        )
        n_adv_options = len(advisory_opts["test_window"])
        assert len(with_adv) == n_base * n_adv_options

    def test_advisory_scenarios_have_advisories(self) -> None:
        """Generated scenarios with advisory options have advisory decisions."""
        schedules = make_schedules()
        sp = self._make_sim_params_with_advisory("test_window")
        advisory_opts = _advisory_sweep_options({"test_window": True}, sp)

        scenarios = generate_trajectory_scenarios(
            schedules, base_hour=12, advisory_options=advisory_opts,
        )
        # Every scenario should have advisory decisions
        for s in scenarios:
            assert "test_window" in s.advisories
            adv = s.advisories["test_window"]
            assert adv.action in ("hold", "close")

    def test_no_advisory_options_unchanged(self) -> None:
        """Empty advisory options produces same scenarios as before."""
        schedules = make_schedules()
        base = generate_trajectory_scenarios(schedules, base_hour=12)
        with_empty = generate_trajectory_scenarios(
            schedules, base_hour=12, advisory_options={},
        )
        assert len(base) == len(with_empty)
        # Scenarios should have no advisories
        for s in with_empty:
            assert not s.advisories


class TestTwoStageAdvisorySweep:
    """Test two-stage advisory sweep helpers and combinatorics."""

    def test_advisory_combo_count(self) -> None:
        opts = {
            "a": [AdvisoryDecision("a", action="hold"), AdvisoryDecision("a", action="close", transition_step=0)],
            "b": [AdvisoryDecision("b", action="hold"), AdvisoryDecision("b", action="close", transition_step=0)],
        }
        assert _advisory_combo_count(opts) == 4

    def test_cross_with_advisory_full(self) -> None:
        """Full product when under cap."""
        hvac = [Scenario(effectors={}), Scenario(effectors={})]
        opts = {
            "w": [AdvisoryDecision("w", action="hold"), AdvisoryDecision("w", action="close", transition_step=12)],
        }
        result = _cross_with_advisory(hvac, opts, max_total=100)
        assert len(result) == 4  # 2 HVAC × 2 advisory
        for s in result:
            assert "w" in s.advisories

    def test_cross_with_advisory_coarsens(self) -> None:
        """Coarsening reduces to hold + instant-return when over cap."""
        hvac = [Scenario(effectors={})]
        opts = {
            "w": [
                AdvisoryDecision("w", action="hold"),
                AdvisoryDecision("w", action="close", transition_step=0),
                AdvisoryDecision("w", action="close", transition_step=12),
                AdvisoryDecision("w", action="close", transition_step=24),
            ],
        }
        # Cap of 2 forces coarsening: hold + step-0 = 2 options, 1 HVAC × 2 = 2
        result = _cross_with_advisory(hvac, opts, max_total=2)
        assert len(result) == 2
        actions = {s.advisories["w"].action for s in result}
        assert "hold" in actions
        steps = {s.advisories["w"].transition_step for s in result if s.advisories["w"].action == "close"}
        assert steps == {0}  # only instant-return survived

    def test_cross_falls_back_if_still_over_cap(self) -> None:
        """Falls back to HVAC-only when even coarsened product exceeds cap."""
        hvac = [Scenario(effectors={})]
        opts = {
            "a": [AdvisoryDecision("a", action="hold"), AdvisoryDecision("a", action="close", transition_step=0)],
            "b": [AdvisoryDecision("b", action="hold"), AdvisoryDecision("b", action="close", transition_step=0)],
        }
        # 1 × 4 = 4, cap of 1 → coarsened 1 × 4 still over → fallback
        result = _cross_with_advisory(hvac, opts, max_total=1)
        assert len(result) == 1  # HVAC-only returned
        assert not result[0].advisories

    def test_two_stage_triggers_above_hard_cap(self) -> None:
        """Verify two-stage triggers: many advisory devices × HVAC > hard cap."""
        schedules = make_schedules()
        hvac = generate_trajectory_scenarios(schedules, base_hour=12, advisory_options=None)
        n_hvac = len(hvac)

        # Simulate 5 devices × 5 options = 3125 combos
        opts: dict[str, list[AdvisoryDecision]] = {}
        for i in range(5):
            name = f"device_{i}"
            opts[name] = [AdvisoryDecision(name, action="hold")]
            for step in [0, 12, 24, 36]:
                opts[name].append(AdvisoryDecision(name, action="close", transition_step=step))

        n_combos = _advisory_combo_count(opts)
        assert n_combos == 3125
        assert n_hvac * n_combos > _ADVISORY_HARD_CAP

        # The old code would return HVAC-only (hold-all). The new _cross_with_advisory
        # with a small max_total coarsens or falls back, but the two-stage logic in
        # sweep_scenarios_physics handles this by reducing HVAC candidates first.
        # Here we test that _cross_with_advisory with a reasonable budget works:
        top_k = hvac[:20]
        result = _cross_with_advisory(top_k, opts, max_total=100_000)
        assert len(result) == 20 * 3125  # full product fits in 62.5K


class TestExtractAdvisoryPlan:
    """Test extract_advisory_plan()."""

    @staticmethod
    def _empty_pred(n_scenarios: int) -> tuple[np.ndarray, dict[str, int]]:
        """Build empty pred_matrix/target_name_index for tests that don't check breaches."""
        return np.zeros((n_scenarios, 0)), {}

    @staticmethod
    def _dummy_schedules() -> list[ComfortSchedule]:
        return []

    def test_basic_per_device_opportunity(self) -> None:
        """Per-device opportunity uses cheapest change vs cheapest hold."""
        environment_states = {"win": True}
        scenarios = [
            Scenario(effectors={}),  # 0: HVAC-only, counts as all-hold baseline
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),  # 1: hold
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="close", transition_step=0)}),  # 2: change
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="close", transition_step=12)}),  # 3: change, worse
        ]
        costs = np.array([5.0, 6.0, 3.0, 4.0])
        pred, tidx = self._empty_pred(4)

        plan = extract_advisory_plan(
            scenarios, costs, environment_states,
            pred_matrix=pred, target_name_index=tidx,
            schedules=self._dummy_schedules(), base_hour=12,
        )
        # Baseline is cheapest all-hold: scenario 0 (no advisories) at 5.0.
        assert plan.baseline_idx == 0
        assert plan.baseline_cost == 5.0
        # One opportunity for "win": best_change = idx 2 (cost 3.0), best_hold = idx 1 (cost 6.0)
        assert len(plan.opportunities) == 1
        opp = plan.opportunities[0]
        assert opp.device == "win"
        assert opp.current_state is True
        assert opp.idx == 2
        assert opp.cost_delta == -3.0  # 3.0 - 6.0
        assert opp.advisory.action == "close"
        assert opp.advisory.transition_step == 0

    def test_no_change_scenarios_no_opportunity(self) -> None:
        """When every scenario holds, device has no opportunity."""
        scenarios = [
            Scenario(effectors={}),
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),
        ]
        costs = np.array([5.0, 4.0])
        pred, tidx = self._empty_pred(2)

        plan = extract_advisory_plan(
            scenarios, costs, {"win": False},
            pred_matrix=pred, target_name_index=tidx,
            schedules=self._dummy_schedules(), base_hour=12,
        )
        # Cheapest all-hold scenario wins.
        assert plan.baseline_idx == 1
        assert plan.baseline_cost == 4.0
        assert plan.opportunities == ()

    def test_proactive_open_default_device(self) -> None:
        """Proactively opening a default-state device appears as an opportunity."""
        environment_states = {"win": False}
        scenarios = [
            Scenario(effectors={}),  # 0: baseline
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),  # 1: also hold
            Scenario(
                effectors={},
                advisories={"win": AdvisoryDecision("win", action="open", transition_step=0, return_step=12)},
            ),  # 2: proactive
            Scenario(
                effectors={},
                advisories={"win": AdvisoryDecision("win", action="open", transition_step=0, return_step=24)},
            ),  # 3: proactive (best)
        ]
        costs = np.array([5.0, 4.5, 3.5, 3.0])
        pred, tidx = self._empty_pred(4)

        plan = extract_advisory_plan(
            scenarios, costs, environment_states,
            pred_matrix=pred, target_name_index=tidx,
            schedules=self._dummy_schedules(), base_hour=12,
        )
        assert plan.baseline_idx == 1  # 4.5 < 5.0 (both all-hold)
        assert plan.baseline_cost == 4.5
        assert len(plan.opportunities) == 1
        opp = plan.opportunities[0]
        assert opp.device == "win"
        assert opp.current_state is False
        assert opp.idx == 3
        assert opp.advisory.action == "open"
        assert opp.advisory.return_step == 24
        # cost_delta vs best_hold (idx 1, cost 4.5)
        assert opp.cost_delta == -1.5

    def test_backup_breach_from_baseline(self) -> None:
        """Backup breaches are computed from the baseline (all-hold) scenario."""
        environment_states = {"win": True}
        schedules = [
            ComfortSchedule(
                sensor="room_temp",
                label="room",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("room", 70, 72, 68, 74, 65, 77)),),
            ),
        ]
        scenarios = [
            Scenario(effectors={}),  # 0: HVAC-only, all-hold baseline
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="close", transition_step=0)}),  # 1: change
        ]
        costs = np.array([6.0, 5.0])
        # Build pred_matrix: 2 scenarios × 1 target column (room_temp_t+72)
        target_name_index = {"room_temp_t+72": 0}
        pred_matrix = np.array([
            [63.0],  # baseline breaches backup_lo (65)
            [66.0],  # change avoids breach
        ])

        plan = extract_advisory_plan(
            scenarios, costs, environment_states,
            pred_matrix=pred_matrix,
            target_name_index=target_name_index,
            schedules=schedules,
            base_hour=12,
        )
        assert plan.baseline_idx == 0
        assert len(plan.backup_breaches) > 0
        assert "63.0" in plan.backup_breaches[0]
        assert "65" in plan.backup_breaches[0]

    def test_multiple_devices_independent_marginals(self) -> None:
        """Each device gets its own best-change/best-hold pair independently."""
        environment_states = {"win_a": True, "win_b": False}
        scenarios = [
            # 0: both hold
            Scenario(effectors={}, advisories={
                "win_a": AdvisoryDecision("win_a", action="hold"),
                "win_b": AdvisoryDecision("win_b", action="hold"),
            }),
            # 1: close win_a, hold win_b
            Scenario(effectors={}, advisories={
                "win_a": AdvisoryDecision("win_a", action="close", transition_step=0),
                "win_b": AdvisoryDecision("win_b", action="hold"),
            }),
            # 2: hold win_a, open win_b
            Scenario(effectors={}, advisories={
                "win_a": AdvisoryDecision("win_a", action="hold"),
                "win_b": AdvisoryDecision("win_b", action="open", transition_step=0, return_step=12),
            }),
            # 3: both change — the "mixed" case the old classifier hid
            Scenario(effectors={}, advisories={
                "win_a": AdvisoryDecision("win_a", action="close", transition_step=0),
                "win_b": AdvisoryDecision("win_b", action="open", transition_step=0, return_step=12),
            }),
        ]
        costs = np.array([6.0, 4.0, 5.0, 3.0])
        pred, tidx = self._empty_pred(4)

        plan = extract_advisory_plan(
            scenarios, costs, environment_states,
            pred_matrix=pred, target_name_index=tidx,
            schedules=self._dummy_schedules(), base_hour=12,
        )
        # Baseline = scenario 0 (only all-hold scenario)
        assert plan.baseline_idx == 0
        assert plan.baseline_cost == 6.0

        by_device = {o.device: o for o in plan.opportunities}
        # win_a: best_change includes idx 3 (cost 3.0), best_hold includes idx 2 (cost 5.0)
        assert by_device["win_a"].idx == 3
        assert by_device["win_a"].cost_delta == -2.0
        assert by_device["win_a"].advisory.action == "close"
        # win_b: best_change includes idx 3 (cost 3.0), best_hold includes idx 1 (cost 4.0)
        assert by_device["win_b"].idx == 3
        assert by_device["win_b"].cost_delta == -1.0
        assert by_device["win_b"].advisory.action == "open"


class TestWriteCommandJsonAdvisory:
    """Test advisory plan data in command JSON output."""

    def _make_decision(self) -> ControlDecision:
        """Create a minimal ControlDecision for testing."""
        return ControlDecision(
            timestamp="2026-04-08T12:00:00+00:00",
            effectors=(EffectorDecision("thermostat_upstairs", mode="off"),),
            command_targets={},
            total_cost=5.0,
            comfort_cost=4.0,
            energy_cost=1.0,
        )

    def test_no_advisory_plan(self, tmp_path: object) -> None:
        """No advisory fields when advisory_plan is None."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_plan=None)
        data = _json.loads(path.read_text())
        assert "advisoryOpportunities" not in data
        assert "advisoryWarnings" not in data

    def test_beneficial_opportunity_serialized(self, tmp_path: object) -> None:
        """Beneficial opportunities appear in JSON with camelCase keys."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        plan = AdvisoryPlan(
            baseline_idx=0,
            baseline_cost=5.0,
            opportunities=(
                DeviceOpportunity(
                    device="piano_window",
                    current_state=True,
                    advisory=AdvisoryDecision("piano_window", action="close", transition_step=12),
                    idx=2,
                    cost_delta=-2.0,
                ),
            ),
        )
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_plan=plan)
        data = _json.loads(path.read_text())
        assert "advisoryOpportunities" in data
        opps = data["advisoryOpportunities"]
        assert len(opps) == 1
        assert opps[0]["device"] == "piano_window"
        assert opps[0]["currentState"] is True
        assert opps[0]["action"] == "close"
        assert opps[0]["inMinutes"] == 60  # 12 steps × 5 min
        assert opps[0]["costDelta"] == -2.0

    def test_backup_breach_warnings(self, tmp_path: object) -> None:
        """Backup breach warnings appear in JSON."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        plan = AdvisoryPlan(
            baseline_idx=0,
            baseline_cost=5.0,
            opportunities=(),
            backup_breaches=("room_temp projected 63.0 at 6h, below backup minimum 65",),
        )
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_plan=plan)
        data = _json.loads(path.read_text())
        assert "advisoryWarnings" in data
        assert len(data["advisoryWarnings"]) == 1
        assert "63.0" in data["advisoryWarnings"][0]["message"]

    def test_proactive_opportunity_with_duration(self, tmp_path: object) -> None:
        """Opportunity with return_step emits durationMinutes."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        plan = AdvisoryPlan(
            baseline_idx=0,
            baseline_cost=4.5,
            opportunities=(
                DeviceOpportunity(
                    device="living_room_window",
                    current_state=False,
                    advisory=AdvisoryDecision(
                        "living_room_window", action="open", transition_step=0, return_step=24,
                    ),
                    idx=3,
                    cost_delta=-0.31,
                ),
            ),
        )
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_plan=plan)
        data = _json.loads(path.read_text())
        assert "advisoryOpportunities" in data
        opps = data["advisoryOpportunities"]
        assert len(opps) == 1
        assert opps[0]["device"] == "living_room_window"
        assert opps[0]["currentState"] is False
        assert opps[0]["action"] == "open"
        assert opps[0]["durationMinutes"] == 120  # (24 - 0) × 5
        assert opps[0]["costDelta"] == -0.31

    def test_non_beneficial_opportunities_excluded(self, tmp_path: object) -> None:
        """Opportunities with cost_delta >= 0 are not written."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        plan = AdvisoryPlan(
            baseline_idx=0,
            baseline_cost=5.0,
            opportunities=(
                DeviceOpportunity(
                    device="win",
                    current_state=True,
                    advisory=AdvisoryDecision("win", action="close", transition_step=0),
                    idx=1,
                    cost_delta=0.3,  # worse than baseline
                ),
            ),
        )
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_plan=plan)
        data = _json.loads(path.read_text())
        assert "advisoryOpportunities" not in data

    def test_opportunities_sorted_by_cost_delta(self, tmp_path: object) -> None:
        """Multiple beneficial opportunities are sorted best-first."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        plan = AdvisoryPlan(
            baseline_idx=0,
            baseline_cost=5.0,
            opportunities=(
                DeviceOpportunity(
                    device="small_win",
                    current_state=False,
                    advisory=AdvisoryDecision("small_win", action="open", transition_step=0),
                    idx=1,
                    cost_delta=-0.5,
                ),
                DeviceOpportunity(
                    device="big_win",
                    current_state=True,
                    advisory=AdvisoryDecision("big_win", action="close", transition_step=0),
                    idx=2,
                    cost_delta=-2.0,
                ),
            ),
        )
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_plan=plan)
        data = _json.loads(path.read_text())
        opps = data["advisoryOpportunities"]
        assert [o["device"] for o in opps] == ["big_win", "small_win"]


# ── Superposition tests ──────────────────────────────────────────────────


class TestSuperposition:
    """Test marginal decomposition (superposition) sweep infrastructure."""

    @staticmethod
    def _make_state(outdoor: float = 42.0, hour: float = 14.5) -> object:
        from weatherstat.simulator import HouseState

        temps = {
            "thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0,
            "bedroom_temp": 68.5, "office_temp": 67.0, "family_room_temp": 69.5,
            "kitchen_temp": 68.0, "piano_temp": 67.5, "bathroom_temp": 68.0,
            "living_room_temp": 69.0,
        }
        return HouseState(
            current_temps=temps,
            outdoor_temp=outdoor,
            forecast_temps=[outdoor] * 12,
            environment_states={},
            hour_of_day=hour,
            solar_fractions=[0.0] * 12,
            solar_elevations=[0.0] * 72,
        )

    def test_option_groups_match_scenarios(self) -> None:
        """Option groups produce the same Cartesian product count as generate_trajectory_scenarios."""
        from weatherstat.control import (
            _build_option_groups,
            _enumerate_options,
        )
        from weatherstat.simulator import load_sim_params

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}

        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        groups = _build_option_groups(per_eff, deps, temps)

        # Count via groups
        group_product = 1
        for g in groups:
            group_product *= len(g.options)

        # Count via generate_trajectory_scenarios
        scenarios = generate_trajectory_scenarios(schedules, 12, None, temps, set(), gains=sim_params.gains)

        assert group_product == len(scenarios), (
            f"Group product {group_product} != scenario count {len(scenarios)}"
        )

    def test_option_groups_contain_all_effectors(self) -> None:
        """Every effector appears in exactly one group."""
        from weatherstat.control import (
            _build_option_groups,
            _enumerate_options,
        )
        from weatherstat.simulator import load_sim_params

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}

        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        groups = _build_option_groups(per_eff, deps, temps)

        all_names: list[str] = []
        for g in groups:
            all_names.extend(g.names)

        # Every independent effector and dependent should appear
        for name in per_eff:
            assert name in all_names, f"Independent effector {name} missing from groups"
        for dep in deps:
            assert dep.name in all_names, f"Dependent effector {dep.name} missing from groups"

        # No duplicates
        assert len(all_names) == len(set(all_names)), "Duplicate effector names across groups"

    def test_compound_group_exists(self) -> None:
        """Thermostat_downstairs + blowers form a compound group."""
        from weatherstat.control import (
            _build_option_groups,
            _enumerate_options,
        )
        from weatherstat.simulator import load_sim_params

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}

        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        groups = _build_option_groups(per_eff, deps, temps)

        compound = [g for g in groups if len(g.names) > 1]
        assert len(compound) >= 1, "Expected at least one compound group (thermostat + blowers)"

        # The compound group should contain thermostat_downstairs
        ds_group = [g for g in compound if "thermostat_downstairs" in g.names]
        assert len(ds_group) == 1
        # And blower dependents
        assert any("blower" in n for n in ds_group[0].names)

    def test_marginals_baseline_shape(self) -> None:
        """Marginal result has correct shapes."""
        from weatherstat.control import (
            CONTROL_HORIZONS,
            _build_option_groups,
            _compute_marginals,
            _enumerate_options,
        )
        from weatherstat.simulator import load_sim_params

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}
        state = self._make_state()

        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        groups = _build_option_groups(per_eff, deps, temps)
        marginals = _compute_marginals(state, groups, sim_params, CONTROL_HORIZONS)

        # t_base should be 1-D: (n_targets,)
        assert marginals.t_base.ndim == 1
        n_targets = len(marginals.t_base)
        assert n_targets > 0

        # One delta array per group
        assert len(marginals.deltas) == len(groups)
        for i, (group, delta) in enumerate(zip(groups, marginals.deltas, strict=True)):
            assert delta.shape == (len(group.options), n_targets), (
                f"Group {i} delta shape {delta.shape} != ({len(group.options)}, {n_targets})"
            )

    def test_superposition_exact_for_trajectory(self) -> None:
        """For trajectory-only effectors, marginal reconstruction matches full predict().

        Trajectory effectors have deterministic activity (not state-dependent),
        so superposition should be exact (within float tolerance).
        """
        from weatherstat.control import (
            CONTROL_HORIZONS,
            _build_option_groups,
            _compute_marginals,
            _enumerate_options,
            _materialize_scenarios,
            _score_combinations,
        )
        from weatherstat.simulator import load_sim_params, predict

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}
        state = self._make_state()

        # Only include trajectory effectors (no regulating)
        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        traj_only = {k: v for k, v in per_eff.items()
                     if EFFECTOR_MAP[k].control_type == "trajectory"}
        traj_deps = [d for d in deps if all(p in traj_only for p in d.depends_on)]

        groups = _build_option_groups(traj_only, traj_deps, temps)
        marginals = _compute_marginals(state, groups, sim_params, CONTROL_HORIZONS)

        # Score all combos approximately
        from weatherstat.control import _ComfortSpec

        comfort_spec = _ComfortSpec.build(schedules, 12, marginals.target_name_index)
        approx_costs, combo_indices, n_combos = _score_combinations(
            marginals, groups, comfort_spec,
        )

        # Materialize ALL combos and do exact simulation
        all_flat = list(range(n_combos))
        all_scenarios = _materialize_scenarios(groups, combo_indices, all_flat)
        exact_names, exact_pred = predict(state, all_scenarios, sim_params, CONTROL_HORIZONS)

        # Compare reconstructed temps vs exact temps
        # Build reconstructed pred matrix
        reconstructed = np.tile(marginals.t_base, (n_combos, 1))
        for delta, idx in zip(marginals.deltas, combo_indices, strict=True):
            reconstructed += delta[idx]

        # Trajectory effectors are deterministic, so reconstruction should be exact
        # (binary effectors with parent gating are also deterministic)
        assert np.allclose(reconstructed, exact_pred, atol=0.01), (
            f"Max deviation: {np.max(np.abs(reconstructed - exact_pred)):.4f}°F"
        )

    def test_superposition_ranking_with_regulating(self) -> None:
        """Top-K from approximate scoring contains the true winner from full simulation.

        Regulating effectors break linearity slightly, but the true winner should
        still be within the top-K approximate candidates.
        """
        from weatherstat.control import (
            _SUPERPOSITION_TOP_K,
            CONTROL_HORIZONS,
            _build_option_groups,
            _compute_marginals,
            _enumerate_options,
            _materialize_scenarios,
            _score_combinations,
        )
        from weatherstat.simulator import load_sim_params, predict

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}
        state = self._make_state()

        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        groups = _build_option_groups(per_eff, deps, temps)
        marginals = _compute_marginals(state, groups, sim_params, CONTROL_HORIZONS)

        from weatherstat.control import _ComfortSpec

        comfort_spec = _ComfortSpec.build(schedules, 12, marginals.target_name_index)
        approx_costs, combo_indices, n_combos = _score_combinations(
            marginals, groups, comfort_spec,
        )

        # Get approximate top-K
        ranked = np.argsort(approx_costs)
        K = min(_SUPERPOSITION_TOP_K, n_combos)
        top_flat = sorted(int(i) for i in ranked[:K])

        # Full simulation of ALL combos
        all_scenarios = _materialize_scenarios(groups, combo_indices, list(range(n_combos)))
        from weatherstat.control import _batch_comfort_cost, _batch_energy_cost

        exact_names, exact_pred = predict(state, all_scenarios, sim_params, CONTROL_HORIZONS)
        tidx = {t: j for j, t in enumerate(exact_names)}
        exact_comfort = _batch_comfort_cost(exact_pred, _ComfortSpec.build(schedules, 12, tidx))
        exact_energy = _batch_energy_cost(all_scenarios, temps)
        exact_costs = exact_comfort + exact_energy

        # The true winner must be in the top-K
        true_best = int(np.argmin(exact_costs))
        assert true_best in top_flat, (
            f"True winner at flat index {true_best} (cost {exact_costs[true_best]:.4f}) "
            f"not in top-{K} (worst in top-K: {approx_costs[ranked[K-1]]:.4f})"
        )

    def test_materialize_scenarios_correct(self) -> None:
        """Materialized scenarios have correct effector decisions."""
        from weatherstat.control import (
            _build_option_groups,
            _enumerate_options,
            _materialize_scenarios,
        )
        from weatherstat.simulator import load_sim_params

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}

        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        groups = _build_option_groups(per_eff, deps, temps)

        # Build combo indices
        option_counts = [len(g.options) for g in groups]
        grids = np.meshgrid(*[np.arange(n) for n in option_counts], indexing="ij")
        combo_indices = [g.ravel() for g in grids]

        # Materialize first and last
        scenarios = _materialize_scenarios(groups, combo_indices, [0, len(combo_indices[0]) - 1])
        assert len(scenarios) == 2

        # First scenario (all option index 0 = all-off)
        first = scenarios[0]
        for g in groups:
            for name in g.names:
                assert first.effectors[name].mode == "off", (
                    f"First scenario should be all-off but {name} is {first.effectors[name].mode}"
                )

    def test_energy_costs_precomputed(self) -> None:
        """Pre-computed energy costs match compute_energy_cost for materialized scenarios."""
        from weatherstat.control import (
            _build_option_groups,
            _enumerate_options,
            _materialize_scenarios,
        )
        from weatherstat.simulator import load_sim_params

        sim_params = load_sim_params()
        schedules = make_schedules()
        temps = {"thermostat_upstairs_temp": 70.0, "thermostat_downstairs_temp": 69.0, "bedroom_temp": 68.5}

        per_eff, deps = _enumerate_options(schedules, 12, None, temps, set(), sim_params.gains)
        groups = _build_option_groups(per_eff, deps, temps)

        # Build combo indices
        option_counts = [len(g.options) for g in groups]
        grids = np.meshgrid(*[np.arange(n) for n in option_counts], indexing="ij")
        combo_indices = [g.ravel() for g in grids]
        n_combos = len(combo_indices[0])

        # Pre-computed sum
        precomputed = np.zeros(n_combos)
        for group, idx in zip(groups, combo_indices, strict=True):
            precomputed += group.energy_costs[idx]

        # Compute via compute_energy_cost on materialized scenarios
        sample_indices = [0, n_combos // 4, n_combos // 2, n_combos - 1]
        sample_indices = [i for i in sample_indices if i < n_combos]
        scenarios = _materialize_scenarios(groups, combo_indices, sample_indices)

        for s_idx, flat_idx in enumerate(sample_indices):
            expected = compute_energy_cost(scenarios[s_idx], temps)
            assert abs(precomputed[flat_idx] - expected) < 0.001, (
                f"Energy cost mismatch at flat {flat_idx}: "
                f"precomputed={precomputed[flat_idx]:.4f} vs expected={expected:.4f}"
            )
