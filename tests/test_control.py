"""Tests for control optimizer constraints.

Tests the sweep constraints without HA or real models. Verifies the
**constraint logic**: comfort cost, energy cost, cautious setpoints,
trajectory generation, window schedule adjustment, and quiet hours.
"""

from __future__ import annotations

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
    classify_advisory_scenario,
    compute_comfort_cost,
    compute_energy_cost,
    extract_planning_layers,
    generate_trajectory_scenarios,
    write_command_json,
)
from weatherstat.types import (
    AdvisoryDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlDecision,
    ControlState,
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

    def test_sweep_fallback_without_gains(self) -> None:
        """Without gains, falls back to naming convention for sensor lookup."""
        eff = self._bedroom_eff()
        options = _regulating_sweep_options(
            eff, self._bedroom_schedule(), 12,
            current_temps={"bedroom_temp": 68.0},
        )
        # Should still find bedroom_temp via naming convention
        assert any(o.mode == "heat" for o in options)


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
        # Room temps below preferred -> heat mode for both splits
        current_temps = {"bedroom_temp": 68.0, "living_room_temp": 69.0}
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
                environment_interaction_betas=tau_model.environment_interaction_betas,
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


class TestClassifyAdvisoryScenario:
    """Test classify_advisory_scenario()."""

    def test_baseline_no_advisories(self) -> None:
        s = Scenario(effectors={})
        assert classify_advisory_scenario(s, {"win": True}) == "baseline"

    def test_reasonable_returns_to_default(self) -> None:
        s = Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="close", transition_step=12)},
        )
        assert classify_advisory_scenario(s, {"win": True}) == "reasonable"

    def test_worst_case_holds_active(self) -> None:
        s = Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="hold")},
        )
        assert classify_advisory_scenario(s, {"win": True}) == "worst_case"

    def test_proactive_activates_default(self) -> None:
        s = Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="open", transition_step=0, return_step=12)},
        )
        assert classify_advisory_scenario(s, {"win": False}) == "proactive"

    def test_hold_on_default_is_baseline(self) -> None:
        """Holding a default-state device is baseline (no change)."""
        s = Scenario(
            effectors={},
            advisories={"win": AdvisoryDecision("win", action="hold")},
        )
        assert classify_advisory_scenario(s, {"win": False}) == "baseline"

    def test_mixed_active_and_proactive(self) -> None:
        """Scenario with both return-to-default and proactive is mixed."""
        s = Scenario(
            effectors={},
            advisories={
                "win1": AdvisoryDecision("win1", action="close", transition_step=0),
                "win2": AdvisoryDecision("win2", action="open", transition_step=0, return_step=12),
            },
        )
        assert classify_advisory_scenario(s, {"win1": True, "win2": False}) == "mixed"


class TestExtractPlanningLayers:
    """Test extract_planning_layers()."""

    def test_basic_layer_extraction(self) -> None:
        """Finds best scenario per layer."""
        environment_states = {"win": True}
        scenarios = [
            Scenario(effectors={}),  # 0: baseline
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),  # 1: worst_case
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="close", transition_step=0)}),  # 2: reasonable
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="close", transition_step=12)}),  # 3: reasonable
        ]
        costs = [5.0, 6.0, 3.0, 4.0]

        result = extract_planning_layers(scenarios, costs, environment_states)
        assert result["baseline_idx"] == 0
        assert result["baseline_cost"] == 5.0
        assert result["reasonable_idx"] == 2  # lowest cost reasonable
        assert result["reasonable_cost"] == 3.0
        assert result["worst_case_idx"] == 1
        assert result["worst_case_cost"] == 6.0

    def test_no_non_default_reasonable_falls_back(self) -> None:
        """When no non-default devices, reasonable = baseline."""
        scenarios = [
            Scenario(effectors={}),  # baseline
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),  # baseline (hold on default)
        ]
        costs = [5.0, 4.0]

        result = extract_planning_layers(scenarios, costs, {"win": False})
        # Both are baseline; second is lower cost
        assert result["baseline_idx"] == 1
        assert result["reasonable_idx"] == 1  # falls back to baseline

    def test_proactive_scoring(self) -> None:
        """Proactive advice shows cost delta vs baseline."""
        environment_states = {"win": False}
        scenarios = [
            Scenario(effectors={}),  # 0: baseline
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),  # 1: also baseline
            Scenario(
                effectors={},
                advisories={"win": AdvisoryDecision("win", action="open", transition_step=0, return_step=12)},
            ),  # 2: proactive
            Scenario(
                effectors={},
                advisories={"win": AdvisoryDecision("win", action="open", transition_step=0, return_step=24)},
            ),  # 3: proactive (better)
        ]
        costs = [5.0, 4.5, 3.5, 3.0]

        result = extract_planning_layers(scenarios, costs, environment_states)
        assert result["baseline_idx"] == 1  # 4.5 < 5.0
        assert "win" in result["proactive"]
        assert result["proactive"]["win"]["idx"] == 3  # best proactive
        assert result["proactive"]["win"]["cost_delta"] == 3.0 - 4.5  # vs baseline

    def test_backup_breach_detection(self) -> None:
        """Backup breaches are detected in worst-case scenario."""
        environment_states = {"win": True}
        schedules = [
            ComfortSchedule(
                sensor="room_temp",
                label="room",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("room", 70, 72, 68, 74, 65, 77)),),
            ),
        ]
        scenarios = [
            Scenario(effectors={}),  # 0: baseline
            Scenario(effectors={}, advisories={"win": AdvisoryDecision("win", action="hold")}),  # 1: worst_case
        ]
        costs = [5.0, 6.0]
        # Worst-case predictions breach backup_lo (65)
        predictions = [
            {"room_temp_t+72": 66.0},  # baseline: fine
            {"room_temp_t+72": 63.0},  # worst_case: below backup 65
        ]

        result = extract_planning_layers(
            scenarios, costs, environment_states,
            predictions_per_scenario=predictions,
            schedules=schedules,
            base_hour=12,
        )
        assert len(result["backup_breaches"]) > 0
        assert "63.0" in result["backup_breaches"][0]
        assert "65" in result["backup_breaches"][0]

    def test_enriched_advisories_in_result(self) -> None:
        """Result includes actual AdvisoryDecision objects for reasonable and proactive layers."""
        environment_states = {"win_a": True, "win_b": False}
        scenarios = [
            Scenario(effectors={}),  # 0: baseline
            Scenario(effectors={}, advisories={
                "win_a": AdvisoryDecision("win_a", action="close", transition_step=0),
            }),  # 1: reasonable
            Scenario(effectors={}, advisories={
                "win_a": AdvisoryDecision("win_a", action="hold"),
            }),  # 2: worst_case
            Scenario(effectors={}, advisories={
                "win_b": AdvisoryDecision("win_b", action="open", transition_step=0, return_step=12),
            }),  # 3: proactive
        ]
        costs = [5.0, 3.0, 6.0, 4.0]

        result = extract_planning_layers(scenarios, costs, environment_states)

        # Reasonable advisories
        assert "reasonable_advisories" in result
        assert "win_a" in result["reasonable_advisories"]
        adv = result["reasonable_advisories"]["win_a"]
        assert adv.action == "close"
        assert adv.transition_step == 0

        # Proactive advisories
        assert "proactive_advisories" in result
        assert "win_b" in result["proactive_advisories"]
        padv = result["proactive_advisories"]["win_b"]
        assert padv.action == "open"
        assert padv.return_step == 12


class TestWriteCommandJsonAdvisory:
    """Test advisory layer data in command JSON output."""

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

    def test_no_advisory_layers(self, tmp_path: object) -> None:
        """No advisory fields when advisory_layers is None."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_layers=None)
        data = _json.loads(path.read_text())
        assert "advisoryRecommendations" not in data
        assert "advisoryWarnings" not in data
        assert "proactiveAdvice" not in data

    def test_reasonable_recommendations(self, tmp_path: object) -> None:
        """Advisory recommendations from reasonable layer appear in JSON."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        layers = {
            "reasonable_advisories": {
                "piano_window": AdvisoryDecision("piano_window", action="close", transition_step=12),
            },
            "reasonable_cost": 3.0,
            "baseline_cost": 5.0,
            "backup_breaches": [],
            "proactive": {},
            "proactive_advisories": {},
        }
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_layers=layers)
        data = _json.loads(path.read_text())
        assert "advisoryRecommendations" in data
        recs = data["advisoryRecommendations"]
        assert len(recs) == 1
        assert recs[0]["device"] == "piano_window"
        assert recs[0]["action"] == "close"
        assert recs[0]["inMinutes"] == 60  # 12 steps × 5 min
        assert recs[0]["layer"] == "reasonable"
        assert recs[0]["costDelta"] == -2.0  # 3.0 - 5.0

    def test_backup_breach_warnings(self, tmp_path: object) -> None:
        """Backup breach warnings appear in JSON."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        layers = {
            "reasonable_advisories": {},
            "backup_breaches": [
                "room_temp projected 63.0 at 6h, below backup minimum 65",
            ],
            "proactive": {},
            "proactive_advisories": {},
        }
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_layers=layers)
        data = _json.loads(path.read_text())
        assert "advisoryWarnings" in data
        assert len(data["advisoryWarnings"]) == 1
        assert "63.0" in data["advisoryWarnings"][0]["message"]
        assert data["advisoryWarnings"][0]["layer"] == "worst_case"

    def test_proactive_advice(self, tmp_path: object) -> None:
        """Proactive advice with duration appears in JSON."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        layers = {
            "reasonable_advisories": {},
            "backup_breaches": [],
            "proactive": {
                "living_room_window": {"idx": 2, "cost_delta": -0.31},
            },
            "proactive_advisories": {
                "living_room_window": AdvisoryDecision(
                    "living_room_window", action="open", transition_step=0, return_step=24,
                ),
            },
        }
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_layers=layers)
        data = _json.loads(path.read_text())
        assert "proactiveAdvice" in data
        advice = data["proactiveAdvice"]
        assert len(advice) == 1
        assert advice[0]["device"] == "living_room_window"
        assert advice[0]["action"] == "open"
        assert advice[0]["durationMinutes"] == 120  # (24 - 0) × 5
        assert advice[0]["costDelta"] == -0.31
        assert advice[0]["layer"] == "proactive"

    def test_hold_only_advisories_excluded(self, tmp_path: object) -> None:
        """Hold-only reasonable advisories don't produce recommendations."""
        import json as _json
        from unittest.mock import patch

        decision = self._make_decision()
        layers = {
            "reasonable_advisories": {
                "win": AdvisoryDecision("win", action="hold"),
            },
            "reasonable_cost": 5.0,
            "baseline_cost": 5.0,
            "backup_breaches": [],
            "proactive": {},
            "proactive_advisories": {},
        }
        with patch("weatherstat.control.PREDICTIONS_DIR", tmp_path):
            path = write_command_json(decision, advisory_layers=layers)
        data = _json.loads(path.read_text())
        assert "advisoryRecommendations" not in data
