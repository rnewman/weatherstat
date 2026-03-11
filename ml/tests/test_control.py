"""Tests for control optimizer constraints.

Tests the sweep constraints without HA or real models. Verifies the
**constraint logic**: comfort cost, energy cost, cautious setpoints,
trajectory generation, window schedule adjustment, and quiet hours.
"""

from __future__ import annotations

from weatherstat.config import BLOWERS, MINI_SPLITS
from weatherstat.control import (
    ABSOLUTE_MAX,
    ABSOLUTE_MIN,
    CONTROL_HORIZONS,
    HORIZON_WEIGHTS,
    _cautious_setpoint,
    _in_hold_window,
    _in_quiet_hours,
    _mini_split_sweep_options,
    adjust_schedules_for_windows,
    compute_comfort_cost,
    compute_energy_cost,
    generate_trajectory_scenarios,
)
from weatherstat.types import (
    BlowerDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlState,
    MiniSplitDecision,
    RoomComfort,
    ThermostatTrajectory,
    TrajectoryScenario,
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
        scenario_both = TrajectoryScenario(
            ThermostatTrajectory(heating=True),
            ThermostatTrajectory(heating=True),
            tuple(BlowerDecision(b.name, "off") for b in BLOWERS),
            tuple(MiniSplitDecision(s.name, "off", 72.0) for s in MINI_SPLITS),
        )
        scenario_one = TrajectoryScenario(
            ThermostatTrajectory(heating=True),
            ThermostatTrajectory(heating=False),
            tuple(BlowerDecision(b.name, "off") for b in BLOWERS),
            tuple(MiniSplitDecision(s.name, "off", 72.0) for s in MINI_SPLITS),
        )

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

    def test_trajectory_scenarios_include_all_off(self) -> None:
        """All-off should be in the trajectory scenario set."""
        scenarios = generate_trajectory_scenarios()
        all_off = [
            s for s in scenarios
            if not s.upstairs.heating and not s.downstairs.heating
            and all(b.mode == "off" for b in s.blowers)
            and all(sp.mode == "off" for sp in s.mini_splits)
        ]
        assert len(all_off) == 1

    def test_trajectory_scenarios_include_delayed_start(self) -> None:
        """Trajectories with delay > 0 should exist."""
        scenarios = generate_trajectory_scenarios()
        delayed = [s for s in scenarios if s.upstairs.heating and s.upstairs.delay_steps > 0]
        assert len(delayed) > 0

    def test_trajectory_scenarios_include_short_duration(self) -> None:
        """Trajectories with finite duration should exist."""
        scenarios = generate_trajectory_scenarios()
        short = [
            s for s in scenarios
            if s.upstairs.heating and s.upstairs.duration_steps is not None and s.upstairs.duration_steps < 72
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
            if s.upstairs.heating:
                assert s.upstairs.delay_steps < max_horizon
            if s.downstairs.heating:
                assert s.downstairs.delay_steps < max_horizon

    def test_trajectory_duration_capped_at_horizon(self) -> None:
        """delay + duration should not exceed max horizon."""
        max_horizon = max(CONTROL_HORIZONS)
        scenarios = generate_trajectory_scenarios()
        for s in scenarios:
            if s.upstairs.heating and s.upstairs.duration_steps is not None:
                assert s.upstairs.delay_steps + s.upstairs.duration_steps <= max_horizon
            if s.downstairs.heating and s.downstairs.duration_steps is not None:
                assert s.downstairs.delay_steps + s.downstairs.duration_steps <= max_horizon


class TestTrajectoryEnergyCost:
    """Test that trajectory energy cost scales with duration."""

    def test_shorter_duration_lower_cost(self) -> None:
        """2h trajectory should cost less than 6h trajectory."""
        short = TrajectoryScenario(
            ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=24),
            ThermostatTrajectory(heating=False),
            (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
            (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
        )
        long = TrajectoryScenario(
            ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=72),
            ThermostatTrajectory(heating=False),
            (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
            (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
        )
        assert compute_energy_cost(short) < compute_energy_cost(long)

    def test_off_trajectory_no_gas_cost(self) -> None:
        """All-off trajectory should have no gas zone cost."""
        off = TrajectoryScenario(
            ThermostatTrajectory(heating=False),
            ThermostatTrajectory(heating=False),
            (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
            (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
        )
        assert compute_energy_cost(off) == 0.0

    def test_full_horizon_trajectory_energy_cost(self) -> None:
        """Full-horizon trajectory should cost one gas zone unit."""
        traj = TrajectoryScenario(
            ThermostatTrajectory(heating=True, delay_steps=0, duration_steps=None),
            ThermostatTrajectory(heating=False),
            (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
            (MiniSplitDecision("bedroom", "off", 72), MiniSplitDecision("living_room", "off", 72)),
        )
        from weatherstat.config import ENERGY_COST_GAS_ZONE
        assert compute_energy_cost(traj) == ENERGY_COST_GAS_ZONE


# ── Target grid sweep tests ──────────────────────────────────────────


class TestMiniSplitSweepOptions:
    """Test target-based mini split sweep option generation."""

    def _bedroom_schedule(self, min_t: float = 68.0, max_t: float = 72.0) -> list[ComfortSchedule]:
        preferred = (min_t + max_t) / 2.0
        return [ComfortSchedule(
            label="bedroom",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", preferred, min_t, max_t)),),
        )]

    def test_sweep_options_include_off(self) -> None:
        """Off should always be an option."""
        options = _mini_split_sweep_options("bedroom", self._bedroom_schedule(), 12, 42.0)
        assert any(o.mode == "off" for o in options)

    def test_sweep_options_preferred_target(self) -> None:
        """Should generate off + preferred target."""
        options = _mini_split_sweep_options("bedroom", self._bedroom_schedule(), 12, 42.0)
        assert len(options) == 2  # off + preferred
        active = [o for o in options if o.mode != "off"]
        assert len(active) == 1
        assert active[0].target == 70.0  # preferred = midpoint of 68-72

    def test_sweep_mode_heat_when_cold(self) -> None:
        """Mode should be 'heat' when outdoor temp is below preferred."""
        options = _mini_split_sweep_options("bedroom", self._bedroom_schedule(), 12, 42.0)
        active = [o for o in options if o.mode != "off"]
        assert all(o.mode == "heat" for o in active)

    def test_sweep_mode_cool_when_hot(self) -> None:
        """Mode should be 'cool' when outdoor temp is above preferred."""
        options = _mini_split_sweep_options("bedroom", self._bedroom_schedule(), 12, 85.0)
        active = [o for o in options if o.mode != "off"]
        assert all(o.mode == "cool" for o in active)

    def test_sweep_no_schedule_returns_off_only(self) -> None:
        """No matching schedule -> only off option."""
        options = _mini_split_sweep_options("bedroom", [], 12, 42.0)
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
            upstairs_setpoint=70.0,
            downstairs_setpoint=70.0,
            mini_split_modes={"bedroom": "heat"},
            mini_split_targets={"bedroom": 70.0},
            mini_split_mode_times={"bedroom": datetime.now(UTC).isoformat()},
        )
        # Hour 23 is inside bedroom's hold window [22, 7]
        options = _mini_split_sweep_options(
            "bedroom", self._bedroom_schedule(), 23, 42.0, prev_state,
        )
        # All options should be "heat" (current mode)
        assert all(o.mode == "heat" for o in options)
        assert len(options) >= 1

    def test_mode_not_locked_outside_hold_window(self) -> None:
        """Outside hold window, mode can change freely."""
        from datetime import UTC, datetime

        prev_state = ControlState(
            last_decision_time=datetime.now(UTC).isoformat(),
            upstairs_setpoint=70.0,
            downstairs_setpoint=70.0,
            mini_split_modes={"bedroom": "heat"},
            mini_split_targets={"bedroom": 70.0},
            mini_split_mode_times={"bedroom": datetime.now(UTC).isoformat()},
        )
        # Hour 12 is outside hold window — mode should be unlocked
        options = _mini_split_sweep_options(
            "bedroom", self._bedroom_schedule(), 12, 42.0, prev_state,
        )
        assert any(o.mode == "off" for o in options)

    def test_idle_split_skipped_when_room_above_target(self) -> None:
        """Heat option skipped when room is well above preferred (split would be idle)."""
        options = _mini_split_sweep_options(
            "bedroom", self._bedroom_schedule(), 12, 42.0,
            current_temps={"bedroom": 73.0},  # 3°F above preferred 70
        )
        # Only off should remain — room is above preferred + proportional_band
        assert all(o.mode == "off" for o in options)

    def test_split_offered_when_room_near_target(self) -> None:
        """Heat option available when room is near preferred."""
        options = _mini_split_sweep_options(
            "bedroom", self._bedroom_schedule(), 12, 42.0,
            current_temps={"bedroom": 70.5},  # within proportional_band of preferred 70
        )
        assert any(o.mode == "heat" for o in options)


# ── Proportional energy cost tests ──────────────────────────────────


class TestProportionalEnergyCost:
    """Test proportional energy cost for mini splits."""

    def test_higher_target_more_energy_when_heating(self) -> None:
        """Higher target vs outdoor -> more expected activity -> higher cost.

        With proportional_band=1.0: target 68.5 @ outdoor 68 -> activity=0.5,
        target 69 @ outdoor 68 -> activity=1.0.
        """
        low_target = TrajectoryScenario(
            ThermostatTrajectory(heating=False),
            ThermostatTrajectory(heating=False),
            (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
            (MiniSplitDecision("bedroom", "heat", 68.5), MiniSplitDecision("living_room", "off", 0.0)),
        )
        high_target = TrajectoryScenario(
            ThermostatTrajectory(heating=False),
            ThermostatTrajectory(heating=False),
            (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
            (MiniSplitDecision("bedroom", "heat", 69.0), MiniSplitDecision("living_room", "off", 0.0)),
        )
        assert compute_energy_cost(low_target, outdoor_temp=68.0) < compute_energy_cost(high_target, outdoor_temp=68.0)

    def test_off_mini_split_no_energy_cost(self) -> None:
        """Off mini split should have zero energy cost."""
        off = TrajectoryScenario(
            ThermostatTrajectory(heating=False),
            ThermostatTrajectory(heating=False),
            (BlowerDecision("family_room", "off"), BlowerDecision("office", "off")),
            (MiniSplitDecision("bedroom", "off", 0.0), MiniSplitDecision("living_room", "off", 0.0)),
        )
        assert compute_energy_cost(off, outdoor_temp=42.0) == 0.0


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
        scenarios = generate_trajectory_scenarios(schedules, base_hour=12, outdoor_temp=42.0)
        # 4 mini split combos (2×2): off + preferred per split
        split_combos = {tuple((sd.name, sd.mode, sd.target) for sd in s.mini_splits) for s in scenarios}
        assert len(split_combos) == 4

    def test_all_off_still_present(self) -> None:
        """All-off baseline should still be in scenario set."""
        schedules = [
            ComfortSchedule(
                label="bedroom",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 68.0, 72.0)),),
            ),
        ]
        scenarios = generate_trajectory_scenarios(schedules, base_hour=12, outdoor_temp=42.0)
        all_off = [
            s for s in scenarios
            if not s.upstairs.heating and not s.downstairs.heating
            and all(b.mode == "off" for b in s.blowers)
            and all(sp.mode == "off" for sp in s.mini_splits)
        ]
        assert len(all_off) == 1
