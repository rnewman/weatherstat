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
    _in_quiet_hours,
    adjust_schedules_for_windows,
    compute_comfort_cost,
    compute_energy_cost,
    generate_trajectory_scenarios,
)
from weatherstat.types import (
    BlowerDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    MiniSplitDecision,
    RoomComfort,
    ThermostatTrajectory,
    TrajectoryScenario,
)

# ── Test helpers ──────────────────────────────────────────────────────────


def make_schedules(**overrides: list[ComfortScheduleEntry]) -> list[ComfortSchedule]:
    """Return default_comfort_schedules() with optional room overrides.

    Usage:
        make_schedules(upstairs=[ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 70, 74))])
    """
    from weatherstat.control import default_comfort_schedules

    defaults = {s.room: s for s in default_comfort_schedules()}
    for room, entries in overrides.items():
        defaults[room] = ComfortSchedule(room=room, entries=tuple(entries))
    return list(defaults.values())


# ── Comfort cost tests ────────────────────────────────────────────────────


class TestComfortCost:
    """Test comfort cost computation."""

    def test_comfort_cost_quadratic_penalty(self) -> None:
        """Room at 76F, max 74F, hot_penalty 2.0 -> cost = (76-74)^2 * 2.0 * weight."""
        schedules = [
            ComfortSchedule(
                room="upstairs",
                entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 70.0, 74.0, hot_penalty=2.0)),),
            ),
        ]
        predictions = {"upstairs_temp_t+12": 76.0}
        cost = compute_comfort_cost(predictions, schedules, base_hour=12)

        expected = (76.0 - 74.0) ** 2 * 2.0 * HORIZON_WEIGHTS[12]
        assert abs(cost - expected) < 0.001

    def test_comfort_cost_respects_schedule_hour(self) -> None:
        """Bedroom at hour 22 (night: max 72F) vs hour 10 (day: max 75F) -> different costs at 74F."""
        from weatherstat.control import default_comfort_schedules

        schedules = default_comfort_schedules()
        bedroom_schedules = [s for s in schedules if s.room == "bedroom"]

        predictions = {"bedroom_temp_t+12": 74.0}

        # Hour 22 (night schedule: max 72F) -> 74 > 72, penalized
        cost_night = compute_comfort_cost(predictions, bedroom_schedules, base_hour=22)

        # Hour 10 (day schedule: max 75F) -> 74 < 75, no penalty
        cost_day = compute_comfort_cost(predictions, bedroom_schedules, base_hour=10)

        assert cost_night > cost_day
        assert cost_day == 0.0

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
            schedules, window_states, cfg.windows, -3.0, 2.0,
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
                ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 74.0, 2.0, 1.0)),
            ],
        )
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )

        bedroom_sched = next(s for s in adjusted if s.room == "bedroom")
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
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )

        upstairs_orig = next(s for s in schedules if s.room == "upstairs")
        upstairs_adj = next(s for s in adjusted if s.room == "upstairs")
        assert upstairs_orig is upstairs_adj  # unchanged

    def test_penalties_preserved(self) -> None:
        """Window adjustment preserves cold_penalty and hot_penalty."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules(
            bedroom=[
                ComfortScheduleEntry(0, 24, RoomComfort("bedroom", 70.0, 74.0, 3.0, 0.5)),
            ],
        )
        window_states = {name: False for name in cfg.windows}
        window_states["bedroom"] = True

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
        )

        bedroom_sched = next(s for s in adjusted if s.room == "bedroom")
        entry = bedroom_sched.entries[0]
        assert entry.comfort.cold_penalty == 3.0
        assert entry.comfort.hot_penalty == 0.5

    def test_window_without_rooms_no_effect(self) -> None:
        """Basement window (rooms=[]) open -> no schedules affected."""
        from weatherstat.yaml_config import load_config

        cfg = load_config()
        schedules = make_schedules()
        window_states = {name: False for name in cfg.windows}
        window_states["basement"] = True  # basement has rooms=[]

        adjusted = adjust_schedules_for_windows(
            schedules, window_states, cfg.windows, -3.0, 2.0,
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
