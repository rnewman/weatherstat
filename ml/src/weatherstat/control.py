"""Control policy — comfort-optimizing HVAC selection.

Receding-horizon controller: "what HVAC settings maximize comfort over the next 6 hours?"
Re-evaluated every 15 minutes. Physics-based trajectory sweep using forward simulation.

Control variables: 2 thermostats (binary on/off with delay x duration grid),
2 blowers (off/low/high), 2 mini-splits (off/heat/cool). Config-driven device lists
make adding blowers a single-line change.

Run:
  uv run python -m weatherstat.control            # single cycle, dry-run
  uv run python -m weatherstat.control --loop      # 15-min loop, dry-run
  uv run python -m weatherstat.control --live      # single cycle, live execution
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weatherstat.config import (
    BLOWERS,
    CONTROL_STATE_FILE,
    ENERGY_COST_BLOWER,
    ENERGY_COST_GAS_ZONE,
    ENERGY_COST_MINI_SPLIT,
    MINI_SPLITS,
    PREDICTION_LABELS,
    PREDICTIONS_DIR,
)
from weatherstat.extract import fetch_recent_history
from weatherstat.types import (
    BlowerDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlDecision,
    ControlState,
    MiniSplitDecision,
    RoomComfort,
    ThermostatTrajectory,
    TrajectoryScenario,
)
from weatherstat.yaml_config import load_config

_CFG = load_config()

# ── Constants ──────────────────────────────────────────────────────────────

# Cautious setpoint offset: when the control loop decides "heat on", we set the
# thermostat to current_temp + CAUTIOUS_OFFSET. If the loop is interrupted, the
# house drifts gently instead of running away to extreme temperatures.
CAUTIOUS_OFFSET = 2  # °F above/below current temp

# Absolute safety bounds (in case of stale current_temp or other weirdness)
ABSOLUTE_MIN = 62
ABSOLUTE_MAX = 78

# Minimum hold time before changing setpoints (seconds).
# Must be less than LOOP_INTERVAL_SECONDS so every cycle produces a fresh decision.
MIN_HOLD_SECONDS = 10 * 60  # 10 minutes

# Maximum data staleness before refusing to execute (seconds)
MAX_STALE_SECONDS = 15 * 60  # 15 minutes

# Maximum predicted 1h temperature change before logging warning
MAX_1H_CHANGE = 5.0  # °F

# Horizon weights: nearer predictions matter more, model is more accurate
HORIZON_WEIGHTS: dict[int, float] = {
    12: 1.0,  # 1h
    24: 0.9,  # 2h
    48: 0.7,  # 4h
    72: 0.5,  # 6h
    144: 0.3,  # 12h (low weight — far out, less reliable)
}

# Minimum cost improvement over all-off required to justify active HVAC.
# Prevents noise-driven decisions when model predictions barely differ between
# scenarios. Equivalent to requiring ~1°F of genuine comfort improvement at
# one horizon before turning on heating.
MIN_IMPROVEMENT = 1.0

# Cold-room override: force zone heating when a room's current temperature is
# this far below its comfort min. Compensates for undertrained models that
# predict rooms warming without HVAC (because training data is all HVAC-on).
COLD_ROOM_OVERRIDE = 1.0  # °F below comfort_min

# Control loop interval
LOOP_INTERVAL_SECONDS = 15 * 60  # 15 minutes

# Horizons used for control (subset of HORIZONS_5MIN, skip 12h for control)
CONTROL_HORIZONS = [12, 24, 48, 72]

# Human-readable labels for horizon steps (5-min intervals)
HORIZON_LABELS: dict[int, str] = {12: "1h", 24: "2h", 48: "4h", 72: "6h", 144: "12h"}

# Trajectory search grid for slow effectors (5-min steps)
# Coarser than PLAN-7's 5×4 to keep sweep under 5s.
# Receding horizon (15-min re-evaluation) compensates for granularity.
TRAJECTORY_DELAYS = [0, 12, 24]  # 0h, 1h, 2h
TRAJECTORY_DURATIONS = [12, 24, 72]  # 1h, 2h, 6h (full horizon)


# ── Default comfort profiles ──────────────────────────────────────────────


def default_comfort_schedules() -> list[ComfortSchedule]:
    """Comfort profiles from YAML config.

    All rooms with schedules are included so the optimizer considers whole-house comfort.
    Rooms without direct HVAC control use lighter penalties — there's only so much
    the thermostats can do for them.
    """
    schedules: list[ComfortSchedule] = []
    for label, entries in _CFG.comfort.items():
        schedule_entries = tuple(
            ComfortScheduleEntry(
                e.start_hour,
                e.end_hour,
                RoomComfort(label, e.preferred, e.min_temp, e.max_temp, e.cold_penalty, e.hot_penalty),
            )
            for e in entries
        )
        schedules.append(ComfortSchedule(label=label, entries=schedule_entries))
    return schedules


def adjust_schedules_for_windows(
    schedules: list[ComfortSchedule],
    window_states: dict[str, bool],
    constraint_labels: set[str],
    min_offset: float,
    max_offset: float,
) -> list[ComfortSchedule]:
    """Widen comfort bounds for constrained sensors with open windows.

    When a window is open, shift min_temp down and max_temp up for the
    matching constraint (by naming convention: window "bedroom" affects
    constraint label "bedroom"). This makes the optimizer less eager to
    heat/cool a sensor with an open window nearby.

    Args:
        schedules: Comfort schedules for all constrained sensors.
        window_states: window_name -> is_open for each window.
        constraint_labels: Set of all constraint labels.
        min_offset: Amount to add to min_temp (negative = lower).
        max_offset: Amount to add to max_temp (positive = higher).

    Returns:
        New list of ComfortSchedule with adjusted bounds for affected sensors.
    """
    # Window name matches constraint label → that window affects that sensor
    labels_with_open_windows: set[str] = {
        wname for wname, is_open in window_states.items()
        if is_open and wname in constraint_labels
    }

    if not labels_with_open_windows:
        return schedules

    adjusted: list[ComfortSchedule] = []
    for schedule in schedules:
        if schedule.label not in labels_with_open_windows:
            adjusted.append(schedule)
            continue
        new_entries = tuple(
            ComfortScheduleEntry(
                e.start_hour,
                e.end_hour,
                RoomComfort(
                    e.comfort.label,
                    e.comfort.preferred,  # preferred unchanged — window doesn't change ideal
                    e.comfort.min_temp + min_offset,
                    e.comfort.max_temp + max_offset,
                    e.comfort.cold_penalty,
                    e.comfort.hot_penalty,
                ),
            )
            for e in schedule.entries
        )
        adjusted.append(ComfortSchedule(label=schedule.label, entries=new_entries))
    return adjusted


def _in_quiet_hours(hour: int, quiet: tuple[int, int]) -> bool:
    """Return True if the given hour falls within quiet hours (wraps midnight)."""
    start, end = quiet
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


# ── Cost function ─────────────────────────────────────────────────────────


# Hard rail multiplier: additional penalty for exceeding min/max bounds.
# Applied on top of the continuous preferred-based cost.
_HARD_RAIL_MULTIPLIER = 10.0


def compute_comfort_cost(
    predictions: dict[str, float],
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> float:
    """Compute total comfort cost across all rooms and horizons.

    Two-layer cost model:
    1. Continuous: quadratic penalty for any deviation from `preferred`,
       weighted asymmetrically by cold_penalty (below) and hot_penalty (above).
    2. Hard rails: steep additional penalty (10×) for exceeding min/max bounds.

    This gives the optimizer a gradient everywhere — it always prefers
    temperatures closer to preferred, not just "anywhere in the band".

    Args:
        predictions: Model predictions keyed like "upstairs_temp_t+12".
        schedules: Comfort schedules for all rooms.
        base_hour: Current hour of day (0-23).

    Returns:
        Total weighted comfort cost.
    """
    cost = 0.0
    horizon_hours = {12: 1, 24: 2, 48: 4, 72: 6}

    for schedule in schedules:
        label = schedule.label
        for h in CONTROL_HORIZONS:
            weight = HORIZON_WEIGHTS.get(h, 0.5)
            hours_ahead = horizon_hours.get(h, h // 12)
            future_hour = (base_hour + hours_ahead) % 24

            comfort = schedule.comfort_at(future_hour)
            if comfort is None:
                continue

            pred_temp = predictions.get(f"{label}_temp_t+{h}")
            if pred_temp is None:
                continue

            # Continuous cost: quadratic deviation from preferred
            if pred_temp < comfort.preferred:
                cost += (comfort.preferred - pred_temp) ** 2 * comfort.cold_penalty * weight
            elif pred_temp > comfort.preferred:
                cost += (pred_temp - comfort.preferred) ** 2 * comfort.hot_penalty * weight

            # Hard rails: steep additional penalty outside min/max
            if pred_temp < comfort.min_temp:
                cost += (comfort.min_temp - pred_temp) ** 2 * comfort.cold_penalty * _HARD_RAIL_MULTIPLIER * weight
            elif pred_temp > comfort.max_temp:
                cost += (pred_temp - comfort.max_temp) ** 2 * comfort.hot_penalty * _HARD_RAIL_MULTIPLIER * weight

    return cost


def compute_comfort_cost_by_sensor(
    predictions: dict[str, float],
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> dict[str, float]:
    """Compute comfort cost per sensor for decision rationale.

    Same model as compute_comfort_cost, but returns per-sensor breakdown
    so we can identify which sensors drive the decision.
    """
    costs: dict[str, float] = {}
    horizon_hours = {12: 1, 24: 2, 48: 4, 72: 6}

    for schedule in schedules:
        label = schedule.label
        sensor_cost = 0.0
        for h in CONTROL_HORIZONS:
            weight = HORIZON_WEIGHTS.get(h, 0.5)
            hours_ahead = horizon_hours.get(h, h // 12)
            future_hour = (base_hour + hours_ahead) % 24

            comfort = schedule.comfort_at(future_hour)
            if comfort is None:
                continue

            pred_temp = predictions.get(f"{label}_temp_t+{h}")
            if pred_temp is None:
                continue

            if pred_temp < comfort.preferred:
                sensor_cost += (comfort.preferred - pred_temp) ** 2 * comfort.cold_penalty * weight
            elif pred_temp > comfort.preferred:
                sensor_cost += (pred_temp - comfort.preferred) ** 2 * comfort.hot_penalty * weight

            if pred_temp < comfort.min_temp:
                sensor_cost += (comfort.min_temp - pred_temp) ** 2 * comfort.cold_penalty * _HARD_RAIL_MULTIPLIER * weight
            elif pred_temp > comfort.max_temp:
                sensor_cost += (pred_temp - comfort.max_temp) ** 2 * comfort.hot_penalty * _HARD_RAIL_MULTIPLIER * weight
        costs[label] = sensor_cost
    return costs


def compute_energy_cost(
    scenario: TrajectoryScenario,
    current_temps: dict[str, float] | None = None,
) -> float:
    """Tiered energy penalty: gas zones > mini-splits > blower fans.

    Used as tiebreaker when comfort cost is equal — prefer less energy usage.
    Gas zone cost is proportional to heating duration within the horizon.
    Mini-split cost is proportional to expected activity (target vs room temp).
    """
    cost = 0.0
    # Gas zones (Navien boiler via thermostat)
    max_h = max(CONTROL_HORIZONS)
    for traj in [scenario.upstairs, scenario.downstairs]:
        if traj.heating:
            dur = traj.duration_steps if traj.duration_steps is not None else (max_h - traj.delay_steps)
            cost += ENERGY_COST_GAS_ZONE * dur / max_h
    # Mini-splits: proportional to expected activity based on room temp
    temps = current_temps or {}
    for sd in scenario.mini_splits:
        if sd.mode == "off":
            continue
        split_cfg = _CFG.mini_splits.get(sd.name)
        p_band = split_cfg.proportional_band if split_cfg else 1.0
        room_temp = temps.get(sd.name)
        if room_temp is not None:
            delta = max(0.0, sd.target - room_temp) if sd.mode == "heat" else max(0.0, room_temp - sd.target)
            avg_activity = min(1.0, delta / p_band)
        else:
            avg_activity = 0.5  # unknown room temp — assume moderate activity
        cost += ENERGY_COST_MINI_SPLIT * avg_activity
    # Blower fans (negligible)
    for bd in scenario.blowers:
        cost += ENERGY_COST_BLOWER.get(bd.mode, 0.0)
    return cost


def _derive_sensor_zones(gains: dict[tuple[str, str], tuple[float, float]]) -> dict[str, str]:
    """Derive sensor → zone mapping from the sysid coupling matrix.

    For each sensor, find the thermostat effector with the highest positive gain.
    Returns label -> zone_name (e.g., {"bedroom": "upstairs"}).
    """
    zones = _CFG.zones  # zone_name -> ZoneConfig(thermostat=...)
    thermostat_to_zone = {f"thermostat_{z.thermostat}": z.name for z in zones.values()}

    # Accumulate best thermostat gain per sensor
    best: dict[str, tuple[str, float]] = {}  # sensor -> (zone, gain)
    for (effector, sensor), (gain, _lag) in gains.items():
        if effector not in thermostat_to_zone or gain <= 0:
            continue
        zone = thermostat_to_zone[effector]
        label = sensor.removeprefix("thermostat_").removesuffix("_temp")
        prev = best.get(label)
        if prev is None or gain > prev[1]:
            best[label] = (zone, gain)

    return {label: zone for label, (zone, _) in best.items()}


def _zone_comfort_max(label: str, schedules: list[ComfortSchedule], hour: int) -> float:
    """Get the comfort max for a constraint label at the given hour."""
    for s in schedules:
        if s.label == label:
            c = s.comfort_at(hour)
            if c is not None:
                return c.max_temp
    return ABSOLUTE_MAX


def _zone_comfort_min(label: str, schedules: list[ComfortSchedule], hour: int) -> float:
    """Get the comfort min for a constraint label at the given hour."""
    for s in schedules:
        if s.label == label:
            c = s.comfort_at(hour)
            if c is not None:
                return c.min_temp
    return ABSOLUTE_MIN


def _cautious_setpoint(current_temp: float, heating: bool, comfort_min: float = ABSOLUTE_MIN) -> float:
    """Compute a cautious setpoint that achieves on/off without runaway risk.

    When heating ON: current_temp + offset (ensure thermostat fires).
    When heating OFF: comfort_min - 1 (thermostat acts as safety net at comfort floor).

    The OFF setpoint uses comfort_min — not current_temp - offset — so the
    thermostat prevents the house from cooling below the comfort range even
    if the control loop is interrupted.
    """
    sp = max(current_temp + CAUTIOUS_OFFSET, comfort_min + CAUTIOUS_OFFSET) if heating else (comfort_min - 1)
    return max(ABSOLUTE_MIN, min(ABSOLUTE_MAX, sp))


# ── Scenario generation & sweep ──────────────────────────────────────────


def _mini_split_sweep_options(
    split_name: str,
    schedules: list[ComfortSchedule],
    base_hour: int,
    prev_state: ControlState | None = None,
    current_temps: dict[str, float] | None = None,
) -> list[MiniSplitDecision]:
    """Generate sweep options for a regulating mini split: off + preferred target.

    Target is the preferred temperature from the comfort schedule.
    Mode is derived from room temp vs preferred: if the room is above
    preferred, cool; if below, heat.  Outdoor temp does not enter — it
    influences trajectories via the simulator, not mode intent.
    During mode hold windows, mode is locked to current.
    If the room is already past target by more than the proportional band,
    the split would be idle — skip the on option and let receding horizon
    re-evaluate when the room actually needs it.
    """
    split_cfg = _CFG.mini_splits.get(split_name)
    options: list[MiniSplitDecision] = [MiniSplitDecision(split_name, "off", 0.0)]

    # Find preferred temperature for this split's sensor
    preferred: float | None = None
    for sched in schedules:
        if sched.label == split_name:
            comfort = sched.comfort_at(base_hour)
            if comfort is not None:
                preferred = comfort.preferred
            break

    if preferred is None:
        return options

    # Derive mode from room temp vs preferred
    room_temp = (current_temps or {}).get(split_name)
    if room_temp is not None:
        mode = "cool" if room_temp > preferred else "heat"
    else:
        # No room temp available — default to heat (safer in winter)
        mode = "heat"

    # Skip on-option if the split would be idle: room already past target
    # by more than the proportional band.  Receding horizon (15-min
    # re-evaluation) will add it back when the room actually needs it.
    p_band = split_cfg.proportional_band if split_cfg else 1.0
    if room_temp is not None:
        if mode == "heat" and room_temp > preferred + p_band:
            return options  # room well above target — split would be idle
        if mode == "cool" and room_temp < preferred - p_band:
            return options  # room well below target — split would be idle

    options.append(MiniSplitDecision(split_name, mode, round(preferred, 1)))

    # Mode hold: lock to current mode during quiet-hours window (no compressor starts/stops)
    hold_window = split_cfg.mode_hold_window if split_cfg else None
    if prev_state and hold_window and _in_hold_window(base_hour, hold_window):
        prev_mode = prev_state.mini_split_modes.get(split_name, "off")
        options = [o for o in options if o.mode == prev_mode]
        if not options:
            prev_target = prev_state.mini_split_targets.get(split_name, 0.0)
            options = [MiniSplitDecision(split_name, prev_mode, prev_target)]

    return options


def _in_hold_window(hour: int, window: tuple[int, int]) -> bool:
    """Check if hour falls within [start, end) window, wrapping midnight."""
    start, end = window
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def generate_trajectory_scenarios(
    schedules: list[ComfortSchedule] | None = None,
    base_hour: int = 12,
    prev_state: ControlState | None = None,
    current_temps: dict[str, float] | None = None,
) -> list[TrajectoryScenario]:
    """Generate trajectory scenarios for physics sweep.

    Slow effectors (thermostats) get a delay × duration grid.
    Fast effectors (blowers) use constant modes.
    Mini splits use comfort-derived target grid (off + 3 targets per split).
    Blower constraint: only active when zone thermostat is heating.
    """
    from itertools import product

    max_horizon = max(CONTROL_HORIZONS)

    # Build thermostat trajectory options: OFF + delay×duration grid
    traj_options: list[ThermostatTrajectory] = [ThermostatTrajectory(heating=False)]
    for delay in TRAJECTORY_DELAYS:
        if delay >= max_horizon:
            continue
        for duration in TRAJECTORY_DURATIONS:
            # Cap duration at remaining horizon
            effective_duration = min(duration, max_horizon - delay)
            traj_options.append(ThermostatTrajectory(
                heating=True,
                delay_steps=delay,
                duration_steps=effective_duration,
            ))

    # Deduplicate (capping creates duplicates)
    traj_options = list(dict.fromkeys(traj_options))

    # Mini-split combinations: target grid from comfort schedule
    if schedules is not None:
        per_split_options = [
            _mini_split_sweep_options(s.name, schedules, base_hour, prev_state, current_temps)
            for s in MINI_SPLITS
        ]
    else:
        # Fallback for backward compat (e.g., scenario counting without schedules)
        per_split_options = [
            [MiniSplitDecision(s.name, "off", 0.0), MiniSplitDecision(s.name, "heat", 70.0)]
            for s in MINI_SPLITS
        ]

    split_combos: list[tuple[MiniSplitDecision, ...]] = [
        combo for combo in product(*per_split_options)
    ]

    # Full cartesian product with blower constraint
    scenarios: list[TrajectoryScenario] = []
    for up_traj in traj_options:
        for dn_traj in traj_options:
            # Blower constraint: only sweep blower levels when zone thermostat
            # starts immediately (delay=0).  Delayed trajectories start heating
            # later — receding horizon will add blower when heat actually begins.
            immediate_heat = {
                "upstairs": up_traj.heating and up_traj.delay_steps == 0,
                "downstairs": dn_traj.heating and dn_traj.delay_steps == 0,
            }
            per_blower_levels = []
            for b in BLOWERS:
                if immediate_heat.get(b.zone, False):
                    per_blower_levels.append(b.levels)
                else:
                    per_blower_levels.append(("off",))

            for levels in product(*per_blower_levels):
                blowers = tuple(BlowerDecision(b.name, lvl) for b, lvl in zip(BLOWERS, levels, strict=True))
                for splits in split_combos:
                    scenarios.append(TrajectoryScenario(up_traj, dn_traj, blowers, splits))

    return scenarios


# ── Physics-based sweep ──────────────────────────────────────────────────


def sweep_scenarios_physics(
    current_temps: dict[str, float],
    outdoor_temp: float,
    forecast_temps: list[float],
    window_states: dict[str, bool],
    sim_params: object,
    hour_of_day: float,
    recent_history: dict[str, list[float]],
    up_current: float,
    dn_current: float,
    current_split_temps: dict[str, float],
    schedules: list[ComfortSchedule],
    base_hour: int,
    prev_state: ControlState | None = None,
    solar_fractions: list[float] | None = None,
) -> tuple[ControlDecision, TrajectoryScenario]:
    """Sweep trajectory scenarios using physics simulator predictions.

    Thermostats are evaluated over a delay × duration grid (trajectory search).
    Fast effectors (blowers, mini-splits) use constant modes.
    """
    from weatherstat.simulator import HouseState, SimParams, predict

    assert isinstance(sim_params, SimParams)

    sensor_zones = _derive_sensor_zones(sim_params.gains)
    scenarios = generate_trajectory_scenarios(schedules, base_hour, prev_state, current_temps)
    pre_count = len(scenarios)
    blocked_reasons: list[str] = []

    # ── Physical constraints ──
    up_max = _zone_comfort_max("upstairs", schedules, base_hour)
    dn_max = _zone_comfort_max("downstairs", schedules, base_hour)
    up_allow = up_current < up_max
    dn_allow = dn_current < dn_max
    if not up_allow or not dn_allow:
        scenarios = [
            s for s in scenarios
            if (up_allow or not s.upstairs.heating) and (dn_allow or not s.downstairs.heating)
        ]
        if not up_allow:
            blocked_reasons.append(f"upstairs at/above max ({up_current:.1f}°F >= {up_max:.0f}°F)")
        if not dn_allow:
            blocked_reasons.append(f"downstairs at/above max ({dn_current:.1f}°F >= {dn_max:.0f}°F)")

    # Note: no "thermal direction" pruning — the trajectory sweep and cost
    # function decide whether heating is justified.  Pre-emptive heating for
    # slow effectors (hydronic slab: 45-75 min lag) requires evaluating
    # futures, not checking current temps against comfort min.

    if blocked_reasons:
        print(f"  Heating blocked: {'; '.join(blocked_reasons)}")
        print(f"  Reduced scenarios: {pre_count} → {len(scenarios)}")

    def _is_all_off(s: TrajectoryScenario) -> bool:
        return (
            not s.upstairs.heating
            and not s.downstairs.heating
            and all(b.mode == "off" for b in s.blowers)
            and all(sp.mode == "off" for sp in s.mini_splits)
        )

    # ── Batch simulate all scenarios ──
    sweep_state = HouseState(
        current_temps=current_temps,
        outdoor_temp=outdoor_temp,
        forecast_temps=forecast_temps,
        window_states=window_states,
        hour_of_day=hour_of_day,
        recent_history=recent_history,
        solar_fractions=solar_fractions or [],
    )
    target_names, pred_matrix = predict(sweep_state, scenarios, sim_params, CONTROL_HORIZONS)
    target_idx = {t: j for j, t in enumerate(target_names)}

    # ── Score each scenario ──
    best_idx = -1
    best_cost = float("inf")
    off_idx = -1
    off_cost = float("inf")

    for i, scenario in enumerate(scenarios):
        predictions = {t: float(pred_matrix[i, j]) for t, j in target_idx.items()}
        comfort = compute_comfort_cost(predictions, schedules, base_hour)
        energy = compute_energy_cost(scenario, current_temps)
        total = comfort + energy

        if _is_all_off(scenario):
            off_idx = i
            off_cost = total

        if total < best_cost:
            best_cost = total
            best_idx = i

    if best_idx < 0:
        raise RuntimeError("No HVAC scenarios evaluated")

    # Minimum improvement safeguard
    if off_idx >= 0 and best_idx != off_idx:
        improvement = off_cost - best_cost
        if improvement < MIN_IMPROVEMENT:
            print(f"  Reverting to all-off: improvement {improvement:.3f} < threshold {MIN_IMPROVEMENT:.1f}")
            best_idx = off_idx

    # ── Cold-sensor safety override ──
    # Force immediate heating (delay=0) when a sensor is significantly below comfort min
    if _is_all_off(scenarios[best_idx]):
        cold_zones: set[str] = set()
        cold_info: list[str] = []
        for schedule in schedules:
            label = schedule.label
            temp = current_temps.get(label)
            if temp is None:
                continue
            comfort = schedule.comfort_at(base_hour)
            if comfort is None:
                continue
            if temp < comfort.min_temp - COLD_ROOM_OVERRIDE:
                zone = sensor_zones.get(label)
                if zone:
                    cold_zones.add(zone)
                    cold_info.append(f"{label} ({temp:.1f}°F < {comfort.min_temp:.0f}°F)")

        if not up_allow:
            cold_zones.discard("upstairs")
        if not dn_allow:
            cold_zones.discard("downstairs")

        if cold_zones:
            constrained_best = -1
            constrained_cost = float("inf")
            for i, scenario in enumerate(scenarios):
                # Cold-room override requires immediate heating (delay=0)
                if "upstairs" in cold_zones and not (scenario.upstairs.heating and scenario.upstairs.delay_steps == 0):
                    continue
                if "downstairs" in cold_zones and not (
                    scenario.downstairs.heating and scenario.downstairs.delay_steps == 0
                ):
                    continue
                predictions = {t: float(pred_matrix[i, j]) for t, j in target_idx.items()}
                c = compute_comfort_cost(predictions, schedules, base_hour)
                e = compute_energy_cost(scenario, current_temps)
                if c + e < constrained_cost:
                    constrained_cost = c + e
                    constrained_best = i
            if constrained_best >= 0:
                print(f"  Cold sensor override: {', '.join(cold_info)}")
                best_idx = constrained_best

    # ── Build ControlDecision ──
    scenario = scenarios[best_idx]
    predictions = {t: float(pred_matrix[best_idx, j]) for t, j in target_idx.items()}
    comfort = compute_comfort_cost(predictions, schedules, base_hour)
    energy = compute_energy_cost(scenario, current_temps)
    total = comfort + energy

    # Immediate action: heating only if trajectory starts now (delay=0)
    up_heating_now = scenario.upstairs.heating and scenario.upstairs.delay_steps == 0
    dn_heating_now = scenario.downstairs.heating and scenario.downstairs.delay_steps == 0

    # Build trajectory info for logging/display
    trajectory_info: dict[str, dict[str, int | None]] = {}
    if scenario.upstairs.heating:
        trajectory_info["upstairs"] = {
            "delay_steps": scenario.upstairs.delay_steps,
            "duration_steps": scenario.upstairs.duration_steps,
        }
    if scenario.downstairs.heating:
        trajectory_info["downstairs"] = {
            "delay_steps": scenario.downstairs.delay_steps,
            "duration_steps": scenario.downstairs.duration_steps,
        }

    label_preds: dict[str, dict[str, float]] = {}
    for label in PREDICTION_LABELS:
        rpred: dict[str, float] = {}
        for h in CONTROL_HORIZONS:
            key = f"{label}_temp_t+{h}"
            val = predictions.get(key)
            if val is not None:
                rpred[HORIZON_LABELS[h]] = round(val, 2)
        if rpred:
            label_preds[label] = rpred

    decision = ControlDecision(
        timestamp=datetime.now(UTC).isoformat(),
        upstairs_heating=up_heating_now,
        downstairs_heating=dn_heating_now,
        upstairs_setpoint=_cautious_setpoint(
            up_current,
            up_heating_now,
            comfort_min=_zone_comfort_min("upstairs", schedules, base_hour),
        ),
        downstairs_setpoint=_cautious_setpoint(
            dn_current,
            dn_heating_now,
            comfort_min=_zone_comfort_min("downstairs", schedules, base_hour),
        ),
        blowers=scenario.blowers,
        mini_splits=scenario.mini_splits,
        total_cost=round(total, 4),
        comfort_cost=round(comfort, 4),
        energy_cost=round(energy, 4),
        predictions=label_preds,
        trajectory_info=trajectory_info,
    )
    return decision, scenario


# ── State persistence ─────────────────────────────────────────────────────


def load_control_state() -> ControlState | None:
    """Load persisted control state, or None if not found.

    Backward-compatible: old state files without blower/mini-split fields load cleanly.
    """
    if not CONTROL_STATE_FILE.exists():
        return None
    try:
        data = json.loads(CONTROL_STATE_FILE.read_text())
        return ControlState(
            last_decision_time=data["last_decision_time"],
            upstairs_setpoint=data["upstairs_setpoint"],
            downstairs_setpoint=data["downstairs_setpoint"],
            blower_modes=data.get("blower_modes", {}),
            mini_split_modes=data.get("mini_split_modes", {}),
            mini_split_targets=data.get("mini_split_targets", {}),
            mini_split_mode_times=data.get("mini_split_mode_times", {}),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def save_control_state(decision: ControlDecision, prev_state: ControlState | None = None) -> None:
    """Persist control state to prevent rapid cycling.

    Tracks mini-split mode change timestamps: only updated when mode actually changes.
    """
    now_iso = decision.timestamp
    new_modes = {sd.name: sd.mode for sd in decision.mini_splits}
    prev_modes = prev_state.mini_split_modes if prev_state else {}
    prev_mode_times = dict(prev_state.mini_split_mode_times) if prev_state else {}

    # Update mode-change timestamps only when mode actually changes
    for name, mode in new_modes.items():
        if mode != prev_modes.get(name, ""):
            prev_mode_times[name] = now_iso

    state: dict[str, object] = {
        "last_decision_time": now_iso,
        "upstairs_setpoint": decision.upstairs_setpoint,
        "downstairs_setpoint": decision.downstairs_setpoint,
        "blower_modes": {bd.name: bd.mode for bd in decision.blowers},
        "mini_split_modes": new_modes,
        "mini_split_targets": {sd.name: sd.target for sd in decision.mini_splits},
        "mini_split_mode_times": prev_mode_times,
    }
    CONTROL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_STATE_FILE.write_text(json.dumps(state, indent=2))


def should_hold(state: ControlState | None) -> bool:
    """Return True if we should hold the current setpoints (min hold time)."""
    if state is None:
        return False
    try:
        last_time = datetime.fromisoformat(state.last_decision_time)
        elapsed = (datetime.now(UTC) - last_time).total_seconds()
        return elapsed < MIN_HOLD_SECONDS
    except (ValueError, TypeError):
        return False


# ── Safety checks ─────────────────────────────────────────────────────────


def check_data_freshness(df: pd.DataFrame) -> bool:
    """Return True if latest data is fresh enough for control."""
    latest_ts_str = df["timestamp"].iloc[-1]
    try:
        latest_ts = pd.to_datetime(latest_ts_str, utc=True)
        age = (datetime.now(UTC) - latest_ts.to_pydatetime()).total_seconds()
        if age > MAX_STALE_SECONDS:
            print(f"  WARNING: Data is {age / 60:.0f}m old (max {MAX_STALE_SECONDS // 60}m)")
            return False
    except (ValueError, TypeError):
        print(f"  WARNING: Cannot parse timestamp: {latest_ts_str}")
        return False
    return True


def check_prediction_sanity(
    decision: ControlDecision,
    current_temps: dict[str, float],
) -> bool:
    """Return True if 1h predictions look reasonable for all labels."""
    safe = True
    for label, preds in decision.predictions.items():
        pred_1h = preds.get("1h")
        current = current_temps.get(label)
        if pred_1h is None or current is None:
            continue
        if abs(pred_1h - current) > MAX_1H_CHANGE:
            print(f"  WARNING: {label} 1h pred {pred_1h:.1f}°F >{MAX_1H_CHANGE}°F from current {current:.1f}°F")
            safe = False
    return safe


# ── Command JSON output ───────────────────────────────────────────────────


def write_command_json(decision: ControlDecision) -> Path:
    """Write executor-compatible command JSON.

    Uses camelCase keys matching the TS Prediction interface.
    All device values come from the decision (no pass-through from current state).
    """
    command: dict[str, object] = {
        "timestamp": decision.timestamp,
        "thermostatUpstairsTarget": decision.upstairs_setpoint,
        "thermostatDownstairsTarget": decision.downstairs_setpoint,
        "confidence": 1.0 - min(decision.total_cost / 10.0, 0.9),
    }

    # Blowers
    for bd in decision.blowers:
        cfg = next(b for b in BLOWERS if b.name == bd.name)
        command[cfg.command_key] = bd.mode

    # Mini-splits
    for sd in decision.mini_splits:
        cfg = next(s for s in MINI_SPLITS if s.name == sd.name)
        command[cfg.command_mode_key] = sd.mode
        command[cfg.command_target_key] = sd.target

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = PREDICTIONS_DIR / f"command_{date_str}.json"
    output_path.write_text(json.dumps(command, indent=2))
    return output_path


# ── Main control cycle ────────────────────────────────────────────────────


def run_control_cycle(live: bool = False) -> ControlDecision | None:
    """Run a single control cycle.

    Args:
        live: If True, write command JSON for executor. If False, dry-run only.

    Returns:
        The control decision, or None if skipped.
    """
    from weatherstat.extract import _check_config

    _check_config()

    # Backfill outcomes for previous decisions before starting this cycle
    from weatherstat.decision_log import backfill_outcomes

    n_backfilled = backfill_outcomes()
    if n_backfilled:
        print(f"[control] Backfilled outcomes for {n_backfilled} previous decision(s)")

    # Fetch data
    print("[control] Fetching recent history from Home Assistant...")
    df_raw, forecast = fetch_recent_history(hours_back=14)
    n_rows = len(df_raw)
    print(f"  Retrieved {n_rows} rows")

    if n_rows < 24:
        print(f"  ERROR: only {n_rows} rows, need >= 24.", file=sys.stderr)
        return None

    # Check freshness
    if live and not check_data_freshness(df_raw):
        print("  Refusing to execute with stale data (dry-run would proceed)")
        return None

    # Current state
    room_temp_columns = _CFG.room_temp_columns

    latest = df_raw.iloc[-1]
    up_current = float(latest.get("thermostat_upstairs_temp", 70))
    dn_current = float(latest.get("thermostat_downstairs_temp", 70))
    up_target = latest.get("thermostat_upstairs_target", "?")
    dn_target = latest.get("thermostat_downstairs_target", "?")
    out_temp = latest.get("outdoor_temp")
    now_str = df_raw["timestamp"].iloc[-1]

    # Build current temperature dict for all rooms (for sanity checks and display)
    current_temps: dict[str, float] = {}
    for room, col in room_temp_columns.items():
        val = latest.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            current_temps[room] = float(val)

    # Current mini-split temps for delta computation during sweep
    current_split_temps: dict[str, float] = {}
    for cfg in MINI_SPLITS:
        val = latest.get(cfg.temp_col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            current_split_temps[cfg.name] = float(val)

    print(f"\n[control] Current state ({now_str}):")
    print(f"  Upstairs:    {up_current:.1f}°F (setpoint: {up_target}°F)")
    print(f"  Downstairs:  {dn_current:.1f}°F (setpoint: {dn_target}°F)")
    if out_temp is not None and not (isinstance(out_temp, float) and np.isnan(out_temp)):
        print(f"  Outdoor:     {float(out_temp):.1f}°F (sensor)")
    window_cols = _CFG.window_display_map
    open_windows = [label for col, label in window_cols.items() if bool(latest.get(col, False))]
    if open_windows:
        print(f"  Windows:     {', '.join(open_windows)} open")
    else:
        print("  Windows:     all closed")
    other_labels = [la for la in PREDICTION_LABELS if la not in ("upstairs", "downstairs")]
    lw = max((len(la) for la in other_labels), default=14) + 2
    for la in other_labels:
        t = current_temps.get(la)
        if t is not None:
            print(f"  {la:<{lw}} {t:.1f}°F")
    # Show current blower/mini-split state
    for cfg in BLOWERS:
        mode = str(latest.get(f"blower_{cfg.name}_mode", "?"))
        print(f"  blower_{cfg.name:<10} {mode}")
    for cfg in MINI_SPLITS:
        mode = str(latest.get(f"mini_split_{cfg.name}_mode", "?"))
        target = latest.get(f"mini_split_{cfg.name}_target", "?")
        print(f"  split_{cfg.name:<12} {mode} @ {target}°F")

    # Show forecast summary
    if forecast:
        from datetime import UTC as _utc

        from weatherstat.forecast import forecast_at_horizons

        ref_time = datetime.now(_utc)
        at_horizons = forecast_at_horizons(forecast, ref_time, [1, 2, 4, 6, 12])
        parts: list[str] = []
        for label in ["1h", "2h", "4h", "6h", "12h"]:
            entry = at_horizons.get(label)
            if entry is not None:
                parts.append(f"{label}:{entry.temperature:.0f}°F")
        if parts:
            print(f"  Forecast:    {', '.join(parts)}")
    else:
        print("  Forecast:    unavailable")

    # Current weather condition for solar fraction
    _current_cond = str(latest.get("weather_condition", "unknown")) if hasattr(latest, "get") else "unknown"
    from weatherstat.weather import condition_to_solar_fraction as _c2sf

    print(f"  Weather:     {_current_cond} (solar fraction: {_c2sf(_current_cond):.0%})")

    # Current local hour for comfort schedule lookup
    from zoneinfo import ZoneInfo

    from weatherstat.config import TIMEZONE

    local_now = datetime.now(ZoneInfo(TIMEZONE))
    base_hour = local_now.hour

    # Comfort schedules + window adjustments
    schedules = default_comfort_schedules()
    window_states_dict = {name: bool(latest.get(f"window_{name}_open", False)) for name in _CFG.windows}
    constraint_labels = {c.label for c in _CFG.constraints}
    schedules = adjust_schedules_for_windows(
        schedules,
        window_states_dict,
        constraint_labels,
        *_CFG.window_open_offset,
    )
    labels_with_open = {wn for wn, ws in window_states_dict.items() if ws and wn in constraint_labels}
    if labels_with_open:
        print(f"  Comfort adjusted for open windows: {', '.join(sorted(labels_with_open))}")

    # Physics-based sweep using forward simulator
    from weatherstat.simulator import extract_recent_history, load_sim_params

    sim_params = load_sim_params()
    recent_hist = extract_recent_history(df_raw, sim_params)

    # Build forecast temp list and solar fractions for simulator
    from weatherstat.weather import condition_to_solar_fraction

    forecast_temp_list: list[float] = []
    solar_fractions: list[float] = []
    # Index 0 = current hour's solar fraction (from latest snapshot condition)
    current_condition = str(latest.get("weather_condition", "unknown")) if hasattr(latest, "get") else "unknown"
    solar_fractions.append(condition_to_solar_fraction(current_condition))

    if forecast:
        from weatherstat.forecast import forecast_at_horizons as _fah

        ref_time = datetime.now(UTC)
        # Get hourly forecasts for the next 12 hours
        at_h = _fah(forecast, ref_time, list(range(1, 13)))
        for h in range(1, 13):
            entry = at_h.get(f"{h}h")
            if entry is not None:
                forecast_temp_list.append(entry.temperature)
                solar_fractions.append(condition_to_solar_fraction(entry.condition))
            elif forecast_temp_list:
                forecast_temp_list.append(forecast_temp_list[-1])
                solar_fractions.append(solar_fractions[-1])
            else:
                forecast_temp_list.append(float(out_temp) if out_temp is not None else 50.0)
                solar_fractions.append(solar_fractions[-1])

    fractional_hour = base_hour + local_now.minute / 60.0

    # Load previous state for mode hold enforcement
    prev_state = load_control_state()

    # Outdoor temp for simulator: prefer first forecast temp (met.no, accurate)
    # over the side sensor (solar-heated, unreliable during daytime).
    if forecast_temp_list:
        outdoor = forecast_temp_list[0]
    elif out_temp is not None and not (isinstance(out_temp, float) and np.isnan(out_temp)):
        outdoor = float(out_temp)
    else:
        outdoor = 50.0
    n_scenarios = len(generate_trajectory_scenarios(schedules, base_hour, prev_state, current_temps))
    print(f"\n[control] Sweeping {n_scenarios} trajectory scenarios...")
    t0 = time.monotonic()
    decision, winning_scenario = sweep_scenarios_physics(
        current_temps,
        outdoor,
        forecast_temp_list,
        window_states_dict,
        sim_params,
        fractional_hour,
        recent_hist,
        up_current,
        dn_current,
        current_split_temps,
        schedules,
        base_hour,
        prev_state,
        solar_fractions,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    print(f"  Sweep completed in {elapsed_ms:.0f}ms ({elapsed_ms / n_scenarios:.1f}ms/combo)")

    # ── Compute baselines for rationale ──
    horizons = [HORIZON_LABELS[h] for h in CONTROL_HORIZONS]

    from weatherstat.simulator import HouseState as _HS
    from weatherstat.simulator import predict as _predict

    sim_state = _HS(
        current_temps=current_temps,
        outdoor_temp=outdoor,
        forecast_temps=forecast_temp_list,
        window_states=window_states_dict,
        hour_of_day=fractional_hour,
        recent_history=recent_hist,
        solar_fractions=solar_fractions,
    )

    # All-off baseline
    all_off = TrajectoryScenario(
        ThermostatTrajectory(heating=False),
        ThermostatTrajectory(heating=False),
        tuple(BlowerDecision(b.name, "off") for b in BLOWERS),
        tuple(MiniSplitDecision(s.name, "off", 0.0) for s in MINI_SPLITS),
    )
    off_targets, off_matrix = _predict(sim_state, [all_off], sim_params, CONTROL_HORIZONS)
    off_preds = {t: float(off_matrix[0, j]) for j, t in enumerate(off_targets)}
    off_comfort = compute_comfort_cost(off_preds, schedules, base_hour)

    # Per-device counterfactuals: winning scenario with each active device removed.
    # This gives true per-device attribution (what does THIS device contribute?).
    counterfactuals: list[TrajectoryScenario] = []
    cf_device_keys: list[str] = []

    if winning_scenario.upstairs.heating:
        cf = TrajectoryScenario(
            ThermostatTrajectory(heating=False),
            winning_scenario.downstairs, winning_scenario.blowers, winning_scenario.mini_splits,
        )
        counterfactuals.append(cf)
        cf_device_keys.append("upstairs_heating")
    if winning_scenario.downstairs.heating:
        cf = TrajectoryScenario(
            winning_scenario.upstairs, ThermostatTrajectory(heating=False),
            winning_scenario.blowers, winning_scenario.mini_splits,
        )
        counterfactuals.append(cf)
        cf_device_keys.append("downstairs_heating")
    for i, bd in enumerate(winning_scenario.blowers):
        if bd.mode != "off":
            new_blowers = tuple(
                BlowerDecision(b.name, "off") if j == i else b
                for j, b in enumerate(winning_scenario.blowers)
            )
            cf = TrajectoryScenario(
                winning_scenario.upstairs, winning_scenario.downstairs,
                new_blowers, winning_scenario.mini_splits,
            )
            counterfactuals.append(cf)
            cf_device_keys.append(f"blower_{bd.name}")
    for i, sd in enumerate(winning_scenario.mini_splits):
        if sd.mode != "off":
            new_splits = tuple(
                MiniSplitDecision(s.name, "off", 0.0) if j == i else s
                for j, s in enumerate(winning_scenario.mini_splits)
            )
            cf = TrajectoryScenario(
                winning_scenario.upstairs, winning_scenario.downstairs,
                winning_scenario.blowers, new_splits,
            )
            counterfactuals.append(cf)
            cf_device_keys.append(f"split_{sd.name}")

    # Simulate all counterfactuals in one batch
    cf_preds: dict[str, dict[str, float]] = {}
    cf_comfort_costs: dict[str, float] = {}
    if counterfactuals:
        cf_targets, cf_matrix = _predict(sim_state, counterfactuals, sim_params, CONTROL_HORIZONS)
        for idx, key in enumerate(cf_device_keys):
            preds = {t: float(cf_matrix[idx, j]) for j, t in enumerate(cf_targets)}
            cf_preds[key] = preds
            cf_comfort_costs[key] = compute_comfort_cost(preds, schedules, base_hour)

    # Decision predictions in flat format for per-sensor breakdown
    dec_flat: dict[str, float] = {}
    for label, preds in decision.predictions.items():
        for h in CONTROL_HORIZONS:
            h_label = HORIZON_LABELS[h]
            if h_label in preds:
                dec_flat[f"{label}_temp_t+{h}"] = preds[h_label]
    dec_sensor_costs = compute_comfort_cost_by_sensor(dec_flat, schedules, base_hour)
    off_sensor_costs = compute_comfort_cost_by_sensor(off_preds, schedules, base_hour)

    up_label = "ON" if decision.upstairs_heating else "OFF"
    dn_label = "ON" if decision.downstairs_heating else "OFF"

    # ── Print decision with counterfactual rationale ──
    print("\n[control] Decision:")

    def _counterfactual_rationale(device_key: str) -> str:
        """Build rationale: what does removing this device change?

        Compares winning scenario vs counterfactual (same scenario minus this device).
        Uses per-sensor comfort cost to find the biggest cost driver, then shows
        the trajectory at the horizon with the biggest delta for that sensor.
        """
        without = cf_preds.get(device_key)
        if without is None:
            return ""

        # Per-sensor comfort costs for counterfactual
        cf_sensor = compute_comfort_cost_by_sensor(without, schedules, base_hour)

        # Total comfort saving from this device
        total_saving = cf_comfort_costs.get(device_key, decision.comfort_cost) - decision.comfort_cost

        if abs(total_saving) < 0.1:
            return "  → no significant effect"

        # Find sensor with biggest per-sensor cost saving from this device
        best_sensor = ""
        best_sensor_saving = 0.0
        for s in set(dec_sensor_costs) | set(cf_sensor):
            s_saving = cf_sensor.get(s, 0) - dec_sensor_costs.get(s, 0)
            if abs(s_saving) > abs(best_sensor_saving):
                best_sensor_saving = s_saving
                best_sensor = s

        # Find the biggest trajectory delta for that sensor
        best_h_label = ""
        best_dec_t = 0.0
        best_cf_t = 0.0
        best_delta = 0.0
        if best_sensor:
            for h_step, h_lbl in zip(CONTROL_HORIZONS, horizons, strict=True):
                dec_t = decision.predictions.get(best_sensor, {}).get(h_lbl)
                cf_t = without.get(f"{best_sensor}_temp_t+{h_step}")
                if dec_t is not None and cf_t is not None:
                    delta = abs(dec_t - cf_t)
                    if delta > best_delta:
                        best_delta = delta
                        best_h_label = h_lbl
                        best_dec_t = dec_t
                        best_cf_t = cf_t

        if best_delta < 0.05:
            return f"  → comfort {total_saving:+.2f} (diffuse effects across sensors)"

        parts = [
            f"  → {best_sensor} at {best_h_label}: {best_dec_t:.1f}° vs {best_cf_t:.1f}° without",
            f" ({best_sensor} {best_sensor_saving:+.2f}",
        ]
        # Show total if it differs significantly from the top sensor
        if abs(total_saving - best_sensor_saving) > 0.5:
            parts.append(f", total {total_saving:+.2f}")
        parts.append(")")
        return "".join(parts)

    print(f"  Upstairs heating:   {up_label} → setpoint {decision.upstairs_setpoint:.0f}°F")
    if decision.upstairs_heating:
        print(_counterfactual_rationale("upstairs_heating"))
    print(f"  Downstairs heating: {dn_label} → setpoint {decision.downstairs_setpoint:.0f}°F")
    if decision.downstairs_heating:
        print(_counterfactual_rationale("downstairs_heating"))
    for bd in decision.blowers:
        if bd.mode == "off":
            print(f"  Blower {bd.name:<14} {bd.mode}")
        else:
            print(f"  Blower {bd.name:<14} {bd.mode}")
            print(_counterfactual_rationale(f"blower_{bd.name}"))
    for sd in decision.mini_splits:
        if sd.mode == "off":
            print(f"  Split {sd.name:<15} off")
        else:
            print(f"  Split {sd.name:<15} {sd.mode} @ {sd.target:.0f}°F")
            print(_counterfactual_rationale(f"split_{sd.name}"))
    if decision.trajectory_info:
        for zone, info in decision.trajectory_info.items():
            delay_h = info["delay_steps"] * 5 / 60
            dur = info.get("duration_steps")
            dur_str = f"{dur * 5 / 60:.0f}h" if dur is not None else "full"
            label = "ON now" if info["delay_steps"] == 0 else f"start in {delay_h:.0f}h"
            print(f"  Trajectory {zone}: {label}, duration {dur_str}")
    print(
        f"  Total cost: {decision.total_cost:.4f}"
        f" (comfort: {decision.comfort_cost:.4f}, energy: {decision.energy_cost:.4f})"
    )
    print(f"  All-off baseline: comfort={off_comfort:.4f}")

    # ── Per-sensor cost breakdown ──
    sensors_with_cost = sorted(
        s for s in set(dec_sensor_costs) | set(off_sensor_costs)
        if dec_sensor_costs.get(s, 0) > 0.001 or off_sensor_costs.get(s, 0) > 0.001
    )
    if sensors_with_cost:
        col_w2 = max(len(s) for s in sensors_with_cost) + 2
        print(f"\n  {'Sensor':<{col_w2}} {'Decision':>10} {'All-off':>10} {'Saving':>10}")
        print(f"  {'-' * (col_w2 + 32)}")
        for s in sensors_with_cost:
            dc = dec_sensor_costs.get(s, 0)
            oc = off_sensor_costs.get(s, 0)
            saving = oc - dc
            print(f"  {s:<{col_w2}} {dc:>10.3f} {oc:>10.3f} {saving:>+10.3f}")

    # ── Predicted temperatures (decision vs all-off) ──
    col_w = max(len(lbl) for lbl in PREDICTION_LABELS) + 2
    header = f"  {'Sensor':<{col_w}}" + "".join(f"{'dec ' + h:>9}{'off ' + h:>9}" for h in horizons)
    print("\n  Predicted temperatures (decision vs all-off):")
    print(header)
    print(f"  {'-' * (col_w + 18 * len(horizons))}")
    for label in PREDICTION_LABELS:
        dec_vals = decision.predictions.get(label, {})
        row = f"  {label:<{col_w}}"
        has_any = False
        for h_step, h_label in zip(CONTROL_HORIZONS, horizons, strict=True):
            dec_t = dec_vals.get(h_label)
            off_key = f"{label}_temp_t+{h_step}"
            off_t = off_preds.get(off_key)
            if dec_t is not None:
                has_any = True
                row += f"{dec_t:>8.1f}°"
            else:
                row += f"{'--':>9}"
            if off_t is not None:
                row += f"{off_t:>8.1f}°"
            else:
                row += f"{'--':>9}"
        if has_any:
            print(row)

    # ── Window advisories (physics-based) ──
    from weatherstat.advisory import evaluate_window_advisories, process_advisories

    original_schedules = default_comfort_schedules()
    adv_state = _HS(
        current_temps=current_temps,
        outdoor_temp=outdoor,
        forecast_temps=forecast_temp_list,
        window_states=window_states_dict,
        hour_of_day=fractional_hour,
        recent_history=recent_hist,
        solar_fractions=solar_fractions,
    )
    window_advisories = evaluate_window_advisories(
        adv_state, winning_scenario, sim_params, original_schedules, base_hour,
    )
    process_advisories(
        window_advisories,
        live=live,
        notification_target=_CFG.notification_target,
        current_hour=base_hour,
    )

    # ── Infrastructure safety checks ──
    from weatherstat.safety import check_device_health, check_thermostat_modes, process_safety_alerts

    safety_alerts = check_thermostat_modes(latest, decision) + check_device_health()
    process_safety_alerts(
        safety_alerts,
        live=live,
        notification_target=_CFG.notification_target,
    )

    # Prediction sanity checks
    sane = check_prediction_sanity(decision, current_temps)

    # Write command JSON
    cmd_path = write_command_json(decision)
    print(f"\n  Command JSON: {cmd_path}")

    # Log decision for outcome tracking
    from weatherstat.decision_log import log_decision

    log_decision(decision, current_temps, latest, schedules, base_hour, live)
    print("  Decision logged")

    if live:
        if not sane:
            print("  SKIPPED: prediction sanity check failed")
            return decision
        # Save state to prevent rapid cycling
        save_control_state(decision, prev_state)
        print("  Mode: LIVE — command written for executor")
    else:
        print("  Mode: DRY-RUN — command written but not executed")

    return decision


def run_control_loop(live: bool = False) -> None:
    """Run the control loop indefinitely at 15-minute intervals."""
    mode_str = "LIVE" if live else "DRY-RUN"
    print(f"[control] Starting control loop ({mode_str}, interval: {LOOP_INTERVAL_SECONDS}s)")
    print("  Press Ctrl+C to stop\n")

    while True:
        try:
            run_control_cycle(live=live)
        except Exception as e:
            print(f"[control] Error in control cycle: {e}", file=sys.stderr)
            traceback.print_exc()
        print(f"\n[control] Next cycle in {LOOP_INTERVAL_SECONDS // 60} minutes...\n")
        time.sleep(LOOP_INTERVAL_SECONDS)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Weatherstat control policy")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute commands (default: dry-run only)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously at 15-minute intervals",
    )
    args = parser.parse_args()

    if args.loop:
        run_control_loop(live=args.live)
    else:
        run_control_cycle(live=args.live)


if __name__ == "__main__":
    main()
