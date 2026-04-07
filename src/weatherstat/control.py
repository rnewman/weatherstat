"""Control policy — comfort-optimizing HVAC selection.

Receding-horizon controller: "what HVAC settings maximize comfort over the next 6 hours?"
Re-evaluated every control_interval (default 5 min). Physics-based trajectory sweep using forward simulation.

Config-driven effector list: thermostats (trajectory on/off with delay x duration grid),
blowers (binary modes), mini-splits (regulating with target temperatures). Adding a device
is a YAML edit — no code changes needed.

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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from weatherstat.config import (
    CONTROL_INTERVAL,
    CONTROL_STATE_FILE,
    EFFECTOR_MAP,
    EFFECTORS,
    PREDICTION_SENSORS,
    PREDICTIONS_DIR,
    SENSOR_LABELS,
    UNIT_SYMBOL,
    abs_temp,
    delta_temp,
)
from weatherstat.extract import fetch_recent_history
from weatherstat.types import (
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlDecision,
    ControlState,
    EffectorDecision,
    RoomComfort,
    Scenario,
)
from weatherstat.yaml_config import ComfortProfile, ConstraintSchedule, MrtCorrectionConfig, load_config

_CFG = load_config()

# ── Constants ──────────────────────────────────────────────────────────────

# Cautious setpoint offset: when the control loop decides "heat on", we set the
# thermostat to current_temp + CAUTIOUS_OFFSET. If the loop is interrupted, the
# house drifts gently instead of running away to extreme temperatures.
CAUTIOUS_OFFSET = _CFG.cautious_offset if _CFG.cautious_offset is not None else delta_temp(2)

# Absolute safety bounds (in case of stale current_temp or other weirdness)
ABSOLUTE_MIN = _CFG.setpoint_min if _CFG.setpoint_min is not None else abs_temp(62)
ABSOLUTE_MAX = _CFG.setpoint_max if _CFG.setpoint_max is not None else abs_temp(78)

# Minimum hold time before changing setpoints (seconds).
# Must be less than LOOP_INTERVAL_SECONDS so every cycle produces a fresh decision.
MIN_HOLD_SECONDS = 3 * 60  # 3 minutes (shorter than default 5-min control interval)

# Maximum data staleness before refusing to execute (seconds)
MAX_STALE_SECONDS = 15 * 60  # 15 minutes

# Maximum predicted 1h temperature change before logging warning
MAX_1H_CHANGE = _CFG.max_1h_change if _CFG.max_1h_change is not None else delta_temp(5.0)

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
# scenarios. Equivalent to requiring ~1° of genuine comfort improvement at
# one horizon before turning on heating.
MIN_IMPROVEMENT = _CFG.min_improvement if _CFG.min_improvement is not None else delta_temp(1.0)

# Cold-room override: force zone heating when a room's current temperature is
# this far below its comfort min. Compensates for undertrained models that
# predict rooms warming without HVAC (because training data is all HVAC-on).
COLD_ROOM_OVERRIDE = _CFG.cold_room_override if _CFG.cold_room_override is not None else delta_temp(1.0)

# Control loop interval (from config, default 5 min)
LOOP_INTERVAL_SECONDS = CONTROL_INTERVAL

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
    for constraint in _CFG.constraints:
        schedule_entries = tuple(
            ComfortScheduleEntry(
                e.start_hour,
                e.end_hour,
                RoomComfort(
                    constraint.label,
                    e.preferred_lo,
                    e.preferred_hi,
                    e.min_temp,
                    e.max_temp,
                    e.cold_penalty,
                    e.hot_penalty,
                ),
            )
            for e in constraint.entries
        )
        schedules.append(ComfortSchedule(sensor=constraint.sensor, label=constraint.label, entries=schedule_entries))
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
        wname for wname, is_open in window_states.items() if is_open and wname in constraint_labels
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
                    e.comfort.preferred_lo,  # preferred unchanged — window doesn't change ideal
                    e.comfort.preferred_hi,
                    e.comfort.min_temp + min_offset,
                    e.comfort.max_temp + max_offset,
                    e.comfort.cold_penalty,
                    e.comfort.hot_penalty,
                ),
            )
            for e in schedule.entries
        )
        adjusted.append(ComfortSchedule(sensor=schedule.sensor, label=schedule.label, entries=new_entries))
    return adjusted


def fetch_active_comfort_profile() -> ComfortProfile | None:
    """Fetch the active comfort profile from Home Assistant.

    Reads the state of the configured comfort_entity (input_select) and
    looks up the corresponding profile from config. Returns None if
    unconfigured, unreachable, or the entity state doesn't match a profile.
    """
    if _CFG.comfort_entity is None:
        return None

    import requests

    from weatherstat.config import HA_TOKEN, HA_URL

    try:
        resp = requests.get(
            f"{HA_URL}/api/states/{_CFG.comfort_entity}",
            headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        state = resp.json().get("state", "")
        return _CFG.comfort_profiles.get(state)
    except Exception:
        return None


def apply_comfort_profile(
    schedules: list[ComfortSchedule],
    profile: ComfortProfile | None,
) -> list[ComfortSchedule]:
    """Apply comfort profile offsets to all schedules.

    If profile is None or has all-zero offsets, returns schedules unchanged.
    """
    if profile is None:
        return schedules
    no_offsets = profile.preferred_offset == 0.0 and profile.min_offset == 0.0 and profile.max_offset == 0.0
    if no_offsets and profile.penalty_scale == 1.0 and profile.preferred_widen == 0.0:
        return schedules

    ps = profile.penalty_scale
    hw = profile.preferred_widen / 2.0  # half-width of band expansion
    adjusted: list[ComfortSchedule] = []
    for schedule in schedules:
        new_entries = tuple(
            ComfortScheduleEntry(
                e.start_hour,
                e.end_hour,
                RoomComfort(
                    e.comfort.label,
                    e.comfort.preferred_lo + profile.preferred_offset - hw,
                    e.comfort.preferred_hi + profile.preferred_offset + hw,
                    e.comfort.min_temp + profile.min_offset,
                    e.comfort.max_temp + profile.max_offset,
                    e.comfort.cold_penalty * ps,
                    e.comfort.hot_penalty * ps,
                ),
            )
            for e in schedule.entries
        )
        adjusted.append(ComfortSchedule(sensor=schedule.sensor, label=schedule.label, entries=new_entries))
    return adjusted


def apply_mrt_correction(
    schedules: list[ComfortSchedule],
    outdoor_temp: float,
    mrt_config: MrtCorrectionConfig | None,
    mrt_weights: dict[str, float] | None = None,
    solar_elevation_gains: dict[str, float] | None = None,
    current_solar_elev: float = 0.0,
    current_solar_fraction: float = 0.0,
) -> tuple[list[ComfortSchedule], float, dict[str, float]]:
    """Adjust comfort targets for mean radiant temperature effects.

    Cold exterior surfaces (walls, windows) lower operative temperature below
    the air temperature reading. This shifts comfort targets up when it's cold
    outside to compensate, and down when warm walls make the air feel warmer.

    Sun streaming through windows heats interior surfaces, raising MRT above
    what outdoor temp alone suggests. Per sensor, the effective outdoor temp
    is increased by the current solar forcing × solar_response, reducing the
    cold-wall correction on sunny days while preserving it on cloudy days.

    Per-sensor mrt_weights (from YAML) still apply as a multiplier on the
    final offset for manual adjustments (e.g., rooms with unusual window area).

    Args:
        schedules: Comfort schedules for all constrained sensors.
        outdoor_temp: Current outdoor temperature (in configured unit).
        mrt_config: Correction parameters, or None to skip.
        mrt_weights: Per-sensor column → weight multiplier (default 1.0).
        solar_elevation_gains: Per-sensor β_solar from sysid.
        current_solar_elev: sin⁺(elevation) right now (0 at night, ~0.7 at noon).
        current_solar_fraction: Weather condition fraction (1.0=sunny, 0.15=cloudy).

    Returns:
        (adjusted_schedules, base_offset, per_sensor_offsets). Base offset is
        from raw outdoor temp (before per-sensor solar adjustment).
        per_sensor_offsets maps sensor column → applied offset (after solar + weight).
    """
    if mrt_config is None:
        return schedules, 0.0, {}

    # Base offset from outdoor temp only (reported for logging)
    raw_offset = mrt_config.alpha * (mrt_config.reference_temp - outdoor_temp)
    base_offset = max(-mrt_config.max_offset, min(mrt_config.max_offset, raw_offset))

    # Fast path: no correction needed when at reference temp and no solar forcing
    if abs(base_offset) < 0.05 and current_solar_elev * current_solar_fraction < 0.001:
        return schedules, 0.0, {}

    adjusted: list[ComfortSchedule] = []
    per_sensor_offsets: dict[str, float] = {}
    for schedule in schedules:
        # Per-sensor solar wall warming raises effective outdoor temp
        solar_gain = (solar_elevation_gains or {}).get(schedule.sensor, 0.0)
        solar_wall_warming = solar_gain * current_solar_elev * current_solar_fraction * mrt_config.solar_response
        effective_outdoor = outdoor_temp + solar_wall_warming

        raw_sensor_offset = mrt_config.alpha * (mrt_config.reference_temp - effective_outdoor)
        sensor_offset = max(-mrt_config.max_offset, min(mrt_config.max_offset, raw_sensor_offset))

        # Manual weight from YAML (for non-solar adjustments like window area)
        weight = (mrt_weights or {}).get(schedule.sensor, 1.0)
        sensor_offset *= weight

        per_sensor_offsets[schedule.sensor] = sensor_offset

        if abs(sensor_offset) < 0.05:
            adjusted.append(schedule)
            continue

        new_entries = tuple(
            ComfortScheduleEntry(
                e.start_hour,
                e.end_hour,
                RoomComfort(
                    e.comfort.label,
                    e.comfort.preferred_lo + sensor_offset,
                    e.comfort.preferred_hi + sensor_offset,
                    e.comfort.min_temp + sensor_offset,
                    e.comfort.max_temp + sensor_offset,
                    e.comfort.cold_penalty,
                    e.comfort.hot_penalty,
                ),
            )
            for e in schedule.entries
        )
        adjusted.append(ComfortSchedule(sensor=schedule.sensor, label=schedule.label, entries=new_entries))
    return adjusted, base_offset, per_sensor_offsets


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
        for h in CONTROL_HORIZONS:
            weight = HORIZON_WEIGHTS.get(h, 0.5)
            hours_ahead = horizon_hours.get(h, h // 12)
            future_hour = (base_hour + hours_ahead) % 24

            comfort = schedule.comfort_at(future_hour)
            if comfort is None:
                continue

            pred_temp = predictions.get(f"{schedule.sensor}_t+{h}")
            if pred_temp is None:
                continue

            # Continuous cost: quadratic deviation from preferred band.
            # Zero cost within [preferred_lo, preferred_hi] (dead band).
            if pred_temp < comfort.preferred_lo:
                cost += (comfort.preferred_lo - pred_temp) ** 2 * comfort.cold_penalty * weight
            elif pred_temp > comfort.preferred_hi:
                cost += (pred_temp - comfort.preferred_hi) ** 2 * comfort.hot_penalty * weight

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
        sensor_cost = 0.0
        for h in CONTROL_HORIZONS:
            weight = HORIZON_WEIGHTS.get(h, 0.5)
            hours_ahead = horizon_hours.get(h, h // 12)
            future_hour = (base_hour + hours_ahead) % 24

            comfort = schedule.comfort_at(future_hour)
            if comfort is None:
                continue

            pred_temp = predictions.get(f"{schedule.sensor}_t+{h}")
            if pred_temp is None:
                continue

            if pred_temp < comfort.preferred_lo:
                sensor_cost += (comfort.preferred_lo - pred_temp) ** 2 * comfort.cold_penalty * weight
            elif pred_temp > comfort.preferred_hi:
                sensor_cost += (pred_temp - comfort.preferred_hi) ** 2 * comfort.hot_penalty * weight

            if pred_temp < comfort.min_temp:
                delta = comfort.min_temp - pred_temp
                sensor_cost += delta**2 * comfort.cold_penalty * _HARD_RAIL_MULTIPLIER * weight
            elif pred_temp > comfort.max_temp:
                delta = pred_temp - comfort.max_temp
                sensor_cost += delta**2 * comfort.hot_penalty * _HARD_RAIL_MULTIPLIER * weight
        costs[schedule.label] = sensor_cost
    return costs


def compute_energy_cost(
    scenario: Scenario,
    current_temps: dict[str, float] | None = None,
) -> float:
    """Per-effector energy penalty from config.

    Used as tiebreaker when comfort cost is equal — prefer less energy usage.
    Trajectory cost is proportional to heating duration within the horizon.
    Regulating cost is proportional to expected activity (target vs room temp).
    Binary cost is a per-mode lookup.
    """
    cost = 0.0
    max_h = max(CONTROL_HORIZONS)
    temps = current_temps or {}

    for ed in scenario.effectors.values():
        eff_cfg = EFFECTOR_MAP.get(ed.name)
        if eff_cfg is None or ed.mode == "off":
            continue

        ec = eff_cfg.energy_cost
        if eff_cfg.control_type == "trajectory":
            dur = ed.duration_steps if ed.duration_steps is not None else (max_h - ed.delay_steps)
            assert isinstance(ec, int | float)
            cost += ec * dur / max_h

        elif eff_cfg.control_type == "regulating":
            p_band = eff_cfg.proportional_band
            label = ed.name.removeprefix("mini_split_")
            room_temp = temps.get(label)
            if room_temp is not None and ed.target is not None:
                delta = max(0.0, ed.target - room_temp) if ed.mode == "heat" else max(0.0, room_temp - ed.target)
                avg_activity = min(1.0, delta / p_band)
            else:
                avg_activity = 0.5
            assert isinstance(ec, int | float)
            cost += ec * avg_activity

        elif eff_cfg.control_type == "binary":
            assert isinstance(ec, dict)
            cost += ec.get(ed.mode, 0.0)

    return cost


def _emergency_effector(
    gains: dict[tuple[str, str], tuple[float, float]],
    constraints: list[ConstraintSchedule],
) -> dict[str, str]:
    """For each constrained sensor, find the trajectory effector with the highest gain.

    Used by the cold-sensor safety override to decide which effector to force on.
    Returns sensor_col -> effector_name (e.g., {"bedroom_temp": "thermostat_upstairs"}).
    """
    trajectory_effectors = {e.name for e in EFFECTORS if e.control_type == "trajectory"}
    constrained_sensors = {c.sensor for c in constraints}

    best: dict[str, tuple[str, float]] = {}  # sensor_col -> (effector_name, gain)
    for (effector, sensor_col), (gain, _lag) in gains.items():
        if effector not in trajectory_effectors or gain <= 0:
            continue
        if sensor_col not in constrained_sensors:
            continue
        prev = best.get(sensor_col)
        if prev is None or gain > prev[1]:
            best[sensor_col] = (effector, gain)

    return {sensor_col: eff_name for sensor_col, (eff_name, _) in best.items()}


def _comfort_max(sensor: str, schedules: list[ComfortSchedule], hour: int) -> float:
    """Get the comfort max for a sensor at the given hour."""
    for s in schedules:
        if s.sensor == sensor:
            c = s.comfort_at(hour)
            if c is not None:
                return c.max_temp
    return ABSOLUTE_MAX


def _comfort_min(sensor: str, schedules: list[ComfortSchedule], hour: int) -> float:
    """Get the comfort min for a sensor at the given hour."""
    for s in schedules:
        if s.sensor == sensor:
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


def _dedup_targets(targets: list[float], tol: float = 0.5) -> list[float]:
    """Deduplicate sorted targets within tolerance, keeping one per cluster."""
    result: list[float] = []
    for t in sorted(set(targets)):
        if not result or abs(t - result[-1]) > tol:
            result.append(t)
    return result


def _regulating_sweep_options(
    eff: object,
    schedules: list[ComfortSchedule],
    base_hour: int,
    gains: dict[tuple[str, str], tuple[float, float]] | None = None,
    prev_state: ControlState | None = None,
    current_temps: dict[str, float] | None = None,
) -> list[EffectorDecision]:
    """Generate sweep options for a regulating effector from all affected sensors.

    Instead of deriving mode/target from one sensor, we:
    1. Find all constrained sensors this effector has meaningful sysid gain on.
    2. Collect their preferred temps (pref_lo for heat targets, pref_hi for cool).
    3. Generate off + heat@each_pref_lo + cool@each_pref_hi.
    4. Idle suppression prunes options where the effector would be inactive
       (room already past target by more than the proportional band).
    5. Mode hold window locks to previous mode during quiet hours.

    The trajectory sweep scores every surviving option across ALL sensors,
    so the optimizer picks whichever mode/target minimizes total comfort cost.
    """
    from weatherstat.config import EffectorConfig

    assert isinstance(eff, EffectorConfig)
    options: list[EffectorDecision] = [EffectorDecision(eff.name)]
    temps = current_temps or {}

    # ── Find constrained sensors this effector affects ──
    schedule_sensors = {s.sensor for s in schedules}
    affected_sensors: list[str] = []
    if gains:
        for (eff_name, sensor_name), (gain, _lag) in gains.items():
            if eff_name == eff.name and gain != 0 and sensor_name in schedule_sensors:
                affected_sensors.append(sensor_name)
    # Fallback: naming convention (backward compat for tests without gains)
    if not affected_sensors:
        fallback = eff.name.removeprefix("mini_split_") + "_temp"
        if fallback in schedule_sensors:
            affected_sensors = [fallback]

    # ── Collect preferred temps from all affected sensors ──
    heat_targets: list[float] = []  # pref_lo values
    cool_targets: list[float] = []  # pref_hi values
    for sensor_col in affected_sensors:
        for sched in schedules:
            if sched.sensor == sensor_col:
                comfort = sched.comfort_at(base_hour)
                if comfort is not None:
                    heat_targets.append(comfort.preferred_lo)
                    cool_targets.append(comfort.preferred_hi)
                break

    if not heat_targets and not cool_targets:
        return options

    heat_targets = _dedup_targets(heat_targets)
    cool_targets = _dedup_targets(cool_targets)

    # ── Reference temp for idle suppression: highest-gain sensor with data ──
    ref_temp: float | None = None
    if gains:
        best_gain = 0.0
        for (eff_name, sensor_name), (gain, _lag) in gains.items():
            if eff_name == eff.name and abs(gain) > best_gain and sensor_name in temps:
                best_gain = abs(gain)
                ref_temp = temps[sensor_name]
    if ref_temp is None:
        # Fallback: naming convention sensor
        fallback = eff.name.removeprefix("mini_split_") + "_temp"
        ref_temp = temps.get(fallback)

    p_band = eff.proportional_band

    # ── Generate options per supported mode ──
    for mode in eff.supported_modes:
        targets = heat_targets if mode == "heat" else cool_targets if mode == "cool" else []
        for target in targets:
            # Idle suppression: skip if effector would be idle
            if ref_temp is not None:
                if mode == "heat" and ref_temp > target + p_band:
                    continue
                if mode == "cool" and ref_temp < target - p_band:
                    continue
            options.append(EffectorDecision(eff.name, mode=mode, target=round(target, 1)))

    # Mode hold: lock to current mode during quiet-hours window (no compressor starts/stops)
    if prev_state and eff.mode_hold_window and _in_hold_window(base_hour, eff.mode_hold_window):
        prev_mode = prev_state.modes.get(eff.name, "off")
        options = [o for o in options if o.mode == prev_mode]
        if not options:
            prev_target = prev_state.setpoints.get(eff.name, 0.0)
            options = [EffectorDecision(eff.name, mode=prev_mode, target=prev_target)]

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
    ineligible_effectors: set[str] | None = None,
    gains: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> list[Scenario]:
    """Generate trajectory scenarios for physics sweep.

    Slow effectors (trajectory) get a delay x duration grid.
    Fast effectors (binary) use constant modes.
    Regulating effectors use comfort-derived target grid (off + preferred).
    Dependent effectors (e.g., blowers) only active when parent is active.
    """
    from itertools import product

    max_horizon = max(CONTROL_HORIZONS)
    _ineligible = ineligible_effectors or set()

    # Build per-effector option lists
    per_effector_options: dict[str, list[EffectorDecision]] = {}

    for eff in EFFECTORS:
        if eff.depends_on:
            # Dependent effectors handled after their parents
            continue

        if eff.control_type == "trajectory":
            if eff.name in _ineligible:
                per_effector_options[eff.name] = [EffectorDecision(eff.name)]
            else:
                options: list[EffectorDecision] = [EffectorDecision(eff.name)]
                for delay in TRAJECTORY_DELAYS:
                    if delay >= max_horizon:
                        continue
                    for duration in TRAJECTORY_DURATIONS:
                        effective_duration = min(duration, max_horizon - delay)
                        options.append(
                            EffectorDecision(
                                eff.name,
                                mode="heating",
                                delay_steps=delay,
                                duration_steps=effective_duration,
                            )
                        )
                # Deduplicate (capping creates duplicates)
                per_effector_options[eff.name] = list(dict.fromkeys(options))

        elif eff.control_type == "regulating":
            if schedules is not None:
                per_effector_options[eff.name] = _regulating_sweep_options(
                    eff,
                    schedules,
                    base_hour,
                    gains,
                    prev_state,
                    current_temps,
                )
            else:
                per_effector_options[eff.name] = [
                    EffectorDecision(eff.name),
                    EffectorDecision(eff.name, mode="heat", target=abs_temp(70.0)),
                ]

        elif eff.control_type == "binary":
            per_effector_options[eff.name] = [EffectorDecision(eff.name, mode=m) for m in eff.supported_modes]

    # Cartesian product of independent effectors
    independent_names = list(per_effector_options.keys())
    independent_options = [per_effector_options[n] for n in independent_names]

    # Dependent effectors: constrained by parent state
    dependent_effectors = [e for e in EFFECTORS if e.depends_on]

    scenarios: list[Scenario] = []
    for combo in product(*independent_options):
        base_decisions = {ed.name: ed for ed in combo}

        # Build dependent options: only sweep when ALL parents are immediately active
        dep_option_lists: list[list[EffectorDecision]] = []
        dep_names: list[str] = []
        for dep in dependent_effectors:
            all_parents_active = all(
                (p := base_decisions.get(pname)) is not None and p.mode != "off" and p.delay_steps == 0
                for pname in dep.depends_on
            )
            if all_parents_active:
                dep_option_lists.append([EffectorDecision(dep.name, mode=m) for m in dep.supported_modes])
            else:
                dep_option_lists.append([EffectorDecision(dep.name)])
            dep_names.append(dep.name)

        if dep_option_lists:
            for dep_combo in product(*dep_option_lists):
                effectors = dict(base_decisions)
                for dep_ed in dep_combo:
                    effectors[dep_ed.name] = dep_ed
                scenarios.append(Scenario(effectors))
        else:
            scenarios.append(Scenario(dict(base_decisions)))

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
    schedules: list[ComfortSchedule],
    base_hour: int,
    prev_state: ControlState | None = None,
    ineligible_effectors: set[str] | None = None,
    *,
    solar_fractions: list[float],
    solar_elevations: list[float],
) -> tuple[ControlDecision, Scenario, dict[str, str]]:
    """Sweep trajectory scenarios using physics simulator predictions.

    Trajectory effectors are evaluated over a delay x duration grid.
    Regulating effectors use comfort-derived targets.
    Binary effectors sweep supported modes.
    Ineligible effectors are fixed to off.

    Returns (decision, winning_scenario, blocked_dict) where blocked_dict
    maps effector name to reason for any physically blocked effectors.
    """
    from weatherstat.simulator import HouseState, SimParams, predict

    assert isinstance(sim_params, SimParams)

    emergency = _emergency_effector(sim_params.gains, _CFG.constraints)
    _ineligible = ineligible_effectors or set()
    scenarios = generate_trajectory_scenarios(
        schedules,
        base_hour,
        prev_state,
        current_temps,
        _ineligible,
        gains=sim_params.gains,
    )
    pre_count = len(scenarios)
    blocked_reasons: list[str] = []
    blocked_dict: dict[str, str] = {}  # effector_name -> reason (returned to caller)

    # ── Physical constraints: block trajectory effectors at/above comfort max ──
    blocked_effectors: set[str] = set()
    for eff in EFFECTORS:
        if eff.control_type != "trajectory" or not eff.temp_col:
            continue
        eff_temp = current_temps.get(eff.temp_col)
        if eff_temp is None:
            continue
        eff_max = _comfort_max(eff.temp_col, schedules, base_hour)
        if eff_temp >= eff_max:
            blocked_effectors.add(eff.name)
            reason = f"at/above max ({eff_temp:.1f}{UNIT_SYMBOL} >= {eff_max:.0f}{UNIT_SYMBOL})"
            blocked_dict[eff.name] = reason
            blocked_reasons.append(f"{eff.name} {reason}")

    if blocked_effectors:
        scenarios = [
            s
            for s in scenarios
            if all(s.effectors[name].mode == "off" for name in blocked_effectors if name in s.effectors)
        ]

    # Note: no "thermal direction" pruning — the trajectory sweep and cost
    # function decide whether heating is justified.  Pre-emptive heating for
    # slow effectors (hydronic slab: 45-75 min lag) requires evaluating
    # futures, not checking current temps against comfort min.

    if blocked_reasons:
        print(f"  Heating blocked: {'; '.join(blocked_reasons)}")
        print(f"  Reduced scenarios: {pre_count} → {len(scenarios)}")

    def _is_all_off(s: Scenario) -> bool:
        return all(ed.mode == "off" for ed in s.effectors.values())

    # ── Batch simulate all scenarios ──
    sweep_state = HouseState(
        current_temps=current_temps,
        outdoor_temp=outdoor_temp,
        forecast_temps=forecast_temps,
        window_states=window_states,
        hour_of_day=hour_of_day,
        recent_history=recent_history,
        solar_fractions=solar_fractions,
        solar_elevations=solar_elevations,
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
        cold_effectors: set[str] = set()
        cold_info: list[str] = []
        for schedule in schedules:
            temp = current_temps.get(schedule.sensor)
            if temp is None:
                continue
            comfort = schedule.comfort_at(base_hour)
            if comfort is None:
                continue
            if temp < comfort.min_temp - COLD_ROOM_OVERRIDE:
                eff_name = emergency.get(schedule.sensor)
                if eff_name:
                    cold_effectors.add(eff_name)
                    cold_info.append(f"{schedule.label} ({temp:.1f}{UNIT_SYMBOL} < {comfort.min_temp:.0f}{UNIT_SYMBOL})")

        # Remove blocked or ineligible effectors from cold override set
        cold_effectors -= blocked_effectors
        cold_effectors -= _ineligible

        if cold_effectors:
            constrained_best = -1
            constrained_cost = float("inf")
            for i, scenario in enumerate(scenarios):
                # Cold-room override requires immediate heating (delay=0) for cold effectors
                if any(
                    not (scenario.effectors[name].mode != "off" and scenario.effectors[name].delay_steps == 0)
                    for name in cold_effectors
                    if name in scenario.effectors
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

    # Build effector decisions and command targets
    effector_decisions: list[EffectorDecision] = list(scenario.effectors.values())
    command_targets: dict[str, float] = {}
    trajectory_info: dict[str, dict[str, int | None]] = {}

    for ed in effector_decisions:
        eff_cfg = EFFECTOR_MAP.get(ed.name)
        if eff_cfg is None:
            continue

        if eff_cfg.control_type == "trajectory":
            # Trajectory effector: compute cautious setpoint
            temp_col = eff_cfg.temp_col
            eff_temp = current_temps.get(temp_col, abs_temp(70.0))
            heating_now = ed.mode != "off" and ed.delay_steps == 0
            setpoint = _cautious_setpoint(
                eff_temp,
                heating_now,
                comfort_min=_comfort_min(temp_col, schedules, base_hour),
            )
            command_targets[ed.name] = setpoint
            if ed.mode != "off":
                trajectory_info[ed.name] = {
                    "delay_steps": ed.delay_steps,
                    "duration_steps": ed.duration_steps,
                }

        elif eff_cfg.control_type == "regulating" and ed.target is not None:
            command_targets[ed.name] = ed.target

    sensor_preds: dict[str, dict[str, float]] = {}
    for sensor_col in PREDICTION_SENSORS:
        rpred: dict[str, float] = {}
        for h in CONTROL_HORIZONS:
            key = f"{sensor_col}_t+{h}"
            val = predictions.get(key)
            if val is not None:
                rpred[HORIZON_LABELS[h]] = round(val, 2)
        if rpred:
            sensor_preds[sensor_col] = rpred

    decision = ControlDecision(
        timestamp=datetime.now(UTC).isoformat(),
        effectors=tuple(effector_decisions),
        command_targets=command_targets,
        total_cost=round(total, 4),
        comfort_cost=round(comfort, 4),
        energy_cost=round(energy, 4),
        predictions=sensor_preds,
        trajectory_info=trajectory_info,
    )
    return decision, scenario, blocked_dict


# ── State persistence ─────────────────────────────────────────────────────


def load_control_state() -> ControlState | None:
    """Load persisted control state, or None if not found."""
    if not CONTROL_STATE_FILE.exists():
        return None
    try:
        data = json.loads(CONTROL_STATE_FILE.read_text())
        return ControlState(
            last_decision_time=data["last_decision_time"],
            setpoints=data.get("setpoints", {}),
            modes=data.get("modes", {}),
            mode_times=data.get("mode_times", {}),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def save_control_state(decision: ControlDecision, prev_state: ControlState | None = None) -> None:
    """Persist control state to prevent rapid cycling.

    Tracks mode change timestamps: only updated when mode actually changes.
    """
    now_iso = decision.timestamp
    new_modes = {ed.name: ed.mode for ed in decision.effectors}
    new_setpoints = dict(decision.command_targets)
    prev_modes = prev_state.modes if prev_state else {}
    prev_mode_times = dict(prev_state.mode_times) if prev_state else {}

    # Update mode-change timestamps only when mode actually changes
    for name, mode in new_modes.items():
        if mode != prev_modes.get(name, ""):
            prev_mode_times[name] = now_iso

    state: dict[str, object] = {
        "last_decision_time": now_iso,
        "setpoints": new_setpoints,
        "modes": new_modes,
        "mode_times": prev_mode_times,
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
    """Return True if 1h predictions look reasonable for all sensors."""
    safe = True
    for sensor_col, preds in decision.predictions.items():
        pred_1h = preds.get("1h")
        current = current_temps.get(sensor_col)
        if pred_1h is None or current is None:
            continue
        if abs(pred_1h - current) > MAX_1H_CHANGE:
            label = SENSOR_LABELS.get(sensor_col, sensor_col)
            print(f"  WARNING: {label} 1h pred {pred_1h:.1f}{UNIT_SYMBOL} >{MAX_1H_CHANGE}{UNIT_SYMBOL} from current {current:.1f}{UNIT_SYMBOL}")
            safe = False
    return safe


# ── Effector eligibility ──────────────────────────────────────────────────


def _fetch_entity_state(entity_id: str) -> str | None:
    """Fetch a single entity's state from HA. Returns None on failure."""
    from weatherstat.config import HA_TOKEN, HA_URL

    try:
        resp = requests.get(
            f"{HA_URL}/api/states/{entity_id}",
            headers={"Authorization": f"Bearer {HA_TOKEN}"},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("state")
    except Exception:
        pass
    return None


def check_effector_eligibility() -> dict[str, str]:
    """Check which effectors are ineligible for control.

    A manually-controlled effector is eligible when:
    1. Its hvac_mode is not "off" — it will respond to setpoint changes.
       (The snapshot stores hvac_action, not hvac_mode, and both report "idle" when
       off — so we fetch the entity state (= hvac_mode) live from HA.)
    2. Its state_device is functional — the delivery system (e.g., boiler) is online.

    Returns effector_name -> reason for each ineligible effector.
    """
    ineligible: dict[str, str] = {}

    for eff in EFFECTORS:
        if eff.mode_control != "manual":
            continue

        # 1. Check hvac_mode (entity state) — "off" means setpoint changes are ignored
        hvac_mode = _fetch_entity_state(eff.entity_id)
        if hvac_mode == "off":
            ineligible[eff.name] = "hvac_mode is off"
            continue

        # 2. Check state_device is functional (gate is open)
        if eff.state_device and eff.state_device in _CFG.state_sensors:
            entity_id = _CFG.state_sensors[eff.state_device].entity_id
            state = _fetch_entity_state(entity_id)
            if state in ("unavailable", "unknown", None):
                ineligible[eff.name] = f"{eff.state_device} is {state or 'unreachable'}"

    return ineligible


# ── Command JSON output ───────────────────────────────────────────────────


def write_command_json(
    decision: ControlDecision,
    opportunities: list | None = None,
    ineligible_effectors: set[str] | None = None,
) -> Path:
    """Write executor-compatible command JSON.

    Uses camelCase keys matching the TS Prediction interface.
    For ineligible effectors, omits their commands (no-op for executor).
    """
    _ineligible = ineligible_effectors or set()
    command: dict[str, object] = {
        "timestamp": decision.timestamp,
        "confidence": 1.0 - min(decision.total_cost / 10.0, 0.9),
    }

    for ed in decision.effectors:
        eff_cfg = EFFECTOR_MAP.get(ed.name)
        if eff_cfg is None or ed.name in _ineligible:
            continue
        # Write each command key for this effector
        for purpose, camel_key in eff_cfg.command_keys.items():
            if purpose == "target":
                target_val = decision.command_targets.get(ed.name)
                if target_val is not None:
                    command[camel_key] = target_val
            elif purpose == "mode":
                command[camel_key] = ed.mode

    # Active window opportunities (informational)
    if opportunities:
        command["opportunities"] = [
            {
                "window": o.window,
                "action": o.action,
                "benefit": o.total_benefit,
                "message": o.message,
            }
            for o in opportunities
        ]

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
    latest = df_raw.iloc[-1]
    # Outdoor temp: prefer configured sensor, fall back to weather entity
    _outdoor_col = _CFG.outdoor_sensor
    out_temp = (latest.get(_outdoor_col) if _outdoor_col else None) or latest.get("met_outdoor_temp")
    now_str = df_raw["timestamp"].iloc[-1]

    # Build current temperature dict keyed by sensor column name
    current_temps: dict[str, float] = {}
    for sensor_col in PREDICTION_SENSORS:
        val = latest.get(sensor_col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            current_temps[sensor_col] = float(val)

    print(f"\n[control] Current state ({now_str}):")
    # Show trajectory effector temps and setpoints
    traj_sensors: set[str] = set()
    for eff in EFFECTORS:
        if eff.control_type == "trajectory" and eff.temp_col:
            traj_sensors.add(eff.temp_col)
            label = SENSOR_LABELS.get(eff.temp_col, eff.name)
            eff_temp = current_temps.get(eff.temp_col, abs_temp(70.0))
            eff_target = latest.get(f"{eff.name}_target", "?")
            print(f"  {label:<14} {eff_temp:.1f}{UNIT_SYMBOL} (setpoint: {eff_target}{UNIT_SYMBOL})")
    if out_temp is not None and not (isinstance(out_temp, float) and np.isnan(out_temp)):
        print(f"  Outdoor:     {float(out_temp):.1f}{UNIT_SYMBOL} (sensor)")
    window_cols = _CFG.window_display_map
    open_windows = [wlabel for col, wlabel in window_cols.items() if bool(latest.get(col, False))]
    if open_windows:
        from weatherstat.yaml_config import window_display
        labels = [f"{label} {kind}" for label, kind in (window_display(w) for w in open_windows)]
        print(f"  Windows:     {', '.join(labels)} open")
    else:
        print("  Windows:     all closed")
    # Show other prediction sensors (not trajectory effectors)
    other_sensors = [s for s in PREDICTION_SENSORS if s not in traj_sensors]
    lw = max((len(SENSOR_LABELS.get(s, s)) for s in other_sensors), default=14) + 2
    for sensor_col in other_sensors:
        t = current_temps.get(sensor_col)
        if t is not None:
            print(f"  {SENSOR_LABELS.get(sensor_col, sensor_col):<{lw}} {t:.1f}{UNIT_SYMBOL}")
    # Show current effector states
    for eff in EFFECTORS:
        if eff.control_type == "trajectory":
            continue  # already shown above
        if eff.control_type == "regulating":
            mode = str(latest.get(f"{eff.name}_mode", "?"))
            target = latest.get(f"{eff.name}_target", "?")
            print(f"  {eff.name:<20} {mode} @ {target}{UNIT_SYMBOL}")
        elif eff.control_type == "binary":
            mode = str(latest.get(f"{eff.name}_mode", "?"))
            print(f"  {eff.name:<20} {mode}")

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
                parts.append(f"{label}:{entry.temperature:.0f}{UNIT_SYMBOL}")
        if parts:
            print(f"  Forecast:    {', '.join(parts)}")
    else:
        print("  Forecast:    unavailable")

    # Current local hour for comfort schedule lookup
    from zoneinfo import ZoneInfo

    from weatherstat.config import TIMEZONE

    local_now = datetime.now(ZoneInfo(TIMEZONE))
    base_hour = local_now.hour

    # Current weather condition for solar fraction
    _current_cond = str(latest.get("weather_condition", "unknown")) if hasattr(latest, "get") else "unknown"
    from weatherstat.weather import condition_to_solar_fraction as _c2sf

    _sf = _c2sf(_current_cond)
    _is_night = base_hour < 7 or base_hour >= 19
    _sf_note = "night, no solar gain" if _is_night else f"solar fraction: {_sf:.0%}"
    print(f"  Weather:     {_current_cond} ({_sf_note})")

    # Comfort schedules: base → profile offsets → MRT correction → window adjustments
    schedules = default_comfort_schedules()
    active_profile = fetch_active_comfort_profile()
    schedules = apply_comfort_profile(schedules, active_profile)
    if active_profile is not None:
        parts = [f"{active_profile.name} profile"]
        offset_items = [
            ("pref", active_profile.preferred_offset),
            ("min", active_profile.min_offset),
            ("max", active_profile.max_offset),
        ]
        offsets = [f"{lbl} {v:+.0f}" for lbl, v in offset_items if v]
        if active_profile.preferred_widen:
            offsets.append(f"band ±{active_profile.preferred_widen / 2:.0f}")
        if active_profile.penalty_scale != 1.0:
            offsets.append(f"penalty ×{active_profile.penalty_scale:.1f}")
        if offsets:
            parts.append(f"({', '.join(offsets)})")
        print(f"  Comfort:     {' '.join(parts)}")
    # Load physics parameters (needed for derived MRT weights and sweep)
    from weatherstat.simulator import extract_recent_history, load_sim_params

    sim_params = load_sim_params()

    # MRT correction: adjust targets for cold/warm wall surface effects.
    # Dynamic: solar wall warming reduces the cold-wall correction on sunny days.
    # Manual mrt_weight from YAML still applies as a multiplier.
    from weatherstat.weather import condition_to_solar_fraction as _csf
    from weatherstat.weather import solar_sin_elevation as _sse

    _out_valid = out_temp is not None and not (isinstance(out_temp, float) and np.isnan(out_temp))
    _mrt_outdoor = float(out_temp) if _out_valid else abs_temp(50.0)
    _mrt_weights = {c.sensor: c.mrt_weight for c in _CFG.constraints if c.mrt_weight != 1.0}
    _mrt_cond = str(latest.get("weather_condition", "unknown")) if hasattr(latest, "get") else "unknown"
    _mrt_solar_frac = _csf(_mrt_cond)
    _mrt_solar_elev = _sse(_CFG.location.latitude, _CFG.location.longitude, datetime.now(UTC))
    schedules, mrt_offset, _mrt_per_sensor = apply_mrt_correction(
        schedules, _mrt_outdoor, _CFG.mrt_correction, _mrt_weights or None,
        solar_elevation_gains=sim_params.solar_elevation_gains,
        current_solar_elev=_mrt_solar_elev,
        current_solar_fraction=_mrt_solar_frac,
    )
    if abs(mrt_offset) >= 0.05:
        _solar_note = f", solar sin⁺={_mrt_solar_elev:.2f}×{_mrt_solar_frac:.0%}" if _mrt_solar_elev > 0 else ""
        print(f"  MRT:         {mrt_offset:+.1f}{UNIT_SYMBOL} base (outdoor {_mrt_outdoor:.0f}{UNIT_SYMBOL}{_solar_note})")
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
        from weatherstat.yaml_config import window_display
        open_labels = [f"{label} {kind}" for label, kind in (window_display(w) for w in sorted(labels_with_open))]
        print(f"  Comfort adjusted for open: {', '.join(open_labels)}")
    recent_hist = extract_recent_history(df_raw, sim_params)

    # Build forecast temp list, solar fractions, and solar elevations for simulator
    from weatherstat.weather import condition_to_solar_fraction, solar_sin_elevation

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
                forecast_temp_list.append(float(out_temp) if out_temp is not None else abs_temp(50.0))
                solar_fractions.append(solar_fractions[-1])

    fractional_hour = base_hour + local_now.minute / 60.0

    # Precompute sin+(elevation) at each 5-min step for the prediction horizon
    lat = _CFG.location.latitude
    lon = _CFG.location.longitude
    now_utc = datetime.now(UTC)
    max_steps = max(CONTROL_HORIZONS)
    solar_elevations: list[float] = [
        solar_sin_elevation(lat, lon, now_utc + timedelta(minutes=5 * step))
        for step in range(1, max_steps + 1)
    ]

    # Load previous state for mode hold enforcement
    prev_state = load_control_state()

    # Outdoor temp for simulator: prefer first forecast temp (met.no, accurate)
    # over the side sensor (solar-heated, unreliable during daytime).
    if forecast_temp_list:
        outdoor = forecast_temp_list[0]
    elif out_temp is not None and not (isinstance(out_temp, float) and np.isnan(out_temp)):
        outdoor = float(out_temp)
    else:
        outdoor = abs_temp(50.0)
    # ── Effector eligibility ──
    ineligible = check_effector_eligibility()
    ineligible_effectors = set(ineligible) if ineligible else None
    if ineligible:
        print("\n[control] Effector eligibility:")
        for eff_name, reason in ineligible.items():
            print(f"  {eff_name}: INELIGIBLE — {reason}")

    n_scenarios = len(
        generate_trajectory_scenarios(
            schedules,
            base_hour,
            prev_state,
            current_temps,
            ineligible_effectors,
            gains=sim_params.gains,
        )
    )
    print(f"\n[control] Sweeping {n_scenarios} trajectory scenarios...")
    t0 = time.monotonic()
    decision, winning_scenario, blocked_dict = sweep_scenarios_physics(
        current_temps,
        outdoor,
        forecast_temp_list,
        window_states_dict,
        sim_params,
        fractional_hour,
        recent_hist,
        schedules,
        base_hour,
        prev_state,
        ineligible_effectors,
        solar_fractions=solar_fractions,
        solar_elevations=solar_elevations,
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
        solar_elevations=solar_elevations,
    )

    # All-off baseline
    all_off = Scenario({e.name: EffectorDecision(e.name) for e in EFFECTORS})
    off_targets, off_matrix = _predict(sim_state, [all_off], sim_params, CONTROL_HORIZONS)
    off_preds = {t: float(off_matrix[0, j]) for j, t in enumerate(off_targets)}
    off_comfort = compute_comfort_cost(off_preds, schedules, base_hour)

    # Per-device counterfactuals: winning scenario with each active device removed.
    # This gives true per-device attribution (what does THIS device contribute?).
    counterfactuals: list[Scenario] = []
    cf_device_keys: list[str] = []

    for eff_name, ed in winning_scenario.effectors.items():
        if ed.mode != "off":
            cf_effectors = dict(winning_scenario.effectors)
            cf_effectors[eff_name] = EffectorDecision(eff_name)
            counterfactuals.append(Scenario(cf_effectors))
            cf_device_keys.append(eff_name)

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
                dec_flat[f"{label}_t+{h}"] = preds[h_label]
    dec_sensor_costs = compute_comfort_cost_by_sensor(dec_flat, schedules, base_hour)
    off_sensor_costs = compute_comfort_cost_by_sensor(off_preds, schedules, base_hour)

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
            return "  -> no significant effect"

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
            return f"  -> comfort {total_saving:+.2f} (diffuse effects across sensors)"

        parts = [
            f"  -> {best_sensor} at {best_h_label}: {best_dec_t:.1f} vs {best_cf_t:.1f} without",
            f" ({best_sensor} {best_sensor_saving:+.2f}",
        ]
        # Show total if it differs significantly from the top sensor
        if abs(total_saving - best_sensor_saving) > 0.5:
            parts.append(f", total {total_saving:+.2f}")
        parts.append(")")
        return "".join(parts)

    def _state_rationale(ed: EffectorDecision) -> str:
        """Build rationale from current state: why is this device on or off?"""
        eff_cfg = EFFECTOR_MAP.get(ed.name)
        if eff_cfg is None:
            return ""
        # Find the sensor this effector most affects
        sensor_col = ""
        if eff_cfg.control_type == "trajectory":
            sensor_col = eff_cfg.temp_col  # e.g., "thermostat_upstairs_temp"
        elif eff_cfg.control_type == "regulating":
            sensor_col = ed.name.removeprefix("mini_split_") + "_temp"
        elif eff_cfg.control_type == "binary":
            # Blowers affect the rooms they serve — use naming convention
            room = ed.name.removeprefix("blower_")
            sensor_col = f"{room}_temp"

        if not sensor_col:
            return ""

        temp = current_temps.get(sensor_col)
        if temp is None:
            return ""

        # Find comfort bounds for this sensor
        for sched in schedules:
            if sched.sensor == sensor_col:
                c = sched.comfort_at(base_hour)
                if c is None:
                    return ""
                label = sched.label
                if ed.mode == "off":
                    if c.preferred_lo <= temp <= c.preferred_hi:
                        return f"{label} {temp:.1f}{UNIT_SYMBOL} in preferred [{c.preferred_lo:.0f}-{c.preferred_hi:.0f}]"
                    if c.min_temp <= temp <= c.max_temp:
                        return f"{label} {temp:.1f}{UNIT_SYMBOL} in comfort [{c.min_temp:.0f}-{c.max_temp:.0f}]"
                    return f"{label} {temp:.1f}{UNIT_SYMBOL} outside [{c.min_temp:.0f}-{c.max_temp:.0f}]"
                # Device is on — explain what it's doing
                target = decision.command_targets.get(ed.name, ed.target)
                if target is not None and eff_cfg.control_type == "trajectory" and temp > target:
                    return f"standby: {label} {temp:.1f}{UNIT_SYMBOL} above {target:.0f}{UNIT_SYMBOL} target"
                return f"{label} {temp:.1f}{UNIT_SYMBOL}, target {target}{UNIT_SYMBOL}" if target else ""
        return ""

    rationale_map: dict[str, str] = {}
    for ed in decision.effectors:
        eff_cfg = EFFECTOR_MAP.get(ed.name)
        if eff_cfg is None:
            continue
        if eff_cfg.control_type == "trajectory":
            label = ed.name.removeprefix("thermostat_")
            setpoint = decision.command_targets.get(ed.name, 0)
            on_label = "ON" if ed.mode != "off" and ed.delay_steps == 0 else "OFF"
            print(f"  {label} heating:  {on_label} -> setpoint {setpoint:.0f}{UNIT_SYMBOL}")
            if ed.mode != "off":
                r = _counterfactual_rationale(ed.name)
                print(r)
                r_text = r.strip().removeprefix("-> ")
                if not r_text or r_text == "no significant effect":
                    r_text = _state_rationale(ed)
                rationale_map[ed.name] = r_text
            else:
                rationale_map[ed.name] = _state_rationale(ed)
        elif eff_cfg.control_type == "regulating":
            if ed.mode == "off":
                print(f"  {ed.name:<20} off")
                rationale_map[ed.name] = _state_rationale(ed)
            else:
                target_str = f" @ {ed.target:.0f}{UNIT_SYMBOL}" if ed.target is not None else ""
                print(f"  {ed.name:<20} {ed.mode}{target_str}")
                r = _counterfactual_rationale(ed.name)
                print(r)
                r_text = r.strip().removeprefix("-> ")
                if not r_text or r_text == "no significant effect":
                    r_text = _state_rationale(ed)
                rationale_map[ed.name] = r_text
        elif eff_cfg.control_type == "binary":
            print(f"  {ed.name:<20} {ed.mode}")
            if ed.mode != "off":
                r = _counterfactual_rationale(ed.name)
                print(r)
                r_text = r.strip().removeprefix("-> ")
                if not r_text or r_text == "no significant effect":
                    r_text = _state_rationale(ed)
                rationale_map[ed.name] = r_text
            else:
                rationale_map[ed.name] = _state_rationale(ed)
    if decision.trajectory_info:
        for eff_name, info in decision.trajectory_info.items():
            delay_h = info["delay_steps"] * 5 / 60
            dur = info.get("duration_steps")
            dur_str = f"{dur * 5 / 60:.0f}h" if dur is not None else "full"
            label = "ON now" if info["delay_steps"] == 0 else f"start in {delay_h:.0f}h"
            print(f"  Trajectory {eff_name}: {label}, duration {dur_str}")
    print(
        f"  Total cost: {decision.total_cost:.4f}"
        f" (comfort: {decision.comfort_cost:.4f}, energy: {decision.energy_cost:.4f})"
    )
    print(f"  All-off baseline: comfort={off_comfort:.4f}")

    # ── Per-sensor cost breakdown ──
    sensors_with_cost = sorted(
        s
        for s in set(dec_sensor_costs) | set(off_sensor_costs)
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
    col_w = max(len(SENSOR_LABELS.get(s, s)) for s in PREDICTION_SENSORS) + 2
    header = f"  {'Sensor':<{col_w}}" + "".join(f"{'dec ' + h:>9}{'off ' + h:>9}" for h in horizons)
    print("\n  Predicted temperatures (decision vs all-off):")
    print(header)
    print(f"  {'-' * (col_w + 18 * len(horizons))}")
    for sensor_col in PREDICTION_SENSORS:
        display = SENSOR_LABELS.get(sensor_col, sensor_col)
        dec_vals = decision.predictions.get(sensor_col, {})
        row = f"  {display:<{col_w}}"
        has_any = False
        for h_step, h_label in zip(CONTROL_HORIZONS, horizons, strict=True):
            dec_t = dec_vals.get(h_label)
            off_key = f"{sensor_col}_t+{h_step}"
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

    # Store rationale data on the decision for TUI display
    object.__setattr__(decision, "rationale", rationale_map)
    object.__setattr__(decision, "sensor_costs", dec_sensor_costs)
    object.__setattr__(decision, "baseline_sensor_costs", off_sensor_costs)
    object.__setattr__(decision, "baseline_cost", off_comfort)

    # ── Window opportunities (persistent, energy-aware) ──
    from weatherstat.advisory import evaluate_window_opportunities, process_opportunities

    opp_state = _HS(
        current_temps=current_temps,
        outdoor_temp=outdoor,
        forecast_temps=forecast_temp_list,
        window_states=window_states_dict,
        hour_of_day=fractional_hour,
        recent_history=recent_hist,
        solar_fractions=solar_fractions,
        solar_elevations=solar_elevations,
    )
    # Pass the same adjusted schedules used for the winning sweep, so the
    # re-sweep's comfort cost is comparable to winning_comfort_cost.
    # Using raw schedules would inflate the benefit by the MRT/profile delta.
    window_opportunities = evaluate_window_opportunities(
        opp_state,
        winning_scenario,
        winning_comfort_cost=decision.comfort_cost,
        winning_energy_cost=decision.energy_cost,
        sim_params=sim_params,
        schedules=schedules,
        base_hour=base_hour,
        prev_state=prev_state,
    )
    active_opps, dismissed_windows = process_opportunities(
        window_opportunities,
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

    # Write command JSON (omit targets for ineligible effectors)
    cmd_path = write_command_json(
        decision,
        opportunities=active_opps,
        ineligible_effectors=ineligible_effectors,
    )
    print(f"\n  Command JSON: {cmd_path}")

    # Log decision for outcome tracking
    from weatherstat.decision_log import log_decision

    log_decision(
        decision, current_temps, latest, schedules, base_hour, live,
        active_profile=active_profile.name if active_profile else None,
        mrt_base_offset=mrt_offset,
        mrt_weights=_mrt_weights,
        ineligible=ineligible,
        blocked_effectors=blocked_dict,
    )
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
            decision = run_control_cycle(live=live)
            if live and decision:
                from weatherstat.executor import execute

                execute()
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
