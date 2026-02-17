"""Control policy — comfort-optimizing HVAC selection.

Receding-horizon controller: "what HVAC settings maximize comfort over the next 6 hours?"
Re-evaluated every 15 minutes.

Control variables: 2 thermostats (binary on/off), 2 blowers (off/low/high),
2 mini-splits (off/heat/cool). 324-combo sweep (~4s). Config-driven device lists
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
    ADVISORY_EFFORT_COST,
    ADVISORY_QUIET_HOURS,
    BLOWERS,
    CONTROL_STATE_FILE,
    ENERGY_COST_BLOWER,
    ENERGY_COST_GAS_ZONE,
    ENERGY_COST_MINI_SPLIT,
    HORIZONS_5MIN,
    MINI_SPLIT_SWEEP_MODES,
    MINI_SPLIT_SWEEP_TARGET,
    MINI_SPLITS,
    PREDICTION_ROOMS,
    PREDICTIONS_DIR,
)
from weatherstat.inference import (
    _prepare_feature_row,
    build_features,
    build_hvac_overrides,
    fetch_recent_history,
    load_feature_columns,
    load_models,
)
from weatherstat.types import (
    Action,
    ActionOption,
    ActionRecommendation,
    BlowerDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlDecision,
    ControlState,
    ExecutionType,
    HVACScenario,
    MiniSplitDecision,
    RoomComfort,
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


# ── Default comfort profiles ──────────────────────────────────────────────


def default_comfort_schedules() -> list[ComfortSchedule]:
    """Comfort profiles from YAML config.

    All rooms with schedules are included so the optimizer considers whole-house comfort.
    Rooms without direct HVAC control use lighter penalties — there's only so much
    the thermostats can do for them.
    """
    schedules: list[ComfortSchedule] = []
    for room, entries in _CFG.comfort.items():
        schedule_entries = tuple(
            ComfortScheduleEntry(
                e.start_hour,
                e.end_hour,
                RoomComfort(room, e.min_temp, e.max_temp, e.cold_penalty, e.hot_penalty),
            )
            for e in entries
        )
        schedules.append(ComfortSchedule(room=room, entries=schedule_entries))
    return schedules


def adjust_schedules_for_windows(
    schedules: list[ComfortSchedule],
    window_states: dict[str, bool],
    window_config: dict[str, object],
    min_offset: float,
    max_offset: float,
) -> list[ComfortSchedule]:
    """Widen comfort bounds for rooms with open windows.

    When a room's window is open, shift min_temp down and max_temp up.
    This makes the optimizer less eager to heat/cool a room with an open window.

    Args:
        schedules: Comfort schedules for all rooms.
        window_states: window_name -> is_open for each window.
        window_config: window_name -> WindowConfig from YAML.
        min_offset: Amount to add to min_temp (negative = lower).
        max_offset: Amount to add to max_temp (positive = higher).

    Returns:
        New list of ComfortSchedule with adjusted bounds for affected rooms.
    """
    # Build set of rooms with open windows
    rooms_with_open_windows: set[str] = set()
    for wname, is_open in window_states.items():
        if is_open:
            wcfg = window_config.get(wname)
            if wcfg is not None:
                for room in wcfg.rooms:
                    rooms_with_open_windows.add(room)

    if not rooms_with_open_windows:
        return schedules

    adjusted: list[ComfortSchedule] = []
    for schedule in schedules:
        if schedule.room not in rooms_with_open_windows:
            adjusted.append(schedule)
            continue
        new_entries = tuple(
            ComfortScheduleEntry(
                e.start_hour,
                e.end_hour,
                RoomComfort(
                    e.comfort.room,
                    e.comfort.min_temp + min_offset,
                    e.comfort.max_temp + max_offset,
                    e.comfort.cold_penalty,
                    e.comfort.hot_penalty,
                ),
            )
            for e in schedule.entries
        )
        adjusted.append(ComfortSchedule(room=schedule.room, entries=new_entries))
    return adjusted


def _in_quiet_hours(hour: int, quiet: tuple[int, int]) -> bool:
    """Return True if the given hour falls within quiet hours (wraps midnight)."""
    start, end = quiet
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


# ── Zone mapping ──────────────────────────────────────────────────────────
# Maps rooms to HVAC zones. Used as fallback in compute_comfort_cost when
# a room-specific model prediction is unavailable.

ROOM_TO_ZONE: dict[str, str] = _CFG.room_to_zone


# ── Cost function ─────────────────────────────────────────────────────────


def compute_comfort_cost(
    predictions: dict[str, float],
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> float:
    """Compute total comfort cost across all rooms and horizons.

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
        zone = ROOM_TO_ZONE.get(schedule.room)
        if zone is None:
            continue

        for h in CONTROL_HORIZONS:
            weight = HORIZON_WEIGHTS.get(h, 0.5)
            hours_ahead = horizon_hours.get(h, h // 12)
            future_hour = (base_hour + hours_ahead) % 24

            comfort = schedule.comfort_at(future_hour)
            if comfort is None:
                continue

            # Prefer room's own model prediction; fall back to zone thermostat prediction
            room = schedule.room
            pred_key = f"{room}_temp_t+{h}"
            pred_temp = predictions.get(pred_key)
            if pred_temp is None:
                pred_key = f"{zone}_temp_t+{h}"
                pred_temp = predictions.get(pred_key)
            if pred_temp is None:
                continue

            # Quadratic penalty for being outside comfort bounds
            if pred_temp < comfort.min_temp:
                cost += (comfort.min_temp - pred_temp) ** 2 * comfort.cold_penalty * weight
            elif pred_temp > comfort.max_temp:
                cost += (pred_temp - comfort.max_temp) ** 2 * comfort.hot_penalty * weight

    return cost


def compute_energy_cost(scenario: HVACScenario) -> float:
    """Tiered energy penalty: gas zones > mini-splits > blower fans.

    Used as tiebreaker when comfort cost is equal — prefer less energy usage.
    """
    cost = 0.0
    # Gas zones (Navien boiler via thermostat)
    if scenario.upstairs_heating:
        cost += ENERGY_COST_GAS_ZONE
    if scenario.downstairs_heating:
        cost += ENERGY_COST_GAS_ZONE
    # Mini-splits (heat pump — efficient but uses electricity)
    for sd in scenario.mini_splits:
        if sd.mode != "off":
            cost += ENERGY_COST_MINI_SPLIT
    # Blower fans (negligible)
    for bd in scenario.blowers:
        cost += ENERGY_COST_BLOWER.get(bd.mode, 0.0)
    return cost


def _zone_comfort_max(zone: str, schedules: list[ComfortSchedule], hour: int) -> float:
    """Get the comfort max for a zone's primary thermostat at the given hour."""
    for s in schedules:
        if s.room == zone:
            c = s.comfort_at(hour)
            if c is not None:
                return c.max_temp
    return ABSOLUTE_MAX


def _zone_comfort_min(zone: str, schedules: list[ComfortSchedule], hour: int) -> float:
    """Get the comfort min for a zone's primary thermostat at the given hour."""
    for s in schedules:
        if s.room == zone:
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


def _apply_newton_floor(
    preds: dict[str, float],
    base_row: pd.DataFrame,
    current_temps: dict[str, float],
) -> list[str]:
    """Cap passive (all-off) predictions at Newton's law values.

    The ML model was trained mostly with HVAC on, so it structurally cannot
    predict passive temperature drift accurately (exploration-exploitation
    problem). Newton's law of cooling provides the physics-correct prediction.

    Handles both thermal directions:
    - Winter (indoor > outdoor): Newton predicts cooling. Model may over-predict
      warmth. Cap downward with min(model, newton) — "floor".
    - Summer (indoor < outdoor): Newton predicts warming. Model may over-predict
      coolness. Cap upward with max(model, newton) — "ceiling".

    Modifies preds in-place. Returns list of cap descriptions for logging.
    """
    if "outdoor_temp" not in base_row.columns:
        return []
    outdoor_temp = float(base_row["outdoor_temp"].iloc[0])
    if np.isnan(outdoor_temp):
        return []

    caps: list[str] = []
    for room in PREDICTION_ROOMS:
        room_temp = current_temps.get(room)
        if room_temp is None or room_temp == outdoor_temp:
            continue
        winter = room_temp > outdoor_temp  # cooling toward outdoor
        for h in CONTROL_HORIZONS:
            label = HORIZON_LABELS[h]
            newton_col = f"{room}_newton_sealed_{label}"
            if newton_col not in base_row.columns:
                continue
            newton_pred = float(base_row[newton_col].iloc[0])
            if np.isnan(newton_pred):
                continue
            key = f"{room}_temp_t+{h}"
            if key not in preds:
                continue
            if winter and preds[key] > newton_pred:
                direction = "floor"
                caps.append(f"{room} {label} {direction}: {preds[key]:.1f}→{newton_pred:.1f}°F")
                preds[key] = newton_pred
            elif not winter and preds[key] < newton_pred:
                direction = "ceiling"
                caps.append(f"{room} {label} {direction}: {preds[key]:.1f}→{newton_pred:.1f}°F")
                preds[key] = newton_pred
    return caps


# ── Scenario generation & sweep ──────────────────────────────────────────


def generate_scenarios() -> list[HVACScenario]:
    """Generate all HVAC combinations from config-driven device lists.

    Cartesian product: thermostats (4) × blowers (levels^n) × mini-splits (modes^n),
    pruned by physical constraints (blowers only useful when their zone is heating).
    """
    from itertools import product

    # Mini-split combinations (independent of thermostats)
    split_mode_lists = [MINI_SPLIT_SWEEP_MODES for _ in MINI_SPLITS]
    split_combos: list[tuple[MiniSplitDecision, ...]] = []
    for modes in product(*split_mode_lists):
        split_combos.append(
            tuple(
                MiniSplitDecision(s.name, mode, MINI_SPLIT_SWEEP_TARGET)
                for s, mode in zip(MINI_SPLITS, modes, strict=True)
            )
        )

    # Full cartesian product with blower constraint:
    # Blowers only redistribute heat from the hydronic slab — useless when zone isn't heating.
    scenarios: list[HVACScenario] = []
    for up_on in [True, False]:
        for dn_on in [True, False]:
            heating = {"upstairs": up_on, "downstairs": dn_on}

            # Build blower levels per device: full range if zone heating, force off otherwise
            per_blower_levels = []
            for b in BLOWERS:
                if heating.get(b.zone, False):
                    per_blower_levels.append(b.levels)
                else:
                    per_blower_levels.append(("off",))

            for levels in product(*per_blower_levels):
                blowers = tuple(
                    BlowerDecision(b.name, lvl) for b, lvl in zip(BLOWERS, levels, strict=True)
                )
                for splits in split_combos:
                    scenarios.append(HVACScenario(up_on, dn_on, blowers, splits))

    return scenarios


def _batch_predict(
    base_row: pd.DataFrame,
    overrides_list: list[dict[str, float]],
    models: dict[str, object],
) -> tuple[list[str], np.ndarray]:
    """Predict all scenarios in batch — one model.predict() call per target.

    Replaces N × M individual predict calls with M batch calls (one per model).
    For the HVAC sweep: 7,200 calls → 40. For the window sweep: 5,120 → 40.

    Args:
        base_row: 1-row DataFrame of base features.
        overrides_list: Per-scenario feature column overrides.
        models: Target name → LightGBM Booster mapping.

    Returns:
        (target_names, predictions) where predictions shape is (n_scenarios, n_targets).
    """
    n = len(overrides_list)
    target_names = list(models.keys())
    if n == 0:
        return target_names, np.empty((0, len(target_names)))

    col_to_idx = {col: i for i, col in enumerate(base_row.columns)}

    # Tile base feature row into (n_scenarios × n_features) numpy array
    X = np.tile(base_row.values.astype(np.float64), (n, 1))

    # Apply per-scenario overrides via direct array indexing
    for i, overrides in enumerate(overrides_list):
        for col, val in overrides.items():
            idx = col_to_idx.get(col)
            if idx is not None:
                X[i, idx] = val

    # Wrap in DataFrame for model compatibility (LightGBM + test mocks accept DataFrames)
    X_df = pd.DataFrame(X, columns=base_row.columns)

    # One batch predict call per target model
    result = np.empty((n, len(target_names)))
    for j, target in enumerate(target_names):
        result[:, j] = models[target].predict(X_df)  # type: ignore[union-attr]

    return target_names, result


def sweep_scenarios(
    base_row: pd.DataFrame,
    feature_columns: list[str],
    models: dict[str, object],
    up_current: float,
    dn_current: float,
    current_split_temps: dict[str, float],
    current_temps: dict[str, float],
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> ControlDecision:
    """Sweep all HVAC combinations and return the best decision.

    Unified sweep over thermostats × blowers × mini-splits.
    Physical constraints prune the search space before the sweep.
    """
    scenarios = generate_scenarios()
    pre_count = len(scenarios)
    blocked_reasons: list[str] = []

    # ── Physical constraint 1: zone thermostat at/above comfort max ────
    # Heating a zone when the thermostat already exceeds comfort is wasteful.
    up_max = _zone_comfort_max("upstairs", schedules, base_hour)
    dn_max = _zone_comfort_max("downstairs", schedules, base_hour)
    up_allow = up_current < up_max
    dn_allow = dn_current < dn_max
    if not up_allow or not dn_allow:
        scenarios = [
            s for s in scenarios
            if (up_allow or not s.upstairs_heating) and (dn_allow or not s.downstairs_heating)
        ]
        if not up_allow:
            blocked_reasons.append(f"upstairs at/above max ({up_current:.1f}°F >= {up_max:.0f}°F)")
        if not dn_allow:
            blocked_reasons.append(f"downstairs at/above max ({dn_current:.1f}°F >= {dn_max:.0f}°F)")

    # ── Physical constraint 2: thermal direction ────
    # If no room is currently below its comfort min (looking at all future
    # horizons' schedules), heating serves no purpose. The 15-min control
    # loop will re-evaluate if temps drop.
    horizon_hours = {12: 1, 24: 2, 48: 4, 72: 6}
    any_room_cold = False
    for schedule in schedules:
        room = schedule.room
        temp = current_temps.get(room)
        if temp is None:
            continue
        for h in CONTROL_HORIZONS:
            hours_ahead = horizon_hours.get(h, h // 12)
            future_hour = (base_hour + hours_ahead) % 24
            comfort = schedule.comfort_at(future_hour)
            if comfort is not None and temp < comfort.min_temp:
                any_room_cold = True
                break
        if any_room_cold:
            break

    if not any_room_cold and (up_allow or dn_allow):
        scenarios = [s for s in scenarios if not s.upstairs_heating and not s.downstairs_heating]
        blocked_reasons.append("no room below comfort min")

    if blocked_reasons:
        print(f"  Heating blocked: {'; '.join(blocked_reasons)}")
        print(f"  Reduced scenarios: {pre_count} → {len(scenarios)}")

    def _is_all_off(s: HVACScenario) -> bool:
        return (
            not s.upstairs_heating
            and not s.downstairs_heating
            and all(b.mode == "off" for b in s.blowers)
            and all(sp.mode == "off" for sp in s.mini_splits)
        )

    # ── Phase 1: batch predict all scenarios ──
    overrides_list = [build_hvac_overrides(s, current_split_temps) for s in scenarios]
    target_names, pred_matrix = _batch_predict(base_row, overrides_list, models)
    target_idx = {t: j for j, t in enumerate(target_names)}

    # ── Newton floor for all-off predictions ──
    # ML models can't predict passive cooling at low temps (trained on HVAC-on
    # data). Newton's law provides the physics-correct passive prediction.
    # Cap all-off model predictions so the sweep sees realistic cooling costs.
    for i, scenario in enumerate(scenarios):
        if not _is_all_off(scenario):
            continue
        off_preds_i = {t: float(pred_matrix[i, j]) for t, j in target_idx.items()}
        caps = _apply_newton_floor(off_preds_i, base_row, current_temps)
        if caps:
            for t, j in target_idx.items():
                pred_matrix[i, j] = off_preds_i[t]
            print(f"  Newton floor ({len(caps)} caps):")
            for cap in caps[:8]:
                print(f"    {cap}")
            if len(caps) > 8:
                print(f"    ... and {len(caps) - 8} more")

    # ── Phase 2: score each scenario ──
    best_idx = -1
    best_cost = float("inf")
    off_idx = -1
    off_cost = float("inf")

    for i, scenario in enumerate(scenarios):
        predictions = {t: float(pred_matrix[i, j]) for t, j in target_idx.items()}
        comfort = compute_comfort_cost(predictions, schedules, base_hour)
        energy = compute_energy_cost(scenario)
        total = comfort + energy

        if _is_all_off(scenario):
            off_idx = i
            off_cost = total

        if total < best_cost:
            best_cost = total
            best_idx = i

    if best_idx < 0:
        raise RuntimeError("No HVAC scenarios evaluated")

    # Minimum improvement safeguard: only choose an active HVAC scenario if it
    # improves total cost over all-off by at least MIN_IMPROVEMENT. Prevents
    # noise-driven decisions when the model barely differentiates scenarios.
    if off_idx >= 0 and best_idx != off_idx:
        improvement = off_cost - best_cost
        if improvement < MIN_IMPROVEMENT:
            print(
                f"  Reverting to all-off: improvement {improvement:.3f}"
                f" < threshold {MIN_IMPROVEMENT:.1f}"
            )
            best_idx = off_idx

    # ── Cold-room safety override ──
    # When the model is undertrained it may predict rooms warming without HVAC
    # (because all training data had HVAC on). If the sweep chose all-off but
    # a room is significantly below comfort min, force zone heating regardless.
    if _is_all_off(scenarios[best_idx]):
        cold_zones: set[str] = set()
        cold_rooms_info: list[str] = []
        for schedule in schedules:
            room = schedule.room
            temp = current_temps.get(room)
            if temp is None:
                continue
            comfort = schedule.comfort_at(base_hour)
            if comfort is None:
                continue
            if temp < comfort.min_temp - COLD_ROOM_OVERRIDE:
                zone = ROOM_TO_ZONE.get(room)
                if zone:
                    cold_zones.add(zone)
                    cold_rooms_info.append(f"{room} ({temp:.1f}°F < {comfort.min_temp:.0f}°F)")

        # Only override zones where heating is allowed (not blocked by comfort max)
        if not up_allow:
            cold_zones.discard("upstairs")
        if not dn_allow:
            cold_zones.discard("downstairs")

        if cold_zones:
            # Find cheapest scenario that heats all cold zones
            constrained_best = -1
            constrained_cost = float("inf")
            for i, scenario in enumerate(scenarios):
                if "upstairs" in cold_zones and not scenario.upstairs_heating:
                    continue
                if "downstairs" in cold_zones and not scenario.downstairs_heating:
                    continue
                predictions = {t: float(pred_matrix[i, j]) for t, j in target_idx.items()}
                c = compute_comfort_cost(predictions, schedules, base_hour)
                e = compute_energy_cost(scenario)
                if c + e < constrained_cost:
                    constrained_cost = c + e
                    constrained_best = i
            if constrained_best >= 0:
                print(f"  Cold room override: {', '.join(cold_rooms_info)}")
                best_idx = constrained_best

    # ── Phase 3: build ControlDecision for winner ──
    scenario = scenarios[best_idx]
    predictions = {t: float(pred_matrix[best_idx, j]) for t, j in target_idx.items()}
    comfort = compute_comfort_cost(predictions, schedules, base_hour)
    energy = compute_energy_cost(scenario)
    total = comfort + energy

    room_preds: dict[str, dict[str, float]] = {}
    for room in PREDICTION_ROOMS:
        rpred: dict[str, float] = {}
        for h in CONTROL_HORIZONS:
            key = f"{room}_temp_t+{h}"
            val = predictions.get(key)
            if val is not None:
                rpred[HORIZON_LABELS[h]] = round(val, 2)
        if rpred:
            room_preds[room] = rpred

    # Compute mini-split command targets from comfort schedule midpoints
    final_splits: list[MiniSplitDecision] = []
    for sd in scenario.mini_splits:
        if sd.mode == "off":
            final_splits.append(sd)
        else:
            target_temp = MINI_SPLIT_SWEEP_TARGET  # fallback
            for sched in schedules:
                if sched.room == sd.name:
                    comfort_entry = sched.comfort_at(base_hour)
                    if comfort_entry is not None:
                        target_temp = (comfort_entry.min_temp + comfort_entry.max_temp) / 2
                    break
            final_splits.append(MiniSplitDecision(sd.name, sd.mode, target_temp))

    best_decision = ControlDecision(
        timestamp=datetime.now(UTC).isoformat(),
        upstairs_heating=scenario.upstairs_heating,
        downstairs_heating=scenario.downstairs_heating,
        upstairs_setpoint=_cautious_setpoint(
            up_current, scenario.upstairs_heating,
            comfort_min=_zone_comfort_min("upstairs", schedules, base_hour),
        ),
        downstairs_setpoint=_cautious_setpoint(
            dn_current, scenario.downstairs_heating,
            comfort_min=_zone_comfort_min("downstairs", schedules, base_hour),
        ),
        blowers=scenario.blowers,
        mini_splits=tuple(final_splits),
        total_cost=round(total, 4),
        comfort_cost=round(comfort, 4),
        energy_cost=round(energy, 4),
        room_predictions=room_preds,
    )

    return best_decision


# ── Advisory (window) sweep ────────────────────────────────────────────────


def build_advisory_actions(window_states: dict[str, bool]) -> list[Action]:
    """Build Action objects for each window from YAML config + current states.

    Args:
        window_states: window_name -> is_open for each window in the config.

    Returns:
        One Action per window, each with "open" and "closed" options.
    """
    actions: list[Action] = []
    for name in _CFG.windows:
        col = f"window_{name}_open"
        open_opt = ActionOption(name="open", feature_overrides={col: 1.0})
        closed_opt = ActionOption(name="closed", feature_overrides={col: 0.0})
        current = "open" if window_states.get(name, False) else "closed"
        actions.append(Action(
            name=name,
            options=(open_opt, closed_opt),
            current=current,
            execution=ExecutionType.ADVISORY,
            effort_cost=ADVISORY_EFFORT_COST,
        ))
    return actions


def _decision_to_overrides(
    decision: ControlDecision,
    current_split_temps: dict[str, float],
) -> dict[str, float]:
    """Reconstruct HVAC feature overrides from a ControlDecision.

    Used to apply the electronic decision's HVAC state as the baseline for
    the window sweep.
    """
    scenario = HVACScenario(
        upstairs_heating=decision.upstairs_heating,
        downstairs_heating=decision.downstairs_heating,
        blowers=decision.blowers,
        mini_splits=decision.mini_splits,
    )
    return build_hvac_overrides(scenario, current_split_temps)


def _build_advisory_message(
    changed_windows: list[str],
    improvement: float,
    adv_preds: dict[str, float],
    baseline_preds: dict[str, float],
    recommend_open: bool,
) -> str:
    """Build human-readable advisory message with per-room temperature effects.

    Args:
        changed_windows: Names of windows to change.
        improvement: Total comfort cost improvement.
        adv_preds: Predictions with the recommended window states.
        baseline_preds: Predictions with current window states.
        recommend_open: True if recommending opening, False if closing.
    """
    action_verb = "Open" if recommend_open else "Close"

    # Build per-room temperature effects at 1h horizon (step 12)
    room_effects: list[str] = []
    for wname in changed_windows:
        wcfg = _CFG.windows.get(wname)
        if wcfg is None:
            continue
        for room in wcfg.rooms:
            key = f"{room}_temp_t+12"
            adv_t = adv_preds.get(key)
            base_t = baseline_preds.get(key)
            if adv_t is not None and base_t is not None:
                delta = adv_t - base_t
                sign = "+" if delta >= 0 else ""
                room_effects.append(f"{sign}{delta:.1f}°F {room}")

    # Format window list
    if len(changed_windows) == 1:
        window_desc = f"the {changed_windows[0]} window"
    else:
        window_desc = "the " + " and ".join(
            ", ".join(changed_windows).rsplit(", ", 1)
        ) + " windows"

    parts = [f"{action_verb} {window_desc}"]
    if room_effects:
        parts.append(f"predicted {', '.join(room_effects)} at 1h")
    parts.append(f"(improvement: {improvement:.1f})")
    return " — ".join(parts)


def evaluate_advisories(
    base_row: pd.DataFrame,
    feature_columns: list[str],
    models: dict[str, object],
    electronic_overrides: dict[str, float],
    advisory_actions: list[Action],
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> list[ActionRecommendation]:
    """Full 2^N window sweep to find the best combination of window changes.

    Args:
        base_row: Feature row (1-row DataFrame) from the current state.
        feature_columns: Ordered feature column names.
        models: Loaded LightGBM models keyed by target name.
        electronic_overrides: HVAC feature overrides from the electronic decision.
        advisory_actions: Action objects for each window.
        schedules: Comfort schedules for all rooms.
        base_hour: Current hour of day (0-23).

    Returns:
        Sorted list of ActionRecommendation for each window that should change.
        Empty if no improvement exceeds the effort threshold.
    """
    from itertools import product

    n = len(advisory_actions)
    if n == 0:
        return []

    # Build X_base with electronic overrides applied
    X_base = base_row.copy()
    for col, val in electronic_overrides.items():
        if col in X_base.columns:
            X_base[col] = val

    # Build current state index (for skipping)
    current_indices = tuple(
        0 if a.current == a.options[0].name else 1
        for a in advisory_actions
    )

    has_any_window = "any_window_open" in X_base.columns

    # Build all combo overrides (baseline = empty overrides as first entry)
    combos: list[tuple[int, ...]] = [current_indices]  # index 0 = baseline
    overrides_list: list[dict[str, float]] = [{}]  # baseline: no window changes

    for combo in product(range(2), repeat=n):
        if combo == current_indices:
            continue

        window_overrides: dict[str, float] = {}
        for i, opt_idx in enumerate(combo):
            option = advisory_actions[i].options[opt_idx]
            window_overrides.update(option.feature_overrides)

        if has_any_window:
            any_open = any(
                advisory_actions[i].options[combo[i]].name == "open"
                for i in range(n)
            )
            window_overrides["any_window_open"] = float(any_open)

        combos.append(combo)
        overrides_list.append(window_overrides)

    # Batch predict all combos (including baseline)
    target_names, pred_matrix = _batch_predict(X_base, overrides_list, models)
    target_idx = {t: j for j, t in enumerate(target_names)}

    # Score baseline (index 0)
    baseline_preds = {t: float(pred_matrix[0, j]) for t, j in target_idx.items()}
    baseline_comfort = compute_comfort_cost(baseline_preds, schedules, base_hour)

    # Score all non-baseline combos
    best_improvement = 0.0
    best_combo: tuple[int, ...] | None = None
    best_preds: dict[str, float] = {}

    for k in range(1, len(combos)):
        combo_preds = {t: float(pred_matrix[k, j]) for t, j in target_idx.items()}
        combo_comfort = compute_comfort_cost(combo_preds, schedules, base_hour)

        improvement = baseline_comfort - combo_comfort
        if improvement > best_improvement:
            best_improvement = improvement
            best_combo = combos[k]
            best_preds = combo_preds

    if best_combo is None:
        return []

    # Determine which windows changed
    changed: list[tuple[str, str, str]] = []  # (name, recommended, current)
    for i, (new_idx, cur_idx) in enumerate(zip(best_combo, current_indices, strict=True)):
        if new_idx != cur_idx:
            action = advisory_actions[i]
            changed.append((
                action.name,
                action.options[new_idx].name,
                action.current,
            ))

    num_changes = len(changed)
    effort_threshold = ADVISORY_EFFORT_COST * num_changes
    if best_improvement <= effort_threshold:
        return []

    # Determine if this is an "open" or "close" recommendation
    recommend_open = any(rec == "open" for _, rec, _ in changed)
    changed_names = [name for name, _, _ in changed]
    message = _build_advisory_message(
        changed_names, best_improvement, best_preds, baseline_preds, recommend_open,
    )

    recommendations: list[ActionRecommendation] = []
    for name, recommended, current in changed:
        recommendations.append(ActionRecommendation(
            action_name=name,
            recommended_state=recommended,
            current_state=current,
            comfort_improvement=best_improvement,
            message=message,
        ))

    return sorted(recommendations, key=lambda r: r.action_name)


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
        )
    except (json.JSONDecodeError, KeyError):
        return None


def save_control_state(decision: ControlDecision) -> None:
    """Persist control state to prevent rapid cycling."""
    state: dict[str, object] = {
        "last_decision_time": decision.timestamp,
        "upstairs_setpoint": decision.upstairs_setpoint,
        "downstairs_setpoint": decision.downstairs_setpoint,
        "blower_modes": {bd.name: bd.mode for bd in decision.blowers},
        "mini_split_modes": {sd.name: sd.mode for sd in decision.mini_splits},
        "mini_split_targets": {sd.name: sd.target for sd in decision.mini_splits},
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
    """Return True if 1h predictions look reasonable for all rooms."""
    safe = True
    for room, preds in decision.room_predictions.items():
        pred_1h = preds.get("1h")
        current = current_temps.get(room)
        if pred_1h is None or current is None:
            continue
        if abs(pred_1h - current) > MAX_1H_CHANGE:
            print(
                f"  WARNING: {room} 1h prediction {pred_1h:.1f}°F is"
                f" >{MAX_1H_CHANGE}°F from current {current:.1f}°F"
            )
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

    # Fetch data
    print("[control] Fetching recent history from Home Assistant...")
    df_raw = fetch_recent_history(hours_back=14)
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
    from weatherstat.features import ROOM_TEMP_COLUMNS

    latest = df_raw.iloc[-1]
    up_current = float(latest.get("thermostat_upstairs_temp", 70))
    dn_current = float(latest.get("thermostat_downstairs_temp", 70))
    up_target = latest.get("thermostat_upstairs_target", "?")
    dn_target = latest.get("thermostat_downstairs_target", "?")
    out_temp = latest.get("outdoor_temp")
    now_str = df_raw["timestamp"].iloc[-1]

    # Build current temperature dict for all rooms (for sanity checks and display)
    current_temps: dict[str, float] = {}
    for room, col in ROOM_TEMP_COLUMNS.items():
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
        print(f"  Outdoor:     {float(out_temp):.1f}°F")
    window_cols = _CFG.window_display_map
    open_windows = [label for col, label in window_cols.items() if bool(latest.get(col, False))]
    if open_windows:
        print(f"  Windows:     {', '.join(open_windows)} open")
    else:
        print("  Windows:     all closed")
    other_rooms = [r for r in PREDICTION_ROOMS if r not in ("upstairs", "downstairs")]
    for room in other_rooms:
        t = current_temps.get(room)
        if t is not None:
            print(f"  {room:<14} {t:.1f}°F")
    # Show current blower/mini-split state
    for cfg in BLOWERS:
        mode = str(latest.get(f"blower_{cfg.name}_mode", "?"))
        print(f"  blower_{cfg.name:<10} {mode}")
    for cfg in MINI_SPLITS:
        mode = str(latest.get(f"mini_split_{cfg.name}_mode", "?"))
        target = latest.get(f"mini_split_{cfg.name}_target", "?")
        print(f"  split_{cfg.name:<12} {mode} @ {target}°F")

    # Load models
    feature_columns = load_feature_columns("full")
    models = load_models("full", HORIZONS_5MIN)
    if not models or not feature_columns:
        print("  ERROR: Full models not found. Run `just train-full` first.", file=sys.stderr)
        return None

    # Build features
    df_feat = build_features(df_raw.copy(), mode="full")
    base_row = _prepare_feature_row(df_feat, feature_columns)

    # Current local hour for comfort schedule lookup
    from zoneinfo import ZoneInfo

    from weatherstat.config import TIMEZONE

    base_hour = datetime.now(ZoneInfo(TIMEZONE)).hour

    # Sweep all HVAC combinations
    schedules = default_comfort_schedules()
    window_states_dict = {
        name: bool(latest.get(f"window_{name}_open", False))
        for name in _CFG.windows
    }
    schedules = adjust_schedules_for_windows(
        schedules, window_states_dict, _CFG.windows, *_CFG.window_open_offset,
    )
    rooms_with_open = set()
    for wn, ws in window_states_dict.items():
        if ws:
            for r in _CFG.windows[wn].rooms:
                rooms_with_open.add(r)
    if rooms_with_open:
        print(f"  Comfort adjusted for open windows: {', '.join(sorted(rooms_with_open))}")
    n_scenarios = len(generate_scenarios())
    print(f"\n[control] Sweeping {n_scenarios} HVAC combinations...")
    t0 = time.monotonic()
    decision = sweep_scenarios(
        base_row,
        feature_columns,
        models,
        up_current,
        dn_current,
        current_split_temps,
        current_temps,
        schedules,
        base_hour,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    print(f"  Sweep completed in {elapsed_ms:.0f}ms ({elapsed_ms / n_scenarios:.1f}ms/combo)")

    up_label = "ON" if decision.upstairs_heating else "OFF"
    dn_label = "ON" if decision.downstairs_heating else "OFF"

    # Print decision
    print("\n[control] Decision:")
    print(f"  Upstairs heating:   {up_label} → setpoint {decision.upstairs_setpoint:.0f}°F")
    print(f"  Downstairs heating: {dn_label} → setpoint {decision.downstairs_setpoint:.0f}°F")
    for bd in decision.blowers:
        print(f"  Blower {bd.name:<14} {bd.mode}")
    for sd in decision.mini_splits:
        if sd.mode == "off":
            print(f"  Split {sd.name:<15} off")
        else:
            print(f"  Split {sd.name:<15} {sd.mode} @ {sd.target:.0f}°F")
    print(
        f"  Total cost: {decision.total_cost:.4f}"
        f" (comfort: {decision.comfort_cost:.4f}, energy: {decision.energy_cost:.4f})"
    )

    # Per-room prediction table (decision vs all-off baseline)
    horizons = [HORIZON_LABELS[h] for h in CONTROL_HORIZONS]

    # Compute all-off baseline prediction for comparison
    all_off = HVACScenario(
        False, False,
        tuple(BlowerDecision(b.name, "off") for b in BLOWERS),
        tuple(MiniSplitDecision(s.name, "off", MINI_SPLIT_SWEEP_TARGET) for s in MINI_SPLITS),
    )
    off_overrides = build_hvac_overrides(all_off, current_split_temps)
    off_targets, off_matrix = _batch_predict(base_row, [off_overrides], models)
    off_preds = {t: float(off_matrix[0, j]) for j, t in enumerate(off_targets)}
    _apply_newton_floor(off_preds, base_row, current_temps)
    off_comfort = compute_comfort_cost(off_preds, schedules, base_hour)

    print(f"\n  All-off baseline: comfort={off_comfort:.4f}")

    header = f"  {'Room':<14}" + "".join(f"{'dec ' + h:>9}{'off ' + h:>9}" for h in horizons)
    print("\n  Predicted temperatures (decision vs all-off):")
    print(header)
    print(f"  {'-' * (14 + 18 * len(horizons))}")
    for room in PREDICTION_ROOMS:
        dec_vals = decision.room_predictions.get(room, {})
        row = f"  {room:<14}"
        has_any = False
        for h_step, h_label in zip(CONTROL_HORIZONS, horizons, strict=True):
            dec_t = dec_vals.get(h_label)
            off_key = f"{room}_temp_t+{h_step}"
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

    # Safety checks
    sane = check_prediction_sanity(decision, current_temps)

    # Write command JSON
    cmd_path = write_command_json(decision)
    print(f"\n  Command JSON: {cmd_path}")

    if live:
        if not sane:
            print("  SKIPPED: prediction sanity check failed")
            return decision
        # Save state to prevent rapid cycling
        save_control_state(decision)
        print("  Mode: LIVE — command written for executor")
    else:
        print("  Mode: DRY-RUN — command written but not executed")

    # ── Model-based advisory sweep ──
    advisory_actions = build_advisory_actions(window_states_dict)
    electronic_overrides = _decision_to_overrides(decision, current_split_temps)

    n_combos = 2 ** len(advisory_actions)
    t1 = time.monotonic()
    recommendations = evaluate_advisories(
        base_row, feature_columns, models,
        electronic_overrides, advisory_actions,
        schedules, base_hour,
    )
    adv_ms = (time.monotonic() - t1) * 1000

    if recommendations:
        # All recommendations share the same message (multi-window)
        print(f"\n[advisory] Window sweep ({n_combos} combos, {adv_ms:.0f}ms):")
        print(f"  {recommendations[0].message}")
        # Dispatch notifications for live mode
        if live:
            from weatherstat.advisory import (
                Advisory,
                AdvisoryType,
                is_on_cooldown,
                load_advisory_state,
                save_advisory_state,
                send_ha_notification,
            )

            state = load_advisory_state()
            # Map recommendation direction to advisory type
            rec_open = any(r.recommended_state == "open" for r in recommendations)
            adv_type = AdvisoryType.FREE_COOLING if rec_open else AdvisoryType.CLOSE_WINDOWS
            if _in_quiet_hours(base_hour, ADVISORY_QUIET_HOURS):
                print("  (quiet hours, not sent)")
            elif not is_on_cooldown(state, adv_type):
                advisory = Advisory(
                    advisory_type=adv_type,
                    title="Window advisory",
                    message=recommendations[0].message,
                )
                if send_ha_notification(advisory, target=_CFG.notification_target):
                    state[adv_type.value] = time.time()
                    save_advisory_state(state)
                    print(f"  → Sent to HA ({_CFG.notification_target})")
            else:
                print("  (on cooldown, not sent)")
    else:
        print(f"\n[advisory] No window recommendations ({n_combos} combos, {adv_ms:.0f}ms)")

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
