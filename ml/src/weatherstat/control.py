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
    BlowerDecision,
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlDecision,
    ControlState,
    HVACScenario,
    MiniSplitDecision,
    RoomComfort,
)

# ── Constants ──────────────────────────────────────────────────────────────

# Cautious setpoint offset: when the control loop decides "heat on", we set the
# thermostat to current_temp + CAUTIOUS_OFFSET. If the loop is interrupted, the
# house drifts gently instead of running away to extreme temperatures.
CAUTIOUS_OFFSET = 2  # °F above/below current temp

# Absolute safety bounds (in case of stale current_temp or other weirdness)
ABSOLUTE_MIN = 62
ABSOLUTE_MAX = 78

# Minimum hold time before changing setpoints (seconds)
MIN_HOLD_SECONDS = 30 * 60  # 30 minutes

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

# Control loop interval
LOOP_INTERVAL_SECONDS = 15 * 60  # 15 minutes

# Horizons used for control (subset of HORIZONS_5MIN, skip 12h for control)
CONTROL_HORIZONS = [12, 24, 48, 72]

# Human-readable labels for horizon steps (5-min intervals)
HORIZON_LABELS: dict[int, str] = {12: "1h", 24: "2h", 48: "4h", 72: "6h", 144: "12h"}


# ── Default comfort profiles ──────────────────────────────────────────────


def default_comfort_schedules() -> list[ComfortSchedule]:
    """Initial comfort profiles from user preferences.

    All 8 rooms have schedules so the optimizer considers whole-house comfort.
    Rooms without direct HVAC control (kitchen, piano, bathroom) use lighter
    penalties — there's only so much the thermostats can do for them.
    """
    return [
        ComfortSchedule(
            room="upstairs",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("upstairs", 70.0, 74.0)),),
        ),
        ComfortSchedule(
            room="downstairs",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("downstairs", 70.0, 74.0)),),
        ),
        ComfortSchedule(
            room="office",
            entries=(
                ComfortScheduleEntry(
                    8,
                    18,
                    RoomComfort("office", 70.0, 74.0, cold_penalty=2.0, hot_penalty=2.0),
                ),
                ComfortScheduleEntry(
                    18,
                    8,
                    RoomComfort("office", 67.0, 76.0, cold_penalty=1.0, hot_penalty=0.5),
                ),
            ),
        ),
        ComfortSchedule(
            room="bedroom",
            entries=(
                # Wake-up: target 72°F
                ComfortScheduleEntry(
                    6,
                    9,
                    RoomComfort("bedroom", 70.0, 73.0, cold_penalty=2.0, hot_penalty=2.0),
                ),
                # Daytime: relaxed
                ComfortScheduleEntry(
                    9,
                    21,
                    RoomComfort("bedroom", 68.0, 72.0, cold_penalty=1.0, hot_penalty=1.5),
                ),
                # Night: cool for sleep
                ComfortScheduleEntry(
                    21,
                    6,
                    RoomComfort("bedroom", 66.0, 69.0, cold_penalty=1.0, hot_penalty=3.0),
                ),
            ),
        ),
        ComfortSchedule(
            room="family_room",
            entries=(ComfortScheduleEntry(0, 24, RoomComfort("family_room", 70.0, 74.0)),),
        ),
        ComfortSchedule(
            room="kitchen",
            entries=(
                ComfortScheduleEntry(
                    0, 24,
                    RoomComfort("kitchen", 69.0, 75.0, cold_penalty=1.0, hot_penalty=0.5),
                ),
            ),
        ),
        ComfortSchedule(
            room="piano",
            entries=(
                ComfortScheduleEntry(
                    0, 24,
                    RoomComfort("piano", 69.0, 75.0, cold_penalty=1.0, hot_penalty=0.5),
                ),
            ),
        ),
        ComfortSchedule(
            room="bathroom",
            entries=(
                # Bathroom has a window often open — relaxed bounds, low penalty
                ComfortScheduleEntry(
                    0, 24,
                    RoomComfort("bathroom", 67.0, 76.0, cold_penalty=0.5, hot_penalty=0.3),
                ),
            ),
        ),
    ]


# ── Zone mapping ──────────────────────────────────────────────────────────
# Maps rooms to HVAC zones. Used as fallback in compute_comfort_cost when
# a room-specific model prediction is unavailable.

ROOM_TO_ZONE: dict[str, str] = {
    "upstairs": "upstairs",
    "bedroom": "upstairs",
    "kitchen": "upstairs",
    "piano": "upstairs",
    "bathroom": "upstairs",
    "downstairs": "downstairs",
    "family_room": "downstairs",
    "office": "downstairs",
}


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


def _cautious_setpoint(current_temp: float, heating: bool) -> float:
    """Compute a cautious setpoint that achieves on/off without runaway risk.

    If the control loop is interrupted, the thermostat will gently coast
    instead of driving to an extreme temperature.
    """
    sp = current_temp + CAUTIOUS_OFFSET if heating else current_temp - CAUTIOUS_OFFSET
    return max(ABSOLUTE_MIN, min(ABSOLUTE_MAX, sp))


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


def sweep_scenarios(
    base_row: pd.DataFrame,
    feature_columns: list[str],
    models: dict[str, object],
    up_current: float,
    dn_current: float,
    current_split_temps: dict[str, float],
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> ControlDecision:
    """Sweep all HVAC combinations and return the best decision.

    Unified sweep over thermostats × blowers × mini-splits.
    """
    scenarios = generate_scenarios()
    best_cost = float("inf")
    best_decision: ControlDecision | None = None

    for scenario in scenarios:
        overrides = build_hvac_overrides(scenario, current_split_temps)
        X = base_row.copy()
        for col, val in overrides.items():
            if col in X.columns:
                X[col] = val

        # Predict at all control horizons
        predictions: dict[str, float] = {}
        for target, model in models.items():
            predictions[target] = float(model.predict(X)[0])  # type: ignore[union-attr]

        comfort = compute_comfort_cost(predictions, schedules, base_hour)
        energy = compute_energy_cost(scenario)
        total = comfort + energy

        if total < best_cost:
            best_cost = total
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
                    # Derive command target from room's comfort schedule midpoint
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
                upstairs_setpoint=_cautious_setpoint(up_current, scenario.upstairs_heating),
                downstairs_setpoint=_cautious_setpoint(dn_current, scenario.downstairs_heating),
                blowers=scenario.blowers,
                mini_splits=tuple(final_splits),
                total_cost=round(total, 4),
                comfort_cost=round(comfort, 4),
                energy_cost=round(energy, 4),
                room_predictions=room_preds,
            )

    if best_decision is None:
        raise RuntimeError("No HVAC scenarios evaluated")

    return best_decision


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

    # Check hold time
    prior_state = load_control_state()
    if should_hold(prior_state) and prior_state is not None:
        last = prior_state.last_decision_time
        print(f"[control] Holding current setpoints (last decision: {last})")
        up_sp = prior_state.upstairs_setpoint
        dn_sp = prior_state.downstairs_setpoint
        print(f"  Upstairs: {up_sp}°F, Downstairs: {dn_sp}°F")
        return None

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

    # Current hour for comfort schedule lookup
    base_hour = datetime.now(UTC).hour

    # Sweep all HVAC combinations
    schedules = default_comfort_schedules()
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

    # Per-room prediction table
    horizons = [HORIZON_LABELS[h] for h in CONTROL_HORIZONS]
    header = f"  {'Room':<14}" + "".join(f"{h:>8}" for h in horizons)
    print("\n  Predicted temperatures:")
    print(header)
    print(f"  {'-' * (14 + 8 * len(horizons))}")
    for room, preds in decision.room_predictions.items():
        vals = "".join(f"{preds.get(h, float('nan')):>7.1f}°" for h in horizons)
        print(f"  {room:<14}{vals}")

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
