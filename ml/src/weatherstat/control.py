"""Control policy — comfort-optimizing setpoint selection.

Receding-horizon controller: "what setpoints maximize comfort over the next 6 hours?"
Re-evaluated every 15 minutes.

Two thermostats (upstairs, downstairs) are the only control variables.
The cost function evaluates comfort across all rooms even though it can only
steer two setpoints — e.g. "setting downstairs to 72 keeps the office comfortable
but makes the bedroom too warm" is a tradeoff the optimizer reasons about.

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
    CONTROL_STATE_FILE,
    HORIZONS_5MIN,
    PREDICTION_ROOMS,
    PREDICTIONS_DIR,
)
from weatherstat.inference import (
    _build_setpoint_overrides,
    _prepare_feature_row,
    build_features,
    fetch_recent_history,
    load_feature_columns,
    load_models,
)
from weatherstat.types import (
    ComfortSchedule,
    ComfortScheduleEntry,
    ControlDecision,
    ControlState,
    RoomComfort,
)

# ── Constants ──────────────────────────────────────────────────────────────

# Setpoint sweep range (°F)
SETPOINT_MIN = 67
SETPOINT_MAX = 76
SETPOINT_STEP = 1

# Absolute safety bounds
ABSOLUTE_MIN = 65
ABSOLUTE_MAX = 78

# Minimum hold time before changing setpoints (seconds)
MIN_HOLD_SECONDS = 30 * 60  # 30 minutes

# Maximum data staleness before refusing to execute (seconds)
MAX_STALE_SECONDS = 15 * 60  # 15 minutes

# Maximum predicted 1h temperature change before logging warning
MAX_1H_CHANGE = 5.0  # °F

# Energy penalty coefficient
ENERGY_PENALTY = 0.01

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


def compute_energy_cost(
    up_setpoint: float,
    dn_setpoint: float,
    up_current: float,
    dn_current: float,
) -> float:
    """Small energy penalty to prefer lower setpoints when comfort is equal."""
    cost = ENERGY_PENALTY * max(0.0, up_setpoint - up_current)
    cost += ENERGY_PENALTY * max(0.0, dn_setpoint - dn_current)
    return cost


# ── Setpoint search ───────────────────────────────────────────────────────


def sweep_setpoints(
    base_row: pd.DataFrame,
    feature_columns: list[str],
    models: dict[str, object],
    up_current: float,
    dn_current: float,
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> ControlDecision:
    """Sweep all setpoint pairs and return the best decision.

    Args:
        base_row: Single-row DataFrame of current features.
        feature_columns: Feature column names expected by models.
        models: Loaded LightGBM models keyed by target name.
        up_current: Current upstairs temperature.
        dn_current: Current downstairs temperature.
        schedules: Comfort schedules.
        base_hour: Current hour of day.

    Returns:
        ControlDecision with the best setpoint pair.
    """
    best_cost = float("inf")
    best_decision: ControlDecision | None = None

    setpoints = list(range(SETPOINT_MIN, SETPOINT_MAX + 1, SETPOINT_STEP))

    for up_sp in setpoints:
        for dn_sp in setpoints:
            overrides = _build_setpoint_overrides(up_current, dn_current, float(up_sp), float(dn_sp))
            X = base_row.copy()
            for col, val in overrides.items():
                if col in X.columns:
                    X[col] = val

            # Predict at all horizons
            predictions: dict[str, float] = {}
            for target, model in models.items():
                predictions[target] = float(model.predict(X)[0])  # type: ignore[union-attr]

            comfort = compute_comfort_cost(predictions, schedules, base_hour)
            energy = compute_energy_cost(float(up_sp), float(dn_sp), up_current, dn_current)
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
                best_decision = ControlDecision(
                    timestamp=datetime.now(UTC).isoformat(),
                    upstairs_setpoint=float(up_sp),
                    downstairs_setpoint=float(dn_sp),
                    total_cost=round(total, 4),
                    comfort_cost=round(comfort, 4),
                    energy_cost=round(energy, 4),
                    room_predictions=room_preds,
                )

    if best_decision is None:
        # Should not happen — setpoints list is non-empty
        raise RuntimeError("No setpoint pairs evaluated")

    return best_decision


# ── State persistence ─────────────────────────────────────────────────────


def load_control_state() -> ControlState | None:
    """Load persisted control state, or None if not found."""
    if not CONTROL_STATE_FILE.exists():
        return None
    try:
        data = json.loads(CONTROL_STATE_FILE.read_text())
        return ControlState(
            last_decision_time=data["last_decision_time"],
            upstairs_setpoint=data["upstairs_setpoint"],
            downstairs_setpoint=data["downstairs_setpoint"],
        )
    except (json.JSONDecodeError, KeyError):
        return None


def save_control_state(decision: ControlDecision) -> None:
    """Persist control state to prevent rapid cycling."""
    state = {
        "last_decision_time": decision.timestamp,
        "upstairs_setpoint": decision.upstairs_setpoint,
        "downstairs_setpoint": decision.downstairs_setpoint,
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


def write_command_json(
    decision: ControlDecision,
    current_state: dict[str, object],
) -> Path:
    """Write executor-compatible command JSON.

    Uses camelCase keys matching the TS Prediction interface.
    Mini split and blower values are passed through from current state.
    """
    command = {
        "timestamp": decision.timestamp,
        "thermostatUpstairsTarget": decision.upstairs_setpoint,
        "thermostatDownstairsTarget": decision.downstairs_setpoint,
        # Pass through current mini split state (not controlled yet)
        "miniSplitBedroomTarget": current_state.get("mini_split_bedroom_target", 72),
        "miniSplitBedroomMode": current_state.get("mini_split_bedroom_mode", "off"),
        "miniSplitLivingRoomTarget": current_state.get("mini_split_living_room_target", 72),
        "miniSplitLivingRoomMode": current_state.get("mini_split_living_room_mode", "off"),
        # Pass through current blower state
        "blowerFamilyRoomMode": current_state.get("blower_family_room_mode", "off"),
        "blowerOfficeMode": current_state.get("blower_office_mode", "off"),
        "confidence": 1.0 - min(decision.total_cost / 10.0, 0.9),
    }

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

    # Sweep setpoints
    schedules = default_comfort_schedules()
    print("\n[control] Sweeping setpoints...")
    t0 = time.monotonic()
    decision = sweep_setpoints(
        base_row,
        feature_columns,
        models,
        up_current,
        dn_current,
        schedules,
        base_hour,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    print(f"  Sweep completed in {elapsed_ms:.1f}ms")

    # Print decision
    print("\n[control] Decision:")
    print(f"  Upstairs setpoint:  {decision.upstairs_setpoint:.0f}°F")
    print(f"  Downstairs setpoint: {decision.downstairs_setpoint:.0f}°F")
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
    if not sane:
        print("  Prediction sanity check FAILED — skipping execution")
        if not live:
            print("  (would skip in live mode; writing command anyway for inspection)")

    # Clamp to absolute bounds
    up_sp = max(ABSOLUTE_MIN, min(ABSOLUTE_MAX, decision.upstairs_setpoint))
    dn_sp = max(ABSOLUTE_MIN, min(ABSOLUTE_MAX, decision.downstairs_setpoint))
    if up_sp != decision.upstairs_setpoint or dn_sp != decision.downstairs_setpoint:
        print(f"  Clamped setpoints: up={up_sp}°F, dn={dn_sp}°F")
        # Rebuild with clamped values
        decision = ControlDecision(
            timestamp=decision.timestamp,
            upstairs_setpoint=up_sp,
            downstairs_setpoint=dn_sp,
            total_cost=decision.total_cost,
            comfort_cost=decision.comfort_cost,
            energy_cost=decision.energy_cost,
            room_predictions=decision.room_predictions,
            dry_run=not live,
        )

    # Extract current pass-through state for command JSON
    current_state: dict[str, object] = {
        "mini_split_bedroom_target": latest.get("mini_split_bedroom_target", 72),
        "mini_split_bedroom_mode": str(latest.get("mini_split_bedroom_mode", "off")),
        "mini_split_living_room_target": latest.get("mini_split_living_room_target", 72),
        "mini_split_living_room_mode": str(latest.get("mini_split_living_room_mode", "off")),
        "blower_family_room_mode": str(latest.get("blower_family_room_mode", "off")),
        "blower_office_mode": str(latest.get("blower_office_mode", "off")),
    }

    # Write command JSON
    cmd_path = write_command_json(decision, current_state)
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
