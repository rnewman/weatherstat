"""Decision logging — records (context, action, prediction, outcome) per control cycle.

Stores decisions in SQLite. Outcomes are backfilled at the start of the next cycle
by reading collector snapshots that have arrived since the decision was made.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from weatherstat.config import DECISION_LOG_DB, PREDICTION_LABELS, SNAPSHOTS_DB

# Horizon durations in minutes, keyed by the label used in predictions
HORIZON_MINUTES: dict[str, int] = {"1h": 60, "2h": 120, "4h": 240, "6h": 360}

# Extra margin (minutes) before we consider a horizon checkable —
# one collector cycle so the snapshot definitely exists.
BACKFILL_MARGIN_MINUTES = 5

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS decisions (
    timestamp TEXT PRIMARY KEY,
    live INTEGER NOT NULL,
    outdoor_temp REAL,
    outdoor_humidity REAL,
    wind_speed REAL,
    weather_condition TEXT,
    current_temps TEXT NOT NULL,
    upstairs_heating INTEGER,
    downstairs_heating INTEGER,
    upstairs_setpoint REAL,
    downstairs_setpoint REAL,
    blowers TEXT,
    mini_splits TEXT,
    predictions TEXT NOT NULL,
    comfort_cost REAL,
    energy_cost REAL,
    total_cost REAL,
    comfort_bounds TEXT,
    trajectory TEXT,
    outcomes TEXT,
    actual_comfort_cost REAL,
    outcome_backfilled INTEGER DEFAULT 0
);
"""


def init_db(db_path: Path | None = None) -> Path:
    """Create the decision log database and table if they don't exist."""
    path = db_path or DECISION_LOG_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(_SCHEMA)
    # Schema migration: add trajectory column if upgrading from pre-PLAN-7
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE decisions ADD COLUMN trajectory TEXT")
    # Schema migration: add unified effector columns
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE decisions ADD COLUMN effector_decisions TEXT")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE decisions ADD COLUMN command_targets TEXT")
    conn.commit()
    conn.close()
    return path


def log_decision(
    decision: object,  # ControlDecision
    current_temps: dict[str, float],
    latest: object,  # pd.Series — latest snapshot row
    schedules: list[object],  # list[ComfortSchedule]
    base_hour: int,
    live: bool,
    db_path: Path | None = None,
) -> None:
    """Record a control decision to the decision log.

    Called after the sweep completes in run_control_cycle().
    """
    from weatherstat.config import EFFECTOR_MAP
    from weatherstat.types import ComfortSchedule, ControlDecision

    assert isinstance(decision, ControlDecision)
    path = init_db(db_path)

    # Extract outdoor conditions from latest snapshot row
    outdoor_temp = _safe_float(latest, "outdoor_temp")
    outdoor_humidity = _safe_float(latest, "outdoor_humidity")
    wind_speed = _safe_float(latest, "wind_speed")
    weather_condition = str(latest.get("weather_condition", "")) if hasattr(latest, "get") else ""

    # Backward compat: fill old columns from new data
    up_heating = any(e.mode != "off" for e in decision.effectors if e.name == "thermostat_upstairs")
    dn_heating = any(e.mode != "off" for e in decision.effectors if e.name == "thermostat_downstairs")
    up_setpoint = decision.command_targets.get("thermostat_upstairs", 0.0)
    dn_setpoint = decision.command_targets.get("thermostat_downstairs", 0.0)

    # Backward compat: serialize blower/mini-split decisions from unified effectors
    blowers_json = json.dumps([
        {"name": e.name.removeprefix("blower_"), "mode": e.mode}
        for e in decision.effectors
        if EFFECTOR_MAP.get(e.name) and EFFECTOR_MAP[e.name].control_type == "binary"
    ])
    mini_splits_json = json.dumps([
        {"name": e.name.removeprefix("mini_split_"), "mode": e.mode, "target": e.target}
        for e in decision.effectors
        if EFFECTOR_MAP.get(e.name) and EFFECTOR_MAP[e.name].control_type == "regulating"
    ])

    # New unified effector columns
    effector_decisions_json = json.dumps([
        {
            "name": e.name, "mode": e.mode, "target": e.target,
            "delay_steps": e.delay_steps, "duration_steps": e.duration_steps,
        }
        for e in decision.effectors
    ])
    command_targets_json = json.dumps(decision.command_targets)

    # Serialize predictions (label -> {horizon -> temp})
    predictions_json = json.dumps(decision.predictions)

    # Serialize comfort bounds at decision time
    comfort_bounds: dict[str, dict[str, float]] = {}
    for sched in schedules:
        if isinstance(sched, ComfortSchedule):
            c = sched.comfort_at(base_hour)
            if c is not None:
                comfort_bounds[sched.label] = {"min": c.min_temp, "max": c.max_temp}
    comfort_bounds_json = json.dumps(comfort_bounds)

    # Serialize trajectory info (if present)
    trajectory_json = json.dumps(decision.trajectory_info) if decision.trajectory_info else None

    conn = sqlite3.connect(str(path))
    conn.execute(
        """\
        INSERT OR REPLACE INTO decisions (
            timestamp, live, outdoor_temp, outdoor_humidity, wind_speed, weather_condition,
            current_temps, upstairs_heating, downstairs_heating, upstairs_setpoint, downstairs_setpoint,
            blowers, mini_splits, predictions, comfort_cost, energy_cost, total_cost,
            comfort_bounds, trajectory, outcome_backfilled, effector_decisions, command_targets
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (
            decision.timestamp,
            int(live),
            outdoor_temp,
            outdoor_humidity,
            wind_speed,
            weather_condition,
            json.dumps(current_temps),
            int(up_heating),
            int(dn_heating),
            up_setpoint,
            dn_setpoint,
            blowers_json,
            mini_splits_json,
            predictions_json,
            decision.comfort_cost,
            decision.energy_cost,
            decision.total_cost,
            comfort_bounds_json,
            trajectory_json,
            effector_decisions_json,
            command_targets_json,
        ),
    )
    conn.commit()
    conn.close()


def backfill_outcomes(db_path: Path | None = None) -> int:
    """Backfill outcomes for past decisions using collector snapshots.

    For each un-backfilled decision, checks whether enough time has elapsed
    for each prediction horizon. Reads actual temperatures from the collector
    DB and computes prediction errors.

    Returns:
        Number of decisions updated.
    """
    path = db_path or DECISION_LOG_DB
    if not path.exists():
        return 0

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, predictions, comfort_bounds FROM decisions WHERE outcome_backfilled = 0"
    ).fetchall()

    if not rows:
        conn.close()
        return 0

    # Load collector snapshots for lookups
    if not SNAPSHOTS_DB.exists():
        conn.close()
        return 0

    snap_conn = sqlite3.connect(str(SNAPSHOTS_DB))
    snap_df = pd.read_sql("SELECT * FROM snapshots ORDER BY timestamp", snap_conn)
    snap_conn.close()

    if snap_df.empty:
        conn.close()
        return 0

    # Parse snapshot timestamps once
    snap_df["_ts"] = pd.to_datetime(snap_df["timestamp"], format="ISO8601", utc=True)

    # Room temp columns from config
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    room_temp_cols = cfg.room_temp_columns

    now = datetime.now(UTC)
    updated = 0

    for row in rows:
        decision_time = datetime.fromisoformat(row["timestamp"])
        if decision_time.tzinfo is None:
            decision_time = decision_time.replace(tzinfo=UTC)

        predictions = json.loads(row["predictions"])
        comfort_bounds = json.loads(row["comfort_bounds"]) if row["comfort_bounds"] else {}
        elapsed_minutes = (now - decision_time).total_seconds() / 60

        outcomes: dict[str, dict[str, dict[str, float | None]]] = {}
        any_filled = False
        all_fillable = True

        for horizon_label, horizon_min in HORIZON_MINUTES.items():
            required_elapsed = horizon_min + BACKFILL_MARGIN_MINUTES
            if elapsed_minutes < required_elapsed:
                all_fillable = False
                continue

            # Find the snapshot closest to decision_time + horizon
            target_time = decision_time + timedelta(minutes=horizon_min)
            diffs = (snap_df["_ts"] - target_time).abs()
            closest_idx = diffs.idxmin()
            closest_diff_min = diffs.iloc[closest_idx].total_seconds() / 60

            # Only use if within 10 minutes of the target
            if closest_diff_min > 10:
                continue

            snap_row = snap_df.iloc[closest_idx]

            for label in PREDICTION_LABELS:
                label_preds = predictions.get(label, {})
                predicted = label_preds.get(horizon_label)
                if predicted is None:
                    continue

                temp_col = room_temp_cols.get(label)
                if temp_col is None:
                    continue

                actual = snap_row.get(temp_col)
                if actual is None or (isinstance(actual, float) and pd.isna(actual)):
                    continue

                actual = float(actual)
                error = round(predicted - actual, 3)

                if label not in outcomes:
                    outcomes[label] = {}
                outcomes[label][horizon_label] = {
                    "predicted": round(predicted, 2),
                    "actual": round(actual, 2),
                    "error": error,
                }
                any_filled = True

        if not any_filled:
            # Check if this decision is old enough that we'll never get data
            # (more than 6h + margin and still nothing — mark as done to avoid re-checking)
            max_horizon = max(HORIZON_MINUTES.values()) + BACKFILL_MARGIN_MINUTES
            if elapsed_minutes > max_horizon + 60:
                conn.execute(
                    "UPDATE decisions SET outcome_backfilled = 1 WHERE timestamp = ?",
                    (row["timestamp"],),
                )
                updated += 1
            continue

        # Compute actual comfort cost from outcomes
        actual_comfort_cost = _compute_actual_comfort_cost(outcomes, comfort_bounds)

        # Mark as fully backfilled only if all horizons are filled (or enough time passed)
        fully_done = all_fillable or elapsed_minutes > max(HORIZON_MINUTES.values()) + BACKFILL_MARGIN_MINUTES + 60

        conn.execute(
            """\
            UPDATE decisions
            SET outcomes = ?, actual_comfort_cost = ?, outcome_backfilled = ?
            WHERE timestamp = ?
            """,
            (json.dumps(outcomes), actual_comfort_cost, int(fully_done), row["timestamp"]),
        )
        updated += 1

    conn.commit()
    conn.close()
    return updated


def _compute_actual_comfort_cost(
    outcomes: dict[str, dict[str, dict[str, float | None]]],
    comfort_bounds: dict[str, dict[str, float]],
) -> float:
    """Compute retroactive comfort cost from actual temperatures.

    Uses the same quadratic penalty as compute_comfort_cost but with actual temps.
    Uses default penalty weights (cold=2.0, hot=1.0) since we don't store
    per-room penalty weights in the decision log.
    """
    # Horizon weights matching control.py
    horizon_weights = {"1h": 1.0, "2h": 0.9, "4h": 0.7, "6h": 0.5}
    cost = 0.0

    for room, horizons in outcomes.items():
        bounds = comfort_bounds.get(room)
        if bounds is None:
            continue
        min_temp = bounds["min"]
        max_temp = bounds["max"]

        for h_label, data in horizons.items():
            actual = data.get("actual")
            if actual is None:
                continue
            weight = horizon_weights.get(h_label, 0.5)

            if actual < min_temp:
                cost += (min_temp - actual) ** 2 * 2.0 * weight  # cold_penalty=2.0
            elif actual > max_temp:
                cost += (actual - max_temp) ** 2 * 1.0 * weight  # hot_penalty=1.0

    return round(cost, 4)


def load_decision_log(db_path: Path | None = None, limit: int = 100) -> pd.DataFrame:
    """Load recent decisions as a DataFrame for analysis.

    Args:
        db_path: Path to decision_log.db. Defaults to DECISION_LOG_DB.
        limit: Maximum number of rows to return (most recent first).

    Returns:
        DataFrame with all columns from the decisions table.
    """
    path = db_path or DECISION_LOG_DB
    if not path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(path))
    df = pd.read_sql(
        f"SELECT * FROM decisions ORDER BY timestamp DESC LIMIT {limit}",
        conn,
    )
    conn.close()
    return df


def _safe_float(row: object, col: str) -> float | None:
    """Extract a float from a pandas Series row, returning None for NaN/missing."""
    import numpy as np

    val = row.get(col) if hasattr(row, "get") else None  # type: ignore[union-attr]
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    return float(val)
