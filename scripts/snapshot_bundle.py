#!/usr/bin/env python3
"""Create a point-in-time debug bundle for replaying control decisions.

Usage:
    python scripts/snapshot_bundle.py                          # bundle at current time
    python scripts/snapshot_bundle.py 2026-03-30T10:31:04+00:00  # bundle at historical time
    python scripts/snapshot_bundle.py --replay <bundle_dir>      # replay a bundle

A bundle captures everything needed to deterministically replay a control cycle:
  - Snapshot data (14 hours of readings before the target time)
  - Thermal parameters (thermal_params.json)
  - Control state (control_state.json — prev_state for mode holds)
  - Config (weatherstat.yaml)
  - Decision log entries around the target time
  - Active profile + effector eligibility (if live)
  - Metadata (target timestamp, local hour, outdoor temp)

Bundles are self-contained directories under ~/.weatherstat/bundles/.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

DATA_DIR = Path.home() / ".weatherstat"
BUNDLES_DIR = DATA_DIR / "bundles"
SNAPSHOTS_DB = DATA_DIR / "snapshots" / "snapshots.db"
THERMAL_PARAMS = DATA_DIR / "thermal_params.json"
CONTROL_STATE = DATA_DIR / "control_state.json"
DECISION_LOG = DATA_DIR / "decision_log.db"
CONFIG_FILE = DATA_DIR / "weatherstat.yaml"


def _find_nearest_snapshot(timestamp: str) -> str | None:
    """Find the snapshot timestamp closest to (but not after) the target."""
    conn = sqlite3.connect(str(SNAPSHOTS_DB))
    row = conn.execute(
        "SELECT MAX(timestamp) FROM readings WHERE timestamp <= ?",
        (timestamp,),
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _find_nearest_decision(timestamp: str) -> tuple[str, dict] | None:
    """Find the decision closest to the target time."""
    if not DECISION_LOG.exists():
        return None
    conn = sqlite3.connect(str(DECISION_LOG))
    row = conn.execute(
        """SELECT timestamp, current_temps, outdoor_temp, weather_condition,
                  effector_decisions, comfort_bounds, comfort_cost, energy_cost,
                  active_profile, mrt_offsets, blocked, trajectory, command_targets
           FROM decisions
           WHERE timestamp <= ?
           ORDER BY timestamp DESC LIMIT 1""",
        (timestamp,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return row[0], {
        "timestamp": row[0],
        "current_temps": json.loads(row[1]) if row[1] else {},
        "outdoor_temp": row[2],
        "weather_condition": row[3],
        "effector_decisions": json.loads(row[4]) if row[4] else [],
        "comfort_bounds": json.loads(row[5]) if row[5] else {},
        "comfort_cost": row[6],
        "energy_cost": row[7],
        "active_profile": row[8],
        "mrt_offsets": json.loads(row[9]) if row[9] else {},
        "blocked": json.loads(row[10]) if row[10] else {},
        "trajectory": json.loads(row[11]) if row[11] else {},
        "command_targets": json.loads(row[12]) if row[12] else {},
    }


def _reconstruct_control_state(timestamp: str) -> dict | None:
    """Reconstruct control_state.json from the decision before the target."""
    if not DECISION_LOG.exists():
        return None
    conn = sqlite3.connect(str(DECISION_LOG))
    row = conn.execute(
        """SELECT timestamp, effector_decisions, command_targets
           FROM decisions WHERE timestamp < ?
           ORDER BY timestamp DESC LIMIT 1""",
        (timestamp,),
    ).fetchone()
    conn.close()
    if not row:
        return None

    prev_ts = row[0]
    effs = json.loads(row[1]) if row[1] else []
    command_targets = json.loads(row[2]) if row[2] else {}

    modes: dict[str, str] = {}
    setpoints: dict[str, float] = {}
    for e in effs:
        name = e.get("name", "")
        mode = e.get("mode", "off")
        target = e.get("target")
        modes[name] = mode
        if target is not None:
            setpoints[name] = float(target)
        elif name in command_targets:
            setpoints[name] = float(command_targets[name])

    return {
        "last_decision_time": prev_ts,
        "setpoints": setpoints,
        "modes": modes,
        "mode_times": {},
    }


def create_bundle(target_time: str | None = None) -> Path:
    """Create a debug bundle at the given time (or now)."""
    if target_time is None:
        target_time = datetime.now(UTC).isoformat()

    # Normalize timestamp for directory name
    safe_ts = target_time.replace(":", "").replace("+", "p").replace("-", "")[:20]
    bundle_dir = BUNDLES_DIR / f"bundle_{safe_ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    snap_ts = _find_nearest_snapshot(target_time)
    print(f"Target time:     {target_time}")
    print(f"Nearest snapshot: {snap_ts}")

    # 1. Extract snapshot data (14 hours before target)
    cutoff = (datetime.fromisoformat(target_time.replace("Z", "+00:00")) - timedelta(hours=14)).isoformat()
    bundle_db = bundle_dir / "snapshots.db"
    conn_src = sqlite3.connect(str(SNAPSHOTS_DB))
    conn_dst = sqlite3.connect(str(bundle_db))
    conn_dst.execute("CREATE TABLE IF NOT EXISTS readings (timestamp TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (timestamp, name))")
    rows = conn_src.execute(
        "SELECT timestamp, name, value FROM readings WHERE timestamp >= ? AND timestamp <= ?",
        (cutoff, target_time),
    ).fetchall()
    conn_dst.executemany("INSERT OR IGNORE INTO readings VALUES (?, ?, ?)", rows)
    conn_dst.commit()
    n_timestamps = conn_dst.execute("SELECT COUNT(DISTINCT timestamp) FROM readings").fetchone()[0]
    conn_src.close()
    conn_dst.close()
    print(f"Snapshots:       {len(rows)} rows ({n_timestamps} timestamps, 14h window)")

    # 2. Copy thermal params
    if THERMAL_PARAMS.exists():
        shutil.copy2(THERMAL_PARAMS, bundle_dir / "thermal_params.json")
        print("Thermal params:  copied")

    # 3. Reconstruct control state at target time
    cs = _reconstruct_control_state(target_time)
    if cs:
        with open(bundle_dir / "control_state.json", "w") as f:
            json.dump(cs, f, indent=2)
        print(f"Control state:   reconstructed from {cs['last_decision_time']}")

    # 4. Copy config
    if CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE, bundle_dir / "weatherstat.yaml")
        print("Config:          copied")

    # 5. Extract nearby decisions
    if DECISION_LOG.exists():
        conn = sqlite3.connect(str(DECISION_LOG))
        decisions = []
        # Get 5 decisions before and 2 after target time
        before = conn.execute(
            """SELECT timestamp, live, comfort_cost, energy_cost,
                      current_temps, effector_decisions, comfort_bounds,
                      active_profile, mrt_offsets, blocked, trajectory,
                      command_targets, predictions, outcomes,
                      actual_comfort_cost, outcome_backfilled
               FROM decisions WHERE timestamp <= ?
               ORDER BY timestamp DESC LIMIT 5""",
            (target_time,),
        ).fetchall()
        after = conn.execute(
            """SELECT timestamp, live, comfort_cost, energy_cost,
                      current_temps, effector_decisions, comfort_bounds,
                      active_profile, mrt_offsets, blocked, trajectory,
                      command_targets, predictions, outcomes,
                      actual_comfort_cost, outcome_backfilled
               FROM decisions WHERE timestamp > ?
               ORDER BY timestamp ASC LIMIT 2""",
            (target_time,),
        ).fetchall()
        conn.close()

        cols = [
            "timestamp", "live", "comfort_cost", "energy_cost",
            "current_temps", "effector_decisions", "comfort_bounds",
            "active_profile", "mrt_offsets", "blocked", "trajectory",
            "command_targets", "predictions", "outcomes",
            "actual_comfort_cost", "outcome_backfilled",
        ]
        for row in reversed(before):
            decisions.append({c: _try_json(v) if isinstance(v, str) and c not in ("timestamp", "active_profile") else v for c, v in zip(cols, row, strict=True)})
        for row in after:
            decisions.append({c: _try_json(v) if isinstance(v, str) and c not in ("timestamp", "active_profile") else v for c, v in zip(cols, row, strict=True)})

        with open(bundle_dir / "decisions.json", "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"Decisions:       {len(decisions)} ({len(before)} before, {len(after)} after)")

    # 6. Write metadata
    decision_info = _find_nearest_decision(target_time)
    meta = {
        "target_time": target_time,
        "nearest_snapshot": snap_ts,
        "created_at": datetime.now(UTC).isoformat(),
        "bundle_dir": str(bundle_dir),
    }
    if decision_info:
        dec_ts, dec_data = decision_info
        meta["nearest_decision"] = dec_ts
        meta["outdoor_temp"] = dec_data.get("outdoor_temp")
        meta["active_profile"] = dec_data.get("active_profile")
        meta["comfort_cost"] = dec_data.get("comfort_cost")
        meta["energy_cost"] = dec_data.get("energy_cost")
        meta["effectors"] = [
            {"name": e.get("name"), "mode": e.get("mode"), "target": e.get("target")}
            for e in dec_data.get("effector_decisions", [])
        ]
    with open(bundle_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBundle created: {bundle_dir}")
    return bundle_dir


def _try_json(s: str) -> object:
    """Try to parse as JSON, return original string on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def replay_bundle(bundle_dir: Path) -> None:
    """Replay a control cycle from a bundle.

    Loads all state from the bundle's SQLite and config files,
    then runs sweep_scenarios_physics directly (no HA calls).
    """
    import os

    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        print(f"Not a valid bundle: {bundle_dir} (no metadata.json)")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    target_time = meta["target_time"]
    print(f"Replaying bundle: {bundle_dir}")
    print(f"Target time: {target_time}")

    # Point weatherstat at the bundle BEFORE importing weatherstat modules
    os.environ["WEATHERSTAT_DATA_DIR"] = str(bundle_dir)

    # Ensure snapshots.db is in the expected subdirectory
    snap_subdir = bundle_dir / "snapshots"
    snap_subdir.mkdir(exist_ok=True)
    if (bundle_dir / "snapshots.db").exists() and not (snap_subdir / "snapshots.db").exists():
        shutil.copy2(bundle_dir / "snapshots.db", snap_subdir / "snapshots.db")

    # Now import after setting env var
    from zoneinfo import ZoneInfo

    from weatherstat.config import PREDICTION_SENSORS, SENSOR_LABELS, TIMEZONE
    from weatherstat.control import (
        adjust_schedules_for_windows,
        apply_comfort_profile,
        apply_mrt_correction,
        default_comfort_schedules,
        sweep_scenarios_physics,
    )
    from weatherstat.simulator import extract_recent_history, load_sim_params
    from weatherstat.types import ControlState
    from weatherstat.weather import condition_to_solar_fraction
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    sim_params = load_sim_params()

    # Load data from bundle's snapshot DB
    snap_db = snap_subdir / "snapshots.db"
    conn = sqlite3.connect(str(snap_db))

    # Find the snapshot closest to target time
    row = conn.execute("SELECT MAX(timestamp) FROM readings WHERE timestamp <= ?", (target_time,)).fetchone()
    snap_ts = row[0] if row else None
    if not snap_ts:
        print("No snapshot data in bundle.")
        return
    print(f"Nearest snapshot: {snap_ts}")

    # Load latest snapshot values
    snap_rows = conn.execute("SELECT name, value FROM readings WHERE timestamp = ?", (snap_ts,)).fetchall()
    snap = dict(snap_rows)

    # Build current temps
    current_temps: dict[str, float] = {}
    for sensor in PREDICTION_SENSORS:
        if sensor in snap:
            with contextlib.suppress(ValueError, TypeError):
                current_temps[sensor] = float(snap[sensor])

    # Outdoor temp
    outdoor_sensor = cfg.outdoor_sensor
    out_temp_raw = snap.get(outdoor_sensor) if outdoor_sensor else snap.get("met_outdoor_temp")
    out_temp = float(out_temp_raw) if out_temp_raw else 50.0

    # Window states
    window_states: dict[str, bool] = {}
    for wname in cfg.windows:
        window_states[wname] = snap.get(f"window_{wname}_open", "0") == "1"

    # Forecast temps and solar fractions
    forecast_temps: list[float] = []
    solar_fractions: list[float] = []
    current_condition = snap.get("weather_condition", "unknown")
    solar_fractions.append(condition_to_solar_fraction(current_condition))
    for h in range(1, 13):
        ft = snap.get(f"forecast_temp_{h}h")
        fc = snap.get(f"forecast_condition_{h}h", "unknown")
        if ft is not None:
            try:
                forecast_temps.append(float(ft))
                solar_fractions.append(condition_to_solar_fraction(fc))
            except (ValueError, TypeError):
                break
        else:
            break

    # Compute local hour from target time
    target_dt = datetime.fromisoformat(target_time.replace("Z", "+00:00"))
    local_dt = target_dt.astimezone(ZoneInfo(TIMEZONE))
    base_hour = local_dt.hour
    fractional_hour = base_hour + local_dt.minute / 60.0

    # Build recent history from bundle's snapshot DB
    import pandas as pd

    # Load all readings into a DataFrame for extract_recent_history
    all_rows = conn.execute(
        "SELECT timestamp, name, value FROM readings ORDER BY timestamp",
    ).fetchall()
    conn.close()

    if all_rows:
        df_eav = pd.DataFrame(all_rows, columns=["timestamp", "name", "value"])
        df_eav["timestamp"] = pd.to_datetime(df_eav["timestamp"], format="ISO8601", utc=True)
        # Pivot to wide format
        df_wide = df_eav.pivot(index="timestamp", columns="name", values="value")
        df_wide.index.name = "timestamp"
        # Convert numeric columns
        for col in df_wide.columns:
            df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce").fillna(df_wide[col])
        recent_hist = extract_recent_history(df_wide, sim_params)
    else:
        recent_hist = {}

    # Reconstruct prev_state from control_state.json in bundle
    prev_state: ControlState | None = None
    cs_path = bundle_dir / "control_state.json"
    if cs_path.exists():
        with open(cs_path) as f:
            cs = json.load(f)
        prev_state = ControlState(
            last_decision_time=cs.get("last_decision_time", ""),
            setpoints=cs.get("setpoints", {}),
            modes=cs.get("modes", {}),
            mode_times=cs.get("mode_times", {}),
        )

    # Outdoor temp for simulator: prefer forecast
    outdoor = forecast_temps[0] if forecast_temps else out_temp

    # Build comfort schedules with profile + MRT adjustments
    schedules = default_comfort_schedules()

    # Apply comfort profile from decision metadata
    profile_name = meta.get("active_profile")
    active_profile = None
    if profile_name and cfg.comfort_profiles:
        active_profile = cfg.comfort_profiles.get(profile_name)
    schedules = apply_comfort_profile(schedules, active_profile)

    # MRT correction
    mrt_weights = {
        c.sensor: (c.mrt_weight if c.mrt_weight != 1.0 else sim_params.mrt_weights.get(c.sensor, 1.0))
        for c in cfg.constraints
    }
    schedules, mrt_offset = apply_mrt_correction(schedules, outdoor, cfg.mrt_correction, mrt_weights)

    # Window adjustments
    constraint_labels = {c.label for c in cfg.constraints}
    schedules = adjust_schedules_for_windows(schedules, window_states, constraint_labels, *cfg.window_open_offset)

    print(f"\nOutdoor: {outdoor:.1f}°F  Weather: {current_condition}  Local hour: {base_hour}")
    print(f"Profile: {profile_name or 'default'}  MRT offset: {mrt_offset:+.1f}°F")
    open_windows = [k for k, v in sorted(window_states.items()) if v]
    print(f"Windows: {', '.join(open_windows) if open_windows else 'all closed'}")
    if prev_state:
        print(f"Prev state: {prev_state.modes}")
    print()

    print("Current temps:")
    for s in sorted(current_temps):
        label = SENSOR_LABELS.get(s, s)
        print(f"  {label:<20} {current_temps[s]:6.1f}°F")
    print()

    # Run sweep
    print("=" * 60)
    print("SWEEP RESULT")
    print("=" * 60)
    decision, scenario, blocked = sweep_scenarios_physics(
        current_temps=current_temps,
        outdoor_temp=outdoor,
        forecast_temps=forecast_temps,
        window_states=window_states,
        sim_params=sim_params,
        hour_of_day=fractional_hour,
        recent_history=recent_hist,
        schedules=schedules,
        base_hour=base_hour,
        prev_state=prev_state,
        solar_fractions=solar_fractions,
    )

    print(f"  Comfort: {decision.comfort_cost:.1f}  Energy: {decision.energy_cost:.3f}")
    print("  Effectors:")
    for e in decision.effectors:
        t = f" @ {e.target}" if e.target is not None else ""
        print(f"    {e.name}: {e.mode}{t}")
    if blocked:
        print(f"  Blocked: {blocked}")
    print()

    print("  Predictions:")
    for sensor in sorted(decision.predictions):
        p = decision.predictions[sensor]
        if "1h" in p:
            print(f"    {sensor:<28} 1h={p['1h']:6.1f}  2h={p['2h']:6.1f}  4h={p['4h']:6.1f}  6h={p['6h']:6.1f}")
    print()

    # Compare with logged decision
    decisions_path = bundle_dir / "decisions.json"
    if decisions_path.exists():
        with open(decisions_path) as f:
            logged = json.load(f)
        # Find the decision closest to target time
        closest = min(logged, key=lambda d: abs(datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")).timestamp() - target_dt.timestamp()))
        print("=" * 60)
        print(f"LOGGED DECISION ({closest['timestamp']})")
        print("=" * 60)
        print(f"  Comfort: {closest.get('comfort_cost', '?')}  Energy: {closest.get('energy_cost', '?')}")
        if closest.get("effector_decisions"):
            print("  Effectors:")
            for e in closest["effector_decisions"]:
                t = f" @ {e.get('target')}" if e.get("target") is not None else ""
                print(f"    {e['name']}: {e.get('mode', '?')}{t}")

        # Show cost difference
        logged_cc = closest.get("comfort_cost", 0)
        if isinstance(logged_cc, (int, float)):
            delta = decision.comfort_cost - logged_cc
            print(f"\n  Replay vs logged comfort cost delta: {delta:+.1f}")
            if abs(delta) > 10:
                print("  (Large delta likely due to missing recent_history or different profile/MRT state)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timestamp", nargs="?", help="Target timestamp (ISO 8601). Default: now.")
    parser.add_argument("--replay", metavar="BUNDLE_DIR", help="Replay a previously created bundle.")
    parser.add_argument("--list", action="store_true", help="List existing bundles.")
    args = parser.parse_args()

    if args.list:
        if not BUNDLES_DIR.exists():
            print("No bundles directory.")
            return
        for d in sorted(BUNDLES_DIR.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                with open(d / "metadata.json") as f:
                    m = json.load(f)
                cost = m.get("comfort_cost", "?")
                profile = m.get("active_profile", "?")
                print(f"  {d.name}  target={m.get('target_time', '?')}  cost={cost}  profile={profile}")
        return

    if args.replay:
        replay_bundle(Path(args.replay))
        return

    create_bundle(args.timestamp)


if __name__ == "__main__":
    main()
