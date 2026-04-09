#!/usr/bin/env python3
"""Replay an environment opportunity evaluation at a given decision timestamp.

Usage:
    python scripts/replay_opportunity.py <timestamp> <env_name>
    python scripts/replay_opportunity.py 2026-03-30T10:31:04.928113+00:00 bathroom

Reconstructs the system state from the decision log and snapshot DB,
then runs both the original sweep and the re-sweep with the environment factor
toggled, showing the full breakdown of why the opportunity was (or wasn't) generated.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from pathlib import Path

from weatherstat.control import (
    CONTROL_HORIZONS,
    HORIZON_WEIGHTS,
    default_comfort_schedules,
    sweep_scenarios_physics,
)
from weatherstat.simulator import load_sim_params
from weatherstat.types import ControlState
from weatherstat.yaml_config import load_config

DATA_DIR = Path.home() / ".weatherstat"
DECISION_LOG = DATA_DIR / "decision_log.db"
SNAPSHOTS_DB = DATA_DIR / "snapshots" / "snapshots.db"


def _load_decision_state(timestamp: str) -> tuple[dict[str, float], float, str]:
    """Load current_temps, outdoor_temp, weather from a decision."""
    conn = sqlite3.connect(str(DECISION_LOG))
    row = conn.execute(
        "SELECT current_temps, outdoor_temp, weather_condition FROM decisions WHERE timestamp = ?",
        (timestamp,),
    ).fetchone()
    conn.close()
    if not row:
        raise ValueError(f"No decision found at {timestamp}")
    return json.loads(row[0]), row[1], row[2] or "unknown"


def _load_snapshot_context(timestamp: str) -> tuple[dict[str, bool], list[float], list[float]]:
    """Load environment states and forecast temps from the nearest snapshot.

    Decision timestamps end in fractional seconds; snapshot timestamps are on
    the minute with '.000Z' suffix. We find the closest snapshot <= decision time.
    """
    cfg = load_config()

    # Build reverse map: column_name -> config_name
    col_to_name: dict[str, str] = {ecfg.column: ename for ename, ecfg in cfg.environment.items()}

    conn = sqlite3.connect(str(SNAPSHOTS_DB))

    # Find closest snapshot <= decision time
    row = conn.execute(
        "SELECT MAX(timestamp) FROM readings WHERE timestamp <= ?",
        (timestamp,),
    ).fetchone()
    snap_ts = row[0] if row else None
    if not snap_ts:
        conn.close()
        raise ValueError(f"No snapshot found at or before {timestamp}")

    rows = conn.execute(
        "SELECT name, value FROM readings WHERE timestamp = ?",
        (snap_ts,),
    ).fetchall()
    conn.close()

    snap = dict(rows)

    environment_states: dict[str, bool] = {}
    for name, val in rows:
        if name in col_to_name:
            environment_states[col_to_name[name]] = val == "1"

    from weatherstat.weather import condition_to_solar_fraction

    forecast_dict: dict[int, float] = {}
    for h in range(1, 13):
        key = f"forecast_temp_{h}h"
        if key in snap:
            with contextlib.suppress(ValueError, TypeError):
                forecast_dict[h] = float(snap[key])

    # Convert to list: forecast_temps[0] = h+1, [1] = h+2, etc.
    forecast_temps: list[float] = []
    for h in range(1, 13):
        if h in forecast_dict:
            forecast_temps.append(forecast_dict[h])
        else:
            break  # stop at first gap

    # Solar fractions from weather conditions
    current_condition = snap.get("weather_condition", "unknown")
    solar_fractions: list[float] = [condition_to_solar_fraction(str(current_condition))]
    for h in range(1, 13):
        fc = snap.get(f"forecast_condition_{h}h", "unknown")
        solar_fractions.append(condition_to_solar_fraction(str(fc)))

    return environment_states, forecast_temps, solar_fractions


def _reconstruct_prev_state(timestamp: str) -> ControlState | None:
    """Reconstruct ControlState from the decision preceding the given timestamp."""
    conn = sqlite3.connect(str(DECISION_LOG))
    row = conn.execute(
        """SELECT timestamp, effector_decisions, command_targets
           FROM decisions WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1""",
        (timestamp,),
    ).fetchone()
    conn.close()
    if not row:
        return None

    prev_ts, eff_json, ct_json = row
    effs = json.loads(eff_json) if eff_json else []
    command_targets = json.loads(ct_json) if ct_json else {}

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

    return ControlState(
        last_decision_time=prev_ts,
        setpoints=setpoints,
        modes=modes,
        mode_times={},  # not tracked in decision log
    )


def _utc_to_local_hour(timestamp: str, tz_offset_hours: int = -7) -> int:
    """Extract hour from ISO timestamp, adjusted for local timezone."""
    # Parse hour from ISO timestamp
    hour_utc = int(timestamp[11:13])
    return (hour_utc + tz_offset_hours) % 24


def replay(timestamp: str, env_name: str) -> None:
    """Replay opportunity evaluation."""
    from datetime import UTC, datetime, timedelta

    from weatherstat.weather import solar_sin_elevation

    cfg = load_config()
    sim_params = load_sim_params()
    schedules = default_comfort_schedules()

    print(f"Replaying opportunity: {env_name} at {timestamp}")
    print()

    # Load state
    current_temps, outdoor_temp, weather = _load_decision_state(timestamp)
    environment_states, forecast_temps, solar_fractions = _load_snapshot_context(timestamp)
    base_hour = _utc_to_local_hour(timestamp)

    # Compute solar elevations at the decision timestamp
    dt = datetime.fromisoformat(timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    max_steps = max(CONTROL_HORIZONS)
    solar_elevations = [
        solar_sin_elevation(cfg.location.latitude, cfg.location.longitude, dt + timedelta(minutes=5 * step))
        for step in range(1, max_steps + 1)
    ]

    print(f"Outdoor: {outdoor_temp:.1f}°F  Weather: {weather}  Local hour: {base_hour}")
    active_env = [k for k, v in sorted(environment_states.items()) if v]
    print(f"Active environment: {', '.join(active_env) if active_env else 'none'}")
    print(f"Forecast: {', '.join(f'{i+1}h={t:.0f}' for i, t in enumerate(forecast_temps))}")
    print()

    # Key temps
    print("Current temps:")
    for k in sorted(current_temps):
        if "_temp" in k:
            print(f"  {k:<30} {current_temps[k]:6.1f}")
    print()

    # Check environment state
    is_active = environment_states.get(env_name, False)
    action = "Activate" if not is_active else "Deactivate"
    print(f"'{env_name}' is currently {'ACTIVE' if is_active else 'DEFAULT'} → proposing to {action}")
    print()

    # Reconstruct prev_state from the preceding decision
    prev_state = _reconstruct_prev_state(timestamp)
    if prev_state:
        print(f"Prev state (for mode holds): {prev_state.modes}")
        print()

    # Recent history: empty dict (we don't have the exact history from the decision)
    recent_history: dict[str, list[float]] = {}

    # Original sweep
    print("=" * 60)
    print("ORIGINAL SWEEP (current environment states)")
    print("=" * 60)
    d1, s1, b1, _ = sweep_scenarios_physics(
        current_temps=current_temps,
        outdoor_temp=outdoor_temp,
        forecast_temps=forecast_temps,
        environment_states=environment_states,
        sim_params=sim_params,
        hour_of_day=base_hour,
        recent_history=recent_history,
        schedules=schedules,
        base_hour=base_hour,
        prev_state=prev_state,
        solar_fractions=solar_fractions,
        solar_elevations=solar_elevations,
    )
    print(f"  Comfort: {d1.comfort_cost:.1f}  Energy: {d1.energy_cost:.3f}  Total: {d1.comfort_cost + d1.energy_cost:.1f}")
    print("  Effectors:")
    for e in d1.effectors:
        t = f" @ {e.target}" if e.target is not None else ""
        print(f"    {e.name}: {e.mode}{t}")
    print("  Predictions:")
    for sensor in sorted(d1.predictions):
        p = d1.predictions[sensor]
        if "1h" in p:
            print(f"    {sensor:<30} 1h={p['1h']:6.1f}  2h={p['2h']:6.1f}  4h={p['4h']:6.1f}  6h={p['6h']:6.1f}")
    print()

    # Toggled sweep
    toggled = dict(environment_states)
    toggled[env_name] = not is_active

    print("=" * 60)
    print(f"RE-SWEEP ({action} {env_name})")
    print("=" * 60)
    d2, s2, b2, _ = sweep_scenarios_physics(
        current_temps=current_temps,
        outdoor_temp=outdoor_temp,
        forecast_temps=forecast_temps,
        environment_states=toggled,
        sim_params=sim_params,
        hour_of_day=base_hour,
        recent_history=recent_history,
        schedules=schedules,
        base_hour=base_hour,
        prev_state=prev_state,
        solar_fractions=solar_fractions,
        solar_elevations=solar_elevations,
    )
    print(f"  Comfort: {d2.comfort_cost:.1f}  Energy: {d2.energy_cost:.3f}  Total: {d2.comfort_cost + d2.energy_cost:.1f}")
    print("  Effectors:")
    for e in d2.effectors:
        t = f" @ {e.target}" if e.target is not None else ""
        print(f"    {e.name}: {e.mode}{t}")

    # Show prediction differences
    print("  Prediction deltas (re-sweep - original):")
    for sensor in sorted(d2.predictions):
        p1 = d1.predictions.get(sensor, {})
        p2 = d2.predictions[sensor]
        if "1h" in p2 and "1h" in p1:
            diffs = {h: p2[h] - p1[h] for h in ["1h", "2h", "4h", "6h"] if h in p2 and h in p1}
            if any(abs(v) > 0.01 for v in diffs.values()):
                parts = [f"{h}={d:+.2f}" for h, d in diffs.items()]
                print(f"    {sensor:<30} {' '.join(parts)}")
    print()

    # Compute benefit
    total_hw = sum(HORIZON_WEIGHTS.get(h, 0.5) for h in CONTROL_HORIZONS)
    cost_norm = len(schedules) * total_hw
    comfort_improvement = (d1.comfort_cost - d2.comfort_cost) / cost_norm
    energy_saving = d1.energy_cost - d2.energy_cost
    total_benefit = comfort_improvement + energy_saving

    print("=" * 60)
    print("BENEFIT CALCULATION")
    print("=" * 60)
    print(f"  cost_norm = {len(schedules)} schedules × {total_hw:.1f} horizon_weight = {cost_norm:.1f}")
    print(f"  comfort: {d1.comfort_cost:.1f} → {d2.comfort_cost:.1f}  (raw delta = {d1.comfort_cost - d2.comfort_cost:.1f})")
    print(f"  comfort_improvement = {d1.comfort_cost - d2.comfort_cost:.1f} / {cost_norm:.1f} = {comfort_improvement:.2f}")
    print(f"  energy_saving = {energy_saving:.3f}")
    print(f"  total_benefit = {total_benefit:.2f}")
    print()
    print("  Opportunity threshold: 0.3  Notification threshold: 1.5")
    would_fire = total_benefit > 0.3
    would_notify = total_benefit > 1.5
    print(f"  Would fire:  {'YES' if would_fire else 'no'}")
    print(f"  Would notify: {'YES' if would_notify else 'no'}")

    # Show which effectors changed
    changed = []
    e1_map = {e.name: e for e in d1.effectors}
    e2_map = {e.name: e for e in d2.effectors}
    for name in sorted(set(e1_map) | set(e2_map)):
        ea = e1_map.get(name)
        eb = e2_map.get(name)
        if ea and eb and (ea.mode != eb.mode or ea.target != eb.target):
            t1 = f"{ea.mode}" + (f"@{ea.target}" if ea.target is not None else "")
            t2 = f"{eb.mode}" + (f"@{eb.target}" if eb.target is not None else "")
            changed.append(f"  {name}: {t1} → {t2}")
    if changed:
        print()
        print("  Effector changes:")
        for c in changed:
            print(f"  {c}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    replay(timestamp=sys.argv[1], env_name=sys.argv[2])
