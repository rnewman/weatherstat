#!/usr/bin/env python3
"""Weatherstat debug inspector.

Usage:
    python scripts/debug_state.py                       # full status summary
    python scripts/debug_state.py temps                  # current temperatures + comfort
    python scripts/debug_state.py gains [SENSOR]         # sysid gains (optionally filtered)
    python scripts/debug_state.py taus                   # tau models + environment betas
    python scripts/debug_state.py decisions [N]          # last N decisions (default 5)
    python scripts/debug_state.py opportunities          # advisory state
    python scripts/debug_state.py snapshots              # snapshot DB stats
    python scripts/debug_state.py comfort [SENSOR]       # comfort schedules (optionally filtered)
    python scripts/debug_state.py why [EFFECTOR]          # explain why active effectors are on
    python scripts/debug_state.py advisory               # advisory states + compound tau effects

    --bundle <dir>   Point at a debug bundle instead of the live data directory.
                     Example: python scripts/debug_state.py --bundle ~/.weatherstat/bundles/bundle_20260330T103104p0000 temps

All output is plain text, suitable for piping to Claude or other tools.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

# ── Paths (overridden by --bundle) ───────────────────────────────────────

def _default_data_dir() -> Path:
    import os
    return Path(os.environ.get("WEATHERSTAT_DATA_DIR", Path.home() / ".weatherstat"))


_data_dir = _default_data_dir()


def _paths() -> dict[str, Path]:
    """Return all data paths relative to the active data directory."""
    d = _data_dir
    return {
        "thermal_params": d / "thermal_params.json",
        "decision_log": d / "decision_log.db",
        "snapshots_db": d / "snapshots" / "snapshots.db",
        "control_state": d / "control_state.json",
        "advisory_state": d / "advisory_state.json",
        "config": d / "weatherstat.yaml",
        "decisions_json": d / "decisions.json",  # bundle format
    }


def _set_bundle(bundle_dir: Path) -> None:
    """Redirect all paths to a bundle directory."""
    global _data_dir
    _data_dir = bundle_dir

    # Bundles store snapshots.db at the top level; ensure the expected
    # subdirectory path also works.
    snap_subdir = bundle_dir / "snapshots"
    snap_subdir.mkdir(exist_ok=True)
    top_db = bundle_dir / "snapshots.db"
    sub_db = snap_subdir / "snapshots.db"
    if top_db.exists() and not sub_db.exists():
        import shutil
        shutil.copy2(top_db, sub_db)

    print(f"[bundle] Using {bundle_dir}\n")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _ago(ts_str: str | None) -> str:
    """Human-readable time-ago from an ISO timestamp."""
    if not ts_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h {(secs % 3600) // 60}m ago"
        return f"{secs // 86400}d ago"
    except (ValueError, TypeError):
        return ts_str


# ── Subcommands ──────────────────────────────────────────────────────────


def cmd_temps() -> None:
    """Current temperatures from latest snapshot + comfort bounds from latest decision."""
    p = _paths()
    conn = sqlite3.connect(str(p["snapshots_db"]))
    row = conn.execute("SELECT MAX(timestamp) FROM readings").fetchone()
    ts = row[0] if row else None
    print(f"Latest snapshot: {ts} ({_ago(ts)})")

    if not ts:
        print("No data.")
        return

    rows = conn.execute(
        "SELECT name, value FROM readings WHERE timestamp = ? AND name LIKE '%_temp' ORDER BY name",
        (ts,),
    ).fetchall()
    conn.close()

    # Load comfort bounds from latest decision
    bounds: dict[str, dict] = {}
    bounds = _load_comfort_bounds(p)

    print()
    print(f"  {'Sensor':<30} {'Temp':>6}  {'Min':>5} {'Pref':>9} {'Max':>5}  Status")
    print(f"  {'─' * 30} {'─' * 6}  {'─' * 5} {'─' * 9} {'─' * 5}  {'─' * 10}")

    for name, value in rows:
        try:
            temp = float(value)
        except (ValueError, TypeError):
            continue
        # Find comfort bounds by label match
        label = name.replace("_temp", "")
        b = bounds.get(label, {})
        cmin = b.get("min", "")
        cmax = b.get("max", "")
        plo = b.get("preferred_lo", "")
        phi = b.get("preferred_hi", "")

        status = ""
        if cmin and cmax:
            if temp < float(cmin):
                status = f"COLD ({temp - float(cmin):+.1f})"
            elif temp > float(cmax):
                status = f"HOT ({temp - float(cmax):+.1f})"
            elif plo and phi:
                if temp < float(plo):
                    status = "below pref"
                elif temp > float(phi):
                    status = "above pref"
                else:
                    status = "in band"

        if plo and phi:
            plo_f, phi_f = float(plo), float(phi)
            pref_str = f"{plo_f:.1f}" if abs(plo_f - phi_f) < 0.05 else f"{plo_f:.1f}-{phi_f:.1f}"
        else:
            pref_str = ""
        cmin_str = f"{float(cmin):.1f}" if cmin else ""
        cmax_str = f"{float(cmax):.1f}" if cmax else ""
        print(f"  {name:<30} {temp:6.1f}  {cmin_str:>5} {pref_str:>9} {cmax_str:>5}  {status}")


def _load_comfort_bounds(p: dict[str, Path]) -> dict[str, dict]:
    """Load comfort bounds from decision log (SQLite) or decisions.json (bundle)."""
    # Try SQLite decision log first
    if p["decision_log"].exists():
        dconn = sqlite3.connect(str(p["decision_log"]))
        drow = dconn.execute(
            "SELECT comfort_bounds, active_profile, mrt_offsets FROM decisions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        dconn.close()
        if drow and drow[0]:
            bounds = json.loads(drow[0])
            profile = drow[1] or "?"
            mrt = json.loads(drow[2]) if drow[2] else {}
            mrt_base = mrt.get("_base", 0)
            print(f"Profile: {profile}  MRT base: {mrt_base:+.1f}°F")
            return bounds

    # Fall back to decisions.json (bundle format)
    if p["decisions_json"].exists():
        with open(p["decisions_json"]) as f:
            decisions = json.load(f)
        if decisions:
            # Last decision in the list is closest to target time
            last = decisions[-1]
            bounds = last.get("comfort_bounds", {})
            profile = last.get("active_profile", "?")
            mrt = last.get("mrt_offsets", {})
            mrt_base = mrt.get("_base", 0) if isinstance(mrt, dict) else 0
            print(f"Profile: {profile}  MRT base: {mrt_base:+.1f}°F")
            return bounds

    return {}


def cmd_gains(sensor_filter: str | None = None) -> None:
    """Show sysid effector→sensor gains."""
    params = _load_json(_paths()["thermal_params"])
    if not params:
        print("No thermal_params.json found.")
        return

    ts = params.get("timestamp", "?")
    n = params.get("n_snapshots", "?")
    print(f"Sysid: {ts} ({n} snapshots)")
    print()

    gains = params.get("effector_sensor_gains", [])
    if sensor_filter:
        gains = [g for g in gains if sensor_filter.lower() in g["sensor"].lower()]

    # Group by sensor
    by_sensor: dict[str, list] = {}
    for g in gains:
        by_sensor.setdefault(g["sensor"], []).append(g)

    for sensor in sorted(by_sensor):
        print(f"  {sensor}:")
        for g in sorted(by_sensor[sensor], key=lambda x: abs(x.get("gain_f_per_hour", 0)), reverse=True):
            gain = g.get("gain_f_per_hour", 0)
            t = g.get("t_statistic", 0)
            lag = g.get("best_lag_minutes", 0)
            neg = g.get("negligible", False)
            flag = " (negligible)" if neg else ""
            print(f"    {g['effector']:<28} {gain:+7.3f} °F/hr  t={t:5.2f}  lag={lag:4.0f}m{flag}")
        print()


def cmd_taus() -> None:
    """Show tau models and environment betas."""
    params = _load_json(_paths()["thermal_params"])
    if not params:
        print("No thermal_params.json found.")
        return

    print(f"Sysid: {params.get('timestamp', '?')}")
    print()

    for t in sorted(params.get("fitted_taus", []), key=lambda x: x["sensor"]):
        tau = t["tau_base"]
        n = t.get("n_segments", "?")
        print(f"  {t['sensor']:<30} tau={tau:6.1f}h  ({n} segments)")
        tau_betas = t.get("environment_tau_betas", t.get("advisory_tau_betas", t.get("window_betas", {})))
        for dev, beta in sorted(tau_betas.items()):
            eff = 1.0 / (1.0 / tau + beta)
            print(f"    advisory {dev:<20} beta={beta:.6f}  (eff tau={eff:.1f}h)")
    print()

    # MRT weights
    mrt = params.get("mrt_weights", {})
    if mrt:
        print("  MRT weights:")
        for s, w in sorted(mrt.items()):
            if abs(w - 1.0) > 0.01:
                print(f"    {s:<30} {w:.3f}")


def cmd_decisions(n: int = 5) -> None:
    """Show recent decisions."""
    p = _paths()

    # Try SQLite first, then bundle JSON
    decisions_data: list[dict] = []

    if p["decision_log"].exists():
        conn = sqlite3.connect(str(p["decision_log"]))
        rows = conn.execute(
            """SELECT timestamp, live, comfort_cost, energy_cost,
                      trajectory, active_profile, blocked,
                      current_temps,
                      actual_comfort_cost, outcome_backfilled,
                      effector_decisions
               FROM decisions ORDER BY timestamp DESC LIMIT ?""",
            (n,),
        ).fetchall()
        conn.close()
        for row in rows:
            ts, live, cc, ec, traj, profile, blocked_json, temps_json, actual_cc, backfilled, eff_json = row
            decisions_data.append({
                "timestamp": ts, "live": live, "comfort_cost": cc, "energy_cost": ec,
                "active_profile": profile, "blocked": json.loads(blocked_json) if blocked_json else {},
                "current_temps": json.loads(temps_json) if temps_json else {},
                "actual_comfort_cost": actual_cc, "outcome_backfilled": backfilled,
                "effector_decisions": json.loads(eff_json) if eff_json else [],
            })
    elif p["decisions_json"].exists():
        with open(p["decisions_json"]) as f:
            all_decisions = json.load(f)
        # Take last N
        decisions_data = all_decisions[-n:]

    if not decisions_data:
        print("No decisions found.")
        return

    for d in decisions_data:
        ts = d.get("timestamp", "?")
        live = d.get("live", False)
        cc = d.get("comfort_cost", 0)
        ec = d.get("energy_cost", 0)
        profile = d.get("active_profile", "?")
        mode = "LIVE" if live else "DRY"
        print(f"─── {ts}  [{mode}]  profile={profile or '?'} ───")
        print(f"  Cost: comfort={cc:.1f}  energy={ec:.3f}  total={cc + ec:.1f}")

        actual_cc = d.get("actual_comfort_cost")
        backfilled = d.get("outcome_backfilled")
        if backfilled and actual_cc is not None:
            print(f"  Actual comfort cost: {actual_cc:.1f}")

        # Effector decisions
        effs = d.get("effector_decisions", [])
        if effs:
            parts = []
            for e in effs:
                name = e.get("name", "?")
                mode_val = e.get("mode", "?")
                target = e.get("target")
                if target is not None:
                    parts.append(f"{name}={mode_val}@{target}")
                else:
                    parts.append(f"{name}={mode_val}")
            print(f"  Effectors: {', '.join(parts)}")

        # Current temps (brief)
        temps = d.get("current_temps", {})
        if temps:
            temp_parts = []
            for k, v in sorted(temps.items()):
                if "_temp" in k:
                    label = k.replace("_temp", "")
                    temp_parts.append(f"{label}={v:.1f}")
            print(f"  Temps: {', '.join(temp_parts)}")

        # Blocked
        blocked = d.get("blocked", {})
        if blocked:
            print(f"  Blocked: {blocked}")

        print()


def cmd_opportunities() -> None:
    """Show advisory state."""
    state = _load_json(_paths()["advisory_state"])
    if not state:
        print("No advisory_state.json found.")
        return

    active = state.get("active", {})
    cooldowns = state.get("cooldowns", {})
    now = datetime.now(UTC).timestamp()

    print("Active opportunities:")
    if active:
        for k, v in sorted(active.items()):
            print(f"  {k}: {json.dumps(v, indent=4)}")
    else:
        print("  (none)")

    print()
    print("Cooldowns:")
    for k, expire in sorted(cooldowns.items()):
        remaining = expire - now
        if remaining > 0:
            mins = int(remaining / 60)
            print(f"  {k:<40} {mins}m remaining")
        else:
            print(f"  {k:<40} expired")


def cmd_snapshots() -> None:
    """Snapshot DB stats."""
    p = _paths()
    if not p["snapshots_db"].exists():
        print("No snapshots.db found.")
        return

    conn = sqlite3.connect(str(p["snapshots_db"]))
    row = conn.execute(
        "SELECT COUNT(DISTINCT timestamp), MIN(timestamp), MAX(timestamp) FROM readings"
    ).fetchone()
    n_ts, ts_min, ts_max = row

    # Count distinct sensors
    n_sensors = conn.execute("SELECT COUNT(DISTINCT name) FROM readings").fetchone()[0]

    # Total rows
    n_rows = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]

    conn.close()

    print(f"Snapshots: {n_ts:,} timestamps, {n_sensors} sensors, {n_rows:,} rows")
    print(f"  First: {ts_min}")
    print(f"  Last:  {ts_max} ({_ago(ts_max)})")


def cmd_why(effector_filter: str | None = None) -> None:
    """Explain why each active effector is on: gains, affected sensors, current temps, predictions."""
    p = _paths()
    params = _load_json(p["thermal_params"])
    cs = _load_json(p["control_state"])
    if not params or not cs:
        print("Missing thermal_params.json or control_state.json.")
        return

    # Load config for environment column mapping
    import os
    if _data_dir != _default_data_dir():
        os.environ["WEATHERSTAT_DATA_DIR"] = str(_data_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from weatherstat.yaml_config import load_config
    cfg = load_config()

    # Build reverse map: column_name -> config_name
    col_to_name: dict[str, str] = {ecfg.column: ename for ename, ecfg in cfg.environment.items()}

    # Load current temps + advisory states from latest snapshot
    conn = sqlite3.connect(str(p["snapshots_db"]))
    row = conn.execute("SELECT MAX(timestamp) FROM readings").fetchone()
    ts = row[0] if row else None
    current_temps: dict[str, float] = {}
    if ts:
        for name, value in conn.execute(
            "SELECT name, value FROM readings WHERE timestamp = ?", (ts,)
        ).fetchall():
            if name.endswith("_temp"):
                with contextlib.suppress(ValueError, TypeError):
                    current_temps[name] = float(value)
    # Per-column latest lookup for env entries (sparse storage may skip columns)
    advisory_active: set[str] = set()
    for col in col_to_name:
        r = conn.execute(
            "SELECT value FROM readings WHERE name = ? ORDER BY timestamp DESC LIMIT 1",
            (col,),
        ).fetchone()
        if r is None:
            continue
        try:
            if float(r[0]) > 0.5:
                advisory_active.add(col_to_name[col])
        except (TypeError, ValueError):
            pass
    conn.close()

    # Show advisory context at top if any windows are open
    if advisory_active:
        # Compute worst-affected tau
        fitted_taus = params.get("fitted_taus", [])
        worst_ratio = 1.0
        worst_sensor = ""
        for t in fitted_taus:
            tau_base = t["tau_base"]
            inv_tau = 1.0 / tau_base
            for dev, beta in t.get("environment_tau_betas", t.get("advisory_tau_betas", t.get("window_betas", {}))).items():
                if dev in advisory_active:
                    inv_tau += beta
            tau_eff = 1.0 / inv_tau if inv_tau > 0 else float("inf")
            ratio = tau_eff / tau_base
            if ratio < worst_ratio:
                worst_ratio = ratio
                worst_sensor = t["sensor"]

        print(f"⚠ Advisory state: {len(advisory_active)} open — {', '.join(sorted(advisory_active))}")
        if worst_ratio < 0.5:
            print(f"  Fastest cooling: {worst_sensor} at {worst_ratio:.0%} of base tau")
        print("  (run `just debug advisory` for full tau breakdown)")
        print()

    # Load comfort bounds + predictions from latest decision
    bounds: dict[str, dict] = {}
    predictions: dict[str, dict] = {}
    if p["decision_log"].exists():
        dconn = sqlite3.connect(str(p["decision_log"]))
        drow = dconn.execute(
            "SELECT comfort_bounds, predictions, comfort_cost FROM decisions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        dconn.close()
        if drow:
            bounds = json.loads(drow[0]) if drow[0] else {}
            predictions = json.loads(drow[1]) if drow[1] else {}

    # Index gains by effector
    gains_by_effector: dict[str, list[dict]] = {}
    for g in params.get("effector_sensor_gains", []):
        gains_by_effector.setdefault(g["effector"], []).append(g)

    # Active effectors from control state
    modes = cs.get("modes", {})
    active = {k: v for k, v in modes.items() if v != "off"}
    if effector_filter:
        active = {k: v for k, v in active.items() if effector_filter.lower() in k.lower()}

    if not active:
        print("No active effectors" + (f" matching '{effector_filter}'" if effector_filter else "") + ".")
        return

    for eff_name, mode in sorted(active.items()):
        target = cs.get("setpoints", {}).get(eff_name)
        target_str = f" @ {target}°F" if target is not None else ""
        print(f"━━━ {eff_name} = {mode}{target_str} ━━━")

        gains = gains_by_effector.get(eff_name, [])
        sig_gains = [g for g in gains if not g.get("negligible", True)]

        if not sig_gains:
            print("  No significant sysid gains — effector has no modeled effect.")
            print()
            continue

        # Sort by absolute gain magnitude
        sig_gains.sort(key=lambda g: abs(g.get("gain_f_per_hour", 0)), reverse=True)

        print(f"  Significant gains ({len(sig_gains)}):")
        for g in sig_gains:
            sensor = g["sensor"]
            gain = g.get("gain_f_per_hour", 0)
            t_stat = g.get("t_statistic", 0)
            temp = current_temps.get(sensor)
            label = sensor.replace("_temp", "")
            b = bounds.get(label, {})

            # Comfort status
            status = ""
            flags = []
            if temp is not None and b:
                cmin = b.get("min")
                cmax = b.get("max")
                plo = b.get("preferred_lo")
                phi = b.get("preferred_hi")
                if cmin is not None and cmax is not None:
                    if temp > float(cmax):
                        status = f"HOT ({temp - float(cmax):+.1f})"
                    elif temp < float(cmin):
                        status = f"COLD ({float(cmin) - temp:+.1f})"
                    elif plo is not None and temp < float(plo):
                        status = "below pref"
                    elif phi is not None and temp > float(phi):
                        status = "above pref"
                    else:
                        status = "in band"

                    # Flag contradictions
                    if temp > float(cmax) and gain > 0:
                        flags.append("⚠ HEATING AN ALREADY-HOT SENSOR")
                    elif temp < float(cmin) and gain < 0:
                        flags.append("⚠ COOLING AN ALREADY-COLD SENSOR")

            # Flag nonsensical gains
            if sensor == "outdoor_temp":
                flags.append("⚠ NONSENSICAL: indoor effector → outdoor temp")
            if abs(gain) > 1.0 and "bookshelf" not in eff_name:
                flags.append(f"⚠ IMPLAUSIBLE MAGNITUDE ({gain:+.2f}°F/hr)")

            temp_str = f"{temp:.1f}°F" if temp is not None else "n/a"
            gain_dir = "warms" if gain > 0 else "cools"
            print(f"    {sensor:<30} {gain:+.4f}°F/hr ({gain_dir})  t={t_stat:.2f}  now={temp_str}  [{status}]")
            for flag in flags:
                print(f"      {flag}")

        # Show predictions for affected constrained sensors
        constrained_sensors = {g["sensor"] for g in sig_gains if g["sensor"].replace("_temp", "") in bounds}
        if constrained_sensors and predictions:
            print("  Predicted trajectories (with current plan):")
            for sensor in sorted(constrained_sensors):
                preds = predictions.get(sensor, {})
                if preds:
                    parts = [f"{h}={v:.1f}" for h, v in preds.items()]
                    temp = current_temps.get(sensor, 0)
                    print(f"    {sensor:<30} now={temp:.1f}  {', '.join(parts)}")

        print()


def cmd_advisory() -> None:
    """Show active advisory states and their compound effect on tau."""
    p = _paths()
    params = _load_json(p["thermal_params"])
    if not params:
        print("No thermal_params.json found.")
        return

    # Load config for environment column mapping
    import os
    if _data_dir != _default_data_dir():
        os.environ["WEATHERSTAT_DATA_DIR"] = str(_data_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from weatherstat.yaml_config import load_config
    cfg = load_config()

    # Build reverse map: column_name -> config_name
    col_to_name: dict[str, str] = {ecfg.column: ename for ename, ecfg in cfg.environment.items()}
    env_columns = set(col_to_name)

    # Get current environment states from latest snapshot
    conn = sqlite3.connect(str(p["snapshots_db"]))
    ts_row = conn.execute("SELECT MAX(timestamp) FROM readings").fetchone()
    ts = ts_row[0] if ts_row else None
    # Query each env column's most recent value independently — sparse columns
    # may not exist at the global max timestamp. Treat > 0.5 as active so both
    # binary ("1") and continuous ("0.85", "1.00") work.
    environment_states: dict[str, bool] = {}
    for name in env_columns:
        row = conn.execute(
            "SELECT value FROM readings WHERE name = ? ORDER BY timestamp DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row is None:
            continue
        try:
            environment_states[col_to_name[name]] = float(row[0]) > 0.5
        except (TypeError, ValueError):
            environment_states[col_to_name[name]] = False
    conn.close()

    active = {k for k, v in environment_states.items() if v}
    inactive = {k for k, v in environment_states.items() if not v}

    print(f"Environment states (snapshot {ts}):")
    print(f"  Active:   {', '.join(sorted(active)) or '(none)'}")
    print(f"  Default:  {', '.join(sorted(inactive)) or '(none)'}")
    print()

    if not active:
        print("No active environment factors — base tau applies everywhere.")
        return

    # Show compound tau effect per sensor
    fitted_taus = params.get("fitted_taus", [])
    advisory_solar = params.get("environment_solar_betas", params.get("advisory_solar_betas", {}))

    print("Compound tau effects (with current active environment factors):")
    print(f"  {'Sensor':<30} {'Base τ':>7} {'Eff τ':>7} {'Ratio':>7}  Contributing")
    print(f"  {'─' * 30} {'─' * 7} {'─' * 7} {'─' * 7}  {'─' * 40}")

    for t in sorted(fitted_taus, key=lambda x: x["sensor"]):
        sensor = t["sensor"]
        tau_base = t["tau_base"]
        tau_betas = t.get("environment_tau_betas", t.get("advisory_tau_betas", t.get("window_betas", {})))

        # Sum betas for active devices
        inv_tau = 1.0 / tau_base
        contributors: list[str] = []
        for dev, beta in tau_betas.items():
            if dev in active:
                inv_tau += beta
                eff = 1.0 / (1.0 / tau_base + beta)
                contributors.append(f"{dev}(→{eff:.0f}h)")

        tau_eff = 1.0 / inv_tau if inv_tau > 0 else float("inf")
        ratio = tau_eff / tau_base

        flag = ""
        if ratio < 0.3:
            flag = "  ⚠ EXTREME"
        elif ratio < 0.5:
            flag = "  ⚠ FAST"

        print(f"  {sensor:<30} {tau_base:6.1f}h {tau_eff:6.1f}h {ratio:6.1%}{flag}  {', '.join(contributors)}")

    # Show environment solar betas for active devices
    active_solar = {dev: betas for dev, betas in advisory_solar.items() if dev in active}
    if active_solar:
        print()
        print("Environment solar betas (active devices):")
        for dev in sorted(active_solar):
            betas = active_solar[dev]
            print(f"  {dev}:")
            for sensor, beta in sorted(betas.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"    {sensor:<30} β_solar={beta:+.4f} °F/hr per sin(elev)")
    print()


def cmd_comfort(sensor_filter: str | None = None) -> None:
    """Show comfort schedules from config."""
    # If pointing at a bundle, set WEATHERSTAT_DATA_DIR so load_config finds the right YAML
    import os
    if _data_dir != _default_data_dir():
        os.environ["WEATHERSTAT_DATA_DIR"] = str(_data_dir)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    for cs in cfg.constraints:
        if sensor_filter and sensor_filter.lower() not in cs.label.lower():
            continue
        print(f"  {cs.label} ({cs.sensor}):")
        for e in cs.entries:
            pref = f"{e.preferred_lo}-{e.preferred_hi}" if e.preferred_lo != e.preferred_hi else f"{e.preferred_lo}"
            print(
                f"    {e.start_hour:02d}-{e.end_hour:02d}:  "
                f"acceptable=[{e.acceptable_lo:.0f},{e.acceptable_hi:.0f}]  pref={pref}  backup=[{e.backup_lo:.0f},{e.backup_hi:.0f}]  "
                f"cold={e.cold_penalty:.1f}  hot={e.hot_penalty:.1f}"
            )
        if cs.mrt_weight != 1.0:
            print(f"    mrt_weight={cs.mrt_weight:.2f}")
        print()


def cmd_summary() -> None:
    """Full status summary."""
    p = _paths()
    print("=== Weatherstat Debug Summary ===\n")

    # Snapshots
    cmd_snapshots()
    print()

    # Sysid
    params = _load_json(p["thermal_params"])
    if params:
        print(f"Sysid: {params.get('timestamp', '?')} ({_ago(params.get('timestamp'))})")
        n_taus = len(params.get("fitted_taus", []))
        n_gains = sum(1 for g in params.get("effector_sensor_gains", []) if not g.get("negligible", True))
        print(f"  {n_taus} taus, {n_gains} significant gains")
    print()

    # Control state
    cs = _load_json(p["control_state"])
    if cs:
        print(f"Control state: {_ago(cs.get('last_decision_time'))}")
        if cs.get("setpoints"):
            parts = [f"{k}={v}" for k, v in sorted(cs["setpoints"].items())]
            print(f"  Setpoints: {', '.join(parts)}")
        if cs.get("modes"):
            parts = [f"{k}={v}" for k, v in sorted(cs["modes"].items())]
            print(f"  Modes: {', '.join(parts)}")
    print()

    # Temps
    cmd_temps()
    print()

    # Opportunities
    cmd_opportunities()


# ── Entry point ──────────────────────────────────────────────────────────

COMMANDS = {
    "temps": lambda args: cmd_temps(),
    "gains": lambda args: cmd_gains(args[0] if args else None),
    "taus": lambda args: cmd_taus(),
    "decisions": lambda args: cmd_decisions(int(args[0]) if args else 5),
    "opportunities": lambda args: cmd_opportunities(),
    "snapshots": lambda args: cmd_snapshots(),
    "comfort": lambda args: cmd_comfort(args[0] if args else None),
    "why": lambda args: cmd_why(args[0] if args else None),
    "advisory": lambda args: cmd_advisory(),
}

if __name__ == "__main__":
    argv = sys.argv[1:]

    # Extract --bundle flag
    if "--bundle" in argv:
        idx = argv.index("--bundle")
        if idx + 1 >= len(argv):
            print("--bundle requires a directory argument")
            sys.exit(1)
        _set_bundle(Path(argv[idx + 1]))
        argv = argv[:idx] + argv[idx + 2:]

    if not argv:
        cmd_summary()
    elif argv[0] in COMMANDS:
        COMMANDS[argv[0]](argv[1:])
    elif argv[0] in ("-h", "--help"):
        print(__doc__)
    else:
        print(f"Unknown command: {argv[0]}")
        print(__doc__)
        sys.exit(1)
