#!/usr/bin/env python3
"""Weatherstat debug inspector.

Usage:
    python scripts/debug_state.py                  # full status summary
    python scripts/debug_state.py temps             # current temperatures + comfort
    python scripts/debug_state.py gains [SENSOR]    # sysid gains (optionally filtered)
    python scripts/debug_state.py taus              # tau models + window betas
    python scripts/debug_state.py decisions [N]     # last N decisions (default 5)
    python scripts/debug_state.py opportunities     # advisory state
    python scripts/debug_state.py snapshots         # snapshot DB stats
    python scripts/debug_state.py comfort [SENSOR]  # comfort schedules (optionally filtered)

All output is plain text, suitable for piping to Claude or other tools.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path.home() / ".weatherstat"
THERMAL_PARAMS = DATA_DIR / "thermal_params.json"
DECISION_LOG = DATA_DIR / "decision_log.db"
SNAPSHOTS_DB = DATA_DIR / "snapshots" / "snapshots.db"
CONTROL_STATE = DATA_DIR / "control_state.json"
EXECUTOR_STATE = DATA_DIR / "executor_state.json"
ADVISORY_STATE = DATA_DIR / "advisory_state.json"


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
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
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
    conn = sqlite3.connect(str(SNAPSHOTS_DB))
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
    if DECISION_LOG.exists():
        dconn = sqlite3.connect(str(DECISION_LOG))
        drow = dconn.execute(
            "SELECT comfort_bounds, active_profile, mrt_offsets FROM decisions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if drow and drow[0]:
            bounds = json.loads(drow[0])
            profile = drow[1] or "?"
            mrt = json.loads(drow[2]) if drow[2] else {}
            mrt_base = mrt.get("_base", 0)
            print(f"Profile: {profile}  MRT base: {mrt_base:+.1f}°F")
        dconn.close()

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


def cmd_gains(sensor_filter: str | None = None) -> None:
    """Show sysid effector→sensor gains."""
    params = _load_json(THERMAL_PARAMS)
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
    """Show tau models and window betas."""
    params = _load_json(THERMAL_PARAMS)
    if not params:
        print("No thermal_params.json found.")
        return

    print(f"Sysid: {params.get('timestamp', '?')}")
    print()

    for t in sorted(params.get("fitted_taus", []), key=lambda x: x["sensor"]):
        tau = t["tau_base"]
        n = t.get("n_segments", "?")
        print(f"  {t['sensor']:<30} tau={tau:6.1f}h  ({n} segments)")
        for win, beta in sorted(t.get("window_betas", {}).items()):
            # effective tau with this window open
            eff = 1.0 / (1.0 / tau + beta)
            print(f"    window {win:<20} beta={beta:.6f}  (eff tau={eff:.1f}h)")
        for pair, beta in sorted(t.get("interaction_betas", {}).items()):
            print(f"    cross  {pair:<20} beta={beta:.6f}")
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
    if not DECISION_LOG.exists():
        print("No decision_log.db found.")
        return

    conn = sqlite3.connect(str(DECISION_LOG))
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

    if not rows:
        print("No decisions logged.")
        return

    for row in rows:
        ts, live, cc, ec, traj, profile, blocked_json, temps_json, actual_cc, backfilled, eff_json = row
        mode = "LIVE" if live else "DRY"
        print(f"─── {ts}  [{mode}]  profile={profile or '?'} ───")
        print(f"  Cost: comfort={cc:.1f}  energy={ec:.3f}  total={cc + ec:.1f}")
        if backfilled and actual_cc is not None:
            print(f"  Actual comfort cost: {actual_cc:.1f}")

        # Effector decisions
        if eff_json:
            effs = json.loads(eff_json)
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
        if temps_json:
            temps = json.loads(temps_json)
            temp_parts = []
            for k, v in sorted(temps.items()):
                if "_temp" in k:
                    label = k.replace("_temp", "")
                    temp_parts.append(f"{label}={v:.1f}")
            print(f"  Temps: {', '.join(temp_parts)}")

        # Blocked
        if blocked_json:
            blocked = json.loads(blocked_json)
            if blocked:
                print(f"  Blocked: {blocked}")

        print()


def cmd_opportunities() -> None:
    """Show advisory state."""
    state = _load_json(ADVISORY_STATE)
    if not state:
        print("No advisory_state.json found.")
        return

    active = state.get("active", {})
    cooldowns = state.get("cooldowns", {})
    now = datetime.now(timezone.utc).timestamp()

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
    if not SNAPSHOTS_DB.exists():
        print("No snapshots.db found.")
        return

    conn = sqlite3.connect(str(SNAPSHOTS_DB))
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


def cmd_comfort(sensor_filter: str | None = None) -> None:
    """Show comfort schedules from config."""
    # Import here to use the config loader
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
                f"min={e.min_temp:.0f}  pref={pref}  max={e.max_temp:.0f}  "
                f"cold={e.cold_penalty:.1f}  hot={e.hot_penalty:.1f}"
            )
        if cs.mrt_weight != 1.0:
            print(f"    mrt_weight={cs.mrt_weight:.2f}")
        print()


def cmd_summary() -> None:
    """Full status summary."""
    print("=== Weatherstat Debug Summary ===\n")

    # Snapshots
    cmd_snapshots()
    print()

    # Sysid
    params = _load_json(THERMAL_PARAMS)
    if params:
        print(f"Sysid: {params.get('timestamp', '?')} ({_ago(params.get('timestamp'))})")
        n_taus = len(params.get("fitted_taus", []))
        n_gains = sum(1 for g in params.get("effector_sensor_gains", []) if not g.get("negligible", True))
        print(f"  {n_taus} taus, {n_gains} significant gains")
    print()

    # Control state
    cs = _load_json(CONTROL_STATE)
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
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_summary()
    elif args[0] in COMMANDS:
        COMMANDS[args[0]](args[1:])
    elif args[0] in ("-h", "--help"):
        print(__doc__)
    else:
        print(f"Unknown command: {args[0]}")
        print(__doc__)
        sys.exit(1)
