"""Post-heating-off residual analysis: is there a slab thermal-mass signature?

Hypothesis: the first-order simulator (forcing ends when activity ends, after
lag) under-predicts room temperature in the hours after zone heating turns
off, because the charged floor slab keeps delivering heat. If true, residuals
(actual - predicted) should be systematically positive shortly after an off
transition in hydronic rooms, decaying over the slab discharge time — and
absent in mini-split rooms (air-source, low mass).

Groups, evaluated at each outcome time (decision ts + horizon):
  POST-OFF   zone heating off, last substantial run (>=30 min) ended <=6h ago
  HEATING-ON zone heating active
  COLD-SOAK  zone heating off >=8h (baseline: model bias without slab effects)

Decisions arrive every 5 minutes, so outcomes near one off-transition are
heavily correlated. The headline comparison therefore aggregates per
off-event (mean residual per event, stats across events), not per row.

Usage: uv run python scripts/slab_residual_analysis.py
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from weatherstat.config import DATA_DIR, DECISION_LOG_DB, SENSOR_LABELS, SNAPSHOTS_DB
from weatherstat.weather import solar_sin_elevation
from weatherstat.yaml_config import load_config

MIN_RUN_MINUTES = 30  # a run shorter than this doesn't meaningfully charge the slab
POST_OFF_HOURS = 6.0
COLD_SOAK_HOURS = 8.0
HORIZON_HOURS = {"1h": 1, "2h": 2, "4h": 4, "6h": 6}

ZONE_ACTION_COLS = {
    "thermostat_upstairs": "thermostat_upstairs_action",
    "thermostat_downstairs": "thermostat_downstairs_action",
}


def load_zone_heating() -> dict[str, pd.Series]:
    """Per-zone boolean heating series (5-min index) from the EAV table."""
    conn = sqlite3.connect(str(SNAPSHOTS_DB))
    out: dict[str, pd.Series] = {}
    for zone, col in ZONE_ACTION_COLS.items():
        df = pd.read_sql(
            "SELECT timestamp, value FROM readings WHERE name = ? ORDER BY timestamp",
            conn,
            params=(col,),
        )
        ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        s = pd.Series((df["value"] == "heating").values, index=ts)
        out[zone] = s[~s.index.duplicated()]
    conn.close()
    return out


def find_off_events(heating: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    """(end_time, run_minutes) for each heating run >= MIN_RUN_MINUTES."""
    events: list[tuple[pd.Timestamp, float]] = []
    run_start: pd.Timestamp | None = None
    prev_t: pd.Timestamp | None = None
    for t, on in heating.items():
        if on and run_start is None:
            run_start = t
        elif not on and run_start is not None:
            run_min = (prev_t - run_start).total_seconds() / 60.0 if prev_t is not None else 0.0
            if run_min >= MIN_RUN_MINUTES:
                events.append((t, run_min))
            run_start = None
        prev_t = t
    return events


def zone_state_at(heating: pd.Series, off_ends: list[pd.Timestamp], t: pd.Timestamp) -> tuple[str, float | None]:
    """Classify a moment: (group, hours_since_off)."""
    idx = heating.index.searchsorted(t)
    if idx == 0:
        return ("unknown", None)
    # Nearest reading at or before t; stale >15 min → unknown
    t_obs = heating.index[idx - 1]
    if (t - t_obs).total_seconds() > 900:
        return ("unknown", None)
    if bool(heating.iloc[idx - 1]):
        return ("heating_on", None)
    # Off: hours since the most recent substantial run ended
    prior = [e for e in off_ends if e <= t]
    if not prior:
        return ("cold_soak", None)
    hours = (t - prior[-1]).total_seconds() / 3600.0
    if hours <= POST_OFF_HOURS:
        return ("post_off", hours)
    if hours >= COLD_SOAK_HOURS:
        return ("cold_soak", hours)
    return ("between", hours)


def sensor_zones() -> tuple[dict[str, str], set[str]]:
    """label -> hydronic zone (max thermostat gain); labels of rooms with their own mini split."""
    with open(DATA_DIR / "thermal_params.json") as f:
        params = json.load(f)
    col_to_label = {col: label for col, label in SENSOR_LABELS.items()}
    best_thermo: dict[str, tuple[str, float]] = {}
    for g in params["effector_sensor_gains"]:
        if g["negligible"] or g["gain_f_per_hour"] <= 0 or not g["effector"].startswith("thermostat_"):
            continue
        if g["gain_f_per_hour"] > best_thermo.get(g["sensor"], ("", 0.0))[1]:
            best_thermo[g["sensor"]] = (g["effector"], g["gain_f_per_hour"])
    zones = {col_to_label.get(sc, sc): eff for sc, (eff, _) in best_thermo.items()}
    # Rooms with their own mini split, as opposed to cross-coupled neighbors
    # (piano etc. have learned gains from the living room split but no split of
    # their own). Hardcoded: config temp_col naming doesn't round-trip through
    # SENSOR_LABELS (the stage-22 living_room naming trap).
    own_split = {"bedroom", "living_room_climate"}
    return zones, own_split


def main() -> None:
    heating = load_zone_heating()
    off_events = {z: find_off_events(s) for z, s in heating.items()}
    for z, ev in off_events.items():
        print(f"{z}: {len(ev)} substantial off-transitions (runs >= {MIN_RUN_MINUTES} min)")

    zones, ms_primary = sensor_zones()
    cfg = load_config()
    lat, lon = cfg.location.latitude, cfg.location.longitude
    print(f"zone map: {zones}")
    print(f"own-mini-split rooms: {sorted(ms_primary) or 'none'}\n")

    conn = sqlite3.connect(str(DECISION_LOG_DB))
    rows = conn.execute(
        "SELECT timestamp, outcomes FROM decisions"
        " WHERE outcome_backfilled = 1 AND outcomes IS NOT NULL AND outcomes != ''"
    ).fetchall()
    conn.close()
    print(f"{len(rows)} backfilled decisions\n")

    off_end_lists = {z: [e for e, _ in ev] for z, ev in off_events.items()}

    # rows_by[group][(hydronic?, horizon)] -> list of residuals
    residuals: dict[tuple[str, bool, str], list[float]] = defaultdict(list)
    night_residuals: dict[tuple[str, bool, str], list[float]] = defaultdict(list)
    # per off-event aggregation for POST-OFF: (zone, event_ts, hydronic?, horizon) -> residuals
    per_event: dict[tuple[str, pd.Timestamp, bool, str], list[float]] = defaultdict(list)
    # decay curve: 30-min bins of hours-since-off (hydronic, 1h+2h horizons)
    decay_bins: dict[int, list[float]] = defaultdict(list)

    for ts_str, outcomes_json in rows:
        ts = pd.Timestamp(ts_str)
        try:
            outcomes = json.loads(outcomes_json)
        except json.JSONDecodeError:
            continue
        for label, horizons in outcomes.items():
            zone = zones.get(label)
            if zone is None:
                continue
            hydronic = label not in ms_primary
            for h_label, rec in horizons.items():
                h = HORIZON_HOURS.get(h_label)
                if h is None or rec.get("error") is None:
                    continue
                residual = -float(rec["error"])  # error = predicted - actual
                t_out = ts + timedelta(hours=h)
                group, hours_since = zone_state_at(heating[zone], off_end_lists[zone], t_out)
                if group in ("unknown", "between"):
                    continue
                residuals[(group, hydronic, h_label)].append(residual)
                if solar_sin_elevation(lat, lon, t_out.to_pydatetime()) <= 0.0:
                    night_residuals[(group, hydronic, h_label)].append(residual)
                if group == "post_off" and hours_since is not None:
                    prior = [e for e in off_end_lists[zone] if e <= t_out]
                    per_event[(zone, prior[-1], hydronic, h_label)].append(residual)
                    if hydronic and h_label in ("1h", "2h"):
                        decay_bins[int(hours_since * 2)].append(residual)

    print("── Row-level residuals (actual − predicted, °F): mean [n] ──")
    print(f"{'group':<12}{'sensors':<12}" + "".join(f"{h:>16}" for h in HORIZON_HOURS))
    for group in ("post_off", "heating_on", "cold_soak"):
        for hydronic in (True, False):
            cells = []
            for h_label in HORIZON_HOURS:
                vals = residuals.get((group, hydronic, h_label), [])
                cells.append(f"{np.mean(vals):+.2f} [{len(vals)}]" if vals else "--")
            kind = "hydronic" if hydronic else "own-split"
            print(f"{group:<12}{kind:<12}" + "".join(f"{c:>16}" for c in cells))

    print("\n── POST-OFF, aggregated per off-event (mean of per-event means) ──")
    for hydronic in (True, False):
        kind = "hydronic" if hydronic else "own-split"
        for h_label in HORIZON_HOURS:
            means = [np.mean(v) for (z, e, hy, hl), v in per_event.items() if hy == hydronic and hl == h_label]
            if len(means) < 3:
                continue
            arr = np.array(means)
            se = arr.std(ddof=1) / np.sqrt(len(arr))
            print(f"  {kind:<11} {h_label}: {arr.mean():+.3f} ± {1.96 * se:.3f} °F  ({len(arr)} events)")

    print("\n── Differential: post_off − cold_soak (same sensors, same horizon) ──")
    for hydronic in (True, False):
        kind = "hydronic" if hydronic else "own-split"
        cells = []
        for h_label in HORIZON_HOURS:
            po = residuals.get(("post_off", hydronic, h_label), [])
            cs = residuals.get(("cold_soak", hydronic, h_label), [])
            cells.append(f"{np.mean(po) - np.mean(cs):+.2f}" if po and cs else "--")
        print(f"  {kind:<11}" + "".join(f"{c:>10}" for c in cells))

    print("\n── NIGHT-ONLY (sun below horizon at outcome time): mean [n] ──")
    for group in ("post_off", "heating_on", "cold_soak"):
        for hydronic in (True, False):
            cells = []
            for h_label in HORIZON_HOURS:
                vals = night_residuals.get((group, hydronic, h_label), [])
                cells.append(f"{np.mean(vals):+.2f} [{len(vals)}]" if vals else "--")
            kind = "hydronic" if hydronic else "own-split"
            print(f"{group:<12}{kind:<12}" + "".join(f"{c:>16}" for c in cells))

    print("\n── Residual vs hours-since-off (hydronic, 1h+2h horizons) ──")
    for b in sorted(decay_bins):
        vals = decay_bins[b]
        lo, hi = b / 2, (b + 1) / 2
        bar = "#" * max(0, int(round(np.mean(vals) * 20)))
        print(f"  {lo:.1f}-{hi:.1f}h: {np.mean(vals):+.3f} °F [{len(vals):5d}] {bar}")


if __name__ == "__main__":
    main()
