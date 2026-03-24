#!/usr/bin/env python3
"""Plot thermostat success: actual temperatures vs comfort bands over time.

Answers the question "is the system working as designed?" at a glance:
- Green band = comfort range, dashed = preferred
- Blue line = actual temperature
- Red/orange shading = violations
- Violation annotations distinguish "capacity exceeded" (system was trying
  its hardest) from "control opportunity" (system could have done more)

Usage:
    uv run python scripts/plot_comfort.py              # last 7 days, save PNG
    uv run python scripts/plot_comfort.py --days 3     # last 3 days
    uv run python scripts/plot_comfort.py --show       # interactive window
    uv run python scripts/plot_comfort.py --predictions # include prediction accuracy panel
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

# ── Data loading ───────────────────────────────────────────────────────────


def _init_weatherstat():
    """Add weatherstat to sys.path and return config + sim params."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))
    from weatherstat.simulator import load_sim_params
    from weatherstat.yaml_config import load_config
    return load_config(), load_sim_params()


def _resolve_paths() -> tuple[Path, Path, Path]:
    """Resolve data paths from weatherstat config module."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))
    from weatherstat.config import DATA_DIR, DECISION_LOG_DB, SNAPSHOTS_DB
    return SNAPSHOTS_DB, DECISION_LOG_DB, DATA_DIR


# Resolved lazily on first use
_PATHS: tuple[Path, Path, Path] | None = None


def _paths() -> tuple[Path, Path, Path]:
    global _PATHS  # noqa: PLW0603
    if _PATHS is None:
        _PATHS = _resolve_paths()
    return _PATHS


def load_sensor_data(days: int) -> pd.DataFrame:
    """Load 5-min sensor readings for the last N days from the EAV table."""
    snapshots_db = _paths()[0]
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(str(snapshots_db))
    df_long = pd.read_sql(
        "SELECT timestamp, name, value FROM readings WHERE timestamp >= ? ORDER BY timestamp",
        conn,
        params=(cutoff,),
    )
    conn.close()
    if df_long.empty:
        return pd.DataFrame()
    df = df_long.pivot(index="timestamp", columns="name", values="value")
    df.columns.name = None
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    for col in df.columns:
        if col != "timestamp" and col != "weather_condition":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_decisions(days: int) -> pd.DataFrame:
    """Load decision log entries for the last N days."""
    decision_log_db = _paths()[1]
    if not decision_log_db.exists():
        return pd.DataFrame()
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(str(decision_log_db))
    df = pd.read_sql(
        "SELECT * FROM decisions WHERE timestamp >= ? ORDER BY timestamp",
        conn,
        params=(cutoff,),
    )
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ── Comfort schedules ─────────────────────────────────────────────────────


def load_comfort_schedules(cfg) -> dict[str, list[dict]]:
    """Load comfort schedules from config.

    Returns: {label: [{start_hour, end_hour, min, max, preferred}, ...]}
    """
    schedules: dict[str, list[dict]] = {}
    for constraint in cfg.constraints:
        schedules[constraint.label] = [
            {
                "start_hour": e.start_hour,
                "end_hour": e.end_hour,
                "min": e.min_temp,
                "max": e.max_temp,
                "preferred": e.preferred,
            }
            for e in constraint.entries
        ]
    return schedules


def comfort_bounds_at_hour(schedule: list[dict], hour: int) -> dict | None:
    """Return the active comfort bounds for a given hour."""
    for entry in schedule:
        s, e = entry["start_hour"], entry["end_hour"]
        if s <= e:
            if s <= hour < e:
                return entry
        else:
            if hour >= s or hour < e:
                return entry
    return None


def compute_comfort_bands(
    timestamps: pd.Series,
    schedule: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute min/max/preferred arrays aligned to timestamps."""
    n = len(timestamps)
    mins = np.full(n, np.nan)
    maxs = np.full(n, np.nan)
    prefs = np.full(n, np.nan)
    for i, ts in enumerate(timestamps):
        local = ts.astimezone(LOCAL_TZ)
        bounds = comfort_bounds_at_hour(schedule, local.hour)
        if bounds:
            mins[i] = bounds["min"]
            maxs[i] = bounds["max"]
            prefs[i] = bounds["preferred"]
    return mins, maxs, prefs


# ── Capacity analysis ─────────────────────────────────────────────────────


def build_sensor_dedicated_effectors(cfg, params) -> dict[str, list[str]]:
    """For each constrained sensor, find the effectors that are 'dedicated' to it.

    An effector is dedicated to a sensor if it has the highest gain for that
    sensor among trajectory effectors, OR it's a regulating/binary effector
    whose name matches the sensor label (e.g., mini_split_bedroom for bedroom).

    Cross-talk gains (bedroom mini split has 0.093 gain to kitchen) are
    intentionally excluded — the optimizer balances those trade-offs, and
    showing them as "available capacity" is misleading.
    """
    label_to_sensor_col = {c.label: c.sensor for c in cfg.constraints}

    # Find best trajectory effector per sensor from coupling matrix
    trajectory_effectors = {n for n, e in cfg.effectors.items() if e.control_type == "trajectory"}
    best_trajectory: dict[str, str] = {}  # sensor_col -> effector_name
    for (effector, sensor_col), (gain, _lag) in params.gains.items():
        if effector not in trajectory_effectors or gain <= 0:
            continue
        prev_gain = params.gains.get((best_trajectory.get(sensor_col, ""), sensor_col), (0.0, 0.0))[0]
        if sensor_col not in best_trajectory or gain > prev_gain:
            best_trajectory[sensor_col] = effector

    dedicated: dict[str, list[str]] = {}
    for label, sensor_col in label_to_sensor_col.items():
        effs: list[str] = []

        # Zone thermostat (highest-gain trajectory effector)
        if sensor_col in best_trajectory:
            effs.append(best_trajectory[sensor_col])

        # Dedicated regulating/binary effectors (name matches label)
        for eff_name, eff_cfg in cfg.effectors.items():
            if eff_cfg.control_type in ("regulating", "binary"):
                # e.g., "mini_split_bedroom" -> "bedroom", "blower_office" -> "office"
                suffix = eff_name.split("_", 1)[1] if "_" in eff_name else eff_name
                if suffix == label:
                    effs.append(eff_name)

        dedicated[label] = effs

    return dedicated


def compute_capacity_utilization(
    decisions: pd.DataFrame,
    dedicated_effectors: dict[str, list[str]],
) -> dict[str, list[dict]]:
    """For each constrained sensor, compute per-decision capacity utilization.

    "At capacity" means all DEDICATED effectors for the sensor are at max.
    Cross-talk from other effectors (e.g., bedroom mini split warming the
    kitchen) is excluded — the optimizer already considers those trade-offs.

    Returns: {label: [{timestamp, temp, min, max, violation, at_capacity}]}
    """
    results: dict[str, list[dict]] = {}

    for _, row in decisions.iterrows():
        temps = json.loads(row["current_temps"]) if isinstance(row["current_temps"], str) else row["current_temps"]
        bounds = json.loads(row["comfort_bounds"]) if isinstance(row["comfort_bounds"], str) else row["comfort_bounds"]
        splits = json.loads(row["mini_splits"]) if isinstance(row["mini_splits"], str) else row["mini_splits"]
        blowers_raw = json.loads(row["blowers"]) if isinstance(row["blowers"], str) else row["blowers"]

        # Build current effector activity: effector_name -> activity (0 to 1)
        activity: dict[str, float] = {}
        activity["thermostat_upstairs"] = float(row.get("upstairs_heating", 0) or 0)
        activity["thermostat_downstairs"] = float(row.get("downstairs_heating", 0) or 0)

        for s in (splits or []):
            activity[f"mini_split_{s['name']}"] = 0.0 if s["mode"] == "off" else 1.0

        blower_levels = {"off": 0.0, "low": 0.5, "high": 1.0}
        for b in (blowers_raw or []):
            activity[f"blower_{b['name']}"] = blower_levels.get(b["mode"], 0.0)

        for label, effs in dedicated_effectors.items():
            if label not in bounds or label not in temps:
                continue

            temp = temps[label]
            min_t = bounds[label]["min"]
            max_t = bounds[label]["max"]

            # "At capacity" = all dedicated effectors are at max
            if effs:
                at_capacity = all(activity.get(e, 0.0) >= 0.99 for e in effs)
            else:
                # No dedicated effectors — entirely dependent on cross-talk
                at_capacity = True  # can't do any better

            violation = 0.0
            if temp < min_t:
                violation = min_t - temp
            elif temp > max_t:
                violation = -(temp - max_t)

            if label not in results:
                results[label] = []
            results[label].append({
                "timestamp": row["timestamp"],
                "temp": temp,
                "min": min_t,
                "max": max_t,
                "violation": violation,
                "at_capacity": at_capacity,
            })

    return results


# ── Statistics ────────────────────────────────────────────────────────────


def compute_stats(
    temps: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
) -> dict[str, float]:
    """Compute comfort statistics from 5-min sensor data."""
    valid = ~np.isnan(temps) & ~np.isnan(mins)
    if valid.sum() == 0:
        return {"pct_in_band": 0.0, "pct_below": 0.0, "pct_above": 0.0, "avg_violation": 0.0}

    t = temps[valid]
    lo = mins[valid]
    hi = maxs[valid]

    in_band = (t >= lo) & (t <= hi)
    below = t < lo
    above = t > hi
    violations = np.where(below, lo - t, np.where(above, t - hi, 0.0))

    return {
        "pct_in_band": in_band.mean() * 100,
        "pct_below": below.mean() * 100,
        "pct_above": above.mean() * 100,
        "avg_violation": violations[violations > 0].mean() if (violations > 0).any() else 0.0,
    }


def compute_capacity_stats(cap_rows: list[dict]) -> dict[str, int]:
    """Compute capacity-aware violation breakdown from decision-level data."""
    empty = {"cold_total": 0, "cold_capacity": 0, "cold_control": 0,
             "hot_total": 0, "hot_capacity": 0, "hot_control": 0}
    if not cap_rows:
        return empty

    cold = [r for r in cap_rows if r["violation"] > 0]
    hot = [r for r in cap_rows if r["violation"] < 0]

    return {
        "cold_total": len(cold),
        "cold_capacity": sum(1 for r in cold if r["at_capacity"]),
        "cold_control": sum(1 for r in cold if not r["at_capacity"]),
        "hot_total": len(hot),
        "hot_capacity": sum(1 for r in hot if r["at_capacity"]),
        "hot_control": sum(1 for r in hot if not r["at_capacity"]),
    }


# ── Prediction accuracy ──────────────────────────────────────────────────


def extract_prediction_errors(decisions: pd.DataFrame) -> pd.DataFrame:
    """Extract prediction errors from backfilled outcomes."""
    rows = []
    for _, dec in decisions.iterrows():
        if not dec.get("outcomes") or pd.isna(dec.get("outcomes")):
            continue
        outcomes = json.loads(dec["outcomes"]) if isinstance(dec["outcomes"], str) else dec["outcomes"]
        ts = dec["timestamp"]
        for label, horizons in outcomes.items():
            for horizon, data in horizons.items():
                if data.get("predicted") is not None and data.get("actual") is not None:
                    rows.append({
                        "timestamp": ts,
                        "label": label,
                        "horizon": horizon,
                        "predicted": data["predicted"],
                        "actual": data["actual"],
                        "error": data["error"],
                    })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────

SENSOR_ORDER = [
    "upstairs", "downstairs", "bedroom", "office", "office_bookshelf",
    "family_room", "kitchen", "piano", "bathroom",
]

HORIZON_COLORS = {"1h": "#2196F3", "2h": "#4CAF50", "4h": "#FF9800", "6h": "#F44336"}


def plot_comfort(
    days: int = 7,
    show: bool = False,
    include_predictions: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Generate the comfort success dashboard."""
    print(f"Loading {days} days of data...")
    cfg, params = _init_weatherstat()
    sensors_df = load_sensor_data(days)
    decisions = load_decisions(days)
    schedules = load_comfort_schedules(cfg)
    label_to_sensor = {c.label: c.sensor for c in cfg.constraints}

    if sensors_df.empty:
        print("No sensor data found!")
        raise SystemExit(1)

    # Capacity analysis: per-sensor dedicated effectors from coupling matrix
    dedicated_effectors = build_sensor_dedicated_effectors(cfg, params)
    capacity_data = compute_capacity_utilization(decisions, dedicated_effectors) if not decisions.empty else {}

    labels = [l for l in SENSOR_ORDER if l in schedules]

    # Layout: summary row + sensor panels + outdoor + optional prediction
    n_sensor_panels = len(labels)
    n_time_panels = n_sensor_panels + 1
    has_pred_panel = include_predictions and not decisions.empty
    n_panels = 1 + n_time_panels + (1 if has_pred_panel else 0)  # +1 for summary

    fig = plt.figure(figsize=(16, 2.2 * n_panels + 1))
    gs = fig.add_gridspec(
        n_panels, 1,
        height_ratios=[1.5] + [1] * n_time_panels + ([1] if has_pred_panel else []),
        hspace=0.15,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    # Share x-axis across time-series panels (not summary or prediction histogram)
    for ax in axes[2:1 + n_time_panels]:
        ax.sharex(axes[1])

    timestamps = sensors_df["timestamp"].dt.tz_convert(LOCAL_TZ)

    # ── Summary panel ──
    ax_summary = axes[0]
    _draw_summary(ax_summary, labels, sensors_df, label_to_sensor, schedules,
                  timestamps, capacity_data, days)

    # ── Per-sensor comfort panels ──
    all_stats: dict[str, dict] = {}

    for i, label in enumerate(labels):
        ax = axes[1 + i]
        sensor_col = label_to_sensor.get(label, label)
        schedule = schedules[label]

        if sensor_col not in sensors_df.columns:
            ax.set_ylabel(label, fontsize=9)
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            continue

        temps = sensors_df[sensor_col].values.astype(float)
        temps[temps < 40.0] = np.nan  # filter bogus sensor readings

        mins, maxs, prefs = compute_comfort_bands(timestamps, schedule)

        # Comfort band fill and preferred line
        ax.fill_between(timestamps, mins, maxs, alpha=0.12, color="#4CAF50")
        ax.plot(timestamps, prefs, color="#4CAF50", linewidth=0.7, alpha=0.5, linestyle="--")

        # Temperature trace
        ax.plot(timestamps, temps, color="#1976D2", linewidth=0.8, alpha=0.85)

        # Violation shading
        below_mask = temps < mins
        above_mask = temps > maxs
        if below_mask.any():
            ax.fill_between(timestamps, temps, mins, where=below_mask,
                            alpha=0.25, color="#F44336", interpolate=True)
        if above_mask.any():
            ax.fill_between(timestamps, temps, maxs, where=above_mask,
                            alpha=0.25, color="#FF9800", interpolate=True)

        stats = compute_stats(temps, mins, maxs)
        cap_stats = compute_capacity_stats(capacity_data.get(label, []))
        all_stats[label] = {**stats, **cap_stats}

        # Annotation: in-band % + capacity breakdown
        parts = [f"{stats['pct_in_band']:.0f}% in band"]
        if cap_stats["cold_total"] > 0:
            pct_cap = 100 * cap_stats["cold_capacity"] / cap_stats["cold_total"]
            parts.append(f"cold: {pct_cap:.0f}% capacity-limited")
        if cap_stats["hot_total"] > 0:
            pct_cap = 100 * cap_stats["hot_capacity"] / cap_stats["hot_total"]
            parts.append(f"hot: {pct_cap:.0f}% capacity-limited")
        ax.text(
            0.01, 0.93, "  |  ".join(parts),
            transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", alpha=0.15)

        # Tight y-axis
        valid_temps = temps[~np.isnan(temps)]
        valid_mins = mins[~np.isnan(mins)]
        valid_maxs = maxs[~np.isnan(maxs)]
        if len(valid_temps) > 0 and len(valid_mins) > 0:
            y_lo = min(valid_temps.min(), valid_mins.min()) - 1.5
            y_hi = max(valid_temps.max(), valid_maxs.max()) + 1.5
            ax.set_ylim(y_lo, y_hi)

    # ── Outdoor + effector panel ──
    ax_env = axes[1 + n_sensor_panels]

    if "outdoor_temp" in sensors_df.columns:
        outdoor = sensors_df["outdoor_temp"].values.astype(float)
        ax_env.plot(timestamps, outdoor, color="#795548", linewidth=0.8, label="outdoor")
        ax_env.set_ylabel("outdoor\n°F", fontsize=9, rotation=0, ha="right", va="center")

    if not decisions.empty:
        dec_ts = decisions["timestamp"].dt.tz_convert(LOCAL_TZ)
        ax_heat = ax_env.twinx()
        for zone, color, lname in [
            ("upstairs_heating", "#E53935", "up heat"),
            ("downstairs_heating", "#1E88E5", "dn heat"),
        ]:
            if zone in decisions.columns:
                vals = decisions[zone].fillna(0).astype(float)
                ax_heat.fill_between(dec_ts, 0, vals * 0.5, alpha=0.2, color=color,
                                     label=lname, step="post")
        ax_heat.set_ylim(0, 1)
        ax_heat.set_yticks([])
        ax_heat.legend(loc="upper right", fontsize=7, ncol=2)

    ax_env.grid(axis="x", alpha=0.3)
    ax_env.tick_params(axis="y", labelsize=8)

    # ── Prediction accuracy panel ──
    if has_pred_panel:
        ax_pred = axes[-1]
        errors_df = extract_prediction_errors(decisions)
        if not errors_df.empty:
            for horizon, color in HORIZON_COLORS.items():
                h_errors = errors_df[errors_df["horizon"] == horizon]["error"]
                if not h_errors.empty:
                    mae = h_errors.abs().mean()
                    bias = h_errors.mean()
                    ax_pred.hist(h_errors, bins=50, alpha=0.4, color=color,
                                label=f"{horizon}: MAE={mae:.2f}, bias={bias:+.2f}",
                                density=True)
            ax_pred.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax_pred.set_xlabel("Prediction error (predicted - actual, °F)", fontsize=9)
            ax_pred.set_ylabel("density", fontsize=9, rotation=0, ha="right", va="center")
            ax_pred.legend(fontsize=7)
            ax_pred.set_xlim(-5, 5)
            ax_pred.tick_params(labelsize=8)
            ax_pred.text(
                0.01, 0.92,
                f"{len(errors_df)} predictions | steering contaminates longer horizons",
                transform=ax_pred.transAxes, fontsize=7.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
        else:
            ax_pred.text(0.5, 0.5, "No backfilled outcomes", transform=ax_pred.transAxes, ha="center")

    # ── Formatting ──
    for ax in axes[1:1 + n_time_panels]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d", tz=LOCAL_TZ))
        ax.xaxis.set_major_locator(mdates.DayLocator(tz=LOCAL_TZ))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18], tz=LOCAL_TZ))
    plt.setp(axes[n_time_panels].xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

    # Night shading on time-series panels
    start_date = timestamps.min().normalize()
    end_date = timestamps.max().normalize() + timedelta(days=1)
    current = start_date
    while current <= end_date:
        for ax in axes[1:1 + n_time_panels]:
            evening = current.replace(hour=22)
            morning = (current + timedelta(days=1)).replace(hour=7)
            ax.axvspan(evening, morning, alpha=0.06, color="black", zorder=0)
        current += timedelta(days=1)

    if output_path is None:
        output_path = _paths()[2] / f"comfort_{days}d.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

    # Console summary
    _print_summary(labels, all_stats, dedicated_effectors, days)

    if show:
        plt.show()

    return output_path


def _draw_summary(
    ax, labels, sensors_df, label_to_sensor, schedules, timestamps, capacity_data, days,
):
    """Draw the top summary bar: per-sensor horizontal stacked bars."""
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of time", fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(f"Comfort Performance — last {days} days", fontsize=12, fontweight="bold", pad=8)

    bar_height = 0.6
    yticks = []
    ylabels_text = []

    for i, label in enumerate(labels):
        sensor_col = label_to_sensor.get(label, label)
        schedule = schedules[label]

        if sensor_col not in sensors_df.columns:
            yticks.append(i)
            ylabels_text.append(label)
            ax.barh(i, 100, height=bar_height, color="#E0E0E0", alpha=0.5)
            ax.text(50, i, "no data", ha="center", va="center", fontsize=8, color="#999")
            continue

        temps = sensors_df[sensor_col].values.astype(float)
        temps[temps < 40.0] = np.nan
        mins, maxs, _prefs = compute_comfort_bands(timestamps, schedule)

        valid = ~np.isnan(temps) & ~np.isnan(mins)
        if valid.sum() == 0:
            yticks.append(i)
            ylabels_text.append(label)
            continue

        t = temps[valid]
        lo = mins[valid]
        hi = maxs[valid]
        n_valid = len(t)

        pct_in = ((t >= lo) & (t <= hi)).sum() / n_valid * 100
        pct_below = (t < lo).sum() / n_valid * 100
        pct_above = (t > hi).sum() / n_valid * 100

        # Split violations by capacity
        cap_stats = compute_capacity_stats(capacity_data.get(label, []))

        pct_below_cap = 0.0
        pct_below_ctrl = 0.0
        if cap_stats["cold_total"] > 0:
            cap_frac = cap_stats["cold_capacity"] / cap_stats["cold_total"]
            pct_below_cap = pct_below * cap_frac
            pct_below_ctrl = pct_below * (1 - cap_frac)

        pct_above_cap = 0.0
        pct_above_ctrl = 0.0
        if cap_stats["hot_total"] > 0:
            cap_frac = cap_stats["hot_capacity"] / cap_stats["hot_total"]
            pct_above_cap = pct_above * cap_frac
            pct_above_ctrl = pct_above * (1 - cap_frac)

        # Draw stacked bar
        left = 0.0
        # In band (green)
        ax.barh(i, pct_in, left=left, height=bar_height, color="#4CAF50", alpha=0.7)
        left += pct_in
        # Cold: capacity-limited (gray-blue — not our fault)
        if pct_below_cap > 0:
            ax.barh(i, pct_below_cap, left=left, height=bar_height, color="#90A4AE", alpha=0.7)
            left += pct_below_cap
        # Cold: control opportunity (red — we could do better)
        if pct_below_ctrl > 0:
            ax.barh(i, pct_below_ctrl, left=left, height=bar_height, color="#F44336", alpha=0.7)
            left += pct_below_ctrl
        # Hot: capacity-limited (light orange)
        if pct_above_cap > 0:
            ax.barh(i, pct_above_cap, left=left, height=bar_height, color="#FFCC80", alpha=0.7)
            left += pct_above_cap
        # Hot: control opportunity (dark orange)
        if pct_above_ctrl > 0:
            ax.barh(i, pct_above_ctrl, left=left, height=bar_height, color="#FF9800", alpha=0.7)
            left += pct_above_ctrl

        # Percentage label
        ax.text(pct_in / 2, i, f"{pct_in:.0f}%", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white" if pct_in > 15 else "black")

        yticks.append(i)
        ylabels_text.append(label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels_text)
    ax.grid(axis="x", alpha=0.2)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#4CAF50", alpha=0.7, label="In comfort band"),
        mpatches.Patch(color="#90A4AE", alpha=0.7, label="Too cold — capacity exceeded"),
        mpatches.Patch(color="#F44336", alpha=0.7, label="Too cold — control opportunity"),
        mpatches.Patch(color="#FFCC80", alpha=0.7, label="Too hot — capacity exceeded"),
        mpatches.Patch(color="#FF9800", alpha=0.7, label="Too hot — control opportunity"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, ncol=3,
              framealpha=0.9)


def _print_summary(labels: list[str], all_stats: dict[str, dict],
                    dedicated_effectors: dict[str, list[str]], days: int):
    """Print console summary table."""
    overall_in = np.mean([s.get("pct_in_band", 0) for s in all_stats.values()]) if all_stats else 0
    print(f"\n{'='*90}")
    print(f"  Comfort Performance — last {days} days — avg {overall_in:.0f}% in band")
    print(f"{'='*90}")
    print(f"{'Sensor':<20} {'In Band':>8} {'Cold':>6} {'(cap)':>6} {'(ctrl)':>6} "
          f"{'Hot':>6} {'(cap)':>6} {'(ctrl)':>6}  {'Dedicated Effectors'}")
    print("-" * 90)

    for label in labels:
        s = all_stats.get(label, {})
        pct_in = s.get("pct_in_band", 0)
        pct_below = s.get("pct_below", 0)
        pct_above = s.get("pct_above", 0)

        cold_total = s.get("cold_total", 0)
        cold_cap = s.get("cold_capacity", 0)
        cold_ctrl = s.get("cold_control", 0)
        hot_total = s.get("hot_total", 0)
        hot_cap = s.get("hot_capacity", 0)
        hot_ctrl = s.get("hot_control", 0)

        def _frac(num, den):
            return f"{100*num/den:.0f}%" if den > 0 else "-"

        effs = dedicated_effectors.get(label, [])
        eff_str = ", ".join(e.replace("thermostat_", "t:").replace("mini_split_", "ms:").replace("blower_", "bl:")
                           for e in effs) if effs else "(cross-talk only)"

        print(
            f"{label:<20} {pct_in:>7.0f}% {pct_below:>5.1f}% "
            f"{_frac(cold_cap, cold_total):>6} {_frac(cold_ctrl, cold_total):>6} "
            f"{pct_above:>5.1f}% {_frac(hot_cap, hot_total):>6} {_frac(hot_ctrl, hot_total):>6}"
            f"  {eff_str}"
        )

    print(f"\n  cap = dedicated effectors at max  |  ctrl = dedicated effectors had headroom")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot thermostat comfort success")
    parser.add_argument("--days", type=int, default=7, help="Number of days to plot (default: 7)")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    parser.add_argument("--predictions", action="store_true", help="Include prediction accuracy panel")
    parser.add_argument("-o", "--output", type=str, help="Output PNG path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    plot_comfort(days=args.days, show=args.show, include_predictions=args.predictions, output_path=output)
