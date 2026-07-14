"""Backfill snapshot DB gaps from HA REST API history.

When the collector goes down, the snapshot DB accumulates a gap. This script
fills it by fetching entity history from the HA REST API (which retains ~10
days of raw state data) and inserting EAV rows into the readings table.

The script reuses the same entity lists, column mappings, and series-to-column
logic as the control loop's fetch_recent_history, so the backfilled data is
identical to what the collector would have written.

Usage:
  uv run python scripts/backfill-collector-gap.py             # auto-detect gap and fill
  uv run python scripts/backfill-collector-gap.py --hours 48   # backfill last 48h
  uv run python scripts/backfill-collector-gap.py --dry-run    # show what would be written
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weatherstat.config import SNAPSHOTS_DB  # noqa: E402
from weatherstat.extract import (  # noqa: E402
    CLIMATE_ENTITIES,
    FAN_ENTITIES,
    NUMERIC_SENSOR_ENTITIES,
    SENSOR_ENTITIES,
    WEATHER_ENTITY,
    _check_config,
    _climate_to_series,
    _fan_to_series,
    _history_to_series,
    _weather_to_series,
    fetch_history,
    fetch_history_with_attributes,
)
from weatherstat.yaml_config import load_config  # noqa: E402

_CFG = load_config()


def _fetch_and_build_df(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch HA history for the given period and build a 5-min DataFrame.

    Same processing logic as fetch_recent_history in extract.py, minus the
    forecast fetch (can't backfill forecasts from the past).
    """
    # Round to 5-minute boundaries so timestamps match the collector format
    def _round_down_5m(dt: datetime) -> datetime:
        return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

    def _round_up_5m(dt: datetime) -> datetime:
        rounded = _round_down_5m(dt)
        return rounded if rounded == dt else rounded + timedelta(minutes=5)

    start = _round_down_5m(start)
    end = _round_up_5m(end)

    print("Fetching sensor history...")
    sensor_ids = list(NUMERIC_SENSOR_ENTITIES.values()) + list(SENSOR_ENTITIES.values())
    sensor_history = fetch_history(sensor_ids, start, end)

    print("Fetching climate/fan/weather history (with attributes)...")
    attr_ids = list(CLIMATE_ENTITIES.values()) + list(FAN_ENTITIES.values()) + [WEATHER_ENTITY]
    attr_history = fetch_history_with_attributes(attr_ids, start, end)

    # Build 5-minute time index
    time_index = pd.date_range(start=start, end=end, freq="5min", tz=UTC)
    result = pd.DataFrame(index=time_index)
    result.index.name = "timestamp"

    # Process temperature/numeric sensors
    for col_name, entity_id in {**NUMERIC_SENSOR_ENTITIES, **SENSOR_ENTITIES}.items():
        records = sensor_history.get(entity_id, [])
        if not records:
            result[col_name] = np.nan
            continue
        series = (
            _history_to_series(records, value_fn=lambda s: s)
            if col_name.endswith("_mode") or col_name in _CFG.state_sensors
            else _history_to_series(records)
        )
        series = series[~series.index.duplicated(keep="last")]
        result[col_name] = series.reindex(time_index, method="ffill")

    # Process climate entities
    for name, entity_id in CLIMATE_ENTITIES.items():
        records = attr_history.get(entity_id, [])
        if not records:
            continue
        series_dict = _climate_to_series(records)
        if name.startswith("thermostat_"):
            for suffix, series in [("target", series_dict["target"]), ("action", series_dict["action"])]:
                s = series[~series.index.duplicated(keep="last")]
                result[f"{name}_{suffix}"] = s.reindex(time_index, method="ffill")
        else:
            for suffix, series in [
                ("temp", series_dict["temp"]),
                ("target", series_dict["target"]),
                ("mode", series_dict["mode"]),
            ]:
                s = series[~series.index.duplicated(keep="last")]
                result[f"{name}_{suffix}"] = s.reindex(time_index, method="ffill")

    # Process fan entities
    for name, entity_id in FAN_ENTITIES.items():
        records = attr_history.get(entity_id, [])
        if not records:
            result[f"{name}_mode"] = "off"
            continue
        series = _fan_to_series(records)
        series = series[~series.index.duplicated(keep="last")]
        result[f"{name}_mode"] = series.reindex(time_index, method="ffill")

    # Process weather entity
    weather_records = attr_history.get(WEATHER_ENTITY, [])
    if weather_records:
        weather_dict = _weather_to_series(weather_records)
        _weather_col_map = {
            "condition": "weather_condition",
            "temperature": "met_outdoor_temp",
            "cloud_coverage": "cloud_coverage",
        }
        for suffix, series in weather_dict.items():
            col = _weather_col_map.get(suffix, f"outdoor_{suffix}")
            s = series[~series.index.duplicated(keep="last")]
            result[col] = s.reindex(time_index, method="ffill")

    # Process environment sensors. Mirrors collector._extract_snapshot (collector.py
    # § "Environment factors"): continuous covers use current_position/100 as openness;
    # binary entries use state == active_state. Attributes are required for
    # current_position, so this branch uses fetch_history_with_attributes (the
    # default fetch_history strips attributes via no_attributes=).
    env_attr_history = fetch_history_with_attributes(_CFG.environment_sensors, start, end)
    for _name, env_cfg in _CFG.environment.items():
        entity_id = env_cfg.entity_id
        col_name = env_cfg.column
        active_state = env_cfg.active_state
        is_continuous = env_cfg.value_type == "continuous"
        records = env_attr_history.get(entity_id, [])
        if not records:
            # No history: let the load-path ffill carry the prior state across the gap.
            continue

        times: list[pd.Timestamp] = []
        values: list[str] = []
        for rec in records:
            ts_str = str(rec.get("last_changed") or rec.get("last_updated", ""))
            state = str(rec.get("state", ""))
            if state in ("unavailable", "unknown"):
                continue
            try:
                ts = pd.to_datetime(ts_str, utc=True, format="ISO8601")
            except (ValueError, TypeError):
                continue

            if is_continuous:
                attrs = rec.get("attributes", {})
                if not isinstance(attrs, dict):
                    attrs = {}
                raw_pos = attrs.get("current_position")
                if raw_pos is not None:
                    try:
                        openness = float(raw_pos) / 100.0
                    except (ValueError, TypeError):
                        openness = 1.0 if state == "open" else 0.0
                else:
                    openness = 1.0 if state == "open" else 0.0
                active_frac = openness if active_state == "open" else 1.0 - openness
                values.append(str(round(active_frac, 2)))
            else:
                values.append("1" if state == active_state else "0")
            times.append(ts)

        if not times:
            continue

        s = pd.Series(values, index=pd.DatetimeIndex(times), dtype=object)
        s = s[~s.index.duplicated(keep="last")]
        result[col_name] = s.reindex(time_index, method="ffill")

    # Finalize
    result = result.reset_index()
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Ensure numeric columns
    numeric_cols = _CFG.numeric_extract_columns
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if "outdoor_wind_speed" in result.columns:
        result = result.rename(columns={"outdoor_wind_speed": "wind_speed"})

    return result


def _df_to_eav_rows(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Convert wide DataFrame to EAV (timestamp, name, value) rows.

    Skips NaN values — they represent sensors with no data in that period.
    """
    rows: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        for col in df.columns:
            if col == "timestamp":
                continue
            val = row[col]
            if val is None:
                continue
            if isinstance(val, float) and np.isnan(val):
                continue
            # Boolean environment columns
            if isinstance(val, (bool, np.bool_)):
                val = "1" if val else "0"
            rows.append((ts, col, str(val)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill snapshot DB gaps from HA history")
    parser.add_argument("--hours", type=float, default=None,
                        help="Hours to backfill (default: auto-detect gap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without writing")
    parser.add_argument("--db", type=Path, default=None,
                        help=f"Database path (default: {SNAPSHOTS_DB})")
    args = parser.parse_args()

    _check_config()

    db_path = args.db or SNAPSHOTS_DB
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    now = datetime.now(UTC)

    if args.hours:
        start = now - timedelta(hours=args.hours)
        end = now
        print(f"Backfilling last {args.hours:.0f}h: {start.isoformat()} to {end.isoformat()}")
    else:
        # Auto-detect gap
        row = conn.execute("SELECT MAX(timestamp) FROM readings").fetchone()
        if not row or not row[0]:
            print("No existing data in readings table. Use --hours to specify backfill range.")
            conn.close()
            sys.exit(1)

        latest_ts = row[0]
        latest_dt = datetime.fromisoformat(latest_ts.replace(".000Z", "+00:00"))
        gap_hours = (now - latest_dt).total_seconds() / 3600

        if gap_hours < 0.2:  # less than ~12 minutes
            print(f"Latest snapshot: {latest_ts} ({gap_hours * 60:.0f}m ago) — no gap to fill.")
            conn.close()
            return

        if gap_hours > 240:  # 10 days
            print(f"Gap is {gap_hours:.0f}h ({gap_hours / 24:.0f}d) — HA history only retains ~10 days.")
            print("Will backfill last 240h (10d). Older data is lost.")
            gap_hours = 240

        start = latest_dt
        end = now
        print(f"Gap detected: {latest_ts} to now ({gap_hours:.1f}h)")

    # Fetch and build DataFrame
    df = _fetch_and_build_df(start, end)
    print(f"Built {len(df)} rows × {len(df.columns) - 1} columns")

    # Convert to EAV
    eav_rows = _df_to_eav_rows(df)
    print(f"Total EAV rows: {len(eav_rows):,}")

    if args.dry_run:
        print("\n(dry-run — no database writes)")
        # Show sample
        if eav_rows:
            first_ts = eav_rows[0][0]
            last_ts = eav_rows[-1][0]
            n_ts = len({r[0] for r in eav_rows})
            n_cols = len({r[1] for r in eav_rows})
            print(f"  Timestamps: {n_ts}")
            print(f"  Columns: {n_cols}")
            print(f"  First: {first_ts}")
            print(f"  Last:  {last_ts}")
        return

    # Insert
    cursor = conn.cursor()
    inserted = 0
    for ts, name, value in eav_rows:
        cursor.execute(
            "INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)",
            (ts, name, value),
        )
        inserted += cursor.rowcount

    conn.commit()
    conn.close()

    skipped = len(eav_rows) - inserted
    print(f"\nInserted {inserted:,} new readings ({skipped:,} already existed)")


if __name__ == "__main__":
    main()
