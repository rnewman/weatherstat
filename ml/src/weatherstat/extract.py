"""Historical data extraction from Home Assistant.

Two extraction modes:
1. Statistics mode (long-term): Hourly temperature statistics via WebSocket API (5+ months)
2. History mode (short-term): Raw state changes via REST API (~10 days), resampled to 5 min

Run: uv run python -m weatherstat.extract
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
import websockets.client

from weatherstat.config import HA_TOKEN, HA_URL, SNAPSHOTS_DB, SNAPSHOTS_DIR
from weatherstat.yaml_config import load_config

if TYPE_CHECKING:
    from weatherstat.forecast import ForecastEntry

_CFG = load_config()

# ── Entity definitions (from YAML config) ──────────────────────────────────

STATISTICS_ENTITIES: dict[str, str] = _CFG.statistics_entities
CLIMATE_ENTITIES: dict[str, str] = _CFG.climate_entities
FAN_ENTITIES: dict[str, str] = _CFG.fan_entities
SENSOR_ENTITIES: dict[str, str] = _CFG.sensor_entities
WEATHER_ENTITY = _CFG.weather_entity
WINDOW_SENSORS = _CFG.window_sensors

ALL_HISTORY_ENTITIES: list[str] = _CFG.all_history_entities


# ── HA API helpers ───────────────────────────────────────────────────────────


def _check_config() -> None:
    if not HA_URL or not HA_TOKEN:
        print("Error: HA_URL and HA_TOKEN must be set in environment.", file=sys.stderr)
        print("       Source your .env file or export them directly.", file=sys.stderr)
        sys.exit(1)


def _rest_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def fetch_history(
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[dict[str, str]]]:
    """Fetch raw state history for entities via HA REST API.

    Returns {entity_id: [{state, last_changed, attributes}, ...]}.
    """
    url = f"{HA_URL}/api/history/period/{start.isoformat()}"
    params = {
        "end_time": end.isoformat(),
        "filter_entity_id": ",".join(entity_ids),
        "minimal_response": "",
        "no_attributes": "",
    }

    resp = requests.get(url, headers=_rest_headers(), params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    result: dict[str, list[dict[str, str]]] = {}
    for entity_history in data:
        if not entity_history:
            continue
        entity_id = entity_history[0].get("entity_id", "")
        result[entity_id] = entity_history
    return result


def fetch_history_with_attributes(
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[dict[str, object]]]:
    """Fetch raw state history WITH attributes for climate/fan/weather entities."""
    url = f"{HA_URL}/api/history/period/{start.isoformat()}"
    params = {
        "end_time": end.isoformat(),
        "filter_entity_id": ",".join(entity_ids),
    }

    resp = requests.get(url, headers=_rest_headers(), params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    result: dict[str, list[dict[str, object]]] = {}
    for entity_history in data:
        if not entity_history:
            continue
        entity_id = str(entity_history[0].get("entity_id", ""))
        result[entity_id] = entity_history
    return result


async def fetch_statistics(
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    period: str = "hour",
) -> dict[str, list[dict[str, object]]]:
    """Fetch long-term statistics via HA WebSocket API.

    Returns {statistic_id: [{start, mean, min, max}, ...]}.
    """
    ws_url = HA_URL.replace("http", "ws", 1) + "/api/websocket"

    async with websockets.client.connect(ws_url, max_size=2**24) as ws:
        # Wait for auth_required
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_required":
            raise RuntimeError(f"Unexpected WS message: {msg}")

        # Authenticate
        await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"Auth failed: {msg}")

        # Request statistics
        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "type": "recorder/statistics_during_period",
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "statistic_ids": entity_ids,
                    "period": period,
                    "types": ["mean", "min", "max"],
                }
            )
        )

        msg = json.loads(await ws.recv())
        if not msg.get("success"):
            raise RuntimeError(f"Statistics request failed: {msg}")

        return msg.get("result", {})


# ── Statistics extraction (long-term hourly) ─────────────────────────────────


def extract_statistics(months_back: int = 7) -> pd.DataFrame:
    """Extract hourly temperature statistics for 5+ months.

    Produces a DataFrame with hourly rows, one column per sensor (mean values).
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=months_back * 30)

    entity_ids = list(STATISTICS_ENTITIES.values())
    col_by_entity = {v: k for k, v in STATISTICS_ENTITIES.items()}

    print(f"Fetching statistics from {start.date()} to {end.date()}...")
    print(f"  Entities: {len(entity_ids)}")

    stats = asyncio.run(fetch_statistics(entity_ids, start, end, period="hour"))

    # Build per-entity DataFrames and merge on timestamp
    frames: list[pd.DataFrame] = []
    for entity_id, records in stats.items():
        col_name = col_by_entity.get(entity_id, entity_id)
        if not records:
            print(f"  Warning: no statistics for {entity_id}")
            continue

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["start"], unit="ms", utc=True)
        df = df.rename(columns={"mean": col_name})
        df = df[["timestamp", col_name]]
        frames.append(df)
        print(f"  {entity_id}: {len(df)} hourly records")

    if not frames:
        print("Error: no statistics data retrieved.", file=sys.stderr)
        sys.exit(1)

    # Merge all on timestamp
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Convert timestamp to string for consistency with snapshot format
    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    print(f"\nStatistics dataset: {len(merged)} rows, {len(merged.columns)} columns")
    null_pct = merged.isnull().mean() * 100
    for col in merged.columns:
        if col != "timestamp" and null_pct[col] > 0:
            print(f"  {col}: {null_pct[col]:.1f}% null")

    return merged


# ── History extraction (short-term full features) ────────────────────────────


def _history_to_series(
    records: list[dict[str, str]],
    value_fn: object = None,
) -> pd.Series:
    """Convert HA history records to a pandas Series indexed by datetime.

    value_fn: optional callable to transform state string to a value.
              Defaults to float conversion (NaN on failure).
    """
    if value_fn is None:

        def value_fn(s: str) -> float | None:
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

    times = []
    values = []
    for rec in records:
        ts_str = rec.get("last_changed") or rec.get("last_updated", "")
        state = rec.get("state", "")
        if state in ("unavailable", "unknown"):
            continue
        try:
            ts = pd.to_datetime(ts_str, utc=True)
        except (ValueError, TypeError):
            continue
        val = value_fn(state)  # type: ignore[operator]
        times.append(ts)
        values.append(val)

    return pd.Series(values, index=pd.DatetimeIndex(times), dtype=object)


def _climate_to_series(
    records: list[dict[str, object]],
) -> dict[str, pd.Series]:
    """Extract climate entity attributes into separate Series."""
    times: list[pd.Timestamp] = []
    states: list[str] = []
    temps: list[float | None] = []
    targets: list[float | None] = []
    actions: list[str] = []

    for rec in records:
        ts_str = str(rec.get("last_changed") or rec.get("last_updated", ""))
        state = str(rec.get("state", ""))
        if state in ("unavailable", "unknown"):
            continue
        try:
            ts = pd.to_datetime(ts_str, utc=True)
        except (ValueError, TypeError):
            continue

        attrs = rec.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        times.append(ts)
        states.append(state)

        ct = attrs.get("current_temperature")
        temps.append(float(ct) if ct is not None else None)

        tgt = attrs.get("temperature")
        targets.append(float(tgt) if tgt is not None else None)

        action = attrs.get("hvac_action", "idle")
        actions.append(str(action))

    idx = pd.DatetimeIndex(times)
    return {
        "mode": pd.Series(states, index=idx, dtype=str),
        "temp": pd.Series(temps, index=idx, dtype=float),
        "target": pd.Series(targets, index=idx, dtype=float),
        "action": pd.Series(actions, index=idx, dtype=str),
    }


def _fan_to_series(records: list[dict[str, object]]) -> pd.Series:
    """Extract fan entity state + preset_mode into a mode Series."""
    times: list[pd.Timestamp] = []
    modes: list[str] = []

    for rec in records:
        ts_str = str(rec.get("last_changed") or rec.get("last_updated", ""))
        state = str(rec.get("state", ""))
        if state in ("unavailable", "unknown"):
            continue
        try:
            ts = pd.to_datetime(ts_str, utc=True)
        except (ValueError, TypeError):
            continue

        if state == "off":
            mode = "off"
        else:
            attrs = rec.get("attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}
            preset = attrs.get("preset_mode", "low")
            mode = str(preset) if preset else "low"

        times.append(ts)
        modes.append(mode)

    return pd.Series(modes, index=pd.DatetimeIndex(times), dtype=str)


def _weather_to_series(
    records: list[dict[str, object]],
) -> dict[str, pd.Series]:
    """Extract weather entity state and attributes."""
    times: list[pd.Timestamp] = []
    conditions: list[str] = []
    humidities: list[float | None] = []
    wind_speeds: list[float | None] = []

    for rec in records:
        ts_str = str(rec.get("last_changed") or rec.get("last_updated", ""))
        state = str(rec.get("state", ""))
        if state in ("unavailable", "unknown"):
            continue
        try:
            ts = pd.to_datetime(ts_str, utc=True)
        except (ValueError, TypeError):
            continue

        attrs = rec.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        times.append(ts)
        conditions.append(state)

        h = attrs.get("humidity")
        humidities.append(float(h) if h is not None else None)

        ws = attrs.get("wind_speed")
        wind_speeds.append(float(ws) if ws is not None else None)

    idx = pd.DatetimeIndex(times)
    return {
        "condition": pd.Series(conditions, index=idx, dtype=str),
        "humidity": pd.Series(humidities, index=idx, dtype=float),
        "wind_speed": pd.Series(wind_speeds, index=idx, dtype=float),
    }


def extract_history(days_back: int = 10) -> pd.DataFrame:
    """Extract raw state history and resample to 5-minute intervals.

    Produces a DataFrame matching the full snapshot schema.
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=days_back)

    print(f"Fetching raw history from {start.date()} to {end.date()}...")

    # 1. Fetch sensor entities (no attributes needed)
    sensor_ids = list(STATISTICS_ENTITIES.values()) + list(SENSOR_ENTITIES.values())
    print(f"  Fetching {len(sensor_ids)} sensor entities...")
    sensor_history = fetch_history(sensor_ids, start, end)

    # 2. Fetch climate + fan + weather entities (with attributes)
    attr_ids = list(CLIMATE_ENTITIES.values()) + list(FAN_ENTITIES.values()) + [WEATHER_ENTITY]
    print(f"  Fetching {len(attr_ids)} attribute entities...")
    attr_history = fetch_history_with_attributes(attr_ids, start, end)

    # 3. Fetch window sensors
    print(f"  Fetching {len(WINDOW_SENSORS)} window sensors...")
    window_history = fetch_history(WINDOW_SENSORS, start, end)

    # Build 5-minute time index
    freq = "5min"
    time_index = pd.date_range(start=start, end=end, freq=freq, tz=UTC)

    # Initialize result DataFrame
    result = pd.DataFrame(index=time_index)
    result.index.name = "timestamp"

    # 4. Process temperature/numeric sensors → resample with ffill
    for col_name, entity_id in {**STATISTICS_ENTITIES, **SENSOR_ENTITIES}.items():
        records = sensor_history.get(entity_id, [])
        if not records:
            print(f"    Warning: no history for {entity_id}")
            result[col_name] = np.nan
            continue

        if col_name.endswith("_mode"):
            # String sensor (boiler mode) — keep as text
            series = _history_to_series(records, value_fn=lambda s: s)
        else:
            series = _history_to_series(records)

        # Deduplicate index
        series = series[~series.index.duplicated(keep="last")]
        # Reindex to 5-min intervals with forward fill
        result[col_name] = series.reindex(time_index, method="ffill")

    # 5. Process climate entities
    for name, entity_id in CLIMATE_ENTITIES.items():
        records = attr_history.get(entity_id, [])
        if not records:
            print(f"    Warning: no history for {entity_id}")
            continue

        series_dict = _climate_to_series(records)

        if name.startswith("thermostat_"):
            # Thermostats: temp comes from the separate sensor, so just target + action
            for suffix, series in [("target", series_dict["target"]), ("action", series_dict["action"])]:
                s = series[~series.index.duplicated(keep="last")]
                result[f"{name}_{suffix}"] = s.reindex(time_index, method="ffill")
        else:
            # Mini splits: state=mode, plus temp, target
            for suffix, series in [
                ("temp", series_dict["temp"]),
                ("target", series_dict["target"]),
                ("mode", series_dict["mode"]),
            ]:
                s = series[~series.index.duplicated(keep="last")]
                result[f"{name}_{suffix}"] = s.reindex(time_index, method="ffill")

    # 6. Process fan entities
    for name, entity_id in FAN_ENTITIES.items():
        records = attr_history.get(entity_id, [])
        if not records:
            print(f"    Warning: no history for {entity_id}")
            result[f"{name}_mode"] = "off"
            continue

        series = _fan_to_series(records)
        series = series[~series.index.duplicated(keep="last")]
        result[f"{name}_mode"] = series.reindex(time_index, method="ffill")

    # 7. Process weather entity
    weather_records = attr_history.get(WEATHER_ENTITY, [])
    if weather_records:
        weather_dict = _weather_to_series(weather_records)
        for suffix, series in weather_dict.items():
            col = f"outdoor_{suffix}" if suffix != "condition" else "weather_condition"
            s = series[~series.index.duplicated(keep="last")]
            result[col] = s.reindex(time_index, method="ffill")

    # 8. Process window sensors → per-window columns + any_window_open
    window_column_map = _CFG.window_column_map
    window_cols_present: list[str] = []
    for entity_id, col_name in window_column_map.items():
        records = window_history.get(entity_id, [])
        if records:
            s = _history_to_series(records, value_fn=lambda s: s == "on")
            s = s[~s.index.duplicated(keep="last")]
            result[col_name] = s.reindex(time_index, method="ffill").astype(bool)
            window_cols_present.append(col_name)
        else:
            result[col_name] = False

    if window_cols_present:
        result["any_window_open"] = result[window_cols_present].any(axis=1).astype(bool)
    else:
        result["any_window_open"] = False

    # Reset index to get timestamp as a column
    result = result.reset_index()
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Ensure numeric columns are float (from YAML config)
    numeric_cols = _CFG.numeric_extract_columns
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Rename outdoor_wind_speed → wind_speed for consistency with snapshot schema
    if "outdoor_wind_speed" in result.columns:
        result = result.rename(columns={"outdoor_wind_speed": "wind_speed"})

    print(f"\nHistory dataset: {len(result)} rows, {len(result.columns)} columns")
    null_pct = result.isnull().mean() * 100
    for col in result.columns:
        if col != "timestamp" and null_pct[col] > 5:
            print(f"  {col}: {null_pct[col]:.1f}% null")

    return result


# ── Collector SQLite reader ───────────────────────────────────────────────


def _load_from_readings(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load from the EAV readings table and pivot to wide format."""
    df_long = pd.read_sql(
        "SELECT timestamp, name, value FROM readings ORDER BY timestamp", conn
    )
    if df_long.empty:
        return pd.DataFrame()

    df = df_long.pivot(index="timestamp", columns="name", values="value")
    df.columns.name = None  # remove "name" label from columns axis

    # Apply types from config
    col_types = _CFG.column_types
    for col in df.columns:
        sql_type = col_types.get(col)
        if sql_type in ("REAL", "INTEGER"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index()


def load_collector_snapshots(db_path: Path | None = None) -> pd.DataFrame:
    """Load all collector snapshots from the SQLite database.

    Reads from the EAV ``readings`` table if it exists and has data,
    otherwise falls back to the legacy wide ``snapshots`` table.

    Args:
        db_path: Path to snapshots.db. Defaults to SNAPSHOTS_DB.

    Returns:
        DataFrame with all snapshots in chronological order.
    """
    path = db_path or SNAPSHOTS_DB
    if not path.exists():
        print(f"No snapshot database found at {path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(path))

    # Check if readings table exists and has data
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "readings" in tables:
        count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        if count > 0:
            df = _load_from_readings(conn)
            conn.close()
            for col in _CFG.window_bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
            return df

    # Fallback to legacy wide table
    df = pd.read_sql("SELECT * FROM snapshots ORDER BY timestamp", conn)
    conn.close()

    for col in _CFG.window_bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df


# ── Live state fetch ─────────────────────────────────────────────────────────


def fetch_recent_history(hours_back: int = 14) -> tuple[pd.DataFrame, list[ForecastEntry] | None]:
    """Fetch recent entity history AND hourly forecast from HA.

    Returns (history_df, forecast_entries). Forecast is None on failure.
    """
    _check_config()

    end = datetime.now(UTC)
    start = end - timedelta(hours=hours_back)

    # Fetch sensor entities (no attributes needed)
    sensor_ids = list(STATISTICS_ENTITIES.values()) + list(SENSOR_ENTITIES.values())
    sensor_history = fetch_history(sensor_ids, start, end)

    # Fetch climate + fan + weather entities (with attributes)
    attr_ids = list(CLIMATE_ENTITIES.values()) + list(FAN_ENTITIES.values()) + [WEATHER_ENTITY]
    attr_history = fetch_history_with_attributes(attr_ids, start, end)

    # Fetch window sensors
    window_history = fetch_history(WINDOW_SENSORS, start, end)

    # Build 5-minute time index
    time_index = pd.date_range(start=start, end=end, freq="5min", tz=UTC)
    result = pd.DataFrame(index=time_index)
    result.index.name = "timestamp"

    # Process temperature/numeric sensors
    for col_name, entity_id in {**STATISTICS_ENTITIES, **SENSOR_ENTITIES}.items():
        records = sensor_history.get(entity_id, [])
        if not records:
            result[col_name] = np.nan
            continue
        if col_name.endswith("_mode"):
            series = _history_to_series(records, value_fn=lambda s: s)
        else:
            series = _history_to_series(records)
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
        for suffix, series in weather_dict.items():
            col = f"outdoor_{suffix}" if suffix != "condition" else "weather_condition"
            s = series[~series.index.duplicated(keep="last")]
            result[col] = s.reindex(time_index, method="ffill")

    # Process window sensors → per-window columns + any_window_open
    window_column_map = _CFG.window_column_map
    window_cols_present: list[str] = []
    for entity_id, col_name in window_column_map.items():
        records = window_history.get(entity_id, [])
        if records:
            s = _history_to_series(records, value_fn=lambda s: s == "on")
            s = s[~s.index.duplicated(keep="last")]
            result[col_name] = s.reindex(time_index, method="ffill").astype(bool)
            window_cols_present.append(col_name)
        else:
            result[col_name] = False

    if window_cols_present:
        result["any_window_open"] = result[window_cols_present].any(axis=1).astype(bool)
    else:
        result["any_window_open"] = False

    # Finalize
    result = result.reset_index()
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Ensure numeric columns (from YAML config)
    numeric_cols = _CFG.numeric_extract_columns
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if "outdoor_wind_speed" in result.columns:
        result = result.rename(columns={"outdoor_wind_speed": "wind_speed"})

    # Fetch hourly forecast (non-fatal)
    forecast: list[ForecastEntry] | None = None
    try:
        from weatherstat.forecast import fetch_forecast

        forecast = fetch_forecast()
        print(f"  Fetched {len(forecast)} forecast entries")
    except Exception as e:
        print(f"  Warning: forecast fetch failed: {e}")

    return result, forecast


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract historical data from Home Assistant")
    parser.add_argument(
        "--mode",
        choices=["all", "statistics", "history"],
        default="all",
        help="Extraction mode (default: all)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=7,
        help="Months of statistics to fetch (default: 7)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Days of raw history to fetch (default: 10)",
    )
    args = parser.parse_args()

    _check_config()
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "statistics"):
        print("=" * 60)
        print("EXTRACTING LONG-TERM STATISTICS (hourly)")
        print("=" * 60)
        stats_df = extract_statistics(months_back=args.months)
        out_path = SNAPSHOTS_DIR / "historical_hourly.parquet"
        stats_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}\n")

    if args.mode in ("all", "history"):
        print("=" * 60)
        print("EXTRACTING RAW HISTORY (5-min intervals)")
        print("=" * 60)
        hist_df = extract_history(days_back=args.days)
        out_path = SNAPSHOTS_DIR / "historical_full.parquet"
        hist_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}\n")

    print("Extraction complete.")


if __name__ == "__main__":
    main()
