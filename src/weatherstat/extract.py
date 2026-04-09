"""Home Assistant data access: live state fetch and collector SQLite reader."""

from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests

from weatherstat.config import HA_TOKEN, HA_URL, SNAPSHOTS_DB
from weatherstat.yaml_config import load_config

if TYPE_CHECKING:
    from weatherstat.forecast import ForecastEntry

_CFG = load_config()

# ── Entity definitions (from YAML config) ──────────────────────────────────

NUMERIC_SENSOR_ENTITIES: dict[str, str] = _CFG.numeric_sensor_entities
CLIMATE_ENTITIES: dict[str, str] = _CFG.climate_entities
FAN_ENTITIES: dict[str, str] = _CFG.fan_entities
SENSOR_ENTITIES: dict[str, str] = _CFG.sensor_entities
WEATHER_ENTITY = _CFG.weather_entity
ENVIRONMENT_SENSORS = _CFG.environment_sensors

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
    temperatures: list[float | None] = []
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

        t = attrs.get("temperature")
        temperatures.append(float(t) if t is not None else None)

        h = attrs.get("humidity")
        humidities.append(float(h) if h is not None else None)

        ws = attrs.get("wind_speed")
        wind_speeds.append(float(ws) if ws is not None else None)

    idx = pd.DatetimeIndex(times)
    return {
        "condition": pd.Series(conditions, index=idx, dtype=str),
        "temperature": pd.Series(temperatures, index=idx, dtype=float),
        "humidity": pd.Series(humidities, index=idx, dtype=float),
        "wind_speed": pd.Series(wind_speeds, index=idx, dtype=float),
    }


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

    Reads from the EAV ``readings`` table and pivots to wide format.

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
    df = _load_from_readings(conn)
    conn.close()

    for col in _CFG.environment_bool_columns:
        if col in df.columns:
            # Use nullable boolean to preserve NaN (sensor didn't exist yet)
            # so downstream code can distinguish "closed" from "unknown".
            df[col] = df[col].astype("boolean")

    return df


def snapshot_status(db_path: Path | None = None) -> tuple[str, int]:
    """Return (latest_timestamp, row_count) from the readings table.

    Lightweight query for TUI status display — avoids loading full DataFrame.
    Returns ("", 0) if the database doesn't exist or is empty.
    """
    path = db_path or SNAPSHOTS_DB
    if not path.exists():
        return ("", 0)
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT MAX(timestamp), COUNT(*) FROM readings").fetchone()
        if row and row[0]:
            return (str(row[0]), int(row[1]))
        return ("", 0)
    finally:
        conn.close()


def latest_snapshot_values(db_path: Path | None = None) -> dict[str, str]:
    """Return {name: value} for the most recent timestamp in readings.

    Lightweight EAV query for TUI — no pivot, no type coercion.
    """
    path = db_path or SNAPSHOTS_DB
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT MAX(timestamp) FROM readings").fetchone()
        if not row or not row[0]:
            return {}
        ts = row[0]
        rows = conn.execute("SELECT name, value FROM readings WHERE timestamp = ?", (ts,)).fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        conn.close()


# ── Live state fetch ─────────────────────────────────────────────────────────


def fetch_recent_history(hours_back: int = 14) -> tuple[pd.DataFrame, list[ForecastEntry] | None]:
    """Fetch recent entity history AND hourly forecast from HA.

    Returns (history_df, forecast_entries). Forecast is None on failure.
    """
    _check_config()

    end = datetime.now(UTC)
    start = end - timedelta(hours=hours_back)

    # Fetch sensor entities (no attributes needed)
    sensor_ids = list(NUMERIC_SENSOR_ENTITIES.values()) + list(SENSOR_ENTITIES.values())
    sensor_history = fetch_history(sensor_ids, start, end)

    # Fetch climate + fan + weather entities (with attributes)
    attr_ids = list(CLIMATE_ENTITIES.values()) + list(FAN_ENTITIES.values()) + [WEATHER_ENTITY]
    attr_history = fetch_history_with_attributes(attr_ids, start, end)

    # Fetch environment sensors (windows, doors, shades, etc.)
    env_history = fetch_history(ENVIRONMENT_SENSORS, start, end)

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
        series = _history_to_series(records, value_fn=lambda s: s) if col_name.endswith("_mode") or col_name in _CFG.state_sensors else _history_to_series(records)
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
        _weather_col_map = {"condition": "weather_condition", "temperature": "met_outdoor_temp"}
        for suffix, series in weather_dict.items():
            col = _weather_col_map.get(suffix, f"outdoor_{suffix}")
            s = series[~series.index.duplicated(keep="last")]
            result[col] = s.reindex(time_index, method="ffill")

    # Process environment sensors → per-entry columns
    for _name, env_cfg in _CFG.environment.items():
        entity_id = env_cfg.entity_id
        col_name = env_cfg.column
        active_state = env_cfg.active_state
        records = env_history.get(entity_id, [])
        if records:
            s = _history_to_series(records, value_fn=lambda s, _as=active_state: s == _as)
            s = s[~s.index.duplicated(keep="last")]
            result[col_name] = s.reindex(time_index, method="ffill").astype(bool)
        else:
            result[col_name] = False

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
