"""Snapshot collector — reads HA entity states and writes to SQLite EAV.

Periodically fetches all monitored entity states via the HA REST API,
extracts column values driven by weatherstat.yaml, fetches weather forecasts,
and writes readings to the EAV table (timestamp, name, value).
"""

from __future__ import annotations

import os
import signal
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta

import requests

from weatherstat.config import SNAPSHOT_INTERVAL_SECONDS, SNAPSHOTS_DB
from weatherstat.yaml_config import load_config

_CFG = load_config()

_CREATE_READINGS_SQL = """\
CREATE TABLE IF NOT EXISTS readings (
  timestamp TEXT NOT NULL,
  name      TEXT NOT NULL,
  value     TEXT NOT NULL,
  PRIMARY KEY (timestamp, name)
)
"""


# ── HA REST helpers ─────────────────────────────────────────────────────────


def _ha_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('HA_TOKEN', '')}",
        "Content-Type": "application/json",
    }


def _ha_url() -> str:
    return os.environ.get("HA_URL", "")


def _fetch_states(entity_ids: list[str]) -> dict[str, dict]:
    """Fetch current state for each entity via REST API.

    Returns {entity_id: {state, attributes, last_changed, last_updated}}.
    """
    url = _ha_url()
    headers = _ha_headers()
    result: dict[str, dict] = {}
    # Batch fetch: GET /api/states returns all entities; filter client-side
    resp = requests.get(f"{url}/api/states", headers=headers, timeout=30)
    resp.raise_for_status()
    all_states = resp.json()
    wanted = set(entity_ids)
    for state in all_states:
        eid = state.get("entity_id", "")
        if eid in wanted:
            result[eid] = state
    return result


def _fetch_forecast(weather_entity: str) -> list[dict]:
    """Fetch hourly weather forecast via HA service call.

    Returns list of forecast entries sorted by datetime.
    """
    url = _ha_url()
    headers = _ha_headers()
    try:
        resp = requests.post(
            f"{url}/api/services/weather/get_forecasts?return_response",
            headers=headers,
            json={"entity_id": weather_entity, "type": "hourly"},
            timeout=30,
        )
        resp.raise_for_status()
        # REST API wraps response: {service_response: {entity_id: {forecast: [...]}}}
        data = resp.json()
        service_resp = data.get("service_response", data)
        entity_data = service_resp.get(weather_entity, {})
        entries = entity_data.get("forecast", []) if isinstance(entity_data, dict) else []
        return sorted(entries, key=lambda e: e.get("datetime", ""))
    except Exception as e:
        print(f"[collector] Forecast fetch failed (non-fatal): {e}", file=sys.stderr)
        return []


# ── Snapshot extraction ─────────────────────────────────────────────────────


def _extract_snapshot(states: dict[str, dict]) -> dict[str, str]:
    """Extract column values from HA entity states.

    Returns {column_name: string_value} for all snapshot columns.
    Driven by weatherstat.yaml column definitions.
    """
    values: dict[str, str] = {}

    def _attr_num(entity_id: str, attr: str, fallback: float = 0) -> str:
        entity = states.get(entity_id)
        if entity:
            val = entity.get("attributes", {}).get(attr)
            if isinstance(val, (int, float)):
                return str(val)
        return str(fallback)

    def _attr_str(entity_id: str, attr: str, fallback: str = "") -> str:
        entity = states.get(entity_id)
        if entity:
            val = entity.get("attributes", {}).get(attr)
            if isinstance(val, str):
                return val
        return fallback

    def _sensor_num(entity_id: str, fallback: float = 0) -> str:
        entity = states.get(entity_id)
        if entity:
            try:
                val = float(entity.get("state", ""))
                if val == val:  # not NaN
                    return str(val)
            except (ValueError, TypeError):
                pass
        return str(fallback)

    def _entity_state(entity_id: str, fallback: str = "") -> str:
        entity = states.get(entity_id)
        return entity.get("state", fallback) if entity else fallback

    def _binary_state(entity_id: str) -> str:
        entity = states.get(entity_id)
        return "1" if entity and entity.get("state") == "on" else "0"

    def _blower_mode(entity_id: str) -> str:
        entity = states.get(entity_id)
        if not entity or entity.get("state") == "off":
            return "off"
        preset = entity.get("attributes", {}).get("preset_mode")
        return str(preset) if isinstance(preset, str) else "low"

    # 1. Effectors
    for name, cfg in _CFG.effectors.items():
        if cfg.domain == "climate":
            values[f"{name}_temp"] = _attr_num(cfg.entity_id, "current_temperature")
            values[f"{name}_target"] = _attr_num(cfg.entity_id, "temperature")
            if cfg.mode_control == "manual":
                values[f"{name}_action"] = _attr_str(cfg.entity_id, "hvac_action", "idle")
            else:
                values[f"{name}_mode"] = _entity_state(cfg.entity_id, "off")
                if cfg.command_encoding and cfg.state_encoding != cfg.command_encoding:
                    values[f"{name}_action"] = _attr_str(cfg.entity_id, "hvac_action", "off")
        else:
            values[f"{name}_mode"] = _blower_mode(cfg.entity_id)

    # 2. State sensors (categorical)
    for col, sensor_cfg in _CFG.state_sensors.items():
        values[col] = _entity_state(sensor_cfg.entity_id)

    # 3. Power sensors (numeric)
    for col, sensor_cfg in _CFG.power_sensors.items():
        values[col] = _sensor_num(sensor_cfg.entity_id)

    # 4. Weather entity
    weather_id = _CFG.weather_entity
    weather = states.get(weather_id)
    if weather:
        attrs = weather.get("attributes", {})
        values["met_outdoor_temp"] = str(attrs.get("temperature", 0))
        values["outdoor_humidity"] = str(attrs.get("humidity", 0))
        values["wind_speed"] = str(attrs.get("wind_speed", 0))
        values["weather_condition"] = weather.get("state", "unknown")
        cc = attrs.get("cloud_coverage")
        if cc is not None:
            values["cloud_coverage"] = str(cc)
    else:
        values["met_outdoor_temp"] = "0"
        values["outdoor_humidity"] = "0"
        values["wind_speed"] = "0"
        values["weather_condition"] = "unknown"

    # 5. Outdoor temp sensor (if configured, separate from weather entity)
    outdoor_col = _CFG.outdoor_sensor
    if outdoor_col:
        for col_name, sensor_cfg in _CFG.temp_sensors.items():
            if col_name == outdoor_col:
                values[outdoor_col] = _sensor_num(sensor_cfg.entity_id)
                break

    # 6. Humidity sensors
    for col, sensor_cfg in _CFG.humidity_sensors.items():
        values[col] = _sensor_num(sensor_cfg.entity_id)

    # 7. Windows
    any_open = False
    for name, win_cfg in _CFG.windows.items():
        is_open = _binary_state(win_cfg.entity_id)
        values[f"window_{name}_open"] = is_open
        if is_open == "1":
            any_open = True
    values["any_window_open"] = "1" if any_open else "0"

    # 8. Per-room temperature sensors (non-thermostat, non-outdoor)
    for col, sensor_cfg in _CFG.temp_sensors.items():
        if col != outdoor_col and not col.startswith("thermostat_"):
            values[col] = _sensor_num(sensor_cfg.entity_id)

    return values


def _inject_forecast(values: dict[str, str], forecast: list[dict]) -> None:
    """Inject hourly forecast data into the snapshot values dict."""
    if not forecast:
        return

    now = datetime.now(UTC)

    def _find_closest(hours_ahead: int) -> dict | None:
        target = now + timedelta(hours=hours_ahead)
        best = None
        best_diff = float("inf")
        for entry in forecast:
            try:
                dt = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
                diff = abs((dt - target).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best = entry
            except (KeyError, ValueError):
                continue
        return best if best_diff <= 5400 else None  # 90 minutes

    # Hourly temps (1h through 12h)
    for h in range(1, 13):
        entry = _find_closest(h)
        values[f"forecast_temp_{h}h"] = str(entry.get("temperature", 0)) if entry else "0"

    # Condition, wind, and cloud coverage at key horizons
    for h in (1, 2, 4, 6, 12):
        entry = _find_closest(h)
        values[f"forecast_condition_{h}h"] = entry.get("condition", "") if entry else ""
        values[f"forecast_wind_{h}h"] = str(entry.get("wind_speed", 0)) if entry else "0"
        cc = entry.get("cloud_coverage") if entry else None
        if cc is not None:
            values[f"forecast_cloud_{h}h"] = str(cc)


# ── Timestamp rounding ──────────────────────────────────────────────────────


def _round_to_five_minutes(dt: datetime) -> str:
    """Round to the nearest 5-minute boundary, return as ISO 8601 string."""
    minutes = dt.minute
    rounded = round(minutes / 5) * 5
    if rounded == 60:
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=rounded, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ── SQLite writer ───────────────────────────────────────────────────────────


_db: sqlite3.Connection | None = None


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is not None:
        return _db
    SNAPSHOTS_DB.parent.mkdir(parents=True, exist_ok=True)
    _db = sqlite3.connect(str(SNAPSHOTS_DB))
    _db.execute("PRAGMA journal_mode = WAL")
    _db.execute(_CREATE_READINGS_SQL)
    _db.commit()
    return _db


def _close_db() -> None:
    global _db
    if _db is not None:
        _db.close()
        _db = None


def _write_readings(timestamp: str, values: dict[str, str]) -> None:
    """Write snapshot values to the EAV readings table."""
    db = _get_db()
    rows = [(timestamp, name, value) for name, value in values.items()]
    db.executemany("INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)", rows)
    db.commit()


# ── Public API ──────────────────────────────────────────────────────────────


def collect_once(*, log: print = print) -> None:
    """Collect a single snapshot and write to SQLite."""
    url = _ha_url()
    if not url:
        print("Error: HA_URL must be set in environment.", file=sys.stderr)
        sys.exit(1)

    # Fetch all entity states in one REST call
    all_entities = _CFG.all_history_entities
    states = _fetch_states(all_entities)

    # Extract column values
    values = _extract_snapshot(states)

    # Fetch and inject forecast
    forecast = _fetch_forecast(_CFG.weather_entity)
    _inject_forecast(values, forecast)

    # Round timestamp and write
    ts = _round_to_five_minutes(datetime.now(UTC))
    _write_readings(ts, values)
    log(f"[collector] Wrote snapshot (ts: {ts}, {len(values)} readings)")


def collect_loop(*, log: print = print) -> None:
    """Run the collection loop indefinitely at 5-minute intervals."""
    interval = SNAPSHOT_INTERVAL_SECONDS
    log(f"[collector] Starting collection loop (interval: {interval}s)")
    log(f"[collector] Database: {SNAPSHOTS_DB}")

    # Graceful shutdown
    def _shutdown(signum: int, _frame: object) -> None:
        log(f"\n[collector] Received signal {signum}, shutting down...")
        _close_db()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        try:
            collect_once(log=log)
        except Exception as e:
            print(f"[collector] Error: {e}", file=sys.stderr)
        time.sleep(interval)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Weatherstat snapshot collector")
    parser.add_argument("command", choices=["collect", "once"], help="collect = loop, once = single snapshot")
    args = parser.parse_args()

    if args.command == "once":
        collect_once()
    else:
        collect_loop()


if __name__ == "__main__":
    main()
