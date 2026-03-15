"""Backfill met_outdoor_temp into the readings table from Open-Meteo.

Uses two APIs to cover the full date range:
  1. Archive API (ERA5 reanalysis): Feb 2026 through ~7 days ago
  2. Historical Forecast API: last ~7 days through yesterday

Data is hourly; sysid interpolates to 5-min intervals automatically.

Usage:
  uv run python scripts/backfill-outdoor-temp.py [--start 2026-02-01] [--dry-run]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

# Add ml/src to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml" / "src"))

from weatherstat.config import SNAPSHOTS_DB  # noqa: E402

LATITUDE = 0.0    # Put yours here.
LONGITUDE = 0.0   # Put yours here.

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

COMMON_PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": "temperature_2m",
    "temperature_unit": "fahrenheit",
    "timezone": "UTC",
}


def fetch_temps(url: str, start: str, end: str) -> list[tuple[str, float]]:
    """Fetch hourly temperatures from Open-Meteo. Returns [(iso_ts, temp_f), ...]."""
    params = {**COMMON_PARAMS, "start_date": start, "end_date": end}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    results: list[tuple[str, float]] = []
    for t_str, temp in zip(times, temps, strict=True):
        if temp is None:
            continue
        # Open-Meteo returns "2026-02-01T00:00" — normalize to collector format
        ts = t_str + ":00.000Z" if len(t_str) == 16 else t_str
        results.append((ts, round(float(temp), 1)))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill met_outdoor_temp from Open-Meteo")
    parser.add_argument("--start", default="2026-02-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing")
    parser.add_argument("--db", type=Path, default=None, help="Database path (default: SNAPSHOTS_DB)")
    args = parser.parse_args()

    db_path = args.db or SNAPSHOTS_DB
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(UTC)
    # Archive API has ~5-7 day lag; split at 8 days ago to be safe
    archive_end = (now - timedelta(days=8)).strftime("%Y-%m-%d")
    forecast_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    forecast_end = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching archive data: {args.start} to {archive_end}...")
    archive_rows = fetch_temps(ARCHIVE_URL, args.start, archive_end)
    print(f"  {len(archive_rows)} hourly readings")

    print(f"Fetching historical forecast data: {forecast_start} to {forecast_end}...")
    forecast_rows = fetch_temps(FORECAST_URL, forecast_start, forecast_end)
    print(f"  {len(forecast_rows)} hourly readings")

    # Merge, preferring archive (ERA5 reanalysis) over forecast for overlap
    seen: set[str] = set()
    all_rows: list[tuple[str, str, float]] = []
    for ts, temp in archive_rows:
        seen.add(ts)
        all_rows.append((ts, "met_outdoor_temp", temp))
    for ts, temp in forecast_rows:
        if ts not in seen:
            all_rows.append((ts, "met_outdoor_temp", temp))

    print(f"\nTotal: {len(all_rows)} readings to insert")

    if args.dry_run:
        print("(dry-run — no database writes)")
        if all_rows:
            print(f"  First: {all_rows[0][0]} = {all_rows[0][2]}°F")
            print(f"  Last:  {all_rows[-1][0]} = {all_rows[-1][2]}°F")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Count existing met_outdoor_temp readings
    existing = cursor.execute(
        "SELECT COUNT(*) FROM readings WHERE name = 'met_outdoor_temp'"
    ).fetchone()[0]
    print(f"Existing met_outdoor_temp readings: {existing}")

    inserted = 0
    for ts, name, value in all_rows:
        cursor.execute(
            "INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)",
            (ts, name, str(value)),
        )
        inserted += cursor.rowcount

    conn.commit()
    conn.close()

    print(f"Inserted {inserted} new readings ({len(all_rows) - inserted} already existed)")


if __name__ == "__main__":
    main()
