"""Backfill shade position data from HA state history into the readings table.

Reads cover entity state-change history from HA REST API (~10 days) and
writes sparse position entries (0.0–1.0) at 5-minute boundaries. The load
path forward-fills gaps, so only state transitions need to be stored.

Transitional states (opening/closing) are skipped — we store the settled
position from the subsequent open/closed state.

Usage:
  uv run python scripts/backfill-shades.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

# Add src to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weatherstat.config import SNAPSHOTS_DB  # noqa: E402
from weatherstat.yaml_config import load_config  # noqa: E402


def _ha_url() -> str:
    env_path = Path.home() / ".weatherstat" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HA_URL="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HA_URL", "")


def _ha_token() -> str:
    env_path = Path.home() / ".weatherstat" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HA_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HA_TOKEN", "")


def _fetch_cover_history(entity_id: str, days: int = 10) -> list[dict]:
    """Fetch state-change history for a cover entity from HA REST API."""
    url = _ha_url()
    token = _ha_token()
    start = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    end = datetime.now(UTC).isoformat()
    resp = requests.get(
        f"{url}/api/history/period/{start}",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "filter_entity_id": entity_id,
            "end_time": end,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else []


def _snap_to_5min(dt: datetime) -> str:
    """Round a datetime down to the nearest 5-minute boundary, as collector timestamp."""
    minute = (dt.minute // 5) * 5
    snapped = dt.replace(minute=minute, second=0, microsecond=0)
    return snapped.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _active_fraction(state: dict, active_state: str) -> float | None:
    """Extract active fraction (0.0–1.0) from an HA cover state entry.

    Active fraction = how much in the non-default state. For covers with
    active_state="closed", openness 0.0 (lowered) → active 1.0; openness
    1.0 (raised) → active 0.0. Returns None for transitional states.
    """
    ha_state = state.get("state", "")
    if ha_state in ("opening", "closing"):
        return None

    attrs = state.get("attributes", {})
    raw_pos = attrs.get("current_position")
    if raw_pos is not None:
        openness = float(raw_pos) / 100.0
    elif ha_state == "closed":
        openness = 0.0
    elif ha_state == "open":
        openness = 1.0
    else:
        return None

    active_frac = openness if active_state == "open" else 1.0 - openness
    return round(active_frac, 2)


def backfill(dry_run: bool = False) -> None:
    cfg = load_config()
    db = sqlite3.connect(str(SNAPSHOTS_DB))

    # Find shade entries in config
    shades = {name: ecfg for name, ecfg in cfg.environment.items() if ecfg.kind == "shade"}
    if not shades:
        print("No shade entries in config.")
        db.close()
        return

    total_inserted = 0

    for name, ecfg in shades.items():
        print(f"\n── {name} ({ecfg.entity_id}) → column: {ecfg.column} ──")

        # Get existing readings for this column
        existing = set(
            r[0] for r in db.execute(
                "SELECT timestamp FROM readings WHERE name = ?", (ecfg.column,)
            ).fetchall()
        )
        print(f"  Existing readings: {len(existing)}")

        # Fetch history from HA
        history = _fetch_cover_history(ecfg.entity_id)
        if not history:
            print("  No history available from HA.")
            continue
        print(f"  HA history entries: {len(history)}")

        # Convert state changes to sparse active-fraction entries
        inserts: list[tuple[str, str, str]] = []
        last_val: float | None = None
        last_ts: str | None = None

        for entry in history:
            val = _active_fraction(entry, ecfg.active_state)
            if val is None:
                continue  # skip transitional states

            dt_str = entry.get("last_changed", entry.get("last_updated", ""))
            if not dt_str:
                continue
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            ts = _snap_to_5min(dt)

            # Skip if same value at same timestamp, or same value as last
            if ts == last_ts:
                continue
            if val == last_val:
                continue
            if ts in existing:
                continue

            inserts.append((ts, ecfg.column, str(val)))
            last_val = val
            last_ts = ts

        print(f"  New readings to insert: {len(inserts)}")

        if inserts and not dry_run:
            db.executemany(
                "INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)",
                inserts,
            )
            total_inserted += len(inserts)

        if dry_run and inserts:
            print("  Sample (first 10):")
            for ts, _name, value in inserts[:10]:
                print(f"    {ts}  active={value}")
            if len(inserts) > 10:
                print(f"    ... and {len(inserts) - 10} more")

    if not dry_run and total_inserted > 0:
        db.commit()
        print(f"\nInserted {total_inserted} total readings.")
    elif not dry_run:
        print("\nNothing to insert.")

    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill shade positions from HA history")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be inserted")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
