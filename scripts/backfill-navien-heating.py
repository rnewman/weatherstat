"""Backfill navien_heating into the readings table from historical data.

Regimes for determining whether heat was actually delivered:

1. Navien controller data available & connected (boiler_navien_mode != "unavailable"):
   copy the boiler_navien_mode value — the Navien controller is authoritative.

2. Navien controller data available but disconnected (boiler_navien_mode == "unavailable"):
   force "Idle" — the boiler wasn't working, so no heat was delivered
   regardless of thermostat intent.

3. No Navien data (pre-install, collector gap, etc.): thermostat action is
   the source of truth. If any thermostat says "heating", we were heating.

Only backfills timestamps where navien_heating does not already exist.

Usage:
  uv run python scripts/backfill-navien-heating.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Add src to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weatherstat.config import SNAPSHOTS_DB  # noqa: E402


def backfill(dry_run: bool = False) -> None:
    db = sqlite3.connect(SNAPSHOTS_DB)
    db.row_factory = sqlite3.Row

    # Get all timestamps that already have navien_heating.
    existing = set(
        r[0]
        for r in db.execute(
            "SELECT timestamp FROM readings WHERE name = 'navien_heating'"
        ).fetchall()
    )
    print(f"Existing navien_heating rows: {len(existing)}")

    inserts: list[tuple[str, str, str]] = []

    # Build set of timestamps that have boiler_navien_mode data.
    navien_mode: dict[str, str] = {}
    for r in db.execute("SELECT timestamp, value FROM readings WHERE name = 'boiler_navien_mode'").fetchall():
        navien_mode[r["timestamp"]] = r["value"]
    print(f"boiler_navien_mode rows: {len(navien_mode)}")

    # --- Regime 1 & 2: timestamps with boiler_navien_mode ---
    regime1_count = 0
    regime2_count = 0
    for ts, mode in navien_mode.items():
        if ts in existing:
            continue
        if mode == "unavailable":
            # Disconnected — no heat delivered.
            inserts.append((ts, "navien_heating", "Idle"))
            regime2_count += 1
        else:
            # Connected — Navien is authoritative.
            inserts.append((ts, "navien_heating", mode))
            regime1_count += 1

    print(f"Regime 1 (Navien connected): {regime1_count} rows")
    print(f"Regime 2 (Navien disconnected → Idle): {regime2_count} rows")

    # --- Regime 3: timestamps with thermostat data but NO navien data ---
    # Covers pre-Navien period and any collector gaps (e.g., YAML change
    # that dropped boiler column before navien_heating was added).
    gap_rows = db.execute(
        """
        SELECT timestamp, MAX(CASE WHEN value = 'heating' THEN 1 ELSE 0 END) AS any_heating
        FROM readings
        WHERE name LIKE 'thermostat_%_action'
        GROUP BY timestamp
        ORDER BY timestamp
        """
    ).fetchall()

    regime3_count = 0
    for row in gap_rows:
        ts = row["timestamp"]
        if ts in existing or ts in navien_mode:
            continue
        value = "Space Heating" if row["any_heating"] else "Idle"
        inserts.append((ts, "navien_heating", value))
        regime3_count += 1

    print(f"Regime 3 (no Navien data, thermostat=truth): {regime3_count} rows")
    print(f"Total to insert: {len(inserts)}")

    if not inserts:
        print("Nothing to do.")
        db.close()
        return

    if dry_run:
        # Show a sample.
        print("\nSample (first 10):")
        for ts, _name, value in inserts[:10]:
            print(f"  {ts}  {value}")
        if len(inserts) > 10:
            print(f"  ... and {len(inserts) - 10} more")
        db.close()
        return

    db.executemany(
        "INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)",
        inserts,
    )
    db.commit()
    print(f"Inserted {db.total_changes} rows.")
    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill navien_heating from historical data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be inserted without writing")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
