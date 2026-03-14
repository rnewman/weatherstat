#!/usr/bin/env python3
"""Migrate existing wide snapshots table to EAV readings table.

One-time migration: unpivots all columns from the wide `snapshots` table
into (timestamp, name, value) rows in the `readings` table.

Safe to run multiple times — uses INSERT OR IGNORE.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "snapshots" / "snapshots.db"


def migrate(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")

    # Create readings table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            timestamp TEXT NOT NULL,
            name      TEXT NOT NULL,
            value     TEXT NOT NULL,
            PRIMARY KEY (timestamp, name)
        )
    """)

    # Get all columns from the snapshots table (excluding timestamp)
    cursor = conn.execute("PRAGMA table_info(snapshots)")
    columns = [row[1] for row in cursor.fetchall() if row[1] != "timestamp"]

    print(f"Migrating {len(columns)} columns from snapshots → readings")

    # Count existing rows for progress
    total_snapshots = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    existing_readings = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    print(f"  Snapshots table: {total_snapshots} rows")
    print(f"  Readings table (before): {existing_readings} rows")

    # Unpivot each column
    migrated = 0
    for col in columns:
        result = conn.execute(f"""
            INSERT OR IGNORE INTO readings (timestamp, name, value)
            SELECT timestamp, ?, CAST("{col}" AS TEXT)
            FROM snapshots
            WHERE "{col}" IS NOT NULL
        """, (col,))
        migrated += result.rowcount

    conn.commit()

    final_readings = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    print(f"  Readings table (after): {final_readings} rows")
    print(f"  New rows inserted: {migrated}")

    # Verify: spot-check a random timestamp
    sample = conn.execute(
        "SELECT timestamp FROM snapshots ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    if sample:
        ts = sample[0]
        wide_cols = conn.execute(
            "SELECT COUNT(*) FROM pragma_table_info('snapshots') WHERE name != 'timestamp'"
        ).fetchone()[0]
        eav_cols = conn.execute(
            "SELECT COUNT(*) FROM readings WHERE timestamp = ?", (ts,)
        ).fetchone()[0]
        print(f"\n  Spot check ({ts}):")
        print(f"    Wide columns: {wide_cols}, EAV readings: {eav_cols}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    migrate(path)
