#!/usr/bin/env python3
"""Migrate EAV column names from window_*_open to environment config columns.

Reads the `environment:` config to build old→new column name mapping:
  old: f"window_{name}_open"  →  new: cfg.column

Also deletes the dead `any_window_open` column.

Usage:
    uv run python scripts/migrate-environment-columns.py          # dry-run (default)
    uv run python scripts/migrate-environment-columns.py --apply  # actually rename
"""

from __future__ import annotations

import argparse
import sqlite3
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate window column names to environment config columns")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (default: dry-run)")
    args = parser.parse_args()

    # Import config (sets up data dir)
    from weatherstat._data_dir import resolve_data_dir
    from weatherstat.yaml_config import load_config

    cfg = load_config()
    data_dir = resolve_data_dir()
    db_path = data_dir / "snapshots" / "snapshots.db"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Build rename mapping: old → new
    renames: dict[str, str] = {}
    for name, env_cfg in cfg.environment.items():
        old_col = f"window_{name}_open"
        new_col = env_cfg.column
        if old_col != new_col:
            renames[old_col] = new_col

    # Check which old names actually exist in the DB
    cursor.execute("SELECT DISTINCT name FROM readings")
    existing_names = {row[0] for row in cursor.fetchall()}

    print(f"Database: {db_path}")
    print(f"Environment entries: {len(cfg.environment)}")
    print(f"Rename candidates: {len(renames)}")
    print()

    applied = 0
    skipped = 0

    for old_name, new_name in sorted(renames.items()):
        if old_name not in existing_names:
            print(f"  SKIP {old_name} → {new_name} (old name not in DB)")
            skipped += 1
            continue

        if new_name in existing_names:
            # Both exist — count rows to decide
            cursor.execute("SELECT COUNT(*) FROM readings WHERE name = ?", (old_name,))
            old_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM readings WHERE name = ?", (new_name,))
            new_count = cursor.fetchone()[0]
            print(f"  SKIP {old_name} → {new_name} (both exist: {old_count} old, {new_count} new rows)")
            skipped += 1
            continue

        cursor.execute("SELECT COUNT(*) FROM readings WHERE name = ?", (old_name,))
        count = cursor.fetchone()[0]
        print(f"  RENAME {old_name} → {new_name} ({count} rows)")

        if args.apply:
            cursor.execute("UPDATE readings SET name = ? WHERE name = ?", (new_name, old_name))
        applied += 1

    # Delete any_window_open
    if "any_window_open" in existing_names:
        cursor.execute("SELECT COUNT(*) FROM readings WHERE name = 'any_window_open'")
        awo_count = cursor.fetchone()[0]
        print(f"\n  DELETE any_window_open ({awo_count} rows)")
        if args.apply:
            cursor.execute("DELETE FROM readings WHERE name = 'any_window_open'")
        applied += 1
    else:
        print("\n  any_window_open: not in DB (already clean)")

    print(f"\nSummary: {applied} changes, {skipped} skipped")

    if args.apply:
        conn.commit()
        print("Changes committed.")
    else:
        print("\nDry run — no changes made. Use --apply to execute.")

    conn.close()


if __name__ == "__main__":
    main()
