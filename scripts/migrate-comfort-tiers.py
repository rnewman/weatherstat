#!/usr/bin/env python3
"""Migrate weatherstat.yaml constraint schedules from two-tier to three-tier comfort bounds.

Converts old min/max format:
    schedule:
      - { hours: [8, 22], preferred: 71, min: 70, max: 74 }

To new three-tier format:
    schedule:
      - { hours: [8, 22], preferred: 71, acceptable: [70, 74], backup: [67, 77] }

The old 'min'/'max' keys become 'acceptable', and a new 'backup' tier is added
at acceptable ± backup_margin (default 3°F).

Old 'min'/'max' keys are removed; new configs use 'acceptable' and 'backup'.
If 'acceptable' or 'backup' already exist, they are preserved (idempotent).

Usage:
    python scripts/migrate-comfort-tiers.py [--backup-margin 3] [config_path]

If config_path is omitted, uses ~/.weatherstat/weatherstat.yaml.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def migrate(data: dict, backup_margin: float = 3.0) -> bool:
    """Migrate constraint schedules in-place. Returns True if any changes made."""
    constraints = data.get("constraints", {})
    schedules = constraints.get("schedules", [])
    changed = False

    for sched in schedules:
        for entry in sched.get("schedule", []):
            # Skip entries already migrated
            if "acceptable" in entry:
                continue

            if "min" not in entry or "max" not in entry:
                continue

            min_t = entry.pop("min")
            max_t = entry.pop("max")
            entry["acceptable"] = [min_t, max_t]

            if "backup" not in entry:
                entry["backup"] = [min_t - backup_margin, max_t + backup_margin]

            changed = True

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate comfort bounds to three-tier format")
    parser.add_argument("config_path", nargs="?", help="Path to weatherstat.yaml")
    parser.add_argument("--backup-margin", type=float, default=3.0, help="Backup margin in degrees (default: 3)")
    args = parser.parse_args()

    if args.config_path:
        path = Path(args.config_path)
    else:
        data_dir = Path.home() / ".weatherstat"
        path = data_dir / "weatherstat.yaml"

    if not path.exists():
        print(f"Config not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = yaml.safe_load(f)

    if not migrate(data, args.backup_margin):
        print("No changes needed — config already uses three-tier bounds or has no schedules.")
        return

    # Back up original
    backup_path = path.with_suffix(".yaml.bak")
    shutil.copy2(path, backup_path)
    print(f"Backed up original to {backup_path}")

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Migrated {path} to three-tier comfort bounds.")
    print("Review the file and adjust backup bounds if needed.")


if __name__ == "__main__":
    main()
