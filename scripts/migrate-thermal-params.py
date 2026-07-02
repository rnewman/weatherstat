#!/usr/bin/env python3
"""Migrate thermal_params.json from window-specific to advisory effector naming.

Renames:
  fitted_taus[].window_betas      → advisory_tau_betas
  fitted_taus[].interaction_betas → advisory_interaction_betas

Idempotent: skips entries that already use the new names.

Usage:
    python scripts/migrate-thermal-params.py [path]

If path is omitted, uses ~/.weatherstat/thermal_params.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def migrate(data: dict) -> bool:
    """Migrate thermal_params in-place. Returns True if any changes made."""
    changed = False

    for ft in data.get("fitted_taus", []):
        if "window_betas" in ft:
            ft["advisory_tau_betas"] = ft.pop("window_betas")
            changed = True
        if "interaction_betas" in ft:
            ft["advisory_interaction_betas"] = ft.pop("interaction_betas")
            changed = True

    return changed


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path.home() / ".weatherstat" / "thermal_params.json"

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if not migrate(data):
        print("No changes needed — already using advisory naming or no fitted_taus.")
        return

    backup_path = path.with_suffix(".json.bak")
    shutil.copy2(path, backup_path)
    print(f"Backed up original to {backup_path}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Migrated {path}: window_betas → advisory_tau_betas, interaction_betas → advisory_interaction_betas.")


if __name__ == "__main__":
    main()
