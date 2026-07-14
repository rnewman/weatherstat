"""Replay the solar residual tracker over collector history.

Walks Feb→now in fixed steps, running the tracker's trailing-window update
chain exactly as production would (each step's estimate is the next step's
prior). The resulting γ(t) curves are the measured per-room canopy
phenology — the acceptance test for PLAN-35 is that west-channel γ for the
kitchen/family-room cluster collapses across April leaf-out, matching the
bare-vs-leafed regression in maple_shading_analysis.py.

Production updates hourly; this replay steps every ~3 days for speed, so
transitions lag slightly more here than they would live.

Usage: uv run python scripts/replay_solar_tracker.py [--params PATH] [--step-days N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from weatherstat.extract import load_collector_snapshots
from weatherstat.sysid import _enumerate_effectors, _enumerate_sensors, _preprocess
from weatherstat.tracker import SOLAR_BLOCK, update_block_for_sensor

_INTERESTING = ["kitchen_temp", "family_room_temp", "living_room_climate_temp", "piano_temp", "bedroom_temp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True, help="thermal_params.json with regression_models")
    parser.add_argument("--step-days", type=float, default=3.0)
    args = parser.parse_args()

    params = json.loads(args.params.read_text())
    models = params.get("regression_models", {})
    if not models:
        raise SystemExit("params file has no regression_models — refit with the branch sysid first")

    effectors = _enumerate_effectors()
    sensors = [s for s in _enumerate_sensors() if s.name in models]

    print("Loading and preprocessing full history...")
    df = load_collector_snapshots()
    df = _preprocess(df, effectors, sensors)
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    start = ts.min() + pd.Timedelta(days=SOLAR_BLOCK.window_days)
    end = ts.max()
    steps = pd.date_range(start, end, freq=pd.Timedelta(days=args.step_days))
    print(f"{len(df)} rows {ts.min().date()} → {end.date()}, {len(steps)} tracker updates\n")

    prev: dict[str, dict[str, float]] = {}
    history: list[tuple[pd.Timestamp, str, dict[str, float], bool, float]] = []

    for t in steps:
        window = df[(ts > t - pd.Timedelta(days=SOLAR_BLOCK.window_days)) & (ts <= t)]
        if len(window) < 500:
            continue
        for sensor in sensors:
            upd = update_block_for_sensor(
                SOLAR_BLOCK, window.reset_index(drop=True), sensor, effectors,
                models[sensor.name], prev.get(sensor.name),
            )
            if upd is None or not upd.coefficients:
                continue
            prev[sensor.name] = upd.coefficients
            history.append((t, sensor.name, upd.coefficients, upd.updated, upd.information))

    # γ_west trajectory for the sentinel sensors
    print(f"{'date':<12}" + "".join(f"{s.replace('_temp', ''):>22}" for s in _INTERESTING))
    print(" " * 12 + "".join(f"{'elev  east  west':>22}" for _ in _INTERESTING))
    by_date: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}
    for t, sensor, coefs, _, _ in history:
        by_date.setdefault(t, {})[sensor] = coefs
    for t in sorted(by_date):
        row = f"{t.date()!s:<12}"
        for s in _INTERESTING:
            c = by_date[t].get(s)
            if c is None:
                row += f"{'--':>22}"
            else:
                row += f"{c.get('_solar_elev', 0):>8.2f}{c.get('_solar_east', 0):>6.2f}{c.get('_solar_west', 0):>6.2f}  "
        print(row)

    held = sum(1 for *_rest, updated, _info in history if not updated)
    print(f"\n{len(history)} sensor-updates total, {held} held by the information gate")


if __name__ == "__main__":
    main()
