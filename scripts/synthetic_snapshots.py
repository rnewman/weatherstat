"""Build a synthetic collector database at production scale, for profiling and testing sysid.

Writes ``<out_dir>/weatherstat.yaml`` (the example config) and
``<out_dir>/snapshots/snapshots.db`` in the collector's exact EAV schema and
value formats. The default — Feb 1 to Nov 10 at 5-minute resolution, 62
names per snapshot — is 81,216 snapshots ≈ 5.0M readings, the size of a
house's first year.

The physics is deliberately simple but real enough that every sysid stage
does representative work and has a known answer: Newton cooling (per-room
tau 30–60h), thermostats with hysteresis and a night setback, two mini
splits, blowers, solar gain through cloud cover, and windows that open on
warm afternoons and cool **only their own room** (so every cross-room window
coupling sysid finds is spurious — a ground truth for the stability and
identification checks). Sensor noise and 0.1° quantization on top.

Usage:
    just synth-db /tmp/ws-synth                 # ~5.0M readings, ~15s
    just synth-db /tmp/ws-synth --days 30       # smaller
    WEATHERSTAT_DATA_DIR=/tmp/ws-synth just sysid
    just bench-sysid periodic /tmp/ws-synth     # see scripts/bench_sysid.py

Never writes outside ``out_dir``; refuses to overwrite an existing database
unless --force.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Names the collector would write that sysid doesn't read (forecast and
# irradiance columns), so the readings count — and load cost — matches a
# real house rather than just the example config's typed columns.
EXTRA_NAMES = (
    [f"forecast_temp_{h}h" for h in (1, 2, 4, 6, 12, 24, 48)]
    + [f"solar_irradiance_{p}" for p in ("h", "n", "e", "s", "w")]
    + ["met_humidity", "met_wind"]
)

ROOMS = [  # temperature columns, in simulation order
    "thermostat_upstairs_temp", "thermostat_downstairs_temp", "family_room_temp", "office_temp",
    "kitchen_temp", "bedroom_temp", "piano_temp", "bathroom_temp",
    "mini_split_bedroom_temp", "mini_split_living_room_temp", "living_room_temp",
]
WINDOWS = ["basement_open", "family_room_open", "bedroom_open", "office_open", "kitchen_open",
           "piano_open", "bathroom_open", "living_room_open"]


def _room_window(room: str) -> int:
    """Index of the window that cools this room, or -1 (each window affects only its own room)."""
    label = room.removeprefix("thermostat_").removeprefix("mini_split_").removesuffix("_temp")
    return WINDOWS.index(f"{label}_open") if f"{label}_open" in WINDOWS else -1


def simulate(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Run the house for n 5-minute steps; returns every column's values."""
    t_h = np.arange(n) * (5 / 60)
    day = t_h / 24
    hour_local = (t_h % 24 - 8) % 24  # PST-ish; close enough for synthetic physics

    season = 52 - 10 * np.cos(2 * np.pi * (day - 15) / 365)
    weather = np.zeros(n)
    shocks = rng.normal(0, 0.15, n)
    for i in range(1, n):
        weather[i] = 0.995 * weather[i - 1] + shocks[i]
    outdoor = season + 7 * np.sin(2 * np.pi * (hour_local - 9) / 24) + weather

    doy = (day + 31) % 365
    decl = np.radians(23.44 * np.sin(2 * np.pi * (doy - 81) / 365))
    lat = np.radians(47.6)
    sun = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(np.radians((hour_local - 12) * 15)), 0, None)
    cloud = np.zeros(n)
    cloud_shocks = rng.normal(0, 3, n)
    for i in range(1, n):
        cloud[i] = np.clip(0.99 * cloud[i - 1] + 0.6 + cloud_shocks[i], 0, 100)
    sun_frac = 1 - 0.75 * (cloud / 100) ** 3.4
    condition = np.where(cloud < 20, "sunny", np.where(cloud < 60, "partlycloudy", np.where(cloud < 90, "cloudy", "rainy")))

    # Windows: occasional afternoon openings, mostly in warm months
    n_days = n // 288 + 1
    win = np.zeros((len(WINDOWS), n), dtype=np.int8)
    for k in range(len(WINDOWS)):
        for d in range(n_days):
            if d * 288 >= n:
                break
            if rng.random() < (0.35 if season[d * 288] > 55 else 0.05):
                s = d * 288 + int(rng.integers(9 * 12, 16 * 12))
                win[k, s : s + int(rng.integers(12, 60))] = 1

    m = len(ROOMS)
    tau = rng.uniform(30, 60, m)
    solar_gain = rng.uniform(0.3, 1.5, m)
    gain_up = np.zeros(m)
    gain_up[[0, 3, 5, 9]] = [2.0, 0.8, 0.6, 0.3]
    gain_dn = np.zeros(m)
    gain_dn[[1, 2, 4, 6, 7]] = [2.0, 1.2, 0.9, 0.7, 0.5]
    gain_split_bed = np.zeros(m)
    gain_split_bed[[5, 8]] = [1.5, 1.6]
    gain_split_liv = np.zeros(m)
    gain_split_liv[[9, 10, 0]] = [1.5, 1.4, 0.4]
    window_of = np.array([_room_window(r) for r in ROOMS])

    T = np.zeros((m, n))
    T[:, 0] = 68
    up, dn, split_bed, split_liv = (np.zeros(n, dtype=np.int8) for _ in range(4))
    noise = rng.normal(0, 0.03, (m, n))
    for i in range(1, n):
        setpoint = 68 if 7 <= hour_local[i] < 22 else 64
        up[i] = 1 if T[0, i - 1] < setpoint - 0.5 else (0 if T[0, i - 1] > setpoint + 0.5 else up[i - 1])
        dn[i] = 1 if T[1, i - 1] < setpoint - 0.5 else (0 if T[1, i - 1] > setpoint + 0.5 else dn[i - 1])
        split_bed[i] = 1 if T[5, i - 1] < 66 else (0 if T[5, i - 1] > 67 else split_bed[i - 1])
        split_liv[i] = 1 if T[9, i - 1] < 67 else (0 if T[9, i - 1] > 68 else split_liv[i - 1])
        open_now = np.where(window_of >= 0, win[np.maximum(window_of, 0), i], 0)
        dT = ((outdoor[i] - T[:, i - 1]) / (tau / (1 + 3 * open_now))
              + gain_up * up[i] + gain_dn * dn[i] + gain_split_bed * split_bed[i] + gain_split_liv * split_liv[i]
              + solar_gain * sun[i] * sun_frac[i])
        T[:, i] = T[:, i - 1] + dT * (5 / 60) + noise[:, i]
    temps = np.round(T + rng.normal(0, 0.05, T.shape), 1)

    blower = rng.random((3, n)) < 0.15
    blower_mode = np.where(blower, np.where(rng.random((3, n)) < 0.5, "low", "high"), "off")
    heating = (up | dn).astype(bool)
    combi = np.where(heating, "Space Heating", np.where(rng.random(n) < 0.05, "Domestic Hot Water", "Idle"))

    def ahead(a: np.ndarray, hours: int) -> np.ndarray:
        return a[np.minimum(np.arange(n) + 12 * hours, n - 1)]

    cols: dict[str, np.ndarray] = {r: temps[k] for k, r in enumerate(ROOMS)}
    action = lambda on: np.where(on, "heating", "idle")  # noqa: E731
    cols |= {
        "thermostat_upstairs_target": np.full(n, 68.0), "thermostat_upstairs_action": action(up),
        "thermostat_downstairs_target": np.full(n, 68.0), "thermostat_downstairs_action": action(dn),
        "mini_split_bedroom_target": np.full(n, 66.5), "mini_split_bedroom_mode": np.full(n, "heat"),
        "mini_split_bedroom_action": action(split_bed),
        "mini_split_living_room_target": np.full(n, 67.5), "mini_split_living_room_mode": np.full(n, "heat"),
        "mini_split_living_room_action": action(split_liv),
        "blower_family_room_mode": blower_mode[0], "blower_office_mode": blower_mode[1], "blower_gym_mode": blower_mode[2],
        "combi_heating": combi, "combi_gas_usage": np.round(heating * 2.1, 2),
        "met_outdoor_temp": np.round(outdoor[np.arange(n) - np.arange(n) % 12], 1),  # hourly met.no
        "outdoor_temp": np.round(outdoor, 1), "outdoor_humidity": np.full(n, 70.0),
        "wind_speed": np.round(np.abs(weather) * 2, 1), "weather_condition": condition,
        "cloud_coverage": np.round(cloud, 0),
        "upstairs_humidity": np.full(n, 40.0), "downstairs_humidity": np.full(n, 42.0), "bedroom_humidity": np.full(n, 45.0),
    }
    cols |= {f"forecast_cloud_{h}h": np.round(ahead(cloud, h), 0) for h in (1, 2, 4, 6, 12)}
    cols |= {w: win[k] for k, w in enumerate(WINDOWS)}
    cols |= {f"forecast_temp_{h}h": np.round(ahead(outdoor, h), 1) for h in (1, 2, 4, 6, 12, 24, 48)}
    cols |= {f"solar_irradiance_{p}": np.round(1000 * sun * sun_frac, 0) for p in ("h", "n", "e", "s", "w")}
    cols |= {"met_humidity": np.full(n, 70.0), "met_wind": np.full(n, 3.0)}
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="data directory to create (gets weatherstat.yaml + snapshots/snapshots.db)")
    parser.add_argument("--days", type=int, default=282, help="days of 5-minute snapshots (default 282 ≈ 5.0M readings)")
    parser.add_argument("--start", default="2026-02-01", help="first day, UTC (default 2026-02-01)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="replace an existing snapshots.db")
    args = parser.parse_args()

    out: Path = args.out_dir.resolve()
    db_path = out / "snapshots" / "snapshots.db"
    if db_path.exists() and not args.force:
        sys.exit(f"{db_path} exists — pass --force to replace it")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not (out / "weatherstat.yaml").exists():
        shutil.copy2(REPO / "weatherstat.yaml.example", out / "weatherstat.yaml")
    for f in (db_path, db_path.with_name("snapshots.db-wal"), db_path.with_name("snapshots.db-shm")):
        f.unlink(missing_ok=True)

    # Config is read at import time: point weatherstat at the new directory first.
    os.environ["WEATHERSTAT_DATA_DIR"] = str(out)
    from weatherstat.collector import _apply_schema
    from weatherstat.yaml_config import load_config

    n = args.days * 288
    cols = simulate(n, np.random.default_rng(args.seed))
    known = {c for c, _ in load_config().snapshot_column_defs()} | set(EXTRA_NAMES)
    unknown = sorted(set(cols) - known)
    if unknown:
        sys.exit(f"generator writes names the example config doesn't define: {unknown}")

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode = WAL")
    _apply_schema(con, log=lambda _m: None)
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    names = list(cols)
    as_text = {k: v.astype(str) for k, v in cols.items()}
    total = 0
    batch: list[tuple[str, str, str]] = []
    for i in range(n):
        ts = (start + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        batch.extend((ts, k, as_text[k][i]) for k in names)
        if len(batch) >= 500_000:
            con.executemany("INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)", batch)
            total += len(batch)
            batch = []
    con.executemany("INSERT OR IGNORE INTO readings (timestamp, name, value) VALUES (?, ?, ?)", batch)
    total += len(batch)
    con.commit()
    _apply_schema(con, log=lambda _m: None)  # planner statistics now that there is data (timestamp skip-scans)
    con.close()
    print(f"{out}: {n:,} snapshots × {len(names)} names = {total:,} readings")


if __name__ == "__main__":
    main()
