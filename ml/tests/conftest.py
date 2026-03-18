"""Test configuration: isolated sandbox using example YAML and synthetic data.

Sets WEATHERSTAT_DATA_DIR to a temporary directory BEFORE any weatherstat
modules are imported, so module-level `_CFG = load_config()` calls pick up
the test config instead of the live ~/.weatherstat.

The temporary directory contains:
  - weatherstat.yaml copied from the repo's weatherstat.yaml.example
  - thermal_params.json with synthetic but realistic parameters
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

# ── Set up isolated data dir BEFORE any weatherstat imports ──────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXAMPLE_YAML = _REPO_ROOT / "weatherstat.yaml.example"

# Create a persistent temp directory for the test session.
# Using a fixed path avoids re-creation across test files within one pytest run.
_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="weatherstat_test_"))
os.environ["WEATHERSTAT_DATA_DIR"] = str(_TEST_DATA_DIR)

# Copy example YAML as the active config
shutil.copy2(_EXAMPLE_YAML, _TEST_DATA_DIR / "weatherstat.yaml")

# Write synthetic thermal_params.json with realistic structure
_SENSOR_COLS = [
    "thermostat_upstairs_temp",
    "thermostat_downstairs_temp",
    "bedroom_temp",
    "bedroom_aggregate_temp",
    "office_temp",
    "office_bookshelf_temp",
    "family_room_temp",
    "kitchen_temp",
    "piano_temp",
    "bathroom_temp",
    "living_room_temp",
    "living_room_climate_temp",
    "basement_temp",
    "upstairs_aggregate_temp",
    "downstairs_aggregate_temp",
]

_THERMAL_PARAMS: dict = {
    "timestamp": "2026-03-14T00:00:00Z",
    "data_start": "2026-02-01T00:00:00Z",
    "data_end": "2026-03-14T00:00:00Z",
    "n_snapshots": 10000,
    "effectors": [
        {"name": "thermostat_upstairs", "encoding": {"heating": 1.0}, "device_type": "thermostat", "state_gate": "navien_heating"},
        {"name": "thermostat_downstairs", "encoding": {"heating": 1.0}, "device_type": "thermostat", "state_gate": "navien_heating"},
        {"name": "blower_family_room", "encoding": {"off": 0, "low": 1, "high": 2}, "device_type": "blower"},
        {"name": "blower_office", "encoding": {"off": 0, "low": 1, "high": 2}, "device_type": "blower"},
        {"name": "blower_gym", "encoding": {"off": 0, "low": 1, "high": 2}, "device_type": "blower"},
        {"name": "mini_split_bedroom", "encoding": {"off": 0, "heat": 1, "cool": -1}, "device_type": "mini_split"},
        {"name": "mini_split_living_room", "encoding": {"off": 0, "heat": 1, "cool": -1}, "device_type": "mini_split"},
    ],
    "state_gates": {
        "navien_heating": {
            "column": "navien_heating",
            "encoding": {"Space Heating": 1.0, "Idle": 0.0, "Domestic Hot Water": 0.0, "DHW Recirculating": 0.0},
        },
    },
    "sensors": [{"name": s} for s in _SENSOR_COLS],
    "fitted_taus": [{"sensor": s, "tau_base": 45.0} for s in _SENSOR_COLS],
    "effector_sensor_gains": [
        # Thermostats → own zone (significant t-statistics)
        {"effector": "thermostat_upstairs", "sensor": "thermostat_upstairs_temp", "gain_f_per_hour": 0.7, "best_lag_minutes": 45, "t_statistic": 3.0, "negligible": False},
        {"effector": "thermostat_upstairs", "sensor": "upstairs_aggregate_temp", "gain_f_per_hour": 0.6, "best_lag_minutes": 45, "t_statistic": 2.5, "negligible": False},
        {"effector": "thermostat_upstairs", "sensor": "bedroom_temp", "gain_f_per_hour": 0.4, "best_lag_minutes": 60, "t_statistic": 2.0, "negligible": False},
        {"effector": "thermostat_upstairs", "sensor": "bedroom_aggregate_temp", "gain_f_per_hour": 0.9, "best_lag_minutes": 60, "t_statistic": 3.0, "negligible": False},
        {"effector": "thermostat_upstairs", "sensor": "bathroom_temp", "gain_f_per_hour": 0.3, "best_lag_minutes": 55, "t_statistic": 1.8, "negligible": False},
        {"effector": "thermostat_downstairs", "sensor": "thermostat_downstairs_temp", "gain_f_per_hour": 0.8, "best_lag_minutes": 45, "t_statistic": 3.5, "negligible": False},
        {"effector": "thermostat_downstairs", "sensor": "downstairs_aggregate_temp", "gain_f_per_hour": 0.7, "best_lag_minutes": 45, "t_statistic": 3.0, "negligible": False},
        {"effector": "thermostat_downstairs", "sensor": "family_room_temp", "gain_f_per_hour": 0.5, "best_lag_minutes": 50, "t_statistic": 2.5, "negligible": False},
        {"effector": "thermostat_downstairs", "sensor": "kitchen_temp", "gain_f_per_hour": 0.4, "best_lag_minutes": 50, "t_statistic": 2.0, "negligible": False},
        {"effector": "thermostat_downstairs", "sensor": "office_temp", "gain_f_per_hour": 0.3, "best_lag_minutes": 55, "t_statistic": 1.5, "negligible": False},
        {"effector": "thermostat_downstairs", "sensor": "office_bookshelf_temp", "gain_f_per_hour": 0.3, "best_lag_minutes": 55, "t_statistic": 1.5, "negligible": False},
        # Mini splits → own room only (high t-stat, no cross-coupling)
        {"effector": "mini_split_bedroom", "sensor": "bedroom_temp", "gain_f_per_hour": 1.3, "best_lag_minutes": 10, "t_statistic": 4.5, "negligible": False},
        {"effector": "mini_split_bedroom", "sensor": "bedroom_aggregate_temp", "gain_f_per_hour": 1.3, "best_lag_minutes": 10, "t_statistic": 4.9, "negligible": False},
        {"effector": "mini_split_living_room", "sensor": "living_room_temp", "gain_f_per_hour": 0.8, "best_lag_minutes": 5, "t_statistic": 2.5, "negligible": False},
        {"effector": "mini_split_living_room", "sensor": "living_room_climate_temp", "gain_f_per_hour": 0.8, "best_lag_minutes": 5, "t_statistic": 2.5, "negligible": False},
        {"effector": "mini_split_living_room", "sensor": "piano_temp", "gain_f_per_hour": 1.2, "best_lag_minutes": 5, "t_statistic": 1.6, "negligible": False},
    ],
    "solar_gains": [
        {"sensor": s, "hour_of_day": h, "gain_f_per_hour": 0.1}
        for s in _SENSOR_COLS
        for h in range(8, 18)
    ],
}

(_TEST_DATA_DIR / "thermal_params.json").write_text(json.dumps(_THERMAL_PARAMS, indent=2))

# Create subdirectories that code might expect
(_TEST_DATA_DIR / "snapshots").mkdir(exist_ok=True)
(_TEST_DATA_DIR / "predictions").mkdir(exist_ok=True)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Clean up temp data directory after test session."""
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)
