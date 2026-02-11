"""Paths and constants for the weatherstat ML pipeline."""

import os
from pathlib import Path

# Project root is three levels up from this file (ml/src/weatherstat/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"

# Snapshot collection interval (should match HA client config)
SNAPSHOT_INTERVAL_SECONDS = 300

# Location for solar position calculations (Seattle)
LATITUDE = 47.66
LONGITUDE = -122.40
ELEVATION = 30.0  # meters

# Home Assistant connection (shared .env with TS client)
HA_URL = os.environ.get("HA_URL", "")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

# Prediction horizons (in steps). At 5-min intervals: 12=1h, 24=2h, 48=4h.
# At hourly intervals: 1=1h, 2=2h, 4=4h.
HORIZONS_5MIN = [12, 24, 48]  # 1h, 2h, 4h
HORIZONS_HOURLY = [1, 2, 4]  # 1h, 2h, 4h

# Zones to predict temperatures for
PREDICTION_ZONES = ["upstairs", "downstairs"]

# LightGBM training parameters — conservative for small datasets
LGBM_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
}

# More conservative params for the small 10-day dataset
LGBM_PARAMS_SMALL: dict[str, object] = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 30,
    "verbose": -1,
}
