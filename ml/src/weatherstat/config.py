"""Paths and constants for the weatherstat ML pipeline."""

from pathlib import Path

# Project root is three levels up from this file (ml/src/weatherstat/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"

MODEL_FILENAME = "weatherstat_lgbm.txt"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Snapshot collection interval (should match HA client config)
SNAPSHOT_INTERVAL_SECONDS = 300

# Location for solar position calculations (placeholder — update for your location)
LATITUDE = 45.5
LONGITUDE = -122.7
ELEVATION = 50.0  # meters

# LightGBM training parameters
LGBM_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
}
