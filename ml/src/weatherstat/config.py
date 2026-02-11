"""Paths and constants for the weatherstat ML pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

# Project root is three levels up from this file (ml/src/weatherstat/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DB = SNAPSHOTS_DIR / "snapshots.db"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"
CONTROL_STATE_FILE = DATA_DIR / "control_state.json"


def experiment_models_dir(name: str) -> Path:
    """Return the models directory for a named experiment.

    Production models live in data/models/. Experiments live in data/models/{name}/.
    The control loop always reads from MODELS_DIR (production).
    """
    return MODELS_DIR / name

# Snapshot collection interval (should match HA client config)
SNAPSHOT_INTERVAL_SECONDS = 300

# Location for solar position calculations (Seattle)
LATITUDE = 47.66
LONGITUDE = -122.40
ELEVATION = 30.0  # meters

# Home Assistant connection (shared .env with TS client)
HA_URL = os.environ.get("HA_URL", "")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

# Prediction horizons (in steps). At 5-min intervals: 12=1h, 24=2h, 48=4h, etc.
# At hourly intervals: 1=1h, 2=2h, 4=4h, etc.
HORIZONS_5MIN = [12, 24, 48, 72, 144]  # 1h, 2h, 4h, 6h, 12h
HORIZONS_HOURLY = [1, 2, 4, 6, 12]  # 1h, 2h, 4h, 6h, 12h

# Rooms to predict temperatures for
PREDICTION_ROOMS = [
    "upstairs",
    "downstairs",
    "bedroom",
    "kitchen",
    "piano",
    "bathroom",
    "family_room",
    "office",
]

# Backward-compatible alias
PREDICTION_ZONES = PREDICTION_ROOMS

# ── HVAC device configuration ─────────────────────────────────────────────


@dataclass(frozen=True)
class BlowerConfig:
    """Configuration for a single blower fan device."""

    name: str  # e.g. "family_room"
    feature_col: str  # e.g. "blower_family_room_mode_enc"
    command_key: str  # e.g. "blowerFamilyRoomMode" (camelCase for executor JSON)
    zone: str = "downstairs"  # thermostat zone — blower only active when zone is heating
    levels: tuple[str, ...] = ("off", "low", "high")  # shorten to ("off", "high") to reduce sweep


@dataclass(frozen=True)
class MiniSplitConfig:
    """Configuration for a single mini-split heat pump device."""

    name: str  # e.g. "bedroom"
    mode_feature_col: str  # e.g. "mini_split_bedroom_mode_enc"
    target_feature_col: str  # e.g. "mini_split_bedroom_target"
    delta_feature_col: str  # e.g. "bedroom_target_delta"
    temp_col: str  # e.g. "mini_split_bedroom_temp"
    command_mode_key: str  # e.g. "miniSplitBedroomMode"
    command_target_key: str  # e.g. "miniSplitBedroomTarget"


BLOWERS: tuple[BlowerConfig, ...] = (
    BlowerConfig(
        name="family_room",
        feature_col="blower_family_room_mode_enc",
        command_key="blowerFamilyRoomMode",
    ),
    BlowerConfig(
        name="office",
        feature_col="blower_office_mode_enc",
        command_key="blowerOfficeMode",
    ),
)

MINI_SPLITS: tuple[MiniSplitConfig, ...] = (
    MiniSplitConfig(
        name="bedroom",
        mode_feature_col="mini_split_bedroom_mode_enc",
        target_feature_col="mini_split_bedroom_target",
        delta_feature_col="bedroom_target_delta",
        temp_col="mini_split_bedroom_temp",
        command_mode_key="miniSplitBedroomMode",
        command_target_key="miniSplitBedroomTarget",
    ),
    MiniSplitConfig(
        name="living_room",
        mode_feature_col="mini_split_living_room_mode_enc",
        target_feature_col="mini_split_living_room_target",
        delta_feature_col="living_room_target_delta",
        temp_col="mini_split_living_room_temp",
        command_mode_key="miniSplitLivingRoomMode",
        command_target_key="miniSplitLivingRoomTarget",
    ),
)

# Mini-split modes swept during control (skip auto/fan_only/dry — not temperature control)
MINI_SPLIT_SWEEP_MODES: tuple[str, ...] = ("off", "heat", "cool")

# Representative target for model override during sweep.
# The actual command target is derived post-sweep from the comfort schedule midpoint.
MINI_SPLIT_SWEEP_TARGET = 72.0

# Mini-split mode encoding (matches features.py split_mode_map)
MINI_SPLIT_MODE_ENC: dict[str, float] = {"off": 0.0, "heat": 1.0, "cool": -1.0}

# Blower mode encoding (matches features.py blower_map)
BLOWER_MODE_ENC: dict[str, float] = {"off": 0.0, "low": 1.0, "high": 2.0}

# Energy cost per device-state (tiebreaker when comfort is equal)
ENERGY_COST_GAS_ZONE = 0.010  # Navien via thermostat — highest
ENERGY_COST_MINI_SPLIT = 0.005  # Heat pump — efficient but uses electricity
ENERGY_COST_BLOWER: dict[str, float] = {"off": 0.0, "low": 0.001, "high": 0.002}


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
