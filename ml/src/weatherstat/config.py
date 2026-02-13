"""Paths and constants for the weatherstat ML pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

from weatherstat.yaml_config import load_config as _load_config

# Project root is three levels up from this file (ml/src/weatherstat/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DB = SNAPSHOTS_DIR / "snapshots.db"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"
METRICS_DIR = DATA_DIR / "metrics"
CONTROL_STATE_FILE = DATA_DIR / "control_state.json"
ADVISORY_STATE_FILE = DATA_DIR / "advisory_state.json"


def experiment_models_dir(name: str) -> Path:
    """Return the models directory for a named experiment.

    Production models live in data/models/. Experiments live in data/models/{name}/.
    The control loop always reads from MODELS_DIR (production).
    """
    return MODELS_DIR / name

# Snapshot collection interval (should match HA client config)
SNAPSHOT_INTERVAL_SECONDS = 300

# ── Config-driven constants (from weatherstat.yaml) ───────────────────────

_CFG = _load_config()

# Location for solar position calculations
LATITUDE = _CFG.location.latitude
LONGITUDE = _CFG.location.longitude
ELEVATION = _CFG.location.elevation
TIMEZONE = _CFG.location.timezone

# Home Assistant connection (shared .env with TS client)
HA_URL = os.environ.get("HA_URL", "")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

# Prediction horizons (in steps). At 5-min intervals: 12=1h, 24=2h, 48=4h, etc.
# At hourly intervals: 1=1h, 2=2h, 4=4h, etc.
HORIZONS_5MIN = [12, 24, 48, 72, 144]  # 1h, 2h, 4h, 6h, 12h
HORIZONS_HOURLY = [1, 2, 4, 6, 12]  # 1h, 2h, 4h, 6h, 12h

# Rooms to predict temperatures for (from YAML config)
PREDICTION_ROOMS = _CFG.prediction_rooms

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


def _snake_to_camel(s: str) -> str:
    """Convert snake_case to camelCase."""
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# Build BlowerConfig from YAML
BLOWERS: tuple[BlowerConfig, ...] = tuple(
    BlowerConfig(
        name=name,
        feature_col=f"blower_{name}_mode_enc",
        command_key=_snake_to_camel(f"blower_{name}_mode"),
        zone=cfg.zone,
        levels=cfg.levels,
    )
    for name, cfg in _CFG.blowers.items()
)

# Build MiniSplitConfig from YAML
MINI_SPLITS: tuple[MiniSplitConfig, ...] = tuple(
    MiniSplitConfig(
        name=name,
        mode_feature_col=f"mini_split_{name}_mode_enc",
        target_feature_col=f"mini_split_{name}_target",
        delta_feature_col=f"{name}_target_delta",
        temp_col=f"mini_split_{name}_temp",
        command_mode_key=_snake_to_camel(f"mini_split_{name}_mode"),
        command_target_key=_snake_to_camel(f"mini_split_{name}_target"),
    )
    for name, cfg in _CFG.mini_splits.items()
)

# Mini-split modes swept during control (from YAML)
MINI_SPLIT_SWEEP_MODES: tuple[str, ...] = next(iter(_CFG.mini_splits.values())).sweep_modes

# Representative target for model override during sweep.
# The actual command target is derived post-sweep from the comfort schedule midpoint.
MINI_SPLIT_SWEEP_TARGET = 72.0

# Mini-split mode encoding (from YAML, filtered to sweep modes for control)
MINI_SPLIT_MODE_ENC: dict[str, float] = {
    mode: enc for mode, enc in next(iter(_CFG.mini_splits.values())).mode_encoding.items()
    if mode in MINI_SPLIT_SWEEP_MODES
}

# Blower mode encoding (from YAML)
BLOWER_MODE_ENC: dict[str, float] = next(iter(_CFG.blowers.values())).level_encoding

# Energy cost per device-state (from YAML, tiebreaker when comfort is equal)
ENERGY_COST_GAS_ZONE = _CFG.energy_costs.gas_zone
ENERGY_COST_MINI_SPLIT = _CFG.energy_costs.mini_split
ENERGY_COST_BLOWER: dict[str, float] = _CFG.energy_costs.blower

# Advisory configuration (from YAML)
ADVISORY_EFFORT_COST: float = _CFG.advisory.effort_cost
ADVISORY_COOLDOWNS: dict[str, int] = _CFG.advisory.cooldowns


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
