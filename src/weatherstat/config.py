"""Paths and constants for the weatherstat ML pipeline."""

import os
from dataclasses import dataclass, field

from weatherstat._data_dir import resolve_data_dir
from weatherstat.yaml_config import load_config as _load_config

DATA_DIR = resolve_data_dir()
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DB = SNAPSHOTS_DIR / "snapshots.db"
PREDICTIONS_DIR = DATA_DIR / "predictions"
CONTROL_STATE_FILE = DATA_DIR / "control_state.json"
ADVISORY_STATE_FILE = DATA_DIR / "advisory_state.json"
DECISION_LOG_DB = DATA_DIR / "decision_log.db"


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
HORIZONS_5MIN = [12, 24, 48, 72, 144]  # 1h, 2h, 4h, 6h, 12h

# Constraint sensor columns and display labels (from YAML config)
PREDICTION_SENSORS = _CFG.prediction_sensors
SENSOR_LABELS = _CFG.sensor_labels  # sensor -> label

# ── HVAC device configuration ─────────────────────────────────────────────


@dataclass(frozen=True)
class EffectorConfig:
    """Unified configuration for any HVAC effector.

    Properties replace categories:
    - control_type: "trajectory" (slow-twitch on/off timing), "regulating"
      (proportional band with target), "binary" (discrete modes)
    - mode_control: "manual" (human controls mode, e.g., thermostat heat/off)
      vs "automatic" (system controls mode, e.g., mini-split)
    - supported_modes: what the device can do ("heat",), ("heat", "cool"), etc.
    - depends_on: effector name this depends on for useful output
    """

    name: str  # "thermostat_upstairs", "mini_split_bedroom", "blower_office"
    entity_id: str
    control_type: str  # "trajectory", "regulating", "binary"
    mode_control: str  # "manual" or "automatic"
    supported_modes: tuple[str, ...]  # ("heat",), ("heat", "cool"), ("off", "low", "high")
    command_keys: dict[str, str]  # purpose -> camelCase key: {"target": "thermostatUpstairsTarget"}
    depends_on: tuple[str, ...] = ()  # effector names: ("thermostat_downstairs",)
    state_device: str | None = None  # state sensor confirming delivery
    proportional_band: float = 1.0  # regulating: activity ramp width (in configured unit)
    mode_hold_window: tuple[int, int] | None = None  # quiet hours for mode changes
    mode_encoding: dict[str, float] = field(default_factory=dict)  # for energy cost / sysid
    temp_col: str = ""  # sensor column for current temp (climate entities)
    energy_cost: float | dict[str, float] = 0.0  # per-effector energy cost (scalar or per-mode dict)


def _snake_to_camel(s: str) -> str:
    """Convert snake_case to camelCase."""
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# Build unified EFFECTORS from YAML — single loop over flat effector dict
_effectors: list[EffectorConfig] = []

for name, cfg in _CFG.effectors.items():
    # Derive command keys from domain and mode_control
    if cfg.domain == "climate":
        if cfg.mode_control == "manual":
            command_keys = {"target": _snake_to_camel(f"{name}_target")}
        else:
            command_keys = {
                "mode": _snake_to_camel(f"{name}_mode"),
                "target": _snake_to_camel(f"{name}_target"),
            }
        temp_col = f"{name}_temp"
    else:
        command_keys = {"mode": _snake_to_camel(f"{name}_mode")}
        temp_col = ""

    _effectors.append(EffectorConfig(
        name=name,
        entity_id=cfg.entity_id,
        control_type=cfg.control_type,
        mode_control=cfg.mode_control,
        supported_modes=cfg.supported_modes,
        command_keys=command_keys,
        depends_on=cfg.depends_on,
        state_device=cfg.state_device,
        proportional_band=cfg.proportional_band,
        mode_hold_window=cfg.mode_hold_window,
        mode_encoding=cfg.command_encoding or cfg.state_encoding,
        temp_col=temp_col,
        energy_cost=cfg.energy_cost,
    ))

EFFECTORS: tuple[EffectorConfig, ...] = tuple(_effectors)
EFFECTOR_MAP: dict[str, EffectorConfig] = {e.name: e for e in EFFECTORS}

# Advisory configuration (from YAML)
ADVISORY_COOLDOWNS: dict[str, int] = _CFG.advisory.cooldowns
ADVISORY_QUIET_HOURS: tuple[int, int] = _CFG.advisory.quiet_hours
ADVISORY_NOTIFICATION_THRESHOLD: float = _CFG.advisory.notification_threshold


# ── Intervals ──────────────────────────────────────────────────────────

CONTROL_INTERVAL: int = _CFG.control_interval  # seconds between control cycles
SYSID_INTERVAL: int = _CFG.sysid_interval  # seconds between automatic sysid (0 = disabled)

# ── Temperature unit helpers ─────────────────────────────────────────────

UNIT_SYMBOL: str = _CFG.unit_symbol


def abs_temp(fahrenheit: float) -> float:
    """Convert an absolute temperature from canonical °F to the configured unit."""
    return _CFG.abs_temp(fahrenheit)


def delta_temp(fahrenheit_delta: float) -> float:
    """Convert a temperature delta from canonical °F to the configured unit."""
    return _CFG.delta_temp(fahrenheit_delta)

