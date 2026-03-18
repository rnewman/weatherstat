"""YAML config loader — single source of truth for all entity IDs, columns, and effectors.

Parses weatherstat.yaml (~/.weatherstat/) into frozen dataclasses.
Loaded once and cached. All modules import from here instead of hardcoding.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ── Config dataclasses ────────────────────────────────────────────────────


@dataclass(frozen=True)
class LocationConfig:
    latitude: float
    longitude: float
    elevation: float
    timezone: str


@dataclass(frozen=True)
class TempSensorConfig:
    column_name: str  # snake_case column in SQLite/Parquet
    entity_id: str
    statistics: bool  # has long-term hourly stats
    role: str = ""  # "outdoor" for the reference sensor


@dataclass(frozen=True)
class HumiditySensorConfig:
    column_name: str
    entity_id: str
    statistics: bool


@dataclass(frozen=True)
class ThermostatConfig:
    name: str  # "upstairs", "downstairs"
    entity_id: str
    zone: str
    state_device: str | None = None  # device that confirms actual delivery (e.g., boiler)


@dataclass(frozen=True)
class MiniSplitYamlConfig:
    name: str  # "bedroom", "living_room"
    entity_id: str
    sweep_modes: tuple[str, ...]
    command_encoding: dict[str, float]  # command: what we set (for control)
    state_encoding: dict[str, float] | None = None  # state: what it's doing (for sysid)
    control_type: str = "binary"  # "regulating" for target-based control
    proportional_band: float = 1.0  # °F — full activity when room is this far from target
    mode_hold_window: tuple[int, int] | None = None  # (start_hour, end_hour) — no mode changes

    @property
    def mode_encoding(self) -> dict[str, float]:
        """Backward compat alias for command_encoding."""
        return self.command_encoding

    @property
    def action_encoding(self) -> dict[str, float] | None:
        """Backward compat alias for state_encoding."""
        return self.state_encoding


@dataclass(frozen=True)
class BlowerYamlConfig:
    name: str  # "family_room", "office"
    entity_id: str
    zone: str
    levels: tuple[str, ...]
    level_encoding: dict[str, float]


@dataclass(frozen=True)
class HealthCheck:
    """Generic device health threshold check."""

    entity_id: str
    min_value: float | None = None  # alert if reading <= this
    max_value: float | None = None  # alert if reading >= this
    severity: str = "warning"
    message: str = ""


@dataclass(frozen=True)
class StateSensorConfig:
    """A categorical sensor with an encoding (e.g., boiler heating mode)."""

    column_name: str  # "navien_heating"
    entity_id: str
    encoding: dict[str, float]


@dataclass(frozen=True)
class PowerSensorConfig:
    """A numeric power/energy sensor."""

    column_name: str  # "navien_gas_usage"
    entity_id: str


@dataclass(frozen=True)
class WindowConfig:
    name: str  # "basement", "family_room", etc.
    entity_id: str


@dataclass(frozen=True)
class ComfortEntry:
    start_hour: int
    end_hour: int
    preferred: float  # ideal temperature — continuous cost in both directions
    min_temp: float  # hard rail — steep additional penalty below this
    max_temp: float  # hard rail — steep additional penalty above this
    cold_penalty: float = 2.0
    hot_penalty: float = 1.0


@dataclass(frozen=True)
class ConstraintSchedule:
    """Comfort constraint on a sensor's value over time."""

    sensor: str  # sensor column name (e.g., "bedroom_temp")
    label: str  # derived display label (e.g., "bedroom")
    entries: tuple[ComfortEntry, ...] = ()


@dataclass(frozen=True)
class ZoneConfig:
    name: str  # "upstairs", "downstairs"
    thermostat: str  # thermostat name within this zone


@dataclass(frozen=True)
class EnergyCostConfig:
    gas_zone: float
    mini_split: float
    blower: dict[str, float]


@dataclass(frozen=True)
class AdvisoryConfig:
    effort_cost: float
    cooldowns: dict[str, int]
    quiet_hours: tuple[int, int] = (22, 7)


@dataclass(frozen=True)
class SafetyConfig:
    cooldowns: dict[str, int] = field(default_factory=dict)


# ── Top-level config ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeatherstatConfig:
    """Top-level config parsed from weatherstat.yaml."""

    location: LocationConfig
    temp_sensors: dict[str, TempSensorConfig]  # col_name -> config
    humidity_sensors: dict[str, HumiditySensorConfig]
    thermostats: dict[str, ThermostatConfig]  # name -> config
    mini_splits: dict[str, MiniSplitYamlConfig]
    blowers: dict[str, BlowerYamlConfig]
    state_sensors: dict[str, StateSensorConfig]
    power_sensors: dict[str, PowerSensorConfig]
    health_checks: list[HealthCheck]
    windows: dict[str, WindowConfig]
    weather_entity: str
    constraints: list[ConstraintSchedule]
    zones: dict[str, ZoneConfig]
    notification_target: str
    energy_costs: EnergyCostConfig
    default_tau: float = 45.0
    advisory: AdvisoryConfig = field(default_factory=lambda: AdvisoryConfig(effort_cost=0.5, cooldowns={}))
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    window_open_offset: tuple[float, float] = (-3.0, 2.0)

    # ── Derived properties (primary) ─────────────────────────────────

    @property
    def prediction_labels(self) -> list[str]:
        """Ordered list of constraint labels for prediction."""
        return [c.label for c in self.constraints]

    @property
    def room_temp_columns(self) -> dict[str, str]:
        """label -> sensor column name for prediction targets."""
        return {c.label: c.sensor for c in self.constraints}

    @property
    def statistics_entities(self) -> dict[str, str]:
        """col_name -> entity_id for sensors with long-term hourly statistics."""
        result: dict[str, str] = {}
        for col, cfg in self.temp_sensors.items():
            if cfg.statistics:
                result[col] = cfg.entity_id
        for col, cfg in self.humidity_sensors.items():
            if cfg.statistics:
                result[col] = cfg.entity_id
        return result

    @property
    def all_temp_columns(self) -> list[str]:
        """All temperature column names (for lag/rolling features)."""
        return list(self.temp_sensors.keys())

    @property
    def climate_entities(self) -> dict[str, str]:
        """prefix -> entity_id for thermostats and mini-splits (history extraction)."""
        result: dict[str, str] = {}
        for name, cfg in self.thermostats.items():
            result[f"thermostat_{name}"] = cfg.entity_id
        for name, cfg in self.mini_splits.items():
            result[f"mini_split_{name}"] = cfg.entity_id
        return result

    @property
    def fan_entities(self) -> dict[str, str]:
        """prefix -> entity_id for blower fans (history extraction)."""
        return {f"blower_{name}": cfg.entity_id for name, cfg in self.blowers.items()}

    @property
    def sensor_entities(self) -> dict[str, str]:
        """column_name -> entity_id for state and power sensors."""
        result: dict[str, str] = {}
        for col, cfg in self.state_sensors.items():
            result[col] = cfg.entity_id
        for col, cfg in self.power_sensors.items():
            result[col] = cfg.entity_id
        return result

    @property
    def window_sensors(self) -> list[str]:
        """Entity IDs for all window sensors."""
        return [cfg.entity_id for cfg in self.windows.values()]

    @property
    def window_column_map(self) -> dict[str, str]:
        """entity_id -> column_name for window sensors."""
        return {cfg.entity_id: f"window_{name}_open" for name, cfg in self.windows.items()}

    @property
    def window_display_map(self) -> dict[str, str]:
        """column_name -> display_name for window sensors."""
        return {f"window_{name}_open": name for name in self.windows}

    @property
    def all_history_entities(self) -> list[str]:
        """All entity IDs for raw history extraction."""
        ids: list[str] = []
        ids.extend(self.statistics_entities.values())
        ids.extend(self.climate_entities.values())
        ids.extend(self.fan_entities.values())
        ids.extend(self.sensor_entities.values())
        ids.append(self.weather_entity)
        ids.extend(self.window_sensors)
        return ids

    @property
    def device_health_checks(self) -> dict[str, list[HealthCheck]]:
        """device_name -> health checks, for generic safety monitoring."""
        result: dict[str, list[HealthCheck]] = {}
        for hc in self.health_checks:
            # Group by entity prefix for display purposes
            key = hc.entity_id.split(".")[-1].split("_")[0] if hc.entity_id else "unknown"
            result.setdefault(key, []).append(hc)
        return result

    def window_columns_for_sensor(self, sensor_name: str) -> list[str]:
        """Derive window columns that might affect a sensor via naming convention.

        Until sysid learns per-window coupling (Phase 3), we use names:
        bedroom_temp → window_bedroom_open (if bedroom window exists).
        """
        label = sensor_name.removeprefix("thermostat_").removesuffix("_temp")
        if label in self.windows:
            return [f"window_{label}_open"]
        return []

    # ── Backward compat properties ───────────────────────────────────

    @property
    def comfort(self) -> dict[str, list[ComfortEntry]]:
        """label -> comfort entries. Backward compat for control.py."""
        return {c.label: list(c.entries) for c in self.constraints}

    # ── Snapshot schema ──────────────────────────────────────────────

    @property
    def exclude_columns(self) -> set[str]:
        """Columns to exclude from features during training.

        Raw categorical strings that have encoded versions.
        """
        cols: set[str] = {"timestamp"}
        for name in self.thermostats:
            cols.add(f"thermostat_{name}_target")
            cols.add(f"thermostat_{name}_action")
        for name in self.mini_splits:
            cols.add(f"mini_split_{name}_mode")
        for name in self.blowers:
            cols.add(f"blower_{name}_mode")
        for col in self.state_sensors:
            cols.add(col)
        cols.add("weather_condition")
        cols.add("any_window_open")
        return cols

    @property
    def hvac_merge_columns(self) -> list[str]:
        """HVAC columns to merge from full-feature sources into baseline hourly data."""
        cols: list[str] = []
        for name in self.thermostats:
            cols.append(f"thermostat_{name}_action")
        for name in self.mini_splits:
            cols.extend([f"mini_split_{name}_temp", f"mini_split_{name}_target", f"mini_split_{name}_mode"])
        for name in self.blowers:
            cols.append(f"blower_{name}_mode")
        for col in self.state_sensors:
            cols.append(col)
        for col in self.power_sensors:
            cols.append(col)
        cols.extend(["weather_condition", "wind_speed", "outdoor_humidity", "met_outdoor_temp"])
        for name in self.windows:
            cols.append(f"window_{name}_open")
        cols.append("any_window_open")
        return cols

    @property
    def numeric_extract_columns(self) -> list[str]:
        """Numeric columns to coerce during extraction."""
        cols: list[str] = list(self.temp_sensors.keys())
        for col in self.humidity_sensors:
            cols.append(col)
        for col in self.power_sensors:
            cols.append(col)
        cols.extend(["outdoor_humidity", "outdoor_wind_speed", "met_outdoor_temp"])
        for name in self.thermostats:
            cols.append(f"thermostat_{name}_target")
        for name in self.mini_splits:
            cols.extend([f"mini_split_{name}_temp", f"mini_split_{name}_target"])
        return cols

    @property
    def window_bool_columns(self) -> list[str]:
        """Window columns that need bool normalization."""
        cols = [f"window_{name}_open" for name in self.windows]
        cols.append("any_window_open")
        return cols

    @property
    def thermostat_action_columns(self) -> list[str]:
        """Thermostat action columns for HVAC encoding."""
        return [f"thermostat_{name}_action" for name in self.thermostats]

    @property
    def mini_split_mode_columns(self) -> list[str]:
        """Mini-split mode columns for HVAC encoding."""
        return [f"mini_split_{name}_mode" for name in self.mini_splits]

    @property
    def blower_mode_columns(self) -> list[str]:
        """Blower mode columns for HVAC encoding."""
        return [f"blower_{name}_mode" for name in self.blowers]

    @property
    def mini_split_delta_pairs(self) -> list[tuple[str, str, str]]:
        """(target_col, temp_col, delta_name) for mini-split delta features."""
        pairs: list[tuple[str, str, str]] = []
        for name in self.mini_splits:
            pairs.append((
                f"mini_split_{name}_target",
                f"mini_split_{name}_temp",
                f"{name}_target_delta",
            ))
        return pairs

    @property
    def column_types(self) -> dict[str, str]:
        """column_name -> SQL_type for all snapshot columns.

        Used by the EAV reader to coerce values after pivoting.
        """
        return {col: sql_type for col, sql_type in self.snapshot_column_defs()}

    def snapshot_column_defs(self) -> list[tuple[str, str]]:
        """(column_name, SQL_type) for all snapshot columns.

        Used for CREATE TABLE and ALTER TABLE migrations.
        """
        defs: list[tuple[str, str]] = []
        # Thermostats
        for name in self.thermostats:
            defs.extend([
                (f"thermostat_{name}_temp", "REAL"),
                (f"thermostat_{name}_target", "REAL"),
                (f"thermostat_{name}_action", "TEXT"),
            ])
        # Mini-splits
        for name, cfg in self.mini_splits.items():
            defs.extend([
                (f"mini_split_{name}_temp", "REAL"),
                (f"mini_split_{name}_target", "REAL"),
                (f"mini_split_{name}_mode", "TEXT"),
            ])
            if cfg.state_encoding:
                defs.append((f"mini_split_{name}_action", "TEXT"))
        # Blowers
        for name in self.blowers:
            defs.append((f"blower_{name}_mode", "TEXT"))
        # State sensors (categorical — stored as TEXT, encoded when used)
        for col in self.state_sensors:
            defs.append((col, "TEXT"))
        # Power sensors (numeric)
        for col in self.power_sensors:
            defs.append((col, "REAL"))
        # Environment
        defs.extend([
            ("outdoor_temp", "REAL"),
            ("met_outdoor_temp", "REAL"),
            ("outdoor_humidity", "REAL"),
            ("wind_speed", "REAL"),
            ("weather_condition", "TEXT"),
        ])
        for col in self.humidity_sensors:
            defs.append((col, "REAL"))
        # Windows
        for name in self.windows:
            defs.append((f"window_{name}_open", "INTEGER"))
        defs.append(("any_window_open", "INTEGER"))
        # Per-room temps (aggregate and individual)
        for col in self.temp_sensors:
            if col not in ("outdoor_temp",) and not col.startswith("thermostat_"):
                defs.append((col, "REAL"))
        return defs


# ── YAML loading ──────────────────────────────────────────────────────────


def _find_yaml_path() -> Path:
    """Find weatherstat.yaml in the data directory (~/.weatherstat/)."""
    from weatherstat._data_dir import resolve_data_dir

    return resolve_data_dir() / "weatherstat.yaml"


def _parse_config(data: dict) -> WeatherstatConfig:
    """Parse raw YAML dict into a WeatherstatConfig."""
    loc = data["location"]
    location = LocationConfig(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        elevation=loc["elevation"],
        timezone=loc["timezone"],
    )

    # Temperature sensors
    temp_sensors: dict[str, TempSensorConfig] = {}
    for col_name, sensor in data["sensors"]["temperature"].items():
        temp_sensors[col_name] = TempSensorConfig(
            column_name=col_name,
            entity_id=sensor["entity_id"],
            statistics=sensor.get("statistics", False),
            role=sensor.get("role", ""),
        )

    # Humidity sensors
    humidity_sensors: dict[str, HumiditySensorConfig] = {}
    for col_name, sensor in data["sensors"]["humidity"].items():
        humidity_sensors[col_name] = HumiditySensorConfig(
            column_name=col_name,
            entity_id=sensor["entity_id"],
            statistics=sensor.get("statistics", False),
        )

    # ── Effectors ────────────────────────────────────────────────────
    eff = data["effectors"]

    thermostats: dict[str, ThermostatConfig] = {}
    for name, dev in eff["thermostats"].items():
        thermostats[name] = ThermostatConfig(
            name=name, entity_id=dev["entity_id"], zone=dev["zone"],
            state_device=dev.get("state_device"),
        )

    mini_splits: dict[str, MiniSplitYamlConfig] = {}
    for name, dev in eff["mini_splits"].items():
        state_enc_raw = dev.get("state_encoding")
        state_enc = {str(k): float(v) for k, v in state_enc_raw.items()} if state_enc_raw else None
        mhw_raw = dev.get("mode_hold_window")
        mode_hold_window = (int(mhw_raw[0]), int(mhw_raw[1])) if mhw_raw else None
        mini_splits[name] = MiniSplitYamlConfig(
            name=name,
            entity_id=dev["entity_id"],
            sweep_modes=tuple(str(m) for m in dev["sweep_modes"]),
            command_encoding={str(k): float(v) for k, v in dev["command_encoding"].items()},
            state_encoding=state_enc,
            control_type=str(dev.get("control_type", "binary")),
            proportional_band=float(dev.get("proportional_band", 3.0)),
            mode_hold_window=mode_hold_window,
        )

    blowers: dict[str, BlowerYamlConfig] = {}
    for name, dev in eff["blowers"].items():
        blowers[name] = BlowerYamlConfig(
            name=name,
            entity_id=dev["entity_id"],
            zone=dev["zone"],
            levels=tuple(str(lv) for lv in dev["levels"]),
            level_encoding={str(k): float(v) for k, v in dev["level_encoding"].items()},
        )

    # ── State sensors (categorical with encoding) ─────────────────────
    state_sensors: dict[str, StateSensorConfig] = {}
    for col_name, sensor in data.get("sensors", {}).get("state", {}).items():
        state_sensors[col_name] = StateSensorConfig(
            column_name=col_name,
            entity_id=sensor["entity_id"],
            encoding={str(k): float(v) for k, v in sensor["encoding"].items()},
        )

    # ── Power sensors (numeric) ──────────────────────────────────────
    power_sensors: dict[str, PowerSensorConfig] = {}
    for col_name, sensor in data.get("sensors", {}).get("power", {}).items():
        power_sensors[col_name] = PowerSensorConfig(
            column_name=col_name,
            entity_id=sensor["entity_id"],
        )

    # ── Health checks (standalone section) ────────────────────────────
    health_checks: list[HealthCheck] = []
    for _name, hc in data.get("health", {}).items():
        health_checks.append(HealthCheck(
            entity_id=hc["entity"],
            min_value=float(hc["min_value"]) if "min_value" in hc else None,
            max_value=float(hc["max_value"]) if "max_value" in hc else None,
            severity=hc.get("severity", "warning"),
            message=hc.get("message", ""),
        ))

    # ── Backward compat: synthesize state sensor from effectors.boiler ──
    if not state_sensors and "boiler" in eff:
        boiler_items = list(eff["boiler"].items())
        boiler_name, boiler_data = boiler_items[0]
        state_sensors[f"{boiler_name}_heating"] = StateSensorConfig(
            column_name=f"{boiler_name}_heating",
            entity_id=boiler_data["mode_entity"],
            encoding={str(k): float(v) for k, v in boiler_data["mode_encoding"].items()},
        )
        # Migrate health checks from boiler config
        for hc_data in boiler_data.get("health", []):
            health_checks.append(HealthCheck(
                entity_id=hc_data["entity"],
                min_value=float(hc_data["min_value"]) if "min_value" in hc_data else None,
                max_value=float(hc_data["max_value"]) if "max_value" in hc_data else None,
                severity=hc_data.get("severity", "warning"),
                message=hc_data.get("message", ""),
            ))
        # Rewrite thermostat state_device refs from old boiler name to new sensor name
        for name, tcfg in thermostats.items():
            if tcfg.state_device == boiler_name:
                thermostats[name] = ThermostatConfig(
                    name=name, entity_id=tcfg.entity_id, zone=tcfg.zone,
                    state_device=f"{boiler_name}_heating",
                )

    # ── Windows ──────────────────────────────────────────────────────
    windows: dict[str, WindowConfig] = {}
    for name, win in data["windows"].items():
        windows[name] = WindowConfig(name=name, entity_id=win["entity_id"])

    # ── Constraints ──────────────────────────────────────────────────
    constraints_data = data.get("constraints", {})
    wo_offset = constraints_data.get("window_open_offset", {"min": -3, "max": 2})
    window_open_offset = (float(wo_offset["min"]), float(wo_offset["max"]))

    constraint_list: list[ConstraintSchedule] = []
    for sched in constraints_data.get("schedules", []):
        sensor = sched["sensor"]
        label = sensor.removeprefix("thermostat_").removesuffix("_temp")
        entries: list[ComfortEntry] = []
        for entry in sched["schedule"]:
            hours = entry["hours"]
            min_t = float(entry["min"])
            max_t = float(entry["max"])
            # Default preferred to midpoint if not specified (backward compat)
            preferred = float(entry.get("preferred", (min_t + max_t) / 2))
            entries.append(ComfortEntry(
                start_hour=hours[0],
                end_hour=hours[1],
                preferred=preferred,
                min_temp=min_t,
                max_temp=max_t,
                cold_penalty=float(entry.get("cold_penalty", 2.0)),
                hot_penalty=float(entry.get("hot_penalty", 1.0)),
            ))
        constraint_list.append(ConstraintSchedule(
            sensor=sensor, label=label, entries=tuple(entries),
        ))

    # ── Zones ────────────────────────────────────────────────────────
    zones: dict[str, ZoneConfig] = {}
    for name, zcfg in data.get("zones", {}).items():
        zones[name] = ZoneConfig(name=name, thermostat=zcfg["thermostat"])

    # ── Energy costs ─────────────────────────────────────────────────
    ec = data["energy_costs"]
    energy_costs = EnergyCostConfig(
        gas_zone=float(ec["gas_zone"]),
        mini_split=float(ec["mini_split"]),
        blower={str(k): float(v) for k, v in ec["blower"].items()},
    )

    # ── Advisory config ──────────────────────────────────────────────
    adv_data = data.get("advisory", {})
    adv_quiet = adv_data.get("quiet_hours", [22, 7])
    advisory_config = AdvisoryConfig(
        effort_cost=float(adv_data.get("effort_cost", 0.5)),
        cooldowns={str(k): int(v) for k, v in adv_data.get("cooldowns", {}).items()},
        quiet_hours=(int(adv_quiet[0]), int(adv_quiet[1])),
    )

    # ── Safety config (just cooldowns — device health is on effectors) ─
    safety_data = data.get("safety", {})
    safety_config = SafetyConfig(
        cooldowns={str(k): int(v) for k, v in safety_data.get("cooldowns", {}).items()},
    )

    # ── Defaults ─────────────────────────────────────────────────────
    defaults = data.get("defaults", {})
    default_tau = float(defaults.get("tau", 45.0))

    return WeatherstatConfig(
        location=location,
        temp_sensors=temp_sensors,
        humidity_sensors=humidity_sensors,
        thermostats=thermostats,
        mini_splits=mini_splits,
        blowers=blowers,
        state_sensors=state_sensors,
        power_sensors=power_sensors,
        health_checks=health_checks,
        windows=windows,
        weather_entity=data["weather"]["entity_id"],
        constraints=constraint_list,
        zones=zones,
        notification_target=data["notifications"]["target"],
        energy_costs=energy_costs,
        default_tau=default_tau,
        advisory=advisory_config,
        safety=safety_config,
        window_open_offset=window_open_offset,
    )


@functools.cache
def load_config(yaml_path: str | None = None) -> WeatherstatConfig:
    """Load and cache the weatherstat config.

    Args:
        yaml_path: Override path to weatherstat.yaml (for testing).
    """
    path = Path(yaml_path) if yaml_path else _find_yaml_path()
    with open(path) as f:
        data = yaml.safe_load(f)
    return _parse_config(data)
