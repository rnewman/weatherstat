"""YAML config loader — single source of truth for all entity IDs, columns, and devices.

Parses weatherstat.yaml (project root) into frozen dataclasses.
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
    mode_encoding: dict[str, float]  # command: what we set (for control)
    action_encoding: dict[str, float] | None = None  # state: what it's doing (for sysid)


@dataclass(frozen=True)
class BlowerYamlConfig:
    name: str  # "family_room", "office"
    entity_id: str
    zone: str
    levels: tuple[str, ...]
    level_encoding: dict[str, float]


@dataclass(frozen=True)
class BoilerConfig:
    name: str  # "navien"
    mode_entity: str
    capacity_entity: str
    mode_encoding: dict[str, float]


@dataclass(frozen=True)
class WindowConfig:
    name: str  # "basement", "family_room", etc.
    entity_id: str
    rooms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoomConfig:
    name: str
    temp_column: str
    zone: str


@dataclass(frozen=True)
class ComfortEntry:
    start_hour: int
    end_hour: int
    min_temp: float
    max_temp: float
    cold_penalty: float = 2.0
    hot_penalty: float = 1.0


@dataclass(frozen=True)
class EnergyCostConfig:
    gas_zone: float
    mini_split: float
    blower: dict[str, float]


@dataclass(frozen=True)
class ExtraTempSensorConfig:
    name: str
    entity_id: str


@dataclass(frozen=True)
class ThermalConfig:
    tau_sealed: dict[str, float]  # room_name -> sealed time constant (hours)
    tau_ventilated: dict[str, float]  # room_name -> ventilated time constant (hours)
    default_tau_sealed: float = 45.0
    default_tau_ventilated: float = 20.0


@dataclass(frozen=True)
class NavienSafetyConfig:
    return_temp_entity: str = ""
    outlet_temp_entity: str = ""
    disconnected_threshold: float = 33.0  # °F


@dataclass(frozen=True)
class SafetyConfig:
    navien: NavienSafetyConfig = field(default_factory=NavienSafetyConfig)
    cooldowns: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AdvisoryConfig:
    effort_cost: float
    cooldowns: dict[str, int]
    quiet_hours: tuple[int, int] = (22, 7)


@dataclass(frozen=True)
class WeatherstatConfig:
    """Top-level config parsed from weatherstat.yaml."""

    location: LocationConfig
    temp_sensors: dict[str, TempSensorConfig]  # col_name -> config
    humidity_sensors: dict[str, HumiditySensorConfig]
    thermostats: dict[str, ThermostatConfig]  # name -> config
    mini_splits: dict[str, MiniSplitYamlConfig]
    blowers: dict[str, BlowerYamlConfig]
    boiler: BoilerConfig
    windows: dict[str, WindowConfig]
    weather_entity: str
    rooms: dict[str, RoomConfig]
    comfort: dict[str, list[ComfortEntry]]
    notification_target: str
    energy_costs: EnergyCostConfig
    extra_temp_sensors: dict[str, ExtraTempSensorConfig] = field(default_factory=dict)
    thermal: ThermalConfig = field(default_factory=lambda: ThermalConfig(tau={}, default_tau=45.0))
    advisory: AdvisoryConfig = field(default_factory=lambda: AdvisoryConfig(effort_cost=0.5, cooldowns={}))
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    window_open_offset: tuple[float, float] = (-3.0, 2.0)  # (min_offset, max_offset)

    # ── Derived properties ────────────────────────────────────────────

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
    def room_temp_columns(self) -> dict[str, str]:
        """room_name -> temp_column for prediction targets."""
        return {name: cfg.temp_column for name, cfg in self.rooms.items()}

    @property
    def prediction_rooms(self) -> list[str]:
        """Ordered list of room names for prediction."""
        return list(self.rooms.keys())

    @property
    def room_to_zone(self) -> dict[str, str]:
        """room_name -> zone_name for control constraints."""
        return {name: cfg.zone for name, cfg in self.rooms.items()}

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
        """prefix -> entity_id for boiler sensors."""
        result: dict[str, str] = {}
        result["navien_heating_mode"] = self.boiler.mode_entity
        result["navien_heat_capacity"] = self.boiler.capacity_entity
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
    def exclude_columns(self) -> set[str]:
        """Columns to exclude from features during training.

        Raw categorical strings that have encoded versions.
        """
        cols: set[str] = {"timestamp"}
        # Thermostat targets (binary controllers — setpoint is not meaningful)
        for name in self.thermostats:
            cols.add(f"thermostat_{name}_target")
            cols.add(f"thermostat_{name}_action")
        # Mini-split raw mode strings
        for name in self.mini_splits:
            cols.add(f"mini_split_{name}_mode")
        # Blower raw mode strings
        for name in self.blowers:
            cols.add(f"blower_{name}_mode")
        # Boiler raw mode string
        cols.add("navien_heating_mode")
        # Weather condition string
        cols.add("weather_condition")
        # Redundant aggregate — per-window features are sufficient
        cols.add("any_window_open")
        return cols

    @property
    def hvac_merge_columns(self) -> list[str]:
        """HVAC columns to merge from full-feature sources into baseline hourly data."""
        cols: list[str] = []
        # Thermostat actions
        for name in self.thermostats:
            cols.append(f"thermostat_{name}_action")
        # Mini-split features
        for name in self.mini_splits:
            cols.extend([f"mini_split_{name}_temp", f"mini_split_{name}_target", f"mini_split_{name}_mode"])
        # Blower modes
        for name in self.blowers:
            cols.append(f"blower_{name}_mode")
        # Boiler
        cols.extend(["navien_heating_mode", "navien_heat_capacity"])
        # Weather + windows
        cols.extend(["weather_condition", "wind_speed", "outdoor_humidity"])
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
        cols.append("navien_heat_capacity")
        cols.extend(["outdoor_humidity", "outdoor_wind_speed"])
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
    def safety_navien(self) -> NavienSafetyConfig | None:
        """Navien safety check config, or None if not configured."""
        cfg = self.safety.navien
        if cfg.return_temp_entity:
            return cfg
        return None

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
            if cfg.action_encoding:
                defs.append((f"mini_split_{name}_action", "TEXT"))
        # Blowers
        for name in self.blowers:
            defs.append((f"blower_{name}_mode", "TEXT"))
        # Boiler
        defs.extend([("navien_heating_mode", "TEXT"), ("navien_heat_capacity", "REAL")])
        # Environment
        defs.extend([
            ("outdoor_temp", "REAL"),
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
    """Find weatherstat.yaml relative to this file (ml/src/weatherstat/ -> project root)."""
    return Path(__file__).resolve().parents[3] / "weatherstat.yaml"


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
        )

    # Humidity sensors
    humidity_sensors: dict[str, HumiditySensorConfig] = {}
    for col_name, sensor in data["sensors"]["humidity"].items():
        humidity_sensors[col_name] = HumiditySensorConfig(
            column_name=col_name,
            entity_id=sensor["entity_id"],
            statistics=sensor.get("statistics", False),
        )

    # Devices
    thermostats: dict[str, ThermostatConfig] = {}
    for name, dev in data["devices"]["thermostats"].items():
        thermostats[name] = ThermostatConfig(
            name=name, entity_id=dev["entity_id"], zone=dev["zone"],
            state_device=dev.get("state_device"),
        )

    mini_splits: dict[str, MiniSplitYamlConfig] = {}
    for name, dev in data["devices"]["mini_splits"].items():
        action_enc_raw = dev.get("action_encoding")
        action_enc = {str(k): float(v) for k, v in action_enc_raw.items()} if action_enc_raw else None
        mini_splits[name] = MiniSplitYamlConfig(
            name=name,
            entity_id=dev["entity_id"],
            sweep_modes=tuple(dev["sweep_modes"]),
            mode_encoding={str(k): float(v) for k, v in dev["mode_encoding"].items()},
            action_encoding=action_enc,
        )

    blowers: dict[str, BlowerYamlConfig] = {}
    for name, dev in data["devices"]["blowers"].items():
        blowers[name] = BlowerYamlConfig(
            name=name,
            entity_id=dev["entity_id"],
            zone=dev["zone"],
            levels=tuple(dev["levels"]),
            level_encoding={str(k): float(v) for k, v in dev["level_encoding"].items()},
        )

    boiler_data = list(data["devices"]["boiler"].values())[0]
    boiler_name = list(data["devices"]["boiler"].keys())[0]
    boiler = BoilerConfig(
        name=boiler_name,
        mode_entity=boiler_data["mode_entity"],
        capacity_entity=boiler_data["capacity_entity"],
        mode_encoding={str(k): float(v) for k, v in boiler_data["mode_encoding"].items()},
    )

    # Windows
    windows: dict[str, WindowConfig] = {}
    for name, win in data["windows"].items():
        windows[name] = WindowConfig(
            name=name,
            entity_id=win["entity_id"],
            rooms=tuple(win.get("rooms", [])),
        )

    # Rooms
    rooms: dict[str, RoomConfig] = {}
    for name, room in data["rooms"].items():
        rooms[name] = RoomConfig(name=name, temp_column=room["temp_column"], zone=room["zone"])

    # Window-open comfort offset (optional, in comfort section)
    comfort_data = data["comfort"]
    wo_offset = comfort_data.get("window_open_offset", {"min": -3, "max": 2})
    window_open_offset = (float(wo_offset["min"]), float(wo_offset["max"]))

    # Comfort schedules
    comfort: dict[str, list[ComfortEntry]] = {}
    for room, entries in comfort_data.items():
        if room == "window_open_offset":
            continue
        comfort[room] = []
        for entry in entries:
            hours = entry["hours"]
            comfort[room].append(ComfortEntry(
                start_hour=hours[0],
                end_hour=hours[1],
                min_temp=float(entry["min"]),
                max_temp=float(entry["max"]),
                cold_penalty=float(entry.get("cold_penalty", 2.0)),
                hot_penalty=float(entry.get("hot_penalty", 1.0)),
            ))

    # Energy costs
    ec = data["energy_costs"]
    energy_costs = EnergyCostConfig(
        gas_zone=float(ec["gas_zone"]),
        mini_split=float(ec["mini_split"]),
        blower={str(k): float(v) for k, v in ec["blower"].items()},
    )

    # Extra temp sensors (optional)
    extra_temp_sensors: dict[str, ExtraTempSensorConfig] = {}
    for name, sensor in data.get("extra_temp_sensors", {}).items():
        extra_temp_sensors[name] = ExtraTempSensorConfig(name=name, entity_id=sensor["entity_id"])

    # Thermal config (optional, with defaults)
    thermal_data = data.get("thermal", {})
    thermal_config = ThermalConfig(
        tau_sealed={str(k): float(v) for k, v in thermal_data.get("tau_sealed", {}).items()},
        tau_ventilated={str(k): float(v) for k, v in thermal_data.get("tau_ventilated", {}).items()},
        default_tau_sealed=float(thermal_data.get("default_tau_sealed", 45.0)),
        default_tau_ventilated=float(thermal_data.get("default_tau_ventilated", 20.0)),
    )

    # Advisory config (optional, with defaults)
    adv_data = data.get("advisory", {})
    adv_quiet = adv_data.get("quiet_hours", [22, 7])
    advisory_config = AdvisoryConfig(
        effort_cost=float(adv_data.get("effort_cost", 0.5)),
        cooldowns={str(k): int(v) for k, v in adv_data.get("cooldowns", {}).items()},
        quiet_hours=(int(adv_quiet[0]), int(adv_quiet[1])),
    )

    # Safety config (optional)
    safety_data = data.get("safety", {})
    navien_data = safety_data.get("navien", {})
    safety_config = SafetyConfig(
        navien=NavienSafetyConfig(
            return_temp_entity=str(navien_data.get("return_temp_entity", "")),
            outlet_temp_entity=str(navien_data.get("outlet_temp_entity", "")),
            disconnected_threshold=float(navien_data.get("disconnected_threshold", 33.0)),
        ),
        cooldowns={str(k): int(v) for k, v in safety_data.get("cooldowns", {}).items()},
    )

    return WeatherstatConfig(
        location=location,
        temp_sensors=temp_sensors,
        humidity_sensors=humidity_sensors,
        thermostats=thermostats,
        mini_splits=mini_splits,
        blowers=blowers,
        boiler=boiler,
        windows=windows,
        weather_entity=data["weather"]["entity_id"],
        rooms=rooms,
        comfort=comfort,
        notification_target=data["notifications"]["target"],
        energy_costs=energy_costs,
        extra_temp_sensors=extra_temp_sensors,
        thermal=thermal_config,
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
