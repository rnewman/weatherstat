"""YAML config loader — single source of truth for all entity IDs, columns, and effectors.

Parses weatherstat.yaml (~/.weatherstat/) into frozen dataclasses.
Loaded once and cached. All modules import from here instead of hardcoding.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from weatherstat.types import EnvironmentEntryConfig

# ── Config dataclasses ────────────────────────────────────────────────────


@dataclass(frozen=True)
class LocationConfig:
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    unit: str = "F"  # "F" or "C"


@dataclass(frozen=True)
class TempSensorConfig:
    column_name: str  # snake_case column in SQLite/Parquet
    entity_id: str
    role: str = ""  # "outdoor" for the reference sensor


@dataclass(frozen=True)
class HumiditySensorConfig:
    column_name: str
    entity_id: str


@dataclass(frozen=True)
class EffectorYamlConfig:
    """Unified effector config — all device types in a single flat dict.

    The effector key in YAML is the full name (e.g., 'thermostat_upstairs').
    Properties replace categories:
    - control_type: "trajectory" (slow-twitch), "regulating" (proportional), "binary" (discrete)
    - mode_control: "manual" (human controls mode) or "automatic" (system controls mode)
    - domain: HA entity domain ("climate" or "fan"), derived from entity_id
    """

    name: str  # "thermostat_upstairs", "mini_split_bedroom", "blower_office"
    entity_id: str
    control_type: str  # "trajectory", "regulating", "binary"
    mode_control: str  # "manual", "automatic"
    supported_modes: tuple[str, ...]  # ("heat",), ("heat", "cool"), ("off", "low", "high")
    state_encoding: dict[str, float]  # maps measured state to numeric (for sysid)
    max_lag_minutes: int = 90  # thermal response lag
    energy_cost: float | dict[str, float] = 0.0  # scalar or per-mode dict
    command_encoding: dict[str, float] | None = None  # separate command encoding (auto-mode climate)
    state_device: str | None = None  # state sensor confirming delivery
    depends_on: tuple[str, ...] = ()  # effector names this depends on (ALL must be active)
    proportional_band: float = 1.0  # regulating: activity ramp width (°F)
    mode_hold_window: tuple[int, int] | None = None  # quiet hours for mode changes

    @property
    def domain(self) -> str:
        """HA entity domain, derived from entity_id."""
        return self.entity_id.split(".")[0]


@dataclass(frozen=True)
class HealthCheck:
    """Generic device health threshold check."""

    name: str  # YAML key, e.g. "navien_connection" — used as alert key
    entity_id: str
    min_value: float | None = None  # alert if reading <= this
    max_value: float | None = None  # alert if reading >= this
    expected_state: str | None = None  # alert if state != this (for binary/enum entities)
    severity: str = "warning"
    message: str = ""
    # Optional condition: only evaluate this check when another entity is in a
    # given state for at least ``when_for_minutes`` minutes.
    when_entity: str | None = None
    when_state: str | None = None
    when_for_minutes: float = 0


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


def environment_display(name: str, cfg: EnvironmentEntryConfig | None = None) -> tuple[str, str]:
    """Return (label, kind) for an environment factor.

    With config: uses config's label and kind directly.
    Without config (fallback): derives from name — 'door_basement' → ('basement', 'door').
    """
    if cfg is not None:
        return cfg.label, cfg.kind
    # Fallback for contexts without config (e.g., historical data).
    # Derive kind from naming convention prefix.
    for prefix in ("door_", "shade_", "vent_", "heater_"):
        if name.startswith(prefix):
            kind = prefix.rstrip("_")
            return name.removeprefix(prefix).replace("_", " "), kind
    return name.replace("_", " "), "window"


@dataclass(frozen=True)
class ComfortEntry:
    start_hour: int
    end_hour: int
    preferred_lo: float  # lower edge of preferred dead band
    preferred_hi: float  # upper edge (= preferred_lo for point target)
    acceptable_lo: float  # steep penalty below this (was min_temp)
    acceptable_hi: float  # steep penalty above this (was max_temp)
    backup_lo: float  # worst-case hedge: defensive action below this
    backup_hi: float  # worst-case hedge: defensive action above this
    cold_penalty: float = 2.0
    hot_penalty: float = 1.0


@dataclass(frozen=True)
class ComfortProfile:
    """Offset-based comfort profile applied on top of base schedules.

    When active, offsets are added to every schedule entry's preferred/acceptable/backup temps.
    An empty profile (all zeros) means "use base schedules unchanged".

    penalty_scale multiplies cold_penalty and hot_penalty. Use < 1.0 to make the
    optimizer care less about reaching preferred while still respecting hard rails
    (acceptable). With penalty_scale=0.1, the optimizer won't spend energy chasing
    preferred but will still act when temps approach acceptable boundaries.

    preferred_widen expands the preferred point into a dead band: preferred becomes
    [preferred - widen/2, preferred + widen/2]. Within the band, comfort cost is zero.
    """

    name: str
    preferred_offset: float = 0.0
    preferred_widen: float = 0.0
    min_offset: float = 0.0  # applied to acceptable_lo and backup_lo
    max_offset: float = 0.0  # applied to acceptable_hi and backup_hi
    penalty_scale: float = 1.0


@dataclass(frozen=True)
class MrtCorrectionConfig:
    """Correction for mean radiant temperature effects on perceived comfort.

    Cold exterior walls and windows lower the mean radiant temperature (MRT),
    making a room feel colder than the air temperature reads. This correction
    raises comfort targets when it's cold outside and lowers them when warm.

    Sun streaming through windows heats interior surfaces, raising MRT above
    what outdoor temp alone suggests. The solar_response parameter controls
    how much the current solar forcing reduces the cold-wall correction.

    Per sensor:
      solar_warming = β_solar(sensor) × sin⁺(elevation) × solar_fraction × solar_response
      effective_outdoor = outdoor_temp + solar_warming
      offset = clamp(alpha × (reference_temp - effective_outdoor), -max_offset, +max_offset)
    """

    alpha: float  # °F comfort shift per °F outdoor deviation from reference
    reference_temp: float  # outdoor temp at which current comfort targets feel right (°F)
    max_offset: float  # cap on correction magnitude (°F)
    solar_response: float = 2.0  # hours-equivalent: how much solar forcing raises effective outdoor


@dataclass(frozen=True)
class ConstraintSchedule:
    """Comfort constraint on a sensor's value over time."""

    sensor: str  # sensor column name (e.g., "bedroom_temp")
    label: str  # derived display label (e.g., "bedroom")
    entries: tuple[ComfortEntry, ...] = ()
    mrt_weight: float = 1.0  # multiplier on global MRT offset for this sensor


@dataclass(frozen=True)
class AdvisoryConfig:
    cooldowns: dict[str, int]
    quiet_hours: tuple[int, int] = (22, 7)
    opportunity_threshold: float = 0.3  # minimum benefit to track
    notification_threshold: float = 1.5  # minimum benefit to push notification


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
    effectors: dict[str, EffectorYamlConfig]  # full_name -> config
    state_sensors: dict[str, StateSensorConfig]
    power_sensors: dict[str, PowerSensorConfig]
    health_checks: list[HealthCheck]
    environment: dict[str, EnvironmentEntryConfig]
    weather_entity: str
    constraints: list[ConstraintSchedule]
    notification_target: str
    default_tau: float = 45.0
    # Control parameters — optional overrides (in configured unit), None = built-in default
    setpoint_min: float | None = None  # absolute minimum setpoint (default: 62°F / 17°C)
    setpoint_max: float | None = None  # absolute maximum setpoint (default: 78°F / 26°C)
    cautious_offset: float | None = None  # setpoint offset above/below current (default: 2°F / 1.1°C)
    max_1h_change: float | None = None  # sanity check: max predicted 1h change (default: 5°F / 2.8°C)
    min_improvement: float | None = None  # min cost improvement to justify HVAC (default: 1°F / 0.6°C)
    cold_room_override: float | None = None  # force heating when this far below min (default: 1°F / 0.6°C)
    backup_margin: float = 3.0  # backup bounds = acceptable ± this margin (default: 3°F / 1.7°C)
    hard_rail_multiplier: float | None = None  # penalty multiplier outside acceptable bounds (default: 3.0)
    control_interval: int = 300  # seconds between control cycles (default: 5 min)
    sysid_interval: int = 3600  # seconds between automatic sysid runs (default: 1 hour, 0 = disabled)
    comfort_entity: str | None = None  # HA input_select controlling active comfort profile
    comfort_profiles: dict[str, ComfortProfile] = field(default_factory=dict)
    mrt_correction: MrtCorrectionConfig | None = None
    advisory: AdvisoryConfig = field(default_factory=lambda: AdvisoryConfig(cooldowns={}))
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # ── Temperature unit helpers ────────────────────────────────────

    @property
    def unit_symbol(self) -> str:
        """Display symbol: '°F' or '°C'."""
        return "°F" if self.location.unit == "F" else "°C"

    @property
    def delta_scale(self) -> float:
        """Multiplier for temperature deltas: 1.0 for °F, 5/9 for °C."""
        return 1.0 if self.location.unit == "F" else 5.0 / 9.0

    def abs_temp(self, fahrenheit: float) -> float:
        """Convert an absolute temperature from canonical °F to the configured unit."""
        if self.location.unit == "F":
            return fahrenheit
        return (fahrenheit - 32.0) * 5.0 / 9.0

    def delta_temp(self, fahrenheit_delta: float) -> float:
        """Convert a temperature delta from canonical °F to the configured unit."""
        return fahrenheit_delta * self.delta_scale

    # ── Derived properties (primary) ─────────────────────────────────

    @property
    def outdoor_sensor(self) -> str | None:
        """Column name of the configured outdoor temp sensor, if any."""
        for col, cfg in self.temp_sensors.items():
            if cfg.role == "outdoor":
                return col
        return None

    @property
    def prediction_sensors(self) -> list[str]:
        """Ordered list of constraint sensor column names for prediction."""
        return [c.sensor for c in self.constraints]

    @property
    def prediction_labels(self) -> list[str]:
        """Ordered list of constraint display labels."""
        return [c.label for c in self.constraints]

    @property
    def sensor_labels(self) -> dict[str, str]:
        """sensor column -> display label for all constraints."""
        return {c.sensor: c.label for c in self.constraints}

    @property
    def numeric_sensor_entities(self) -> dict[str, str]:
        """col_name -> entity_id for all temperature and humidity sensors."""
        result: dict[str, str] = {}
        for col, cfg in self.temp_sensors.items():
            result[col] = cfg.entity_id
        for col, cfg in self.humidity_sensors.items():
            result[col] = cfg.entity_id
        return result

    @property
    def all_temp_columns(self) -> list[str]:
        """All temperature column names (for lag/rolling features)."""
        return list(self.temp_sensors.keys())

    @property
    def climate_entities(self) -> dict[str, str]:
        """name -> entity_id for climate effectors (history extraction)."""
        return {name: cfg.entity_id for name, cfg in self.effectors.items() if cfg.domain == "climate"}

    @property
    def fan_entities(self) -> dict[str, str]:
        """name -> entity_id for fan effectors (history extraction)."""
        return {name: cfg.entity_id for name, cfg in self.effectors.items() if cfg.domain == "fan"}

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
    def environment_sensors(self) -> list[str]:
        """Entity IDs for all environment factor sensors."""
        return [cfg.entity_id for cfg in self.environment.values()]

    @property
    def environment_column_map(self) -> dict[str, str]:
        """entity_id -> column_name for environment sensors."""
        return {cfg.entity_id: cfg.column for cfg in self.environment.values()}

    @property
    def advisory_environment(self) -> dict[str, EnvironmentEntryConfig]:
        """Environment entries with advisory: true — included in advisory sweep."""
        return {k: v for k, v in self.environment.items() if v.advisory}

    @property
    def all_history_entities(self) -> list[str]:
        """All entity IDs for raw history extraction."""
        ids: list[str] = []
        ids.extend(self.numeric_sensor_entities.values())
        ids.extend(self.climate_entities.values())
        ids.extend(self.fan_entities.values())
        ids.extend(self.sensor_entities.values())
        ids.append(self.weather_entity)
        ids.extend(self.environment_sensors)
        return ids

    @property
    def device_health_checks(self) -> list[HealthCheck]:
        """All configured health checks, for generic safety monitoring."""
        return self.health_checks

    def environment_columns_for_sensor(self, sensor_name: str) -> list[str]:
        """Derive environment columns that might affect a sensor via naming convention.

        bedroom_temp → bedroom environment entry's column (if it exists).
        """
        label = sensor_name.removeprefix("thermostat_").removesuffix("_temp")
        if label in self.environment:
            return [self.environment[label].column]
        return []

    # ── Snapshot schema ──────────────────────────────────────────────

    @property
    def exclude_columns(self) -> set[str]:
        """Columns to exclude from features during training.

        Raw categorical strings that have encoded versions.
        """
        cols: set[str] = {"timestamp"}
        for name, cfg in self.effectors.items():
            if cfg.domain == "climate":
                cols.add(f"{name}_target")
                if cfg.mode_control == "manual":
                    cols.add(f"{name}_action")
                else:
                    cols.add(f"{name}_mode")
            else:
                cols.add(f"{name}_mode")
        for col in self.state_sensors:
            cols.add(col)
        cols.add("weather_condition")
        return cols

    @property
    def hvac_merge_columns(self) -> list[str]:
        """HVAC columns to merge from full-feature sources into baseline hourly data."""
        cols: list[str] = []
        for name, cfg in self.effectors.items():
            if cfg.domain == "climate":
                if cfg.mode_control == "manual":
                    cols.append(f"{name}_action")
                else:
                    cols.extend([f"{name}_temp", f"{name}_target", f"{name}_mode"])
            else:
                cols.append(f"{name}_mode")
        for col in self.state_sensors:
            cols.append(col)
        for col in self.power_sensors:
            cols.append(col)
        cols.extend(["weather_condition", "wind_speed", "outdoor_humidity", "met_outdoor_temp"])
        for cfg in self.environment.values():
            cols.append(cfg.column)
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
        for name, cfg in self.effectors.items():
            if cfg.domain == "climate":
                if cfg.mode_control == "manual":
                    cols.append(f"{name}_target")
                else:
                    cols.extend([f"{name}_temp", f"{name}_target"])
        return cols

    @property
    def environment_bool_columns(self) -> list[str]:
        """Environment columns that need bool normalization."""
        return [cfg.column for cfg in self.environment.values()]

    @property
    def thermostat_action_columns(self) -> list[str]:
        """Action columns for manual-mode climate effectors."""
        return [f"{name}_action" for name, cfg in self.effectors.items()
                if cfg.domain == "climate" and cfg.mode_control == "manual"]

    @property
    def mini_split_mode_columns(self) -> list[str]:
        """Mode columns for automatic-mode climate effectors."""
        return [f"{name}_mode" for name, cfg in self.effectors.items()
                if cfg.domain == "climate" and cfg.mode_control == "automatic"]

    @property
    def blower_mode_columns(self) -> list[str]:
        """Mode columns for fan effectors."""
        return [f"{name}_mode" for name, cfg in self.effectors.items() if cfg.domain == "fan"]

    @property
    def mini_split_delta_pairs(self) -> list[tuple[str, str, str]]:
        """(target_col, temp_col, delta_name) for regulating climate effectors."""
        pairs: list[tuple[str, str, str]] = []
        for name, cfg in self.effectors.items():
            if cfg.domain == "climate" and cfg.mode_control == "automatic":
                # Strip prefix for delta name (e.g., "mini_split_bedroom" -> "bedroom")
                label = name.split("_", 2)[-1] if "_" in name else name
                pairs.append((f"{name}_target", f"{name}_temp", f"{label}_target_delta"))
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
        Column layout depends on HA entity domain:
        - climate (mode_control=manual): _temp, _target, _action
        - climate (mode_control=automatic): _temp, _target, _mode, optionally _action
        - fan: _mode
        """
        defs: list[tuple[str, str]] = []
        for name, cfg in self.effectors.items():
            if cfg.domain == "climate":
                defs.extend([
                    (f"{name}_temp", "REAL"),
                    (f"{name}_target", "REAL"),
                ])
                if cfg.mode_control == "manual":
                    defs.append((f"{name}_action", "TEXT"))
                else:
                    defs.append((f"{name}_mode", "TEXT"))
                    if cfg.command_encoding and cfg.state_encoding != cfg.command_encoding:
                        defs.append((f"{name}_action", "TEXT"))
            else:
                defs.append((f"{name}_mode", "TEXT"))
        # State sensors (categorical — stored as TEXT, encoded when used)
        for col in self.state_sensors:
            defs.append((col, "TEXT"))
        # Power sensors (numeric)
        for col in self.power_sensors:
            defs.append((col, "REAL"))
        # Environment (weather entity always provides met_outdoor_temp)
        defs.append(("met_outdoor_temp", "REAL"))
        if self.outdoor_sensor:
            defs.append((self.outdoor_sensor, "REAL"))
        defs.extend([
            ("outdoor_humidity", "REAL"),
            ("wind_speed", "REAL"),
            ("weather_condition", "TEXT"),
        ])
        for col in self.humidity_sensors:
            defs.append((col, "REAL"))
        # Environment factors (windows, doors, shades, etc.)
        for cfg in self.environment.values():
            defs.append((cfg.column, "INTEGER"))
        # Per-room temps (aggregate and individual)
        for col in self.temp_sensors:
            if col != self.outdoor_sensor and not col.startswith("thermostat_"):
                defs.append((col, "REAL"))
        return defs


# ── YAML loading ──────────────────────────────────────────────────────────


def _find_yaml_path() -> Path:
    """Find weatherstat.yaml in the data directory (~/.weatherstat/)."""
    from weatherstat._data_dir import resolve_data_dir

    return resolve_data_dir() / "weatherstat.yaml"


def _parse_depends_on(raw: str | list | None) -> tuple[str, ...]:
    """Parse depends_on: accepts a string, list of strings, or None."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(x) for x in raw)


def _parse_config(data: dict) -> WeatherstatConfig:
    """Parse raw YAML dict into a WeatherstatConfig."""
    loc = data["location"]
    location = LocationConfig(
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        elevation=loc["elevation"],
        timezone=loc["timezone"],
        unit=str(loc.get("unit", "F")).upper(),
    )

    # Temperature sensors
    temp_sensors: dict[str, TempSensorConfig] = {}
    for col_name, sensor in data["sensors"]["temperature"].items():
        temp_sensors[col_name] = TempSensorConfig(
            column_name=col_name,
            entity_id=sensor["entity_id"],
            role=sensor.get("role", ""),
        )

    # Humidity sensors
    humidity_sensors: dict[str, HumiditySensorConfig] = {}
    for col_name, sensor in data["sensors"]["humidity"].items():
        humidity_sensors[col_name] = HumiditySensorConfig(
            column_name=col_name,
            entity_id=sensor["entity_id"],
        )

    # ── Effectors (flat dict, each declares its own properties) ─────
    effectors: dict[str, EffectorYamlConfig] = {}
    for name, dev in data.get("effectors", {}).items():
        se_raw = dev.get("state_encoding")
        state_enc = {str(k): float(v) for k, v in se_raw.items()} if se_raw else {}
        ce_raw = dev.get("command_encoding")
        cmd_enc = {str(k): float(v) for k, v in ce_raw.items()} if ce_raw else None
        mhw_raw = dev.get("mode_hold_window")
        mode_hold_window = (int(mhw_raw[0]), int(mhw_raw[1])) if mhw_raw else None
        ec_raw = dev.get("energy_cost", 0.0)
        energy_cost = {str(k): float(v) for k, v in ec_raw.items()} if isinstance(ec_raw, dict) else float(ec_raw)
        effectors[name] = EffectorYamlConfig(
            name=name,
            entity_id=dev["entity_id"],
            control_type=str(dev.get("control_type", "trajectory")),
            mode_control=str(dev.get("mode_control", "manual")),
            supported_modes=tuple(str(m) for m in dev.get("supported_modes", ())),
            state_encoding=state_enc,
            max_lag_minutes=int(dev.get("max_lag_minutes", 90)),
            energy_cost=energy_cost,
            command_encoding=cmd_enc,
            state_device=dev.get("state_device"),
            depends_on=_parse_depends_on(dev.get("depends_on")),
            proportional_band=float(dev.get("proportional_band", 1.0)),
            mode_hold_window=mode_hold_window,
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
    for hc_name, hc in data.get("health", {}).items():
        when = hc.get("when", {})
        health_checks.append(HealthCheck(
            name=hc_name,
            entity_id=hc["entity"],
            min_value=float(hc["min_value"]) if "min_value" in hc else None,
            max_value=float(hc["max_value"]) if "max_value" in hc else None,
            expected_state=str(hc["expected_state"]) if "expected_state" in hc else None,
            severity=hc.get("severity", "warning"),
            message=hc.get("message", ""),
            when_entity=when.get("entity"),
            when_state=str(when["state"]) if "state" in when else None,
            when_for_minutes=float(when.get("for_minutes", 0)),
        ))

    # ── Environment factors (windows, doors, shades, vents, etc.) ──
    environment: dict[str, EnvironmentEntryConfig] = {}
    for name, env_data in data.get("environment", {}).items():
        environment[name] = EnvironmentEntryConfig(
            name=name,
            entity_id=env_data["entity_id"],
            column=env_data["column"],
            kind=env_data.get("kind", "window"),
            default_state=env_data.get("default_state", "closed"),
            active_state=str(env_data["active_state"]),
            advisory=bool(env_data.get("advisory", False)),
        )

    # ── Constraints ──────────────────────────────────────────────────
    constraints_data = data.get("constraints", {})

    # Comfort profiles (home/away mode)
    comfort_entity: str | None = constraints_data.get("comfort_entity")
    comfort_profiles: dict[str, ComfortProfile] = {}
    for prof_name, prof_data in constraints_data.get("profiles", {}).items():
        prof_data = prof_data or {}  # "Home: {}" parses as None in YAML
        comfort_profiles[prof_name] = ComfortProfile(
            name=prof_name,
            preferred_offset=float(prof_data.get("preferred_offset", 0.0)),
            preferred_widen=float(prof_data.get("preferred_widen", 0.0)),
            min_offset=float(prof_data.get("min_offset", 0.0)),
            max_offset=float(prof_data.get("max_offset", 0.0)),
            penalty_scale=float(prof_data.get("penalty_scale", 1.0)),
        )

    # MRT correction (operative temperature adjustment for wall surface effects)
    mrt_data = constraints_data.get("mrt_correction")
    mrt_correction: MrtCorrectionConfig | None = MrtCorrectionConfig(
        alpha=float(mrt_data["alpha"]),
        reference_temp=float(mrt_data["reference_temp"]),
        max_offset=float(mrt_data.get("max_offset", 3.0)),
        solar_response=float(mrt_data.get("solar_response", 2.0)),
    ) if mrt_data else None

    # Backup margin: how much wider backup bounds are than acceptable (default 3°F / 1.7°C)
    backup_margin = float(data.get("defaults", {}).get("backup_margin", 3.0))

    constraint_list: list[ConstraintSchedule] = []
    for sched in constraints_data.get("schedules", []):
        sensor = sched["sensor"]
        label = sensor.removeprefix("thermostat_").removesuffix("_temp")
        entries: list[ComfortEntry] = []
        for entry in sched["schedule"]:
            hours = entry["hours"]
            # Three-tier bounds: acceptable (was min/max), backup (new)
            # Backward compat: "min"/"max" → acceptable, "acceptable" overrides
            acc_raw = entry.get("acceptable")
            if acc_raw is not None:
                if isinstance(acc_raw, list):
                    acc_lo, acc_hi = float(acc_raw[0]), float(acc_raw[1])
                else:
                    raise ValueError(f"'acceptable' must be [lo, hi], got: {acc_raw}")
            else:
                # Backward compat: read from min/max
                acc_lo = float(entry["min"])
                acc_hi = float(entry["max"])

            backup_raw = entry.get("backup")
            if backup_raw is not None:
                if isinstance(backup_raw, list):
                    bak_lo, bak_hi = float(backup_raw[0]), float(backup_raw[1])
                else:
                    raise ValueError(f"'backup' must be [lo, hi], got: {backup_raw}")
            else:
                # Default: acceptable ± backup_margin
                bak_lo = acc_lo - backup_margin
                bak_hi = acc_hi + backup_margin

            # preferred: float (point target) or [lo, hi] (dead band)
            pref_raw = entry.get("preferred", (acc_lo + acc_hi) / 2)
            if isinstance(pref_raw, list):
                pref_lo, pref_hi = float(pref_raw[0]), float(pref_raw[1])
            else:
                pref_lo = pref_hi = float(pref_raw)
            entries.append(ComfortEntry(
                start_hour=hours[0],
                end_hour=hours[1],
                preferred_lo=pref_lo,
                preferred_hi=pref_hi,
                acceptable_lo=acc_lo,
                acceptable_hi=acc_hi,
                backup_lo=bak_lo,
                backup_hi=bak_hi,
                cold_penalty=float(entry.get("cold_penalty", 2.0)),
                hot_penalty=float(entry.get("hot_penalty", 1.0)),
            ))
        constraint_list.append(ConstraintSchedule(
            sensor=sensor, label=label, entries=tuple(entries),
            mrt_weight=float(sched.get("mrt_weight", 1.0)),
        ))

    # ── Advisory config ──────────────────────────────────────────────
    adv_data = data.get("advisory", {})
    adv_quiet = adv_data.get("quiet_hours", [22, 7])
    advisory_config = AdvisoryConfig(
        cooldowns={str(k): int(v) for k, v in adv_data.get("cooldowns", {}).items()},
        quiet_hours=(int(adv_quiet[0]), int(adv_quiet[1])),
        opportunity_threshold=float(adv_data.get("opportunity_threshold", 0.3)),
        notification_threshold=float(adv_data.get("notification_threshold", 1.5)),
    )

    # ── Safety config (just cooldowns — device health is on effectors) ─
    safety_data = data.get("safety", {})
    safety_config = SafetyConfig(
        cooldowns={str(k): int(v) for k, v in safety_data.get("cooldowns", {}).items()},
    )

    # ── Defaults ─────────────────────────────────────────────────────
    defaults = data.get("defaults", {})
    default_tau = float(defaults.get("tau", 45.0))

    def _opt_float(key: str) -> float | None:
        v = defaults.get(key)
        return float(v) if v is not None else None

    def _opt_int(key: str, default: int) -> int:
        v = defaults.get(key)
        return int(v) if v is not None else default

    return WeatherstatConfig(
        location=location,
        temp_sensors=temp_sensors,
        humidity_sensors=humidity_sensors,
        effectors=effectors,
        state_sensors=state_sensors,
        power_sensors=power_sensors,
        health_checks=health_checks,
        environment=environment,
        weather_entity=data["weather"]["entity_id"],
        constraints=constraint_list,
        notification_target=data["notifications"]["target"],
        default_tau=default_tau,
        setpoint_min=_opt_float("setpoint_min"),
        setpoint_max=_opt_float("setpoint_max"),
        cautious_offset=_opt_float("cautious_offset"),
        max_1h_change=_opt_float("max_1h_change"),
        min_improvement=_opt_float("min_improvement"),
        cold_room_override=_opt_float("cold_room_override"),
        hard_rail_multiplier=_opt_float("hard_rail_multiplier"),
        backup_margin=backup_margin,
        control_interval=_opt_int("control_interval", 300),
        sysid_interval=_opt_int("sysid_interval", 3600),
        comfort_entity=comfort_entity,
        comfort_profiles=comfort_profiles,
        mrt_correction=mrt_correction,
        advisory=advisory_config,
        safety=safety_config,
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
