"""Domain types mirroring the TypeScript side, using frozen dataclasses and StrEnum."""

from dataclasses import dataclass
from enum import StrEnum


class HVACMode(StrEnum):
    OFF = "off"
    HEAT = "heat"
    COOL = "cool"
    HEAT_COOL = "heat_cool"
    AUTO = "auto"
    FAN_ONLY = "fan_only"
    DRY = "dry"


class HVACAction(StrEnum):
    HEATING = "heating"
    COOLING = "cooling"
    IDLE = "idle"
    OFF = "off"
    DRYING = "drying"
    FAN = "fan"


class BlowerMode(StrEnum):
    OFF = "off"
    LOW = "low"
    HIGH = "high"


class NavienHeatingMode(StrEnum):
    SPACE_HEATING = "Space Heating"
    IDLE = "Idle"


class WeatherCondition(StrEnum):
    """All met.no weather conditions that HA can report."""

    CLEAR_NIGHT = "clear-night"
    CLOUDY = "cloudy"
    EXCEPTIONAL = "exceptional"
    FOG = "fog"
    HAIL = "hail"
    LIGHTNING = "lightning"
    LIGHTNING_RAINY = "lightning-rainy"
    PARTLY_CLOUDY = "partlycloudy"
    POURING = "pouring"
    RAINY = "rainy"
    SNOWY = "snowy"
    SNOWY_RAINY = "snowy-rainy"
    SUNNY = "sunny"
    WINDY = "windy"
    WINDY_VARIANT = "windy-variant"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ThermostatState:
    entity_id: str
    current_temp: float
    target_temp: float
    hvac_mode: HVACMode
    hvac_action: HVACAction


@dataclass(frozen=True)
class MiniSplitState:
    entity_id: str
    current_temp: float
    target_temp: float
    hvac_mode: HVACMode
    fan_mode: str


@dataclass(frozen=True)
class WeatherState:
    temperature: float
    humidity: float
    wind_speed: float
    wind_bearing: float
    condition: WeatherCondition


@dataclass(frozen=True)
class SnapshotRow:
    """Matches the Parquet schema written by the HA client."""

    timestamp: str
    # Thermostat zones
    thermostat_upstairs_temp: float
    thermostat_upstairs_target: float
    thermostat_upstairs_action: str
    thermostat_downstairs_temp: float
    thermostat_downstairs_target: float
    thermostat_downstairs_action: str
    # Mini splits (named by location)
    mini_split_bedroom_temp: float
    mini_split_bedroom_target: float
    mini_split_bedroom_mode: str
    mini_split_living_room_temp: float
    mini_split_living_room_target: float
    mini_split_living_room_mode: str
    # Blowers (mode captures speed level)
    blower_family_room_mode: str
    blower_office_mode: str
    # Navien
    navien_heating_mode: str
    navien_heat_capacity: float
    # Environment
    outdoor_temp: float
    outdoor_humidity: float
    wind_speed: float
    weather_condition: str
    indoor_humidity: float
    any_window_open: bool
    # Per-room temperatures
    upstairs_aggregate_temp: float
    downstairs_aggregate_temp: float
    family_room_temp: float
    office_temp: float
    bedroom_temp: float
    kitchen_temp: float
    living_room_temp: float


@dataclass(frozen=True)
class Prediction:
    """Output of the ML inference pipeline, written as JSON."""

    timestamp: str
    thermostat_upstairs_target: float
    thermostat_downstairs_target: float
    mini_split_bedroom_target: float
    mini_split_bedroom_mode: HVACMode
    mini_split_living_room_target: float
    mini_split_living_room_mode: HVACMode
    blower_family_room_mode: BlowerMode
    blower_office_mode: BlowerMode
    confidence: float
