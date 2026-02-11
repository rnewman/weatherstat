"""Domain types mirroring the TypeScript side, using frozen dataclasses and StrEnum."""

from dataclasses import dataclass
from enum import StrEnum


class HVACMode(StrEnum):
    OFF = "off"
    HEAT = "heat"
    COOL = "cool"
    HEAT_COOL = "heat_cool"
    AUTO = "auto"


class HVACAction(StrEnum):
    HEATING = "heating"
    COOLING = "cooling"
    IDLE = "idle"
    OFF = "off"


class WeatherCondition(StrEnum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    PARTLY_CLOUDY = "partlycloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    FOG = "fog"
    CLEAR_NIGHT = "clear-night"
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
    thermostat_upstairs_temp: float
    thermostat_upstairs_target: float
    thermostat_upstairs_action: str
    thermostat_downstairs_temp: float
    thermostat_downstairs_target: float
    thermostat_downstairs_action: str
    mini_split_1_temp: float
    mini_split_1_target: float
    mini_split_1_mode: str
    mini_split_2_temp: float
    mini_split_2_target: float
    mini_split_2_mode: str
    floor_heat_on: bool
    blower_1_on: bool
    blower_2_on: bool
    outdoor_temp: float
    outdoor_humidity: float
    wind_speed: float
    weather_condition: str
    navien_heater_active: bool
    any_window_open: bool
    indoor_temps_json: str


@dataclass(frozen=True)
class Prediction:
    """Output of the ML inference pipeline, written as JSON."""

    timestamp: str
    thermostat_upstairs_target: float
    thermostat_downstairs_target: float
    mini_split_1_target: float
    mini_split_1_mode: HVACMode
    mini_split_2_target: float
    mini_split_2_mode: HVACMode
    floor_heat_on: bool
    blower_1_on: bool
    blower_2_on: bool
    confidence: float
