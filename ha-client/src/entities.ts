/**
 * Home Assistant entity ID constants.
 *
 * See docs/entities.md for the full entity reference with zone associations.
 */

// Z-Wave thermostats (control Navien hydronic floor heat zones)
export const THERMOSTAT_UPSTAIRS = "climate.upstairs_thermostat";
export const THERMOSTAT_DOWNSTAIRS = "climate.downstairs_thermostat";

// Mini splits (ESPHome via M5NanoC6 IR blasters)
export const MINI_SPLIT_BEDROOM = "climate.m5nanoc6_bed_split_bedroom_split";
export const MINI_SPLIT_LIVING_ROOM = "climate.m5nanoc6_lr_split_living_room_split";

// In-wall blowers (Z-Wave fan entities with off/low/high presets)
export const BLOWER_FAMILY_ROOM = "fan.blower_1";
export const BLOWER_OFFICE = "fan.blower_office";

// Temperature sensors (per-room)
export const TEMP_SENSORS = {
  upstairs_aggregate: "sensor.upstairs_temperatures",
  downstairs_aggregate: "sensor.downstairs_temperatures",
  family_room: "sensor.climate_family_room_air_temperature",
  office: "sensor.climate_office_air_temperature",
  kitchen: "sensor.kitchen_air_temperature",
  bedroom: "sensor.climate_bedroom_air_temperature",
  piano: "sensor.climate_piano_air_temperature",
  bathroom: "sensor.climate_bathroom_air_temperature",
  bedroom_aggregate: "sensor.bedroom_aggregate_temperature",
  living_room: "sensor.living_room_aggregate_temperature",
} as const;

// Outdoor temperature
export const OUTDOOR_TEMP = "sensor.climate_side_air_temperature";

// Indoor humidity
export const INDOOR_HUMIDITY = "sensor.upstairs_thermostat_humidity";

// Window/door sensors (for any_window_open)
export const WINDOW_SENSORS = [
  "binary_sensor.window_basement_intrusion",
  "binary_sensor.window_family_room_intrusion",
  "binary_sensor.window_balcony_intrusion",
  "binary_sensor.window_bedroom_intrusion",
  "binary_sensor.window_office_window_door_is_open",
  "binary_sensor.kitchen_window_sensor_intrusion",
];

// Weather entity (met.no integration)
export const WEATHER_ENTITY = "weather.forecast_home";

// Navien tankless water heater
export const NAVIEN_HEATING_MODE = "sensor.navien_navien_heating_mode";
export const NAVIEN_HEAT_CAPACITY = "sensor.navien_navien_heat_capacity";

/** All entity IDs we subscribe to for snapshots. */
export const ALL_MONITORED_ENTITIES: string[] = [
  THERMOSTAT_UPSTAIRS,
  THERMOSTAT_DOWNSTAIRS,
  MINI_SPLIT_BEDROOM,
  MINI_SPLIT_LIVING_ROOM,
  BLOWER_FAMILY_ROOM,
  BLOWER_OFFICE,
  ...Object.values(TEMP_SENSORS),
  OUTDOOR_TEMP,
  INDOOR_HUMIDITY,
  ...WINDOW_SENSORS,
  WEATHER_ENTITY,
  NAVIEN_HEATING_MODE,
  NAVIEN_HEAT_CAPACITY,
];
