/**
 * Home Assistant entity ID constants.
 *
 * These are placeholders — replace with real entity IDs from your HA instance.
 * Run `just collect` with logging to discover available entities.
 */

// Z-Wave thermostats
export const THERMOSTAT_UPSTAIRS = "climate.thermostat_upstairs";
export const THERMOSTAT_DOWNSTAIRS = "climate.thermostat_downstairs";

// Mini splits (ESPHome)
export const MINI_SPLIT_1 = "climate.mini_split_1";
export const MINI_SPLIT_2 = "climate.mini_split_2";

// Hydronic under-floor heat (upstairs)
export const FLOOR_HEAT = "switch.floor_heat";

// In-wall blowers (ESPHome-controllable ones)
export const BLOWER_1 = "switch.blower_1";
export const BLOWER_2 = "switch.blower_2";

// Temperature sensors
export const TEMP_SENSORS: Record<string, string> = {
  living_room: "sensor.living_room_temperature",
  bedroom: "sensor.bedroom_temperature",
  office: "sensor.office_temperature",
  kitchen: "sensor.kitchen_temperature",
  upstairs_hall: "sensor.upstairs_hall_temperature",
  basement: "sensor.basement_temperature",
};

// Window/door sensors
export const WINDOW_SENSORS = [
  "binary_sensor.window_1",
  "binary_sensor.window_2",
  "binary_sensor.window_3",
];

// Weather entity (HA built-in or integration)
export const WEATHER_ENTITY = "weather.home";

// Navien heater status
export const NAVIEN_HEATER = "binary_sensor.navien_heater";

/** All entity IDs we subscribe to for snapshots. */
export const ALL_MONITORED_ENTITIES: string[] = [
  THERMOSTAT_UPSTAIRS,
  THERMOSTAT_DOWNSTAIRS,
  MINI_SPLIT_1,
  MINI_SPLIT_2,
  FLOOR_HEAT,
  BLOWER_1,
  BLOWER_2,
  ...Object.values(TEMP_SENSORS),
  ...WINDOW_SENSORS,
  WEATHER_ENTITY,
  NAVIEN_HEATER,
];
