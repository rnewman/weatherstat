/**
 * Home Assistant entity ID constants.
 *
 * Thin re-exports from the YAML config (weatherstat.yaml).
 * See docs/entities.md for the full entity reference with zone associations.
 */

import { config } from "./yaml-config.ts";

// Z-Wave thermostats (control Navien hydronic floor heat zones)
export const THERMOSTAT_UPSTAIRS = config.thermostats["upstairs"]!.entityId;
export const THERMOSTAT_DOWNSTAIRS = config.thermostats["downstairs"]!.entityId;

// Mini splits (ESPHome via M5NanoC6 IR blasters)
export const MINI_SPLIT_BEDROOM = config.miniSplits["bedroom"]!.entityId;
export const MINI_SPLIT_LIVING_ROOM = config.miniSplits["living_room"]!.entityId;

// In-wall blowers (Z-Wave fan entities with off/low/high presets)
export const BLOWER_FAMILY_ROOM = config.blowers["family_room"]!.entityId;
export const BLOWER_OFFICE = config.blowers["office"]!.entityId;

// Temperature sensors (per-room)
export const TEMP_SENSORS = config.tempSensors;

// Outdoor temperature
export const OUTDOOR_TEMP = config.outdoorTempEntity;

// Indoor humidity
export const INDOOR_HUMIDITY = config.indoorHumidityEntity;

// Window/door sensors (for any_window_open)
export const WINDOW_SENSORS = config.windowSensorEntityIds;

// Weather entity (met.no integration)
export const WEATHER_ENTITY = config.weatherEntity;

// Navien tankless water heater
export const NAVIEN_HEATING_MODE = config.boiler["navien"]!.modeEntity;
export const NAVIEN_HEAT_CAPACITY = config.boiler["navien"]!.capacityEntity;

/** All entity IDs we subscribe to for snapshots. */
export const ALL_MONITORED_ENTITIES = config.allMonitoredEntities;
