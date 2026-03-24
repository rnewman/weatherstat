/**
 * Home Assistant entity ID constants.
 *
 * Re-exports from the YAML config (weatherstat.yaml).
 * Device-specific entity IDs (thermostats, mini-splits, blowers) are accessed
 * directly from config in the executor. Only shared/non-device constants here.
 */

import { config } from "./yaml-config.ts";

// Temperature sensors (per-room)
export const TEMP_SENSORS = config.tempSensors;

// Window/door sensors (for any_window_open)
export const WINDOW_SENSORS = config.windowSensorEntityIds;

// Weather entity (met.no integration)
export const WEATHER_ENTITY = config.weatherEntity;

/** All entity IDs we subscribe to for snapshots. */
export const ALL_MONITORED_ENTITIES = config.allMonitoredEntities;
