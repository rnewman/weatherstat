/**
 * YAML configuration loader for the TS client.
 *
 * Reads weatherstat.yaml (shared with Python) and derives entity IDs,
 * snapshot column definitions, SQL schema, and monitored entity lists.
 * Adding a sensor: add to weatherstat.yaml + update SnapshotRow in types.ts.
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

import type { HAEntityState } from "./types.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const YAML_PATH = resolve(__dirname, "../../weatherstat.yaml");

// ---- Raw YAML types (match the YAML file structure) ----

interface RawConfig {
  sensors: {
    temperature: Record<string, { entity_id: string; statistics: boolean }>;
    humidity: Record<string, { entity_id: string; statistics: boolean }>;
  };
  devices: {
    thermostats: Record<string, { entity_id: string; zone: string }>;
    mini_splits: Record<
      string,
      { entity_id: string; sweep_modes: string[]; mode_encoding: Record<string, number> }
    >;
    blowers: Record<
      string,
      { entity_id: string; zone: string; levels: string[]; level_encoding: Record<string, number> }
    >;
    boiler: Record<
      string,
      { mode_entity: string; capacity_entity: string; mode_encoding: Record<string, number> }
    >;
  };
  extra_temp_sensors?: Record<string, { entity_id: string }>;
  windows: Record<string, { entity_id: string }>;
  weather: { entity_id: string };
  notifications: { target: string };
}

// ---- Column definition types ----

/** Extract a value from the HA entity state map. */
export type ExtractFn = (stateMap: Map<string, HAEntityState>) => string | number | boolean;

/** One column in the snapshots table, with its extraction logic. */
export interface ColumnDef {
  snake: string;
  camel: string;
  sqlType: "REAL" | "TEXT" | "INTEGER";
  extract: ExtractFn;
}

// ---- Helpers ----

function snakeToCamel(s: string): string {
  return s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}

// ---- Value extractors ----

function attrNum(entityId: string, attr: string, fallback: number): ExtractFn {
  return (stateMap) => {
    const val = stateMap.get(entityId)?.attributes[attr];
    return typeof val === "number" ? val : fallback;
  };
}

function attrStr(entityId: string, attr: string, fallback: string): ExtractFn {
  return (stateMap) => {
    const val = stateMap.get(entityId)?.attributes[attr];
    return typeof val === "string" ? val : fallback;
  };
}

function sensorNum(entityId: string, fallback: number): ExtractFn {
  return (stateMap) => {
    const s = stateMap.get(entityId);
    if (!s) return fallback;
    const val = parseFloat(s.state);
    return Number.isFinite(val) ? val : fallback;
  };
}

function entityState(entityId: string, fallback: string): ExtractFn {
  return (stateMap) => stateMap.get(entityId)?.state ?? fallback;
}

function binarySensorState(entityId: string): ExtractFn {
  return (stateMap) => stateMap.get(entityId)?.state === "on";
}

function blowerModeExtract(entityId: string): ExtractFn {
  return (stateMap) => {
    const s = stateMap.get(entityId);
    if (s?.state === "off" || !s?.state) return "off";
    const preset = s.attributes["preset_mode"];
    if (typeof preset === "string") return preset;
    return "low"; // fan is on but no preset — default to low
  };
}

// ---- Config loader ----

function loadYamlConfig() {
  const raw = parse(readFileSync(YAML_PATH, "utf-8")) as RawConfig;

  // ── Build column definitions in canonical snapshot order ──────────────

  const columnDefs: ColumnDef[] = [];

  // 1. Thermostats (3 cols each: temp, target, action)
  for (const [name, therm] of Object.entries(raw.devices.thermostats)) {
    const prefix = `thermostat_${name}`;
    columnDefs.push({
      snake: `${prefix}_temp`,
      camel: snakeToCamel(`${prefix}_temp`),
      sqlType: "REAL",
      extract: attrNum(therm.entity_id, "current_temperature", 0),
    });
    columnDefs.push({
      snake: `${prefix}_target`,
      camel: snakeToCamel(`${prefix}_target`),
      sqlType: "REAL",
      extract: attrNum(therm.entity_id, "temperature", 0),
    });
    columnDefs.push({
      snake: `${prefix}_action`,
      camel: snakeToCamel(`${prefix}_action`),
      sqlType: "TEXT",
      extract: attrStr(therm.entity_id, "hvac_action", "idle"),
    });
  }

  // 2. Mini splits (3 cols each: temp, target, mode)
  for (const [name, split] of Object.entries(raw.devices.mini_splits)) {
    const prefix = `mini_split_${name}`;
    columnDefs.push({
      snake: `${prefix}_temp`,
      camel: snakeToCamel(`${prefix}_temp`),
      sqlType: "REAL",
      extract: attrNum(split.entity_id, "current_temperature", 0),
    });
    columnDefs.push({
      snake: `${prefix}_target`,
      camel: snakeToCamel(`${prefix}_target`),
      sqlType: "REAL",
      extract: attrNum(split.entity_id, "temperature", 0),
    });
    columnDefs.push({
      snake: `${prefix}_mode`,
      camel: snakeToCamel(`${prefix}_mode`),
      sqlType: "TEXT",
      extract: entityState(split.entity_id, "off"),
    });
  }

  // 3. Blowers (1 col each: mode)
  for (const [name, blw] of Object.entries(raw.devices.blowers)) {
    const snake = `blower_${name}_mode`;
    columnDefs.push({
      snake,
      camel: snakeToCamel(snake),
      sqlType: "TEXT",
      extract: blowerModeExtract(blw.entity_id),
    });
  }

  // 4. Boiler (2 cols each: heating_mode, heat_capacity)
  for (const [name, boiler] of Object.entries(raw.devices.boiler)) {
    columnDefs.push({
      snake: `${name}_heating_mode`,
      camel: snakeToCamel(`${name}_heating_mode`),
      sqlType: "TEXT",
      extract: entityState(boiler.mode_entity, "Idle"),
    });
    columnDefs.push({
      snake: `${name}_heat_capacity`,
      camel: snakeToCamel(`${name}_heat_capacity`),
      sqlType: "REAL",
      extract: sensorNum(boiler.capacity_entity, 0),
    });
  }

  // 5. Outdoor temp
  const outdoorSensor = raw.sensors.temperature["outdoor_temp"];
  if (outdoorSensor) {
    columnDefs.push({
      snake: "outdoor_temp",
      camel: "outdoorTemp",
      sqlType: "REAL",
      extract: sensorNum(outdoorSensor.entity_id, 0),
    });
  }

  // 6. Weather-derived columns
  const weatherId = raw.weather.entity_id;
  columnDefs.push({
    snake: "outdoor_humidity",
    camel: "outdoorHumidity",
    sqlType: "REAL",
    extract: (stateMap) => {
      const w = stateMap.get(weatherId);
      return w ? ((w.attributes["humidity"] as number) ?? 0) : 0;
    },
  });
  columnDefs.push({
    snake: "wind_speed",
    camel: "windSpeed",
    sqlType: "REAL",
    extract: (stateMap) => {
      const w = stateMap.get(weatherId);
      return w ? ((w.attributes["wind_speed"] as number) ?? 0) : 0;
    },
  });
  columnDefs.push({
    snake: "weather_condition",
    camel: "weatherCondition",
    sqlType: "TEXT",
    extract: entityState(weatherId, "unknown"),
  });

  // 7. Humidity sensors
  for (const [col, sensor] of Object.entries(raw.sensors.humidity)) {
    columnDefs.push({
      snake: col,
      camel: snakeToCamel(col),
      sqlType: "REAL",
      extract: sensorNum(sensor.entity_id, 0),
    });
  }

  // 8. Windows (1 col each + any_window_open)
  const windowExtractors: ExtractFn[] = [];
  for (const [name, win] of Object.entries(raw.windows)) {
    const snake = `window_${name}_open`;
    const extractFn = binarySensorState(win.entity_id);
    windowExtractors.push(extractFn);
    columnDefs.push({
      snake,
      camel: snakeToCamel(snake),
      sqlType: "INTEGER",
      extract: extractFn,
    });
  }
  columnDefs.push({
    snake: "any_window_open",
    camel: "anyWindowOpen",
    sqlType: "INTEGER",
    extract: (stateMap) => windowExtractors.some((fn) => fn(stateMap) === true),
  });

  // 9. Per-room temperature sensors (exclude outdoor_temp and thermostat_* handled above)
  for (const [col, sensor] of Object.entries(raw.sensors.temperature)) {
    if (col === "outdoor_temp" || col.startsWith("thermostat_")) continue;
    columnDefs.push({
      snake: col,
      camel: snakeToCamel(col),
      sqlType: "REAL",
      extract: sensorNum(sensor.entity_id, 0),
    });
  }

  // ── Derived arrays ───────────────────────────────────────────────────

  const snapshotColumns = columnDefs.map((d) => d.snake);

  const camelToSnake: Record<string, string> = {};
  for (const d of columnDefs) {
    camelToSnake[d.camel] = d.snake;
  }

  const createTableSql = [
    "CREATE TABLE IF NOT EXISTS snapshots (",
    "  timestamp TEXT PRIMARY KEY,",
    ...columnDefs.map(
      (d, i) => `  ${d.snake} ${d.sqlType}${i < columnDefs.length - 1 ? "," : ""}`,
    ),
    ")",
  ].join("\n");

  // All monitored entities (unique set)
  const allMonitoredEntities = Array.from(
    new Set([
      ...Object.values(raw.devices.thermostats).map((t) => t.entity_id),
      ...Object.values(raw.devices.mini_splits).map((s) => s.entity_id),
      ...Object.values(raw.devices.blowers).map((b) => b.entity_id),
      ...Object.values(raw.devices.boiler).flatMap((b) => [b.mode_entity, b.capacity_entity]),
      ...Object.values(raw.sensors.temperature).map((s) => s.entity_id),
      ...Object.values(raw.sensors.humidity).map((s) => s.entity_id),
      ...Object.values(raw.windows).map((w) => w.entity_id),
      raw.weather.entity_id,
      ...Object.values(raw.extra_temp_sensors ?? {}).map((s) => s.entity_id),
    ]),
  );

  // Window sensor entity IDs (for backward-compatible WINDOW_SENSORS export)
  const windowSensorEntityIds = Object.values(raw.windows).map((w) => w.entity_id);

  // TEMP_SENSORS map: short name → entity_id (for backward compatibility)
  // Includes non-thermostat, non-outdoor sensors from sensors.temperature + extra_temp_sensors.
  const tempSensors: Record<string, string> = {};
  for (const [col, sensor] of Object.entries(raw.sensors.temperature)) {
    if (col.startsWith("thermostat_") || col === "outdoor_temp") continue;
    const key = col.replace(/_temp$/, "");
    tempSensors[key] = sensor.entity_id;
  }
  for (const [name, sensor] of Object.entries(raw.extra_temp_sensors ?? {})) {
    tempSensors[name] = sensor.entity_id;
  }

  return {
    // Raw device configs
    thermostats: Object.fromEntries(
      Object.entries(raw.devices.thermostats).map(([name, t]) => [
        name,
        { entityId: t.entity_id, zone: t.zone },
      ]),
    ),
    miniSplits: Object.fromEntries(
      Object.entries(raw.devices.mini_splits).map(([name, s]) => [
        name,
        { entityId: s.entity_id, sweepModes: s.sweep_modes, modeEncoding: s.mode_encoding },
      ]),
    ),
    blowers: Object.fromEntries(
      Object.entries(raw.devices.blowers).map(([name, b]) => [
        name,
        { entityId: b.entity_id, zone: b.zone, levels: b.levels, levelEncoding: b.level_encoding },
      ]),
    ),
    boiler: Object.fromEntries(
      Object.entries(raw.devices.boiler).map(([name, b]) => [
        name,
        { modeEntity: b.mode_entity, capacityEntity: b.capacity_entity },
      ]),
    ),

    // Sensor entity IDs
    outdoorTempEntity: raw.sensors.temperature["outdoor_temp"]?.entity_id ?? "",
    indoorHumidityEntity: raw.sensors.humidity["indoor_humidity"]?.entity_id ?? "",
    weatherEntity: raw.weather.entity_id,
    notificationTarget: raw.notifications.target,

    // Backward-compatible maps
    tempSensors,
    windowSensorEntityIds,

    // Derived schema
    columnDefs,
    snapshotColumns,
    camelToSnake,
    createTableSql,
    allMonitoredEntities,
  } as const;
}

/** Singleton config loaded from weatherstat.yaml. */
export const config = loadYamlConfig();
