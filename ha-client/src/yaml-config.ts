/**
 * YAML configuration loader for the TS client.
 *
 * Reads weatherstat.yaml (~/.weatherstat/) and derives entity IDs,
 * snapshot column definitions, SQL schema, and monitored entity lists.
 * Adding a sensor: just add to weatherstat.yaml — SnapshotRow is dynamic.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "yaml";

import type { HAEntityState } from "./types.ts";
import { dataDir } from "./data-dir.ts";

const YAML_PATH = resolve(dataDir, "weatherstat.yaml");

// ---- Raw YAML types (match the YAML file structure) ----

interface RawConfig {
  sensors: {
    temperature: Record<string, { entity_id: string; statistics?: boolean; role?: string }>;
    humidity: Record<string, { entity_id: string; statistics?: boolean }>;
    state?: Record<string, { entity_id: string; encoding: Record<string, number> }>;
    power?: Record<string, { entity_id: string }>;
  };
  effectors: Record<
    string,
    {
      entity_id: string;
      control_type: string;
      mode_control: string;
      supported_modes?: string[];
      state_encoding?: Record<string, number>;
      command_encoding?: Record<string, number>;
      state_device?: string;
      depends_on?: string | string[];
      proportional_band?: number;
      mode_hold_window?: [number, number];
      max_lag_minutes?: number;
      energy_cost?: number | Record<string, number>;
    }
  >;
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

  // 1. Effectors — column layout depends on HA entity domain
  for (const [name, eff] of Object.entries(raw.effectors)) {
    const domain = eff.entity_id.split(".")[0];
    if (domain === "climate") {
      // Climate entities: temp, target, and action/mode
      columnDefs.push({
        snake: `${name}_temp`,
        camel: snakeToCamel(`${name}_temp`),
        sqlType: "REAL",
        extract: attrNum(eff.entity_id, "current_temperature", 0),
      });
      columnDefs.push({
        snake: `${name}_target`,
        camel: snakeToCamel(`${name}_target`),
        sqlType: "REAL",
        extract: attrNum(eff.entity_id, "temperature", 0),
      });
      if (eff.mode_control === "manual") {
        // Manual mode (thermostat): only action (hvac_action attribute)
        columnDefs.push({
          snake: `${name}_action`,
          camel: snakeToCamel(`${name}_action`),
          sqlType: "TEXT",
          extract: attrStr(eff.entity_id, "hvac_action", "idle"),
        });
      } else {
        // Automatic mode (mini-split): mode (entity state) + optional action
        columnDefs.push({
          snake: `${name}_mode`,
          camel: snakeToCamel(`${name}_mode`),
          sqlType: "TEXT",
          extract: entityState(eff.entity_id, "off"),
        });
        if (eff.state_encoding && eff.command_encoding) {
          columnDefs.push({
            snake: `${name}_action`,
            camel: snakeToCamel(`${name}_action`),
            sqlType: "TEXT",
            extract: attrStr(eff.entity_id, "hvac_action", "off"),
          });
        }
      }
    } else {
      // Fan entities: mode from state/preset_mode
      columnDefs.push({
        snake: `${name}_mode`,
        camel: snakeToCamel(`${name}_mode`),
        sqlType: "TEXT",
        extract: blowerModeExtract(eff.entity_id),
      });
    }
  }

  // 4. State sensors (categorical — stored as TEXT)
  for (const [col, sensor] of Object.entries(raw.sensors.state ?? {})) {
    columnDefs.push({
      snake: col,
      camel: snakeToCamel(col),
      sqlType: "TEXT",
      extract: entityState(sensor.entity_id, ""),
    });
  }

  // 4b. Power sensors (numeric)
  for (const [col, sensor] of Object.entries(raw.sensors.power ?? {})) {
    columnDefs.push({
      snake: col,
      camel: snakeToCamel(col),
      sqlType: "REAL",
      extract: sensorNum(sensor.entity_id, 0),
    });
  }

  // 5. Outdoor temp sensor (optional — identified by role: "outdoor")
  const outdoorEntry = Object.entries(raw.sensors.temperature).find(
    ([, s]) => s.role === "outdoor",
  );
  const outdoorCol = outdoorEntry?.[0];
  if (outdoorEntry) {
    columnDefs.push({
      snake: outdoorEntry[0],
      camel: snakeToCamel(outdoorEntry[0]),
      sqlType: "REAL",
      extract: sensorNum(outdoorEntry[1].entity_id, 0),
    });
  }

  // 6. Weather-derived columns
  const weatherId = raw.weather.entity_id;
  columnDefs.push({
    snake: "met_outdoor_temp",
    camel: "metOutdoorTemp",
    sqlType: "REAL",
    extract: (stateMap) => {
      const w = stateMap.get(weatherId);
      return w ? ((w.attributes["temperature"] as number) ?? 0) : 0;
    },
  });
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

  // 6b. Forecast columns (populated by collector from service call, not entity state)
  // Hourly outdoor temperature forecasts (1h through 12h)
  for (let h = 1; h <= 12; h++) {
    columnDefs.push({
      snake: `forecast_temp_${h}h`,
      camel: `forecastTemp${h}h`,
      sqlType: "REAL",
      extract: () => 0,  // placeholder — collector injects real values
    });
  }
  // Condition and wind at key horizons (1h, 2h, 4h, 6h, 12h)
  for (const h of [1, 2, 4, 6, 12]) {
    columnDefs.push({
      snake: `forecast_condition_${h}h`,
      camel: `forecastCondition${h}h`,
      sqlType: "TEXT",
      extract: () => "",  // placeholder — collector injects real values
    });
    columnDefs.push({
      snake: `forecast_wind_${h}h`,
      camel: `forecastWind${h}h`,
      sqlType: "REAL",
      extract: () => 0,  // placeholder — collector injects real values
    });
  }

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

  // 9. Per-room temperature sensors (exclude outdoor and thermostat_* handled above)
  for (const [col, sensor] of Object.entries(raw.sensors.temperature)) {
    if (col === outdoorCol || col.startsWith("thermostat_")) continue;
    columnDefs.push({
      snake: col,
      camel: snakeToCamel(col),
      sqlType: "REAL",
      extract: sensorNum(sensor.entity_id, 0),
    });
  }

  // ── Derived arrays ───────────────────────────────────────────────────

  const camelToSnake: Record<string, string> = {};
  for (const d of columnDefs) {
    camelToSnake[d.camel] = d.snake;
  }

  const createReadingsTableSql = [
    "CREATE TABLE IF NOT EXISTS readings (",
    "  timestamp TEXT NOT NULL,",
    "  name      TEXT NOT NULL,",
    "  value     TEXT NOT NULL,",
    "  PRIMARY KEY (timestamp, name)",
    ")",
  ].join("\n");

  // All monitored entities (unique set)
  const allMonitoredEntities = Array.from(
    new Set([
      ...Object.values(raw.effectors).map((e) => e.entity_id),
      ...Object.values(raw.sensors.temperature).map((s) => s.entity_id),
      ...Object.values(raw.sensors.humidity).map((s) => s.entity_id),
      ...Object.values(raw.sensors.state ?? {}).map((s) => s.entity_id),
      ...Object.values(raw.sensors.power ?? {}).map((s) => s.entity_id),
      ...Object.values(raw.windows).map((w) => w.entity_id),
      raw.weather.entity_id,
    ]),
  );

  // Window sensor entity IDs
  const windowSensorEntityIds = Object.values(raw.windows).map((w) => w.entity_id);

  // TEMP_SENSORS map: short name → entity_id
  // Includes non-thermostat, non-outdoor sensors from sensors.temperature.
  const tempSensors: Record<string, string> = {};
  for (const [col, sensor] of Object.entries(raw.sensors.temperature)) {
    if (col.startsWith("thermostat_") || col === outdoorCol) continue;
    const key = col.replace(/_temp$/, "");
    tempSensors[key] = sensor.entity_id;
  }

  // Effector configs for executor — flat dict keyed by effector name
  const effectors = Object.fromEntries(
    Object.entries(raw.effectors).map(([name, e]) => [
      name,
      {
        entityId: e.entity_id,
        domain: e.entity_id.split(".")[0] as "climate" | "fan",
        controlType: e.control_type,
        modeControl: e.mode_control,
      },
    ]),
  );

  return {
    effectors,
    // Sensor entity IDs
    weatherEntity: raw.weather.entity_id,
    notificationTarget: raw.notifications.target,

    // Backward-compatible maps
    tempSensors,
    windowSensorEntityIds,

    // Derived schema
    columnDefs,
    camelToSnake,
    createReadingsTableSql,
    allMonitoredEntities,
  } as const;
}

/** Singleton config loaded from weatherstat.yaml. */
export const config = loadYamlConfig();
