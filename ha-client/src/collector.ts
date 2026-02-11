/**
 * State snapshot collector.
 *
 * Periodically reads entity states via HAClient and writes snapshots
 * to a SQLite database. Depends on the HAClient interface, not WebSocket directly.
 */

import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import Database from "better-sqlite3";
import type { Database as DatabaseType } from "better-sqlite3";

import type { HAClient, SnapshotRow } from "./types.ts";
import type { Config } from "./config.ts";
import {
  THERMOSTAT_UPSTAIRS,
  THERMOSTAT_DOWNSTAIRS,
  MINI_SPLIT_BEDROOM,
  MINI_SPLIT_LIVING_ROOM,
  BLOWER_FAMILY_ROOM,
  BLOWER_OFFICE,
  TEMP_SENSORS,
  OUTDOOR_TEMP,
  INDOOR_HUMIDITY,
  WINDOW_SENSORS,
  WEATHER_ENTITY,
  NAVIEN_HEATING_MODE,
  NAVIEN_HEAT_CAPACITY,
  ALL_MONITORED_ENTITIES,
} from "./entities.ts";

/** All columns in the snapshots table (excluding timestamp). */
const SNAPSHOT_COLUMNS = [
  "thermostat_upstairs_temp",
  "thermostat_upstairs_target",
  "thermostat_upstairs_action",
  "thermostat_downstairs_temp",
  "thermostat_downstairs_target",
  "thermostat_downstairs_action",
  "mini_split_bedroom_temp",
  "mini_split_bedroom_target",
  "mini_split_bedroom_mode",
  "mini_split_living_room_temp",
  "mini_split_living_room_target",
  "mini_split_living_room_mode",
  "blower_family_room_mode",
  "blower_office_mode",
  "navien_heating_mode",
  "navien_heat_capacity",
  "outdoor_temp",
  "outdoor_humidity",
  "wind_speed",
  "weather_condition",
  "indoor_humidity",
  "any_window_open",
  "upstairs_aggregate_temp",
  "downstairs_aggregate_temp",
  "family_room_temp",
  "office_temp",
  "bedroom_temp",
  "kitchen_temp",
  "piano_temp",
  "bathroom_temp",
  "living_room_temp",
] as const;

/** Map camelCase SnapshotRow keys to snake_case SQLite columns. */
const CAMEL_TO_SNAKE: Record<string, string> = {
  thermostatUpstairsTemp: "thermostat_upstairs_temp",
  thermostatUpstairsTarget: "thermostat_upstairs_target",
  thermostatUpstairsAction: "thermostat_upstairs_action",
  thermostatDownstairsTemp: "thermostat_downstairs_temp",
  thermostatDownstairsTarget: "thermostat_downstairs_target",
  thermostatDownstairsAction: "thermostat_downstairs_action",
  miniSplitBedroomTemp: "mini_split_bedroom_temp",
  miniSplitBedroomTarget: "mini_split_bedroom_target",
  miniSplitBedroomMode: "mini_split_bedroom_mode",
  miniSplitLivingRoomTemp: "mini_split_living_room_temp",
  miniSplitLivingRoomTarget: "mini_split_living_room_target",
  miniSplitLivingRoomMode: "mini_split_living_room_mode",
  blowerFamilyRoomMode: "blower_family_room_mode",
  blowerOfficeMode: "blower_office_mode",
  navienHeatingMode: "navien_heating_mode",
  navienHeatCapacity: "navien_heat_capacity",
  outdoorTemp: "outdoor_temp",
  outdoorHumidity: "outdoor_humidity",
  windSpeed: "wind_speed",
  weatherCondition: "weather_condition",
  indoorHumidity: "indoor_humidity",
  anyWindowOpen: "any_window_open",
  upstairsAggregateTemp: "upstairs_aggregate_temp",
  downstairsAggregateTemp: "downstairs_aggregate_temp",
  familyRoomTemp: "family_room_temp",
  officeTemp: "office_temp",
  bedroomTemp: "bedroom_temp",
  kitchenTemp: "kitchen_temp",
  pianoTemp: "piano_temp",
  bathroomTemp: "bathroom_temp",
  livingRoomTemp: "living_room_temp",
};

const CREATE_TABLE_SQL = `
CREATE TABLE IF NOT EXISTS snapshots (
  timestamp TEXT PRIMARY KEY,
  thermostat_upstairs_temp REAL,
  thermostat_upstairs_target REAL,
  thermostat_upstairs_action TEXT,
  thermostat_downstairs_temp REAL,
  thermostat_downstairs_target REAL,
  thermostat_downstairs_action TEXT,
  mini_split_bedroom_temp REAL,
  mini_split_bedroom_target REAL,
  mini_split_bedroom_mode TEXT,
  mini_split_living_room_temp REAL,
  mini_split_living_room_target REAL,
  mini_split_living_room_mode TEXT,
  blower_family_room_mode TEXT,
  blower_office_mode TEXT,
  navien_heating_mode TEXT,
  navien_heat_capacity REAL,
  outdoor_temp REAL,
  outdoor_humidity REAL,
  wind_speed REAL,
  weather_condition TEXT,
  indoor_humidity REAL,
  any_window_open INTEGER,
  upstairs_aggregate_temp REAL,
  downstairs_aggregate_temp REAL,
  family_room_temp REAL,
  office_temp REAL,
  bedroom_temp REAL,
  kitchen_temp REAL,
  piano_temp REAL,
  bathroom_temp REAL,
  living_room_temp REAL
)`;

const INSERT_SQL = `
INSERT OR IGNORE INTO snapshots (
  timestamp, ${SNAPSHOT_COLUMNS.join(", ")}
) VALUES (
  @timestamp, ${SNAPSHOT_COLUMNS.map((c) => `@${c}`).join(", ")}
)`;

/** Module-level database handle — opened once and reused. */
let _db: DatabaseType | null = null;

function getDb(dbPath: string): DatabaseType {
  if (_db) return _db;
  _db = new Database(dbPath);
  _db.pragma("journal_mode = WAL");
  _db.exec(CREATE_TABLE_SQL);

  // Migrate existing databases: add new columns if missing
  for (const col of ["piano_temp", "bathroom_temp"]) {
    try {
      _db.exec(`ALTER TABLE snapshots ADD COLUMN ${col} REAL`);
    } catch {
      // Column already exists — ignore
    }
  }

  return _db;
}

/** Round a timestamp to the nearest 5-minute boundary for dedup. */
function roundToFiveMinutes(isoTimestamp: string): string {
  const d = new Date(isoTimestamp);
  const minutes = d.getMinutes();
  const rounded = Math.round(minutes / 5) * 5;
  d.setMinutes(rounded, 0, 0);
  return d.toISOString();
}

function getBlowerMode(state: string | undefined, presetMode: unknown): string {
  if (state === "off" || !state) return "off";
  if (typeof presetMode === "string") return presetMode;
  return "low"; // fan is on but no preset — default to low
}

async function buildSnapshot(client: HAClient): Promise<SnapshotRow> {
  const states = await client.getStates(ALL_MONITORED_ENTITIES);
  const stateMap = new Map(states.map((s) => [s.entity_id, s]));

  const getAttrNum = (entityId: string, attr: string, fallback: number): number => {
    const s = stateMap.get(entityId);
    const val = s?.attributes[attr];
    return typeof val === "number" ? val : fallback;
  };

  const getAttrStr = (entityId: string, attr: string, fallback: string): string => {
    const s = stateMap.get(entityId);
    const val = s?.attributes[attr];
    return typeof val === "string" ? val : fallback;
  };

  const getSensorNum = (entityId: string, fallback: number): number => {
    const s = stateMap.get(entityId);
    if (!s) return fallback;
    const val = parseFloat(s.state);
    return Number.isFinite(val) ? val : fallback;
  };

  const anyWindowOpen = WINDOW_SENSORS.some(
    (id) => stateMap.get(id)?.state === "on",
  );

  const weather = stateMap.get(WEATHER_ENTITY);

  const blowerFR = stateMap.get(BLOWER_FAMILY_ROOM);
  const blowerOff = stateMap.get(BLOWER_OFFICE);

  return {
    timestamp: new Date().toISOString(),
    // Thermostat zones
    thermostatUpstairsTemp: getAttrNum(THERMOSTAT_UPSTAIRS, "current_temperature", 0),
    thermostatUpstairsTarget: getAttrNum(THERMOSTAT_UPSTAIRS, "temperature", 0),
    thermostatUpstairsAction: getAttrStr(THERMOSTAT_UPSTAIRS, "hvac_action", "idle"),
    thermostatDownstairsTemp: getAttrNum(THERMOSTAT_DOWNSTAIRS, "current_temperature", 0),
    thermostatDownstairsTarget: getAttrNum(THERMOSTAT_DOWNSTAIRS, "temperature", 0),
    thermostatDownstairsAction: getAttrStr(THERMOSTAT_DOWNSTAIRS, "hvac_action", "idle"),
    // Mini splits
    miniSplitBedroomTemp: getAttrNum(MINI_SPLIT_BEDROOM, "current_temperature", 0),
    miniSplitBedroomTarget: getAttrNum(MINI_SPLIT_BEDROOM, "temperature", 0),
    miniSplitBedroomMode: stateMap.get(MINI_SPLIT_BEDROOM)?.state ?? "off",
    miniSplitLivingRoomTemp: getAttrNum(MINI_SPLIT_LIVING_ROOM, "current_temperature", 0),
    miniSplitLivingRoomTarget: getAttrNum(MINI_SPLIT_LIVING_ROOM, "temperature", 0),
    miniSplitLivingRoomMode: stateMap.get(MINI_SPLIT_LIVING_ROOM)?.state ?? "off",
    // Blowers
    blowerFamilyRoomMode: getBlowerMode(blowerFR?.state, blowerFR?.attributes["preset_mode"]),
    blowerOfficeMode: getBlowerMode(blowerOff?.state, blowerOff?.attributes["preset_mode"]),
    // Navien
    navienHeatingMode: stateMap.get(NAVIEN_HEATING_MODE)?.state ?? "Idle",
    navienHeatCapacity: getSensorNum(NAVIEN_HEAT_CAPACITY, 0),
    // Environment
    outdoorTemp: getSensorNum(OUTDOOR_TEMP, 0),
    outdoorHumidity: weather ? (weather.attributes["humidity"] as number ?? 0) : 0,
    windSpeed: weather ? (weather.attributes["wind_speed"] as number ?? 0) : 0,
    weatherCondition: weather?.state ?? "unknown",
    indoorHumidity: getSensorNum(INDOOR_HUMIDITY, 0),
    anyWindowOpen,
    // Per-room temperatures
    upstairsAggregateTemp: getSensorNum(TEMP_SENSORS.upstairs_aggregate, 0),
    downstairsAggregateTemp: getSensorNum(TEMP_SENSORS.downstairs_aggregate, 0),
    familyRoomTemp: getSensorNum(TEMP_SENSORS.family_room, 0),
    officeTemp: getSensorNum(TEMP_SENSORS.office, 0),
    bedroomTemp: getSensorNum(TEMP_SENSORS.bedroom, 0),
    kitchenTemp: getSensorNum(TEMP_SENSORS.kitchen, 0),
    pianoTemp: getSensorNum(TEMP_SENSORS.piano, 0),
    bathroomTemp: getSensorNum(TEMP_SENSORS.bathroom, 0),
    livingRoomTemp: getSensorNum(TEMP_SENSORS.living_room, 0),
  };
}

function writeSnapshot(snapshot: SnapshotRow, dbPath: string): string {
  const db = getDb(dbPath);
  const roundedTs = roundToFiveMinutes(snapshot.timestamp);

  // Build snake_case params for the INSERT
  const params: Record<string, string | number> = { timestamp: roundedTs };
  for (const [camel, snake] of Object.entries(CAMEL_TO_SNAKE)) {
    const val = snapshot[camel as keyof SnapshotRow];
    if (typeof val === "boolean") {
      params[snake] = val ? 1 : 0;
    } else {
      params[snake] = val as string | number;
    }
  }

  db.prepare(INSERT_SQL).run(params);
  return dbPath;
}

export async function collectOnce(client: HAClient, config: Config): Promise<void> {
  await mkdir(dirname(config.dbPath), { recursive: true });
  const snapshot = await buildSnapshot(client);
  const filepath = writeSnapshot(snapshot, config.dbPath);
  console.log(`[collector] Wrote snapshot to ${filepath} (ts: ${roundToFiveMinutes(snapshot.timestamp)})`);
}

export async function collectLoop(client: HAClient, config: Config): Promise<never> {
  console.log(
    `[collector] Starting collection loop (interval: ${config.snapshotIntervalMs / 1000}s)`,
  );
  console.log(`[collector] Database: ${config.dbPath}`);

  while (true) {
    try {
      await collectOnce(client, config);
    } catch (err) {
      console.error("[collector] Error collecting snapshot:", err);
    }
    await new Promise((resolve) => setTimeout(resolve, config.snapshotIntervalMs));
  }
}

/** Close the database handle (for clean shutdown). */
export function closeDb(): void {
  if (_db) {
    _db.close();
    _db = null;
  }
}
