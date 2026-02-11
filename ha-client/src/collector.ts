/**
 * State snapshot collector.
 *
 * Periodically reads entity states via HAClient and writes Parquet snapshots
 * to data/snapshots/. Depends on the HAClient interface, not WebSocket directly.
 */

import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { ParquetSchema, ParquetWriter } from "@dsnp/parquetjs";

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

const SNAPSHOT_SCHEMA = new ParquetSchema({
  timestamp: { type: "UTF8" },
  // Thermostat zones
  thermostatUpstairsTemp: { type: "DOUBLE" },
  thermostatUpstairsTarget: { type: "DOUBLE" },
  thermostatUpstairsAction: { type: "UTF8" },
  thermostatDownstairsTemp: { type: "DOUBLE" },
  thermostatDownstairsTarget: { type: "DOUBLE" },
  thermostatDownstairsAction: { type: "UTF8" },
  // Mini splits
  miniSplitBedroomTemp: { type: "DOUBLE" },
  miniSplitBedroomTarget: { type: "DOUBLE" },
  miniSplitBedroomMode: { type: "UTF8" },
  miniSplitLivingRoomTemp: { type: "DOUBLE" },
  miniSplitLivingRoomTarget: { type: "DOUBLE" },
  miniSplitLivingRoomMode: { type: "UTF8" },
  // Blowers
  blowerFamilyRoomMode: { type: "UTF8" },
  blowerOfficeMode: { type: "UTF8" },
  // Navien
  navienHeatingMode: { type: "UTF8" },
  navienHeatCapacity: { type: "DOUBLE" },
  // Environment
  outdoorTemp: { type: "DOUBLE" },
  outdoorHumidity: { type: "DOUBLE" },
  windSpeed: { type: "DOUBLE" },
  weatherCondition: { type: "UTF8" },
  indoorHumidity: { type: "DOUBLE" },
  anyWindowOpen: { type: "BOOLEAN" },
  // Per-room temperatures
  upstairsAggregateTemp: { type: "DOUBLE" },
  downstairsAggregateTemp: { type: "DOUBLE" },
  familyRoomTemp: { type: "DOUBLE" },
  officeTemp: { type: "DOUBLE" },
  bedroomTemp: { type: "DOUBLE" },
  kitchenTemp: { type: "DOUBLE" },
  livingRoomTemp: { type: "DOUBLE" },
});

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
    livingRoomTemp: getSensorNum(TEMP_SENSORS.living_room, 0),
  };
}

async function writeSnapshot(snapshot: SnapshotRow, snapshotsDir: string): Promise<string> {
  await mkdir(snapshotsDir, { recursive: true });

  const date = snapshot.timestamp.slice(0, 10); // YYYY-MM-DD
  const filename = `snapshot_${date}.parquet`;
  const filepath = resolve(snapshotsDir, filename);

  const writer = await ParquetWriter.openFile(SNAPSHOT_SCHEMA, filepath);
  await writer.appendRow({ ...snapshot } as Record<string, unknown>);
  await writer.close();
  return filepath;
}

export async function collectOnce(client: HAClient, config: Config): Promise<void> {
  const snapshot = await buildSnapshot(client);
  const filepath = await writeSnapshot(snapshot, config.snapshotsDir);
  console.log(`[collector] Wrote snapshot to ${filepath}`);
}

export async function collectLoop(client: HAClient, config: Config): Promise<never> {
  console.log(
    `[collector] Starting collection loop (interval: ${config.snapshotIntervalMs / 1000}s)`,
  );

  while (true) {
    try {
      await collectOnce(client, config);
    } catch (err) {
      console.error("[collector] Error collecting snapshot:", err);
    }
    await new Promise((resolve) => setTimeout(resolve, config.snapshotIntervalMs));
  }
}
