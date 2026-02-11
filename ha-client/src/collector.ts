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
  MINI_SPLIT_1,
  MINI_SPLIT_2,
  FLOOR_HEAT,
  BLOWER_1,
  BLOWER_2,
  TEMP_SENSORS,
  WINDOW_SENSORS,
  WEATHER_ENTITY,
  NAVIEN_HEATER,
  ALL_MONITORED_ENTITIES,
} from "./entities.ts";

const SNAPSHOT_SCHEMA = new ParquetSchema({
  timestamp: { type: "UTF8" },
  thermostatUpstairsTemp: { type: "DOUBLE" },
  thermostatUpstairsTarget: { type: "DOUBLE" },
  thermostatUpstairsAction: { type: "UTF8" },
  thermostatDownstairsTemp: { type: "DOUBLE" },
  thermostatDownstairsTarget: { type: "DOUBLE" },
  thermostatDownstairsAction: { type: "UTF8" },
  miniSplit1Temp: { type: "DOUBLE" },
  miniSplit1Target: { type: "DOUBLE" },
  miniSplit1Mode: { type: "UTF8" },
  miniSplit2Temp: { type: "DOUBLE" },
  miniSplit2Target: { type: "DOUBLE" },
  miniSplit2Mode: { type: "UTF8" },
  floorHeatOn: { type: "BOOLEAN" },
  blower1On: { type: "BOOLEAN" },
  blower2On: { type: "BOOLEAN" },
  outdoorTemp: { type: "DOUBLE" },
  outdoorHumidity: { type: "DOUBLE" },
  windSpeed: { type: "DOUBLE" },
  weatherCondition: { type: "UTF8" },
  navienHeaterActive: { type: "BOOLEAN" },
  anyWindowOpen: { type: "BOOLEAN" },
  // indoorTemps stored as JSON string for flexibility
  indoorTempsJson: { type: "UTF8" },
});

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

  const isOn = (entityId: string): boolean => {
    return stateMap.get(entityId)?.state === "on";
  };

  const anyWindowOpen = WINDOW_SENSORS.some((id) => isOn(id));

  const indoorTemps: Record<string, number> = {};
  for (const [location, entityId] of Object.entries(TEMP_SENSORS)) {
    const s = stateMap.get(entityId);
    if (s) {
      indoorTemps[location] = parseFloat(s.state) || 0;
    }
  }

  const weather = stateMap.get(WEATHER_ENTITY);

  return {
    timestamp: new Date().toISOString(),
    thermostatUpstairsTemp: getAttrNum(THERMOSTAT_UPSTAIRS, "current_temperature", 0),
    thermostatUpstairsTarget: getAttrNum(THERMOSTAT_UPSTAIRS, "temperature", 0),
    thermostatUpstairsAction: getAttrStr(THERMOSTAT_UPSTAIRS, "hvac_action", "idle"),
    thermostatDownstairsTemp: getAttrNum(THERMOSTAT_DOWNSTAIRS, "current_temperature", 0),
    thermostatDownstairsTarget: getAttrNum(THERMOSTAT_DOWNSTAIRS, "temperature", 0),
    thermostatDownstairsAction: getAttrStr(THERMOSTAT_DOWNSTAIRS, "hvac_action", "idle"),
    miniSplit1Temp: getAttrNum(MINI_SPLIT_1, "current_temperature", 0),
    miniSplit1Target: getAttrNum(MINI_SPLIT_1, "temperature", 0),
    miniSplit1Mode: stateMap.get(MINI_SPLIT_1)?.state ?? "off",
    miniSplit2Temp: getAttrNum(MINI_SPLIT_2, "current_temperature", 0),
    miniSplit2Target: getAttrNum(MINI_SPLIT_2, "temperature", 0),
    miniSplit2Mode: stateMap.get(MINI_SPLIT_2)?.state ?? "off",
    floorHeatOn: isOn(FLOOR_HEAT),
    blower1On: isOn(BLOWER_1),
    blower2On: isOn(BLOWER_2),
    outdoorTemp: weather ? (weather.attributes["temperature"] as number ?? 0) : 0,
    outdoorHumidity: weather ? (weather.attributes["humidity"] as number ?? 0) : 0,
    windSpeed: weather ? (weather.attributes["wind_speed"] as number ?? 0) : 0,
    weatherCondition: weather?.state ?? "unknown",
    navienHeaterActive: isOn(NAVIEN_HEATER),
    anyWindowOpen,
    indoorTemps,
  };
}

async function writeSnapshot(snapshot: SnapshotRow, snapshotsDir: string): Promise<string> {
  await mkdir(snapshotsDir, { recursive: true });

  const date = snapshot.timestamp.slice(0, 10); // YYYY-MM-DD
  const filename = `snapshot_${date}.parquet`;
  const filepath = resolve(snapshotsDir, filename);

  const writer = await ParquetWriter.openFile(SNAPSHOT_SCHEMA, filepath);

  await writer.appendRow({
    ...snapshot,
    indoorTempsJson: JSON.stringify(snapshot.indoorTemps),
  });

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
