/**
 * Prediction executor.
 *
 * Reads the latest prediction JSON from data/predictions/ and executes
 * HVAC control commands via HAClient. Depends on the HAClient interface,
 * not WebSocket directly.
 */

import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

import type { HAClient, Prediction } from "./types.ts";
import type { Config } from "./config.ts";
import {
  THERMOSTAT_UPSTAIRS,
  THERMOSTAT_DOWNSTAIRS,
  MINI_SPLIT_1,
  MINI_SPLIT_2,
  FLOOR_HEAT,
  BLOWER_1,
  BLOWER_2,
} from "./entities.ts";

async function loadLatestPrediction(predictionsDir: string): Promise<Prediction | null> {
  let files: string[];
  try {
    files = await readdir(predictionsDir);
  } catch {
    return null;
  }

  const jsonFiles = files.filter((f) => f.endsWith(".json")).sort();
  const latest = jsonFiles[jsonFiles.length - 1];
  if (!latest) return null;

  const content = await readFile(resolve(predictionsDir, latest), "utf-8");
  return JSON.parse(content) as Prediction;
}

async function setThermostat(
  client: HAClient,
  entityId: string,
  targetTemp: number,
): Promise<void> {
  await client.callService({
    domain: "climate",
    service: "set_temperature",
    target: { entity_id: entityId },
    serviceData: { temperature: targetTemp },
  });
  console.log(`[executor] Set ${entityId} target to ${targetTemp}`);
}

async function setMiniSplit(
  client: HAClient,
  entityId: string,
  targetTemp: number,
  mode: string,
): Promise<void> {
  await client.callService({
    domain: "climate",
    service: "set_hvac_mode",
    target: { entity_id: entityId },
    serviceData: { hvac_mode: mode },
  });
  await client.callService({
    domain: "climate",
    service: "set_temperature",
    target: { entity_id: entityId },
    serviceData: { temperature: targetTemp },
  });
  console.log(`[executor] Set ${entityId} to ${mode} @ ${targetTemp}`);
}

async function setSwitch(
  client: HAClient,
  entityId: string,
  on: boolean,
): Promise<void> {
  await client.callService({
    domain: "switch",
    service: on ? "turn_on" : "turn_off",
    target: { entity_id: entityId },
  });
  console.log(`[executor] Set ${entityId} to ${on ? "on" : "off"}`);
}

export async function executePrediction(client: HAClient, config: Config): Promise<void> {
  const prediction = await loadLatestPrediction(config.predictionsDir);
  if (!prediction) {
    console.log("[executor] No prediction found, skipping");
    return;
  }

  console.log(`[executor] Executing prediction from ${prediction.timestamp} (confidence: ${prediction.confidence})`);

  // Apply thermostat targets
  await setThermostat(client, THERMOSTAT_UPSTAIRS, prediction.thermostatUpstairsTarget);
  await setThermostat(client, THERMOSTAT_DOWNSTAIRS, prediction.thermostatDownstairsTarget);

  // Apply mini split settings
  await setMiniSplit(client, MINI_SPLIT_1, prediction.miniSplit1Target, prediction.miniSplit1Mode);
  await setMiniSplit(client, MINI_SPLIT_2, prediction.miniSplit2Target, prediction.miniSplit2Mode);

  // Apply floor heat and blowers
  await setSwitch(client, FLOOR_HEAT, prediction.floorHeatOn);
  await setSwitch(client, BLOWER_1, prediction.blower1On);
  await setSwitch(client, BLOWER_2, prediction.blower2On);

  console.log("[executor] All commands executed");
}
