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
  MINI_SPLIT_BEDROOM,
  MINI_SPLIT_LIVING_ROOM,
  BLOWER_FAMILY_ROOM,
  BLOWER_OFFICE,
} from "./entities.ts";

async function loadLatestPrediction(predictionsDir: string): Promise<Prediction | null> {
  let files: string[];
  try {
    files = await readdir(predictionsDir);
  } catch {
    return null;
  }

  const jsonFiles = files.filter((f) => f.startsWith("command_") && f.endsWith(".json")).sort();
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
  // Only set temperature when actively heating/cooling — "off" doesn't need one
  if (mode !== "off") {
    await client.callService({
      domain: "climate",
      service: "set_temperature",
      target: { entity_id: entityId },
      serviceData: { temperature: targetTemp },
    });
  }
  console.log(`[executor] Set ${entityId} to ${mode}${mode !== "off" ? ` @ ${targetTemp}` : ""}`);
}

async function setBlower(
  client: HAClient,
  entityId: string,
  mode: string,
): Promise<void> {
  if (mode === "off") {
    await client.callService({
      domain: "fan",
      service: "turn_off",
      target: { entity_id: entityId },
    });
  } else {
    await client.callService({
      domain: "fan",
      service: "turn_on",
      target: { entity_id: entityId },
    });
    await client.callService({
      domain: "fan",
      service: "set_preset_mode",
      target: { entity_id: entityId },
      serviceData: { preset_mode: mode },
    });
  }
  console.log(`[executor] Set ${entityId} to ${mode}`);
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
  await setMiniSplit(client, MINI_SPLIT_BEDROOM, prediction.miniSplitBedroomTarget, prediction.miniSplitBedroomMode);
  await setMiniSplit(client, MINI_SPLIT_LIVING_ROOM, prediction.miniSplitLivingRoomTarget, prediction.miniSplitLivingRoomMode);

  // Apply blower modes
  await setBlower(client, BLOWER_FAMILY_ROOM, prediction.blowerFamilyRoomMode);
  await setBlower(client, BLOWER_OFFICE, prediction.blowerOfficeMode);

  console.log("[executor] All commands executed");
}
