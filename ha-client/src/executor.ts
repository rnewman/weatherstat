/**
 * Prediction executor.
 *
 * Reads the latest prediction JSON from data/predictions/ and executes
 * HVAC control commands via HAClient.
 *
 * Features:
 * - Lazy execution: skips commands when HA is already in the desired state
 * - Override detection: skips devices that have been manually changed since
 *   the last execution (respects user overrides until cleared)
 * - Force mode (--force): bypasses override detection
 */

import { readdir, readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";

import type { HAClient, HAEntityState, Prediction } from "./types.ts";
import type { Config } from "./config.ts";
import {
  THERMOSTAT_UPSTAIRS,
  THERMOSTAT_DOWNSTAIRS,
  MINI_SPLIT_BEDROOM,
  MINI_SPLIT_LIVING_ROOM,
  BLOWER_FAMILY_ROOM,
  BLOWER_OFFICE,
} from "./entities.ts";

// ── Types ────────────────────────────────────────────────────────────────

interface DeviceState {
  mode?: string;
  target?: number;
}

interface ExecutorState {
  timestamp: string;
  devices: Record<string, DeviceState>;
}

// ── Constants ────────────────────────────────────────────────────────────

/** Ignore temperature differences smaller than this (°F). */
const TARGET_TOLERANCE = 0.5;

// ── State persistence ────────────────────────────────────────────────────

function executorStatePath(config: Config): string {
  return resolve(config.dataDir, "executor_state.json");
}

async function loadExecutorState(config: Config): Promise<ExecutorState | null> {
  try {
    const content = await readFile(executorStatePath(config), "utf-8");
    return JSON.parse(content) as ExecutorState;
  } catch {
    return null;
  }
}

async function saveExecutorState(config: Config, state: ExecutorState): Promise<void> {
  const p = executorStatePath(config);
  await mkdir(dirname(p), { recursive: true });
  await writeFile(p, JSON.stringify(state, null, 2));
}

// ── HA state readers ─────────────────────────────────────────────────────

function readThermostat(stateMap: Map<string, HAEntityState>, entityId: string): DeviceState {
  const entity = stateMap.get(entityId);
  const target = entity?.attributes["temperature"];
  return { target: typeof target === "number" ? target : undefined };
}

function readMiniSplit(stateMap: Map<string, HAEntityState>, entityId: string): DeviceState {
  const entity = stateMap.get(entityId);
  const target = entity?.attributes["temperature"];
  return {
    mode: entity?.state ?? "unknown",
    target: typeof target === "number" ? target : undefined,
  };
}

function readBlower(stateMap: Map<string, HAEntityState>, entityId: string): DeviceState {
  const entity = stateMap.get(entityId);
  if (!entity || entity.state === "off") return { mode: "off" };
  const preset = entity.attributes["preset_mode"];
  return { mode: typeof preset === "string" ? preset : "low" };
}

// ── Device command helpers ───────────────────────────────────────────────

async function applyThermostat(
  client: HAClient,
  entityId: string,
  target: number,
): Promise<void> {
  await client.callService({
    domain: "climate",
    service: "set_temperature",
    target: { entity_id: entityId },
    serviceData: { temperature: target },
  });
}

async function applyMiniSplit(
  client: HAClient,
  entityId: string,
  mode: string,
  target: number,
): Promise<void> {
  await client.callService({
    domain: "climate",
    service: "set_hvac_mode",
    target: { entity_id: entityId },
    serviceData: { hvac_mode: mode },
  });
  if (mode !== "off") {
    await client.callService({
      domain: "climate",
      service: "set_temperature",
      target: { entity_id: entityId },
      serviceData: { temperature: target },
    });
  }
}

async function applyBlower(
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
}

// ── Prediction loader ────────────────────────────────────────────────────

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

// ── Main executor ────────────────────────────────────────────────────────

export async function executePrediction(
  client: HAClient,
  config: Config,
  force: boolean = false,
): Promise<void> {
  const prediction = await loadLatestPrediction(config.predictionsDir);
  if (!prediction) {
    console.log("[executor] No prediction found, skipping");
    return;
  }

  console.log(
    `[executor] Executing prediction from ${prediction.timestamp}`
    + ` (confidence: ${prediction.confidence.toFixed(2)})`,
  );
  if (force) console.log("[executor] Force mode — overrides will be ignored");

  // Fetch current HA states for all controlled entities
  const controlledEntities = [
    THERMOSTAT_UPSTAIRS, THERMOSTAT_DOWNSTAIRS,
    MINI_SPLIT_BEDROOM, MINI_SPLIT_LIVING_ROOM,
    BLOWER_FAMILY_ROOM, BLOWER_OFFICE,
  ];
  const states = await client.getStates(controlledEntities);
  const stateMap = new Map<string, HAEntityState>(states.map((s) => [s.entity_id, s]));

  // Load last-applied state (what the executor actually set last time)
  const lastState = await loadExecutorState(config);
  const last = lastState?.devices ?? {};

  // Clone last-applied — updated per-device only when we actually apply
  const updated: Record<string, DeviceState> = { ...last };

  let applied = 0;
  let lazySkips = 0;
  let overrideSkips = 0;

  // ── Thermostats ────────────────────────────────────────────────────────

  const thermostats: [string, string, number][] = [
    ["upstairs", THERMOSTAT_UPSTAIRS, prediction.thermostatUpstairsTarget],
    ["downstairs", THERMOSTAT_DOWNSTAIRS, prediction.thermostatDownstairsTarget],
  ];

  for (const [name, entityId, desiredTarget] of thermostats) {
    const key = `thermostat_${name}`;
    const current = readThermostat(stateMap, entityId);

    // Lazy: already at desired target
    if (current.target !== undefined && Math.abs(current.target - desiredTarget) < TARGET_TOLERANCE) {
      console.log(`[executor] ${key}: already at ${current.target}°F, skipping`);
      lazySkips++;
      continue;
    }

    // Override: target changed from what we last set
    const lastApplied = last[key];
    if (
      !force
      && lastApplied?.target !== undefined
      && current.target !== undefined
      && Math.abs(current.target - lastApplied.target) > TARGET_TOLERANCE
    ) {
      console.log(
        `[executor] ${key}: override detected`
        + ` (current ${current.target}°F, we last set ${lastApplied.target}°F), skipping`,
      );
      overrideSkips++;
      continue;
    }

    await applyThermostat(client, entityId, desiredTarget);
    console.log(`[executor] ${key}: set to ${desiredTarget}°F`);
    updated[key] = { target: desiredTarget };
    applied++;
  }

  // ── Mini splits ────────────────────────────────────────────────────────

  const miniSplits: [string, string, string, number][] = [
    ["bedroom", MINI_SPLIT_BEDROOM, prediction.miniSplitBedroomMode, prediction.miniSplitBedroomTarget],
    ["living_room", MINI_SPLIT_LIVING_ROOM, prediction.miniSplitLivingRoomMode, prediction.miniSplitLivingRoomTarget],
  ];

  for (const [name, entityId, desiredMode, desiredTarget] of miniSplits) {
    const key = `mini_split_${name}`;
    const current = readMiniSplit(stateMap, entityId);

    // Lazy: mode and target already match
    const modeMatch = current.mode === desiredMode;
    const targetMatch = desiredMode === "off"
      || (current.target !== undefined && Math.abs(current.target - desiredTarget) < TARGET_TOLERANCE);
    if (modeMatch && targetMatch) {
      const desc = desiredMode === "off" ? "off" : `${desiredMode} @ ${current.target}°F`;
      console.log(`[executor] ${key}: already ${desc}, skipping`);
      lazySkips++;
      continue;
    }

    // Override: mode changed from what we last set
    const lastApplied = last[key];
    if (!force && lastApplied?.mode !== undefined && current.mode !== lastApplied.mode) {
      console.log(
        `[executor] ${key}: override detected`
        + ` (currently ${current.mode}, we last set ${lastApplied.mode}), skipping`,
      );
      overrideSkips++;
      continue;
    }

    await applyMiniSplit(client, entityId, desiredMode, desiredTarget);
    const desc = desiredMode === "off" ? "off" : `${desiredMode} @ ${desiredTarget}°F`;
    console.log(`[executor] ${key}: set to ${desc}`);
    updated[key] = { mode: desiredMode, target: desiredTarget };
    applied++;
  }

  // ── Blowers ────────────────────────────────────────────────────────────

  const blowers: [string, string, string][] = [
    ["family_room", BLOWER_FAMILY_ROOM, prediction.blowerFamilyRoomMode],
    ["office", BLOWER_OFFICE, prediction.blowerOfficeMode],
  ];

  for (const [name, entityId, desiredMode] of blowers) {
    const key = `blower_${name}`;
    const current = readBlower(stateMap, entityId);

    // Lazy: already at desired mode
    if (current.mode === desiredMode) {
      console.log(`[executor] ${key}: already ${desiredMode}, skipping`);
      lazySkips++;
      continue;
    }

    // Override: mode changed from what we last set
    const lastApplied = last[key];
    if (!force && lastApplied?.mode !== undefined && current.mode !== lastApplied.mode) {
      console.log(
        `[executor] ${key}: override detected`
        + ` (currently ${current.mode}, we last set ${lastApplied.mode}), skipping`,
      );
      overrideSkips++;
      continue;
    }

    await applyBlower(client, entityId, desiredMode);
    console.log(`[executor] ${key}: set to ${desiredMode}`);
    updated[key] = { mode: desiredMode };
    applied++;
  }

  // ── Save state ─────────────────────────────────────────────────────────

  await saveExecutorState(config, {
    timestamp: new Date().toISOString(),
    devices: updated,
  });

  console.log(
    `[executor] Done: ${applied} applied, ${lazySkips} already correct,`
    + ` ${overrideSkips} overrides respected`,
  );
}
