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
import { config as yamlConfig } from "./yaml-config.ts";

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

/** Convert snake_case to camelCase for prediction JSON key lookup. */
function snakeToCamel(s: string): string {
  const parts = s.split("_");
  return parts[0] + parts.slice(1).map((p) => (p ? p[0]!.toUpperCase() + p.slice(1) : "")).join("");
}

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
  const controlledEntities = Object.values(yamlConfig.effectors).map((e) => e.entityId);
  const states = await client.getStates(controlledEntities);
  const stateMap = new Map<string, HAEntityState>(states.map((s) => [s.entity_id, s]));

  // Load last-applied state (what the executor actually set last time)
  const lastState = await loadExecutorState(config);
  const OVERRIDE_STALE_MS = 30 * 60 * 1000; // 30 minutes (2× loop interval)

  let last = lastState?.devices ?? {};
  if (lastState?.timestamp) {
    const age = Date.now() - new Date(lastState.timestamp).getTime();
    if (age > OVERRIDE_STALE_MS) {
      console.log(
        `[executor] Executor state is ${Math.round(age / 60000)}m old — clearing overrides`,
      );
      last = {};
    }
  }

  // Clone last-applied — updated per-device only when we actually apply
  const updated: Record<string, DeviceState> = { ...last };

  let applied = 0;
  let lazySkips = 0;
  let overrideSkips = 0;

  // ── Iterate all effectors from config ─────────────────────────────────

  for (const [name, ecfg] of Object.entries(yamlConfig.effectors)) {
    if (ecfg.domain === "climate" && ecfg.modeControl === "manual") {
      // ── Manual-mode climate (thermostat): target only ───────────────
      const targetKey = snakeToCamel(`${name}_target`);
      const desiredTarget = prediction[targetKey] as number | undefined;

      if (desiredTarget === undefined) {
        console.log(`[executor] ${name}: ineligible, skipping`);
        continue;
      }

      const current = readThermostat(stateMap, ecfg.entityId);

      if (current.target !== undefined && Math.abs(current.target - desiredTarget) < TARGET_TOLERANCE) {
        console.log(`[executor] ${name}: already at ${current.target}°F, skipping`);
        lazySkips++;
        continue;
      }

      const lastApplied = last[name];
      if (
        !force
        && lastApplied?.target !== undefined
        && current.target !== undefined
        && Math.abs(current.target - lastApplied.target) > TARGET_TOLERANCE
      ) {
        console.log(
          `[executor] ${name}: override detected`
          + ` (current ${current.target}°F, we last set ${lastApplied.target}°F), skipping`,
        );
        overrideSkips++;
        continue;
      }

      await applyThermostat(client, ecfg.entityId, desiredTarget);
      console.log(`[executor] ${name}: set to ${desiredTarget}°F`);
      updated[name] = { target: desiredTarget };
      applied++;

    } else if (ecfg.domain === "climate") {
      // ── Automatic-mode climate (mini-split): mode + target ──────────
      const modeKey = snakeToCamel(`${name}_mode`);
      const targetKey = snakeToCamel(`${name}_target`);
      const desiredMode = prediction[modeKey] as string | undefined;
      const desiredTarget = prediction[targetKey] as number | undefined;

      if (desiredMode === undefined) {
        console.log(`[executor] ${name}: not in command, skipping`);
        continue;
      }

      const current = readMiniSplit(stateMap, ecfg.entityId);

      const modeMatch = current.mode === desiredMode;
      const targetMatch = desiredMode === "off"
        || (desiredTarget !== undefined && current.target !== undefined
          && Math.abs(current.target - desiredTarget) < TARGET_TOLERANCE);
      if (modeMatch && targetMatch) {
        const desc = desiredMode === "off" ? "off" : `${desiredMode} @ ${current.target}°F`;
        console.log(`[executor] ${name}: already ${desc}, skipping`);
        lazySkips++;
        continue;
      }

      const lastApplied = last[name];
      if (!force && lastApplied?.mode !== undefined && current.mode !== lastApplied.mode) {
        console.log(
          `[executor] ${name}: override detected`
          + ` (currently ${current.mode}, we last set ${lastApplied.mode}), skipping`,
        );
        overrideSkips++;
        continue;
      }

      await applyMiniSplit(client, ecfg.entityId, desiredMode, desiredTarget ?? 72);
      const desc = desiredMode === "off" ? "off" : `${desiredMode} @ ${desiredTarget}°F`;
      console.log(`[executor] ${name}: set to ${desc}`);
      updated[name] = { mode: desiredMode, target: desiredTarget };
      applied++;

    } else {
      // ── Fan entity (blower): mode only ─────────────────────────────
      const modeKey = snakeToCamel(`${name}_mode`);
      const desiredMode = prediction[modeKey] as string | undefined;

      if (desiredMode === undefined) {
        console.log(`[executor] ${name}: not in command, skipping`);
        continue;
      }

      const current = readBlower(stateMap, ecfg.entityId);

      if (current.mode === desiredMode) {
        console.log(`[executor] ${name}: already ${desiredMode}, skipping`);
        lazySkips++;
        continue;
      }

      const lastApplied = last[name];
      if (!force && lastApplied?.mode !== undefined && current.mode !== lastApplied.mode) {
        console.log(
          `[executor] ${name}: override detected`
          + ` (currently ${current.mode}, we last set ${lastApplied.mode}), skipping`,
        );
        overrideSkips++;
        continue;
      }

      await applyBlower(client, ecfg.entityId, desiredMode);
      console.log(`[executor] ${name}: set to ${desiredMode}`);
      updated[name] = { mode: desiredMode };
      applied++;
    }
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
