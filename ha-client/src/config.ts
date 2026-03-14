import { resolve } from "node:path";

import { dataDir } from "./data-dir.ts";

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export interface Config {
  haUrl: string;
  haToken: string;
  snapshotIntervalMs: number;
  dataDir: string;
  snapshotsDir: string;
  predictionsDir: string;
  dbPath: string;
}

export function loadConfig(): Config {
  const snapshotsDir = resolve(dataDir, "snapshots");

  return {
    haUrl: requireEnv("HA_URL"),
    haToken: requireEnv("HA_TOKEN"),
    snapshotIntervalMs:
      (parseInt(process.env["SNAPSHOT_INTERVAL"] ?? "300", 10)) * 1000,
    dataDir,
    snapshotsDir,
    predictionsDir: resolve(dataDir, "predictions"),
    dbPath: resolve(snapshotsDir, "snapshots.db"),
  };
}
