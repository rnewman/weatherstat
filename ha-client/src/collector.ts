/**
 * State snapshot collector.
 *
 * Periodically reads entity states via HAClient and writes snapshots
 * to a SQLite database. Column definitions and entity dispatch are
 * driven by weatherstat.yaml via the config loader.
 */

import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import Database from "better-sqlite3";
import type { Database as DatabaseType } from "better-sqlite3";

import type { HAClient, HAEntityState, SnapshotRow } from "./types.ts";
import type { Config } from "./config.ts";
import { config } from "./yaml-config.ts";

const INSERT_SQL = `
INSERT OR IGNORE INTO snapshots (
  timestamp, ${config.snapshotColumns.join(", ")}
) VALUES (
  @timestamp, ${config.snapshotColumns.map((c) => `@${c}`).join(", ")}
)`;

/** Module-level database handle — opened once and reused. */
let _db: DatabaseType | null = null;

function getDb(dbPath: string): DatabaseType {
  if (_db) return _db;
  _db = new Database(dbPath);
  _db.pragma("journal_mode = WAL");
  _db.exec(config.createTableSql);

  // Generic migration: add any columns from config that are missing
  const existing = new Set(
    (_db.pragma("table_info(snapshots)") as Array<{ name: string }>).map((r) => r.name),
  );
  for (const col of config.columnDefs) {
    if (!existing.has(col.snake)) {
      _db.exec(`ALTER TABLE snapshots ADD COLUMN ${col.snake} ${col.sqlType}`);
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

async function buildSnapshot(client: HAClient): Promise<SnapshotRow> {
  const states = await client.getStates(config.allMonitoredEntities);
  const stateMap = new Map<string, HAEntityState>(states.map((s) => [s.entity_id, s]));

  // Build snapshot using config-driven extraction dispatch
  const row: Record<string, string | number | boolean> = {
    timestamp: new Date().toISOString(),
  };

  for (const col of config.columnDefs) {
    row[col.camel] = col.extract(stateMap);
  }

  return row as unknown as SnapshotRow;
}

function writeSnapshot(snapshot: SnapshotRow, dbPath: string): string {
  const db = getDb(dbPath);
  const roundedTs = roundToFiveMinutes(snapshot.timestamp);

  // Build snake_case params for the INSERT
  const params: Record<string, string | number> = { timestamp: roundedTs };
  for (const [camel, snake] of Object.entries(config.camelToSnake)) {
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

export async function collectOnce(client: HAClient, appConfig: Config): Promise<void> {
  await mkdir(dirname(appConfig.dbPath), { recursive: true });
  const snapshot = await buildSnapshot(client);
  const filepath = writeSnapshot(snapshot, appConfig.dbPath);
  console.log(
    `[collector] Wrote snapshot to ${filepath} (ts: ${roundToFiveMinutes(snapshot.timestamp)})`,
  );
}

export async function collectLoop(client: HAClient, appConfig: Config): Promise<never> {
  console.log(
    `[collector] Starting collection loop (interval: ${appConfig.snapshotIntervalMs / 1000}s)`,
  );
  console.log(`[collector] Database: ${appConfig.dbPath}`);

  while (true) {
    try {
      await collectOnce(client, appConfig);
    } catch (err) {
      console.error("[collector] Error collecting snapshot:", err);
    }
    await new Promise((resolve) => setTimeout(resolve, appConfig.snapshotIntervalMs));
  }
}

/** Close the database handle (for clean shutdown). */
export function closeDb(): void {
  if (_db) {
    _db.close();
    _db = null;
  }
}
