/**
 * State snapshot collector.
 *
 * Periodically reads entity states via HAClient and writes readings
 * to a SQLite EAV table (timestamp, name, value). Entity dispatch and
 * column definitions are driven by weatherstat.yaml via the config loader.
 */

import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import Database from "better-sqlite3";
import type { Database as DatabaseType } from "better-sqlite3";

import type { HAClient, HAEntityState, SnapshotRow, WeatherForecastEntry } from "./types.ts";
import type { Config } from "./config.ts";
import { config } from "./yaml-config.ts";

const INSERT_READING_SQL = `
INSERT OR IGNORE INTO readings (timestamp, name, value)
VALUES (@timestamp, @name, @value)
`;

/** Module-level database handle — opened once and reused. */
let _db: DatabaseType | null = null;

function getDb(dbPath: string): DatabaseType {
  if (_db) return _db;
  _db = new Database(dbPath);
  _db.pragma("journal_mode = WAL");

  // EAV readings table — canonical format
  _db.exec(config.createReadingsTableSql);

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

  return row as SnapshotRow;
}

function writeSnapshot(snapshot: SnapshotRow, dbPath: string): string {
  const db = getDb(dbPath);
  const roundedTs = roundToFiveMinutes(snapshot.timestamp);

  // Build snake_case (name, value) pairs for EAV insert
  const insertReading = db.prepare(INSERT_READING_SQL);
  const writeAll = db.transaction(() => {
    for (const [camel, snake] of Object.entries(config.camelToSnake)) {
      const val = snapshot[camel];
      const value = typeof val === "boolean" ? (val ? "1" : "0") : String(val);
      insertReading.run({ timestamp: roundedTs, name: snake, value });
    }
  });
  writeAll();

  return dbPath;
}

/** Fetch hourly forecast and flatten into snapshot row fields. */
async function injectForecast(
  client: HAClient,
  row: SnapshotRow,
  weatherEntityId: string,
): Promise<void> {
  try {
    const response = await client.callServiceWithResponse({
      domain: "weather",
      service: "get_forecasts",
      target: { entity_id: weatherEntityId },
      serviceData: { type: "hourly" },
    });

    const entityData = response[weatherEntityId] as { forecast?: WeatherForecastEntry[] } | undefined;
    const entries = entityData?.forecast ?? [];
    if (entries.length === 0) return;

    // Sort by datetime
    const sorted = [...entries].sort((a, b) => a.datetime.localeCompare(b.datetime));

    const now = new Date();

    // Find the entry closest to each hour ahead
    const findClosest = (hoursAhead: number): WeatherForecastEntry | null => {
      const target = new Date(now.getTime() + hoursAhead * 3600_000);
      let best: WeatherForecastEntry | null = null;
      let bestDiff = Infinity;
      for (const entry of sorted) {
        const diff = Math.abs(new Date(entry.datetime).getTime() - target.getTime());
        if (diff < bestDiff) {
          bestDiff = diff;
          best = entry;
        }
      }
      // Accept if within 90 minutes
      return bestDiff <= 5400_000 ? best : null;
    };

    // Hourly temps (1h through 12h)
    for (let h = 1; h <= 12; h++) {
      const entry = findClosest(h);
      const camel = `forecastTemp${h}h`;
      row[camel] = entry?.temperature ?? 0;
    }

    // Condition and wind at key horizons
    for (const h of [1, 2, 4, 6, 12]) {
      const entry = findClosest(h);
      row[`forecastCondition${h}h`] = entry?.condition ?? "";
      row[`forecastWind${h}h`] = entry?.windSpeed ?? 0;
    }
  } catch (err) {
    console.warn("[collector] Forecast fetch failed (non-fatal):", err);
  }
}

export async function collectOnce(client: HAClient, appConfig: Config): Promise<void> {
  await mkdir(dirname(appConfig.dbPath), { recursive: true });
  const snapshot = await buildSnapshot(client);

  // Inject forecast data from service call
  await injectForecast(client, snapshot, config.weatherEntity);

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
