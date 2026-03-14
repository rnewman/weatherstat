import { resolve } from "node:path";
import { homedir } from "node:os";

/** Weatherstat data directory: WEATHERSTAT_DATA_DIR or ~/.weatherstat */
export const dataDir: string =
  process.env["WEATHERSTAT_DATA_DIR"] ?? resolve(homedir(), ".weatherstat");
