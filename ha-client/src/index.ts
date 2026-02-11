/**
 * Weatherstat HA client entry point.
 *
 * Usage:
 *   npx tsx src/index.ts collect    # Run snapshot collection loop
 *   npx tsx src/index.ts execute    # Execute latest prediction
 */

import { loadConfig } from "./config.ts";
import { WebSocketHAClient } from "./connection.ts";
import { collectLoop, collectOnce, closeDb } from "./collector.ts";
import { executePrediction } from "./executor.ts";

const COMMANDS = ["collect", "execute", "once"] as const;
type Command = (typeof COMMANDS)[number];

function printUsage(): void {
  console.log("Usage: weatherstat <command>");
  console.log("");
  console.log("Commands:");
  console.log("  collect   Start the snapshot collection loop");
  console.log("  once      Collect a single snapshot");
  console.log("  execute   Execute the latest prediction");
}

async function main(): Promise<void> {
  const command = process.argv[2] as Command | undefined;

  if (!command || !COMMANDS.includes(command)) {
    printUsage();
    process.exit(command ? 1 : 0);
  }

  const config = loadConfig();
  const client = new WebSocketHAClient(config.haUrl, config.haToken);

  // Graceful shutdown
  const shutdown = async () => {
    console.log("\nShutting down...");
    closeDb();
    await client.disconnect();
    process.exit(0);
  };
  process.on("SIGINT", () => void shutdown());
  process.on("SIGTERM", () => void shutdown());

  try {
    switch (command) {
      case "collect":
        await collectLoop(client, config);
        break;
      case "once":
        await collectOnce(client, config);
        break;
      case "execute":
        await executePrediction(client, config);
        break;
    }
  } finally {
    await client.disconnect();
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
