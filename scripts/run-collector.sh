#!/usr/bin/env bash
# Durable collector wrapper with auto-restart and health monitoring.
#
# - Sources .env for HA_URL/HA_TOKEN
# - Runs the collector in a restart loop with exponential backoff
# - Spawns a background health check every 5 minutes
# - Clean shutdown on SIGINT/SIGTERM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HA_CLIENT_DIR="$PROJECT_ROOT/ha-client"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/collector.log"
HEALTH_CHECK="$SCRIPT_DIR/check-health.sh"
HEALTH_INTERVAL=300  # 5 minutes

# Backoff parameters
BACKOFF_INITIAL=5
BACKOFF_MAX=300
BACKOFF_FACTOR=2

# Track child PIDs for cleanup
COLLECTOR_PID=""
HEALTH_PID=""

cleanup() {
    echo ""
    echo "[run-collector] Shutting down..."
    [[ -n "$HEALTH_PID" ]] && kill "$HEALTH_PID" 2>/dev/null && wait "$HEALTH_PID" 2>/dev/null
    [[ -n "$COLLECTOR_PID" ]] && kill "$COLLECTOR_PID" 2>/dev/null && wait "$COLLECTOR_PID" 2>/dev/null
    echo "[run-collector] Clean shutdown complete."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Source .env
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
    echo "[run-collector] Loaded .env"
else
    echo "[run-collector] WARNING: No .env file found at $PROJECT_ROOT/.env"
fi

# Verify credentials
if [[ -z "${HA_URL:-}" ]] || [[ -z "${HA_TOKEN:-}" ]]; then
    echo "[run-collector] ERROR: HA_URL and HA_TOKEN must be set"
    exit 1
fi

# Start health monitor in background
health_monitor() {
    while true; do
        sleep "$HEALTH_INTERVAL"
        bash "$HEALTH_CHECK" 2>&1 | while IFS= read -r line; do
            echo "[health] $line"
        done
    done
}
health_monitor &
HEALTH_PID=$!
echo "[run-collector] Health monitor started (PID: $HEALTH_PID, interval: ${HEALTH_INTERVAL}s)"

# Restart loop with exponential backoff
backoff=$BACKOFF_INITIAL
attempt=0

while true; do
    attempt=$((attempt + 1))
    echo "[run-collector] Starting collector (attempt $attempt)..." | tee -a "$LOG_FILE"

    # Run the collector, teeing output to log and stdout
    cd "$HA_CLIENT_DIR"
    npx tsx src/index.ts collect 2>&1 | tee -a "$LOG_FILE" &
    COLLECTOR_PID=$!

    # Wait for collector to exit
    set +e
    wait "$COLLECTOR_PID"
    exit_code=$?
    set -e
    COLLECTOR_PID=""

    # If we were signalled (cleanup), exit immediately
    if [[ $exit_code -eq 130 ]] || [[ $exit_code -eq 143 ]]; then
        break
    fi

    echo "[run-collector] Collector exited with code $exit_code, restarting in ${backoff}s..." | tee -a "$LOG_FILE"
    sleep "$backoff"

    # Exponential backoff with cap
    backoff=$((backoff * BACKOFF_FACTOR))
    if [[ $backoff -gt $BACKOFF_MAX ]]; then
        backoff=$BACKOFF_MAX
    fi

    # Reset backoff if collector ran for >5 minutes (likely a real session, not immediate crash)
    # This is approximate — we just reset on each successful restart attempt
done
