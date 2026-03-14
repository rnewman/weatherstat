#!/usr/bin/env bash
# Weekly retraining wrapper.
#
# - Sources .env for HA_URL/HA_TOKEN
# - Runs `just retrain` (train full model from collector data)
# - Logs to logs/retrain-YYYY-MM-DD.log
# - Single run, exits (designed for launchd scheduling)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
TODAY="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/retrain-${TODAY}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

log() {
    echo "[retrain $(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

# Source .env from data directory
WEATHERSTAT_DATA_DIR="${WEATHERSTAT_DATA_DIR:-$HOME/.weatherstat}"
export WEATHERSTAT_DATA_DIR
if [[ -f "$WEATHERSTAT_DATA_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$WEATHERSTAT_DATA_DIR/.env"
    set +a
    log "Loaded $WEATHERSTAT_DATA_DIR/.env"
else
    log "WARNING: No .env file found at $WEATHERSTAT_DATA_DIR/.env"
fi

# Verify credentials
if [[ -z "${HA_URL:-}" ]] || [[ -z "${HA_TOKEN:-}" ]]; then
    log "ERROR: HA_URL and HA_TOKEN must be set"
    exit 1
fi

log "Starting weekly retrain..."
log "Project: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Run the retrain pipeline (train full model from collector data)
if just retrain 2>&1 | tee -a "$LOG_FILE"; then
    log "Retrain completed successfully"
else
    exit_code=$?
    log "Retrain failed with exit code $exit_code"
    exit $exit_code
fi

log "Done. Metrics saved to $WEATHERSTAT_DATA_DIR/metrics/"
