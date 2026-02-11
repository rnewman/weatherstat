#!/usr/bin/env bash
# Standalone health check for the collector.
# Queries snapshots.db for the latest timestamp and alerts if stale.
#
# Exit 0 = fresh, Exit 1 = stale or error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DB_PATH="${DB_PATH:-$PROJECT_ROOT/data/snapshots/snapshots.db}"
STALE_MINUTES="${STALE_MINUTES:-10}"

if [[ ! -f "$DB_PATH" ]]; then
    echo "ERROR: Database not found at $DB_PATH"
    exit 1
fi

# Get the latest timestamp from SQLite
latest_ts=$(sqlite3 "$DB_PATH" "SELECT MAX(timestamp) FROM snapshots;" 2>/dev/null)

if [[ -z "$latest_ts" ]]; then
    echo "ERROR: No snapshots in database"
    exit 1
fi

# Convert to epoch seconds
latest_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${latest_ts%%.*}" "+%s" 2>/dev/null || \
    date -d "${latest_ts}" "+%s" 2>/dev/null || echo "0")

now_epoch=$(date "+%s")
age_seconds=$((now_epoch - latest_epoch))
age_minutes=$((age_seconds / 60))

if [[ $age_minutes -ge $STALE_MINUTES ]]; then
    msg="Collector stale: last snapshot ${age_minutes}m ago (${latest_ts})"
    echo "WARNING: $msg"
    # macOS notification
    if command -v osascript &>/dev/null; then
        osascript -e "display notification \"$msg\" with title \"Weatherstat\" sound name \"Basso\"" 2>/dev/null || true
    fi
    exit 1
else
    echo "OK: last snapshot ${age_minutes}m ago (${latest_ts})"
    exit 0
fi
