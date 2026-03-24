# Weatherstat task runner

# List available commands
default:
    @just --list

# Run HA state collector (5-min loop)
collect:
    cd ha-client && npx tsx src/index.ts collect

# Collect a single snapshot
collect-once:
    cd ha-client && npx tsx src/index.ts once

# Run collector with auto-restart + health monitoring
collect-durable:
    bash scripts/run-collector.sh

# Check collector health (is data fresh?)
health:
    bash scripts/check-health.sh

# Extract historical data from HA
extract *ARGS:
    cd ml && uv run python -m weatherstat.extract {{ARGS}}

# Lint both packages
lint: lint-ts lint-py

# Lint TypeScript
lint-ts:
    cd ha-client && pnpm exec tsc --noEmit

# Lint Python
lint-py:
    cd ml && uv run ruff check src/

lint-fix:
    uv run ruff check --fix ml/src/ ml/tests/

# Format Python
fmt:
    cd ml && uv run ruff format src/ tests/

# Test both packages
test: test-ts test-py

# Test TypeScript (placeholder)
test-ts:
    @echo "No TS tests yet"

# Test Python
test-py:
    cd ml && uv run pytest tests/

# TypeScript type-check
typecheck:
    cd ha-client && pnpm exec tsc --noEmit

# Single control cycle (dry-run, physics trajectory sweep)
control:
    cd ml && uv run python -m weatherstat.control

# Run control loop (dry-run, 15-min interval)
control-loop:
    cd ml && uv run python -m weatherstat.control --loop

# Single control cycle with live execution
control-live:
    cd ml && uv run python -m weatherstat.control --live

# Live control loop (15-min interval, generates + executes via HA)
control-loop-live:
    #!/usr/bin/env bash
    set -uo pipefail
    REPO_ROOT="$(git rev-parse --show-toplevel)"
    echo "[control-loop-live] Starting (15-min interval, Ctrl+C to stop)"
    while true; do
        echo ""
        echo "── Control cycle: $(date) ──"
        if cd "$REPO_ROOT/ml" && uv run python -m weatherstat.control --live; then
            cd "$REPO_ROOT"
            echo "[control-loop-live] Executing command via HA..."
            cd ha-client && npx tsx src/index.ts execute; cd "$REPO_ROOT"
        else
            cd "$REPO_ROOT"
            echo "[control-loop-live] Control cycle failed (HA down?), will retry next cycle"
        fi
        echo "[control-loop-live] Next cycle in 15 minutes..."
        sleep 900
    done

# Execute latest command JSON via HA
execute:
    cd ha-client && npx tsx src/index.ts execute

# Execute latest command, ignoring manual overrides
execute-force:
    cd ha-client && npx tsx src/index.ts execute --force

# System identification: extract thermal parameters from collector data
sysid *ARGS:
    cd ml && uv run python -m weatherstat.sysid {{ARGS}}

# Comfort performance dashboard (last 7 days by default)
comfort *ARGS:
    cd ml && uv run python ../scripts/plot_comfort.py {{ARGS}}

# Verify live config parses correctly (both TS and Python)
verify:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Verifying weatherstat.yaml..."
    node -e "const { config } = require('./ha-client/src/yaml-config.ts'); console.log('  TS: OK (' + Object.keys(config.effectors).length + ' effectors, ' + config.columnDefs.length + ' columns)')"
    cd ml && uv run python -c "from weatherstat.yaml_config import load_config; cfg = load_config(); print(f'  Python: OK ({len(cfg.effectors)} effectors, {len(cfg.constraints)} constraints)'); from weatherstat.config import EFFECTORS; print(f'  Config: OK ({len(EFFECTORS)} effectors in EFFECTORS)')"
    echo "Config OK."

# ── Setup ────────────────────────────────────────────────────────────────

# Initialize ~/.weatherstat data directory
init:
    #!/usr/bin/env bash
    set -euo pipefail
    DATA_DIR="${WEATHERSTAT_DATA_DIR:-$HOME/.weatherstat}"
    echo "Initializing data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR/snapshots"
    mkdir -p "$DATA_DIR/predictions"
    mkdir -p "$DATA_DIR/models"
    mkdir -p "$DATA_DIR/metrics"
    if [[ ! -f "$DATA_DIR/weatherstat.yaml" ]]; then
        if [[ -f "weatherstat.yaml.example" ]]; then
            cp weatherstat.yaml.example "$DATA_DIR/weatherstat.yaml"
            echo "Copied weatherstat.yaml.example -> $DATA_DIR/weatherstat.yaml"
            echo "  Edit $DATA_DIR/weatherstat.yaml with your entity IDs"
        else
            echo "ERROR: weatherstat.yaml.example not found in repo root"
            exit 1
        fi
    else
        echo "Config exists: $DATA_DIR/weatherstat.yaml"
    fi
    if [[ ! -f "$DATA_DIR/.env" ]]; then
        echo "WARNING: No .env at $DATA_DIR/.env — create with HA_URL and HA_TOKEN"
    else
        echo "Env exists: $DATA_DIR/.env"
    fi
    echo "Ready: $DATA_DIR"

# Install all dependencies
install:
    cd ha-client && pnpm install
    cd ml && uv sync
