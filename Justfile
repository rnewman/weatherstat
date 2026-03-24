# Weatherstat task runner

# List available commands
default:
    @just --list

# Run HA state collector (5-min loop, auto-recovery)
collect:
    cd ml && uv run python -u -m weatherstat.collector collect

# Collect a single snapshot
collect-once:
    cd ml && uv run python -m weatherstat.collector once

# Check collector health (is data fresh?)
health:
    bash scripts/check-health.sh


# Lint Python
lint:
    cd ml && uv run ruff check src/

lint-fix:
    uv run ruff check --fix ml/src/ ml/tests/

# Format Python
fmt:
    cd ml && uv run ruff format src/ tests/

# Test Python
test:
    cd ml && uv run pytest tests/

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
            cd "$REPO_ROOT/ml"
            echo "[control-loop-live] Executing command via HA..."
            uv run python -m weatherstat.executor
        else
            cd "$REPO_ROOT"
            echo "[control-loop-live] Control cycle failed (HA down?), will retry next cycle"
        fi
        echo "[control-loop-live] Next cycle in 15 minutes..."
        sleep 900
    done

# Execute latest command JSON via HA
execute:
    cd ml && uv run python -m weatherstat.executor

# Execute latest command, ignoring manual overrides
execute-force:
    cd ml && uv run python -m weatherstat.executor --force

# Discover HA entities and generate starter config
discover *ARGS:
    cd ml && uv run python ../scripts/discover.py {{ARGS}}

# System identification: extract thermal parameters from collector data
sysid *ARGS:
    cd ml && uv run python -m weatherstat.sysid {{ARGS}}

# Interactive TUI dashboard
tui:
    cd ml && uv run --extra tui python -m weatherstat.tui

# Comfort performance dashboard (last 7 days by default)
comfort *ARGS:
    cd ml && uv run python ../scripts/plot_comfort.py {{ARGS}}

# Verify live config parses correctly
verify:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Verifying weatherstat.yaml..."
    cd ml && uv run python -c "from weatherstat.yaml_config import load_config; cfg = load_config(); print(f'  Config: OK ({len(cfg.effectors)} effectors, {len(cfg.constraints)} constraints)'); from weatherstat.config import EFFECTORS; print(f'  Effectors: OK ({len(EFFECTORS)} in EFFECTORS)')"
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
    cd ml && uv sync
