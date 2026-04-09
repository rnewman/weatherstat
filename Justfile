# Weatherstat task runner

# List available commands
default:
    @just --list

# Run HA state collector (5-min loop, auto-recovery)
collect:
    uv run python -u -m weatherstat.collector collect

# Collect a single snapshot
collect-once:
    uv run python -m weatherstat.collector once

# Check collector health (is data fresh?)
health:
    bash scripts/check-health.sh


# Lint Python
lint:
    uv run ruff check src/

lint-fix:
    uv run ruff check --fix src/ tests/

# Format Python
fmt:
    uv run ruff format src/ tests/

# Test Python
test:
    uv run pytest tests/

# Control cycle (flags: --live, --loop)
control *ARGS:
    uv run python -m weatherstat.control {{ARGS}}

# Execute latest command JSON via HA (flag: --force to ignore overrides)
execute *ARGS:
    uv run python -m weatherstat.executor {{ARGS}}

# Discover HA entities and generate starter config
discover *ARGS:
    uv run python ../scripts/discover.py {{ARGS}}

# System identification: extract thermal parameters from collector data
sysid *ARGS:
    uv run python -m weatherstat.sysid {{ARGS}}

# Interactive TUI dashboard
tui *ARGS:
    uv run --extra tui python -m weatherstat.tui {{ARGS}}

# Comfort performance dashboard (last 7 days by default)
comfort *ARGS:
    uv run python ./scripts/plot_comfort.py {{ARGS}}

# Debug inspector (subcommands: temps, gains, taus, decisions, opportunities, snapshots, comfort; --bundle <dir> for bundles)
debug *ARGS:
    uv run python ./scripts/debug_state.py {{ARGS}}

# Replay an environment opportunity evaluation (args: <timestamp> <env_name>)
replay *ARGS:
    uv run python ./scripts/replay_opportunity.py {{ARGS}}

# Create or replay a point-in-time debug bundle (args: [timestamp], --replay <dir>, --list)
bundle *ARGS:
    uv run python ./scripts/snapshot_bundle.py {{ARGS}}

# Verify live config parses correctly
verify:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Verifying weatherstat.yaml..."
    uv run python -c "from weatherstat.yaml_config import load_config; cfg = load_config(); print(f'  Config: OK ({len(cfg.effectors)} effectors, {len(cfg.constraints)} constraints)'); from weatherstat.config import EFFECTORS; print(f'  Effectors: OK ({len(EFFECTORS)} in EFFECTORS)')"
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

# Install all dependencies (including dev tools: pytest, ruff)
install:
    uv sync --extra dev
