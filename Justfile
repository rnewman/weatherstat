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

# Train LightGBM baseline model (hourly temp-only, 5+ months)
train-baseline:
    cd ml && uv run python -m weatherstat.train --mode baseline

# Train LightGBM full model (5-min all features, ~10 days)
train-full:
    cd ml && uv run python -m weatherstat.train --mode full

# Train both models
train: train-baseline train-full

# Retrain: re-extract data then train both models
retrain: extract train

# Evaluate and compare models
evaluate:
    cd ml && uv run python -m weatherstat.evaluate

# View training metrics (latest summary, history, or compare two runs)
metrics *ARGS:
    cd ml && uv run python -m weatherstat.metrics {{ARGS}}

# Visualize extracted data
visualize *ARGS:
    cd ml && uv run python -m weatherstat.visualize {{ARGS}}

# Predict: fetch live state from HA, predict with both models
predict:
    cd ml && uv run python -m weatherstat.inference

# Predict from collector snapshot files
predict-snapshot:
    cd ml && uv run python -m weatherstat.inference --snapshot

# Counterfactual: predict under HVAC on/off scenarios
counterfactual:
    cd ml && uv run python -m weatherstat.inference --counterfactual

# Lint both packages
lint: lint-ts lint-py

# Lint TypeScript
lint-ts:
    cd ha-client && pnpm exec tsc --noEmit

# Lint Python
lint-py:
    cd ml && uv run ruff check src/

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

# Single control cycle (dry-run)
control:
    cd ml && uv run python -m weatherstat.control

# Run control loop (dry-run, 15-min interval)
control-loop:
    cd ml && uv run python -m weatherstat.control --loop

# Single control cycle with live execution
control-live:
    cd ml && uv run python -m weatherstat.control --live

# Execute latest command JSON via HA
execute:
    cd ha-client && npx tsx src/index.ts execute

# ── Experiments ──────────────────────────────────────────────────────────

# Create a git worktree for an experiment branch (shares data/)
worktree NAME *PATH:
    bash scripts/worktree.sh {{NAME}} {{PATH}}

# Train both models into an experiment directory
train-experiment NAME:
    cd ml && uv run python -m weatherstat.train --mode baseline --experiment {{NAME}}
    cd ml && uv run python -m weatherstat.train --mode full --experiment {{NAME}}

# Train just the full model into an experiment directory
train-experiment-full NAME:
    cd ml && uv run python -m weatherstat.train --mode full --experiment {{NAME}}

# Compare an experiment's models against production
experiment-compare NAME:
    cd ml && uv run python -m weatherstat.experiment compare {{NAME}}

# List all experiments
experiments:
    cd ml && uv run python -m weatherstat.experiment list

# ── Retraining ─────────────────────────────────────────────────────────

# Run manual retrain (extract + train, with logging)
retrain-manual:
    bash scripts/run-retrain.sh

# Install weekly retrain launchd agent (Sunday 3 AM)
retrain-install:
    mkdir -p ~/Library/LaunchAgents
    cp com.twinql.weatherstat.retrain.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.twinql.weatherstat.retrain.plist
    @echo "Retrain agent installed (Sunday 3:00 AM)"

# Uninstall weekly retrain launchd agent
retrain-uninstall:
    -launchctl unload ~/Library/LaunchAgents/com.twinql.weatherstat.retrain.plist
    rm -f ~/Library/LaunchAgents/com.twinql.weatherstat.retrain.plist
    @echo "Retrain agent uninstalled"

# ── Setup ────────────────────────────────────────────────────────────────

# Install all dependencies
install:
    cd ha-client && pnpm install
    cd ml && uv sync
