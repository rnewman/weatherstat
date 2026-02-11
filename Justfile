# Weatherstat task runner

# List available commands
default:
    @just --list

# Run HA state collector
collect:
    cd ha-client && npx tsx src/index.ts collect

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

# Install all dependencies
install:
    cd ha-client && pnpm install
    cd ml && uv sync
