# Weatherstat task runner

# List available commands
default:
    @just --list

# Run HA state collector
collect:
    cd ha-client && npx tsx src/index.ts collect

# Train LightGBM model
train:
    cd ml && uv run python -m weatherstat.train

# Run inference pipeline
infer:
    cd ml && uv run python -m weatherstat.inference

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
