# Weatherstat

Hysteresis-aware smart thermostat system for hydronic floor heat with massive thermal lag. Two components:

- **ha-client/** — TypeScript (pnpm, Node 25 native TS). Connects to Home Assistant to collect sensor snapshots and execute HVAC predictions.
- **ml/** — Python (uv, hatchling). LightGBM training/inference pipeline.

## Architecture

- HA interface is abstracted behind `HAClient` (types.ts). WebSocket is one implementation; add-on/integration would use HA internal API.
- Communication between TS and Python is file-based: Parquet snapshots in `data/snapshots/`, JSON predictions in `data/predictions/`.
- Weather data comes from HA's `weather.forecast_home` entity (met.no) — same source for training and inference (no feature skew).
- Feature engineering lives in `ml/src/weatherstat/features.py` and is shared by both training and inference.
- Entity IDs are real and live in `ha-client/src/entities.ts`. Full reference in `docs/entities.md`.

## Development Stages

This project develops iteratively. Each stage builds on the previous:

1. **Data collection** (Stage 1A) — Get the collector running ASAP. Every day of missed data is unrecoverable. Extract historical data from HA.
2. **Basic modeling** (Stage 1B-C) — Train baseline thermal-dynamics model on 5+ months of hourly temp data, and a proof-of-concept HVAC-aware model on ~10 days of full-feature data.
3. **Multi-horizon prediction** (Stage 1D) — Predict temperature at T+1h, T+2h, T+4h per zone. Compare baseline vs HVAC-augmented model.
4. **Evaluation** — Assess predictions against physical expectations. Does the model capture thermal dynamics? Do HVAC features add signal?
5. **Iterate** — Retrain weekly as data accumulates. Revisit feature engineering and model architecture after 4-8 weeks. Seasonal coverage grows over months.

Current stage: **Stage 1 complete (pipeline built)**. Next: extract real data, train, evaluate, start collector for ongoing collection.

## Commands

```bash
just                  # List available tasks
just collect          # Run HA state collector (5-min snapshots)
just extract          # Extract historical data from HA
just train            # Train both baseline and full models
just train-baseline   # Train hourly temp-only model (5+ months)
just train-full       # Train 5-min full-feature model (~10 days)
just evaluate         # Compare baseline vs full model
just retrain          # Re-extract + retrain everything
just infer            # Run inference pipeline
just lint             # Lint both packages
just test             # Test both packages
just typecheck        # TypeScript type-check
```

## Key Documentation

- `docs/entities.md` — Full HA entity reference organized by zone
- Entity IDs: `ha-client/src/entities.ts`
- Snapshot schema: `ha-client/src/types.ts` (SnapshotRow interface)
- Python mirror: `ml/src/weatherstat/types.py`

## Conventions

- TypeScript: strict mode, ESM, no default exports
- Python: ruff for linting/formatting, frozen dataclasses, StrEnum for enums
- All source files use explicit types — no `any` in TS, full type hints in Python
- Temperatures in Fahrenheit (matching HA configuration)
- Parquet column names use snake_case (matching Python conventions)
- TS interface properties use camelCase (matching TS conventions)

## Environment

HA credentials in `.env` (not committed):
```
HA_URL=https://ha.example.com
HA_TOKEN=<long-lived access token>
```

Both the TS client and Python extraction script read from these env vars.
