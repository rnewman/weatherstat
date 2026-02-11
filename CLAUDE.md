# Weatherstat

Hysteresis-aware smart thermostat system. Two components:

- **ha-client/** — TypeScript (pnpm, Node 25 native TS). Connects to Home Assistant to collect sensor snapshots and execute HVAC predictions.
- **ml/** — Python (uv, hatchling). LightGBM training/inference pipeline.

## Architecture

- HA interface is abstracted behind `HAClient` (types.ts). WebSocket is one implementation; add-on/integration would use HA internal API.
- Communication between TS and Python is file-based: Parquet snapshots in `data/snapshots/`, JSON predictions in `data/predictions/`.
- Weather data comes from HA's `weather.*` entity — same source for training and inference (no feature skew).
- Feature engineering lives in `ml/src/weatherstat/features.py` and is shared by both training and inference.

## Commands

```bash
just                  # List available tasks
just collect          # Run HA state collector
just train            # Train LightGBM model
just infer            # Run inference
just lint             # Lint both packages
just test             # Test both packages
just typecheck        # TypeScript type-check
```

## Conventions

- TypeScript: strict mode, ESM, no default exports
- Python: ruff for linting/formatting, frozen dataclasses, StrEnum for enums
- All source files use explicit types — no `any` in TS, full type hints in Python
- Entity IDs in `ha-client/src/entities.ts` are placeholders — fill in real ones from your HA instance
