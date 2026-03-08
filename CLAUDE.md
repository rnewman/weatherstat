# Weatherstat

Hysteresis-aware smart thermostat system for hydronic floor heat with massive thermal lag. Two components:

- **ha-client/** — TypeScript (pnpm, Node 25 native TS). Connects to Home Assistant to collect sensor snapshots and execute HVAC predictions.
- **ml/** — Python (uv, hatchling). LightGBM training/inference pipeline.

## Architecture

- HA interface is abstracted behind `HAClient` (types.ts). WebSocket is one implementation; add-on/integration would use HA internal API.
- Collector writes 5-min snapshots to SQLite (`data/snapshots/snapshots.db`). Training uses collector data exclusively. Control/prediction output goes to JSON in `data/predictions/`.
- Weather data comes from HA's `weather.forecast_home` entity (met.no) — same source for training and inference (no feature skew).
- Feature engineering lives in `ml/src/weatherstat/features.py` and is shared by both training and inference.
- Entity IDs are real and live in `ha-client/src/entities.ts`. Full reference in `docs/entities.md`.
- Collector SQLite schema matches the Parquet snapshot schema (same columns, snake_case). Only difference: `any_window_open` is INTEGER in SQLite vs bool in Parquet (normalized at load time).

## Development Stages

1. **Pipeline & data** (done) — Collector running, LightGBM training on collector data, evaluation framework.
2. **Control loop** (done) — Setpoint sweep, comfort schedules, safety rails, dry-run + live execution.
3. **Data accumulation** (in progress) — Collector running since Feb 2026. Retrain weekly. All data is winter so far.
4. **System identification** (done) — `sysid.py` fits tau, effector×sensor gains, and solar profiles from collector data.
5. **Per-room models & blower control** (done) — 8 rooms × 5 horizons, blower fan speed in control sweep.
6. **Forecast training + HVAC features** (done) — Training uses stored met.no forecasts (no train/serve skew), retrospective HVAC duty cycle features.
7. **Effector inertia planning** (done) — Trajectory search for slow effectors, physics-only control (ML removed).

See `docs/FUTURE.md` for the roadmap and `docs/plans/` for detailed plans.

## Commands

```bash
just                  # List available tasks
just collect          # Run HA state collector (5-min loop)
just collect-once     # Collect a single snapshot
just collect-durable  # Auto-restart collector + health monitoring
just health           # Check if collector data is fresh
just extract          # Extract historical data from HA
just train            # Train LightGBM models on collector data
just evaluate         # Evaluate model on validation set
just retrain          # Retrain from collector data
just predict          # Fetch live state, predict with full model
just control          # Single control cycle (dry-run, physics trajectory sweep)
just control-loop     # 15-min control loop (dry-run)
just control-live     # Single control cycle with live execution
just execute          # Apply latest command JSON to HA
just sysid            # System identification (fit thermal params from data)
just lint             # Lint both packages
just test             # Test both packages
just typecheck        # TypeScript type-check

# Experiments
just worktree NAME    # Create git worktree for experiment branch
just train-experiment NAME      # Train models to data/models/{NAME}/
just experiment-compare NAME    # Compare experiment vs production
just experiments      # List all experiments
```

## Key Documentation

- `docs/entities.md` — Full HA entity reference organized by zone
- Entity IDs: `ha-client/src/entities.ts`
- Snapshot schema: `ha-client/src/types.ts` (SnapshotRow interface)
- Python mirror: `ml/src/weatherstat/types.py`
- System identification: `ml/src/weatherstat/sysid.py`

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
