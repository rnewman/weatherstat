# Weatherstat

Hysteresis-aware smart thermostat system for hydronic floor heat with massive thermal lag. Two components:

- **ha-client/** — TypeScript (pnpm, Node 25 native TS). Connects to Home Assistant to collect sensor snapshots and execute HVAC commands.
- **ml/** — Python (uv, hatchling). Physics-based control: system identification, forward simulation, trajectory sweep.

## Architecture

- HA interface is abstracted behind `HAClient` (types.ts). WebSocket is one implementation; add-on/integration would use HA internal API.
- Collector writes 5-min snapshots to SQLite (`~/.weatherstat/snapshots/snapshots.db`) in EAV format (`readings` table: timestamp, name, value). Python reader pivots to wide DataFrame at load time, applying types from config. Sysid reads this for parameter fitting. Control output goes to JSON in `~/.weatherstat/predictions/`.
- Weather forecasts come from HA's `weather.forecast_home` entity (met.no). The collector stores hourly forecast snapshots; the simulator uses live forecasts for piecewise outdoor temp integration.
- Entity IDs are real and live in `ha-client/src/entities.ts`. Full reference in `docs/entities.md`.
- Sensor-to-zone mapping is derived from the sysid coupling matrix (which thermostat has the highest gain for each sensor), not configured in YAML.

## Development Stages

1. **Pipeline & data** (done) — Collector running, LightGBM training on collector data, evaluation framework.
2. **Control loop** (done) — Setpoint sweep, comfort schedules, safety rails, dry-run + live execution.
3. **Data accumulation** (in progress) — Collector running since Feb 2026. All data is winter so far.
4. **System identification** (done) — `sysid.py` fits tau, effector×sensor gains, and solar profiles from collector data.
5. **Per-room models & blower control** (done) — 8 rooms × 5 horizons, blower fan speed in control sweep.
6. **Forecast training + HVAC features** (done) — Training uses stored met.no forecasts (no train/serve skew), retrospective HVAC duty cycle features.
7. **Effector inertia planning** (done) — Trajectory search for slow effectors, physics-only control (ML removed).
8. **Virtual effectors Phase 1** (done) — Physics-based window advisories integrated into control loop.
9. **Generalized architecture Phase 1** (done) — Sensor/effector/constraint model replaces room-centric config. Generic device health checks. Boiler column generalization. Humidity sensor expansion.
10. **Narrow storage Phase 2** (done) — EAV `readings` table is canonical storage. Legacy wide `snapshots` table and dual-write removed. No schema changes needed to add sensors.
11. **Learned window effects Phase 3** (done) — `TauModel` with per-window `window_betas` replaces binary sealed/ventilated tau. Sysid learns window cooling rate coefficients and cross-breeze interactions from regression.
12. **Derived zone mapping** (done) — Sensor-to-zone mapping derived from sysid coupling matrix (highest thermostat gain per sensor). Zone removed from constraints YAML and `ConstraintSchedule`. Legacy wide table dropped from collector.
13. **Mini split target temperature control** (done) — Mini splits treated as regulating effectors: sweep target temperatures (from comfort schedule `preferred`) instead of modes. Proportional activity model in simulator (activity ramps 0→1 as room deviates from target within `proportional_band`). Mode hold windows prevent noisy mode changes during quiet hours. Two-layer comfort cost: continuous quadratic from `preferred` + 10× steep penalty outside hard rails (min/max). Asymmetric cold/hot penalties.

See `docs/FUTURE.md` for the roadmap and `docs/plans/` for detailed plans.

## Commands

```bash
just                  # List available tasks
just init             # Initialize ~/.weatherstat data directory
just migrate          # Migrate data from repo data/ to ~/.weatherstat
just collect          # Run HA state collector (5-min loop)
just collect-once     # Collect a single snapshot
just collect-durable  # Auto-restart collector + health monitoring
just health           # Check if collector data is fresh
just extract          # Extract historical data from HA
just sysid            # System identification (fit thermal params from data)
just control          # Single control cycle (dry-run, physics trajectory sweep)
just control-loop     # 15-min control loop (dry-run)
just control-live     # Single control cycle with live execution
just execute          # Apply latest command JSON to HA
just comfort          # Comfort performance dashboard (last 7 days)
just lint             # Lint both packages
just test             # Test both packages
just typecheck        # TypeScript type-check
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

## Data Directory

Runtime data lives in `~/.weatherstat/` (override with `WEATHERSTAT_DATA_DIR` env var):

```
~/.weatherstat/
  weatherstat.yaml            # house config (from weatherstat.yaml.example)
  .env                        # HA credentials (HA_URL, HA_TOKEN)
  snapshots/snapshots.db      # collector output (SQLite EAV)
  predictions/command_*.json  # control decisions
  thermal_params.json         # sysid output
  decision_log.db             # control decision history
  control_state.json          # last decision state
  executor_state.json         # executor override tracking
  advisory_state.json         # window advisory cooldowns
```

First-time setup: `just init` creates the directory and copies `weatherstat.yaml.example`.
Migration from repo `data/`: `just migrate`.
