# Weatherstat

Hysteresis-aware smart thermostat system for hydronic floor heat with massive thermal lag.

- **ml/** — Python (uv, hatchling). All pipeline stages: snapshot collection, system identification, forward simulation, trajectory sweep, command execution, TUI dashboard.

## Architecture

- Collector fetches HA entity states via REST API and writes 5-min snapshots to SQLite (`~/.weatherstat/snapshots/snapshots.db`) in EAV format (`readings` table: timestamp, name, value). Python reader pivots to wide DataFrame at load time, applying types from config. Sysid reads this for parameter fitting. Control output goes to JSON in `~/.weatherstat/predictions/`.
- Executor reads command JSON and applies HVAC commands via HA REST API, with lazy execution (skip if already correct) and override detection (respect manual changes).
- Weather forecasts come from HA's `weather.forecast_home` entity (met.no). The collector stores hourly forecast snapshots; the simulator uses live forecasts for piecewise outdoor temp integration.
- Entity IDs are configured in `weatherstat.yaml`. Full reference in `docs/entities.md`.
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
14. **Gain filtering & decision rationale** (done) — t-statistic threshold (|t| ≥ 1.5) and magnitude cap (≤ 3.0°F/hr) prune confounded/implausible sysid gains at simulator load time. Mode-direction clamp prevents heating from cooling or cooling from warming. Selectively standardized ridge regression (solar/window features scaled by std, effectors unscaled) shrinks confounded gains while preserving solar gain estimates. Counterfactual per-device attribution in control output (simulate winning scenario minus each active device). Per-sensor comfort cost breakdown. Outdoor temp for simulator uses weather forecast instead of solar-heated side sensor. Mini split heat/cool mode derived from room temp vs preferred, not outdoor temp. Test sandbox uses example YAML + synthetic thermal params (no live data dependency).
15. **Boiler to state sensor** (done) — Boiler removed from effectors. Heating mode is now a `sensors.state` entry (`navien_heating`) with an encoding — a categorical sensor, not an effector. Thermostat `state_device` references the state sensor for delivery confirmation. `_REGRESSION_SKIP_TYPES` eliminated (no boiler in regression). Health checks moved to standalone `health` YAML section. Sysid outputs `state_gates` for simulator to confirm thermostat history. Backward compat: old `effectors.boiler` YAML auto-synthesizes state sensor. Power sensor (`navien_gas_usage`) added under `sensors.power`.
16. **Comfort profiles & MRT correction** (done) — Named comfort profiles (Home/Away) controlled by HA `input_select.thermostat_mode`, with offset-based temperature adjustments. Mean radiant temperature correction uses outdoor temp as proxy for wall surface effects: `offset = clamp(alpha × (ref - outdoor), -max, +max)` shifts comfort targets to compensate for cold/warm surfaces. Per-sensor `mrt_weight` multiplier (configurable in YAML, derived from sysid solar profiles). Pipeline: base schedules → profile offsets → MRT correction → window adjustments. Navien connection health check (`expected_state` on binary sensor). See `docs/mrt-correction.md`.
17. **Persistent window opportunities** (done) — Fire-and-forget advisories replaced with persistent, energy-aware "opportunities" model. Two thresholds: opportunity (track in state) and notification (push to phone). Re-sweep: evaluates best HVAC plan with window toggled to capture energy savings (e.g., open window + turn off mini split). Lifecycle management: new opportunities added, still-valid kept, expired dismissed via `persistent_notification/dismiss`. Per-window notification IDs prevent stacking.
18. **Unified effector model** (done) — Single `EffectorDecision` type replaces `ThermostatTrajectory`, `BlowerDecision`, `MiniSplitDecision`. `EffectorConfig` with `control_type`/`mode_control`/`depends_on`/`command_keys` replaces per-type config classes. Scenario generation iterates `EFFECTORS` tuple from config with `itertools.product`. No hardcoded device names anywhere. TS executor iterates config dicts dynamically. Effector eligibility gate: pre-sweep check excludes manual-mode effectors that are off or whose state_device is unavailable. Dead code pruned: legacy advisory types, boiler backward-compat, LGBM params, encoding aliases.

See `docs/FUTURE.md` for the roadmap and `docs/plans/` for detailed plans.

## Commands

```bash
just                  # List available tasks
just init             # Initialize ~/.weatherstat data directory
just collect          # Run HA state collector (5-min loop, auto-recovery)
just collect-once     # Collect a single snapshot
just health           # Check if collector data is fresh
just sysid            # System identification (fit thermal params from data)
just control          # Single control cycle (dry-run, physics trajectory sweep)
just control-loop     # 15-min control loop (dry-run)
just control-live     # Single control cycle with live execution
just control-loop-live # Loop control cycle with live execution
just execute          # Apply latest command JSON to HA
just execute-force    # Apply ignoring manual overrides
just tui              # Interactive TUI dashboard
just comfort          # Comfort performance dashboard (last 7 days)
just lint             # Lint both packages
just lint-fix         # Lint and fix both packages
just test             # Test both packages
```

## Key Documentation

- Entity IDs: `weatherstat.yaml` (configured per-house)
- Domain types: `ml/src/weatherstat/types.py`
- System identification: `ml/src/weatherstat/sysid.py`
- Collector: `ml/src/weatherstat/collector.py`
- Executor: `ml/src/weatherstat/executor.py`

## Conventions

- Python: ruff for linting/formatting, frozen dataclasses, StrEnum for enums
- All source files use explicit types — full type hints in Python
- Temperatures in Fahrenheit (matching HA configuration)
- Snapshot column names use snake_case

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
