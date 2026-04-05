# Weatherstat

Hysteresis-aware smart thermostat system for hydronic floor heat with massive thermal lag.

- **src/** — Python (uv, hatchling). All pipeline stages: snapshot collection, system identification, forward simulation, trajectory sweep, command execution, TUI dashboard.

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
16. **Comfort profiles & MRT correction** (done) — Named comfort profiles (Home/Away) controlled by HA `input_select.thermostat_mode`, with offset-based temperature adjustments. Mean radiant temperature correction uses outdoor temp as proxy for wall surface effects, now sun-aware: per-sensor effective outdoor temp is raised by current solar forcing (β_solar × sin⁺(elev) × weather_fraction × solar_response), reducing cold-wall correction on sunny days. Pipeline: base schedules → profile offsets → MRT correction → window adjustments. Navien connection health check (`expected_state` on binary sensor). See `docs/mrt-correction.md`.
17. **Persistent window opportunities** (done) — Fire-and-forget advisories replaced with persistent, energy-aware "opportunities" model. Two thresholds: opportunity (track in state) and notification (push to phone). Re-sweep: evaluates best HVAC plan with window toggled to capture energy savings (e.g., open window + turn off mini split). Lifecycle management: new opportunities added, still-valid kept, expired dismissed via `persistent_notification/dismiss`. Per-window notification IDs prevent stacking.
18. **Unified effector model** (done) — Single `EffectorDecision` type replaces `ThermostatTrajectory`, `BlowerDecision`, `MiniSplitDecision`. `EffectorConfig` with `control_type`/`mode_control`/`depends_on`/`command_keys` replaces per-type config classes. Scenario generation iterates `EFFECTORS` tuple from config with `itertools.product`. No hardcoded device names anywhere. TS executor iterates config dicts dynamically. Effector eligibility gate: pre-sweep check excludes manual-mode effectors that are off or whose state_device is unavailable. Dead code pruned: legacy advisory types, boiler backward-compat, LGBM params, encoding aliases.
19. **Smoothed derivative & gain recovery** (done) — 5-minute central differences amplified sensor noise (~10°F/hr) drowning thermostat signals (~0.3°F/hr). Smoothed derivative (15-min half-window rolling mean + wider central difference) reduces noise ~5×, recovering thermostat gains from 1/32 surviving to 25/32. Mode-direction sign filter prevents heating-only effectors from having negative gains. Per-sensor cost display bug fixed (key format mismatch). TUI comfort bars now reflect active comfort profile + MRT correction. See `docs/debugging-notes.md` § "Derivative Noise".
20. **Celsius support & configurable defaults** (done) — `unit: F` or `unit: C` in location block. All hardcoded temperature constants converted via `abs_temp()`/`delta_temp()` from canonical °F at load time. Control thresholds (`setpoint_min`, `setpoint_max`, `cautious_offset`, `max_1h_change`, `min_improvement`, `cold_room_override`) configurable in `defaults:` section. Display formatting uses `UNIT_SYMBOL` throughout. Dead-band preferred range: `preferred` can be a point or `[lo, hi]` range with zero cost inside the band; `preferred_widen` in profiles expands point targets into dead bands.
21. **Enriched decision logging & faster loops** (done) — Decision log enriched with `active_profile`, `mrt_offsets`, `blocked` columns. `comfort_bounds` now includes `preferred_lo/hi` and `cold/hot_penalty`. Fixed `_compute_actual_comfort_cost` key mismatch bug (was always returning 0.0). Control interval configurable (default 5 min, was 15 min). Sysid split into `fit_sysid()` + `save_sysid_result()` for quality-gated periodic refitting (default hourly in TUI). Comfort plotter uses historical decision bounds (reflects actual profiles/MRT/windows over time) and shows per-sensor control authority (% time system had full control).
22. **Gains-aware regulating sweep & solar irradiance collection** (done) — Regulating effectors (mini splits) now derive sweep options from all constrained sensors with meaningful sysid gain, not a single naming-convention sensor. Both heat and cool modes are offered with targets from affected sensors' preferred temps; idle suppression prunes the nonsensical direction; the trajectory scorer picks the winner. Fixed bug where `mini_split_living_room` could never activate (naming convention produced `living_room_temp` but comfort schedule was on `living_room_climate_temp`). Solar irradiance data collection via forecast.solar HA integration: 5 planes (horizontal + cardinal walls) at 1kWp, collecting W data for future irradiance-based solar model to replace per-hour sysid coefficients × weather-condition fraction.
23. **Elevation-based solar model & cloud coverage collection** (done) — Replaced 11 per-hour solar features (one per hour 7–17) with a single continuous feature: `sin⁺(solar_elevation) × weather_fraction`. Solar elevation computed analytically from lat/lon/timestamp (Spencer 1971 declination). One regression coefficient per sensor instead of 11. Automatically captures seasonal variation (Feb noon sin⁺=0.49, Apr=0.74, Jun=0.91 at Seattle) — the old model underpredicted spring solar gain by ~35% because winter-fitted coefficients had no seasonal awareness. Simulator precomputes `solar_elevations` at 5-min resolution for the prediction horizon. Backward-compatible: falls back to legacy per-hour profiles when `solar_elevation_gains` is absent from thermal_params.json. Collector now stores `cloud_coverage` (0–100%) from weather entity and `forecast_cloud_{h}h` at key horizons for future continuous solar fraction.
24. **Sun-aware MRT correction** (done) — MRT correction now uses current solar state to differentiate sunny vs cloudy days at the same outdoor temp. Per-sensor effective outdoor temp: `effective_outdoor = outdoor + β_solar × sin⁺(elev) × weather_fraction × solar_response`. High-solar-gain rooms (e.g., piano with large windows) get less cold-wall correction on sunny days because sun streaming through windows heats interior surfaces. `solar_response` configurable in MRT config (default 2.0). Static sysid-derived `mrt_weights` removed — dynamic solar calculation replaces them. Manual `mrt_weight` in YAML constraint schedules still applies as a multiplier for non-solar overrides.

See `docs/FUTURE.md` for the roadmap and `docs/plans/` for detailed plans.

## Commands

```bash
just                  # List available tasks
just init             # Initialize ~/.weatherstat data directory
just collect          # Run HA state collector (5-min loop, auto-recovery)
just collect-once     # Collect a single snapshot
just health           # Check if collector data is fresh
just sysid            # System identification (fit thermal params from data)
just control          # Single control cycle (dry-run)
just control --live   # Single cycle with live execution + HA commands
just control --loop   # 5-min control loop (dry-run)
just control --live --loop  # Live control loop (production)
just execute          # Apply latest command JSON to HA
just execute --force  # Apply ignoring manual overrides
just tui              # Interactive TUI dashboard
just tui --live       # TUI starting in live mode
just comfort          # Comfort performance dashboard (last 7 days)
just lint             # Lint both packages
just lint-fix         # Lint and fix both packages
just test             # Test both packages
```

## Key Documentation

- Entity IDs: `weatherstat.yaml` (configured per-house)
- Domain types: `src/weatherstat/types.py`
- System identification: `src/weatherstat/sysid.py`
- Collector: `src/weatherstat/collector.py`
- Executor: `src/weatherstat/executor.py`

## Conventions

- Python: ruff for linting/formatting, frozen dataclasses, StrEnum for enums
- All source files use explicit types — full type hints in Python
- Temperature unit configurable via `unit` in weatherstat.yaml location block (F or C). Built-in defaults are canonical °F, converted at load time via `abs_temp()`/`delta_temp()`. All runtime values are in the configured unit.
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
