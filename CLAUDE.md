# Weatherstat

Hysteresis-aware smart thermostat system for hydronic floor heat with massive thermal lag.

- **src/** — Python (uv, hatchling). All pipeline stages: snapshot collection, system identification, forward simulation, trajectory sweep, command execution, TUI dashboard.

## Architecture

- Collector fetches HA entity states via REST API and writes 5-min snapshots to SQLite (`~/.weatherstat/snapshots/snapshots.db`) in EAV format (`readings` table: timestamp, name, value). Python reader pivots to wide DataFrame at load time, applying types from config. Sysid reads this for parameter fitting. Control output goes to JSON in `~/.weatherstat/predictions/`.
- Executor reads command JSON and applies HVAC commands via HA REST API, with lazy execution (skip if already correct) and override detection (respect manual changes).
- Weather forecasts come from HA's `weather.forecast_home` entity (met.no). The collector stores hourly forecast snapshots; the simulator uses live forecasts for piecewise outdoor temp integration.
- Entity IDs are configured in `weatherstat.yaml`.
- Sensor-to-zone mapping is derived from the sysid coupling matrix (which thermostat has the highest gain for each sensor), not configured in YAML.

## Development Stages

1. **Pipeline & data** (done) — Collector running, LightGBM training on collector data, evaluation framework.
2. **Control loop** (done) — Setpoint sweep, comfort schedules, safety rails, dry-run + live execution.
3. **Data accumulation** (done) — Collector running since Feb 2026. Winter + spring data.
4. **System identification** (done) — `sysid.py` fits tau, effector×sensor gains, and solar profiles from collector data.
5. **Per-room models & blower control** (done) — 8 rooms × 5 horizons, blower fan speed in control sweep.
6. **Forecast training + HVAC features** (done) — Training uses stored met.no forecasts (no train/serve skew), retrospective HVAC duty cycle features.
7. **Effector inertia planning** (done) — Trajectory search for slow effectors, physics-only control (ML removed).
8. **Virtual effectors Phase 1** (done) — Physics-based window advisories integrated into control loop.
9. **Generalized architecture Phase 1** (done) — Sensor/effector/constraint model replaces room-centric config. Generic device health checks. Boiler column generalization. Humidity sensor expansion.
10. **Narrow storage Phase 2** (done) — EAV `readings` table is canonical storage. Legacy wide `snapshots` table and dual-write removed. No schema changes needed to add sensors.
11. **Learned window effects Phase 3** (done) — `TauModel` with per-entry tau betas replaces binary sealed/ventilated tau. Sysid learns environment factor cooling rate coefficients and interaction terms from regression.
12. **Derived zone mapping** (done) — Sensor-to-zone mapping derived from sysid coupling matrix (highest thermostat gain per sensor). Zone removed from constraints YAML and `ConstraintSchedule`. Legacy wide table dropped from collector.
13. **Mini split target temperature control** (done) — Mini splits treated as regulating effectors: sweep target temperatures (from comfort schedule `preferred`) instead of modes. Proportional activity model in simulator (activity ramps 0→1 as room deviates from target within `proportional_band`). Mode hold windows prevent noisy mode changes during quiet hours. Two-layer comfort cost: continuous quadratic from `preferred` + 10× steep penalty outside hard rails (min/max). Asymmetric cold/hot penalties.
14. **Gain filtering & decision rationale** (done) — t-statistic threshold (|t| ≥ 1.5) and magnitude cap (≤ 3.0°F/hr) prune confounded/implausible sysid gains at simulator load time. Mode-direction clamp prevents heating from cooling or cooling from warming. Selectively standardized ridge regression (solar/environment features scaled by std, effectors unscaled) shrinks confounded gains while preserving solar gain estimates. Counterfactual per-device attribution in control output (simulate winning scenario minus each active device). Per-sensor comfort cost breakdown. Outdoor temp for simulator uses weather forecast instead of solar-heated side sensor. Mini split heat/cool mode derived from room temp vs preferred, not outdoor temp. Test sandbox uses example YAML + synthetic thermal params (no live data dependency).
15. **Boiler to state sensor** (done) — Boiler removed from effectors. Heating mode is now a `sensors.state` entry (`navien_heating`) with an encoding — a categorical sensor, not an effector. Thermostat `state_device` references the state sensor for delivery confirmation. `_REGRESSION_SKIP_TYPES` eliminated (no boiler in regression). Health checks moved to standalone `health` YAML section. Sysid outputs `state_gates` for simulator to confirm thermostat history. Power sensor (`navien_gas_usage`) added under `sensors.power`.
16. **Comfort profiles & MRT correction** (done) — Named comfort profiles (Home/Away) controlled by HA `input_select.thermostat_mode`, with offset-based temperature adjustments. Mean radiant temperature correction uses outdoor temp as proxy for wall surface effects, now sun-aware: per-sensor effective outdoor temp is raised by current solar forcing (β_solar × sin⁺(elev) × weather_fraction × solar_response), reducing cold-wall correction on sunny days. Pipeline: base schedules → profile offsets → MRT correction → environment adjustments. Navien connection health check (`expected_state` on binary sensor). See `docs/mrt-correction.md`.
17. **Persistent window opportunities** (done) — Fire-and-forget advisories replaced with persistent, energy-aware "opportunities" model. Two thresholds: opportunity (track in state) and notification (push to phone). Re-sweep: evaluates best HVAC plan with window toggled to capture energy savings (e.g., open window + turn off mini split). Lifecycle management: new opportunities added, still-valid kept, expired dismissed via `persistent_notification/dismiss`. Per-window notification IDs prevent stacking.
18. **Unified effector model** (done) — Single `EffectorDecision` type replaces `ThermostatTrajectory`, `BlowerDecision`, `MiniSplitDecision`. `EffectorConfig` with `control_type`/`mode_control`/`depends_on`/`command_keys` replaces per-type config classes. Scenario generation iterates `EFFECTORS` tuple from config with `itertools.product`. No hardcoded device names anywhere. Executor iterates config dicts dynamically. Effector eligibility gate: pre-sweep check excludes manual-mode effectors that are off or whose state_device is unavailable. Dead code pruned: legacy advisory types, boiler backward-compat, LGBM params, encoding aliases.
19. **Smoothed derivative & gain recovery** (done) — 5-minute central differences amplified sensor noise (~10°F/hr) drowning thermostat signals (~0.3°F/hr). Smoothed derivative (15-min half-window rolling mean + wider central difference) reduces noise ~5×, recovering thermostat gains from 1/32 surviving to 25/32. Mode-direction sign filter prevents heating-only effectors from having negative gains. Per-sensor cost display bug fixed (key format mismatch). TUI comfort bars now reflect active comfort profile + MRT correction. See `docs/debugging-notes.md` § "Derivative Noise".
20. **Celsius support & configurable defaults** (done) — `unit: F` or `unit: C` in location block. All hardcoded temperature constants converted via `abs_temp()`/`delta_temp()` from canonical °F at load time. Control thresholds (`setpoint_min`, `setpoint_max`, `cautious_offset`, `max_1h_change`, `cold_room_override`) configurable in `defaults:` section. Display formatting uses `UNIT_SYMBOL` throughout. Dead-band preferred range: `preferred` can be a point or `[lo, hi]` range with zero cost inside the band; `preferred_widen` in profiles expands point targets into dead bands.
21. **Enriched decision logging & faster loops** (done) — Decision log enriched with `active_profile`, `mrt_offsets`, `blocked` columns. `comfort_bounds` now includes `preferred_lo/hi` and `cold/hot_penalty`. Fixed `_compute_actual_comfort_cost` key mismatch bug (was always returning 0.0). Control interval configurable (default 5 min, was 15 min). Sysid split into `fit_sysid()` + `save_sysid_result()` for quality-gated periodic refitting (default hourly in TUI). Comfort plotter uses historical decision bounds (reflects actual profiles/MRT/windows over time) and shows per-sensor control authority (% time system had full control).
22. **Gains-aware regulating sweep & solar irradiance collection** (done) — Regulating effectors (mini splits) now derive sweep options from all constrained sensors with meaningful sysid gain, not a single naming-convention sensor. Both heat and cool modes are offered with targets from affected sensors' preferred temps; idle suppression prunes the nonsensical direction; the trajectory scorer picks the winner. Fixed bug where `mini_split_living_room` could never activate (naming convention produced `living_room_temp` but comfort schedule was on `living_room_climate_temp`). Solar irradiance data collection via forecast.solar HA integration: 5 planes (horizontal + cardinal walls) at 1kWp, collecting W data for future irradiance-based solar model to replace per-hour sysid coefficients × weather-condition fraction.
23. **Elevation-based solar model & cloud coverage collection** (done) — Replaced 11 per-hour solar features (one per hour 7–17) with a single continuous feature: `sin⁺(solar_elevation) × weather_fraction`. Solar elevation computed analytically from lat/lon/timestamp (Spencer 1971 declination). One regression coefficient per sensor instead of 11. Automatically captures seasonal variation (Feb noon sin⁺=0.49, Apr=0.74, Jun=0.91 at Seattle) — the old model underpredicted spring solar gain by ~35% because winter-fitted coefficients had no seasonal awareness. Simulator precomputes `solar_elevations` at 5-min resolution for the prediction horizon. Backward-compatible: falls back to legacy per-hour profiles when `solar_elevation_gains` is absent from thermal_params.json. Collector now stores `cloud_coverage` (0–100%) from weather entity and `forecast_cloud_{h}h` at key horizons for future continuous solar fraction.
24. **Sun-aware MRT correction** (done) — MRT correction now uses current solar state to differentiate sunny vs cloudy days at the same outdoor temp. Per-sensor effective outdoor temp: `effective_outdoor = outdoor + β_solar × sin⁺(elev) × weather_fraction × solar_response`. High-solar-gain rooms (e.g., piano with large windows) get less cold-wall correction on sunny days because sun streaming through windows heats interior surfaces. `solar_response` configurable in MRT config (default 2.0). Static sysid-derived `mrt_weights` removed — dynamic solar calculation replaces them. Manual `mrt_weight` in YAML constraint schedules still applies as a multiplier for non-solar overrides.
25. **Advisory effectors in trajectory sweep** (done) — Generalized windows (and space heaters, blinds, vent fans) into "advisory effectors": user-operated devices the system observes, models in physics, sweeps in trajectory search, but only advises on (notifications, not commands). Three planning layers from a single sweep: reasonable (user cooperates at optimal time → drives HVAC), worst-case (user does nothing → backup breach detection + defensive HVAC), proactive (activate default-state devices → "open window for 2h"). Three-tier comfort bounds: `preferred` (dead band), `acceptable` (normal cost), `backup` (worst-case hedge). Per-step advisory timelines in simulator enable mid-horizon transitions. `AdvisoryDecision` with `return_step` enables two-transition proactive advice. Combinatorics management: coarsening at >50K scenarios, hold-all fallback at >100K. Fast path: scalar tau when no advisory effects exist. Advisory entries marked with `advisory: true` in `environment:` config. Advisory recommendations persisted to state file for TUI display.
26. **Two-stage advisory sweep & configurable penalties** (done) — When advisory combinatorics exceed the hard cap (e.g., 5 open windows × 5 options = 3,125 advisory combos × ~14K HVAC = 43.75M), split into two stages: Stage 1 scores HVAC-only scenarios (scalar tau fast path), selects top-K cheapest plans; Stage 2 crosses top-K with advisory product (per-step tau dynamics), extracts planning layers. K auto-computed: `max(MIN_K=10, 50K / n_advisory_combos)`. Coarsening fallback reduces options to [hold, best action] per device. `_cross_with_advisory()` helper centralizes advisory×HVAC product crossing. `_HARD_RAIL_MULTIPLIER` configurable via `defaults.hard_rail_multiplier` (default 3.0, was hardcoded 10.0). Debug tool: `just debug advisory` shows environment states, compound tau effects per sensor, advisory betas.
27. **Unified environment config** (done) — Replaced separate `windows:` and `advisory_effectors:` YAML sections with a single `environment:` section. Each entry is an `EnvironmentEntryConfig` with `name`, `entity_id`, `column`, `kind` (window/door/shade/vent/heater), `default_state`, `active_state`, and optional `advisory: true`. Kind-specific action verbs (close/open, lower/raise, turn off/on) for display and notifications. Column names from config (`cfg.column`) instead of hardcoded `window_{name}_open`. `environment_states` throughout (renamed from `advisory_states`). `EnvironmentPanel` (renamed from `WindowPanel`), `environment_display()` (renamed from `window_display()`). Dead code `any_window_open` removed. Solar interaction features (`environment_solar_betas`) restricted to `kind: shade` only — windows/doors/vents/heaters affect tau (already modeled), not solar gain magnitude.
28. **Sysid & prediction validation** (done) — `src/weatherstat/validate.py`: two-layer automated validation. Layer 1 (sysid diagnostics): per-sensor `RegressionDiagnostics` with R², Durbin-Watson, bootstrap coefficient stability (CV per advisory/solar feature), VIF (warning >10, error >50), holdout RMSE degradation (warning >20%), environment tau/solar beta magnitude bounds (eff tau < tau_base/3). Bootstrap warnings only for *kept* features (β>0, |t|≥threshold, CV>1). Layer 2 (prediction envelope): absolute temperature bounds [35°F, 100°F], rate of change ≤8°F/hr, scenario spread >30°F. **Health summary**: per-sensor letter grades (A/B/C/F) from R², DW, segment count, gain count, holdout, unstable-kept count. Gain stability: compares current gains with previous `thermal_params.json`, flags >50% drift. R² thresholds calibrated for smoothed-derivative data (0.08 A, 0.04 B); DW thresholds account for structural autocorrelation from 15-min rolling mean (0.20 A, 0.10 B). `_MIN_ACTIVE_ROWS=50` gate skips advisory features with near-zero variance. Test suite: `tests/test_validate.py` (42 tests).
29. **Generic delay sweep & prefix cleanup** (done) — All effector types now sweep delay × duration combinations (`[0, 12, 24] × [12, 24, 72]` steps = 0/1/2h delays × 1/2/6h durations), not just trajectory effectors. The optimizer can express "cool in 2h for 1h" — cheaper than "cool now for 6h" with the same comfort outcome, because energy cost scales by `active_fraction = duration / max_horizon`. Regulating effectors carry `delay_steps`/`duration_steps` through `_RegulatingEffector` into the simulator Euler loop, where an active mask (`step >= delay & step < delay + duration`) gates the proportional activity computation. Binary effectors get the same delay/duration time mask. `MIN_IMPROVEMENT` gate removed — all-off is already a scored scenario; if doing nothing is cheapest, it wins naturally without a special comparison gate. `min_improvement` config field removed. Five prefix-stripping lookups in `control.py` replaced with configuration-driven alternatives: `eff_cfg.temp_col` for energy cost and idle suppression, `sim_params.gains` for rationale sensor lookup. Naming-convention fallback in `_regulating_sweep_options` removed (gains are required).

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
just tui-web          # TUI with embedded mobile web frontend (default 0.0.0.0:8765)
just comfort          # Comfort performance dashboard (last 7 days)
just debug            # Full debug summary (temps, gains, tau, state)
just debug temps      # Current temps + comfort bounds
just debug gains      # Sysid effector→sensor gains
just debug taus       # Tau models + advisory betas
just debug why        # Explain why active effectors are on
just debug advisory   # Advisory states + compound tau effects
just debug decisions  # Recent decision log
just lint             # Lint src + tests
just lint-fix         # Lint and fix src + tests
just test             # Run full test suite
just validate         # Quick sysid + prediction validation smoke test
```

## Key Documentation

- Entity IDs: `weatherstat.yaml` (configured per-house)
- Domain types: `src/weatherstat/types.py`
- System identification: `src/weatherstat/sysid.py`
- Collector: `src/weatherstat/collector.py`
- Executor: `src/weatherstat/executor.py`
- Validation: `src/weatherstat/validate.py` (sysid diagnostics + prediction envelope)

## Conventions

- Python: ruff for linting/formatting, frozen dataclasses, StrEnum for enums
- All source files use explicit types — full type hints in Python
- Temperature unit configurable via `unit` in weatherstat.yaml location block (F or C). Built-in defaults are canonical °F, converted at load time via `abs_temp()`/`delta_temp()`. All runtime values are in the configured unit.
- Snapshot column names use snake_case
- **TUI is the primary interface.** Any change to control logic, comfort schedules, MRT correction, sysid output, or display data must also be reflected in `src/weatherstat/tui/app.py`. The TUI reads temperatures, applies comfort profiles + MRT correction, and displays effector state independently from the control loop. If you change how something is computed in control.py, check whether app.py computes the same thing for display.
- **No silent exceptions in the TUI.** Every `except Exception` must log the error via `self._log()` with the exception message. Critical paths (temp refresh, control cycle, sysid) should also log `traceback.format_exc()`. The only exception is `_log()` itself (can't log a logging failure).
- **Validation after sysid and prediction changes.** Any change to sysid regression (new features, changed regularization, feature engineering) must pass existing `test_validate.py` tests — especially VIF and holdout RMSE checks. New sysid features should be tested for collinearity (VIF <10 for advisory/solar features). Any change to simulator or control predictions must pass prediction envelope tests (absolute bounds, rate limits). Run `just validate` as a quick smoke test after sysid or prediction changes.
- **Use Just for all commands.** Run `just test`, `just lint`, `just validate`, etc. — not raw `uv run` commands. The Justfile is the canonical interface for all development tasks. Keep the CLAUDE.md Commands section in sync with the Justfile.

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
  advisory_state.json         # environment entry advisory cooldowns
```

First-time setup: `just init` creates the directory and copies `weatherstat.yaml.example`.
