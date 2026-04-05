# Weatherstat Architecture

## Goal

Maximize thermal comfort across a multi-room house while minimizing energy cost, by anticipating rather than reacting to temperature changes.

The primary challenge is hydronic floor heating with 2-4 hour thermal lag, compounded by solar gain, inter-room coupling, and Seattle's variable marine climate. A conventional thermostat — purely reactive — is the wrong tool for this job. The system must predict hours ahead and act preemptively.

## Design principles

1. **Physics first.** The house is a thermal system with known structure. Use physics to model what we understand (envelope loss, heating input, thermal mass) and learn only the parameters we can't derive from first principles (gains, delays, solar profiles).

2. **Causality over correlation.** The system must answer counterfactual questions: "what happens if I turn on the heat now vs. in 2 hours?" This requires a model that understands cause and effect, not one trained on observational data where actions and conditions are confounded.

3. **Configuration-driven.** Adding a sensor, effector, or constraint should be a YAML edit, not a code change. The system discovers its own shape from configuration.

4. **Sensors and effectors, not rooms.** The system optimizes sensor values by actuating effectors. Rooms are display labels, not structural entities. Constraints target sensors directly. Effector dependencies (e.g., a blower depends on a thermostat) are expressed as direct references, not spatial groupings.

5. **Learned over configured.** Sysid discovers physics (tau, gains, window effects) from data. Configuration declares what exists, not how things interact. Command/state pairs: every effector has intent (command) and reality (measured state). Sysid trains on state; control sets commands.

6. **Safe by default.** Dry-run before live. Override detection. Hold times. Setpoint clamps. Generic device health checks. The system should be hard to misconfigure into damaging equipment or freezing pipes.

7. **Observable.** Every decision is logged with its inputs, alternatives considered, and reasoning. Predictions are saved for later comparison with reality.

---

## System overview

```
                        +-----------------+
                        |  Home Assistant  |
                        |   (REST API)    |
                        +--------+--------+
                                 |
                    reads state   |   executes commands
                   +-------------+-------------+
                   |                           |
            +------v------+             +------v------+
            |  Collector   |             |  Executor    |
            |  (5-min)     |             |              |
            +------+------+             +------^------+
                   |                           |
                   | SQLite snapshots           | command JSON
                   |                           |
            +------v--------------------------+------+
            |           Thermal Model                 |
            |  (grey-box forward simulator)            |
            |                                         |
            |  +-------------+  +------------------+  |
            |  | Physics Core |  | Learned Params   |  |
            |  | Newton decay |  | Heating gain/lag |  |
            |  | Heating input|  | Solar model      |  |
            |  | Inter-room   |  | Coupling coeffs  |  |
            |  +-------------+  +------------------+  |
            +-----------+-----------------------------+
                        |
                        | predicted temperatures
                        |
            +-----------v-----------------------------+
            |           Controller (MPC)              |
            |                                         |
            |  Comfort schedules + energy costs        |
            |  Sweep candidate trajectories            |
            |  Score with thermal model predictions    |
            |  Select best trajectory                  |
            |  Emit: commands (executor) +             |
            |        advisories (notifications)        |
            +-----------------------------------------+
```

---

## Components

### 1. Configuration (`weatherstat.yaml`)

Single source of truth for all entity IDs, effector definitions, comfort constraints, energy costs, and safety thresholds.

The system is organized around four fundamental concepts:
- **Sensors** — observable quantities (temperature, humidity) with entity IDs and types
- **Effectors** — actuatable devices (thermostats, mini splits, blowers) described by properties (`control_type`, `mode_control`, `depends_on`, `command_keys`) rather than categories. All share a unified `EffectorConfig` with per-effector command/state encodings and optional health checks
- **Constraints** — scoring objectives on sensor values, with time-of-day bounds and asymmetric penalty weights, referencing sensors directly (not rooms)
- **Windows** — environmental modifiers with binary state; their effect on sensor dynamics is learned by sysid (per-window cooling rate coefficients in TauModel)

**Contents:**
- `location` — lat/lon/elevation/timezone (for solar position)
- `sensors` — temperature and humidity entities (10+ humidity sensors)
- `effectors` — flat dict of all HVAC devices (thermostats, mini splits, blowers), each declaring `control_type` (trajectory/regulating/binary), `mode_control` (manual/automatic), `supported_modes`, `state_encoding`, `max_lag_minutes`, `energy_cost`, and optional `depends_on` (string or list — ALL must be active), `state_device`, `proportional_band`, `mode_hold_window`
- `windows` — window/door sensors (effects learned by sysid, not configured)
- `constraints` — per-sensor, time-of-day comfort bands with asymmetric penalty weights (sensor-to-effector coupling derived from sysid coupling matrix)
- `advisory` — effort costs, quiet hours, cooldown timers
- `safety` — cooldown timers for infrastructure alerts
- `defaults` — fallback values (e.g., `tau: 45.0` before sysid has run), timing intervals (`control_interval`, `sysid_interval`), and optional control thresholds

**Read by:** every other component (via `yaml_config.py`). The loader produces typed configuration objects. Adding a sensor or device to the YAML automatically propagates to collection, prediction, and control.

**Currently:** manually maintained. See "Config Generator" in future components.

---

### 2. Collector (`src/weatherstat/collector.py`)

Periodically sample the full state of the house and persist it for training and analysis.

**Behavior:**
- Runs every 5 minutes.
- Reads all monitored entities from HA via REST API (`GET /api/states`).
- Extracts values using config-driven column definitions (temperature attributes, HVAC actions/modes/targets, window states, weather conditions).
- Captures weather forecast snapshots (`forecast_temp_{1..12}h`, `forecast_condition_{1,2,4,6,12}h`, `forecast_wind_{1,2,4,6,12}h`, `forecast_cloud_{1,2,4,6,12}h`) from HA's met.no integration via service call. Current `cloud_coverage` (0–100%) is also stored from the weather entity attributes.
- Deduplicates by rounding timestamps to the snapshot interval.
- Writes to SQLite (`~/.weatherstat/snapshots/snapshots.db`) in EAV format: `readings` table with `(timestamp, name, value)` triples. No schema changes needed to add sensors.

**Output:** SQLite database with 5-minute resolution. The `readings` table stores `(timestamp, name, value)` triples (~80 readings per snapshot). The Python reader pivots this to a wide DataFrame at load time, applying types from config.

**Operational:** `just collect` runs the 5-min loop with auto-recovery. `just health` checks data freshness.

---

### 3. Thermal Model

The core prediction engine. A grey-box model: physics provides the structure, data provides the parameters.

#### 3a. Physics Core (`src/weatherstat/simulator.py`)

Forward-simulates room temperatures by Euler-integrating the thermal dynamics of each sensor at 5-minute resolution:

```
dT/dt = (T_outdoor - T) / tau
        + Σ gain_e * activity_e(t - lag_e)                 # effector heating, delayed
        + β_solar(sensor) × sin⁺(elevation) × SF           # solar gain (elevation-based)
        + Σ β_solar(sensor, plane) × irradiance(plane, t)  # solar gain (future: irradiance)
```

Where:
- `tau` is the effective envelope time constant, computed from `TauModel`: `1/tau_eff = 1/tau_base + Σ β_w × open_w + Σ β_{ww'} × open_w × open_w'`
- Each effector `e` contributes a gain (°F/hr per activity unit) delayed by its fitted lag
- Solar gain uses an elevation-based model: one `β_solar` per sensor × `sin⁺(solar_elevation)` × weather-conditioned solar fraction (sunny=1.0, cloudy=0.15, etc.). Solar elevation is computed analytically from latitude, longitude, and time — no external data needed. `sin⁺(elevation)` naturally captures both hour-of-day variation (low at sunrise/sunset, high at noon) and seasonal variation (winter noon ~30° at Seattle, April ~48°, summer ~66°). This replaced the prior per-hour model (11 coefficients per sensor, hours 7–17) which had no seasonal awareness — coefficients fitted from winter data underpredicted spring solar gain by ~35%. The per-sensor regression coefficient absorbs compound house geometry (window orientations, roof pitch, glass vs wall). **Planned next step:** per-sensor irradiance gain coefficients fitted against 5-plane irradiance data (horizontal + 4 cardinal walls) from forecast.solar, which will also capture directionality (e.g., west rooms warm more in afternoon). Data collection started 2026-04-05; see `docs/FUTURE.md` § "Irradiance-Based Solar Model"

**Effector control types** (property of each effector, not separate code categories):
- **Trajectory:** Pre-computed binary activity from delay/duration parameters. Used for slow-twitch effectors (e.g., hydronic thermostats with 45-75 min lag).
- **Binary:** Constant activity from mode (off=0, low=0.5, high=1.0). Used for discrete-level effectors (e.g., blowers).
- **Regulating:** Proportional activity computed inside the Euler loop: `activity = clip((target - T) / proportional_band, 0, 1)`. Activity drops to zero as the room reaches the target. Used for self-regulating climate devices (e.g., mini splits).

**Effector dependencies:** Some effectors only produce useful output when all their dependencies are active (e.g., a blower circulating air over hydronic coils only helps when heat is flowing through them). `depends_on` references one or more parent effectors by name; the dependency is met when *all* parents are active (AND gate). The simulator models this via multiplicative activity gates; scenario generation prunes combinations where any parent is inactive.

**Integration:** Euler steps at 5-minute resolution, chaining hourly weather forecast segments for the outdoor temperature trajectory.

**Batch simulation:** The controller calls `predict()` with thousands of candidate scenarios. Each scenario specifies effector timelines (trajectory-parameterized), and the simulator evaluates all of them against the same initial conditions. Internally, Euler integration is vectorized across all scenarios using numpy — the outer loop is over sensors and timesteps, with numpy broadcasting handling all scenarios simultaneously.

#### 3b. System Identification (`src/weatherstat/sysid.py`)

Fits all thermal model parameters from observed collector data using a two-stage approach.

**Stage 1 — Tau fitting (scipy `curve_fit`):** For each temperature sensor, selects all nighttime (10pm–6am) periods where all HVAC effectors are off AND all windows are closed (sealed envelope). Fits Newton cooling (`T(t) = T_out + (T_0 - T_out) * exp(-t/tau)`) via nonlinear least squares on each contiguous segment. Multiple segments → weighted median → `tau_base` (sealed envelope time constant).

**Stage 2 — Effector gains, solar, and window effects (ridge regression):** With tau_base fitted, computes Newton residuals at every timestep (`dT/dt_observed - dT/dt_newton`). These residuals are explained by a linear regression on: lagged effector activity (coarse time bins capturing delay), a solar elevation feature (`sin⁺(elevation) × weather_fraction` — one continuous feature replacing the prior 11 per-hour indicators), weather control features (ΔT², wind×ΔT, dT_outdoor/dt), per-window `window_state × (T_out - T)` features (cooling rate when open), and window pair interactions (cross-breeze effects). One regression per sensor.

The regression uses selectively standardized ridge (L2 penalty λ = 0.01×n). Solar and window features are pre-scaled by their standard deviation so the penalty falls proportionally; effector features are left in raw scale for full regularization against confounded gains. T-statistics flag negligible gains (|gain| < 0.05°F/hr AND |t-stat| < 2.0).

**Smoothed derivative:** The dT/dt computation uses a smoothed central difference rather than naive 5-minute intervals. Temperature is first smoothed with a centered rolling mean (default: 15-minute half-window), then differentiated over the smoothed values. This is critical: naive 5-minute central differences amplify sensor noise (~±0.1°F jitter) into ~10°F/hr of derivative noise, drowning effector signals of ~0.3°F/hr. The smoothed derivative reduces noise ~5× while preserving signals on the timescale of effector lags (≥15 min). Note: only Stage 2 needs this — Stage 1 (tau fitting) operates on raw temperature curves via `curve_fit`, where noise averages out naturally (integration suppresses noise; differentiation amplifies it). The smoothing introduces mild positive autocorrelation (~35-min kernel), reducing effective sample size by roughly 3×, but this is negligible with 10K+ snapshots. See `docs/debugging-notes.md` § "Derivative Noise" for the full analysis.

**Gain filtering at load time:** The simulator applies additional filters when loading gains from `thermal_params.json`:
- **t-statistic threshold** (|t| ≥ 1.5): prunes gains that are likely confounded (e.g., bedroom split correlates with other warming sources but doesn't cause it). Without this, OLS attributes correlated warming to whichever effector happens to be active.
- **Magnitude cap** (≤ 3.0°F/hr): catches physically implausible gains from sensors near vents, unobserved heat sources, or short data histories.
- **Mode-direction sign filter**: heating-only effectors (trajectory with `supported_modes: [heat]`) cannot have negative gains, and vice versa. Confounded OLS can produce these nonsensical cross-coupling effects.
- **Mode-direction clamp**: in the Euler loop for regulating effectors, heating contributions are clamped ≥ 0 and cooling contributions ≤ 0, preventing confounded gains from producing physically impossible effects.

**What it extracts:**
- **Effector × sensor gain matrix**: heating rate (°F/hr) and effective delay for each (effector, sensor) pair. Multiple effectors active simultaneously? The regression decomposes their contributions.
- **Solar elevation gains**: per-sensor `β_solar` coefficient (°F/hr per unit sin(elevation)×fraction). One value per sensor replaces the prior 11 per-hour coefficients.
- **Tau per sensor**: `tau_base` (sealed envelope time constant).
- **Window coupling coefficients**: per-window `β_w` (additional cooling rate when window is open) and cross-breeze interaction terms `β_{ww'}`. The simulator computes effective tau as `1 / (1/tau_base + Σ β_w × open_w + Σ β_{ww'} × open_w × open_w')`.

**Config-driven:** Effectors and sensors enumerated from `weatherstat.yaml`. Adding a device or sensor = YAML edit + rerun.

**Output:** `~/.weatherstat/thermal_params.json` — the full coupling matrix, tau fits, and solar elevation gains. Run via `just sysid`.

**Two-phase API:** `fit_sysid()` generates a `SysIdResult` without writing to disk; `save_sysid_result()` persists it. `run_sysid()` is a convenience wrapper that calls both. The TUI uses the two-phase API with a quality gate (rejects fits with zero taus or zero significant gains) for automatic periodic refitting.

**Dependencies:** numpy, scipy (curve_fit), pandas. No ML frameworks required.

#### 3c. Prediction Interface

The thermal model exposes a clean prediction interface to the controller:

```python
predict(state: HouseState, scenarios: list[Scenario],
        params: SimParams, horizons: list[int]) -> (target_names, prediction_matrix)
```

`HouseState` bundles all environmental state (current room temps, outdoor temp, forecast temps, window states, hour of day, recent HVAC history). `Scenario` encodes the HVAC plan (a dict of `EffectorDecision` per effector, each specifying mode, target, delay, and duration as applicable). The controller treats this as a black box — it provides state and candidate actions, and receives `prediction_matrix[i, j]` (predicted temperature for scenario `i` at target `j`, where targets are room × horizon combinations).

**Vectorization:** Activity matrices `(n_scenarios, n_total_steps)` are built per effector using numpy broadcasting over delay/duration parameters. Total effector forcing is pre-computed per sensor via slice operations, so the Euler integration loop (72 timesteps) contains only numpy vector operations over all scenarios. This reduces Python loop iterations from ~4.3M to ~576 for a typical 7400-scenario sweep.

---

### 4. Controller (`src/weatherstat/control.py`)

Decides what HVAC actions to take right now, using receding-horizon optimization (MPC with trajectory search).

**Each control cycle:**
1. Read current house state (temperatures, HVAC states, window states, weather).
2. Fetch weather forecast for the prediction horizon.
3. Enumerate candidate trajectories: each thermostat gets a delay × duration grid (e.g., "delay 1h, heat for 2h, coast"), crossed with blower modes and mini-split target temperatures (~5,000–15,000 scenarios depending on config).
4. For each candidate: forward-simulate all sensor temperatures. Score the resulting trajectories against comfort schedules and energy costs.
5. Select the trajectory with the best score.
6. Emit electronic commands for the executor.

**Trajectory search:** Effector options are generated per control_type: trajectory effectors get delay × duration grids, regulating effectors get gains-aware mode + target combinations (heat targets from affected sensors' `pref_lo`, cool targets from `pref_hi`, with idle suppression when the highest-gain sensor is already past target), binary effectors get their supported modes. The sweep takes the cartesian product, with dependent effectors constrained by their parent's state. Boiler activity is confirmed via state_gate multiplication in the simulator.

**Receding horizon:** Only the immediate action matters. A trajectory of "delay 2h then heat" means "stay off now." At the next cycle (default 5 minutes, configurable via `control_interval`), the controller re-evaluates with fresh data and may choose differently.

**Scoring:**

Two-layer comfort cost model with dead-band preferred:
1. **Preferred band (dead band):** `preferred` can be a point (`72`) or a range (`[71, 73]`). Zero cost within the band; quadratic penalty outside it, weighted by `cold_penalty` (below) and `hot_penalty` (above). This prevents the optimizer from wasting energy chasing a single degree — any temperature within the band is equally acceptable.
2. **Hard rails:** Steep additional penalty (10×) for exceeding `min`/`max` bounds.

```
score = sum over (sensors, horizons) of:
    dead_band_cost(T_predicted, preferred_lo, preferred_hi)  # zero inside band
  + hard_rail_penalty(T_predicted, min, max)                  # steep outside bounds
  + energy_cost(actions, duration)
```

**Comfort profiles** (Home/Away) apply global offsets to all schedules: `preferred_offset` shifts the band, `preferred_widen` expands it (±half into a dead band), `min_offset`/`max_offset` shift hard rails, and `penalty_scale` multiplies penalties. A typical Away profile widens the preferred band so the optimizer maintains safety limits without spending energy on optimization within the band.

Window-open states widen the comfort band to avoid fighting ventilation.

**Horizon weighting:** Closer predictions weighted higher (more accurate, more actionable).

**Energy cost scaling:** Each effector's cost comes from its per-effector `energy_cost` config. Trajectory effectors: cost proportional to `duration / max_horizon`. Regulating effectors: cost proportional to expected activity (target vs outdoor temperature within proportional band). Binary effectors: per-mode cost dict.

**Physical constraints:**
- Dependent effectors forced off when their dependency is inactive (e.g., blowers off when thermostat isn't calling for heat).
- Setpoint clamps (absolute safety bounds: 62–78°F).
- Hold times: 3-minute minimum between setpoint changes; 2-hour minimum between mini-split mode changes.
- Mode hold window: per-device configurable hours (e.g., 10pm–7am) during which mini-split mode changes are forbidden — only silent target temperature adjustments are allowed.
- Cold-sensor override: force immediate zone heating (delay=0) when any sensor is significantly below comfort minimum.

**Decision rationale:** After selecting the best scenario, the controller runs counterfactual simulations — the winning scenario with each active device individually removed. This gives true per-device attribution: "what does THIS device contribute?" rather than conflating multi-device effects. Each active device's rationale shows the sensor most affected, the trajectory difference, and the per-sensor comfort cost impact. A per-sensor cost breakdown table (decision vs all-off) shows which sensors drive the overall decision.

**Output:** `ControlDecision` JSON containing recommended device states, predicted outcomes, trajectory info, and decision reasoning.

---

### 5. Executor (`src/weatherstat/executor.py`)

Applies the controller's commands to Home Assistant with safety checks.

**Behavior:**
- Reads the latest `command_*.json` from `predictions/`.
- For each effector, checks current HA state via REST API (lazy execution — skip if already in desired state).
- Calls HA services: `climate.set_temperature`, `climate.set_hvac_mode`, `fan.set_preset_mode`, etc.
- Detects manual overrides: if device state doesn't match what the executor last set, a human intervened. Respects the override for 30 minutes (or `--force`).
- Persists executor state for override tracking across restarts.
- Returns structured `ExecutorResult` with per-device actions, used by the TUI to display override status.

**Safety:**
- Never executes without explicit `--live` flag.
- Override detection prevents fighting with manual adjustments.
- All actions logged for audit.

---

### 6. Window Opportunities (`src/weatherstat/advisory.py`)

Persistent, energy-aware recommendations for window state changes. Uses the physics simulator to evaluate whether toggling window states would improve comfort and/or save energy, given the committed electronic plan.

**Two-tier evaluation:**
1. **Quick check:** Simulate winning scenario with window toggled → comfort delta.
2. **Re-sweep** (if promising): Full scenario sweep with toggled window to find the best HVAC plan. Captures "open window + turn off mini split" energy savings.

**Two-threshold model:**
- **Opportunity threshold** (0.3): minimum benefit to track (visible in control output).
- **Notification threshold** (1.5): minimum benefit to push a mobile notification.

**Lifecycle management:** Opportunities persist across control cycles. New ones are added, still-valid ones kept, expired ones dismissed (including HA persistent notifications). Per-window notification IDs prevent stacking.

**Key design constraint:** The electronic plan does NOT change based on opportunities. The controller commits to the best electronic trajectory given current window states. If the human acts (e.g., opens a window), the next control cycle re-evaluates and naturally adjusts.

**Dispatch:**
- Per-window cooldown timers prevent notification fatigue.
- Quiet hours suppress push notifications (still tracked).
- Dismissed opportunities automatically clear HA persistent notifications.

---

### 7. Safety System (`src/weatherstat/safety.py`)

Detects infrastructure problems that prevent the control loop from working.

**Check types:**
- **Manual-mode effector check:** Detects when a manual-mode effector (e.g., thermostat) has `hvac_mode` "off" while the controller wants it active — setting the target has no effect in this state. Iterates all effectors with `mode_control: "manual"` from config.
- **Generic device health:** Iterates health check definitions from the `health` YAML section. Fetches each entity's current state from HA and compares against configured min/max thresholds or expected state values.

**Dispatch:** Safety alerts bypass quiet hours (these are urgent). Cooldown tracking prevents notification fatigue. Alerts reuse the advisory notification infrastructure.

**Configuration-driven:** Health checks are defined in the `health` YAML section with entity ID, min/max thresholds, expected state, severity, and message. Adding a health check = YAML edit.

---

### 8. Decision Log (`src/weatherstat/decision_log.py`)

Records every control decision with full context.

**Stores:**
- Timestamp, live/dry-run mode.
- Input state snapshot (outdoor conditions, all room temperatures, HVAC states).
- Predicted outcomes at each horizon per room.
- Selected action (setpoints, blowers, mini-splits) and trajectory info.
- Comfort cost, energy cost, total cost.
- Active comfort profile name (Home/Away/etc.).
- Resolved comfort bounds per sensor (min, max, preferred_lo, preferred_hi, cold/hot penalties) — the actual targets after profile offsets, MRT correction, and window adjustments.
- MRT correction offsets (base offset + per-sensor applied offsets).
- Blocked effectors (ineligible + physically blocked, with reasons).

**Outcome backfill:** At the start of each control cycle, the system checks whether enough time has elapsed to compare predictions to actual temperatures. For each horizon (1h, 2h, 4h, 6h), it finds the closest collector snapshot and records prediction error. Retroactive comfort cost uses the full two-layer model (preferred band + hard rails) with the enriched bounds logged at decision time.

**Used by:**
- Humans debugging decisions ("why did it turn on the heat at 3 AM?").
- Prediction error analysis: compare predicted vs actual temperatures at each logged horizon.
- Comfort performance dashboard (`just comfort`) for capacity analysis and control authority tracking.

---

### 9. Comfort Dashboard (`scripts/plot_comfort.py`)

Answers "is the system working as designed?" at a glance. Run via `just comfort`.

**Output:** A multi-panel PNG with:
- **Summary bar:** Per-sensor horizontal stacked bar showing % in comfort band, % too cold (capacity-limited vs control opportunity), % too hot (capacity-limited vs control opportunity).
- **Temperature traces:** Per-sensor time series with comfort band overlay (green fill = min/max, dashed/filled = preferred band), violation shading (red = below min, orange = above max), and control authority background tinting (red = all blocked/offline, amber = partially blocked, gray = no decision data).
- **Outdoor + effector panel:** Outdoor temperature with heating state overlays.
- **Prediction accuracy** (optional, `--predictions`): Error histogram by horizon with MAE and bias.

**Historical comfort bands:** When decision log data is available, comfort bands are derived from the logged comfort_bounds per decision (reflecting the actual profile, MRT correction, and window adjustments active at the time) rather than the current config. This means the plot accurately shows what the system was targeting, even if profiles or config changed during the period.

**Capacity analysis:** Violations are classified as "capacity exceeded" (all dedicated effectors at max — building physics problem) vs "control opportunity" (dedicated effectors had headroom). Dedicated effectors are identified per sensor: zone thermostat (from coupling matrix) plus any name-matched mini split or blower. Cross-talk gains from effectors that primarily serve other sensors are excluded — the optimizer already balances those trade-offs.

**Control authority:** Per-sensor tracking of when the system had full control (no relevant effectors blocked or overridden), partial control, or no control (all blocked, system offline, or dry-run mode). Displayed as background tinting on time-series panels and a "Ctrl %" column in the console summary.

**Console output:** Summary table with per-sensor breakdown: % in band, control authority %, capacity/control violation split, and dedicated effector list.

---

## Component interfaces

### Collector -> Thermal Model

**Interface:** SQLite database (`~/.weatherstat/snapshots/snapshots.db`).
- EAV `readings` table: `(timestamp, name, value)` triples, driven by YAML config.
- One snapshot per 5-minute interval (~74 readings per snapshot).
- Names include: temperature sensors, HVAC states (action, mode, target per effector), window states, weather, humidity sensors, forecast snapshots.
- Python reader pivots to wide DataFrame at load time, applying SQL types from config.

The collector is language-agnostic from the model's perspective.

### Thermal Model -> Controller

**Interface:** `predict(state: HouseState, scenarios, params, horizons) -> (target_names, prediction_matrix)`

The controller treats the thermal model as a black box. `HouseState` bundles all environmental inputs; `Scenario` encodes all HVAC plans (as a dict of `EffectorDecision` per effector). The prediction engine can be swapped without touching sweep logic, scoring, or constraints.

### Controller -> Executor

**Interface:** `ControlDecision` JSON file in `~/.weatherstat/predictions/`.

Contains:
- Effector decisions: per-effector mode, target, delay, duration.
- Command targets: per-effector setpoints for HA commands.
- Predictions: per-sensor, per-horizon temperatures.
- Trajectory info: delay and duration per trajectory effector.
- Timestamp and live/dry-run flag.

### Controller -> Advisory System

**Interface:** Function calls within the control loop:
1. `evaluate_window_opportunities(state, winning_scenario, ...)` — runs `predict()` per window toggle, re-sweeps promising candidates for energy savings, returns `WindowOpportunity` list.
2. `process_opportunities(opportunities, live, ...)` — manages lifecycle (add/keep/expire/dismiss), dispatches notifications to HA.

### Configuration -> Everything

**Interface:** `weatherstat.yaml` parsed by `yaml_config.py`.

---

## Future components

See `docs/FUTURE.md` for the full roadmap. Key upcoming areas:

- **Online Learning:** Automatic parameter adjustment from prediction error trends. The decision log already records predicted vs actual temperatures at each horizon.
- **Virtual Thermostats:** Per-sensor HA climate entities for user-adjustable comfort targets from the dashboard.
- **Summer/cooling adaptation:** Currently winter-only data. Mini splits are the only cooling effectors; capacity analysis will be important.
- **Additional blower automation:** More blowers would improve heat distribution to capacity-limited sensors like the kitchen.
- **Anomaly Detection:** Alert when prediction errors suggest changed conditions (window left open, equipment issues).

---

## Retrospective: what we tried and learned

### The ML approach (Feb-Mar 2026)

We built a full LightGBM pipeline: 8 rooms x 5 horizons = 40 models trained on collector snapshots (5-min resolution) and historical hourly statistics.

**What worked well:**
- **Pipeline infrastructure.** Config-driven collection, feature engineering shared between training and inference, experiment framework with worktrees and backtesting — all reusable.
- **Comfort schedules and scoring.** The sweep/score/constraint framework in the controller is sound and carries forward unchanged.
- **Newton cooling fits.** Overnight cooling analysis produced reliable tau values per room (sealed and ventilated). These become parameters in the grey-box model.
- **Physics-as-features experiment** (`physics_v1`): rate-of-change features improved 12h predictions significantly for downstairs rooms. This validated that physics signals help, but encoding them as ML features is indirect.
- **Sweep and executor.** The action enumeration, physical constraint enforcement, override detection, and lazy execution all work correctly and carry forward.

**What didn't work:**
- **Counterfactual prediction.** The model learned "heat is ON when it's cold" (correlation) not "heat ON causes warming" (causation). Asking "what if heat is off?" in cold conditions produces unreliable answers because the training data rarely contains that scenario — the thermostat prevented it.
- **Hybrid blending experiment** (`hybrid_physics`): blending ML with uncalibrated Newton decay made predictions worse at all horizons beyond 1h. The physics model needs to be good on its own before blending helps.
- **Guardrail accumulation.** Newton floor/ceiling on sweep predictions, cold-room overrides, and safety clamps were all patches for the ML model's causal blindness. Each was individually reasonable but together they indicated the wrong architecture — the model needed external help to make physically plausible predictions.

**The lesson:** ML is a powerful function approximator, but for a physical system with known structure, encoding that structure directly produces better predictions with less data, better extrapolation, and — crucially — reliable counterfactual reasoning. ML remains the right tool for components that are genuinely hard to model (solar gain, occupancy effects).

See `docs/EXPERIMENTS.md` for detailed experiment results and metrics. The ML training pipeline has been archived to `archive/` for reference.
