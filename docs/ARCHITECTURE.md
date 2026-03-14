# Weatherstat Architecture

## Goal

Maximize thermal comfort across a multi-room house while minimizing energy cost, by anticipating rather than reacting to temperature changes.

The primary challenge is hydronic floor heating with 2-4 hour thermal lag, compounded by solar gain, inter-room coupling, and Seattle's variable marine climate. A conventional thermostat — purely reactive — is the wrong tool for this job. The system must predict hours ahead and act preemptively.

## Design principles

1. **Physics first.** The house is a thermal system with known structure. Use physics to model what we understand (envelope loss, heating input, thermal mass) and learn only the parameters we can't derive from first principles (gains, delays, solar profiles).

2. **Causality over correlation.** The system must answer counterfactual questions: "what happens if I turn on the heat now vs. in 2 hours?" This requires a model that understands cause and effect, not one trained on observational data where actions and conditions are confounded.

3. **Configuration-driven.** Adding a sensor, effector, or constraint should be a YAML edit, not a code change. The system discovers its own shape from configuration.

4. **Sensors and effectors, not rooms.** The system optimizes sensor values by actuating effectors. Rooms are display labels, not structural entities. Constraints target sensors directly. Zones are a property of effectors (which thermostat controls which heating circuit), not of sensors.

5. **Learned over configured.** Sysid discovers physics (tau, gains, window effects) from data. Configuration declares what exists, not how things interact. Command/state pairs: every effector has intent (command) and reality (measured state). Sysid trains on state; control sets commands.

6. **Safe by default.** Dry-run before live. Override detection. Hold times. Setpoint clamps. Generic device health checks. The system should be hard to misconfigure into damaging equipment or freezing pipes.

7. **Observable.** Every decision is logged with its inputs, alternatives considered, and reasoning. Predictions are saved for later comparison with reality.

---

## System overview

```
                        +-----------------+
                        |  Home Assistant  |
                        |  (WebSocket +   |
                        |   REST API)     |
                        +--------+--------+
                                 |
                    reads state   |   executes commands
                   +-------------+-------------+
                   |                           |
            +------v------+             +------v------+
            |  Collector   |             |  Executor    |
            |  (TS, 5-min) |             |  (TS)        |
            +------+------+             +------^------+
                   |                           |
                   | SQLite snapshots           | command JSON
                   |                           |
            +------v--------------------------+------+
            |           Thermal Model                 |
            |  (Python: grey-box forward simulator)   |
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

The system is organized around five fundamental concepts:
- **Sensors** — observable quantities (temperature, humidity) with entity IDs and types
- **Effectors** — actuatable devices (thermostats, mini splits, blowers, boiler) with command/state encodings and optional health checks
- **Constraints** — scoring objectives on sensor values, with time-of-day bounds and asymmetric penalty weights, referencing sensors directly (not rooms)
- **Windows** — environmental modifiers with binary state; their effect on sensor dynamics is learned by sysid (per-window cooling rate coefficients in TauModel)
- **Zones** — topological grouping of effectors (which thermostat controls which heating circuit). Sensor-to-zone mapping is derived from the sysid coupling matrix (which thermostat has the highest gain for each sensor), not configured.

**Contents:**
- `location` — lat/lon/elevation/timezone (for solar position)
- `sensors` — temperature and humidity entities (10+ humidity sensors)
- `effectors` — thermostats, mini splits, blowers, boiler with mode encodings and health check thresholds
- `windows` — window/door sensors (effects learned by sysid, not configured)
- `constraints` — per-sensor, time-of-day comfort bands with asymmetric penalty weights (no zone assignment — derived from coupling matrix)
- `zones` — thermostat-to-circuit mapping
- `energy_costs` — per-device energy cost for the optimizer
- `advisory` — effort costs, quiet hours, cooldown timers
- `safety` — cooldown timers for infrastructure alerts
- `defaults` — fallback values (e.g., `tau: 45.0` before sysid has run)

**Read by:** every other component (TS via `yaml-config.ts`, Python via `yaml_config.py`). Both loaders produce typed configuration objects. Adding a sensor or device to the YAML automatically propagates to collection, prediction, and control.

**Currently:** manually maintained. See "Config Generator" in future components.

---

### 2. Collector (`ha-client/src/collector.ts`)

Periodically sample the full state of the house and persist it for training and analysis.

**Behavior:**
- Runs every 5 minutes.
- Reads all monitored entities from HA via WebSocket subscription.
- Extracts values using config-driven column definitions (temperature attributes, HVAC actions/modes/targets, window states, weather conditions).
- Captures weather forecast snapshots (`forecast_temp_{1..12}h`, `forecast_condition_{1,2,4,6,12}h`, `forecast_wind_{1,2,4,6,12}h`) from HA's met.no integration for use by the simulator.
- Deduplicates by rounding timestamps to the snapshot interval.
- Writes to SQLite (`~/.weatherstat/snapshots/snapshots.db`) in EAV format: `readings` table with `(timestamp, name, value)` triples. No schema changes needed to add sensors.

**Output:** SQLite database with 5-minute resolution. The `readings` table stores `(timestamp, name, value)` triples (~74 readings per snapshot). The Python reader pivots this to a wide DataFrame at load time, applying types from config.

**Operational:** `just collect-durable` runs with auto-restart and health monitoring. `just health` checks data freshness.

---

### 3. Thermal Model

The core prediction engine. A grey-box model: physics provides the structure, data provides the parameters.

#### 3a. Physics Core (`ml/src/weatherstat/simulator.py`)

Forward-simulates room temperatures by Euler-integrating the thermal dynamics of each sensor at 5-minute resolution:

```
dT/dt = (T_outdoor - T) / tau
        + Σ gain_e * activity_e(t - lag_e)     # effector heating, delayed
        + solar(hour_of_day) * solar_fraction   # solar gain, weather-modulated
```

Where:
- `tau` is the effective envelope time constant, computed from `TauModel`: `1/tau_eff = 1/tau_base + Σ β_w × open_w + Σ β_{ww'} × open_w × open_w'`
- Each effector `e` contributes a gain (°F/hr per activity unit) delayed by its fitted lag
- Solar gain is a per-sensor, per-hour profile from sysid, modulated by a weather-conditioned solar fraction (sunny=1.0, cloudy=0.15, clear-night=0.0, etc.)

**Effector types:**
- **Trajectory (thermostats):** Pre-computed binary activity from delay/duration parameters.
- **Discrete (blowers):** Constant activity from mode (off=0, low=0.5, high=1.0).
- **Regulating (mini splits):** Proportional activity computed inside the Euler loop: `activity = clip((target - T) / proportional_band, 0, 1)`. Activity drops to zero as the room reaches the target, modeling the mini split's PID controller.

**Integration:** Euler steps at 5-minute resolution, chaining hourly weather forecast segments for the outdoor temperature trajectory.

**Batch simulation:** The controller calls `predict()` with thousands of candidate scenarios. Each scenario specifies effector timelines (trajectory-parameterized), and the simulator evaluates all of them against the same initial conditions. Internally, Euler integration is vectorized across all scenarios using numpy — the outer loop is over sensors and timesteps, with numpy broadcasting handling all scenarios simultaneously.

#### 3b. System Identification (`ml/src/weatherstat/sysid.py`)

Fits all thermal model parameters from observed collector data using a two-stage approach.

**Stage 1 — Tau fitting (scipy `curve_fit`):** For each temperature sensor, selects all nighttime (10pm–6am) periods where all HVAC effectors are off AND all windows are closed (sealed envelope). Fits Newton cooling (`T(t) = T_out + (T_0 - T_out) * exp(-t/tau)`) via nonlinear least squares on each contiguous segment. Multiple segments → weighted median → `tau_base` (sealed envelope time constant).

**Stage 2 — Effector gains, solar profiles, and window effects (numpy linear regression):** With tau_base fitted, computes Newton residuals at every timestep (`dT/dt_observed - dT/dt_newton`). These residuals are explained by a linear regression on: lagged effector activity (coarse time bins capturing delay), hour-of-day indicators (solar gain), per-window `window_state × (T_out - T)` features (cooling rate when open), and window pair interactions (cross-breeze effects). One regression per sensor.

The regression uses `np.linalg.lstsq` (OLS) with automatic fallback to ridge regression (`np.linalg.solve` with L2 penalty) when the condition number indicates collinear effectors. T-statistics flag negligible gains (|gain| < 0.05°F/hr AND |t-stat| < 2.0).

**What it extracts:**
- **Effector × sensor gain matrix**: heating rate (°F/hr) and effective delay for each (effector, sensor) pair. Multiple effectors active simultaneously? The regression decomposes their contributions.
- **Solar gain profiles**: per-sensor, per-hour-of-day gain coefficients.
- **Tau per sensor**: `tau_base` (sealed envelope time constant).
- **Window coupling coefficients**: per-window `β_w` (additional cooling rate when window is open) and cross-breeze interaction terms `β_{ww'}`. The simulator computes effective tau as `1 / (1/tau_base + Σ β_w × open_w + Σ β_{ww'} × open_w × open_w')`.

**Config-driven:** Effectors and sensors enumerated from `weatherstat.yaml`. Adding a device or sensor = YAML edit + rerun.

**Output:** `~/.weatherstat/thermal_params.json` — the full coupling matrix, tau fits, and solar profiles. Run via `just sysid`.

**Dependencies:** numpy, scipy (curve_fit), pandas. No ML frameworks required.

#### 3c. Prediction Interface

The thermal model exposes a clean prediction interface to the controller:

```python
predict(state: HouseState, scenarios: list[TrajectoryScenario],
        params: SimParams, horizons: list[int]) -> (target_names, prediction_matrix)
```

`HouseState` bundles all environmental state (current room temps, outdoor temp, forecast temps, window states, hour of day, recent HVAC history). `TrajectoryScenario` encodes the HVAC plan (thermostat trajectories with delay/duration, blower and mini-split modes). The controller treats this as a black box — it provides state and candidate actions, and receives `prediction_matrix[i, j]` (predicted temperature for scenario `i` at target `j`, where targets are room × horizon combinations).

**Vectorization:** Activity matrices `(n_scenarios, n_total_steps)` are built per effector using numpy broadcasting over delay/duration parameters. Total effector forcing is pre-computed per sensor via slice operations, so the Euler integration loop (72 timesteps) contains only numpy vector operations over all scenarios. This reduces Python loop iterations from ~4.3M to ~576 for a typical 7400-scenario sweep.

---

### 4. Controller (`ml/src/weatherstat/control.py`)

Decides what HVAC actions to take right now, using receding-horizon optimization (MPC with trajectory search).

**Each control cycle:**
1. Read current house state (temperatures, HVAC states, window states, weather).
2. Fetch weather forecast for the prediction horizon.
3. Enumerate candidate trajectories: each thermostat gets a delay × duration grid (e.g., "delay 1h, heat for 2h, coast"), crossed with blower modes and mini-split target temperatures (~5,000–15,000 scenarios depending on config).
4. For each candidate: forward-simulate all sensor temperatures. Score the resulting trajectories against comfort schedules and energy costs.
5. Select the trajectory with the best score.
6. Emit electronic commands for the executor.

**Trajectory search:** Slow effectors (thermostats, 45-75 min lag) are parameterized as `[OFF × delay] → [ON × duration] → [OFF × remainder]`. Blowers use constant modes (off/low/high). Mini splits are treated as regulating effectors: the sweep generates target temperatures derived from the comfort schedule (off + preferred), not mode permutations. The boiler timeline is derived as the OR of both thermostat timelines. Blowers follow their zone thermostat (no heat to redistribute when the slab/radiators aren't active).

**Receding horizon:** Only the immediate action matters. A trajectory of "delay 2h then heat" means "stay off now." At the next 15-minute cycle, the controller re-evaluates with fresh data and may choose differently.

**Scoring:**

Two-layer comfort cost model:
1. **Continuous:** Quadratic penalty for any deviation from `preferred` temperature, weighted asymmetrically by `cold_penalty` (below preferred) and `hot_penalty` (above preferred). This gives the optimizer a gradient everywhere — it always prefers temperatures closer to preferred, not just "anywhere in the band."
2. **Hard rails:** Steep additional penalty (10×) for exceeding `min`/`max` bounds.

```
score = sum over (sensors, horizons) of:
    (T_predicted - preferred)^2 * penalty_weight        # continuous
  + hard_rail_penalty(T_predicted, min, max)             # steep outside bounds
  + energy_cost(actions, duration)
```

Window-open states widen the comfort band to avoid fighting ventilation.

**Horizon weighting:** Closer predictions weighted higher (more accurate, more actionable).

**Energy cost scaling:** Gas zone cost is proportional to `duration / max_horizon`. Mini-split cost is proportional to expected activity (target vs outdoor temperature within the proportional band).

**Physical constraints:**
- Blowers forced off when their zone's thermostat is off (no cold air circulation from radiator coils).
- Setpoint clamps (absolute safety bounds: 62–78°F).
- Hold times: 10-minute minimum between setpoint changes; 2-hour minimum between mini-split mode changes.
- Mode hold window: per-device configurable hours (e.g., 10pm–7am) during which mini-split mode changes are forbidden — only silent target temperature adjustments are allowed.
- Cold-sensor override: force immediate zone heating (delay=0) when any sensor is significantly below comfort minimum.

**Output:** `ControlDecision` JSON containing recommended device states, predicted outcomes, trajectory info, and decision reasoning.

---

### 5. Executor (`ha-client/src/executor.ts`)

Applies the controller's commands to Home Assistant with safety checks.

**Behavior:**
- Reads the latest `ControlDecision` JSON.
- For each device command, checks current HA state (lazy execution — skip if already in desired state).
- Calls HA services: `climate.set_temperature`, `climate.set_hvac_mode`, `fan.set_preset_mode`, etc.
- Detects manual overrides: if device state doesn't match what the executor last set, a human intervened. Respects the override until the next control cycle (or `--force`).
- Persists executor state for override tracking across restarts.

**Safety:**
- Never executes without explicit `--live` flag.
- Override detection prevents fighting with manual adjustments.
- All actions logged for audit.

---

### 6. Advisory System (`ml/src/weatherstat/advisory.py`)

Generates human-actionable recommendations for things the system can't do electronically. Uses the physics simulator to evaluate whether toggling window states would improve comfort, given the committed electronic plan.

**Advisory types:**
- **Free cooling:** "Open [window] — outdoor temp is lower, room cools toward comfort range."
- **Close windows:** "Close [window] — heating efficiency improves with window sealed."

**Evaluation:** After the trajectory sweep commits to an electronic plan, the advisory system re-runs `predict()` for each window, toggling its state. If toggling improves comfort cost (measured against original, non-adjusted constraint schedules) by more than the effort threshold (from YAML `advisory.effort_cost`), an advisory is generated. Advisory messages identify the most-affected constrained sensor (data-driven from the constraint list and temperature deltas).

**Key design constraint:** The electronic plan does NOT change based on advisories. The controller commits to the best electronic trajectory given current window states. Advisories are a separate, post-sweep evaluation. If the human acts on the advisory (e.g., opens a window), the next 15-minute cycle re-evaluates with the new state and naturally adjusts.

**Dispatch:**
- All triggered advisories are combined into a single notification (no spam).
- Per-window cooldown timers prevent notification fatigue.
- Quiet hours suppress push notifications (still logged).
- Notifications use a fixed tag so each control cycle replaces the previous advisory.

---

### 7. Safety System (`ml/src/weatherstat/safety.py`)

Detects infrastructure problems that prevent the control loop from working.

**Check types:**
- **Thermostat mode check:** Detects when a thermostat's `hvac_mode` is "off" while the controller wants to heat — setting the target has no effect in this state.
- **Generic device health:** Iterates health check definitions from effector configs (e.g., `effectors.boiler.navien.health`). Fetches each entity's current state from HA and compares against configured min/max thresholds.

**Dispatch:** Safety alerts bypass quiet hours (these are urgent). Cooldown tracking prevents notification fatigue. Alerts reuse the advisory notification infrastructure.

**Configuration-driven:** Health checks are defined per-effector in YAML with entity ID, min/max thresholds, severity, and message. Adding a health check = YAML edit.

---

### 8. Decision Log (`ml/src/weatherstat/decision_log.py`)

Records every control decision with full context.

**Stores:**
- Timestamp, live/dry-run mode.
- Input state snapshot (outdoor conditions, all room temperatures, HVAC states).
- Predicted outcomes at each horizon per room.
- Selected action (setpoints, blowers, mini-splits) and trajectory info.
- Comfort cost, energy cost, total cost.

**Outcome backfill:** At the start of each control cycle, the system checks whether enough time has elapsed to compare predictions to actual temperatures. For each horizon (1h, 2h, 4h, 6h), it finds the closest collector snapshot and records prediction error. This enables systematic accuracy tracking without a separate process.

**Used by:**
- Humans debugging decisions ("why did it turn on the heat at 3 AM?").
- Prediction error analysis: compare predicted vs actual temperatures at each logged horizon.
- Comfort performance dashboard (`just comfort`) for capacity analysis.

---

### 9. Comfort Dashboard (`scripts/plot_comfort.py`)

Answers "is the system working as designed?" at a glance. Run via `just comfort`.

**Output:** A multi-panel PNG with:
- **Summary bar:** Per-sensor horizontal stacked bar showing % in comfort band, % too cold (capacity-limited vs control opportunity), % too hot (capacity-limited vs control opportunity).
- **Temperature traces:** Per-sensor time series with comfort band overlay (green fill = min/max, dashed = preferred), violation shading (red = below min, orange = above max).
- **Outdoor + effector panel:** Outdoor temperature with heating state overlays.
- **Prediction accuracy** (optional, `--predictions`): Error histogram by horizon with MAE and bias.

**Capacity analysis:** Violations are classified as "capacity exceeded" (all dedicated effectors at max — building physics problem) vs "control opportunity" (dedicated effectors had headroom). Dedicated effectors are identified per sensor: zone thermostat (from coupling matrix) plus any name-matched mini split or blower. Cross-talk gains from effectors that primarily serve other sensors are excluded — the optimizer already balances those trade-offs.

**Console output:** Summary table with per-sensor breakdown and dedicated effector list.

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

The controller treats the thermal model as a black box. `HouseState` bundles all environmental inputs; `TrajectoryScenario` encodes all HVAC plans. The prediction engine can be swapped without touching sweep logic, scoring, or constraints.

### Controller -> Executor

**Interface:** `ControlDecision` JSON file in `~/.weatherstat/predictions/`.

Contains:
- Device states: thermostat setpoints, blower modes, mini-split modes + target temperatures.
- Predictions: per-sensor, per-horizon temperatures.
- Trajectory info: delay and duration per zone.
- Timestamp and live/dry-run flag.

### Controller -> Advisory System

**Interface:** Function calls within the control loop:
1. `evaluate_window_advisories(state, winning_scenario, sim_params, schedules, base_hour)` — runs `predict()` per window toggle, returns advisories.
2. `process_advisories(advisories, live, notification_target, current_hour)` — applies cooldowns, quiet hours, dispatches to HA.

### Configuration -> Everything

**Interface:** `weatherstat.yaml` parsed by `yaml-config.ts` (TS) and `yaml_config.py` (Python).

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

See `docs/EXPERIMENTS.md` for detailed experiment results and metrics. The ML training pipeline has been archived to `archive/ml/` for reference.
