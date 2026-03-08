# Weatherstat Architecture

## Goal

Maximize thermal comfort across a multi-room house while minimizing energy cost, by anticipating rather than reacting to temperature changes.

The primary challenge is hydronic floor heating with 2-4 hour thermal lag, compounded by solar gain, inter-room coupling, and Seattle's variable marine climate. A conventional thermostat — purely reactive — is the wrong tool for this job. The system must predict hours ahead and act preemptively.

## Design principles

1. **Physics first.** The house is a thermal system with known structure. Use physics to model what we understand (envelope loss, heating input, thermal mass) and learn only the parameters we can't derive from first principles (gains, delays, solar profiles).

2. **Causality over correlation.** The system must answer counterfactual questions: "what happens if I turn on the heat now vs. in 2 hours?" This requires a model that understands cause and effect, not one trained on observational data where actions and conditions are confounded.

3. **Configuration-driven.** Adding a sensor, room, or device should be a YAML edit, not a code change. The system discovers its own shape from configuration.

4. **Safe by default.** Dry-run before live. Override detection. Hold times. Setpoint clamps. The system should be hard to misconfigure into damaging equipment or freezing pipes.

5. **Observable.** Every decision is logged with its inputs, alternatives considered, and reasoning. Predictions are saved for later comparison with reality.

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

Single source of truth for all entity IDs, device definitions, room topology, comfort schedules, energy costs, and thermal parameters.

**Contents:**
- `location` — lat/lon/elevation/timezone (for solar position)
- `sensors` — every monitored temperature and humidity entity, with column names
- `devices` — thermostats, mini splits, blowers, boiler, with mode encodings
- `windows` — window/door sensors with room associations
- `rooms` — room definitions mapping to temperature columns and heating zones
- `comfort` — per-room, time-of-day comfort bands with asymmetric penalty weights
- `thermal` — fitted tau values (sealed and ventilated) per room
- `energy_costs` — per-device energy cost for the optimizer
- `advisory` — effort costs, quiet hours, cooldown timers

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
- Writes rows to SQLite (`data/snapshots/snapshots.db`).
- Schema generated dynamically from YAML — adding a sensor automatically adds a column.

**Output:** SQLite database, one row per 5-minute interval, each row a complete house state snapshot.

**Operational:** `just collect-durable` runs with auto-restart and health monitoring. `just health` checks data freshness.

---

### 3. Thermal Model

The core prediction engine. A grey-box model: physics provides the structure, data provides the parameters.

#### 3a. Physics Core (`ml/src/weatherstat/simulator.py`)

Forward-simulates room temperatures by Euler-integrating the thermal dynamics of each sensor at 5-minute resolution:

```
dT/dt = (T_outdoor - T) / tau
        + Σ gain_e * activity_e(t - lag_e)     # effector heating, delayed
        + solar(hour_of_day)                    # solar gain profile
```

Where:
- `tau` is the envelope loss time constant (sealed or ventilated, depending on window state)
- Each effector `e` contributes a gain (°F/hr per activity unit) delayed by its fitted lag
- Solar gain is a per-sensor, per-hour profile from sysid

**Integration:** Euler steps at 5-minute resolution, chaining hourly weather forecast segments for the outdoor temperature trajectory.

**Batch simulation:** The controller calls `predict()` with thousands of candidate scenarios. Each scenario specifies effector timelines (trajectory-parameterized), and the simulator evaluates all of them against the same initial conditions. Internally, Euler integration is vectorized across all scenarios using numpy — the outer loop is over sensors and timesteps, with numpy broadcasting handling all scenarios simultaneously.

#### 3b. System Identification (`ml/src/weatherstat/sysid.py`)

Fits all thermal model parameters from observed collector data using a two-stage approach.

**Stage 1 — Tau fitting (scipy `curve_fit`):** For each temperature sensor, selects all nighttime (10pm–6am) periods where all HVAC effectors are off. Fits Newton cooling (`T(t) = T_out + (T_0 - T_out) * exp(-t/tau)`) via nonlinear least squares on each contiguous segment, separated by window state. Multiple segments → weighted median → tighter estimate than single-night fitting. Produces `tau_sealed` and `tau_ventilated` per sensor.

**Stage 2 — Effector gains and solar profiles (numpy linear regression):** With tau fitted, computes Newton residuals at every timestep (`dT/dt_observed - dT/dt_newton`). These residuals are explained by a linear regression on lagged effector activity (coarse time bins capturing delay) and hour-of-day indicators (capturing solar gain). One regression per sensor; coefficients across all sensors form the full coupling matrix.

The regression uses `np.linalg.lstsq` (OLS) with automatic fallback to ridge regression (`np.linalg.solve` with L2 penalty) when the condition number indicates collinear effectors. T-statistics flag negligible gains (|gain| < 0.05°F/hr AND |t-stat| < 2.0).

**What it extracts:**
- **Effector × sensor gain matrix**: heating rate (°F/hr) and effective delay for each (effector, sensor) pair. Multiple effectors active simultaneously? The regression decomposes their contributions.
- **Solar gain profiles**: per-sensor, per-hour-of-day gain coefficients.
- **Tau per sensor**: envelope loss time constants (sealed and ventilated), more robust than single-night fits.

**Config-driven:** Effectors and sensors enumerated from `weatherstat.yaml`. Adding a device or sensor = YAML edit + rerun.

**Output:** `data/thermal_params.json` — the full coupling matrix, tau fits, and solar profiles. Run via `just sysid`.

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
3. Enumerate candidate trajectories: each thermostat gets a delay × duration grid (e.g., "delay 1h, heat for 2h, coast"), crossed with blower modes and mini-split modes (~7,400 scenarios).
4. For each candidate: forward-simulate all room temperatures. Score the resulting trajectories against comfort schedules and energy costs.
5. Select the trajectory with the best score.
6. Emit electronic commands for the executor.

**Trajectory search:** Slow effectors (thermostats, 45-75 min lag) are parameterized as `[OFF × delay] → [ON × duration] → [OFF × remainder]`. Fast effectors (blowers, mini-splits) use constant modes. The boiler timeline is derived as the OR of both thermostat timelines. Blowers follow their zone thermostat (no heat to redistribute when the slab isn't active).

**Receding horizon:** Only the immediate action matters. A trajectory of "delay 2h then heat" means "stay off now." At the next 15-minute cycle, the controller re-evaluates with fresh data and may choose differently.

**Scoring:**

```
score = sum over (rooms, horizons) of:
    comfort_violation(T_predicted, comfort_band, penalty_weights)
  + energy_cost(actions, duration)
```

Comfort violations are asymmetric: being too cold incurs a higher penalty than being too warm (configurable per room and time of day via YAML). Window-open states widen the comfort band to avoid fighting ventilation.

**Horizon weighting:** Closer predictions weighted higher (more accurate, more actionable).

**Energy cost scaling:** For trajectory scenarios, gas zone cost is proportional to `duration / max_horizon` — a 2h heating plan costs less than a 6h plan.

**Physical constraints:**
- Blowers forced off when their zone's thermostat is off (no cold air circulation).
- Setpoint clamps (min/max safety bounds).
- Hold times (minimum interval between changes to avoid short-cycling).
- Cold-room override: force immediate zone heating (delay=0) when any room exceeds a threshold below comfort minimum.

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

Generates human-actionable recommendations for things the system can't do electronically.

**Current advisories:**
- **Free cooling:** "Open windows in [rooms] to cool down for free."
- **Close windows:** "Heating is active — close windows in [rooms]."

**Behavior:**
- Cooldown timers prevent notification fatigue.
- Quiet hours suppress push notifications (still logged).
- HA persistent notifications (replaces stale notification via `notification_id`).

**Current status:** The advisory infrastructure exists but is not integrated into the control loop. See `docs/plans/PLAN-8-virtual-effectors.md` for the plan to reconnect it via virtual effector modeling.

---

### 7. Decision Log (`ml/src/weatherstat/decision_log.py`)

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

---

## Component interfaces

### Collector -> Thermal Model

**Interface:** SQLite database (`data/snapshots/snapshots.db`).
- Schema driven by YAML config.
- One row per 5-minute interval.
- Columns: timestamp, all temperature sensors, HVAC states (action, mode, target per device), window states, weather, humidity, forecast snapshots.

The collector is language-agnostic from the model's perspective.

### Thermal Model -> Controller

**Interface:** `predict(state: HouseState, scenarios, params, horizons) -> (target_names, prediction_matrix)`

The controller treats the thermal model as a black box. `HouseState` bundles all environmental inputs; `TrajectoryScenario` encodes all HVAC plans. The prediction engine can be swapped without touching sweep logic, scoring, or constraints.

### Controller -> Executor

**Interface:** `ControlDecision` JSON file in `data/predictions/`.

Contains:
- Device states: thermostat setpoints, blower modes, mini-split modes/targets.
- Predictions: per-room, per-horizon temperatures.
- Trajectory info: delay and duration per zone.
- Timestamp and live/dry-run flag.

### Controller -> Advisory System

**Interface:** Function call within the control loop. Receives current state + control decision + simulator predictions, emits notifications.

### Configuration -> Everything

**Interface:** `weatherstat.yaml` parsed by `yaml-config.ts` (TS) and `yaml_config.py` (Python).

---

## Future components

### Virtual Effectors / Advisory-Driven Planning

Model human-actionable changes (windows, blinds, space heaters, doors) as virtual effectors in the trajectory sweep. The controller finds the best electronic-only plan AND evaluates whether a human action would improve outcomes. See `docs/plans/PLAN-8-virtual-effectors.md`.

### Online Learning

Continuously improve thermal model parameters by comparing predictions to outcomes. The decision log already records predicted vs actual temperatures. The next step is automatic parameter adjustment (exponential moving averages on gain/delay/tau) when systematic prediction errors appear.

### Virtual Thermostats

Per-room HA climate entities for user-adjustable comfort targets from the dashboard. Each room appears as a `climate` entity with target_temp_high and target_temp_low. YAML comfort schedules become defaults.

### Config Generator

Auto-generate `weatherstat.yaml` from Home Assistant entity discovery.

### Anomaly Detection

Detect when the house behaves differently than the model predicts. Persistent prediction error above a threshold triggers investigation ("cooling rate doubled overnight — window left open?").

### Shade/Cover Control

Manage solar gain via motorized blinds. Solar gain model predicts which rooms will overshoot; close blinds preemptively.

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
