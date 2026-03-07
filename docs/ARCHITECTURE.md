# Weatherstat Architecture

## Goal

Maximize thermal comfort across a multi-room house while minimizing energy cost, by anticipating rather than reacting to temperature changes.

The primary challenge is hydronic floor heating with 2-4 hour thermal lag, compounded by solar gain, inter-room coupling, and Seattle's variable marine climate. A conventional thermostat — purely reactive — is the wrong tool for this job. The system must predict hours ahead and act preemptively.

## Design principles

1. **Physics first, ML second.** The house is a thermal system with known structure. Use physics to model what we understand (envelope loss, heating input, thermal mass) and ML only for what we can't easily model from first principles (solar gain patterns, inter-room effects).

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
            |  Sweep candidate actions                 |
            |  Score with thermal model predictions    |
            |  Select best action set                  |
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
- Deduplicates by rounding timestamps to the snapshot interval.
- Writes rows to SQLite (`data/snapshots/snapshots.db`).
- Schema generated dynamically from YAML — adding a sensor automatically adds a column.

**Output:** SQLite database, one row per 5-minute interval, each row a complete house state snapshot.

**Operational:** `just collect-durable` runs with auto-restart and health monitoring. `just health` checks data freshness.

**Should also store:** weather forecast snapshots (the hourly forecast at each collection time) for training the solar/weather models. This is a gap today.

---

### 3. Thermal Model

The core prediction engine. A grey-box model: physics provides the structure, data provides the parameters.

#### 3a. Physics Core

Models the known thermal dynamics of each room:

```
dT/dt = (1/C) * [
    Q_heat(t - lag)              # floor heating, delayed by slab thermal mass
    + Q_split(t)                 # mini split, near-instant
    + Q_solar(time, clouds)      # solar gain, room-dependent
    + sum(k_ij * (T_j - T_i))   # inter-room heat transfer
    - k_out * (T_i - T_outdoor)  # envelope loss
    - Q_vent(windows)            # ventilation loss when windows open
]
```

**Parameters (fitted by sysid):**
- `tau_sealed`, `tau_ventilated` per sensor — envelope loss time constants, fitted from all nighttime HVAC-off periods (stage 1). YAML values from single-night fit (Feb 2026) serve as fallbacks.
- `Q_heat` gain and `lag` per effector × sensor — from regression on Newton residuals (stage 2). Captures floor heat delays (30–90 min), mini split immediacy (2–10 min), and cross-zone coupling.
- `Q_solar` per sensor × hour-of-day — solar gain profile from hour indicators in residual regression (stage 2).
- `k_ij` inter-room coupling — implicit in the effector × sensor gain matrix (e.g., upstairs thermostat warms piano at +1.0°F/hr).

**Integration:** Euler steps at 5-minute resolution, chaining hourly weather forecast segments for the outdoor temperature trajectory. `forecast.py` already implements piecewise Newton for the cooling-only case; this extends it with heating and solar terms.

#### 3b. Solar Gain Model

Solar gain depends on window orientation, size, shading, and cloud patterns — hard to model from first principles without building geometry, but easy to learn from data.

**Approach:**
- During heating-off daytime periods, the deviation from Newton cooling is predominantly solar.
- Extract residuals: `Q_solar_observed = C * (dT/dt_actual - dT/dt_newton_no_solar)`.
- Train a small model (random forest or lookup table) on `(hour_of_day, day_of_year, cloud_cover, room)` -> `Q_solar`.
- Each room gets its own solar profile reflecting window orientation and exposure.

Even a few weeks of collector data should produce a useful initial model since the solar pattern is strongly periodic.

#### 3c. System Identification (`ml/src/weatherstat/sysid.py`)

Fits all thermal model parameters from observed data using a two-stage approach that uses ALL collector data, not just rare clean episodes.

**Stage 1 — Tau fitting:** For each temperature sensor, selects all nighttime (10pm–6am) periods where all HVAC effectors are off. Fits Newton cooling (`T(t) = T_out + (T_0 - T_out) * exp(-t/tau)`) via `curve_fit` on each contiguous segment, separated by window state. Multiple segments → weighted median → tighter estimate than single-night fitting. Produces `tau_sealed` and `tau_ventilated` per sensor.

**Stage 2 — Effector gains and solar profiles:** With tau fitted, computes Newton residuals at every timestep (`dT/dt_observed - dT/dt_newton`). These residuals are explained by a linear regression on lagged effector activity (coarse time bins capturing delay) and hour-of-day indicators (capturing solar gain). One regression per sensor; coefficients across all sensors form the full coupling matrix.

**What it extracts:**
- **Effector × sensor gain matrix**: heating rate (°F/hr) and effective delay for each (effector, sensor) pair. Multiple effectors active simultaneously? The regression decomposes their contributions.
- **Solar gain profiles**: per-sensor, per-hour-of-day gain coefficients.
- **Tau per sensor**: envelope loss time constants (sealed and ventilated), more robust than single-night fits.

**Config-driven:** Effectors and sensors enumerated from `weatherstat.yaml`. Adding a device or sensor = YAML edit + rerun.

**Output:** `data/thermal_params.json` — the full coupling matrix, tau fits, and solar profiles. Run via `just sysid`.

**Safeguards:** Sanity checks (ventilated tau must be < sealed tau), minimum data threshold for regression (500 rows), negligible-gain flagging (|gain| < 0.05°F/hr AND |t-stat| < 2.0), ridge regression fallback for collinear effectors.

#### 3d. Prediction Interface

The thermal model exposes a single interface to the controller:

```python
predict(state, actions, horizons) -> {room: {horizon: temperature}}
```

The controller doesn't know or care whether predictions come from physics simulation, ML, or a hybrid. This is the key abstraction boundary.

**Implementation:** Forward-simulate from current state using physics core + solar model + learned parameters + weather forecast, under the proposed action set, at each requested horizon.

---

### 4. Controller

Decides what HVAC actions to take right now, using receding-horizon optimization (MPC).

**Each control cycle:**
1. Read current house state (temperatures, HVAC states, window states, weather).
2. Fetch weather forecast for the prediction horizon.
3. Enumerate candidate action sets (thermostat on/off x blower levels x mini split modes).
4. For each candidate: call `predict(state, actions, horizons)`. Score the resulting trajectories against comfort schedules and energy costs.
5. Select the action set with the best score.
6. Emit electronic commands (for executor) and human advisories (for notifications).

**Scoring:**

```
score = sum over (rooms, horizons) of:
    comfort_violation(T_predicted, comfort_band, penalty_weights)
  + energy_cost(actions, duration)
  + effort_cost(advisory_actions)
```

Comfort violations are asymmetric: being too cold incurs a higher penalty than being too warm (configurable per room and time of day via YAML). Window-open states widen the comfort band to avoid fighting ventilation.

**Horizon weighting:** Closer predictions weighted higher (more accurate, more actionable).

**Physical constraints:**
- Blowers forced off when their zone's thermostat is off (no cold air circulation).
- Setpoint clamps (min/max safety bounds).
- Hold times (minimum interval between changes to avoid short-cycling).
- Cold-room override: force zone heating when any room exceeds a threshold below comfort minimum.

**Action types:**
- **Electronic** (executed automatically): thermostat setpoints, mini split mode/target, blower speed.
- **Advisory** (human notification): open/close windows, adjust blinds. Penalized by effort cost and suppressed during quiet hours.

**Output:** `ControlDecision` JSON containing recommended device states, predicted outcomes, and decision reasoning.

**Future (full MPC):** With a fast forward simulator, extend from single-step action selection to multi-step trajectory optimization. "Turn heat on now, off in 2 hours, on again at 6 AM" becomes a single optimized plan rather than three independent decisions.

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
- Evaluated after each control cycle.
- Cooldown timers prevent notification fatigue.
- Quiet hours suppress push notifications (still logged).
- HA persistent notifications (replaces stale notification via `notification_id`).

**Future advisories:**
- "Close blinds — solar gain will overshoot comfort."
- "Bedroom will be cold at wake-up — consider pre-heating." (or just pre-heat automatically)
- "Office is 3 degrees warmer than expected — something changed."

---

### 7. Decision Log (`ml/src/weatherstat/decision_log.py`)

Records every control decision with full context.

**Stores:**
- Timestamp.
- Input state snapshot (all temperatures, HVAC states, weather).
- Candidate actions evaluated and their scores.
- Selected action and reasoning.
- Predicted outcomes at each horizon.

**Used by:**
- Humans debugging decisions ("why did it turn on the heat at 3 AM?").
- Online learning: compare predicted vs actual temperatures at each logged horizon.
- Systematic prediction error analysis.

---

### 8. Online Learning

Continuously improves thermal model parameters by comparing predictions to outcomes.

**Mechanism:**
- Each control cycle records predictions at specific horizons.
- When those horizons arrive, compare predicted temperatures to actuals from the collector.
- Update parameters:
  - Heating gain/lag: if post-heating temps consistently undershoot, increase gain estimate.
  - Solar model: if daytime temps consistently exceed predictions, solar is underestimated.
  - Tau values: if overnight cooling is faster/slower than predicted, adjust.
- Track prediction errors over time to detect model drift or building changes.

**This is not retraining.** It's incremental parameter adjustment (exponential moving averages or similar). Full retraining of the solar model happens on a separate schedule.

---

## Component interfaces

### Collector -> Thermal Model

**Interface:** SQLite database (`data/snapshots/snapshots.db`).
- Schema driven by YAML config.
- One row per 5-minute interval.
- Columns: timestamp, all temperature sensors, HVAC states (action, mode, target per device), window states, weather, humidity.

The collector is language-agnostic from the model's perspective.

### Thermal Model -> Controller

**Interface:** `predict(state, actions, horizons) -> {room: {horizon: temperature}}`

The controller treats the thermal model as a black box. This is the key abstraction: the prediction engine can be swapped without touching sweep logic, scoring, or constraints.

### Controller -> Executor

**Interface:** `ControlDecision` JSON file in `data/predictions/`.

Contains:
- `commands`: device_id -> desired state (mode, target, speed).
- `predictions`: predicted temperatures per room per horizon.
- `reasoning`: human-readable explanation.
- `timestamp`: when decided.

### Controller -> Advisory System

**Interface:** Function call within the control loop. Receives current state + control decision, emits notifications.

### Configuration -> Everything

**Interface:** `weatherstat.yaml` parsed by `yaml-config.ts` (TS) and `yaml_config.py` (Python).

---

## Future components

### Config Generator

Auto-generate `weatherstat.yaml` from Home Assistant entity discovery.

- Query HA for all climate, fan, sensor, and binary_sensor entities.
- Match naming patterns to identify device types and room associations.
- Use HA area assignments and labels for room grouping.
- Present a draft config for human review.
- Generate comfort schedules with sensible defaults.

The current YAML is hand-maintained. This works for one house but doesn't generalize.

### Virtual Thermostats

Expose per-room comfort targets as HA climate entities, editable from the dashboard or phone.

- Each room appears as a `climate` entity with target_temp_high and target_temp_low.
- User adjusts from the dashboard; control loop reads targets each cycle.
- YAML comfort schedules become defaults.
- Solves the "adjust from bed at 2 AM" problem.

### Anomaly Detection

Detect when the house behaves differently than the model predicts.

- Persistent prediction error above a threshold triggers investigation.
- "Cooling rate doubled overnight — window left open?"
- "Kitchen consistently warmer at 6 PM — cooking pattern."
- Over time, distinguish recurring patterns from true anomalies.

### Shade/Cover Control

Manage solar gain via motorized blinds. Solar gain model predicts which rooms will overshoot; close blinds preemptively. Advisory-only until motorized covers are installed.

### ERV Integration

Control an energy recovery ventilator for air exchange without thermal penalty. Integrates into the thermal model as a controlled ventilation term alongside window state.

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

See `docs/EXPERIMENTS.md` for detailed experiment results and metrics.
