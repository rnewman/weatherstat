# Plan: Grey-Box Forward Simulator (`simulator.py`)

## Context

The control loop works end-to-end but has a fundamental weakness: the ML model
can't answer counterfactual questions reliably. It learned "heat is ON when cold"
(correlation), not "heat ON causes warming" (causation). We patch this with
Newton floor/ceiling guardrails and cold-room overrides, but these are band-aids.

Sysid now gives us the physical parameters we need: per-sensor tau, per-effector
gain and delay for each sensor, and solar profiles. The next step is a forward
simulator that uses these parameters to predict temperature trajectories under
any HVAC scenario — replacing ML predictions in the sweep with physics-based
predictions that handle counterfactuals correctly.

**Goal:** Create `simulator.py` that the controller can call instead of (or
alongside) the ML models. Same output format, same sweep interface, but
physics-based predictions using sysid parameters.

## What the simulator computes

For each sensor, at each 5-minute timestep:

```
dT/dt = (T_outdoor(t) - T_sensor(t)) / tau(t)          # envelope loss
      + Σ_e gain_e * activity_e(t - lag_e)               # effector heating
      + solar_gain(hour_of_day(t))                        # solar
```

Euler integration: `T(t + dt) = T(t) + dt * dT/dt` where dt = 1/12 hour.

- `tau(t)` switches between sealed/ventilated based on window state
- `T_outdoor(t)` uses forecast outdoor temps (piecewise hourly, from forecast.py)
- `activity_e(t)` is the proposed HVAC state (the scenario being evaluated)
- `lag_e` is the delay from sysid (the effect doesn't start until lag minutes later)
- `solar_gain` is the hour-of-day profile from sysid

## Design decisions

### Integration with the controller

The controller calls `_batch_predict(base_row, overrides_list, models)` which
returns `(target_names, np.ndarray)` of shape `(n_scenarios, n_targets)`.
Targets are like `"upstairs_temp_t+12"` (room + horizon in 5-min steps).

The simulator provides a drop-in replacement:

```python
def batch_simulate(
    current_temps: dict[str, float],      # sensor_name -> current °F
    outdoor_temp: float,
    forecast_temps: list[float],          # hourly forecast [h+1..h+12]
    window_states: dict[str, bool],       # window_name -> is_open
    scenarios: list[HVACScenario],        # from generate_scenarios()
    params: SimParams,                    # loaded from thermal_params.json
    hour_of_day: int,
    horizons: list[int],                  # [12, 24, 48, 72] (5-min steps)
) -> tuple[list[str], np.ndarray]        # same shape as _batch_predict
```

This returns the same `(target_names, predictions)` tuple, so the controller's
scoring, constraint checking, and decision logic don't change at all.

### HVAC scenario → effector activity

Each `HVACScenario` specifies thermostat on/off, blower modes, mini split modes.
The simulator converts these to numeric activity levels using the same encodings
from sysid's `EffectorSpec.encoding`:

- Thermostat upstairs heating=True → `thermostat_upstairs` activity = 1.0
- Blower office mode="high" → `blower_office` activity = 2.0
- Mini split bedroom mode="heat" → `mini_split_bedroom` activity = 1.0
- Navien fires when either thermostat is on (same logic as `build_hvac_overrides`)

### Delay handling

Sysid gives `best_lag_minutes` per effector-sensor pair. During simulation:

- At t=0 the scenario's HVAC state takes effect
- The thermal effect on a sensor doesn't appear until `lag` minutes later
- Before the lag, the effector contribution is zero for that sensor

But crucially, we also need to account for **pre-existing thermal charge**. If the
boiler has been running before t=0, there's already heat in the slab that will
continue radiating. The simulator needs the recent HVAC history (last ~90 min)
to model this correctly.

**Approach:** Accept `recent_history: dict[str, list[float]]` — the last N
5-minute activity values per effector (from the current snapshot data). During
simulation, for timesteps where `t < lag`, look back into the real history.
After `t >= lag`, use the scenario's proposed state.

### What sensors to simulate

Sysid processes all 13 temperature sensors. But the controller only needs
predictions for the 8 rooms in `PREDICTION_ROOMS`. The simulator should predict
all rooms that have sysid parameters with sufficient data (i.e., non-negligible
gains from at least one effector), and return targets matching the controller's
expected format.

### Solar gain

Sysid provides per-sensor, per-hour solar gain coefficients. During simulation,
look up the gain for the current hour of each timestep. Interpolation between
hours is unnecessary — the underlying data is hourly and the model fit is noisy
enough that step changes are fine.

Winter data only means the solar profiles are winter-specific. As more data
accumulates through spring/summer, sysid will refit with seasonal variation.

### Loading parameters

`SimParams` is a frozen dataclass loaded from `data/thermal_params.json`.
Provides fast lookup:

```python
@dataclass(frozen=True)
class SimParams:
    taus: dict[str, tuple[float, float]]       # sensor -> (sealed, vent)
    gains: dict[tuple[str, str], tuple[float, float]]  # (eff, sensor) -> (gain, lag)
    solar: dict[tuple[str, int], float]        # (sensor, hour) -> gain
    sensors: list[str]                         # sensor names with params
    effectors: list[str]                       # effector names
```

Loaded once at startup, same as ML models. A `load_sim_params()` function
reads the JSON and builds the lookup structures.

## Implementation plan

### Files to create/modify

- **Create:** `ml/src/weatherstat/simulator.py`
- **Modify:** `ml/src/weatherstat/control.py` (add `--physics` flag, wire up simulator)
- **Modify:** `Justfile` (add `just control-physics` convenience command)

### 1. `SimParams` and loading (`load_sim_params`)

Load `data/thermal_params.json`, build lookup dicts for fast access during
simulation. Handle missing params gracefully (sensors with no sysid data
fall back to YAML tau + zero gains = pure Newton cooling).

### 2. `simulate_sensor` — single sensor, single scenario

Core simulation loop for one sensor under one scenario:

```python
def simulate_sensor(
    sensor: str,
    current_temp: float,
    outdoor_temp: float,
    forecast_temps: list[float],
    tau_sealed: float,
    tau_vent: float,
    is_ventilated: bool,
    effector_activities: dict[str, list[float]],  # eff -> activity per 5-min step
    gains: dict[str, tuple[float, float]],        # eff -> (gain, lag_minutes)
    solar_profile: dict[int, float],              # hour -> gain
    start_hour: int,
    n_steps: int,                                 # total 5-min steps to simulate
) -> list[float]:                                 # temp at each step
```

- Euler integrate at 5-min steps
- Outdoor temp: use forecast (piecewise hourly) via the same logic as
  `piecewise_newton_prediction` in forecast.py
- Effector contribution: for each effector, look up activity at `t - lag`
  (from pre-built activity timeline that includes recent history + scenario)
- Solar: lookup by hour of current timestep
- Returns temperature at every step (caller picks horizon values)

### 3. `build_activity_timeline` — merge history + scenario

For each effector, build a complete activity timeline:
- Steps [-N..0]: from recent snapshot history (actual HVAC states)
- Steps [1..max_horizon]: from the scenario being evaluated

This handles the pre-existing thermal charge problem. The slab was heated
for the last 2 hours? That history feeds into the delay model naturally.

```python
def build_activity_timeline(
    effector: EffectorSpec,
    scenario_activity: float,     # proposed state for this effector
    recent_history: list[float],  # last N actual activities (oldest first)
    n_future_steps: int,
) -> list[float]:                 # full timeline including history
```

### 4. `batch_simulate` — all sensors, all scenarios

The public interface. For each scenario:
1. Convert HVACScenario → effector activities (using sysid encodings)
2. Build activity timelines (merge recent history + scenario)
3. Simulate each sensor
4. Extract temperatures at requested horizons
5. Pack into the same `(target_names, np.ndarray)` format as `_batch_predict`

Vectorization: the inner loop (Euler steps) runs ~144 steps (12h) per sensor
per scenario. With 8 sensors × 324 scenarios = ~2,592 simulations × 144 steps
= ~373K floating point ops. This is fast enough in plain Python/numpy (~10ms).
If needed, vectorize across scenarios (all scenarios for one sensor at once).

### 5. Controller integration

Add a `--physics` flag to `control.py`:

```python
if args.physics:
    params = load_sim_params()
    target_names, pred_matrix = batch_simulate(
        current_temps, outdoor_temp, forecast_temps,
        window_states, scenarios, params, base_hour, CONTROL_HORIZONS,
    )
else:
    overrides_list = [build_hvac_overrides(s, current_split_temps) for s in scenarios]
    target_names, pred_matrix = _batch_predict(base_row, overrides_list, models)
```

Everything downstream (scoring, constraints, cold-room override, decision
building) stays identical. The Newton floor is no longer needed for physics
mode (the simulator IS the physics), but can remain as a no-op safety check.

### 6. Comparison mode

Add `just simulate` that runs a single control cycle with both ML and physics
predictions side by side, printing a comparison table. This lets us validate
the simulator against ML predictions before switching over.

## Edge cases

**Sensors without sysid data** (living_room_climate_temp, basement_temp):
Simulate with pure Newton (zero effector gains). These sensors contribute
to comfort scoring only if they're in PREDICTION_ROOMS.

**Effectors not in scenario** (e.g., navien): Derived from thermostat state
(fires when either thermostat is on), same logic as `build_hvac_overrides`.

**No forecast data**: Fall back to constant outdoor temp (current value) for
all horizons. Same degradation as the ML path.

**Negative gains from sysid**: Some gains are negative (e.g., blower_family_room
on piano_temp = -0.34°F/hr — redistributing heat away). The simulator handles
this naturally; negative gains mean the effector cools that sensor.

**Very short recent history**: If collector has < 90 min of history, pad with
zeros (effectors assumed off). Conservative — may underestimate slab charge.

## Verification

1. `just control-physics` runs without error
2. Physics predictions are physically reasonable:
   - All-off: temperatures decay toward outdoor (Newton)
   - Heating on: temperatures rise, with delay matching sysid lag
   - Mini split bedroom on: bedroom warms quickly (~1.5°F/hr), others barely affected
   - Blower effects: redistribution visible (some sensors warm, some cool)
3. Comparison mode shows physics and ML broadly agree on direction
4. Physics correctly handles counterfactuals: "heat off in cold weather" → cooling
   (unlike ML which may predict stable temps)
5. Physics sweep picks the same or better decisions as ML sweep
   (especially when Newton floor was doing heavy lifting)
6. Performance: full sweep completes in < 100ms

## Dependencies

- `numpy` — already available
- No new dependencies. Sysid JSON loading uses stdlib `json` + dataclasses.
  `piecewise_newton_prediction` from `forecast.py` can be reused or its logic
  inlined (it's 20 lines).
