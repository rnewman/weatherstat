# Plan: Phase 6 — Forecast Training, Retrospective HVAC, Pre-Heating

## Context: How the System Works

### The prediction problem

We're building a physics-based smart thermostat for a house with hydronic
floor heat (massive thermal lag: 2-4 hours from boiler firing to room warming).
The control loop runs every 15 minutes and asks: "what HVAC settings will
maximize comfort over the next 6 hours?"

To answer this, the controller sweeps ~324 HVAC combinations (2 thermostats ×
blower modes × mini-split modes) and **predicts** room temperatures at 1h, 2h,
4h, and 6h horizons for each combination. It then picks the combo that
minimizes a comfort cost function (quadratic penalty for being outside
per-room, time-of-day comfort bounds) plus a small energy tiebreaker.

### Two prediction engines

**ML models** (LightGBM, 40 models: 8 rooms × 5 horizons): Trained on
collector data (5-min snapshots since Feb 2026). Features include current
temperatures, HVAC states, weather, forecast, Newton cooling predictions.
Problem: the ML model learned correlations, not causation. "Heat is ON when
it's cold" ≠ "heat ON causes warming." It predicts temperatures stabilizing
around 72-73°F regardless of whether heating continues — the equilibrium it
always saw in training data.

**Physics simulator** (`simulator.py`, just completed): Euler-integrates
Newton's law of cooling with effector gains, delays, and solar profiles from
system identification (`sysid.py`). For each sensor:
```
dT/dt = (T_outdoor - T) / tau + Σ gain_e * activity_e(t - lag_e) + solar(hour)
```
Correctly predicts continued warming when heating is on (+3.6°F vs ML at 6h).
Handles counterfactuals causally. This is now the primary prediction engine.

### Forecast data flow

Weather forecasts come from Home Assistant (met.no integration). The collector
(TypeScript) fetches hourly forecasts every 5 minutes and stores them in
SQLite alongside sensor snapshots:
- `forecast_temp_{1..12}h` — hourly outdoor temps (12 columns)
- `forecast_condition_{1,2,4,6,12}h` — weather condition at key horizons
- `forecast_wind_{1,2,4,6,12}h` — wind speed at key horizons

4268 of 6734 snapshots have forecast data (collection started ~mid-Feb).

At inference time, `fetch_forecast()` fetches live forecasts from HA.
The physics simulator uses these for piecewise outdoor temp integration.
ML models use them as features (forecast outdoor temp, condition, wind).

### The train/serve skew bug

During training, ML models don't see the stored forecasts. Instead,
`add_forecast_features()` creates "perfect forecasts" by shifting
`outdoor_temp` forward in time. The model trains on zero-error forecasts
but runs inference on real met.no forecasts (typical error: 2-5°F in
Seattle winters). This is a train/serve skew that makes the model
over-trust forecast features.

### What we're dropping

The **baseline model** (hourly, 7-month historical stats, temp-only) was a
bootstrap for when we had limited 5-min data. With 25+ days of collector data
and the physics simulator, it no longer serves a purpose. The control loop
doesn't use it. The `--mode baseline` training path, `historical_hourly.parquet`,
and `load_baseline_data()` can be removed.

The **historical Parquet extraction** (`historical_full.parquet`) was a one-time
bootstrap from HA's 10-day raw history. The collector has accumulated more data
than this now. The `--collector-only` flag is already the default for `just train`.
We can simplify `load_full_data()` to be collector-only.

---

## Workstream 1: Use Stored Forecasts in Training

### Problem

Training creates fake "perfect" forecasts by shifting actual outdoor_temp.
Inference uses real met.no forecasts. This skew means the model doesn't
learn forecast error patterns (met.no systematically underestimates overnight
cooling in Seattle winters).

### Solution

In `add_forecast_features()` training mode: if the DataFrame has
`forecast_temp_1h` columns (from collector SQLite), use them directly
instead of shifting outdoor_temp. This is a small change — the feature
column names produced are the same either way.

### Files to modify

**`ml/src/weatherstat/features.py`** — `add_forecast_features()`:

Currently the training branch (line ~405-425) does:
```python
df[f"forecast_outdoor_temp_{label}"] = df["outdoor_temp"].shift(-shift)
```

Change to:
```python
# Use stored forecasts from collector if available
stored_col = f"forecast_temp_{label}"  # SQLite column name
if stored_col in df.columns:
    df[f"forecast_outdoor_temp_{label}"] = df[stored_col]
else:
    # Fall back to shifted actuals (pre-collector data)
    df[f"forecast_outdoor_temp_{label}"] = df["outdoor_temp"].shift(-shift)
```

Same for the hourly forecast temps (`forecast_outdoor_temp_{N}h`):
```python
stored = f"forecast_temp_{h}h"
if stored in df.columns:
    df[f"forecast_outdoor_temp_{h}h"] = df[stored]
else:
    df[f"forecast_outdoor_temp_{h}h"] = df["outdoor_temp"].shift(-shift)
```

For condition codes: stored column is `forecast_condition_{1,2,4,6,12}h` (text).
Need to encode with `encode_weather_condition()` before assigning.

For wind: stored column is `forecast_wind_{1,2,4,6,12}h` (numeric). Direct copy.

**`ml/src/weatherstat/extract.py`** — verify `load_collector_snapshots()`
loads the forecast columns. Since it does `SELECT *`, it should already
include them, but verify the column names match expectations.

### Cleanup: drop baseline and historical parquet paths

**`ml/src/weatherstat/train.py`**:
- Remove `load_baseline_data()` and the `--mode baseline` path
- Simplify `load_full_data()`: remove `collector_only` parameter, always
  load from collector only. Remove the historical_full.parquet merge.
- Remove `_resample_to_hourly()` (only used by baseline)
- Keep one training mode (what was `--mode full --collector-only`)

**`ml/src/weatherstat/inference.py`**:
- Remove baseline model loading and comparison
- The inference CLI (`just predict`) becomes physics-only or full-model-only

**`ml/src/weatherstat/config.py`**:
- Remove `HORIZONS_HOURLY`, `LGBM_PARAMS_SMALL` (only used by baseline)

**`Justfile`**:
- Remove `train-baseline`, `train-full`, simplify `train` to just train collector models
- Remove experiment commands that reference baseline

**`CLAUDE.md`**: Update to reflect simplified training

### Verification

- `just train` succeeds on collector data
- Inspect trained model's forecast features: values should differ from
  actual outdoor_temp by 2-5°F (not zero as before)
- `just control-physics` still works
- `just simulate` (comparison) still works

---

## Workstream 2: Retrospective HVAC Features

### Problem

Binary "boiler ON/OFF" doesn't capture slab thermal charge. The slab is
a thermal battery — 3 hours of heating stores vastly more energy than 5
minutes. The ML model can't distinguish these states from a single ON/OFF
bit, and the physics simulator uses sysid gains that assume constant-rate
heating. Retrospective features give both engines better initial conditions.

### Features to add

For thermostats and navien (high thermal lag):

| Feature | Definition | Window |
|---------|-----------|--------|
| `{eff}_duty_30m` | fraction of last 30 min active | 6 steps |
| `{eff}_duty_1h` | fraction of last 1h active | 12 steps |
| `{eff}_duty_2h` | fraction of last 2h active | 24 steps |
| `{eff}_cumulative_2h` | total active minutes in last 2h | sum × 5 |
| `{eff}_minutes_since_on` | minutes since last on→off | lookback |
| `{eff}_minutes_since_off` | minutes since last off→on | lookback |

For mini-splits and blowers (low lag): duty_30m and duty_1h only.

Cap `minutes_since_*` at 360 (6h) to prevent extreme values.

### Files to modify

**`ml/src/weatherstat/features.py`**: Add `add_retrospective_hvac_features(df)`.
- Uses the encoded HVAC columns already created by `add_hvac_features()`
  (e.g., `thermostat_upstairs_action_enc`, `navien_heating_mode_enc`)
- Rolling mean for duty cycles: `df[col].rolling(window, min_periods=1).mean()`
- Cumulative: rolling sum × 5 (minutes per step)
- Minutes since transition: diff of encoded column, find last change, compute elapsed
- Call from `build_features()` after `add_hvac_features()`

**`ml/src/weatherstat/simulator.py`**: Optionally use duty cycle as initial
condition modifier for pre-existing thermal charge. This is a future
enhancement — for now, the simulator already handles recent history through
the activity timeline.

### Verification

- Feature values are in expected ranges (duty 0-1, cumulative 0-120 min, etc.)
- No NaN at start of dataset (min_periods=1 handles this)
- Retrain; check feature importance — expect duty_2h and cumulative_2h
  to rank high for thermostat/navien predictions

---

## Workstream 3: Pre-Heating Logic

### Problem

The control loop is reactive: it heats when a room is already cold. With
2-4 hour hydronic lag, this means rooms can be uncomfortably cold for hours
during overnight temperature drops. The physics simulator + forecast can
now predict this in advance and start heating proactively.

### Design

Add `check_preheat()` to `simulator.py`. Called before the sweep in physics
mode:

```python
def check_preheat(
    current_temps: dict[str, float],
    outdoor_temp: float,
    forecast_temps: list[float],      # hourly forecast [h+1..h+12]
    window_states: dict[str, bool],
    params: SimParams,
    hour_of_day: float,
    recent_history: dict[str, list[float]],
    schedules: list[ComfortSchedule],
    base_hour: int,
) -> list[PreHeatRecommendation]:
```

Algorithm:
1. Simulate all-off trajectory for 12h using forecast outdoor temps
2. For each room, find the first horizon where predicted temp drops below
   the comfort min at that future hour (check comfort schedule at each step)
3. If breach found within 6h:
   a. Simulate with zone heating ON from now
   b. Find where the heated trajectory first rises above comfort min
   c. `lead_time` = how many hours of pre-heating needed
   d. `start_by` = `breach_time - lead_time`
   e. `urgent` = `start_by <= 0` (must start heating now)
4. Return list of `PreHeatRecommendation` per zone

### Types

**`ml/src/weatherstat/types.py`**: Add:
```python
@dataclass(frozen=True)
class PreHeatRecommendation:
    zone: str                    # "upstairs" or "downstairs"
    breach_room: str             # room triggering the recommendation
    breach_temp: float           # predicted temp at breach time
    comfort_min: float           # comfort minimum at breach time
    breach_horizon_hours: float  # hours until breach (all-off)
    lead_time_hours: float       # hours of heating needed
    start_by_hours: float        # start heating by this many hours from now
    urgent: bool                 # True if start_by <= 0
```

### Integration with control loop

**`ml/src/weatherstat/control.py`**:

In `run_control_cycle()` (physics mode), before the sweep:
```python
preheat_recs = check_preheat(
    current_temps, outdoor, forecast_temp_list, window_states_dict,
    sim_params, fractional_hour, recent_hist, schedules, base_hour,
)
for rec in preheat_recs:
    if rec.urgent:
        print(f"  PRE-HEAT: {rec.zone} must start now "
              f"({rec.breach_room} reaches {rec.breach_temp:.1f}°F "
              f"in {rec.breach_horizon_hours:.1f}h, need {rec.lead_time_hours:.1f}h lead)")
```

In `sweep_scenarios_physics()`: if any `urgent` recommendation exists, add
it as a constraint (force the zone ON), similar to cold-room override.

**`Justfile`**: No new commands needed — pre-heat runs automatically in
physics mode.

### Verification

- Simulate a scenario: outdoor drops from 45→35°F overnight, rooms at 69°F,
  comfort min 68°F. Pre-heat should recommend starting before the breach.
- Verify lead time is physically reasonable (2-4h for hydronic)
- Stable outdoor temp → no pre-heat recommendation
- Rooms well above comfort → no pre-heat recommendation
- Pre-heat constraint forces zone ON in sweep when urgent

---

## Implementation Order

1. **Workstream 1: Stored forecasts + cleanup** — fixes a real bug, simplifies codebase
2. **Workstream 2: Retrospective HVAC features** — improves both ML and physics
3. **Workstream 3: Pre-heating** — the payoff feature, requires physics mode

Each is independently committable. All three can be done in one session.

## Files Summary

| File | Workstream | Change |
|------|-----------|--------|
| `ml/src/weatherstat/features.py` | 1, 2 | Stored forecast in training; retrospective HVAC features |
| `ml/src/weatherstat/train.py` | 1 | Drop baseline mode, simplify to collector-only |
| `ml/src/weatherstat/inference.py` | 1 | Drop baseline model comparison |
| `ml/src/weatherstat/config.py` | 1 | Remove baseline-only constants |
| `ml/src/weatherstat/simulator.py` | 3 | `check_preheat()` function |
| `ml/src/weatherstat/control.py` | 3 | Pre-heat integration, force-ON constraint |
| `ml/src/weatherstat/types.py` | 3 | `PreHeatRecommendation` dataclass |
| `ml/tests/test_features.py` | 1, 2 | Tests for forecast features, retrospective features |
| `ml/tests/test_simulator.py` | 3 | Tests for pre-heat logic |
| `Justfile` | 1 | Remove baseline commands, simplify train |
| `CLAUDE.md` | all | Update development stages |
