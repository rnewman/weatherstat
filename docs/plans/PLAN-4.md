# Plan: System Identification Module (`sysid.py`)

## Context

We're building the foundation of a grey-box thermal model (see `docs/ARCHITECTURE.md`).
The model needs physical parameters: how much does each effector heat/cool each
sensor location? How long is the delay? What is the envelope loss rate? How much
solar gain does each sensor see? System identification extracts these from observed
collector data.

**Key constraint:** HVAC devices cycle on and off — the boiler fires and rests, mini
split compressors cycle, blowers turn on when thermostats call for heat. "Clean"
episodes where exactly one device transitions and nothing else changes are rare.
We need an approach that uses ALL the data, not just rare clean windows.

**Generalizability:** This must be config-driven. A new user lists their sensors and
effectors in YAML, runs sysid, and gets their home's thermal map. No hardcoded
device names or sensor lists.

## Core abstraction: sensors and effectors

The physics operates on **sensors** (temperature measurement points) and
**effectors** (devices that add or remove heat). "Rooms" are a human convenience
label — the hydronic slab doesn't care about walls, and you might have multiple
sensors in the same room measuring slightly different things.

- **Sensor**: a temperature measurement point with a column name (e.g.,
  `bedroom_temp`, `living_room_climate_temp`). Has an envelope loss rate (tau)
  and a solar gain profile. Comfort bounds are defined per sensor.
- **Effector**: an HVAC device with a state column and encoding (e.g.,
  `thermostat_upstairs_action`, active when "heating"). Has a gain and delay
  on each sensor.

The sysid output is: **effector × sensor → {gain, delay}** — the full thermal
coupling matrix. Plus **tau per sensor** and **solar profile per sensor**.

In the existing YAML, sensors come from `sensors.temperature` (each entry has a
column name). Effectors come from `devices` (thermostats, mini splits, blowers,
boiler). The `rooms` section is a grouping/labeling layer on top; sysid operates
below it at the sensor/effector level.

## Approach: two-stage fitting

### Stage 1: Fit tau per sensor (envelope loss)

Tau is derived from data, not assumed. The existing `ml/scripts/fit_tau.py` fits
tau from a single overnight window; sysid generalizes: use ALL nighttime HVAC-off
periods for a more robust estimate.

For each sensor, select all periods where:
- Nighttime (10pm–6am local, minimal solar confound)
- All effectors off (thermostat action = "idle" for both zones, mini splits off)
- Associated window state is constant (fit sealed and ventilated separately)

Fit Newton cooling: `T(t) = T_out + (T_0 - T_out) * exp(-t/tau)` using
`scipy.optimize.curve_fit`. Multiple overnight segments → more data → tighter
confidence intervals than the single-night fit.

Output: `tau_sealed` and `tau_ventilated` per sensor, with number of segments used.

### Stage 2: Regression for effector gains and solar

With tau fitted, compute Newton residuals:

```
dT/dt_newton = (T_outdoor - T_sensor) / tau
residual[t] = dT/dt_observed - dT/dt_newton
```

The residual must be explained by effector activity and solar gain:

```
residual[t] ≈ Σ_e(gain_e * effector_activity_e[t]) + Q_solar(hour[t]) + noise
```

This is a **linear regression**. Every 5-minute snapshot is a data point. We use
all ~6,700 rows, not just clean episodes. Multiple effectors active simultaneously?
The regression decomposes their contributions. Cycling? Captured by time-averaged
activity levels.

For **delay estimation**: include lagged effector states in coarse bins
(0–15min, 15–30min, 30–60min, 60–90min for floor heat; 0–5min, 5–15min for
mini splits). The bin with the highest coefficient reveals the effective delay.

For **solar gain**: include hour-of-day indicator variables (hours 7–17). The
coefficients give the solar profile directly.

One regression per sensor. The coefficients across all sensors form the full
effector × sensor coupling matrix.

## What's config-driven

**From existing YAML — no changes needed:**

Effectors and their state columns (from `devices` section):
- Thermostats: `thermostat_{name}_action`, active when = "heating"
  - `mode_encoding` not directly present, but action column is binary heating/idle
- Mini splits: `mini_split_{name}_mode`, with `mode_encoding` in YAML
- Blowers: `blower_{name}_mode`, with `level_encoding` in YAML
- Boiler: `navien_heating_mode`, with `mode_encoding` in YAML

Sensors and their temp columns (from `sensors.temperature` section):
- Each has `entity_id` and a key that doubles as column name
- Note: not all sensors map 1:1 to `rooms` entries. Some sensors (e.g.,
  `living_room_climate_temp`, `upstairs_aggregate_temp`) exist in `sensors`
  but not in `rooms`. Sysid should process ALL sensors, not just those with
  room entries.

Thermal parameters (from `thermal` section):
- `tau_sealed` / `tau_ventilated` per room name — used for comparison only,
  since sysid fits its own tau values
- Sensors not in the `thermal` section use defaults

Window sensors (from `windows` section):
- Per-window room associations → determines tau_sealed vs tau_ventilated

Sysid reads all of this from config. Adding an effector or sensor = YAML edit + rerun.

## Output data structures

```python
@dataclass(frozen=True)
class EffectorSpec:
    """An effector (HVAC device) derived from config."""
    name: str                    # "thermostat_upstairs", "mini_split_bedroom"
    state_column: str            # "thermostat_upstairs_action"
    encoding: dict[str, float]   # state_value -> numeric activity level
    max_lag_minutes: int         # how far back to test for delayed effects
    device_type: str             # "thermostat", "mini_split", "blower", "boiler"

@dataclass(frozen=True)
class SensorSpec:
    """A temperature sensor derived from config."""
    name: str                    # "bedroom_temp", "living_room_climate_temp"
    temp_column: str             # column name in snapshot DataFrame
    window_columns: list[str]    # which window sensors affect this location
    yaml_tau_sealed: float       # existing YAML value (for comparison)
    yaml_tau_ventilated: float

@dataclass(frozen=True)
class FittedTau:
    """Envelope loss rate fitted from overnight cooling data."""
    sensor: str
    tau_sealed: float            # hours
    tau_ventilated: float | None # hours (None if insufficient ventilated data)
    n_segments_sealed: int
    n_segments_ventilated: int

@dataclass(frozen=True)
class EffectorSensorGain:
    """How one sensor responds to one effector."""
    effector: str
    sensor: str
    gain_f_per_hour: float       # heating rate per unit of effector activity
    best_lag_minutes: float      # lag bin with strongest effect
    t_statistic: float           # statistical significance
    negligible: bool             # True if below noise threshold

@dataclass(frozen=True)
class SolarGainProfile:
    sensor: str
    hour_of_day: int             # 0-23
    gain_f_per_hour: float
    std_error: float
    t_statistic: float

@dataclass(frozen=True)
class SysIdResult:
    timestamp: str
    data_start: str
    data_end: str
    n_snapshots: int
    effectors: list[EffectorSpec]             # inputs used
    sensors: list[SensorSpec]                 # inputs used
    fitted_taus: list[FittedTau]              # stage 1
    effector_sensor_gains: list[EffectorSensorGain]  # stage 2: the coupling matrix
    solar_gains: list[SolarGainProfile]       # stage 2
```

The `effector_sensor_gains` list is the full coupling matrix. Consumers filter
by effector ("what does this thermostat do to each sensor?") or by sensor
("what affects this measurement point?").

## Implementation plan

### 1. Effector enumeration (`_enumerate_effectors`)

Read `devices` from YAML config. For each device type:
- Derive `state_column` from naming convention (`thermostat_{name}_action`,
  `mini_split_{name}_mode`, `blower_{name}_mode`)
- Use `mode_encoding` / `level_encoding` from YAML for numeric conversion
- For thermostats (which have no explicit encoding): `{"heating": 1, "idle": 0}`
- Set `max_lag_minutes` by device type:
  - Thermostat (floor heat): 90 min
  - Mini split: 15 min
  - Blower: 5 min

### 2. Sensor enumeration (`_enumerate_sensors`)

Read `sensors.temperature` from YAML config. For each sensor:
- Column name = the YAML key (e.g., `bedroom_temp`)
- Find associated windows: check `windows.*.rooms` for a matching room name
  (sensors that share a room name with a window entry get that window's column)
- Existing tau from `thermal.tau_sealed` / `tau_ventilated` (by room name if
  available, else defaults)

### 3. Preprocessing (`_preprocess`)

- Load snapshots via `load_collector_snapshots()`
- Parse timestamps: `pd.to_datetime(df["timestamp"], format="ISO8601")`
- Localize to configured timezone (for hour-of-day solar features)
- Sort by timestamp
- Compute `dT/dt` for each sensor: central differences `(T[t+1] - T[t-1]) / 10min`
  (forward/backward at edges)
- Encode effector states to numeric using `EffectorSpec.encoding`
- Generate lagged effector features in coarse bins:
  - Floor heat: 0–15min, 15–30min, 30–60min, 60–90min (mean activity in each bin)
  - Mini splits: 0–5min, 5–15min
  - Blowers: 0–5min (immediate)
- Add hour-of-day indicators for solar (hours 7–17)

### 4. Tau fitting (`_fit_tau`)

Reuse the approach from `ml/scripts/fit_tau.py` (curve_fit on exp(-t/tau)) but
generalized:
- Select all nighttime (10pm–6am local) periods where all effectors are off
- For each sensor: extract contiguous segments, separated by window state changes
- Fit sealed segments (window closed) and ventilated segments (window open) separately
- Minimum segment length: 1 hour (12 steps at 5-min)
- Aggregate multiple segments: weighted median by segment length
- For sensors without ventilated data: estimate using sealed × ratio from a
  sensor that has both (same approach as `fit_tau.py` line 134)

Output: `FittedTau` per sensor. Used for Newton residuals in stage 2.

### 5. Newton residuals

With fitted tau, compute residuals for every row:
```
residual = dT/dt_observed - (outdoor_temp - sensor_temp) / tau
```
Using `tau_sealed` when sensor's window(s) are closed, `tau_ventilated` when open.

### 6. Regression per sensor (`_fit_sensor_model`)

For each sensor, fit:

```
residual[t] = Σ_e Σ_lag(β_{e,lag} * effector_activity_e[t - lag])
            + Σ_h(γ_h * is_hour_h[t])
            + ε[t]
```

Using **ordinary least squares** (`numpy.linalg.lstsq`).

For each effector:
- The lag bin with the largest |β| is `best_lag_minutes`
- The sum of β across all lag bins ≈ total gain
- t-statistics from the covariance matrix for significance

For solar:
- Hour coefficients give the solar profile directly
- Daytime hours only (7–17); nighttime is the implicit baseline

**Collinearity handling:** Coarse lag bins already reduce collinearity. If the
condition number is still high, fall back to ridge regression with small penalty.

### 7. Aggregate and output (`_build_results`)

- For each (effector, sensor): extract best lag, gain, significance
- Flag negligible gains: |gain| < 0.05 °F/hr AND |t-statistic| < 2.0
- Build solar profiles from hour coefficients
- Package into `SysIdResult`
- Serialize to JSON at `data/thermal_params.json`

### 8. Report (`print_report`)

Formatted tables:
- **Tau fits**: per sensor, sealed and ventilated, with comparison to YAML values
- **Effector × sensor gain matrix**: rows = effectors, columns = sensors,
  cells = gain (°F/hr) with delay in parentheses. Negligible gains shown as "–"
- **Solar profiles**: per sensor by hour
- **Data summary**: date range, n_snapshots, effector activity statistics
  (% of time each effector was active)

### 9. CLI entry point

```
python -m weatherstat.sysid [--output PATH] [--verbose]
```

Also: `just sysid` in the Justfile.

## Handling edge cases

**Effectors that never activate:** Report as "no data". Don't crash — a device
might be configured but unused in winter.

**Sensors with NaN periods (new sensors):** Drop those timesteps for that sensor's
regression only. Other sensors still use the full dataset.

**Collinear effectors (both thermostats often on together):** Wider confidence
intervals for individual gains. Report t-statistics so the user can see which
gains are well-identified. Resolves as more varied data accumulates.

**Sensors without window associations:** Always use tau_sealed. This is correct
for interior sensors with no nearby window.

**Sensors not in `rooms` config:** Process them anyway — sysid operates on
`sensors.temperature`, not `rooms`. Sensors like `upstairs_aggregate_temp` or
`living_room_climate_temp` get the same treatment. Their tau comes from defaults
if not in the `thermal` section.

## Files to create/modify

- **Create:** `ml/src/weatherstat/sysid.py`
- **Modify:** `Justfile` (add `just sysid`)

## Dependencies

- `scipy` for `optimize.curve_fit` (tau fitting) — already used in `fit_tau.py`
- `numpy` for `linalg.lstsq` (regression) — already available
- `pandas` — already available

## Verification

1. `just sysid` runs without error on current collector data
2. Output JSON is valid and contains all sections
3. **Fitted tau**: 30–55h range for sealed, consistent with existing YAML values
4. **Floor heat gains**: 0.5–3 °F/hr for sensors near heating loops, 30–90 min delay
5. **Cross-zone effects**: small but detectable gains for sensors on other floors
   (e.g., downstairs thermostat → living_room sensors above)
6. **Mini splits**: higher gain, shorter delay (< 15 min) for nearest sensor
7. **Solar gains**: positive during daytime, near-zero at night
8. **Blower effects**: redistribution visible as differential gains on sensors
   in the same zone
9. **Generalizability**: adding a hypothetical device to YAML and rerunning works
   (missing data column → reported as "no data")
10. **Reproducibility**: running twice on the same data produces identical output
