# PLAN-9: Generalized Architecture

## Motivation

The system works but has accumulated home-specific assumptions that prevent
reuse. Three categories of problems:

1. **Room-centric config**: Properties of sensors (tau), constraints on sensor
   values (comfort), and device topology are tangled under a "rooms" concept
   that isn't fundamental. The control system optimizes *sensor values* by
   actuating *effectors*. Rooms are labels, not entities.

2. **Hardcoded relationships**: Window effects are configured (`affects: [bedroom]`),
   not learned. Tau switches between two fixed values (sealed/ventilated).
   Boiler column names embed the brand ("navien"). Safety checks are
   Navien-specific code.

3. **Wide snapshot format**: Every sensor is a SQLite column. Adding a sensor
   requires schema migration. Column names become load-bearing API contracts
   that propagate through Python properties, TS extractors, and sysid.

The fix is to align the implementation with the architecture doc's own
principles: *configuration-driven, physics first, observable*.

---

## Core Model

The system has five fundamental concepts:

```
SENSORS         Observable quantities with time-series values.
                Each has an entity_id, a type, and (for temperature
                sensors) a base tau fitted by sysid.

EFFECTORS       Actuatable devices with command/state pairs.
                Each has an entity_id, encoding, energy cost,
                and optional health checks.

COUPLINGS       Physics relationships between effectors and sensors:
                gain (°F/hr per activity unit) and delay (minutes).
                Learned by sysid, not configured.

CONSTRAINTS     Scoring objectives on sensor values.
                Time-of-day bounds with asymmetric penalty weights.
                Reference sensors directly, not rooms.

WINDOWS         Environmental modifiers with binary state.
                Each has an entity_id. Their effect on each sensor's
                thermal dynamics (the "tau graph") is learned by sysid.
```

**Zones** remain as a topological concept — which thermostat controls which
heating circuit, which blowers serve which circuit. But zones are a property
of effectors, not of sensors. The sensor-to-zone mapping is derived from the
coupling matrix (which thermostat has significant gain for which sensors).

**Rooms** become optional display labels, not structural entities.

---

## Workstream 1: Narrow Storage Format

### Problem

The collector writes one wide row per 5-minute snapshot:
```
timestamp | bedroom_temp | kitchen_temp | navien_heating_mode | ...
```

Adding a sensor = ALTER TABLE + code changes. Column names like
`navien_heating_mode` bake brand names into the schema. Different homes
would have completely different schemas.

### Solution

Replace with an EAV (entity-attribute-value) table:

```sql
CREATE TABLE readings (
  timestamp TEXT NOT NULL,
  name      TEXT NOT NULL,
  value     TEXT NOT NULL,
  PRIMARY KEY (timestamp, name)
);
```

Each reading is a (time, name, value) triple. The config maps entity_ids to
names. Adding a sensor = writing new rows. No schema changes, ever.

### Collector changes (TypeScript)

Currently `yaml-config.ts` builds `ColumnDef[]` with extract functions and
SQL types. The collector calls each extract function and inserts one wide row.

New approach:
- `ColumnDef` stays (still need name + extract function + type info).
- INSERT becomes a batch of `(timestamp, name, value)` tuples.
- `createTableSql` generates the EAV schema, not the wide schema.
- Schema migration: the EAV table is created alongside the existing wide
  table. Old data is preserved. The collector writes to both during a
  transition period, then the wide table is dropped.

### Python reader changes

`load_collector_snapshots()` in `extract.py` currently does `SELECT *` and
gets a wide DataFrame. New approach:

```python
def load_collector_snapshots() -> pd.DataFrame:
    rows = conn.execute(
        "SELECT timestamp, name, value FROM readings ORDER BY timestamp"
    ).fetchall()
    df_long = pd.DataFrame(rows, columns=["timestamp", "name", "value"])
    df = df_long.pivot(index="timestamp", columns="name", values="value")
    # Apply types from config
    for col, dtype in _CFG.column_types.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") if dtype == "REAL" else df[col]
    return df.reset_index()
```

### Data migration

One-time script to unpivot existing wide data:
```sql
INSERT INTO readings (timestamp, name, value)
SELECT timestamp, 'bedroom_temp', CAST(bedroom_temp AS TEXT)
FROM snapshots WHERE bedroom_temp IS NOT NULL;
-- ... repeat for each column
```

This is mechanical — generate the INSERT statements from the existing schema.

### What it eliminates

- `snapshot_column_defs()` no longer generates CREATE TABLE SQL with
  per-column types.
- `sensor_entities`, `exclude_columns`, `hvac_merge_columns`,
  `numeric_extract_columns` — all the hardcoded column-name properties
  in `yaml_config.py` — become unnecessary or trivially derived from config.
- No more `ALTER TABLE ADD COLUMN` migrations.
- `allMonitoredEntities` in TS stays (needed for WebSocket subscriptions).

---

## Workstream 2: Sensor-Centric Config

### Problem

Room properties are fragmented across three YAML sections:
```yaml
rooms:
  bedroom:
    temp_column: bedroom_temp
    zone: upstairs
thermal:
  tau_sealed:
    bedroom: 53.4
comfort:
  bedroom:
    - { hours: [6, 9], min: 70, ... }
```

Tau is keyed by room but fitted per sensor. Comfort is keyed by room but
constrains sensor values. Adding a room requires edits in three places.

### Solution

Tau moves to the sensor definition (where it's actually fitted). Comfort
becomes constraints on sensors. Rooms become optional labels.

#### New YAML structure

```yaml
location:
  latitude: 47.66
  longitude: -122.40
  elevation: 30
  timezone: America/Los_Angeles

sensors:
  temperature:
    bedroom_temp:
      entity_id: sensor.climate_bedroom_air_temperature
    kitchen_temp:
      entity_id: sensor.kitchen_air_temperature
    outdoor_temp:
      entity_id: sensor.climate_side_air_temperature
      role: outdoor  # Newton cooling reference
    thermostat_upstairs_temp:
      entity_id: sensor.upstairs_thermostat_air_temperature
    thermostat_downstairs_temp:
      entity_id: sensor.downstairs_thermostat_air_temperature
    # ... all temperature sensors
  humidity:
    upstairs_humidity:
      entity_id: sensor.upstairs_thermostat_humidity
    downstairs_humidity:
      entity_id: sensor.downstairs_thermostat_humidity
    bedroom_humidity:
      entity_id: sensor.climate_bedroom_humidity
    office_humidity:
      entity_id: sensor.climate_office_humidity
    family_room_humidity:
      entity_id: sensor.climate_family_room_humidity
    kitchen_humidity:
      entity_id: sensor.climate_kitchen_humidity  # verify entity name
    piano_humidity:
      entity_id: sensor.climate_piano_humidity
    bathroom_humidity:
      entity_id: sensor.climate_bathroom_humidity
    living_room_humidity:
      entity_id: sensor.climate_living_room_humidity
    basement_humidity:
      entity_id: sensor.climate_basement_humidity
    outdoor_humidity:
      entity_id: sensor.climate_side_humidity
      role: outdoor

effectors:
  thermostats:
    upstairs:
      entity_id: climate.upstairs_thermostat
      zone: upstairs
      state_device: navien
    downstairs:
      entity_id: climate.downstairs_thermostat
      zone: downstairs
      state_device: navien
  mini_splits:
    bedroom:
      entity_id: climate.m5nanoc6_bed_split_bedroom_split
      sweep_modes: ["off", heat, cool]
      command_encoding: { "off": 0, heat: 1, cool: -1, fan_only: 0.5, dry: 0.25, auto: 0.5, heat_cool: 0.5 }
      state_encoding: { heating: 1, cooling: -1, drying: 0.25, idle: 0, "off": 0 }
    living_room:
      entity_id: climate.m5nanoc6_lr_split_living_room_split
      sweep_modes: ["off", heat, cool]
      command_encoding: { "off": 0, heat: 1, cool: -1, fan_only: 0.5, dry: 0.25, auto: 0.5, heat_cool: 0.5 }
      state_encoding: { heating: 1, cooling: -1, drying: 0.25, idle: 0, "off": 0 }
  blowers:
    family_room:
      entity_id: fan.blower_1
      zone: downstairs
      levels: ["off", low, high]
      level_encoding: { "off": 0, low: 1, high: 2 }
    office:
      entity_id: fan.blower_office
      zone: downstairs
      levels: ["off", low, high]
      level_encoding: { "off": 0, low: 1, high: 2 }
  boiler:
    navien:
      mode_entity: sensor.navien_navien_heating_mode
      capacity_entity: sensor.navien_navien_heat_capacity
      mode_encoding: { "Space Heating": 1, "Idle": 0, "Domestic Hot Water": 0, "DHW Recirculating": 0 }
      health:
        - entity: sensor.navien_navien_sh_return_temp
          min_value: 33
          severity: critical
          message: "Boiler return temp too low — may be in Zone Pump mode"
        - entity: sensor.navien_navien_sh_return_temp
          check: unavailable
          severity: warning
          message: "Boiler diagnostic sensor offline"

windows:
  basement:
    entity_id: binary_sensor.window_basement_intrusion
  family_room:
    entity_id: binary_sensor.window_family_room_intrusion
  balcony:
    entity_id: binary_sensor.window_balcony_intrusion
  bedroom:
    entity_id: binary_sensor.window_bedroom_intrusion
  office:
    entity_id: binary_sensor.window_office_window_door_is_open
  kitchen:
    entity_id: binary_sensor.kitchen_window_sensor_intrusion
  piano:
    entity_id: binary_sensor.window_piano_is_open
  bathroom:
    entity_id: binary_sensor.bathroom_high_window_open

weather:
  entity_id: weather.forecast_home

# Constraints reference sensors directly, not rooms.
# The optimizer minimizes penalty-weighted violations of these bounds.
constraints:
  - sensor: thermostat_upstairs_temp
    schedule:
      - { hours: [0, 24], min: 70, max: 75, hot_penalty: 0.5 }
  - sensor: thermostat_downstairs_temp
    schedule:
      - { hours: [0, 24], min: 70, max: 74 }
  - sensor: bedroom_temp
    schedule:
      - { hours: [6, 9], min: 70, max: 73, cold_penalty: 2.0, hot_penalty: 2.0 }
      - { hours: [9, 21], min: 69, max: 75, cold_penalty: 1.0, hot_penalty: 0.5 }
      - { hours: [21, 6], min: 68, max: 72, cold_penalty: 2.0, hot_penalty: 0.75 }
  - sensor: office_temp
    schedule:
      - { hours: [8, 18], min: 70, max: 74, cold_penalty: 2.0, hot_penalty: 2.0 }
      - { hours: [18, 8], min: 67, max: 76, cold_penalty: 1.0, hot_penalty: 0.5 }
  - sensor: family_room_temp
    schedule:
      - { hours: [0, 24], min: 70, max: 74 }
  - sensor: kitchen_temp
    schedule:
      - { hours: [0, 24], min: 69, max: 75, cold_penalty: 1.0, hot_penalty: 0.5 }
  - sensor: piano_temp
    schedule:
      - { hours: [0, 24], min: 69, max: 75, cold_penalty: 1.0, hot_penalty: 0.5 }
  - sensor: bathroom_temp
    schedule:
      - { hours: [0, 24], min: 68, max: 76, cold_penalty: 0.5, hot_penalty: 0.3 }

# Zones are topological: which thermostat controls which heating circuit.
# The sensor-to-zone mapping is derived from sysid's coupling matrix.
zones:
  upstairs:
    thermostat: upstairs
  downstairs:
    thermostat: downstairs

energy_costs:
  gas_zone: 0.010
  mini_split: 0.005
  blower: { "off": 0.0, low: 0.001, high: 0.002 }

notifications:
  target: notify.mobile_app_richard_s_15_pro

advisory:
  effort_cost: 0.5
  quiet_hours: [22, 7]
  cooldowns:
    free_cooling: 14400
    close_windows: 3600

safety:
  cooldowns:
    thermostat_off: 3600
    device_fault: 1800

defaults:
  tau: 45.0  # hours — used before sysid has run
```

#### Key changes from current config

| Current | New | Why |
|---------|-----|-----|
| `rooms.bedroom.temp_column: bedroom_temp` | Dropped | Constraints reference sensors directly |
| `rooms.bedroom.zone: upstairs` | Derived from sysid coupling matrix | Which thermostat heats bedroom? The one with the highest gain |
| `thermal.tau_sealed.bedroom: 53.4` | Dropped | Sysid learns base tau per sensor |
| `thermal.tau_ventilated.bedroom: 23.5` | Dropped | Sysid learns per-window coupling coefficients |
| `comfort.bedroom: [...]` | `constraints: [{sensor: bedroom_temp, ...}]` | Constraints target sensors |
| `windows.bedroom.rooms: [bedroom]` | Dropped | Sysid learns window-sensor couplings |
| `safety.navien: {...}` | `effectors.boiler.navien.health: [...]` | Health checks belong to devices |
| `devices` (top-level) | `effectors` (top-level) | Clearer naming |
| `sensors.humidity` (1 sensor) | `sensors.humidity` (11 sensors) | All available HA humidity sensors |

#### What `rooms` used to provide

| Role | Now provided by |
|------|----------------|
| Map room name → temp column | Constraints reference sensors directly |
| Map room → zone | Sysid coupling matrix (or explicit zone config) |
| Display grouping | Sensor names are descriptive enough |
| Tau lookup key | Sysid output keyed by sensor name |
| Comfort lookup key | Constraints keyed by sensor name |

---

## Workstream 3: Learned Window Effects (Tau Graph)

### Problem

Window effects are currently binary and hardcoded:
- Config: `windows.bedroom.rooms: [bedroom]` (which sensors this window affects)
- Sysid: fits `tau_sealed` and `tau_ventilated` separately, switches between them
- Simulator: `tau = tau_vent if is_ventilated else tau_sealed`

This misses:
- **Cross-room effects**: opening the living room window cools the kitchen and
  piano too, through airflow coupling.
- **Cross-breeze effects**: bedroom + basement windows open together create a
  draft that cools faster than either alone.
- **Per-window magnitude**: the balcony door (large opening) has a bigger
  effect than the bathroom window (small, high).

### Solution: windows as environmental effectors in sysid

The thermal equation becomes:

```
dT/dt = (T_out - T) × (1/tau_base + Σ β_w × open_w + Σ β_{ww'} × open_w × open_w')
      + Σ gain_e × activity_e(t - lag_e)
      + solar(hour)
```

Where:
- `tau_base` = sealed envelope time constant (all windows closed)
- `β_w` = per-window cooling rate coefficient for this sensor
- `β_{ww'}` = cross-breeze interaction coefficient for window pair

The effective tau at any moment is:
```
1/tau_eff = 1/tau_base + Σ β_w × open_w + Σ β_{ww'} × open_w × open_w'
```

#### Sysid changes

**Stage 1 (tau fitting):** Only use all-windows-closed segments. This gives
`tau_base` (pure sealed envelope). Currently we separate sealed and
ventilated segments — instead, discard ventilated segments for tau fitting.

**Stage 2 (regression):** The Newton residual uses `tau_base`:
```python
residual = dT/dt_observed - (T_out - T) / tau_base
```

Add window regressors alongside effector lag features and solar indicators:
```python
# Per-window: window_state × (T_outdoor - T)
for win_col in all_window_columns:
    X_window = df[win_col] * (df["outdoor_temp"] - df[temp_col])
    features.append(X_window)

# Window interactions: pairs with enough co-open data
for i, w1 in enumerate(all_window_columns):
    for w2 in all_window_columns[i+1:]:
        co_open = df[w1] & df[w2]
        if co_open.sum() >= MIN_INTERACTION_ROWS:
            X_interact = co_open * (df["outdoor_temp"] - df[temp_col])
            features.append(X_interact)
```

The regression coefficient for each window feature gives `β_w` directly.
The coefficient for each interaction gives `β_{ww'}`.

**Physical constraint:** `β_w` should be positive (opening a window increases
cooling rate). Negative coefficients indicate insufficient data or
collinearity — flag as negligible.

#### SysIdResult output changes

Currently outputs `SensorSpec.window_columns` and `FittedTau.tau_sealed/tau_ventilated`.

New output per sensor:
```json
{
  "sensor": "bedroom_temp",
  "tau_base": 53.4,
  "n_segments": 8,
  "window_couplings": {
    "bedroom": { "beta": 0.015, "t_statistic": 4.2 },
    "basement": { "beta": 0.002, "t_statistic": 1.1 },
    "balcony": { "beta": 0.001, "t_statistic": 0.3 }
  },
  "window_interactions": {
    "bedroom+basement": { "beta": 0.008, "t_statistic": 2.7 }
  }
}
```

#### Simulator changes

`SimParams.taus` changes from `dict[str, (tau_sealed, tau_vent)]` to
`dict[str, TauModel]`:

```python
@dataclass(frozen=True)
class TauModel:
    tau_base: float                           # sealed envelope (hours)
    window_betas: dict[str, float]            # window_name -> beta
    interaction_betas: dict[str, float]       # "w1+w2" -> beta

    def effective_tau(self, window_states: dict[str, bool]) -> float:
        inv_tau = 1.0 / self.tau_base
        for win, beta in self.window_betas.items():
            if window_states.get(win, False):
                inv_tau += beta
        for key, beta in self.interaction_betas.items():
            w1, w2 = key.split("+")
            if window_states.get(w1, False) and window_states.get(w2, False):
                inv_tau += beta
        return 1.0 / max(inv_tau, 0.01)  # safety floor
```

The Euler integration loop replaces `tau = tau_vent if is_vent else tau_sealed`
with `tau = tau_model.effective_tau(window_states)`.

For the vectorized `predict()` path, pre-compute `tau_eff` per sensor once
before the loop (window states don't change during a prediction horizon).

#### What this eliminates

- `SensorSpec.window_columns` — sysid learns which windows matter
- `SensorSpec.yaml_tau_sealed / yaml_tau_ventilated` — sysid fits from data
- `FittedTau.tau_ventilated` — replaced by window coupling coefficients
- `_estimate_vent_ratio()` — no need for ratio estimation
- `thermal.tau_sealed` and `thermal.tau_ventilated` in YAML — gone
- `windows.*.rooms` in YAML — gone
- `sensor_window_cols` in SimParams — replaced by TauModel.window_betas

#### Bootstrapping (before sysid)

Before sysid runs, `defaults.tau` provides a reasonable starting point. The
simulator uses `defaults.tau` for any sensor without a fitted TauModel, with
no window effects. After first sysid run, learned values take over.

---

## Workstream 4: Generic Safety / Device Health

### Problem

`safety.py` has `check_navien_health()` and `_check_navien_esphome()` — 90
lines of Navien-specific code. `NavienSafetyConfig` is a dedicated
dataclass. The control loop calls `check_navien_health(latest)` by name.

### Solution

Health checks move into the effector definition and the safety system
becomes a generic loop:

```yaml
effectors:
  boiler:
    navien:
      health:
        - entity: sensor.navien_navien_sh_return_temp
          min_value: 33
          severity: critical
          message: "Boiler return temp too low — may be in Zone Pump mode"
```

```python
@dataclass(frozen=True)
class HealthCheck:
    entity_id: str
    min_value: float | None = None    # alert if reading <= this
    max_value: float | None = None    # alert if reading >= this
    severity: str = "warning"
    message: str = ""

def check_device_health(device_name: str, checks: list[HealthCheck]) -> list[SafetyAlert]:
    """Generic health check: fetch entity, compare to thresholds."""
    alerts = []
    for check in checks:
        state = fetch_ha_state(check.entity_id)
        if state == "unavailable" or state == "unknown":
            alerts.append(SafetyAlert(
                key=f"{device_name}_unavailable",
                title=f"{device_name} sensor unavailable",
                message=f"Diagnostic sensor {check.entity_id} is '{state}'",
                severity="warning",
            ))
            continue
        value = float(state)
        if check.min_value is not None and value <= check.min_value:
            alerts.append(SafetyAlert(
                key=f"{device_name}_fault",
                title=f"{device_name} health check failed",
                message=check.message or f"{check.entity_id} = {value}",
                severity=check.severity,
            ))
    return alerts
```

The control loop calls:
```python
for device_name, health_checks in _CFG.device_health_checks.items():
    safety_alerts.extend(check_device_health(device_name, health_checks))
```

### What it eliminates

- `NavienSafetyConfig` dataclass
- `safety_navien` property
- `check_navien_health()` function
- `_check_navien_esphome()` function
- `safety.navien` YAML section
- All "navien" references in safety.py

The thermostat-off check (`check_thermostat_modes()`) stays — it's already
generic (loops over zones, no brand names).

---

## Workstream 5: Boiler Column Generalization

### Problem

Boiler snapshot columns are `navien_heating_mode` and `navien_heat_capacity`.
These names are hardcoded in:
- `yaml_config.py`: `sensor_entities`, `exclude_columns`, `hvac_merge_columns`,
  `numeric_extract_columns`, `snapshot_column_defs()`
- `yaml-config.ts`: boiler section of `loadYamlConfig()`
- `features.py`: HVAC encoding and `HVAC_ACTIVITY_MAP`
- `entities.ts`: `NAVIEN_HEATING_MODE`, `NAVIEN_HEAT_CAPACITY`
- `sysid.py`: `_enumerate_effectors()` (already uses `_CFG.boiler.name`)
- `types.py`: `NavienHeatingMode` enum

### Solution

Column names follow the same pattern as other devices:
```
boiler_{name}_mode       (was: {name}_heating_mode)
boiler_{name}_capacity   (was: {name}_heat_capacity)
```

For the current installation: `boiler_navien_mode`, `boiler_navien_capacity`.

#### Changes

1. `yaml_config.py`: All properties that reference boiler columns derive
   names from `f"boiler_{self.boiler.name}_mode"` etc.

2. `yaml-config.ts`: Boiler section uses `boiler_${name}_mode` pattern.

3. `entities.ts`: Replace hardcoded `config.boiler["navien"]` with
   iteration over boiler config.

4. `features.py`: HVAC encoding derives boiler column names from config.

5. `types.py`: Remove `NavienHeatingMode` enum (it's just string values,
   and the encoding is in YAML).

6. SQLite migration: Rename existing columns.
   ```sql
   ALTER TABLE snapshots RENAME COLUMN navien_heating_mode TO boiler_navien_mode;
   ALTER TABLE snapshots RENAME COLUMN navien_heat_capacity TO boiler_navien_capacity;
   ```
   If using the new EAV format (Workstream 1), instead:
   ```sql
   UPDATE readings SET name = 'boiler_navien_mode' WHERE name = 'navien_heating_mode';
   UPDATE readings SET name = 'boiler_navien_capacity' WHERE name = 'navien_heat_capacity';
   ```

---

## Workstream 6: Humidity Sensor Expansion

### Problem

HA reports 10+ humidity sensors matching rooms we track. We collect only one.

### Solution

Add all per-room humidity sensors to `sensors.humidity` in the new YAML.
With the EAV storage format, no schema change is needed — the collector
just starts writing new `(timestamp, name, value)` rows.

Sensors to add (see Workstream 2 YAML for full list):
- downstairs_humidity, bedroom_humidity, office_humidity,
  family_room_humidity, piano_humidity, bathroom_humidity,
  living_room_humidity, basement_humidity, outdoor_humidity

Humidity data isn't used by the physics simulator yet, but collecting it
enables future features (mold risk, humidifier control, comfort modeling
that accounts for humidity, apparent temperature).

---

## Implementation Order

### Phase 1: Config restructure (low risk, no data format change)

1. Write new YAML schema (Workstream 2 + 4 + 5 + 6)
2. Update Python `yaml_config.py` to parse the new schema
3. Update TypeScript `yaml-config.ts` to parse the new schema
4. Update `safety.py` to use generic health checks
5. Remove `NavienHeatingMode` enum, `NavienSafetyConfig`, room-centric properties
6. Rename boiler columns in SQLite
7. Add humidity sensors

All code continues reading wide-format SQLite. The config changes are
internal — sysid, simulator, control loop get the same data types.

### Phase 2: Narrow storage (medium risk, new data path)

1. Create EAV table alongside existing wide table
2. Collector writes to both (dual-write transition)
3. Migrate existing wide data to EAV
4. Update `load_collector_snapshots()` to read from EAV + pivot
5. Verify sysid/control produce identical results
6. Drop wide table, remove dual-write

### Phase 3: Learned window effects (medium risk, sysid/simulator changes)

1. Update sysid Stage 1: only use all-sealed segments for tau_base
2. Update sysid Stage 2: add window × ΔT regressors and interactions
3. New `TauModel` output replacing `FittedTau.tau_sealed/tau_ventilated`
4. Update simulator to use `TauModel.effective_tau()`
5. Verify predictions are reasonable — compare to current fixed-tau results
6. Drop `thermal` config section, `SensorSpec.window_columns`,
   `_estimate_vent_ratio()`

### Phase 4: Derived zone mapping (low risk, controller refinement)

1. After sysid runs, derive sensor → zone mapping from coupling matrix:
   "bedroom_temp is most coupled to thermostat_upstairs → upstairs zone"
2. Controller uses derived mapping instead of configured `rooms.*.zone`
3. Cold-room override uses derived mapping to find which thermostat to force
4. Drop `rooms` from config entirely

---

## Files Summary

| File | Phase | Change |
|------|-------|--------|
| `weatherstat.yaml` | 1 | New schema: sensors/effectors/constraints/windows |
| `ml/src/weatherstat/yaml_config.py` | 1 | Parse new schema, drop room/thermal/comfort sections |
| `ha-client/src/yaml-config.ts` | 1, 2 | Parse new schema, EAV writer |
| `ha-client/src/entities.ts` | 1 | Remove hardcoded navien references |
| `ml/src/weatherstat/safety.py` | 1 | Generic device health checks |
| `ml/src/weatherstat/types.py` | 1 | Remove NavienHeatingMode enum |
| `ml/src/weatherstat/features.py` | 1 | Derive boiler column names from config |
| `ml/src/weatherstat/extract.py` | 1, 2 | Boiler column rename, EAV reader |
| `ml/src/weatherstat/sysid.py` | 1, 3 | Boiler naming, window tau graph |
| `ml/src/weatherstat/simulator.py` | 3 | TauModel, effective_tau() |
| `ml/src/weatherstat/control.py` | 1, 4 | Generic safety calls, derived zone mapping |
| `ml/src/weatherstat/config.py` | 1 | Remove NavienHeatingMode references |
| `ha-client/src/collector.ts` | 2 | EAV writer |
| `docs/ARCHITECTURE.md` | all | Update to reflect new model |
| `CLAUDE.md` | all | Update development stages |

---

## Risk Assessment

**Phase 1** is low risk: config restructure with no data format change. The
system reads the same data, produces the same results. Tests verify.

**Phase 2** is medium risk: dual-write ensures no data loss. The pivot at
read time produces the same DataFrame the wide table did. Can be verified
by comparing outputs before/after.

**Phase 3** is the most interesting: learned window effects could produce
different (hopefully better) predictions. Needs careful comparison:
- Run sysid with current approach, save predictions
- Run sysid with new approach, compare predictions
- The new approach should show stronger cooling when windows are open in
  rooms with large windows, and cross-breeze effects when multiple windows
  are open simultaneously.

**Phase 4** is low risk once sysid produces reliable coupling matrices. The
derived zone mapping should match the currently configured one (because the
physics are right). If it doesn't, that's a signal that the config was wrong.
