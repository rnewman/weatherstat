# Per-Room Temperature Prediction

## Context

The system currently predicts temperatures for 2 zones (upstairs thermostat hall, downstairs thermostat hall). But we have sensors in 7+ rooms and 6 HVAC devices (2 thermostats, 2 mini splits, 2 blowers, more blowers planned). Per-room prediction lets us:
- Understand how each HVAC device affects each room
- Predict comfort in rooms that don't have direct control (kitchen, bathroom)
- Eventually optimize multi-device control per-room

**Phasing:** Prediction first, then control (separate future work).

## Physical Layout (verified from HA)

**Upstairs** (heated by upstairs thermostat → Navien boiler):
- Upstairs hall — thermostat sensor (`sensor.upstairs_thermostat_air_temperature`)
- Bedroom — dedicated sensor + Aranet4; **mini split: bedroom**
- Kitchen — dedicated sensor; part of open-plan space; **mini split: living room** (indirect)
- Piano — dedicated sensor; part of open-plan space; **mini split: living room** (indirect)
- Bathroom — dedicated sensor; window often open, hard to control

**Downstairs** (heated by downstairs thermostat → Navien boiler):
- Downstairs hall — thermostat sensor (`sensor.downstairs_thermostat_air_temperature`)
- Family room — dedicated sensor; **blower: family room**
- Office — dedicated sensor; **blower: office**

**Aggregates** (computed from room sensors, used by HVAC but not prediction targets):
- `sensor.upstairs_temperatures` = mean(bedroom, kitchen, piano, upstairs_thermostat)
- `sensor.downstairs_temperatures` = mean(family_room, office, downstairs_thermostat)
- `sensor.bedroom_aggregate_temperature` = mean(bedroom, aranet4) — bedroom mini split sensor
- `sensor.living_room_aggregate_temperature` = mean(kitchen, piano) — living room mini split sensor

## Prediction Targets: 8 rooms

| Room | Column | Floor | Direct HVAC | Sensor entity |
|------|--------|-------|-------------|---------------|
| upstairs | thermostat_upstairs_temp | Up | Thermostat (existing) | `sensor.upstairs_thermostat_air_temperature` |
| downstairs | thermostat_downstairs_temp | Down | Thermostat (existing) | `sensor.downstairs_thermostat_air_temperature` |
| bedroom | bedroom_temp | Up | Mini split (existing) | `sensor.climate_bedroom_air_temperature` |
| kitchen | kitchen_temp | Up | — (existing) | `sensor.kitchen_air_temperature` |
| family_room | family_room_temp | Down | Blower (existing) | `sensor.climate_family_room_air_temperature` |
| office | office_temp | Down | Blower (existing) | `sensor.climate_office_air_temperature` |
| piano | piano_temp | Up | — (**NEW sensor**) | `sensor.climate_piano_air_temperature` |
| bathroom | bathroom_temp | Up | — (**NEW sensor**) | `sensor.climate_bathroom_air_temperature` |

Models: 8 rooms × 5 horizons = **40 models** per training mode (was 2 × 5 = 10).

## Part 1: Add piano + bathroom sensors to data pipeline

### 1a. TypeScript schema (`ha-client/`)

**`ha-client/src/entities.ts`** — add to TEMP_SENSORS:
```
piano: "sensor.climate_piano_air_temperature",
bathroom: "sensor.climate_bathroom_air_temperature",
```

**`ha-client/src/types.ts`** — add to SnapshotRow:
```
pianoTemp: number;
bathroomTemp: number;
```

**`ha-client/src/collector.ts`** — 4 places:
- SNAPSHOT_COLUMNS: add `"piano_temp"`, `"bathroom_temp"`
- CAMEL_TO_SNAKE: add `pianoTemp: "piano_temp"`, `bathroomTemp: "bathroom_temp"`
- CREATE_TABLE_SQL: add `piano_temp REAL`, `bathroom_temp REAL`
- buildSnapshot(): add `pianoTemp: getSensorNum(TEMP_SENSORS.piano, 0)`, same for bathroom
- **SQLite migration**: After CREATE TABLE IF NOT EXISTS, run `ALTER TABLE snapshots ADD COLUMN piano_temp REAL` (and bathroom_temp) wrapped in try/catch (column already exists = no-op). This handles the existing DB gracefully.

### 1b. Python schema (`ml/`)

**`ml/src/weatherstat/types.py`** — add to SnapshotRow dataclass:
```
piano_temp: float
bathroom_temp: float
```

**`ml/src/weatherstat/extract.py`**:
- STATISTICS_ENTITIES: add `"piano_temp": "sensor.climate_piano_air_temperature"`, `"bathroom_temp": "sensor.climate_bathroom_air_temperature"`
- numeric_cols list in `extract_history()`: add `"piano_temp"`, `"bathroom_temp"`

### 1c. Feature columns (`ml/src/weatherstat/features.py`)

- TEMP_COLUMNS_FULL: add `"piano_temp"`, `"bathroom_temp"`
- TEMP_COLUMNS_HOURLY: add `"piano_temp"`, `"bathroom_temp"`

This automatically gives piano and bathroom lag/rolling features via the existing generic code.

## Part 2: Expand prediction to per-room

### 2a. Rename + expand prediction config

**`ml/src/weatherstat/features.py`** — rename and expand:
```python
# Was ZONE_TEMP_COLUMNS with 2 entries
ROOM_TEMP_COLUMNS = {
    "upstairs": "thermostat_upstairs_temp",
    "downstairs": "thermostat_downstairs_temp",
    "bedroom": "bedroom_temp",
    "kitchen": "kitchen_temp",
    "piano": "piano_temp",
    "bathroom": "bathroom_temp",
    "family_room": "family_room_temp",
    "office": "office_temp",
}
```

**`ml/src/weatherstat/config.py`** — rename and expand:
```python
# Was PREDICTION_ZONES = ["upstairs", "downstairs"]
PREDICTION_ROOMS = [
    "upstairs", "downstairs",
    "bedroom", "kitchen", "piano", "bathroom",
    "family_room", "office",
]
```

### 2b. Per-room delta features

**`ml/src/weatherstat/features.py`** — expand `add_delta_features()`:

**Indoor-outdoor deltas** for all 8 rooms (was 2 zones):
```python
room_outdoor_cols = {
    "bedroom": "bedroom_temp",
    "kitchen": "kitchen_temp",
    "piano": "piano_temp",
    "bathroom": "bathroom_temp",
    "family_room": "family_room_temp",
    "office": "office_temp",
    "upstairs": "thermostat_upstairs_temp",
    "downstairs": "thermostat_downstairs_temp",
}
```

**Room-to-zone thermostat target deltas** (new — how far each room is from its zone's setpoint):
```python
ROOM_ZONE_TARGET = {
    "bedroom": "thermostat_upstairs_target",
    "kitchen": "thermostat_upstairs_target",
    "piano": "thermostat_upstairs_target",
    "bathroom": "thermostat_upstairs_target",
    "family_room": "thermostat_downstairs_target",
    "office": "thermostat_downstairs_target",
}
```
Produces e.g. `bedroom_zone_target_delta = thermostat_upstairs_target - bedroom_temp`.

Keep existing mini split target deltas (bedroom_target_delta, living_room_target_delta) — these capture the mini split control gap.

### 2c. Update training pipeline

**`ml/src/weatherstat/train.py`**:
- Import `ROOM_TEMP_COLUMNS` instead of `ZONE_TEMP_COLUMNS`, `PREDICTION_ROOMS` instead of `PREDICTION_ZONES`
- Update `get_target_columns()` call to use new names
- Update `add_future_targets()` call to use `ROOM_TEMP_COLUMNS`

**`ml/src/weatherstat/train.py`** — EXCLUDE_COLUMNS_BASE: no changes needed (target cols are excluded dynamically via `exclude = EXCLUDE_COLUMNS_BASE | set(target_cols)`)

### 2d. Update inference pipeline

**`ml/src/weatherstat/inference.py`**:
- Update imports (PREDICTION_ROOMS, ROOM_TEMP_COLUMNS)
- Load 40 models instead of 10
- Per-room predictions in output JSON

### 2e. Update evaluation

**`ml/src/weatherstat/evaluate.py`**:
- Update imports and zone references to use PREDICTION_ROOMS

## Part 3: Handle missing piano/bathroom data gracefully

Piano and bathroom won't have 5-min collector data until after the schema change. They WILL have hourly statistics going back months (the HA sensors have been reporting).

**Training strategy:**
- **Baseline mode (hourly)**: All 8 rooms immediately available after `just extract`
- **Full mode (5-min)**: Piano/bathroom columns will be NaN in historical_full.parquet and early collector rows. Options:
  - (a) Train piano/bathroom models only once enough 5-min data exists (skip if >50% NaN)
  - (b) Fill NaN piano/bathroom from their nearest known values

**Recommended: option (a)** — skip rooms with insufficient data. In `train_mode()`, after building features, check which rooms have enough non-NaN target values and train only those. Print a warning for skipped rooms. As collector data accumulates, piano/bathroom models will automatically start training.

Implementation: filter `target_cols` to only those with >50% non-NaN before entering the training loop.

## Files Modified (summary)

| File | Changes |
|------|---------|
| `ha-client/src/entities.ts` | +2 TEMP_SENSORS entries |
| `ha-client/src/types.ts` | +2 SnapshotRow fields |
| `ha-client/src/collector.ts` | +2 cols in 4 places + ALTER TABLE migration |
| `ml/src/weatherstat/types.py` | +2 SnapshotRow fields |
| `ml/src/weatherstat/extract.py` | +2 STATISTICS_ENTITIES + 2 numeric_cols |
| `ml/src/weatherstat/features.py` | +2 temp cols, rename ZONE→ROOM, expand deltas |
| `ml/src/weatherstat/config.py` | Rename PREDICTION_ZONES → PREDICTION_ROOMS, expand |
| `ml/src/weatherstat/train.py` | Update imports, use ROOM_TEMP_COLUMNS, skip-if-insufficient |
| `ml/src/weatherstat/inference.py` | Update imports and model loading |
| `ml/src/weatherstat/evaluate.py` | Update imports and zone references |

## Verification

1. `just lint` — clean across both packages
2. `just extract` — fetches piano + bathroom historical data
3. `just train-baseline` — trains 8 rooms × 5 horizons = 40 models (piano/bathroom included from hourly stats)
4. `just train-full` — trains existing 6 rooms; skips piano/bathroom with warning (no 5-min data yet)
5. After collector restart: `just collect-once` — snapshot includes piano_temp, bathroom_temp
6. `just predict` — per-room predictions in output
7. `just evaluate` — per-room evaluation comparison

## Future: Multi-Device Control (not this plan)

Once per-room prediction is validated, the control loop needs to expand:
- **Sweep space**: 2 thermostat targets × 2 mini split targets × 2 blower states (+ more blowers as added)
- **Hierarchical optimization**: Sweep thermostats first → best pair → sweep mini splits → sweep blowers. ~130 combos vs ~10,000 brute force.
- **Per-room comfort scoring**: Already partially implemented (`control.py` has per-room schedules + ROOM_TO_ZONE mapping). Replace zone lookup with direct room prediction.
- **Command JSON**: Expand to include mini split targets/modes and blower on/off decisions.
