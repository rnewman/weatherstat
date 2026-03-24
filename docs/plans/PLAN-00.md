# Durable Collection + Control Loop

## Context

Two problems to solve:

1. **The collector has a data loss bug**: `ParquetWriter.openFile()` truncates the file on each write. Every 5-min cycle overwrites the previous snapshot for that day, keeping only the last row. Must fix before starting collection — every day of data is unrecoverable.

2. **The control loop is almost complete**: inference + counterfactual sweep + executor all exist. The missing piece is a control policy that picks the best setpoint and writes executor-compatible JSON.

The system is a receding-horizon controller: "what settings maximize comfort over the next 6 hours?" re-evaluated every 15 minutes. 6h covers hydronic overshoot (~60 min lag), the mid-afternoon solar spike, and the 3am temperature trough.

## Part 1: Fix Collector Storage (Parquet → SQLite)

### Problem
`collector.ts:writeSnapshot()` calls `ParquetWriter.openFile()` which uses `fs.createWriteStream` with default flag `'w'` (truncate). Each 5-min cycle destroys the previous day's data.

### Solution
Replace Parquet with SQLite. SQLite gives us:
- Atomic append (INSERT OR IGNORE for dedup on timestamp)
- No truncation risk
- Easy querying for health checks (`SELECT MAX(timestamp)`)
- Standard format readable from Python (`import sqlite3`)

### Changes

**`ha-client/package.json`** — Replace `@dsnp/parquetjs` with `better-sqlite3` (synchronous SQLite for Node, no native compile needed on modern Node 25). Add `@types/better-sqlite3`.

**`ha-client/src/collector.ts`** — Rewrite `writeSnapshot()`:
- Open/create `data/snapshots/snapshots.db` (single file, not daily)
- CREATE TABLE IF NOT EXISTS with all SnapshotRow columns
- UNIQUE constraint on `timestamp` (rounded to nearest 5 min for dedup)
- INSERT OR IGNORE for each snapshot
- Keep the writer as a module-level db handle (open once, reuse)

```sql
CREATE TABLE IF NOT EXISTS snapshots (
  timestamp TEXT PRIMARY KEY,
  thermostat_upstairs_temp REAL,
  thermostat_upstairs_target REAL,
  thermostat_upstairs_action TEXT,
  ...all other SnapshotRow fields...
);
```

**`ha-client/src/config.ts`** — Add `dbPath` to Config (default: `data/snapshots/snapshots.db`).

**`ml/src/weatherstat/extract.py`** — Add `load_collector_snapshots()` that reads from SQLite:
```python
def load_collector_snapshots(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM snapshots ORDER BY timestamp", conn)
    conn.close()
    return df
```

**`ml/src/weatherstat/inference.py`** — Update `load_latest_snapshots()` to read from SQLite instead of daily Parquet files.

**`.gitignore`** — Add `data/snapshots/*.db`.

### Migration note
The existing `historical_full.parquet` and `historical_hourly.parquet` (from extract.py) stay as Parquet — they're extracted historical data, not collector output. Only the collector's live snapshots move to SQLite.

## Part 2: Collector Shell Wrapper + Health Monitoring

### `scripts/run-collector.sh`
- Sources `.env` for HA_URL/HA_TOKEN
- Runs `npx tsx src/index.ts collect` in a restart loop with exponential backoff (5s → 5min max)
- Spawns a background health check loop (every 5 min)
- Health check: query SQLite for `SELECT MAX(timestamp)`, alert if >10 min stale
- Alert via `osascript -e 'display notification ...'` (macOS)
- Logs to `logs/collector.log`
- Clean shutdown on SIGINT (kills both collector and health monitor)

### `scripts/check-health.sh`
- Standalone one-shot health check
- Queries `data/snapshots/snapshots.db` for latest timestamp
- Exits 0 if fresh, 1 if stale (with notification)

### Justfile additions
```
collect-durable    # Run collector with auto-restart + health monitoring
health             # Check collector health
collect-once       # Single snapshot (already exists in TS, add recipe)
```

### Other
- Add `logs/` to `.gitignore`
- Create `logs/.gitkeep`

## Part 3: Control Policy

### New module: `ml/src/weatherstat/control.py`

Core concept: **per-room virtual thermostats** with weighted comfort optimization.

Remember that this needs to be testable, so it's important to have some kind of virtualized/dry-run output.

#### Comfort Profile

```python
@dataclass(frozen=True)
class RoomComfort:
    room: str                    # "bedroom", "office", "upstairs", "downstairs"
    min_temp: float              # lower comfort bound
    max_temp: float              # upper comfort bound
    cold_penalty: float = 2.0   # penalty weight for being below min
    hot_penalty: float = 1.0    # penalty weight for being above max

@dataclass(frozen=True)
class ComfortSchedule:
    """Time-of-day comfort profile for a room."""
    room: str
    entries: list[tuple[int, int, RoomComfort]]  # (start_hour, end_hour, comfort)
```

Initial defaults from user's preferences:
- **Upstairs**: 70-74°F all day
- **Downstairs**: 70-74°F all day
- **Office**: 70-74°F during day (8am-6pm), high hot_penalty (hard to cool)
- **Bedroom**: 72°F target at wake-up, 66-68°F at night, high hot_penalty always (sleep quality)

The controller currently controls only two thermostats (upstairs/downstairs), but the comfort profile is per-room — the cost function evaluates comfort across all rooms even though it can only steer two setpoints. This means: "setting downstairs to 72 keeps the office comfortable but makes the bedroom too warm" is a tradeoff the cost function can reason about.

**Note**: Currently the model only predicts `upstairs_temp` and `downstairs_temp`, not per-room. For the initial version, upstairs comfort maps to upstairs predictions, downstairs comfort maps to downstairs predictions. Per-room predictions can be added later when we train per-room models.

#### Cost Function

For a candidate setpoint pair (up_sp, dn_sp):
1. Predict temperatures at T+1h through T+6h using `_build_setpoint_overrides()` + model
2. For each zone, at each horizon:
   - Look up the active `RoomComfort` for that time-of-day (current time + horizon)
   - If predicted temp < min: `cost += (min - pred)^2 * cold_penalty * horizon_weight`
   - If predicted temp > max: `cost += (pred - max)^2 * hot_penalty * horizon_weight`
3. Add small energy penalty: `cost += 0.01 * max(0, setpoint - current_temp)` per zone
4. Horizon weights: {1h: 1.0, 2h: 0.9, 4h: 0.7, 6h: 0.5} — nearer matters more, and model is more accurate

#### Setpoint Search

Sweep (up_sp, dn_sp) pairs over [68, 76] in 1°F steps. 9 * 9 = 81 pairs, each needing 10 model predictions (2 zones * 5 horizons). At ~0.1ms per LightGBM prediction, full sweep ~8ms. Negligible.

Sweep must be coupled (not independent per zone) because Navien fires when either zone calls for heat, and upstairs heat is ceiling heat for downstairs.

#### State Persistence

`data/control_state.json` tracks last decision time and setpoints, enforces minimum hold time (30 min default) to avoid cycling.

#### Safety Rails

- Clamp setpoints to absolute bounds [65, 78]
- If any 1h prediction shows >5°F change from current, log warning, skip execution
- If latest data timestamp is >15 min old, refuse to execute (stale data)
- **Dry-run mode is the default** — logs decisions, writes JSON, does NOT call executor

## Part 4: Executor Integration

### Prediction file naming

Control module writes `data/predictions/command_YYYYMMDD_HHMMSS.json` (camelCase keys matching the TS `Prediction` interface). Existing `prediction_*.json` and `counterfactual_*.json` from inference.py are diagnostic.

**`ha-client/src/executor.ts`** — Filter for `command_*.json`:
```typescript
const jsonFiles = files.filter((f) => f.startsWith("command_") && f.endsWith(".json")).sort();
```

### Mini split / blower pass-through

For initial thermostat-only control, mini split and blower values in the command JSON are passed through from the current HA state (not controlled).

### CLI and Justfile

```
control           # Single control cycle (dry-run)
control-loop      # Run control loop (dry-run, 15-min interval)
control-live      # Single control cycle with live execution
execute           # Execute latest command JSON via HA
```

## Files to Create

| File | Purpose |
|------|---------|
| `ml/src/weatherstat/control.py` | Control policy: comfort profiles, cost function, setpoint search, decision loop, CLI |
| `scripts/run-collector.sh` | Shell wrapper: restart loop + health monitor |
| `scripts/check-health.sh` | Standalone health check |

## Files to Modify

| File | Changes |
|------|---------|
| `ha-client/src/collector.ts` | Replace Parquet writer with SQLite |
| `ha-client/src/config.ts` | Add `dbPath` to Config |
| `ha-client/package.json` | Replace `@dsnp/parquetjs` → `better-sqlite3` + types |
| `ha-client/src/executor.ts` | Filter for `command_*.json` |
| `ml/src/weatherstat/inference.py` | Update `load_latest_snapshots()` for SQLite |
| `ml/src/weatherstat/config.py` | Add CONTROL_LOG_DIR, CONTROL_STATE_FILE paths |
| `ml/src/weatherstat/types.py` | Add RoomComfort, ComfortSchedule, ControlDecision, ControlState |
| `Justfile` | Add collect-durable, health, collect-once, control, control-loop, control-live, execute |
| `.gitignore` | Add `data/snapshots/*.db`, `data/control_log/`, `logs/` |

## Implementation Order

1. **SQLite migration** (collector.ts, package.json, config.ts) — must be first, blocks collection
2. **Health scripts** (run-collector.sh, check-health.sh) — enables durable collection
3. **Justfile recipes** for collection
4. **Start collecting data** — critical path, do this as soon as steps 1-3 are done
5. **Control types** (types.py, config.py)
6. **Control policy** (control.py)
7. **Executor integration** (executor.ts filter, command JSON output)
8. **Control Justfile recipes**
9. **Update inference.py** (load from SQLite)

## Verification

1. `just collect-once` → `snapshots.db` has 1 row; run again → dedup works (still 1 row if within same 5-min window)
2. `just collect-durable` → snapshots accumulate; kill → restarts; health alert fires if stale
3. `just control` → prints comfort scores, writes `command_*.json`, logs decision (dry-run)
4. Inspect `command_*.json` → valid Prediction format with camelCase keys
5. `just execute` → TS executor reads command JSON, calls HA services (verify in HA logbook)
6. `just lint` → clean

## Future Phases (not in this implementation)

- Phase 2: Per-room temperature models + blower control
- Phase 3: Mini split target control
- Phase 4: Mini split mode (heat/cool) based on season
- Phase 5: Human advisory ("open windows tonight")
