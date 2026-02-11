# Operations Guide

Running the weatherstat system: data collection, model training, and control.

## Prerequisites

```bash
just install          # pnpm + uv dependencies
cp .env.example .env  # then fill in HA_URL and HA_TOKEN
```

## 1. Data Collection

The collector writes 5-minute snapshots to `data/snapshots/snapshots.db` (SQLite).
Every day of missed data is unrecoverable — start this first.

### Start collecting

```bash
just collect-durable   # recommended: auto-restart + health monitoring
```

This runs the collector in a restart loop with exponential backoff (5s to 5min)
and a background health check every 5 minutes. Logs go to `logs/collector.log`.
Stop with Ctrl+C for clean shutdown.

For quick testing:

```bash
just collect-once      # single snapshot
just collect           # loop without restart wrapper
```

### Health check

```bash
just health            # is the collector writing fresh data?
```

Exits 0 if the latest snapshot is <10 minutes old, 1 if stale.
The durable collector runs this automatically and shows macOS notifications
when data goes stale.

### Inspecting the database

```bash
sqlite3 data/snapshots/snapshots.db "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM snapshots;"
```

## 2. Historical Extraction

The collector only captures data going forward. For historical data (needed for
initial model training), extract from HA:

```bash
just extract                       # both statistics + history
just extract --mode statistics     # 7 months of hourly temps
just extract --mode history        # 10 days of 5-min full-feature data
```

This writes `data/snapshots/historical_hourly.parquet` and
`data/snapshots/historical_full.parquet`. These are separate from the collector's
SQLite database.

## 3. Model Training

Train both models:

```bash
just train             # baseline (hourly, 5+ months) + full (5-min, ~10 days)
just train-baseline    # just the hourly baseline
just train-full        # just the 5-min full-feature model
```

Models are saved to `data/models/`.

### When to retrain

- **Weekly** as collector data accumulates (more full-feature data improves the model)
- **After schema changes** to feature engineering
- **Seasonally** — all initial data is winter; spring/summer data changes the dynamics

```bash
just retrain           # re-extract historical + retrain both models
```

## 4. Prediction & Counterfactual

```bash
just predict           # fetch live state from HA, predict with both models
just counterfactual    # "what if setpoints were 68/70/72/74/76?"
```

## 5. Control Loop

The controller picks optimal thermostat setpoints by sweeping 81 setpoint pairs
and evaluating comfort + energy cost over a 6-hour horizon.

### Dry-run (default — no changes to HA)

```bash
just control           # single cycle, prints decision + writes command JSON
just control-loop      # 15-min loop, writes command JSON but doesn't execute
```

### Live execution

```bash
just control-live      # single cycle, writes command JSON AND saves state
just execute           # tell HA to apply the latest command_*.json
```

The control module writes `data/predictions/command_YYYYMMDD_HHMMSS.json`.
The executor reads the latest `command_*.json` and applies it via HA services.

### Safety rails

- Setpoints clamped to [65, 78]°F
- 30-minute minimum hold time between setpoint changes
- Refuses to execute if data is >15 minutes stale
- Warns (and skips in live mode) if 1h prediction shows >5°F change
- Dry-run is always the default

### Comfort profiles

Defaults in `ml/src/weatherstat/control.py:default_comfort_schedules()`:

| Room       | Time       | Range     | Notes                          |
|------------|------------|-----------|--------------------------------|
| Upstairs   | All day    | 70-74°F   |                                |
| Downstairs | All day    | 70-74°F   |                                |
| Office     | 8am-6pm   | 70-74°F   | High hot penalty (hard to cool)|
| Office     | 6pm-8am   | 66-76°F   | Relaxed                        |
| Bedroom    | 6-9am      | 70-73°F   | Wake-up comfort                |
| Bedroom    | 9am-9pm    | 68-74°F   | Daytime relaxed                |
| Bedroom    | 9pm-6am    | 64-68°F   | Sleep: high hot penalty        |

## Typical Workflow

```bash
# Day 1: Start collecting + extract historical data
just collect-durable &       # background, or in a tmux/screen session
just extract                 # get historical data for initial training

# Day 1: Train initial models
just train

# Day 1+: Verify predictions look sane
just predict
just counterfactual

# Day 1+: Start dry-run control loop
just control-loop            # watch the logs, verify decisions

# When confident: go live
just control-live            # single cycle to test
just execute                 # apply to HA

# Weekly: retrain with accumulated data
just retrain
```

## File Layout

```
data/
  snapshots/
    snapshots.db               # collector output (SQLite, ongoing)
    historical_hourly.parquet  # extracted hourly stats (7 months)
    historical_full.parquet    # extracted 5-min history (10 days)
  models/
    baseline_*.txt             # hourly LightGBM models
    full_*.txt                 # 5-min LightGBM models
  predictions/
    prediction_*.json          # diagnostic predictions
    counterfactual_*.json      # setpoint sweep results
    command_*.json             # control decisions (executor reads these)
  control_state.json           # last setpoint decision (anti-cycling)
logs/
  collector.log                # collector + health check output
```
