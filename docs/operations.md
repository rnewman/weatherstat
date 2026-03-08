# Operations Guide

Running the weatherstat system: data collection, system identification, and physics-based control.

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

## 2. System Identification

Extract thermal parameters from collector data. This fits the physics model
that the controller uses for all predictions.

```bash
just sysid             # fit all parameters, write data/thermal_params.json
just sysid -v          # verbose: show per-sensor details
just sysid --output custom_path.json  # custom output path
```

**What it produces** (`data/thermal_params.json`):
- `tau_sealed` and `tau_ventilated` per sensor — envelope heat loss time constants
- Effector × sensor gain matrix — heating rate (°F/hr) and delay (minutes) for each (device, sensor) pair
- Solar gain profiles — per-sensor, per-hour-of-day gain coefficients

### When to rerun

- After accumulating significantly more collector data (monthly)
- After physical changes (new insulation, window replacement, new HVAC device)
- Seasonally — spring/summer solar profiles differ from winter

Sysid uses all available collector data. More data = tighter parameter estimates.

## 3. Control Loop

The controller runs a physics-based trajectory sweep: for each combination of
thermostat trajectories (delay × duration × on/off), blower modes, and mini-split
modes (~7,400 scenarios), it forward-simulates room temperatures over a 6-hour
horizon and selects the trajectory that minimizes comfort cost + energy cost.

Re-evaluated every 15 minutes (receding horizon — only the immediate action
is executed, then re-planned with fresh data).

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

For continuous operation:

```bash
just control-loop-live # 15-min loop: control cycle + execute via HA
```

The control module writes `data/predictions/command_YYYYMMDD_HHMMSS.json`.
The executor reads the latest `command_*.json` and applies it via HA services.

### Safety rails

- Setpoints clamped to [65, 78]°F
- 30-minute minimum hold time between setpoint changes
- Refuses to execute if data is >15 minutes stale
- Warns (and skips in live mode) if 1h prediction shows >5°F change
- Cold-room override: forces immediate heating when a room is >1°F below comfort min
- Dry-run is always the default

### Comfort profiles

Defined in `weatherstat.yaml` under the `comfort` section. Per-room, per-time-of-day
comfort bands with asymmetric penalty weights (too-cold penalized more than too-hot).

### Decision logging

Every control cycle logs its decision to `data/decision_log.db` (SQLite):
inputs, predictions, action chosen, and trajectory info. Outcomes are backfilled
automatically by comparing predictions to actual temperatures from subsequent
snapshots.

```bash
sqlite3 data/decision_log.db "SELECT timestamp, comfort_cost, energy_cost, trajectory FROM decisions ORDER BY timestamp DESC LIMIT 10;"
```

## Typical Workflow

```bash
# Day 1: Start collecting
just collect-durable &       # background, or in a tmux/screen session

# After a few days of data: fit thermal parameters
just sysid -v

# Start dry-run control loop
just control                 # single cycle to inspect output
just control-loop            # watch decisions over time

# When confident: go live
just control-live            # single cycle to test
just execute                 # apply to HA

# Ongoing: refit parameters as data accumulates
just sysid
```

## File Layout

```
data/
  snapshots/
    snapshots.db               # collector output (SQLite, ongoing)
  predictions/
    command_*.json             # control decisions (executor reads these)
  thermal_params.json          # sysid output (effector gains, tau, solar)
  decision_log.db              # control decision history + outcome tracking
  control_state.json           # last decision state (anti-cycling)
  executor_state.json          # executor override tracking
logs/
  collector.log                # collector + health check output
```
