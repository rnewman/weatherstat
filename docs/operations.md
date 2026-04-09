# Operations guide

Running the weatherstat system: data collection, system identification, and physics-based control.

## Prerequisites

```bash
just install          # install dependencies
just init             # create ~/.weatherstat, copy example config
# then edit ~/.weatherstat/weatherstat.yaml with your entity IDs
# and create ~/.weatherstat/.env with HA_URL and HA_TOKEN
```

All runtime data lives in `~/.weatherstat/` by default. Override with
`WEATHERSTAT_DATA_DIR` env var.

## 1. Data collection

The collector writes 5-minute snapshots to `~/.weatherstat/snapshots/snapshots.db` (SQLite).
Every day of missed data is unrecoverable — start this first.

### Start collecting

```bash
just collect           # 5-min loop with auto-recovery (Ctrl+C to stop)
just collect-once      # single snapshot (for diagnostics)
```

The collector fetches all entity states via REST API, so each snapshot is
independent. If a fetch fails, it logs the error and retries next cycle.

### Health check

```bash
just health            # is the collector writing fresh data?
```

Exits 0 if the latest snapshot is <10 minutes old, 1 if stale.
The durable collector runs this automatically and shows macOS notifications
when data goes stale.

### Inspecting the database

```bash
sqlite3 ~/.weatherstat/snapshots/snapshots.db "SELECT COUNT(DISTINCT timestamp), MIN(timestamp), MAX(timestamp) FROM readings;"
```

## 2. System identification

Extract thermal parameters from collector data. This fits the physics model
that the controller uses for all predictions.

```bash
just sysid             # fit all parameters, write ~/.weatherstat/thermal_params.json
just sysid -v          # verbose: show per-sensor details
just sysid --output custom_path.json  # custom output path
```

**What it produces** (`~/.weatherstat/thermal_params.json`):
- `TauModel` per sensor — `tau_base` (sealed envelope time constant) plus per-environment-entry `environment_tau_betas` (additional cooling/warming rate when active) and pairwise `environment_interaction_betas`
- Effector × sensor gain matrix — heating rate (°F/hr) and delay (minutes) for each (device, sensor) pair
- Solar elevation gains — per-sensor `β_solar` coefficient (°F/hr per unit sin(elevation)×fraction), automatically seasonal

### When to rerun

- **Automatically:** The TUI runs sysid periodically (default: hourly, configurable via `sysid_interval` in the `defaults` section). A quality gate rejects bad fits (zero taus or zero significant gains), so the existing params are preserved if the new fit is worse.
- **Manually:** After physical changes (new insulation, window replacement, new HVAC device) or seasonally — spring/summer solar profiles differ from winter.

Sysid uses all available collector data. More data = tighter parameter estimates. The two-phase API (`fit_sysid()` + `save_sysid_result()`) allows callers to validate before writing.

## 3. Control loop

The controller runs a physics-based trajectory sweep: for each combination of
effector options (trajectory effectors get delay × duration grids, regulating
effectors get mode + target combinations, binary effectors get their supported
modes), it forward-simulates sensor temperatures over a 6-hour horizon and
selects the scenario that minimizes comfort cost + energy cost (~5,000–15,000
scenarios).

Re-evaluated every 5 minutes by default (configurable via `control_interval` in
the `defaults` section). Receding horizon — only the immediate action is
executed, then re-planned with fresh data.

### Dry-run (default — no changes to HA)

```bash
just control           # single cycle, prints decision + writes command JSON
just control --loop    # 5-min loop, writes command JSON but doesn't execute
```

### Live execution

```bash
just control --live    # single cycle, writes command JSON AND saves state
just execute           # apply the latest command_*.json to HA
just execute --force   # apply ignoring manual overrides
```

For continuous operation:

```bash
just control --live --loop  # 5-min loop: control cycle + execute via HA
just tui                   # interactive dashboard: monitor, control, execute (recommended)
just tui --live            # TUI starting in live mode
```

The control module writes `~/.weatherstat/predictions/command_YYYYMMDD_HHMMSS.json`.
The executor reads the latest `command_*.json` and applies it via HA REST API,
with lazy execution (skips devices already in the desired state) and override
detection (respects manual changes for 30 minutes).

### Safety rails

- Setpoints clamped to [62, 78]°F (absolute bounds, configurable)
- 3-minute minimum hold time between thermostat setpoint changes
- 2-hour minimum hold time between mini-split mode changes
- Per-device mode hold window (e.g., 10pm–7am): no mini-split mode changes during quiet hours, only target temperature adjustments
- Refuses to execute if data is >15 minutes stale
- Cold-sensor override: forces immediate trajectory-effector activation when any sensor is significantly below comfort minimum
- Dry-run is always the default

### Comfort profiles

Defined in `weatherstat.yaml` under `constraints.schedules`. Per-sensor, per-time-of-day
comfort profiles with `preferred` temperature, hard `min`/`max` rails, and asymmetric
`cold_penalty`/`hot_penalty` weights. The optimizer uses a two-layer cost: continuous
quadratic from preferred + 10× steep penalty outside min/max.

### Decision logging

Every control cycle logs its decision to `~/.weatherstat/decision_log.db` (SQLite):
inputs, predictions, action chosen, trajectory info, active comfort profile,
MRT correction offsets, blocked/ineligible effectors, and full resolved comfort
bounds (min, max, preferred band, penalty weights). Outcomes are backfilled
automatically by comparing predictions to actual temperatures from subsequent
snapshots, using the enriched bounds for accurate retroactive cost computation.

```bash
sqlite3 ~/.weatherstat/decision_log.db "SELECT timestamp, comfort_cost, energy_cost, trajectory FROM decisions ORDER BY timestamp DESC LIMIT 10;"
```

## 4. Comfort Dashboard

Visualize how well the system is maintaining comfort. Answers "is it working?"
at a glance.

```bash
just comfort                     # last 7 days, save PNG to ~/.weatherstat/comfort_7d.png
just comfort --days 3            # last 3 days
just comfort --predictions       # include prediction accuracy histogram
just comfort --show              # interactive matplotlib window
just comfort -o report.png       # custom output path
```

**Summary bar:** Per-sensor stacked bar showing % in comfort band, % too cold
(capacity exceeded vs control opportunity), % too hot (same breakdown).

**Historical comfort bands:** Uses logged comfort_bounds from the decision log,
so bands reflect the actual comfort profile, MRT correction, and environment
adjustments active at each point in time (not just the current config).

**Capacity analysis:** Violations are classified by whether the sensor's
dedicated effectors (zone thermostat + name-matched mini split/blower) were
already at maximum. "Capacity exceeded" = building physics limitation.
"Control opportunity" = the system had headroom it didn't use.

**Control authority:** Per-sensor tracking of when the system had full control
vs blocked/overridden/offline. Background tinting on time-series panels and a
"Ctrl %" column in the console summary.

**Console output:** Summary table with per-sensor breakdown: in band %,
control authority %, capacity/control violation split, and dedicated effector
list.

## Typical Workflow

```bash
# First time: initialize data directory
just init
# Edit ~/.weatherstat/weatherstat.yaml and ~/.weatherstat/.env

# Day 1: Start collecting
just collect &               # background, or in a tmux/screen session

# After a few days of data: fit thermal parameters
just sysid -v

# Start dry-run control loop
just control                 # single cycle to inspect output
just control --loop          # watch decisions over time

# When confident: go live
just control --live          # single cycle to test
just tui --live              # interactive dashboard in live mode (recommended)

# Sysid runs automatically in the TUI (hourly, with quality gate).
# Manually refit after physical changes:
just sysid
```

## Data Directory Layout

All runtime data lives in `~/.weatherstat/` (override with `WEATHERSTAT_DATA_DIR`):

```
~/.weatherstat/
  weatherstat.yaml             # house configuration (from weatherstat.yaml.example)
  .env                         # HA credentials (HA_URL, HA_TOKEN)
  snapshots/
    snapshots.db               # collector output (SQLite EAV, ongoing)
  predictions/
    command_*.json             # control decisions (executor reads these)
  thermal_params.json          # sysid output (TauModel, gains, solar)
  decision_log.db              # control decision history + outcome tracking
  control_state.json           # last decision state (anti-cycling, mode hold times)
  executor_state.json          # executor override tracking
  advisory_state.json          # per-environment-entry advisory cooldown timestamps
  comfort_*.png                # comfort dashboard output
```

Source code in the repo:
```
scripts/
  plot_comfort.py              # comfort performance dashboard
logs/
  collector.log                # collector + health check output
```
