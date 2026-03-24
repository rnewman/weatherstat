# Getting Started with Weatherstat

This guide walks you through setting up weatherstat for your house. You'll need a running Home Assistant instance with temperature sensors and climate entities.

## Prerequisites

- **Home Assistant** with WebSocket API access
- **Long-lived access token** (HA → Profile → Security → Long-Lived Access Tokens)
- **Temperature sensors** in the rooms you want to control (at least one)
- **An outdoor temperature sensor** (or use HA's weather entity)
- **Climate entities** (thermostats, mini-splits) and/or fans you want weatherstat to manage
- **Python 3.12+** and **uv** (for the control pipeline)

Optional but valuable:
- Window/door binary sensors (sysid learns their thermal effect)
- Humidity sensors
- A weather integration (met.no is the default; provides forecasts)

## 1. Clone and install

```bash
git clone https://github.com/your-org/weatherstat.git
cd weatherstat
just install          # installs dependencies
```

## 2. Initialize the data directory

```bash
just init
```

This creates `~/.weatherstat/` with subdirectories for snapshots, predictions, and models. It also copies `weatherstat.yaml.example` as a starting point.

## 3. Set up credentials

Create `~/.weatherstat/.env`:

```bash
HA_URL=https://your-ha-instance.local:8123
HA_TOKEN=your_long_lived_access_token_here
```

All weatherstat components (collector, sysid, control, executor) read from this file.

## 4. Discover your entities

Weatherstat includes a discovery script that connects to your HA instance, finds all relevant entities, and generates a starter config:

```bash
just discover
```

This prints a summary of what it found and writes YAML to stdout. To save to a file:

```bash
just discover -- --output ~/.weatherstat/weatherstat.yaml
```

The script finds:
- **Climate entities** — classified as thermostats (heat-only) or mini-splits (heat+cool). This might well be wrong for your house.
- **Fan entities** — blowers, circulation fans
- **Temperature sensors** — by `device_class` or unit of measurement
- **Humidity sensors** — by `device_class`
- **Window/door sensors** — binary sensors with window/door `device_class`
- **Weather entities** — forecast providers
- **Location** — latitude, longitude, elevation, timezone from `zone.home`

If you prefer to write the config manually, use `weatherstat.yaml.example` as a template.

## 5. Review and customize the config

The generated config needs your review. Before running the collector or control loop, open `~/.weatherstat/weatherstat.yaml` and check:

### Sensors

- **State sensors** (optional): If your boiler or furnace exposes a mode sensor (e.g., "Heating" / "Idle"), add it under `sensors.state` with an encoding. This lets sysid distinguish "thermostat is calling for heat" from "heat is actually being delivered."
- Optional: **remove sensors you don't care about.** You don't need every temperature sensor in your house. However, it's useful to collect data in case you want to use it later. You don't need to set schedules for all of them.

### Outdoor temperature

- Optionally **pick an outdoor sensor.** One temperature sensor can have `role: outdoor`. The physics model needs outdoor temperature for envelope heat loss calculations. If you don't define one, the weather forecast provider will be used instead.

### Effectors

Each HVAC device needs:

- **`control_type`**: How the system searches for the best action.
  - `trajectory` — For slow-response heating (hydronic, radiant). The sweep searches over when to turn on and how long to run.
  - `regulating` — For self-regulating devices (mini-splits, some radiator valves). The sweep searches over target temperatures.
  - `binary` — For discrete-mode devices (fans, dampers). The sweep tries each mode.

- **`mode_control`**: Who controls the device mode.
  - `manual` — You control the mode (e.g., you flip the thermostat to "heat"). The system only adjusts the target temperature.
  - `automatic` — The system controls the mode (e.g., it decides whether a mini-split should heat, cool, or be off).

- **`max_lag_minutes`**: How long between turning on the device and seeing its effect. Hydronic floor heat: 60–90 minutes. Forced air: 15–30 minutes. Mini-split: 10–20 minutes. Fans: 5 minutes. This drives the trajectory search grid.

- **`energy_cost`**: Relative cost per unit of operation. Doesn't need to be in real currency — it's a weight that balances comfort against energy use. Higher = more expensive = the optimizer avoids using it unless necessary.

- **`depends_on`** (optional): Name(s) of parent effector(s) this device depends on. A duct fan that blows air over hydronic coils should declare `depends_on: your_thermostat_name` — it's only useful when the thermostat is actively heating. Multiple parents means ALL must be active (AND gate).

- **`state_device`** (optional): A state sensor that confirms the effector is actually delivering. For thermostats with a boiler, the thermostat might be "calling for heat" but the boiler isn't firing. If you have a boiler mode sensor, reference it here.

### Constraints

Constraints define your comfort targets. Each constraint references a temperature sensor and defines what temperature range you want at each time of day:

```yaml
constraints:
  schedules:
    - sensor: bedroom_temp
      schedule:
        - { hours: [6, 9], preferred: 71, min: 70, max: 73, cold_penalty: 2.0 }
        - { hours: [9, 21], preferred: 70, min: 69, max: 75 }
        - { hours: [21, 6], preferred: 70, min: 68, max: 72, hot_penalty: 0.5 }
```

Start simple: one 24-hour schedule per room with `preferred`, `min`, and `max`. Refine later.

**Not every sensor needs a constraint.** A constraint means "I want the system to actively optimize this sensor." Sensors without constraints are still collected and used by sysid — they just don't drive HVAC decisions.

### Windows (optional)

If you have window/door sensors, list them. You don't need to configure which rooms they affect — sysid learns the thermal coupling from data.

### Defaults

`tau: 45.0` is the envelope time constant used before sysid runs. If your house is well-insulated, 40–60 is reasonable. Poorly insulated: 10–20. This is just a starting guess; sysid replaces it with a fitted value.

## 6. Verify the config

```bash
just verify
```

This runs the config parser to catch errors before you start collecting data.

## 7. Start collecting data

```bash
just collect           # 5-min loop with auto-recovery (Ctrl+C to stop)
```

The collector writes snapshots to `~/.weatherstat/snapshots/snapshots.db` in EAV format. Each snapshot captures the full state of your house: temperatures, HVAC states, window states, weather conditions, and forecasts.

**Let it run for at least 3–5 days** before proceeding. Sysid needs:
- Overnight cooling curves (all HVAC off) to fit τ
- Periods with individual effectors active to fit gains
- Ideally some variety in weather conditions

More data = better model. A week is good; a month is great.

## 8. Run system identification

```bash
just sysid
```

This reads your collector data and fits the thermal model parameters: τ (envelope time constant), per-effector gains (how much each device warms/cools each sensor), effective lags, solar gain profiles, and window coupling coefficients.

Output goes to `~/.weatherstat/thermal_params.json`. Review the console output — it shows fitted parameters, data quality metrics, and any warnings about insufficient data or confounded gains.

### What to look for

- **τ values** should be physically plausible (10–60 hours for a house).
- **Gains** should make sense: the thermostat in a room should have the highest gain for sensors in that room. Cross-room gains should be smaller.
- **Warnings about t-statistics** mean the gain estimate is uncertain — usually not enough data with that specific effector active.
- **Magnitude caps** mean a gain was implausibly large (>3°F/hr) — likely caused by confounded data (the effector was correlated with other warming sources).

Rerun sysid as you accumulate more data. The model improves with more diverse conditions.

## 9. First control cycle (dry run)

```bash
just control
```

This runs a single control cycle in dry-run mode: reads the current state, generates candidate plans, simulates them, picks the best, and prints the decision — but doesn't execute anything.

Review the output:
- Current temperatures and weather conditions
- Number of scenarios evaluated
- Winning plan (which effectors are on, at what settings)
- Per-device rationale (counterfactual attribution)
- Predicted temperatures at each horizon
- Window opportunity recommendations (if any)

The command JSON is written to `~/.weatherstat/predictions/` but not applied.

## 10. Go live

Once you're satisfied with the dry-run decisions:

```bash
just control-live         # single live cycle
just control-loop-live    # continuous 15-min loop (production)
```

This executes the decisions: sets thermostat targets, changes mini-split modes, adjusts fan speeds. The executor checks current HA state before acting (lazy execution — skip if already correct) and detects manual overrides.

## Ongoing operations

- **`just health`** — Check if the collector data is fresh.
- **`just comfort`** — Comfort performance dashboard (last 7 days): time in/out of comfort band per sensor.
- **`just sysid`** — Rerun periodically as data accumulates, especially after seasonal changes.
- **`just verify`** — Run after editing the config to catch parser errors.

## What the system learns vs. what you configure

**You configure:**
- What sensors and effectors exist (entity IDs)
- How effectors behave (control_type, mode_control, lag, energy cost)
- What temperatures you want (comfort schedules)
- Physical dependencies between devices (depends_on)

**The system learns:**
- How fast your house loses heat (τ)
- How much each effector warms/cools each sensor (gains)
- Effective response delays (lag bins)
- Solar gain by room and time of day
- How each window affects each sensor's temperature dynamics
- Cross-breeze interactions between windows

This separation is deliberate: you declare what exists, the system discovers how things interact. When in doubt, add sensors and effectors to the config and let sysid figure out whether they matter.
