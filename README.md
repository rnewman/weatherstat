# Weatherstat

<img width="1798" height="1229" alt="Screenshot 2026-03-24 at 9 41 10 PM" src="https://github.com/user-attachments/assets/f48ce427-c7ae-4ba3-a89f-c9c86cfbbd9d" />

## Docs

* [Onboarding](docs/onboarding.md)
* [Operations](docs/operations.md)
* [Architecture](docs/ARCHITECTURE.md)

## What is weatherstat?

Weatherstat is a physics-based smart climate controller for houses where conventional thermostats don't work well — particularly those with high thermal mass (hydronic floor heating, radiant panels, masonry walls) where the lag between turning on the heat and feeling the warmth is measured in hours, not minutes.

Weatherstat provides fine-grained commands and a comprehensive terminal UI.

I built this because my home's conventional climate system — two thermostats driving hydronic heat (under-floor on one level, in-wall on another), two mini-splits primarily for cooling, and six in-wall blowers with manual two-speed switches, not to mention seemingly countless windows and shades — left my rooms a patchwork of too cold, too hot, and sometimes both in the same day.

Weatherstat started out as an ML experiment: could a decision tree model do a good job of controlling these components?

The answer was no, but I ended up with an [architecture](docs/ARCHITECTURE.md) that could.

Examples in this README are all in °F, my local unit; the system itself is unit-agnostic, just like the math. Constants and thresholds are configurable.

## The problems

A conventional thermostat is a reactive control system: it measures the temperature, compares it to a setpoint, and turns the heat on or off. This fails badly in at least four ways.

Firstly, it delivers a poor experience for high-lag systems. Forced-air HVAC systems can respond quite quickly, but hydronic floor heating cannot. You set the thermostat to 71°F. The room drops to 70°F. The thermostat calls for heat. Hot water starts flowing through the floor — but the thermal mass of the floor means the room won't warm up for 45–90 minutes. By the time the room reaches 71°F and the thermostat tells the heater to stop heating and pumping water, the floor and system have stored enough heat that, over the next hour, the air temperature overshoots to 73°F. By the time the room is cold again, so is the hydronic system, so the cycle repeats with the same overshoot. The result: wide temperature swings, wasted energy, and the thermostat is always fighting the last war.

Secondly, it ignores insolation. You get up and it's cold, so you set the thermostat to 72° — maybe you have it on a schedule. The house begins warming up, finally getting up to temperature around 11am. The hydronic heat system continues to dump heat into the air for a couple more hours… but at this point the sun is fully up and is baking the south-facing windows. By 4 or 5pm, the living room is a toasty 77°. The thermostat will stay off until the living room cools, guaranteeing a cold-start problem again the following morning. If your thermostat had only known that today would be a sunny day, perhaps it would have only heated the basement, keeping the living room at the lowest acceptable temperature with the expectation that the sun would finish the job.

Thirdly, it can't account for open windows. At that toasty 77°, the living room could be comfortable with half an hour of ventilation on a cool day — but you won't know until you go upstairs to make dinner. Wouldn't it be nice if your thermostat could tell you when opening or closing a window would make a big (and cheap!) difference?

Fourthly, the typical thermostat has one or two sensors. Unless you have well-tuned central air with well-balanced registers that account for insolation, multiple rooms in a shared zone will end up at different temperatures — a bedroom with an open window might be 10°F cooler on one side of the room than the other. I want to be able to monitor many more than two points in my home, and automatically adjust all of my climate devices to make as many places comfortable as possible.

The fix for the first two of these is to predict the future and act at the right time. If you know the room will be too cold in an hour, start heating now. If you know the slab will dump another degree or two into the air and you're close to target, stop heating. If you know it's going to be sunny today, don't add heat that the sun will provide for free.

The fix for the second two is to be able to predict and optimize across **many** sensors and effectors: to explore the space of possible configurations of heating, cooling, windows, *etc.*, to maximize comfort at minimal cost.

This prediction and simulation requires a model of how the house actually behaves — how fast it loses heat, how much heat each device delivers, how solar gain and weather affect each room. That's what weatherstat builds.

### Sensors and effectors, not rooms

The system doesn't think in terms of rooms. It optimizes **sensor** values by actuating **effectors**. A thermostat is an effector; a thermometer is a sensor.

The coupling between sensors and effectors (how much does this heater warm that thermometer?) is discovered from data, not configured. The same approach can also apply to ventilation and humidity.

Thermal coupling doesn't respect room boundaries. The living room thermostat warms the kitchen (weakly) through the wall; the downstairs heating warms the bedroom through the floor. The mini-split in the bedroom cools the hallway through the open door. The system captures these cross-couplings naturally.

## The approach: physics with learned parameters

Weatherstat uses a grey-box model: the structure comes from physics, the parameters come from data.

The physics is Newton's law of cooling with additional forcing terms:

```
dT/dt = (T_outdoor - T) / tau           # envelope heat loss/gain
      + Σ gain × activity(t - lag)      # heating/cooling from each effector
      + solar(hour) × solar_fraction    # solar gain by time of day
```

**Tau** (τ) is the thermal time constant of the building envelope — how many hours it takes to lose 63% of the indoor-outdoor temperature difference if you turned off all heating. A well-insulated house might have τ = 40–60 hours. A drafty one might be 10–20. Opening a window dramatically reduces τ.

**Gains** describe how much each HVAC device (effector) warms or cools each temperature sensor. The upstairs thermostat might warm the upstairs by 0.8°F/hr but the kitchen by only 0.15°F/hr. A mini-split cools the bedroom at 1.2°F/hr. These are the coupling coefficients between effectors and sensors.

**Lag** is the delay between turning on an effector and seeing its effect. Forced air: 5–15 minutes. Mini-split: 10–20 minutes. Hydronic floor heat: 45–90 minutes. This is the fundamental parameter that makes high-lag systems hard to control reactively.

**Solar gain** varies by room and hour of day — a south-facing room gets significant warming from 10am to 3pm on a sunny day; a north-facing room gets almost none.

The system doesn't assume or configure these parameters. It *learns* them from your data through **system identification** (sysid): fit τ from (typically overnight) cooling curves when the heating is off, then fit gains, lags, and solar profiles from the residuals using regression. The more data you collect, the better the model gets.

## What the system does

Every 15 minutes (configurable), weatherstat runs a control cycle:

### 1. Read the current state

Pull the latest sensor data from Home Assistant: sensor temperatures, outdoor temperature, weather forecast, window states, climate entity states. This is collected continuously by a snapshot collector that writes to a local SQLite database every 5 minutes.

### 2. Generate candidate plans

Enumerate thousands of possible HVAC actions. For a trajectory effector like a thermostat: "turn on now and heat for 3 hours", "wait 1 hour then heat for 2 hours", "stay off." For a regulating effector like a mini-split: "heat to 72°F", "cool to 70°F", "turn off." For a binary effector like a blower motor: "off", "low", "high." The sweep takes the cartesian product of all effector options — typically 5,000–15,000 scenarios.

### 3. Simulate each plan

For every candidate, forward-simulate all sensor temperatures for the next 6 hours using the physics model. This uses the fitted parameters (τ, gains, lags, solar profiles), the weather forecast for outdoor temperature, and the current window states. The simulator runs in batch: numpy broadcasts across all scenarios simultaneously, so evaluating 7,000+ plans takes about 30 milliseconds.

### 4. Score against comfort and energy

Each simulation produces predicted temperatures at 1, 2, 4, 6, and 12 hours for every sensor. These are scored against comfort schedules — what temperature do you want in each room at each time of day?

The scoring model has three layers:
- **Dead band:** Zero cost within your preferred range. `preferred` can be a single temperature (point target) or a range like `[70, 73]` — anywhere inside the range is equally good, eliminating wasteful hunting.
- **Comfort band:** Quadratic penalty for deviation outside the preferred range but within `min`/`max`, with asymmetric weights (you might care more about being too cold than too warm).
- **Hard rails:** Steep additional penalty (10×) for exceeding minimum/maximum bounds.

Plus an energy cost term for each active effector. The total score is comfort cost + energy cost.

### 5. Pick the best plan and execute

Select the plan with the lowest total cost. But only execute the *immediate* action — this is the **receding horizon** principle. A plan that says "wait 2 hours then heat" means "stay off right now." In 15 minutes, the controller re-evaluates with fresh data and may choose differently. The future part of the plan is never committed; it's just context for choosing the right action now.

### 6. Explain why

After selecting the best plan, run counterfactual simulations: what would happen if each active device were individually turned off? This gives true per-device attribution — "the upstairs thermostat is preventing the bedroom from dropping below 69°F by 4pm" — rather than vague "the heat is on."

## Key concepts

### Effector types

Effectors differ in their response characteristics, not in fundamental kind:

- **Trajectory** (e.g., hydronic thermostats): Slow response, large thermal mass. The control variables are *when* to turn on and *how long* to run. The sweep searches over delay × duration grids. The setpoint is a control lever — set slightly above current temp to ensure the thermostat calls for heat, or slightly below comfort minimum (as a safety net) when off.

- **Regulating** (e.g., mini-splits): Fast response, self-regulating. The control variable is the *target temperature*. Activity ramps proportionally as the room deviates from the target — the mini-split works harder when the room is far from target and tapers off as it approaches. The sweep searches over target temperatures drawn from your comfort schedule.

- **Binary** (e.g., duct fans): Instant response, discrete modes. The control variable is the *mode* (off/low/high). A blower over a hydronic coil circulates warm air to a room, changing the effectiveness of the hydronic zone at heating the air.

### Dependencies

Some effectors only produce useful output when their dependencies are active. A duct fan blowing air over a hydronic coil is useless unless the thermostat is calling for heat AND the boiler is actually firing — cold air circulation doesn't help. The `depends_on` field declares these physical dependencies (AND gate: all dependencies must be active). The scenario generator prunes impossible combinations, and the simulator gates dependent effector activity by dependency state.

### Comfort schedules and profiles

Comfort constraints define what "comfortable" means for each sensor, varying by time of day:

```yaml
- sensor: bedroom_temp
  schedule:
    - { hours: [6, 9], preferred: 71, min: 70, max: 73, cold_penalty: 2.0 }
    - { hours: [9, 21], preferred: [70, 73], min: 69, max: 75 }
    - { hours: [21, 6], preferred: 70, min: 68, max: 72 }
```

`preferred` is the ideal temperature — either a point (`preferred: 71`) or a range (`preferred: [70, 73]`). A point target means the optimizer always gently pushes toward that value. A range defines a dead band — zero cost anywhere inside, so the system won't waste energy chasing a single degree. `min` and `max` are hard boundaries with steep penalties. `cold_penalty` and `hot_penalty` let you express asymmetric preferences ("I'd rather be 1°F too warm than 1°F too cold in the morning").

**Comfort profiles** (Home, Away) apply offsets to the base schedules. `preferred_widen` expands point targets into dead bands — for example, an Away mode with `preferred_widen: 6` turns `preferred: 71` into a [68, 74] dead band where the system won't act, avoiding wasteful cool-then-heat cycles overnight.

**MRT correction** adjusts comfort targets based on outdoor temperature. When it's cold outside, walls and windows radiate less infrared toward you — you need a warmer air temperature to feel the same comfort. The system automatically raises targets in cold weather and lowers them in warm weather, with per-sensor weights (a room with large exterior-facing windows gets more correction than an interior room).

### Window effects and opportunities

Open windows dramatically change a room's thermal behavior. The system handles this at two levels:

**In the physics model:** Sysid learns per-window cooling rate coefficients. An open window reduces the effective τ — the room loses heat faster. Cross-breeze interactions (two windows open simultaneously) have their own learned coefficients. The simulator uses the current window state in every prediction.

**In the advisory system:** After choosing the best electronic HVAC plan, the system evaluates whether toggling each window would improve comfort and/or save energy. If opening a window would let you turn off the mini-split and still stay comfortable, that's an energy-saving opportunity. Persistent notifications track these opportunities across control cycles, with cooldown periods to prevent nagging.

### Safety rails

The system is conservative by design:

- **Setpoint clamps** (62–78°F): Never sets a thermostat outside safe bounds.
- **Hold times**: Minimum 10 minutes between setpoint changes.
- **Mode hold windows**: No mini-split mode changes during configurable quiet hours (e.g., 10pm–7am), so you're not woken by compressor starts — only silent target temperature adjustments.
- **Cold-room override**: If any sensor drops significantly below its comfort minimum, constrain the sweep to scenarios where the most-coupled trajectory effector (from sysid's coupling matrix) is heating immediately — overriding delay and cost trade-offs.
- **Minimum improvement threshold**: Don't turn on HVAC unless it improves the score by at least 1.0 units over doing nothing.
- **Override detection**: If a human manually adjusts a thermostat, respect the override.
- **Health monitoring**: Alert if critical infrastructure (boiler connection, return temperature) is in a bad state.

### Data dependency

The system needs data to work. Specifically:

- **At least a few days of collector data** before sysid can fit reliable parameters. τ needs overnight cooling curves; gains need periods where individual effectors were active.
- **Seasonal data matters.** Parameters fitted from winter data may not be accurate in summer (solar gain profiles change dramatically). The system improves as it accumulates data across seasons.
- **Weather forecasts** from Home Assistant's met.no integration provide the outdoor temperature trajectory for forward simulation.
- **5-minute snapshots** capture the full state of the house: every temperature, every HVAC action, every window state, weather conditions. This is the training data for sysid and the input to control.

The collector runs continuously, writing 5-minute snapshots to a local SQLite database.

## Architecture

```
Home Assistant  ←──── REST API ────→  Collector (Python, 5-min)
       ↑                                    │
       │ REST API                           │ SQLite snapshots
       │                                    ↓
  Executor (Python)                  System Identification
       ↑                              (sysid, Python)
       │ command JSON                       │
       │                                    │ thermal_params.json
       ↓                                    ↓
  Controller (Python) ←── predict() ── Simulator (Python)
       │
       ├── Comfort schedules + energy costs
       ├── Scenario sweep (5,000-15,000 plans)
       ├── Window opportunities (advisory)
       ├── Safety checks (health)
       └── Decision log (outcomes)
```

The system communicates with Home Assistant via REST API for both reading state (collector) and executing commands (executor). Data flows through SQLite (collector → sysid) and JSON files (controller → executor).

Everything is driven by a single YAML config file that declares what sensors, effectors, windows, and comfort constraints exist. Adding a new device is a config edit — the collector picks it up automatically, sysid fits its parameters when re-run, and the control loop incorporates it into the sweep.

## What it doesn't do

- **It's not a replacement for climate entities.** It sets thermostat targets and modes; the climate entities still do their own local control. If weatherstat stops running, your thermostats continue working at whatever setpoints were last applied. It is entirely possible for you to build those yourself with relays and Home Assistant, but I chose to keep a layer of redundancy.
- **It doesn't learn occupancy.** It doesn't know when you're home (unless you tell it via the comfort profile entity). This is intentional — occupancy detection is a different problem. You can use occupancy to drive behavior by flipping the profile.
- **It's not magic.** Weatherstat cannot tell when you have a window open without a sensor, run a space heater, flip the breaker for your electric heat without turning off the thermostat, *etc.* Adding or removing heat from the system without being able to tell makes it much harder to extract correct relationships.
- **It doesn't control non-HVAC devices (yet).** No lights, blinds, or appliances. Only climate entities, fans, and advisory notifications.
- **It needs Home Assistant.** All sensor data and device control goes through HA's REST API.
