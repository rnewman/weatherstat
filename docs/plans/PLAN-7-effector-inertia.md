# Plan 7: Trajectory Search for Slow Effectors

# Context

You're in ~/repos/weatherstat. Read docs/ARCHITECTURE.md and docs/FUTURE.md in full to understand the context of what you're doing.

## The problem

We're building a physics-based smart thermostat for a house with hydronic
floor heat (massive thermal lag: 2-4 hours from boiler firing to room warming).

The controller sweeps ~324 HVAC scenarios every 15 minutes, asking: "if I
hold these settings constant for 6 hours, what happens?" This constant-action
assumption systematically mis-evaluates slow effectors (hydronic floor heat:
45-75 min lag, hours of thermal mass).

**Over-costs heating ON**: the sweep prices 6h of continuous gas, but the
real controller will turn it off once rooms are warm.

**Under-values near-term heating**: at the 1h horizon (weight 1.0), a
thermostat turned on now has barely started delivering heat (45 min lag).
The sweep sees "costly action, minimal benefit" and picks all-off.

**Misses proactive starts**: "start heating in 2 hours because a cold front
is arriving" isn't representable — every scenario starts NOW.

**Doesn't value continuations**: if the heater has been on for 40 minutes
and the slab is partially charged but the room hasn't warmed yet, "turn off"
wastes the invested charge. "Continue for 80 more minutes then coast" is
the right move, but it's not in the candidate set.

The result: the controller is reactive. It waits until rooms are cold, then
turns on heating that won't arrive for 45+ minutes. For a system designed
to anticipate rather than react, this is the core gap.

## The insight

The constant-action assumption is the disease. Previous iterations of this
plan proposed a "lookahead constraint" — detect future breaches, force
effectors on as a pre-sweep override. That treats the symptom: it patches
the sweep's wrong answers rather than fixing the evaluation.

The fix: **expand the search space from constant actions to trajectories.**
Each scenario becomes a piecewise-constant action schedule: "effector X runs
for duration D after delay S, then stops." The existing multi-sensor scorer
(comfort cost across all rooms at all horizons) evaluates each trajectory
and picks the best one. No separate lookahead, no constraints, no
zone-specific heuristics.

This is the first step toward full MPC (ARCHITECTURE.md: "extend from
single-step action selection to multi-step trajectory optimization") but
without optimizing arbitrary action sequences — just parameterized
trajectory shapes.

## Trajectory shape

For each slow effector (lag > 15 min), the trajectory has two parameters:

```
[OFF × S steps] → [ON × D steps] → [OFF × remainder]
```

- **delay** S: how long to wait before activating (0 = start now)
- **duration** D: how long to run before shutting off

For fast effectors (mini-splits, blowers: lag < 15 min), constant activity
over the full horizon remains a good approximation — their response time is
under one control loop interval. They keep the current binary/multi-mode
search with no delay/duration parameters.

## Search space

### Candidate values

Delays (slow effectors only):
- 0h, 1h, 2h, 3h, 4h (5 values, in 15-min-step units: 0, 12, 24, 36, 48)

Durations (slow effectors only, when active):
- 1h, 2h, 4h, 6h (4 values, in steps: 12, 24, 48, 72)

Plus OFF (no heating). Some (delay, duration) pairs are redundant when
delay + duration > horizon, and can be pruned.

### Size estimate

Per slow effector: 5 delays × 4 durations + 1 OFF = 21 options.
Two thermostats: 21 × 21 = 441 combinations.
With blowers (3 modes each, 2 blowers) × mini-splits (3 modes each, 2 splits):
441 × 9 × 9 ≈ 35,700 scenarios.

With constraint pruning (blowers off when zone not heating at that timestep,
heating blocked when zone above comfort max, thermal direction check), this
drops substantially. And we can prune delay+duration combos that extend
past the 6h scoring horizon, reducing each effector to ~15 options.

The physics simulator runs ~72 Euler steps × 13 sensors per scenario. At
the current ~30ms for 324 scenarios, 10,000 scenarios would take ~1 second.
Acceptable for a 15-minute control loop.

### Blower coupling

Blowers redistribute heat from the hydronic slab — they multiply the
thermostat's effect on the room. This coupling is captured in the sysid gain
matrix (blower gain depends on thermostat activity) and will be learned from
data. No programmatic coupling is needed.

That said, a blower running while the zone thermostat is off provides no
heating benefit (there's no slab heat to redistribute). The existing
constraint "blowers off when zone not heating" applies per-timestep within
the trajectory: if the thermostat shuts off at step 24, the blower timeline
should also stop at step 24. This is physical, not a modeling choice.

### Boiler (navien)

The boiler fires whenever either thermostat calls for heat. Its timeline
is derived: the OR of both thermostat timelines. No independent search
dimension needed.

## Implementation

### `simulator.py`

**`build_activity_timeline`** — add `switch_off_step` parameter:

```python
def build_activity_timeline(
    scenario_activity: float,
    recent_history: list[float],
    n_future_steps: int,
    switch_on_step: int = 0,
    switch_off_step: int | None = None,
) -> list[float]:
    padded = ([0.0] * max(0, _HISTORY_STEPS - len(recent_history))
              + recent_history[-_HISTORY_STEPS:])
    future = []
    for step in range(n_future_steps):
        if step < switch_on_step or (switch_off_step is not None and step >= switch_off_step):
            future.append(0.0)
        else:
            future.append(scenario_activity)
    return padded + future
```

**`batch_simulate`** — accept per-effector trajectory parameters. The
`HVACScenario` type (or a wrapper) carries delay/duration per slow effector.
Timeline generation uses these to produce the piecewise schedule.

### `types.py`

Extend `HVACScenario` or create a `TrajectoryScenario` that includes
per-thermostat delay and duration:

```python
@dataclass(frozen=True)
class ThermostatTrajectory:
    heating: bool
    delay_steps: int = 0       # steps before activation
    duration_steps: int | None = None  # steps of activation (None = full horizon)

@dataclass(frozen=True)
class TrajectoryScenario:
    upstairs: ThermostatTrajectory
    downstairs: ThermostatTrajectory
    blowers: tuple[BlowerDecision, ...]
    mini_splits: tuple[MiniSplitDecision, ...]
```

### `control.py`

**`generate_scenarios`** — produce `TrajectoryScenario` objects with the
delay × duration grid for thermostats, constant for fast effectors. Apply
existing constraint pruning (comfort max, thermal direction) to the
trajectory space.

**`sweep_scenarios_physics`** — passes trajectory scenarios to
`batch_simulate`. Scoring is unchanged: `compute_comfort_cost` evaluates
all sensors at all horizons exactly as before.

**Blower timeline coupling**: when building timelines, the blower's
switch_off_step matches the thermostat's switch_off_step for its zone.
If the thermostat is OFF for the first S steps, the blower is OFF too.

**Cold-room override**: unchanged. If the sweep picks all-off but a room is
below comfort_min - threshold, re-score with constrained heating. The
constraint now applies to trajectory candidates (must include heating
with delay=0).

### `decision_log.py`

Log the winning trajectory's delay and duration for each slow effector, so
we can observe what the system is planning and verify it makes physical
sense.

## What this subsumes

- **PLAN-7 "inertia check"**: the original pre-sweep lookahead that detects
  future breaches and forces heating as a constraint. This is subsumed by
  the trajectory search: "start heating in 2h for 2h" is just a candidate
  that the scorer evaluates alongside "start now" and "don't heat." If
  delayed start is optimal, the search finds it without a separate system.

- **Pre-heating logic** (FUTURE.md): "use forecast + sysid to start heating
  before outdoor temp drops." Trajectory candidates with delay=0 and
  long duration achieve exactly this — the scorer sees that early start
  prevents breaches that delayed starts can't.

- **MPC trajectory planning** (FUTURE.md, partial): this is a constrained
  version of MPC. Not arbitrary action sequences, but parameterized
  trajectories (delay + duration). It captures the most important structure
  (when to start, when to stop) without the full optimization complexity.

## Verification

1. **Overnight cooling** (outdoor drops 45 -> 35F, rooms at 69F, comfort
   min 68F): system selects a trajectory with early start (delay=0) and
   moderate duration (2-3h), not constant-ON or reactive late start.

2. **Mid-charge continuation** (heater on for 40 min, room hasn't warmed):
   system selects "continue for 1-2h more" (delay=0, duration=1-2h) over
   "stop now" (OFF), because the trajectory shows the slab delivering heat
   soon.

3. **Proactive start** (cold front arriving in 4h): system selects delay=2h,
   duration=2h — pre-charges the slab so warmth arrives as cold hits.

4. **No heating needed** (all rooms well above comfort): trajectory search
   agrees with constant-action search — OFF is best at all delays/durations.

5. **Multi-sensor tradeoff**: upstairs thermostat heating for 4h overcooks
   upstairs_temp but keeps bedroom_temp comfortable. System picks 2-3h
   duration as the whole-house optimum.

6. **Blower interaction**: with downstairs heating for 2h, system discovers
   that blower=high produces better comfort than blower=off (more heat
   distributed from slab to room), selected purely by the scorer.

## What we explicitly defer

- **Full MPC**: arbitrary multi-segment action sequences ("heat 2h, off 1h,
  heat 1h"). The single-segment trajectory captures most of the value.
  Receding horizon (re-evaluate every 15 min) handles multi-segment
  naturally over time.

- **Continuous duration optimization**: searching over discrete durations
  (1h, 2h, 4h, 6h). Gradient-based optimization over continuous
  durations is possible but the discrete search is simple and the
  receding horizon compensates for granularity.

- **Probabilistic scoring**: the scorer uses point predictions. A
  survival/hazard framing would account for forecast uncertainty and
  prediction error. See FUTURE.md for this direction.

- **Online parameter adjustment**: using prediction errors to update
  sysid gains and lags at runtime.

## Prerequisites

- Physics simulator complete (done)
- Sysid parameters available (done)
- Forecast data flowing to control cycle (done)
- Comfort schedules with time-of-day awareness (done)
