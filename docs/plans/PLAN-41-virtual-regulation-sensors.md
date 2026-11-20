# PLAN-41: Virtual regulation sensors

## Problem

A regulating effector (mini split) runs its own control loop. It compares
its target to the temperature it reads and modulates itself. Sysid learns
the *effect* side of that loop: when the unit runs at activity *a*, a room
warms at gₛ·*a* °F/hr. Nothing models the *cause* side, which is how hard
the unit will run for a given target. That depends on what the unit reads.

The simulator filled that gap by computing a separate activity for each room
from that room's own temperature, as if every room were the unit's sensor.
That is physically impossible, and the optimizer exploited it: it planned
"cool" with the room the unit serves already below its band, because the
plan "cooled" a warmer neighbour at full activity, did nothing to the cold
room, and cost little energy. Separately, idle suppression judged "would
the unit be idle?" against the highest-gain room, so a near tie between two
rooms' gains (0.002 apart) decided that every heat option was idle.

What a unit actually reads is usually a hand-built Home Assistant average
(a `min_max` helper, say) pushed to it as its remote temperature. Such
averages are a balancing act, drift as the setup changes without anyone
deciding they should, and fail silently: when a member's battery dies and
its reading freezes, the average keeps including the stale value.

## Idea

Weatherstat **provides** the sensor each regulating effector regulates on.
The value is computed from the real sensors, weighted by the effector's
learned gains, and published every minute. The unit's input is then known
exactly, because it is Weatherstat's own formula. The simulator evaluates
the same formula on simulated room temperatures. No model of the unit's
input is fitted, and no average is configured.

**Why weight by gain.** Let the unit's activity be *a*, its learned gains
gₛ, and the comfort cost quadratic around each preferred temperature. The
cost changes with activity in proportion to Σ gₛ·(prefₛ − Tₛ). With
V = Σ wₛ·Tₛ and wₛ ∝ gₛ, the unit's error (target − V) is the comfort
gradient along the one direction the unit can push the house. Between
planning cycles the unit's fast internal loop works on the right signal. The
planner's job reduces to choosing the target on V's scale every 5 minutes.

## Design

### 1. Weights (`regulation.py`)

- **Source gains.** The effector → constrained-sensor gains *after*
  load-time filtering (`load_sim_params`: t-stat, magnitude, sign), the
  gains the simulator uses, so V and the simulation agree by construction.
- **Members.** Positive gains only, wₛ = gₛ / Σg. A negative or filtered
  gain is not a member.
- **Stability.** Weights are recomputed after each sysid refit but replace
  the adopted set only when some weight moves by more than
  `defaults.regulation_weight_tolerance` (0.05). Refit noise would otherwise
  step V and make the unit cycle.
- **History.** Every adopted set is appended to `regulation_history.jsonl`,
  so any published V can be reproduced.

### 2. The virtual input

V = Σ wₛ·Tₛ over the **available** members, renormalized over the weight
they carry.

- **Available** means present in HA, not `unavailable`/`unknown`, and not
  frozen (stage 40).
- Below `defaults.regulation_min_coverage` (0.6) of the weight, V is
  *unavailable* and the unit falls back to its own thermistor; a wrong
  number would be worse.
- A membership change shifts V's level, so it starts an immediate control
  cycle to re-plan the target on the new scale (the overlay wake path).

### 3. Publishing (MQTT bridge)

- One `sensor.weatherstat_<effector>_input` per regulating effector:
  `device_class: temperature`, `state_class: measurement`, the configured
  unit, availability from the bridge's last will, and attributes for the
  weights, members used, dropped members with the reason, coverage and the
  weight-set timestamp.
- `force_update: true` records every publish (`last_reported` advances). It
  does **not** keep the unit fed: HA's ESPHome integration forwards a
  subscribed entity to a board only when its state changes, so the board
  re-asserts the last value itself (`ha/README.md` § Mini split inputs).
- `expire_after: 180`, so a hung process (not just a disconnected one)
  makes the entity unavailable.
- Every 60 s, on a timer independent of the control cycle: one
  `/api/states` fetch, compute, publish. It runs in the TUI and the headless
  loop; a lock file in the data dir keeps it to one publisher.

### 4. Simulator: one activity per unit

- The Euler loop steps all sensors jointly, state shaped
  `(n_scenarios, n_sensors)`: rooms don't couple to each other, but a
  unit's input mixes several rooms.
- Each step: V = Wᵀ·T from the members' simulated temperatures; activity
  from the unit's learned response to (target − V) (§ Findings); each
  sensor's contribution is gₛ × activity, with lag.
- **Signed gains.** The signed gain replaces `abs(gain)`, keeping the
  mode-direction clamp (heating cannot cool, cooling cannot warm). `abs()`
  had turned a negative cross gain, displayed as "cools", into warming.
- `predict` also returns each unit's activity trace per scenario, for
  dedupe and energy.
- Superposition (stage 30) holds only approximately for regulating units:
  activity depends on V, and V on the other effectors. Exact refinement of
  the top K corrects for it, as before.

### 5. Planner

- **Setpoints on V's scale.** The bottom, middle and top of the
  gain-weighted preferred band [Σ wₛ·pref_loₛ, Σ wₛ·pref_hiₛ], each turned
  into the setpoint that makes this unit hold that level given its learned
  response (`ResponseCurve.setpoint_for`), in 0.5° steps, clipped to
  `setpoint_min`/`setpoint_max`, crossed with delays and 2 h / 6 h
  durations.
  - The levels are absolute. In heat mode a level below V is a floor: the
    unit idles, and holds the level when the house drifts down to it; a
    level above V heats up to it and stops. Cool mode mirrors this. The
    planner picks the level by cost, and the unit's thermostat regulates
    between cycles.
  - No level exceeds the preferred band, so one cold low-weight room cannot
    drive the setpoint above what every room the unit serves prefers.
  - *As first built* the candidates were V + {0.5, 1, 2}. Relative
    setpoints move with V at every re-plan, so the unit never reached one
    and never stopped itself, and for a unit that overshoots, "V + 0.5"
    meant flat out. A request to cool a room was then met with heat for
    hours whenever the horizon past the request's expiry wanted the room
    warm.
- **No heuristic pruning.** Idle suppression is removed.
- **Dedupe.** Options whose marginal activity traces are identical collapse
  into one, and an all-zero trace collapses into off. This is exact for the
  marginal, and it is where the search space shrinks.
- **No MRT target adjustment.** `target_air = preferred + mrt_offset` is
  removed: MRT already applies to predictions at scoring time, so the
  target is just a point on V's scale.
- **Mode hold** works on the reduced option set: the held mode's options
  are exempt from dedupe-to-off.

### 6. Energy

Regulating energy cost is `energy_cost` × the mean simulated activity over
the horizon (from the marginal traces), replacing a guess from the current
temperature.

### 7. Health: does the unit follow V?

Once a unit reads V, its reported `current_temperature` should equal V
within the unit's rounding. Units report their 0.5°C steps as whole °F, so
the tolerance is 1.5°F; a larger gap for more than 10 minutes raises an
alert (TUI, `sensor.weatherstat_status`, a persistent notification). It
means the remote reading timed out, the board reads another entity, or the
unit is on its own thermistor. The check runs for effectors listed in
`defaults.regulation_live`.

### 8. Display

TUI and web show V, members and weights, coverage and the follows-V status
per unit; `just debug regulation` adds history and the learned response;
`decision-why` explains a regulating decision in V terms; the decision log
records V and the weight-set timestamp (`regulation` column).

## Rollout

1. **Build** §§ 1–8, with the planner using V while the units still read
   whatever they read before.
2. **Shadow run.** V is published and planning uses it, but the units still
   regulate on the old input. The mismatch (V vs what the unit reads) is
   measured, not hidden.
3. **Switch-over**, in the house's Home Assistant rather than this repo:
   find the consumers of whatever the units read now, point each unit's
   remote temperature at `sensor.weatherstat_<effector>_input` (board
   configuration in `ha/README.md`), drop any collected column for the old
   average from the config, and add the effector to
   `defaults.regulation_live`.

## Validation

**Unit tests.**
- Weights ∝ positive filtered gains, sum to 1, exclude negative and
  filtered gains; changes under the tolerance are not adopted, changes over
  it are, and go to history.
- A frozen or unavailable member is dropped and the rest renormalized;
  coverage below the minimum → unavailable; a membership change requests a
  control cycle.
- Every sensor's regulating contribution is gₛ times the *same* activity
  trace. The exploit case (warm neighbour, cold served room, cool setpoint
  between them) simulates exactly as off; heat warms both rooms in
  proportion to their gains. A negative cross gain never warms under heat
  or cools under cool.
- Identical traces collapse, all-zero → off, the held mode survives dedupe;
  superposition stays within the approximate-vs-exact tolerance.
- Discovery carries `force_update`, `expire_after`, `device_class` and
  availability; one publisher under the lock, every 60 s; unavailable when
  coverage is insufficient.

**Offline, before the shadow run.**
- From each unit's observed reading and target, the response model must
  predict its observed heating/cooling/idle state.
- Prediction error at 1/2/4/6 h must be no worse in aggregate, and should
  be better on the rooms the units affect.
- Option count and cycle time, before and after.

**After switch-over.** The unit's reading tracks V within rounding and the
follows-V check stays green; stopping Weatherstat makes the entity
unavailable within 3 minutes and the unit falls back to its thermistor;
comfort over the following week is no worse than a comparable week before.

## Findings

**Units switch at an offset, and heat and cool differ.** A unit's on/off
behaviour against (target − its own reading) is close to a step, but the
step sits away from the setpoint, by an amount that differs per unit and per
mode: a unit with a built-in heating offset holds a few degrees past its
heat setpoint. So the response is learned, not configured (it replaces
`proportional_band`): sysid fits a ramp clip((sign·(target − V) + offset) /
band, 0, 1) per unit and per mode from (error, running) history, on a grid
over the half-on point from 2° short of the setpoint to 4° past it and bands
up to 3°. A mode with fewer than 500 rows uses the pooled fit. Units that
report whole degrees leave a gap between "always on" and "never on" in which
every switch point scores the same, so the fit takes the middle of the tie.
A first, pooled fit on a grid capped at 2° put one unit's heat switch point
more than a degree short, so every heat setpoint held its rooms warmer than
planned; the planner's setpoint translation uses the curve for the plan's
mode.

**Delivery data adds little.** Refitting sysid with the units' activity
regressor built from reported `hvac_action`, and from target-only ramp and
sigmoid models (`scripts/delegated_regulation_experiment.py`): the best
target-only model explains about 70% of what `hvac_action` explains, and
sub mode and power add little to `hvac_action`, except that some "heating"
rows are idle sub-mode rows. So gains stay fitted on observed action, the
best signal with full history. Fitting them on model-implied activity would
rescale them and discard it. The simulator still needs the response model,
because it cannot observe future action. A sub-mode gate is a later
refinement once enough data exists.

**Accuracy is unchanged; behaviour is fixed.** Backtest
(`scripts/backtest_regulation.py`) on windows in which every effector's
command and state held constant: RMSE matches the old simulator within
±0.01°F for every sensor at 1 h and 2 h; with a unit running, some rooms are
slightly better and some slightly worse. "No worse in aggregate" is met;
"better on the rooms the units affect" is not. The change buys correct
behaviour (no per-room exploit, no wrongly pruned heat), not accuracy.

**Sweep cost.** Built with the relative grid, the sweep was about 35% slower
than before (the old count was low because idle suppression pruned options,
including heat options that were needed). Band-anchored levels roughly halve
the combinations and the sweep time again. A third or more of each unit's
options fold away as exact duplicates. Memory is dominated by
`_score_combinations` (combinations × targets × 8 bytes).

## Later

- **Comfort-weighted inputs** (wₛ ∝ cₛ·gₛ, using the cold/hot penalty) to
  bias the unit's fast loop toward rooms that matter more.
- **Modelling the unit's 0.5°C quantization** of V in the simulator.
- **Virtual inputs for trajectory effectors** (floor thermostats regulate
  on their own wall sensors and are not remotely fed).
