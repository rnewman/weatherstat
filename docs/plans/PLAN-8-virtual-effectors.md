# PLAN-8: Virtual Effectors and Advisory-Driven Planning

## Problem

The system can control thermostats, blowers, and mini-splits electronically.
But many actions that affect room temperatures require human intervention:
opening/closing windows, adjusting blinds, turning on a space heater,
opening interior doors, starting a fireplace. These actions have thermal
effects that the physics model can predict — but the system can't execute
them. It can only recommend.

Currently, the advisory system (`advisory.py`) exists but is disconnected
from the control loop. It was previously driven by ML predictions, which
have been removed. The infrastructure (HA notifications, cooldown timers,
quiet hours) is intact but unused.

## Core Design Constraint

**A human may not act on the advisory, or may act late.**

The controller must plan for the world as it is, not as it hopes it will be.
This means:

1. **Always commit to the best electronic-only plan.** The trajectory sweep
   finds the optimal thermostat/blower/mini-split trajectory assuming windows
   stay as they are. Execute that plan immediately.

2. **Separately evaluate: would a human action improve things?** After
   choosing the electronic plan, ask "if the human opened the bedroom window,
   would we get a better outcome?" If yes, send an advisory.

3. **React to reality.** If the human acts (sensor detects window opened),
   the next 15-minute cycle re-evaluates with the new state. The controller
   naturally adjusts its electronic plan to complement the human action.
   If the human doesn't act, the electronic plan is still good.

This is the same pattern the executor uses for manual overrides: plan for
the current state, adapt when it changes.

## Virtual Effector Concept

A **virtual effector** is anything that:
- Has a measurable state (sensor exists)
- Affects room temperatures (sysid can fit a gain)
- Cannot be electronically actuated (requires human action)

Examples:

| Virtual Effector | Sensor | Thermal Effect | Sysid Parameter |
|-----------------|--------|----------------|-----------------|
| Window (open/close) | binary_sensor | Changes tau (sealed→ventilated) | tau_sealed, tau_ventilated |
| Interior door | binary_sensor | Changes inter-room coupling | Could be modeled as coupling gain |
| Space heater | power sensor or switch | Direct heating | Gain (°F/hr) per sensor |
| Blinds/shades | cover entity | Modulates solar gain | Solar profile modifier |
| Fireplace | temperature spike detection | Direct heating | Gain (°F/hr) per sensor |

Windows are special: they don't add or remove heat, they change the envelope
loss rate. The simulator already handles this via `tau_sealed` vs
`tau_ventilated`. Other virtual effectors would enter the physics model the
same way as electronic effectors — gain × activity in the dT/dt equation.

## Implementation

### Phase 1: Window Advisories via Physics Simulator

Windows are the first virtual effector because the simulator already models
them (sealed vs ventilated tau), and binary sensors already exist.

**After the trajectory sweep completes and commits to an electronic plan:**

1. For each window (or combination of windows), re-run `batch_simulate`
   with the winning electronic trajectory but with toggled window state(s).

2. Score the toggled prediction against comfort schedules.

3. If toggling a window improves comfort cost beyond the effort threshold
   (from YAML `advisory.effort_cost`), recommend the action.

4. Dispatch via the existing advisory infrastructure: HA notification with
   cooldown timer and quiet hour suppression.

**Key detail:** The electronic plan does NOT change based on the advisory.
The advisory is purely informational. The controller has already committed
to the best electronic trajectory given current window states.

**Scenarios this handles:**

- *Summer afternoon, office overheating:* Sweep picks mini-split cooling.
  Advisory check: "opening the office window with mini-split off would be
  cheaper and achieve the same comfort." Notify user. If user opens window,
  next cycle sees the open window and turns off the mini-split.

- *Winter morning, heating active:* Sweep picks thermostat on. Advisory
  check: toggling windows makes things worse. No advisory sent.

- *Spring evening, all rooms warm:* Sweep picks all-off. Advisory check:
  "opening bedroom and family room windows would cool to sleep temp faster."
  Notify user.

### Phase 2: Expand to Other Virtual Effectors

For each new virtual effector type:

1. **Add to YAML config:** Sensor entity, affected rooms, expected thermal
   effect direction.

2. **Add to sysid:** Treat as an effector in the stage-2 regression. The
   regression will fit its gain and delay from observed data, just as it
   does for thermostats and blowers.

3. **Add to simulator:** Include in timeline building and dT/dt integration.

4. **Add to advisory evaluation:** After electronic plan is committed,
   evaluate virtual effector toggles.

The same pattern scales to any number of virtual effectors because:
- Sysid handles arbitrary effectors (YAML-driven)
- The simulator handles arbitrary activity timelines
- Advisory evaluation is a post-sweep check, not part of the main sweep

### Phase 3: Combined Advisory Evaluation

With multiple virtual effectors, evaluate combinations. A 2^N sweep over
N virtual effectors is feasible for small N (we have ~8 windows, giving
256 combos). For larger N, use greedy coordinate descent (toggle each
independently, take the best, repeat).

## What This Is Not

- **Not co-optimization.** The electronic and virtual effector plans are
  not jointly optimized. The electronic plan is committed first; virtual
  effector advisories are evaluated second. This is intentional — joint
  optimization would produce plans that rely on human compliance.

- **Not reinforcement learning.** There's no reward signal for whether
  the human followed the advisory. We don't penalize or adapt based on
  compliance. If the human consistently ignores window advisories, the
  system keeps sending them (within cooldown limits) because each advisory
  is evaluated on its own merits.

- **Not a new sweep dimension.** Virtual effectors are not added to the
  main trajectory sweep. That would multiply the search space by 2^N for
  no benefit — we can't execute the virtual actions anyway.

## Files to Modify

| File | Change |
|------|--------|
| `control.py` | After sweep: evaluate virtual effector toggles, call advisory |
| `advisory.py` | Already has notification infrastructure; add physics-based evaluation |
| `simulator.py` | Already handles window states; may need to accept per-step window timelines for doors/blinds |
| `weatherstat.yaml` | Virtual effector definitions (mostly already there for windows) |
| `sysid.py` | Treat virtual effectors as effectors in regression (future, for non-window types) |

## Prerequisites

- Trajectory sweep working (PLAN-7, done)
- `advisory.py` notification infrastructure intact (it is)
- Window binary sensors configured and collecting data (they are)
