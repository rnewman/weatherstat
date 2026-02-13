# Weatherstat Roadmap

## Done

### Stage 1: Pipeline & Data Infrastructure
- HA WebSocket + REST extraction (hourly statistics, 5-min raw history)
- Collector: 5-min snapshots to SQLite, durable runner with health monitoring
- LightGBM training: baseline (5mo hourly) and full (5-min multi-source)
- Full-model training merges historical Parquet + collector SQLite automatically
- Feature engineering: time, solar position, weather, HVAC state, lag/rolling
- Multi-horizon prediction: T+1h through T+12h per zone
- Evaluation framework comparing baseline vs full model

### Stage 2: Control Loop
- Setpoint sweep over thermostat pairs
- Comfort schedules (per-room, time-of-day, asymmetric penalties)
- Safety rails: setpoint clamps, hold time, staleness check, sanity limits
- Dry-run mode (default) + live execution via HA services
- Counterfactual analysis ("what if setpoints were X?")

### Stage 3: Per-Room Models & Full HVAC Control
- 8-room prediction: upstairs, downstairs, bedroom, kitchen, piano, bathroom, family_room, office
- 5 horizons each (1h, 2h, 4h, 6h, 12h) → 40 models per training mode
- Unified HVAC sweep: thermostats (4) × blowers (3^2) × mini-splits (3^2) = 180 combos (~2.4s)
  - Physical constraint: blowers forced off when their zone's thermostat is off
  - Mini-split sweep over off/heat/cool with fixed representative target; command target derived from comfort schedule midpoint post-sweep
- Config-driven device lists (`BlowerConfig`, `MiniSplitConfig`) — adding a blower is a one-line change
- Tiered energy cost: gas > heat pump > fan (tiebreaker when comfort is equal)
- Executor handles all device types: thermostats, mini-splits (mode + target), blowers (off/low/high)

## In Progress

### Data Accumulation
- Collector running continuously (started Feb 2026)
- All current data is winter — seasonal coverage needs months
- HVAC feature importance should increase as dataset grows
- Retrain weekly to incorporate new collector data

### Dry-Run Control Validation
- Monitoring control loop decisions before going live
- Verifying recommendations match physical intuition (e.g. blower artifact caught and constrained)

### Human Advisory Notifications
- Window advisories: "open windows for free cooling" and "close windows — heating is active"
- Evaluates after each control cycle using current sensor state
- Sends HA persistent notifications (replaces stale notifications via notification_id)
- Cooldown timers prevent notification fatigue (4h free cooling, 1h close windows)
- Future: comfort breach warnings, solar gain/blinds, room migration suggestions

## Near-Term

### Experiment Infrastructure
- Git worktrees sharing `data/` so experiments don't disrupt the running collector/control loop
- Experiment-namespaced model directories (`data/models/{experiment}/`)
- Eval harness comparing experiment models against production baseline
- Enables physics features and MPC work in parallel without risk

### Physics-Informed Features
- Heating/cooling rate (dT/dt), thermal deficit, estimated time-to-setpoint
- Helps the model learn control-relevant dynamics without a physics simulation
- Implement as an experiment branch first, promote if metrics improve

### Virtual Thermostats (Per-Room Climate Entities)
- Expose per-room target high/low as HA climate entities
- User adjusts comfort targets from the HA dashboard or phone (no YAML editing)
- Control loop reads targets from HA each cycle instead of (or defaulting from) YAML
- Solves the "adjust from bed at 2 AM" problem — change targets without restarting the loop
- Complements device-level override detection (target overrides vs device overrides)
- Climate entity shows: current room temp, target range, heating/cooling/idle action
- YAML comfort schedule provides defaults; HA entities override when set
- Could add "reset to schedule" automation that restores YAML defaults at a set time

## Medium-Term

### Hybrid Physics + ML
- Blend ML predictions (good at short horizons) with exponential-decay physics model (long horizons)
- Fixes extrapolation issues when setpoints move outside training distribution
- See `docs/modeling-strategy.md` for the math

### Automated Retraining
- Cron job or launchd plist to retrain weekly
- Track model metrics over time to detect drift

### Unified Action Framework
Every controllable action is a feature perturbation — the model doesn't care whether
a feature changed electronically or because someone opened a window. The optimization
should be unified; only the execution method differs.

**Action abstraction:**
- Each action has: name, feature columns + state→value mapping, current state,
  execution type (electronic or advisory), energy cost, effort cost
- Electronic: thermostats on/off, blowers off/low/high, mini-splits off/heat/cool
- Advisory: windows open/closed, blinds open/closed
- Future: motorized blinds (electronic), ventilation fans, etc.

**Unified sweep:**
- Evaluate all actions together in one optimization pass
- Output splits into electronic commands (executor) and advisory notifications
- Effort cost penalizes bothering the human — only suggest physical actions when
  the predicted improvement materially exceeds what electronic actions achieve alone
- Naturally discovers cross-domain optimizations: "open the window instead of running
  the mini split on cool", "close blinds to prevent afternoon overshoot"

**Replaces rule-based advisories:**
- Current `advisory.py` heuristics (free cooling, close windows) become emergent
  properties of the optimizer rather than hand-coded rules
- Advisory suggestions come with predicted temperature impact ("closing the bedroom
  window would keep the room at 69°F instead of dropping to 66°F")

**Prerequisites:**
- Window→room mapping in YAML (needed for targeted suggestions)
- Blind sensors in HA (cover entities or binary sensors)
- Effort cost tuning (how much "cost" to assign to human interruption)

### Full MPC
- Model Predictive Control: optimize over multi-step HVAC trajectories
- Exploration vs exploitation strategy for broadening training distribution
- Multi-zone coordination (upstairs heat is ceiling heat for downstairs)

## Long-Term

### HA Packaging
Three integration levels under consideration:
- **Add-on** (Docker, easiest): package as-is, supervisor manages lifecycle, file-based IPC stays.
  Lowest effort but clunky.
- **Integration** (Python HA component): collector becomes `DataUpdateCoordinator`, predictions
  stored as HA entities, control decisions as automations. Native but significant rewrite.
- **Hybrid** (likely best): training stays external (too heavy for HA), inference + control
  become a lightweight HA integration reading models from a shared volume.
  Collector and executor merge into the integration, eliminating file-based IPC.
