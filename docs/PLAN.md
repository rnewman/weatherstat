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
- Window-open comfort adjustment (soft penalty — widen comfort bounds when window open)
- Advisory quiet hours (suppress push notifications during sleep, still log)
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

### Sweep Scalability
- Current: 180 HVAC combos + 128 window combos, ~4.4s total
- At 6 blowers + 10 windows: ~26K HVAC combos + 1,024 window combos — untenable
- Batch prediction (10–50× speedup, no algorithmic change) is the first step
- Window decomposition (O(N) vs O(2^N)) and greedy coordinate descent (O(Σ levels) vs O(∏ levels)) follow
- See `docs/FUTURE.md` for the full scalability plan

## Near-Term

### Weather Forecast Integration (next)
- Fetch hourly forecasts from HA (`weather.get_forecasts` service on `weather.forecast_home`)
- Forecast outdoor temp, condition, wind speed at each prediction horizon
- Piecewise Newton integration: chain hourly segments using forecast outdoor temps
  instead of constant current outdoor temp
- Forecast features for ML: outdoor temp, cloud cover, wind at each horizon
- Solar irradiance estimate: solar elevation × cloud factor from forecast condition
- Store forecast snapshots in collector for training data
- See `docs/FUTURE.md` for full design

### Retrospective HVAC Features (parallel with forecast)
- Cumulative heating minutes, duty cycle, time-since-transition features
- Captures slab thermal charge state — "boiler ran 3h" ≠ "boiler ran 5min"
- Computed from recent history already fetched during feature engineering
- See `docs/FUTURE.md` for feature list

### Physics as Sweep Guardrails (done)
- Newton floor/ceiling on all-off predictions during HVAC sweep
- Handles winter (model over-predicts warmth) and summer (model over-predicts coolness)
- Cold-room safety override as secondary fallback
- Design: physics guardrails at sweep level, ML predicts absolute temps with physics features

### Virtual Thermostats (Per-Room Climate Entities)
- Expose per-room target high/low as HA climate entities
- User adjusts comfort targets from the HA dashboard or phone (no YAML editing)
- Control loop reads targets from HA each cycle instead of (or defaulting from) YAML
- Solves the "adjust from bed at 2 AM" problem — change targets without restarting the loop

## Medium-Term

### Pre-Heating Logic
- Uses forecast + HVAC response curves to start heating before outdoor temp drops
- Critical for hydronic floor heat with 2–4 hour thermal lag
- Requires forecast integration + retrospective features to estimate slab charge state

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

**Prerequisites:**
- Window→room mapping in YAML ✓ done
- Sweep scalability for larger action spaces — see `docs/FUTURE.md`

### Full MPC
- Model Predictive Control: optimize over multi-step HVAC trajectories
- Multi-zone coordination (upstairs heat is ceiling heat for downstairs)
- Requires fast-evaluating thermal model — physics + modular ML approach provides foundation

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
