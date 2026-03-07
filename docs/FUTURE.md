# Future Work

Items are roughly in priority order within each section.

---

## Grey-Box Forward Simulator

The next major step: use sysid parameters to build a physics-based forward
simulator that answers counterfactual questions causally.

**What we have now:**
- `sysid.py` fits the full effector × sensor gain matrix, delays, tau values,
  and solar profiles from collector data
- The controller sweeps candidate actions and scores them using ML predictions
- Newton cooling is used as guardrails (floor/ceiling) on ML predictions

**What's needed:**
- Forward-simulate from current state using sysid parameters:
  ```
  T(t+dt) = T(t) + dt * [
      (T_outdoor - T) / tau           # envelope loss
      + Σ gain_e * activity_e(t-lag)  # effector heating
      + Q_solar(hour)                 # solar gain
  ]
  ```
- Euler integration at 5-minute steps, chaining hourly forecast outdoor temps
- Replace ML predictions in the sweep with physics predictions (or blend)
- The key payoff: reliable counterfactual reasoning ("what if heat is off?")
  that the ML model can't provide due to observational confounding

**Prerequisites:** sysid output is stable and reasonable (done — verified Mar 2026).

---

## Forecast Collector Storage

Store weather forecast snapshots alongside sensor snapshots so training data
includes what the forecast predicted at each point in time. Currently forecasts
are fetched at inference time but not persisted for training.

This would allow training models that learn forecast accuracy patterns (e.g.,
met.no consistently underestimates overnight cooling in Seattle winters).

---

## Pre-Heating Logic

Use forecast + sysid HVAC response curves to start heating before outdoor temp
drops. Critical for hydronic floor heat with 2-4 hour thermal lag.

Requires the forward simulator — the controller needs to simulate "if I start
heating now, will the slab be charged by the time the cold front arrives?"

---

## MPC Trajectory Planning

Optimize HVAC *sequences* (heat 2h then coast 4h) instead of single-step
settings. The current controller picks the best action for *right now* and
re-evaluates every 15 minutes. Full MPC would plan multi-step trajectories.

Requires a fast forward simulator (above). The physics model is fast enough;
the question is whether the optimization surface is tractable.

---

## Sweep Scalability

Batch prediction gives ~30ms per sweep (148x speedup). Further optimization
is only needed when the device count grows significantly.

**Future approaches (in priority order):**
- **Window decomposition** — O(N) vs O(2^N). Independent per-window toggle.
  Needed at 10+ windows.
- **Greedy coordinate descent** — O(Σ levels) vs O(∏ levels). Iterative
  single-device optimization. Needed at 6+ blowers.
- **Marginal screening** — Fast linear approximation, full eval on top-K.
- **Spatial decomposition** — Only re-predict rooms affected by each device
  change. Composes with all above.

---

## Solar Irradiance Estimation

Approximate solar irradiance from existing + forecast data:
- `solar_elevation` (already computed per-row)
- `cloud_cover` (from forecast condition: sunny/partly_cloudy/cloudy)
- `irradiance_estimate = max(0, sin(elevation)) * (1 - cloud_factor) * PEAK`

Sysid already fits per-sensor, per-hour solar gain profiles. This would extend
that to a continuous model that accounts for cloud cover variation within hours.

---

## Virtual Thermostats

Per-room HA climate entities for user-adjustable comfort targets from the
dashboard. Each room appears as a `climate` entity with target_temp_high and
target_temp_low. YAML comfort schedules become defaults.

---

## HA Packaging

Package as an HA add-on or custom integration. See `docs/plans/` for
integration level options.

---

## Anomaly Detection

Detect when the house behaves differently than the model predicts. Persistent
prediction error above a threshold triggers investigation ("cooling rate
doubled overnight — window left open?"). With sysid parameters as a baseline,
residual monitoring becomes straightforward.
