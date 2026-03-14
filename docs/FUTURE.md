# Future Work

Items are roughly in priority order within each section.

---

## Summer / Cooling Adaptation

All current data is winter (collector running since Feb 2026). The system
will face new challenges in summer:
- Mini splits are the only cooling effectors — capacity analysis will be
  important for rooms without them.
- Auto mode (heat_cool) may be needed in shoulder seasons where the room
  needs heating in the morning and cooling in the afternoon.
- Solar gain profiles will change significantly — sysid needs spring/summer
  data to fit accurate seasonal profiles.
- Window advisory logic should recommend opening windows for free cooling
  more aggressively when outdoor temp is below indoor but above comfort min.

---

## Additional Blower Automation

More blowers in the hydronic circuit would improve heat distribution to
capacity-limited sensors (e.g., kitchen: three exterior walls, crawlspace,
end of the hydronic circuit — consistently below comfort min even with zone
thermostat at max). Physical installation, not software.

---

## Full MPC Trajectory Planning

Optimize multi-segment HVAC sequences ("heat 2h, off 1h, heat 1h") instead
of the single-segment trajectories in PLAN-7. The trajectory search captures
most of the value; full MPC handles cases where the optimal plan has multiple
heating/cooling phases within one horizon.

The physics simulator is fast enough; the question is whether the multi-segment
optimization surface is tractable or whether receding-horizon re-evaluation
(every 15 min) effectively discovers multi-segment plans through repeated
single-segment decisions.

---

## Discrete Hazard / Survival Model for Comfort

Frame comfort prediction probabilistically: at each future timestep, for each
sensor, estimate a hazard rate h(t, s) = P(breach at t | no breach before t)
conditional on the chosen trajectory. The survival function S(t) = P(staying
within comfort through time t) replaces point-prediction scoring.

**Why this matters:**
- Point predictions ignore uncertainty. The simulator says "room hits 68.0F at
  step 48" but the real outcome is a distribution (forecast error, model error,
  unmodeled disturbances like doors opening).
- A survival framing values trajectories that keep breach probability low across
  the full horizon, not just at scored checkpoints (1h, 2h, 4h, 6h).
- It naturally handles asymmetric risk: a trajectory where the 95th-percentile
  outcome breaches comfort is worse than one where the median is slightly
  further from optimal but the tail is safe.

**Practical approach:**
- Estimate prediction uncertainty from historical residuals (simulator
  prediction vs actual outcome, binned by horizon and conditions).
- At each scored horizon, compute P(T < comfort_min) and P(T > comfort_max)
  from the predicted mean and estimated variance.
- Weight comfort cost by breach probability rather than deterministic penalty.

**Prerequisites:** enough decision-log history to estimate prediction error
distributions by horizon. The comfort dashboard (`just comfort --predictions`)
already shows error distributions by horizon.

---

## Online Learning

Continuously improve thermal model parameters by comparing predictions to
outcomes. The decision log already records predicted vs actual temperatures
at each horizon. The next step is automatic parameter adjustment (exponential
moving averages on gain/delay/tau) when systematic prediction errors appear.

The comfort dashboard shows a consistent warm bias (+0.15 to +0.30°F,
increasing with horizon) — this likely represents underestimated heat loss
(air infiltration, radiation) that the tau model doesn't fully capture.

---

## Sweep Scalability

Batch prediction gives ~30ms per sweep. Further optimization is only needed
when the device count grows significantly.

**Future approaches (in priority order):**
- **Window decomposition** — O(N) vs O(2^N). Independent per-window toggle.
  Needed at 10+ windows.
- **Greedy coordinate descent** — O(Σ levels) vs O(∏ levels). Iterative
  single-device optimization. Needed at 6+ blowers.
- **Marginal screening** — Fast linear approximation, full eval on top-K.
- **Spatial decomposition** — Only re-predict sensors affected by each device
  change. Composes with all above.

---

## Solar Irradiance Estimation

Approximate solar irradiance from existing + forecast data:
- `solar_elevation` (already computed per-row)
- Weather-conditioned solar fractions (already implemented: sunny=1.0,
  cloudy=0.15, etc.) modulate the per-hour solar gain profiles.
- Future extension: continuous irradiance model from cloud cover percentage
  (if available from met.no) rather than discrete condition codes.

---

## Virtual Thermostats

Per-sensor HA climate entities for user-adjustable comfort targets from the
dashboard. Each sensor appears as a `climate` entity with target_temp_high and
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
