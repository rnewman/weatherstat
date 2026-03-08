# Future Work

Items are roughly in priority order within each section.

---

## Virtual Effectors / Advisory Planning (PLAN-8, next)

Model human-actionable changes (windows, blinds, space heaters, doors) as
virtual effectors. After the electronic trajectory sweep commits to a plan,
evaluate whether toggling virtual effectors would improve comfort and send
advisories via HA notifications. The electronic plan never depends on human
compliance. See `docs/plans/PLAN-8-virtual-effectors.md`.

---

## Archive ML Pipeline

The LightGBM training/inference/experiment pipeline is orphaned — the
control loop uses the physics simulator exclusively. Move training, inference,
evaluation, experiment, and metrics modules to `archive/ml/`. Extract
`fetch_recent_history` to `extract.py` first (it's data plumbing used by
the live control path). See `docs/plans/PLAN-archive-ml.md`.

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

**How it connects to the trajectory search:**
- Each trajectory candidate produces a different hazard curve for each sensor.
  "Heat now for 2h then coast" front-loads hazard reduction. "Delay 2h then
  heat" has higher near-term hazard but similar long-term reduction.
- The scorer becomes: maximize expected comfort-survival across all sensors,
  weighted by room importance and penalty asymmetry.

**Practical approach:**
- Estimate prediction uncertainty from historical residuals (simulator
  prediction vs actual outcome, binned by horizon and conditions).
- At each scored horizon, compute P(T < comfort_min) and P(T > comfort_max)
  from the predicted mean and estimated variance.
- Weight comfort cost by breach probability rather than deterministic penalty.
- This is incremental over the deterministic scorer — same structure, but
  penalties are expected values over the predictive distribution.

**Prerequisites:** enough decision-log history to estimate prediction error
distributions by horizon. The trajectory search (PLAN-7) should come first;
probabilistic scoring refines it.

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
