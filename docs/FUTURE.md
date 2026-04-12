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
- ~~Window advisory logic~~ → Done: generalized as "advisory effectors" (Stage 25).
  Windows, space heaters, blinds, vent fans are swept in the trajectory search
  alongside HVAC effectors. Three planning layers (reasonable/worst-case/proactive)
  replace the old per-window re-sweep.
- Proactive free-cooling advice: advisory sweep evaluates "open this window for
  2h" scenarios and surfaces them when cost delta is significant.

---

## Irradiance-Based Solar Model

**Status:** Data collection started 2026-04-05. Elevation-based model deployed
2026-04-05 as an intermediate step (see below).

**Intermediate step (done):** The per-hour model (11 coefficients per sensor,
hours 7–17) was replaced by an elevation-based model: one coefficient per
sensor × `sin⁺(solar_elevation)` × weather-conditioned fraction. Solar
elevation is computed analytically from lat/lon/time — no external data needed.
This captures seasonal variation (Feb noon ~30°, Apr ~48°, Jun ~66° at Seattle)
and hour-of-day variation in a single continuous feature. The old model
underpredicted spring solar gain by ~35% because winter-fitted coefficients had
no way to increase with sun angle. Cloud coverage (0–100%) is now collected in
the snapshot data alongside the discrete weather condition, for future use as a
continuous solar fraction.

**What the elevation model doesn't capture:** directionality. A west-facing
room heats more in afternoon than morning at the same elevation. The irradiance
model (below) will handle this with per-plane coefficients.

**Data collection:** forecast.solar HA integration provides irradiance
forecasts for 5 planes (horizontal + 4 cardinal walls) at hourly resolution.
Power sensors (`solar_power_{horizontal,south,west,east,north}`) are
collected every 5 minutes in the EAV table.

**Target model:** Replace the single elevation coefficient per sensor with
a small number of irradiance gain coefficients per sensor. Each sensor's
solar forcing becomes a linear combination of the 5 plane irradiances:

    solar_gain(sensor, t) = Σ_plane β(sensor, plane) × irradiance(plane, t)

Sysid fits β coefficients from regression against temperature derivatives.
The compound geometry of the house (windows facing multiple directions, roof
pitch, glass vs wall, shading) is captured in the β values — no physical
modeling of individual surfaces needed.

**Prerequisites:** Weeks to months of irradiance + temperature data to fit
reliable β coefficients across different weather conditions and sun angles.

**Benefit:** The irradiance signal naturally encodes sun altitude (seasonal),
hour of day, cloud cover, and orientation — all in one W/m² number per plane.
The elevation model already handles seasonal + hourly variation; the irradiance
model adds directional resolution and replaces the coarse weather-condition
fraction with actual measured energy.

---

## Accumulated Wall Heat for MRT Correction

**Problem:** The current MRT correction uses instantaneous solar forcing
(`sin⁺(elev) × weather_fraction`) to estimate wall surface temperature.
But wall temperature is the integral of energy flux over time — a thermal
capacitor, not a resistor. The same solar angle produces very different
wall temperatures at 9am vs 3pm:

| Time | Solar elev | Outdoor | Exterior wall | Interior wall | MRT correction needed |
|------|-----------|---------|---------------|---------------|----------------------|
| 9am  | 35°       | 42°F    | ~42°F (cold from night) | ~62°F | Large (cold walls) |
| 3pm  | 35°       | 50°F    | ~75°F (6h of sun) | ~72°F | Small or negative (warm walls) |

Our current model sees the same solar angle and similar outdoor temp at
both times, producing similar MRT corrections. But the 9am walls are cold
(need more heating) and the 3pm walls are warm (need less). The error is
asymmetric: we under-correct cold mornings and over-correct warm
afternoons.

**Subtlety — double counting:** The air temperature sensor already
captures much of the accumulated solar heating. A room that's had hours
of sun reads high 70s, and the control system responds to that reading
directly. The MRT correction is only for the *residual* — the perceived
comfort difference between air temperature and operative temperature
caused by warm/cold surfaces radiating more/less IR. This residual is
real (you feel warmer near a sun-heated wall even if the air temp is the
same), but it's smaller than the total wall heating effect. Getting the
magnitude right requires separating "wall warmth that raised the air
temp" from "wall warmth that raises perceived comfort above air temp."

**Model:** Replace the instantaneous solar term in the MRT correction with
an exponential moving average of solar forcing:

```
accumulated_solar(t) = Σᵢ sin⁺(elev(t - i·Δt)) × SF(t - i·Δt) × exp(-i·Δt / τ_wall) × Δt
```

where `τ_wall` is the wall thermal time constant (estimated 3–6 hours for
typical residential walls — much shorter than the air time constants sysid
fits, because this is surface temperature, not through-wall conduction).

The effective outdoor temperature becomes:

```
effective_outdoor = outdoor + β_solar × accumulated_solar × solar_response
```

instead of the current:

```
effective_outdoor = outdoor + β_solar × sin⁺(elev) × SF × solar_response
```

**Behavior:**
- **Morning:** accumulated_solar is low (sun just rose, little energy
  stored) → full cold-wall correction → system heats more aggressively.
  Matches lived experience that the house feels coldest first thing.
- **Afternoon:** accumulated_solar is high (hours of solar energy in
  walls) → reduced correction → system backs off. Matches lived
  experience that late afternoon feels warmest.
- **After sunset:** accumulated_solar decays exponentially → correction
  gradually increases as walls cool. Captures the evening cool-down.
- **Cloudy day after sunny morning:** accumulated_solar reflects the
  morning sun even though current SF is low. Walls don't forget.

**Implementation path:**

1. At MRT correction time, read recent `weather_condition` values from the
   collector (last ~4×τ_wall hours). Compute historical `sin⁺(elev)` for
   each timestamp analytically (no data dependency — just lat/lon/time).
2. Weight by exponential decay with time constant `τ_wall`.
3. Pass the scalar `accumulated_solar` to `apply_mrt_correction` in place
   of (or alongside) the instantaneous `current_solar_elev × SF`.
4. Add `tau_wall` as a configurable parameter in `MrtCorrectionConfig`
   (default ~4 hours).

The collector data read is the main new dependency — currently MRT
correction is a pure function of current conditions. The control loop
already has snapshot access nearby (`latest_snapshot_values`), but we'd
need a lookback query. `_load_from_readings()` with a time filter is
straightforward.

**Calibration challenge:** The `solar_response` scaling factor would need
re-tuning after switching from instantaneous to accumulated solar, since
accumulated values are larger (sum vs single sample) and have different
units (energy-like vs power-like). The τ_wall parameter itself could
potentially be learned from data — if we observe systematic morning
under-heating and afternoon over-heating in the decision log residuals,
the optimal τ_wall minimizes that pattern.

**Priority:** Medium. The magnitude of the instantaneous→accumulated
error is estimated at 0.3–1.0°F in MRT correction, primarily affecting
the first and last hours of the solar day. Worth implementing when the
morning cold-wall issue becomes noticeable in comfort metrics, or when
the irradiance model (above) provides better per-plane solar data that
makes the accumulated model more accurate.

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
(every 5 min) effectively discovers multi-segment plans through repeated
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

Sysid now runs automatically in the TUI (default hourly, configurable via
`sysid_interval`), with a quality gate that rejects bad fits. This handles
gradual seasonal drift but doesn't yet do incremental parameter updates.

The next step beyond periodic refitting is automatic parameter adjustment
(exponential moving averages on gain/delay/tau) when systematic prediction
errors appear. The decision log records predicted vs actual temperatures at
each horizon, providing the data for this.

The comfort dashboard shows a consistent warm bias (+0.15 to +0.30°F,
increasing with horizon) — this likely represents underestimated heat loss
(air infiltration, radiation) that the tau model doesn't fully capture.

---

## Sweep Scalability

Batch prediction gives ~160ms per sweep (~14K scenarios). The unified effector
model generates scenarios as the cartesian product of per-effector options,
so count grows as `|options|^N_independent`.

**Current:** All effector types sweep delay × duration grids (9 combos per
mode). 2 trajectory (10 each) × 2 regulating (1 + modes×targets×9 delay/dur
each, ~37 with idle suppression) × 3 binary (1 + modes×9, ~19 each) —
potentially large cross products, managed by the existing coarsening logic
(reduce at >50K, hold-all at >100K).
Adding a 3rd trajectory effector: ~100K+ scenarios, ~1s sweep.

**Future approaches (in priority order):**
- **Trajectory grid reduction** — Coarser delay/duration grid for N>2
  trajectory effectors. Receding horizon compensates.
- **Window decomposition** — O(N) vs O(2^N). Independent per-window toggle.
  Needed at 10+ windows.
- **Greedy coordinate descent** — O(Σ levels) vs O(∏ levels). Iterative
  single-device optimization. Needed at 6+ blowers.
- **Marginal screening** — Fast linear approximation, full eval on top-K.
- **Spatial decomposition** — Only re-predict sensors affected by each device
  change. Composes with all above.

---

## Causal Gain Identification

The current sysid OLS regression fits effector→sensor gains from observational
data, which produces confounded estimates. Examples observed (Mar 2026):

- **Bedroom mini split → every sensor in the house** had positive gains (t<1.0).
  The split runs when the house is warming for other reasons (solar, thermostats),
  so OLS attributes the warming to the split. When the optimizer sets the split to
  cool, it inverts all these gains and thinks cooling the bedroom also cools the
  bathroom, kitchen, etc.
- **Office bookshelf sensor** had gains of 4-6°F/hr from thermostats — physically
  implausible. Likely caused by unobserved heat sources (occupant body heat, space
  heater) correlated with HVAC activity.
- **Negative downstairs→living room coupling** (-2.5°F/hr): downstairs heating
  only activates in genuinely cold weather, which is also when the living room
  cools fastest. Simpson's paradox, not a real causal effect.

**Current mitigations** (guard rails, not root fixes):
- t-statistic threshold (|t| ≥ 1.5) prunes statistically insignificant gains
- Gain magnitude cap (≤ 3.0°F/hr) catches physically implausible outliers
- Mode-direction clamp prevents cooling from warming or heating from cooling

**Future approaches:**
- Include outdoor temperature as a control variable in the gain regression
  (partial out weather effects before fitting effector gains)
- Asymmetric gain fitting: train heating and cooling gains separately (hot air
  and cold air flow differently — convection patterns differ)
- Instrumental variable approaches if sufficient data accumulates
- Per-sensor data quality flags (short history, known unobserved heat sources)

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
