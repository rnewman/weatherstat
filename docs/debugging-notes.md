# Debugging Notes

Recurring issues, root causes, and diagnostic approaches for the
weatherstat control loop. Updated as new oddities arise.

---

## Prediction Sanity Failures

### Symptom: sensor 1h prediction >5°F from current reading

**Root cause taxonomy:**

1. **No effective gains** — all sysid-learned effector gains filtered by
   |t| < 1.5 or magnitude > 3.0°F/hr in the simulator. The sensor
   becomes pure Newton cooling toward outdoor temp, losing all HVAC
   influence. Check `thermal_params.json`: look for the sensor's gains
   and count how many pass the simulator's filter.

2. **Confounded gains** — OLS learns spurious correlation (e.g., heating
   correlates with cold weather → inflated gains). The gain magnitude
   looks reasonable but the sign or cross-sensor coupling is wrong.
   Ridge regularization (added 2026-03-21) mitigates but doesn't
   eliminate this.

3. **Insufficient tau segments** — sensors with < 5 nighttime HVAC-off
   segments get tau_base = 500h (the cap), which is unrealistically
   slow cooling. This overpredicts temperature stability when HVAC is
   off but underpredicts cooling when outdoor is cold (because Newton
   term is divided by tau).

### Case study: office_bookshelf_temp (2026-03-21)

Current reading 72.0°F, predicted 66.2°F at 1h — a 5.8°F drop.

**Investigation:**
- tau_base = 500h (only 3 segments — unreliable fit, hit cap)
- 0/7 effector gains pass simulator filter (all |t| < 1.5)
- No window_betas learned (empty)
- 4 windows open (bedroom, piano, bathroom, living_room) but no
  cross-coupling modeled
- Prediction is pure Newton cooling: dT/dt = (T_outdoor - 72) / 500
  → should be ~0.06°F/hr, not 5.8. But the simulator uses forecast
  outdoor temps, and with tau=500 the Newton term is tiny — the
  solar gain corrections at this hour may actually dominate.
  **TODO**: run `predict` with verbose to trace per-term contributions.

**Resolution**: removed office_bookshelf_temp from constraint schedules.
office_temp (same room, different sensor) has 18 tau segments and
stable (if still low-t) gains. The bookshelf sensor remains as a
monitoring input.

**Structural issue**: the sensor is confounded by unobserved heat sources
(occupant body heat, space heater) that correlate with occupancy, which
also correlates with HVAC usage patterns.

---

## Opportunity Benefit Values

### Symptom: benefit=714.29 (absurdly large)

**Root cause**: comfort cost is quadratic, summed across N rooms × 4
horizons with weights. Raw cost difference between scenarios can be
100-1000+ while thresholds are 0.3-1.5.

**Fix (2026-03-21)**: normalize by `len(schedules) * sum(horizon_weights)`.
This gives per-room per-weighted-horizon average cost improvement.
Thresholds now represent:
- 0.3 = ~0.5°F average deviation improvement (track)
- 1.5 = ~1.2°F average deviation improvement (notify)

If benefits still seem high, check whether many windows are open
simultaneously — genuine large improvements are expected when closing
a window on a cold night.

---

## Ridge Regularization Effects (2026-03-21)

### What changed

Sysid gain regression switched from OLS (with ridge fallback for
high condition numbers) to always-on ridge with λ = 0.01 × n_rows.

### Observed effects

**Positive:**
- Confounded thermostat_downstairs gains shrunk 6-217× across sensors
  (these were all spurious — caused by heating↔cold-weather correlation)
- office_bookshelf thermostat_downstairs: -10.978 → -1.293 (8.5× shrink)
- bedroom_temp → mini_split_bedroom t-stat improved 1.22 → 1.52
  (now passes simulator's |t| ≥ 1.5 filter — was previously filtered out!)
- bedroom_aggregate → mini_split_bedroom: t improved 5.69 → 7.54
- bedroom_aggregate → blower_family_room: t improved 1.65 → 1.80

**Negative / trade-off:**
- Solar gain estimates also shrunk (~5-8× for piano). No solar gains
  are now statistically significant. This means MRT weights are all
  empty (no differentiation between sun-facing and north-facing rooms).
- Root cause: binary solar indicators have low variance (~1/24 active),
  so their coefficients need to be large to fit the data. Ridge penalizes
  large coefficients regardless of feature scale.

### Potential improvement: standardized ridge

Scale each feature column to mean=0, std=1 before regression, then
un-standardize coefficients afterward. This would penalize all features
equally regardless of their natural scale, preserving solar gains while
still shrinking confounded effector gains.

### Tuning notes

- Current λ = 0.01 × n_rows. For n ≈ 10000, λ = 100.
- Reducing to 0.001 × n would preserve more signal but less shrinkage.
- Per-feature-group λ (different for effectors vs solar vs windows)
  is more principled but more complex.
- Cross-validation for λ selection is the textbook approach but adds
  compute time to sysid.

---

## Diagnostic Playbook

### "Why is sensor X predicted at Y°F?"

1. Check `thermal_params.json` for the sensor's gains, tau, window_betas
2. Count how many gains pass simulator filter (|t| ≥ 1.5, magnitude ≤ 3.0)
3. If zero gains pass: prediction is pure Newton + solar. Check tau_base
   and forecast outdoor temp.
4. Run a single predict with verbose logging to see per-term contributions.
5. Check window states — open windows with learned betas increase cooling.

### "Why did the system recommend opening/closing window X?"

1. Check the opportunity's comfort_improvement and energy_saving fields
2. Re-run with the window toggled to see which rooms benefit most
3. Look at the re-sweep: does changing the window change the optimal
   HVAC plan? (e.g., "open window + turn off mini split")

### Future: `just explain <sensor>`

A debugging tool that:
1. Shows the sensor's thermal parameters (tau, gains, window_betas)
2. Runs predict for each term individually (Newton, each effector, solar)
3. Prints a breakdown of temperature contributions at each horizon
4. Identifies which terms dominate and whether gains are filtered

This would replace the manual investigation done for office_bookshelf.
