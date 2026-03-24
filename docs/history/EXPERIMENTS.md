# Experiments

Log of feature engineering and modeling experiments run against the production
full models. All experiments use the same dataset and 80/20 chronological
validation split.

**Baseline context:** The full model has ~3°F RMSE at 1–6h horizons and ~1.5°F
at 12h due to a small dataset (~3k rows) and a chronological val split that
happens to be easier at 12h. All current data is winter 2026.

---

## physics_v1 — Physics-informed features

**Branch:** `experiment/physics_v1`
**Worktree:** `weatherstat-physics_v1`
**Date:** Feb 2026

**Hypothesis:** Adding temperature rate-of-change features (dT/dt, d²T/dt²,
heating power interaction) gives the ML model physics-informed signals that
improve long-horizon predictions.

**Features added** (in `features.py`):
- `{room}_dT_3` / `{room}_dT_6` / `{room}_dT_12` — temperature rate of change
  over 3, 6, and 12 time steps (15min, 30min, 1h windows)
- `{room}_d2T` — thermal acceleration (second derivative)
- `heating_power_interaction` — thermostat action × outdoor temperature delta

**Results:** 15 wins, 13 losses, 12 ties (vs production)

| Room | 1h | 2h | 4h | 6h | 12h |
|------|----|----|----|----|-----|
| upstairs | exp -0.02 | tie | exp -0.01 | exp -0.04 | prod +0.03 |
| downstairs | tie | tie | tie | exp -0.01 | **exp -0.08** |
| bedroom | exp -0.01 | prod +0.01 | exp -0.07 | exp -0.01 | exp -0.02 |
| kitchen | tie | prod +0.02 | prod +0.03 | prod +0.01 | **exp -0.15** |
| piano | prod +0.02 | exp -0.01 | prod +0.03 | exp -0.03 | tie |
| bathroom | prod +0.01 | exp -0.03 | prod +0.04 | prod +0.03 | tie |
| family_room | tie | tie | tie | prod +0.01 | **exp -0.24** |
| office | tie | prod +0.01 | prod +0.01 | tie | **exp -0.18** |

*Deltas are RMSE change in °F (negative = experiment is better).*

**Conclusions:**
- Physics features help most at 12h, with large improvements for downstairs
  rooms (family_room -0.24, office -0.18, kitchen -0.15). These rooms have
  the most thermal coupling to outdoor conditions.
- At 1–6h horizons, results are mixed — roughly as many regressions as
  improvements, and the magnitudes are small (<0.07°F).
- The 12h wins are the most significant because that's where the production
  model is already best (lower RMSE), so relative improvements matter more.
- **Verdict:** Worth merging the rate-of-change features. The 12h gains are
  real and the short-horizon regressions are negligible.

---

## hybrid_physics — Exponential decay blending

**Branch:** `experiment/hybrid_physics`
**Worktree:** `weatherstat-hybrid_physics`
**Date:** Feb 2026

**Hypothesis:** Blending ML predictions with a physics-based exponential decay
model (Newton's law of cooling) improves long-horizon accuracy. At short
horizons ML dominates; at long horizons the physics prior (temperature decays
toward equilibrium) should be more robust.

**Approach:**
- Newton's law of cooling: `T(t) = T_eq + (T_now - T_eq) * exp(-t / tau)`
- Equilibrium temperature depends on HVAC state (setpoint when heating, outdoor
  blend when off, mini-split target when active)
- Blending weight: `w_ml = exp(-hours_ahead / crossover_hours)` with
  crossover=3.5h gives ML 75% at 1h, 32% at 4h, 3% at 12h
- Evaluation is row-by-row using per-row HVAC state for physics predictions

**Results:** 6 hybrid wins, 33 ML wins, 1 tie (crossover=3.5h, 568 val rows)

| Room | 1h | 2h | 4h | 6h | 12h |
|------|----|----|----|----|-----|
| upstairs | hybrid -0.12 | ml +0.02 | ml +0.79 | ml +2.24 | ml +7.26 |
| downstairs | ml +0.06 | ml +0.29 | ml +1.28 | ml +2.78 | ml +7.63 |
| bedroom | hybrid -0.01 | ml +0.06 | ml +0.89 | ml +2.27 | ml +6.78 |
| kitchen | hybrid -0.04 | ml +0.07 | ml +1.00 | ml +2.13 | ml +7.02 |
| piano | hybrid -0.16 | hybrid -0.14 | ml +0.88 | ml +2.42 | ml +7.06 |
| bathroom | hybrid -0.05 | tie | ml +0.72 | ml +1.81 | ml +6.29 |
| family_room | ml +0.06 | ml +0.32 | ml +1.38 | ml +2.90 | ml +7.84 |
| office | ml +0.08 | ml +0.38 | ml +1.60 | ml +3.29 | ml +8.61 |

*Deltas are RMSE change in °F (negative = hybrid is better).*

**Why it failed:**
1. **Uncalibrated physics model.** The evaluation uses default tau values (6h
   heating, 8h cooling) because there's no recent time series per validation
   row to estimate tau from. Physics-only RMSE is 4–10°F — far worse than ML.
2. **Blending degrades ML at long horizons.** At 6h+ where physics weight is
   82%+, the uncalibrated physics model's high error dominates. The blend
   makes predictions worse by +2–9°F compared to ML alone.
3. **ML is already flat across horizons.** The full model's RMSE is ~3°F at all
   horizons (an artifact of the small dataset + chronological split), so
   there's no "ML falls apart at long horizons" gap for physics to fill.

**Conclusions:**
- Post-hoc blending with an uncalibrated exponential decay model does not work.
- The physics_v1 approach (physics as ML features) is strictly better — it
  gives the ML model the physics signal without replacing its predictions.
- A calibrated version (estimating tau from a sliding window of recent data)
  might work, but would need careful engineering and much more data to validate.
- **Verdict:** Do not merge. The physics-as-features approach from physics_v1
  is the right way to incorporate thermal dynamics.
