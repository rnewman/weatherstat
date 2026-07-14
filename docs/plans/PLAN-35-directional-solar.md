# PLAN-35: Directional solar with tracked canopy response

## Problem

The solar model is a single per-sensor coefficient on `sin⁺(elevation) × SF`.
Two measured failures (July 2026 residual analyses, `scripts/slab_residual_analysis.py`
and `scripts/maple_shading_analysis.py`):

1. **No directionality.** The afternoon post-off residual rises to +0.4–0.5°F
   late afternoon; the west forecast.solar plane delivers 311 W at 17:00 when
   `sin⁺(elev)` is already declining. West-facing evening sun is structurally
   invisible to the model.
2. **Obstruction is not static.** The property is ringed by deciduous trees
   (fig by the kitchen, maples over the living/family rooms, vine maple,
   dogwood) plus evergreens and structures. Measured: the kitchen's
   west-plane response was +0.67 ± 0.15 °F/hr per unit feature with bare
   trees (Feb–early Apr) and −0.04 ± 0.04 after leaf-out (t = 4.7). A static
   coefficient averages the regimes and is wrong in both, flipping error
   sign at leaf-drop.

**Rejected: configured phenology.** An earlier draft modulated the βs with a
fixed calendar leaf-fraction curve. Phenology follows accumulated warmth,
not the calendar — transition dates shift by weeks year to year, so a
configured curve is systematically wrong at exactly the transitions. The
canopy's effect is directly observable in the residual at the timescale it
changes (weeks), so the design below *tracks* it instead of predicting it.

## Design

### Features: analytic plane irradiance

Same code at fit and predict time (no train/serve skew, computable for the
full history back to Feb 2026):

```
f_H(t) = sin⁺(elev)                       × SF(t)   # today's feature, unchanged
f_E(t) = max(0, cos(elev)·cos(az −  90°)) × SF(t)   # east wall
f_W(t) = max(0, cos(elev)·cos(az − 270°)) × SF(t)   # west wall
```

- East/west near-orthogonal (measured r = −0.05); south redundant with
  horizontal (r = 0.96), omitted.
- Azimuth via the same Spencer-formula family as `solar_elevation`;
  validated against measured forecast.solar planes (correct phase) in
  `scripts/maple_shading_analysis.py`. Promote `solar_azimuth` and
  `plane_feature` into `weather.py` with tests.
- forecast.solar stays calibration-only (it is a model, not a sensor — it
  cannot see trees).

### Two-speed fitting

**Slow (batch, hourly refit as today):** the full-history Stage-2 regression
fits effector gains, tau betas, weather controls, and *baseline* plane βs
per sensor. These βs are regime-averaged — they serve as prior/fallback,
not as the runtime solar response. Existing machinery (selective
standardization, t-stat, bootstrap CV, VIF) applies unchanged.

**Fast (tracked block, runs with the hourly refit):** conceptually, every
sysid coefficient is a state with a timescale; the batch fit is the
timescale-∞ special case. This plan adds the general mechanism — a
per-feature-block trailing-window tracker with a random-walk prior — and
instantiates it for one block: solar transmission. Solar goes first because
it is the only block with *measured* drift (the canopy) and because its
carrier is sharp and exogenous (zero at night, plane-specific phase,
astronomically scheduled, independent of controller actions), which is what
makes fast tracking identifiable. Tau's carrier (T_out − T) is broadband and
effector activity is closed-loop confounded — granting them finite
timescales without new evidence (see the gain-drift comparison in the
health summary) or excitation would reintroduce the confounding the ML
retrospective documents. The tracker code is block-generic; enabling a
second block is config plus evidence, not new machinery.

Per sensor, a small trailing-window regression (default ~14 days) of the
*residual after all non-solar terms* on the three plane features yields the
current directional response γ_p, regularized toward the previous estimate
(random-walk prior):

```
minimize ‖r − F·γ‖² + λ_track·‖γ − γ_prev‖²
```

so γ moves only as fast as the data demands. Quality gates:
- **Information gate:** if the window's solar variance is too low (dark
  Seattle fortnight), hold γ_prev (`_MIN_ACTIVE_ROWS` pattern).
- Stage-34 magnitude cap and sign constraint (γ ≥ 0) apply.
- Cold start / degraded mode: fall back to the batch baseline βs.

**Why the lag is acceptable:** the tracker trails a transition by roughly
half a window — but it can only lag when the sun isn't shining (no signal →
no update → also no prediction error). When sunny days expose the new canopy
state, the same days update γ. The parameter is only stale when it doesn't
matter. A calendar curve "anticipates" transitions but with the wrong dates
every year; lagging a weeks-long observable beats predicting it wrongly.

**Free byproduct:** the logged γ_p(t) time series *is* the measured
per-room canopy phenology — next year's leaf-out is data, not assumption.
It also absorbs every other slow directional drift for free: pruning a
tree, a new fence, window film, seasonal blind habits.

### Simulator / control

- `thermal_params.json`: `solar_plane_gains` (batch baseline) +
  tracker state (γ per sensor per plane, with timestamp and window
  information metric). `solar_elevation_gains` kept as backward-compat
  fallback per the established pattern.
- `HouseState`: extend precomputed per-step `solar_elevations` with per-step
  plane factors (control.py computes analytically, as elevations today).
  Forcing: `Σ_p γ_p · f_p(t)`. Solar stays scenario-independent —
  superposition (stage 30) untouched.
- MRT: `effective_outdoor` uses the per-sensor directional sum — west rooms
  feel-like warmer at 18:00.
- TUI parity (CLAUDE.md convention): app.py mirrors the new solar and MRT
  computation; display γ alongside gains in `just debug gains`.

## Phases

1. **weather.py**: `solar_azimuth`, `plane_feature` + tests (phase check
   against stored forecast.solar data).
2. **sysid batch**: plane features replace the single elevation feature,
   behind the experiment harness. Variants: `baseline` (frozen) → `+planes`,
   judged on holdout RMSE, VIF, per-sensor grades. Sentinels: kitchen,
   piano.
3. **Canopy tracker**: trailing-window γ fit + random-walk prior +
   information gate; runs from the hourly refit path (runtime.sysid_refit);
   γ history logged. Tests: synthetic transmission step converges within
   ~window; dark-window hold; cap/sign gates.
4. **simulator + control + MRT + TUI**: plane forcing from γ, params
   backward-compat, per-step precomputation, display parity.
5. **Acceptance**: rerun `scripts/slab_residual_analysis.py` with the new
   model — daytime post-off residuals should collapse toward the night
   values (≈ 0); the afternoon decay-curve rise should flatten. Bare→leafed
   replay: run the tracker over Feb–Jul history and confirm the kitchen γ_W
   reproduces the measured collapse. `just validate` + full suite.

## Risks

- **Weak identification in dark windows** is handled by the information
  gate, but a marginal window could still pass with noisy γ. The
  random-walk prior bounds the step size; tune λ_track on the Feb–Jul
  replay.
- **Confound absorption is by design** (γ soaks up any directional
  residual drift) but means γ is not purely "canopy" — e.g., a systematic
  forecast bias could leak in. The γ history log makes such drift visible.
- **VIF** between f_H and wall planes is the main batch-fit risk; harness
  gates it. Fallback: fit E/W as asymmetry (f_W − f_E) over the shared f_H.
- **Two parameter speeds** add operational surface: the tracker must never
  outlive its gates (a runaway γ is a comfort bug). Validation envelope
  tests extend to tracker output.
