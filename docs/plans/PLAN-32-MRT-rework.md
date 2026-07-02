# Plan: Per-horizon MRT correction on predictions

## Context

MRT correction compensates for the gap between air temperature (what sensors measure) and operative temperature (what humans feel). Cold exterior walls lower perceived temp; warm/sunny walls raise it.

**Current approach**: Modifies comfort schedule bounds once per cycle using current outdoor temp and solar state. Two problems:
1. The correction is frozen at current conditions but predictions span 1–6 hours, during which outdoor temp, solar elevation, and cloud cover change significantly (e.g., sunny afternoon → evening sunset).
2. Shifting bounds away from user-configured values causes confusing display messages ("76.7°F above backup 77°F" when backup_hi was shifted from 77 to 76.5 by MRT).

**New approach**: Leave comfort bounds at user-configured values. Apply per-horizon MRT offsets to predicted temperatures before comfort scoring ("feels-like" correction). The offset at each horizon uses that horizon's forecast outdoor temp, solar elevation, and solar fraction — all data we already have.

**Math equivalence**: `pred > (bound + offset)` ↔ `(pred - offset) > bound`. The sign convention: `offset = α × (ref - effective_outdoor)`. Positive when cold (need warmer air). `feels_like = air_temp - offset` (cold walls → feels colder than measured).

## Files to modify

### 1. `src/weatherstat/control.py` — Core changes

**a. Replace `apply_mrt_correction` (lines 217–305)**

Replace with two functions:
- `compute_mrt_offset(outdoor_temp, mrt_config, *, solar_gain, solar_elev, solar_frac, mrt_weight) → float`
  Single-sensor, single-moment offset. Used for current-moment needs (regulating targets, cold-room override, display).
- `compute_horizon_mrt_offsets(sensors, mrt_config, forecast_temps, solar_elevations, solar_fractions, solar_elevation_gains, mrt_weights) → dict[str, dict[int, float]]`
  Per-sensor, per-horizon offsets. Maps `sensor_col → {horizon_step → offset}`. Uses forecast outdoor temp at each horizon, solar elevation at each horizon step, solar fraction at each horizon's hour.

Keep the old function signature as a thin wrapper for backward compat in tests if needed, or update all callers.

**b. `_ComfortSpec` (lines 471–527)**

Add `mrt_offsets: np.ndarray` field (shape `(n_active,)`, default zeros). In `build()`, accept optional `mrt_offsets: dict[str, dict[int, float]]` parameter. During the `(schedule, horizon)` loop, look up `mrt_offsets[schedule.sensor][h]` for each entry and store in the array.

**c. `_batch_comfort_cost` (lines 530–546)**

After extracting `temps = pred_matrix[:, spec.col_indices]`, apply MRT:
```python
temps = temps - spec.mrt_offsets  # broadcasts (n_active,) over (N, n_active)
```
This converts air temp predictions to feels-like before comparing to (uncorrected) bounds.

**d. `compute_comfort_cost` and `compute_comfort_cost_by_sensor` (lines 324–420)**

Add `mrt_offsets: dict[str, dict[int, float]] | None = None` parameter. Inside the loop, look up offset for `(schedule.sensor, h)` and subtract from `pred_temp` before scoring.

**e. `_check_backup_breaches` (lines 1372–1405)**

Add `mrt_offsets` parameter. Subtract per-horizon offset from `pred` before comparing to backup bounds. Display format already changed to `:.1f` (from earlier fix).

**f. Cold-room override (lines 1656–1671)**

Currently compares `temp < comfort_entry.acceptable_lo - COLD_ROOM_OVERRIDE` using MRT-shifted bounds. Change to apply current-moment MRT offset to `temp`:
```python
feels_like = temp - current_mrt_offsets.get(schedule.sensor, 0.0)
if feels_like < comfort_entry.acceptable_lo - COLD_ROOM_OVERRIDE:
```

**g. Regulating sweep options (lines 734–743) — IMPORTANT**

Currently reads `preferred_lo`/`preferred_hi` from MRT-corrected schedules to
generate target temperatures for regulating effectors (mini splits). After the
refactor, schedules are uncorrected, so targets need explicit MRT adjustment.

**Why regulating targets need MRT correction:**

A regulating effector (e.g., mini split) has a proportional activity model:
activity ramps from 0→1 as room air temp deviates from the target within a
`proportional_band`. The target temperature is the *air* temperature the device
physically aims for. But the user's comfort preference is about *perceived*
(operative) temperature.

If the user wants to feel 72°F and MRT says cold walls make 72°F air feel like
70.5°F, the device must target 73.5°F air to achieve 72°F perceived. The
relationship is:

```
target_air = preferred_operative + mrt_offset
```

where `mrt_offset = α × (ref - effective_outdoor)` is positive when cold.

Without this adjustment, the optimizer would only have "off" and "heat@72" in
its candidate set. Scoring would penalize 72°F air (which feels like 70.5°F)
for being below preferred, but no better option exists in the sweep — the
optimizer can't discover 73.5°F if it's not offered. The result: the mini split
heats to 72°F, the room feels cold, and the optimizer reports a nonzero comfort
cost it can't reduce.

**Implementation:** Use current-moment MRT offset (not per-horizon), because
the target temperature is set immediately and held for the duration:
```python
mrt_adj = current_mrt_offsets.get(sensor_col, 0.0)
heat_targets.append(comfort.preferred_lo + mrt_adj)
cool_targets.append(comfort.preferred_hi + mrt_adj)
```

Add `current_mrt_offsets: dict[str, float] | None = None` parameter to
`_regulating_sweep_options`. Document the relationship between operative
preference and air target in the function docstring.

**h. Control loop caller (lines 2268–2282)**

- Remove `apply_mrt_correction` call that modifies schedules
- Compute current-moment per-sensor offsets (for regulating targets, cold-room, display)
- Compute per-horizon offsets (for scoring)
- Pass horizon offsets through to `_ComfortSpec.build()`, `compute_comfort_cost()`, `_check_backup_breaches()`
- Pass current-moment offsets to `_regulating_sweep_options()` and cold-room override

**i. `_score_all` inner function (line 1612–1621)**

Pass `mrt_offsets` to `_ComfortSpec.build()`.

**j. `sweep_scenarios_physics` signature**

Add `mrt_offsets` parameters (current-moment dict and horizon dict) or compute them inside the function from forecast data that's already assembled there.

### 2. `src/weatherstat/tui/controller.py` — TUI snapshot building

- Stop calling `apply_mrt_correction` to modify schedules (line 335)
- Compute per-sensor current-moment offsets for display (keep the offset computation)
- Pass uncorrected comfort bounds to `StatusSnapshot`
- Continue passing `mrt_per_sensor` offsets for grey display text

### 3. `src/weatherstat/decision_log.py` — Decision logging

- `comfort_bounds` logged at lines 145–162 will now be uncorrected (user-configured values) — this is an improvement for the comfort plotter
- `mrt_offsets` JSON (lines 164–170): update to store per-sensor applied offsets from the new computation. Consider storing per-sensor offsets (already does) rather than `base × weight`
- `log_decision` signature: change `mrt_base_offset`/`mrt_weights` params to accept `mrt_per_sensor: dict[str, float]` directly

### 4. `tests/test_control.py` — Test updates

- `TestMrtCorrection` (lines 316–456): Rewrite tests to verify `compute_mrt_offset` and `compute_horizon_mrt_offsets` return correct offsets (not that they shift schedule bounds)
- `TestSunAwareMrt` (lines 462–547): Same — test offset values, not shifted schedules
- Add new tests:
  - Per-horizon offsets differ when forecast temps change
  - Per-horizon offsets differ when solar elevation changes (day→night)
  - Scoring with MRT offsets matches old scoring with shifted bounds (equivalence test)
  - `_batch_comfort_cost` applies MRT offsets correctly

### 5. Display paths (no functional changes)

- **TUI widgets** (`src/weatherstat/tui/widgets.py`): No change — already displays raw temp + offset
- **Web** (`src/weatherstat/web.py`): No change — same display pattern
- **Debug script** (`scripts/debug_state.py`): No change — shows MRT from decision log
- **Comfort plotter** (`scripts/plot_comfort.py`): Will naturally show uncorrected bounds now (improvement — bounds won't drift with weather)

## Implementation order

1. Add `compute_mrt_offset` and `compute_horizon_mrt_offsets` functions
2. Add `mrt_offsets` field to `_ComfortSpec`, update `build()` 
3. Update `_batch_comfort_cost` to apply offsets
4. Update `compute_comfort_cost`, `compute_comfort_cost_by_sensor`, `_check_backup_breaches`
5. Update `_regulating_sweep_options` to accept current-moment offsets
6. Update cold-room override
7. Update control loop caller to compute and pass offsets instead of modifying schedules
8. Update `_score_all` and `sweep_scenarios_physics` plumbing
9. Update TUI controller
10. Update decision logging
11. Update tests

## Data flow after refactor

```
Control loop:
  outdoor_temp, forecast_temps, solar_elevations, solar_fractions
    ↓
  compute_horizon_mrt_offsets() → {sensor: {horizon: offset}}
  compute_mrt_offset() per sensor → {sensor: offset}  (current moment)
    ↓
  Schedules: UNCORRECTED (user-configured values)
    ↓
  Regulating targets: preferred + current_mrt_offset[sensor]
    ↓
  Cold-room: (current_temp - current_mrt_offset) vs acceptable_lo
    ↓
  Scoring: _ComfortSpec.build(schedules, mrt_offsets=horizon_offsets)
           → spec.mrt_offsets array
           → _batch_comfort_cost: feels_like = temps - mrt_offsets
           → compare feels_like to uncorrected bounds
    ↓
  Backup breaches: (pred - horizon_offset) vs backup bounds
    ↓
  Display: raw temp + offset shown, bounds at configured values
```

## Documentation updates

### 6. `docs/mrt-correction.md` — Rewrite

The current doc says "shifts all comfort targets uniformly" and has a "Design
Choices" section defending the static (non-per-horizon) approach with "Wall
surfaces have high thermal mass." Update to reflect the new approach:

- **Core change:** MRT correction now applies to predicted temperatures
  ("feels-like"), not comfort bounds. Bounds stay at user-configured values.
- **Per-horizon correction:** Each prediction horizon uses that horizon's
  forecast outdoor temp, solar elevation, and solar fraction. This captures
  afternoon→evening transitions where solar MRT relief fades at sunset.
- **Remove** the "Why static, not per-horizon?" design choice paragraph (the
  argument was about wall thermal mass, but MRT correction is about *perceived*
  comfort, not wall dynamics — and we now have per-horizon data anyway).
- **Update** the "Why shift all targets uniformly?" paragraph — this question
  no longer applies since we don't shift targets at all.
- **Add** explanation of regulating target adjustment: targets are air temps
  that produce the desired operative temp, so they must account for MRT.
- **Add** sign convention documentation: `feels_like = air_temp - offset`
  where offset > 0 when cold (positive offset = cold walls = feels colder).

### 7. `docs/ARCHITECTURE.md` — Update scoring/decision sections

- Line 342: Update "Resolved comfort bounds per sensor... after MRT correction"
  → comfort bounds are now uncorrected; MRT is applied to predictions.
- Line 365: Update comfort plotter note — bounds in decision log are now
  uncorrected (user-configured values), which is simpler and more stable.
- Add brief note in the scoring section about feels-like correction.

### 8. `CLAUDE.md` — Update stage 16 and conventions

- Stage 16 description currently says "offset-based temperature adjustments"
  for comfort profiles. Add a note that MRT correction was moved from
  bound-adjustment to prediction-correction (new stage number or amendment).
- Add stage 32 entry describing this change.
- The convention "TUI reads temperatures, applies comfort profiles + MRT
  correction" (line 101) should clarify that MRT is now a prediction
  correction, not a schedule modification.

### 9. Memory update

Update `MEMORY.md` entry for MRT correction to reflect:
- `apply_mrt_correction` replaced by `compute_mrt_offset` / `compute_horizon_mrt_offsets`
- Correction applied to predictions (feels-like), not comfort bounds
- Per-horizon offsets using forecast data

## Verification

1. `just test` — all existing tests pass (after updates)
2. `just lint` — clean
3. Manual: `just debug decision-why` — verify comfort bounds show configured values, MRT offset shown separately
4. Manual: `just control` — dry-run cycle, verify comfort costs are comparable to pre-refactor
5. Check: backup breach messages now show sensible numbers (e.g., "77.2°F above backup 77°F" instead of "76.7°F above backup 77°F")
