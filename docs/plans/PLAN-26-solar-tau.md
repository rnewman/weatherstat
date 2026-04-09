# Joint Tau + Solar Estimation

## Context

Tau fitting (Stage 1 of sysid) currently restricts to "dark" segments — HVAC off, solar
elevation < 5°. This was necessary because the exponential decay model assumes the only
driver is `(T_indoor - T_outdoor)`, so any unmodeled heat source (solar) biases tau.

**The problem:** Even at solar elev < 5°, building thermal mass retains accumulated solar
heat. The kitchen wall faces south/west (same wall as the solar-blasted side sensor), so
nighttime/twilight segments starting after a sunny day show faster initial cooling from
stored wall heat — biasing tau *downward*. Kitchen tau jumped from 32h to 63h just by adding
twilight data (more segments diluted the biased ones), confirming solar contamination.

**The fix:** Include solar forcing directly in the tau fitting model. Instead of restricting
to segments where solar ≈ 0, model it:

```
dT/dt = -(T - T_outdoor) / τ + β_solar × S(t)
```

This lets us use ALL HVAC-off segments (day and night), dramatically increasing data while
producing a tau that represents the true envelope thermal constant, uncontaminated by solar.

## Physics Model

### Current (dark-only, one parameter)

Per segment: normalize temperatures, fit pure exponential:
```
T(t) = T_outdoor + (T₀ - T_outdoor) × exp(-t / τ)
```
Requires S(t) ≈ 0. One parameter: τ.

### Proposed (joint, two parameters)

Per segment: fit the linear ODE with time-varying solar forcing:
```
dT/dt = -(T - T_outdoor) / τ + β_solar × S(t)
```

Where `S(t) = sin⁺(solar_elevation(t)) × weather_condition_fraction(t)` — the same
`_solar_elev` feature already computed in `_preprocess()`.

**Exact stepwise solution** (piecewise-constant S over 5-min intervals):
```
T_eq(t) = T_outdoor + τ × β_solar × S(t)
T(t + dt) = T_eq(t) + (T(t) - T_eq(t)) × exp(-dt / τ)
```

No Euler approximation. Two parameters: τ, β_solar. Fit via `scipy.optimize.curve_fit`
with solar forcing captured in a closure.

β_solar is a **nuisance parameter** — we need it to get accurate τ, but the "real" solar
gain comes from Stage 2 regression (more data, lag structure, interaction terms). However,
we log β_solar for diagnostics: it should agree roughly with Stage 2's
`solar_elevation_gain`.

### Segment selection changes

Remove the solar elevation filter. Keep everything else:
- All HVAC effectors off (encoded state = 0)
- Stable environment factor state within segment (no window/shade transitions)
- Sensor + outdoor temps non-null
- Minimum segment length ≥ 12 steps (1 hour)
- Contiguous (gaps > 10 min break segments)

This should roughly double the usable segment count (adding all daytime HVAC-off periods).

## Assessment Metrics

These metrics compare the current dark-only approach with the joint approach. The prototype
runs both and prints a comparison table — no structural changes to the pipeline until the
numbers justify switching.

### Primary: Tau inter-segment consistency (CV)

**What:** Coefficient of variation (σ/μ) of per-segment τ estimates, weighted by segment length.

**Why:** If the current approach's τ scatter is mainly from unmodeled solar, the joint
approach should have lower CV. This is the single most important metric — it directly
measures whether we're getting more consistent physics.

**Threshold:** Joint CV < dark-only CV for at least 6/9 sensors = clear win.

### Primary: Segment count

**What:** Number of qualifying segments per sensor.

**Why:** More segments = more robust weighted median, less sensitivity to outliers.
Currently ~22 for mature sensors. Should increase to ~40+ with daytime HVAC-off periods.

### Secondary: Per-segment fit RMSE

**What:** RMS of `T_predicted - T_observed` within each segment, averaged across segments.

**Why:** Joint model should fit daytime segments better (it models solar) and dark segments
equivalently (β_solar × 0 ≈ 0). If joint RMSE is *worse* on dark segments, something is
wrong with the fitting.

**Comparison:** Report mean RMSE for dark-only segments under both models (apples-to-apples)
and joint-only RMSE on daytime segments (no comparison possible for dark-only model there).

### Secondary: Leave-one-out cross-validation RMSE

**What:** For each sensor, fit τ from N-1 segments, predict held-out segment temps. Report
mean RMSE across folds.

**Why:** Tests generalization — does the τ from other segments predict a new segment well?
Joint model should generalize better because τ is less biased.

### Diagnostic: β_solar plausibility

**What:** Per-segment β_solar values, aggregated per sensor (median + IQR).

**Checks:**
- Positive for all sensors (sun heats rooms)
- Higher for solar-exposed rooms (piano > bathroom)
- Near zero for dark segments (segments where S(t) ≈ 0 should give poorly-determined
  β_solar — check that fit didn't invent one)
- Inter-segment CV < 2 (reasonably consistent)

### Diagnostic: β_solar vs Stage 2 consistency

**What:** Compare median per-segment β_solar with Stage 2's `solar_elevation_gain`.

**Why:** Both measure "°F/hr per unit of solar forcing" from different data. Large
disagreement (>3×) suggests identifiability problems in one or both stages.

### Diagnostic: τ–β_solar correlation

**What:** From `curve_fit` covariance matrix, compute correlation coefficient between τ and
β_solar estimates per segment.

**Why:** High correlation (|r| > 0.8) means the fit can't distinguish "slow cooling" from
"solar heating" — identifiability failure. This is the main risk of the joint approach.

**Mitigation:** Segments that span a solar transition (sunset included) have good
identifiability because S changes within the segment. Pure-daytime segments with constant
high S may have poor identifiability — flag these.

### Diagnostic: Tau stability (joint vs current)

**What:** Compare τ estimates between current and joint approaches per sensor.

**Why:** Large shifts (>50%) in a consistent direction suggest the current approach has
systematic bias. Small/random differences suggest solar bias isn't a major factor for that
sensor.

## Source Code Changes

### `src/weatherstat/sysid.py`

#### New: `_fit_tau_solar_curve()`

```python
def _fit_tau_solar_curve(
    t_hours: np.ndarray,
    temps: np.ndarray,
    t_outdoor: float,
    solar_forcing: np.ndarray,
) -> tuple[float, float, float, np.ndarray] | None:
    """Fit tau and beta_solar jointly from a segment with solar forcing.

    Returns (tau, beta_solar, rmse, predicted_temps) or None on failure.
    Uses exact piecewise-constant ODE solution (no Euler error).
    """
```

Parameters:
- `t_hours`: relative time in hours (array)
- `temps`: observed temperatures (array)
- `t_outdoor`: mean outdoor temp for segment (scalar)
- `solar_forcing`: `_solar_elev` values for segment (array, same length)

Bounds: τ ∈ [1, 500], β_solar ∈ [0, 50]. Initial guess: τ=40, β_solar=2.

Return includes `pcov` correlation for the τ–β_solar diagnostic.

Actually, refine: return a small result dataclass or named tuple for clarity:

```python
@dataclass(frozen=True)
class SegmentFitResult:
    tau: float
    beta_solar: float
    rmse: float
    tau_beta_correlation: float  # from pcov
    n_points: int
    has_solar_variation: bool  # S varies meaningfully within segment
```

#### Modified: `_find_uncontrolled_segments()`

Add `require_dark: bool = True` parameter:
- `require_dark=True`: current behavior (solar elev < 5°)
- `require_dark=False`: no solar filter, all HVAC-off segments qualify

Everything else unchanged (stable windows, contiguous, min length, valid temps).

#### Modified: `_fit_tau()`

Run both approaches:
1. `_find_uncontrolled_segments(df, effectors, sensor)` — dark-only (current)
2. `_find_uncontrolled_segments(df, effectors, sensor, require_dark=False)` — all HVAC-off

For approach 1: fit with `_fit_tau_curve()` (existing).
For approach 2: fit with `_fit_tau_solar_curve()` (new).

Collect per-segment results for both. Compute weighted median τ from each.
Print comparison table.

Return value: unchanged (`list[FittedTau]`). Initially return the dark-only results
(conservative). Switch to joint results once metrics confirm improvement.

#### New: `_print_tau_comparison()`

Print per-sensor comparison table:
```
Tau fitting comparison (dark-only vs joint tau+solar):
  Sensor                  │ Dark segs │ Joint segs │ τ_dark │ τ_joint │ CV_dark │ CV_joint │ β_solar_med
  thermostat_upstairs     │    22     │     41     │  42.7  │   45.1  │  0.35   │   0.18   │    1.8
  kitchen                 │    22     │     38     │  62.8  │   58.3  │  0.52   │   0.22   │    4.2
  piano                   │    22     │     43     │  25.5  │   28.1  │  0.41   │   0.19   │    6.1
  ...
```

Also print per-sensor diagnostics:
- β_solar range and consistency
- τ–β_solar correlation (flagged if |r| > 0.8)
- LOO RMSE comparison

### No other files change

This is a prototype — comparison output only. The rest of the pipeline continues to use
the dark-only τ values. Once we verify the joint approach is better, a follow-up commit
switches `_fit_tau()` to use joint results and removes the dark-only code path.

## Implementation Order

1. Add `SegmentFitResult` dataclass
2. Add `_fit_tau_solar_curve()` function
3. Add `require_dark` parameter to `_find_uncontrolled_segments()`
4. Modify `_fit_tau()` to run both approaches, collect comparison data
5. Add `_print_tau_comparison()` for the comparison table
6. Run `just sysid` and assess the metrics

## Verification

1. `just test` — existing tests pass (no behavioral changes)
2. `just lint` — clean
3. `just sysid` — outputs comparison table with metrics
4. Manual assessment of metrics against thresholds:
   - Joint CV < dark-only CV for ≥6/9 sensors?
   - β_solar positive and plausible (higher for solar rooms)?
   - τ–β_solar correlation < 0.8 for most segments?
   - LOO RMSE equal or better?
5. If metrics pass: follow-up commit switches to joint results
6. If metrics fail: investigate which sensors/segments have identifiability problems

## Decision Criteria

**Switch to joint approach if:**
- Tau CV improves for ≥6/9 sensors
- β_solar is positive for ≥7/9 sensors
- τ–β_solar correlation < 0.8 for ≥75% of segments
- LOO RMSE is equal or better for ≥6/9 sensors

**Keep dark-only if:**
- τ–β_solar correlation > 0.8 for many segments (identifiability failure)
- β_solar is negative or erratic (model misspecification)
- Joint τ has *higher* CV (worse consistency)

**Investigate further if:**
- Mixed results (some sensors better, some worse)
- Identifiability is good for nighttime segments but poor for daytime
  → could use joint fit but weight nighttime segments more heavily
