# Future Work

## Architecture: What the ML Model Should Do

The future temperature of a room is determined by:

```
T_future = T_passive + ΔT_hvac + ΔT_solar + ΔT_interroom + ΔT_ventilation
```

**Newton's law** handles `T_passive` well — exponential decay toward outdoor temp
with per-room τ. This is physics we can compute directly.

**The ML model's job** is learning the *corrections* that physics can't easily compute:
1. **HVAC transfer function**: boiler ON → °F/hour gained in each room, with what lag?
   Hydronic slab thermal mass makes this a multi-hour delayed response.
2. **Solar gain per room**: south-facing windows (piano) vs north (bathroom). Depends
   on solar elevation × cloud cover × window area. The structure is physics but the
   coefficients are house-specific.
3. **Inter-room coupling**: warm rooms heat adjacent cool rooms via stairway, duct,
   and convection paths.
4. **Ventilation sensitivity**: wind speed × window state → additional heat loss.

**The ML model should NOT try to learn**:
- Passive thermal decay (Newton does this)
- That "it's warm at 3pm" (time-of-day ≠ causation)
- Future outdoor temperature trends (forecast provides this)

### Design principles

- **Modular capabilities**: forecast, retrospective HVAC features, Newton integration,
  and ML prediction are independent modules. Each can be improved or replaced without
  touching the others.
- **Physics as guardrails, not targets**: Newton provides sweep-time floor/ceiling on
  all-off predictions. The ML model predicts absolute temperatures with physics features
  as inputs. We do NOT train on residuals (risk of overfitting to Newton's errors and
  compounding).
- **Keep the model boundary visible**: the sweep, cost function, and physics layers
  should be inspectable. The "squishy middle" (LightGBM) handles what it's good at;
  everything else is explicit.

---

## Modular Capabilities Roadmap

### 1. Weather Forecast Integration (next)

**The biggest single improvement available.** Currently Newton features and ML predictions
use current outdoor temp for all horizons — wrong by 10–15°F at 6h in shoulder seasons.

**Data source**: HA `weather.get_forecasts` service on `weather.forecast_home` (met.no).
Returns hourly forecasts with: temperature, condition, wind speed, precipitation.

**Components**:
- **Forecast fetcher**: Python module to call HA forecast service via REST API.
  Returns structured hourly forecast for next 12–24h.
- **Piecewise Newton integration**: Chain hourly segments using forecast outdoor temps:
  ```
  T(t₁) = T_out_h0 + (T₀ - T_out_h0) × exp(-1/τ)
  T(t₂) = T_out_h1 + (T(t₁) - T_out_h1) × exp(-1/τ)
  ```
  Exact for piecewise-constant outdoor temps (which hourly forecasts provide).
- **Forecast ML features**: at each horizon, include forecast outdoor temp, cloud cover
  (proxy for solar irradiance), and wind speed. Combined with solar elevation at the
  future timestamp → approximate solar irradiance estimate.
- **Collector storage**: store forecast snapshots alongside sensor snapshots so training
  data includes what was predicted at each point in time.

### 2. Retrospective HVAC Features (next, parallel with forecast)

**Current gap**: "boiler ON" doesn't capture how much heat is stored in the slab.
The hydronic thermal mass means the boiler running for 3 hours has deposited far more
energy than 5 minutes of runtime. Current features only see the instantaneous state.

**Features to add** (computed from recent history already fetched):
- `heating_minutes_{1h,2h,4h}`: cumulative 5-min intervals with heating ON
- `heating_duty_cycle_{1h,2h}`: fraction of time heating was on
- `time_since_heat_start`: minutes since last off→on transition
- `time_since_heat_stop`: minutes since last on→off transition
- Same for each mini-split and blower
- `navien_runtime_{1h,2h,4h}`: cumulative Navien Space Heating minutes

These capture the slab's thermal charge state. Combined with the HVAC transfer function
the model learns, this enables understanding "the slab is fully charged and will radiate
for 2 more hours" vs "the boiler just kicked on."

### 3. Solar Irradiance Estimation

**Approximate solar irradiance** from existing + forecast data:
- `solar_elevation` (already computed per-row)
- `cloud_cover` (from forecast condition: sunny=0, partly_cloudy=0.5, cloudy=1.0)
- `irradiance_estimate = max(0, sin(elevation)) × (1 - cloud_factor) × PEAK_IRRADIANCE`

This is crude but captures the key dynamics: south-facing rooms warm in winter
afternoon sun, not on cloudy days. Per-room coefficients (window orientation, area)
are for the ML model to learn.

---

## Sweep Scalability

Batch prediction (approach #1) is done — 148× speedup. Remaining approaches are
relevant when the device count grows.

### Current cost (post-batch)

| Component | Combos | Batch calls | Wall time |
|-----------|--------|-------------|-----------|
| HVAC sweep | 180 | 40 | ~30ms |
| Window advisory | 128 | 40 | ~28ms |

### Future optimization (in priority order)

**2. Window decomposition** — O(N) vs O(2^N). Independent toggle per window,
verification pass on combined result. Needed at 10+ windows.

**3. Greedy coordinate descent** — O(Σ levels) vs O(∏ levels). Iterative
single-device optimization. Needed at 6+ blowers.

**4. Marginal screening** — Fast linear approximation, full eval on top-K.
Best of both worlds for large action spaces.

**5. Spatial decomposition** — Only re-predict rooms affected by each device
change. Composes with all above approaches.

---

## Other Future Work

### Pre-Heating Logic
Uses forecast + HVAC response curves to start heating before outdoor temp drops.
Critical for hydronic floor heat with 2–4 hour thermal lag. Requires forecast
integration (#1) and retrospective features (#2) to estimate slab charge state.

### MPC Trajectory Planning
Optimize HVAC *sequences* (heat 2h then coast 4h) instead of single settings.
Requires fast-evaluating thermal model. The physics + modular ML approach provides
the foundation — evaluate Newton + ML correction for each step in the trajectory.

### Virtual Thermostats
Per-room HA climate entities for user-adjustable comfort targets from the dashboard.

### HA Packaging
See PLAN.md for integration levels (add-on, integration, hybrid).
