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
- No environment tau betas learned (empty)
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
high condition numbers) to selectively standardized ridge (λ = 0.01 × n).

Three approaches were evaluated; selective standardization won:

| Metric             | OLS   | Naive ridge | Full std | **Selective** |
|--------------------|-------|-------------|----------|---------------|
| Gains passing filter | 6   | 6           | 3        | **3**         |
| bedroom→mini_split | –     | t=1.52      | –        | **t=1.51**    |
| bedroom_agg→m.s.   | t=5.69| t=7.54      | t=5.73   | **t=7.05**    |
| piano→mini_split_lr| t=1.85| t=1.94      | t=1.79   | **t=1.83**    |
| Significant solar  | 0     | 0           | 9        | **9**         |
| MRT weights        | empty | empty       | populated| **populated** |
| Confounded gains   | wild  | shrunk      | exploded | **shrunk**    |

**Selective standardization**: only solar (`_solar_*`) and window
(`_win_*`, `_winx_*`) features are divided by their std before
regression. Effector and weather features stay in raw scale. This
protects sparse low-variance features (solar indicators) from
over-penalization while maintaining full regularization on confounded
effector gains. Coefficients are transformed back to original units
after the solve.

### Known issue: living_room_climate solar spikes

`living_room_climate_temp` shows ±17°F/hr solar gains (h=8, h=10).
Root cause: only 3 tau segments, tau_base=500h cap. The solar
coefficients are fitting noise in sparse data. Solar gains are NOT
filtered by the simulator's magnitude cap (only effector gains are).
Guard rail: add a solar gain magnitude cap in the simulator, or
filter at sysid time for sensors with < N tau segments.

### Tuning notes

- λ = 0.01 × n_rows. For n ≈ 10000, λ = 100.
- High collinearity (cond > 1e6) uses 10× λ.
- Standardization applies only to solar/window features.
- Pre-ridge backups saved as `.pre-ridge`, `.naive-ridge`, `.std-ridge`
  in `~/.weatherstat/` for A/B comparison.

---

## Derivative Noise and Smoothed Differentiation (2026-03-24)

### The problem

Sysid fits gains by regressing Newton residuals (`dT/dt_observed - dT/dt_newton`) on effector activity. The `dT/dt_observed` term is computed numerically from 5-minute temperature snapshots. The naive approach — central differences, `(T[i+1] - T[i-1]) / (2 × 5min)` — amplifies sensor noise catastrophically.

A temperature sensor with ±0.1°F jitter produces ±1.2°F/hr of derivative noise. With real sensor behavior (air currents, HVAC cycling, measurement quantization), the 5-minute dT/dt residual has a standard deviation of **~10.5°F/hr**. A thermostat gain of 0.3°F/hr is a 35:1 noise-to-signal ratio.

The lag-2 autocorrelation of the 5-minute residual is **-0.615** — a hallmark of central-difference noise amplification. The reading `T[i+1]` appears with positive sign at step `i` and negative sign at step `i+2`, creating strong anti-correlation between adjacent derivative estimates.

### Diagnosis

Thermostat gains had reasonable magnitudes (0.2–0.6°F/hr) but t-statistics below 1.5 for almost all sensors, causing the simulator to prune them. The control loop treated thermostats as having zero effect. The univariate correlation between thermostat activity and the Newton residual was ~0.01 — a real signal buried under noise.

Testing with longer differentiation windows revealed the signal clearly:

| Window | Residual std | Thermostat t-stat | Gain |
|--------|-------------|-------------------|------|
| 5 min | 10.5°F/hr | 1.05 | 0.25 |
| 15 min | 2.0°F/hr | **6.28** | 0.29 |
| 30 min | 0.8°F/hr | **18.3** | 0.33 |
| 60 min | 0.5°F/hr | **34.5** | 0.33 |

The gain is stable across all windows (~0.3°F/hr), confirming it's a real physical effect. Only the noise changes.

### The fix

Smooth temperature with a centered rolling mean (default half-window: 3 steps = 15 min, total kernel 35 min), then compute central differences on the smoothed series. This reduces noise ~5× while preserving signals on the timescale of effector lags (≥15 min for fast effectors like blowers, ≥45 min for hydronic).

After the fix: thermostat gains go from 1 surviving (out of 32) to **25 surviving**, all with positive signs and physically reasonable magnitudes. The control loop now correctly turns on thermostats when heating is needed.

### The fundamental principle

Numerical differentiation amplifies high-frequency noise by a factor proportional to `1/Δt`. The signal-to-noise ratio of the derivative depends on the relationship between three timescales:

1. **Signal timescale** (τ_signal): how fast the effect you're measuring changes. For thermostat heating, this is the thermal lag — 15–90 minutes.
2. **Sampling interval** (Δt): the gap between observations. Here, 5 minutes.
3. **Noise amplitude** (σ_noise): the sensor measurement jitter. Here, ~±0.1°F.

The derivative SNR scales as:

```
SNR_derivative ∝ (signal_amplitude × Δt_diff) / σ_noise
```

where `Δt_diff` is the differentiation window. When `Δt_diff` is much shorter than the signal timescale, you're wasting SNR on temporal resolution you don't need. The optimal differentiation window is **as long as you can afford** without blurring the signal — roughly the shortest effector lag you need to resolve.

In practice: if your fastest effector has a 15-minute lag and your data is sampled every 5 minutes, differentiating over 5 minutes gives 3× more temporal resolution than you need while amplifying noise 3×. A 15-minute window matches the signal timescale, maximizing SNR without sacrificing useful temporal detail.

### Secondary finding: weather control absorption

Even after noise reduction, the weather control features (ΔT², wind×ΔT, dT_outdoor/dt) partially absorb the thermostat signal — the thermostat t-stat drops from 6.28 to ~5.3 when weather controls are included. This is expected: thermostats run more when it's cold, and the weather features capture cold-weather nonlinearities. The signal is strong enough now that this absorption doesn't push gains below significance, but it's a factor to monitor.

### Secondary finding: mode-direction sign filter

During investigation, discovered that `thermostat_downstairs → living_room_climate_temp` had a negative gain (−0.087°F/hr, t=−1.69) that passed the t-stat filter. A negative gain from a heating-only effector is physical nonsense — it says "turning on the heat cools the living room." Added a mode-direction sign filter in `simulator.load_sim_params()`: heating-only effectors (`supported_modes: [heat]`) cannot have negative gains, and cooling-only effectors cannot have positive gains.

### Why this only affects Stage 2

Stage 1 (tau fitting) fits `T(t) = T_out + (T_0 - T_out) × exp(-t/τ)` via `scipy.curve_fit` on **raw temperature curves**, not derivatives. Fitting a smooth model to noisy data is integration-like: noise averages out. Stage 2 differentiates first, then regresses — differentiation amplifies noise. So the two stages have opposite noise sensitivity, and only Stage 2 needed the smoothed derivative.

### Autocorrelation from smoothing

The 35-minute smoothing kernel means adjacent derivative estimates share data, introducing positive autocorrelation. OLS standard errors assume independence, so t-statistics are slightly overstated — the effective sample size is roughly N/3 rather than N. With 11,800 rows this still leaves ~4,000 effective observations, more than enough for a ~50-feature regression. The t-stats of 5–10 would remain well above significance even after correction.

---

## Diagnostic Playbook

### "Why is sensor X predicted at Y°F?"

1. Check `thermal_params.json` for the sensor's gains, tau, environment tau betas
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
