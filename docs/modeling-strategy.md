# Modeling Strategy: From Regression to Control

## The Problem Reframed

This is a **control system**, not a forecasting system. The question is never "what will the temperature be in 12 hours?" — it's:

> **"What should I set all HVAC devices to right now in order to maximize comfort over the next few hours?"**

We re-ask that question periodically (every 5-30 minutes), creating a **receding horizon** control loop. This means:

- **Short-horizon accuracy matters most.** T+1h and T+2h predictions drive decisions. T+4h is useful for anticipating solar gain or weather shifts. T+6h+ is nice-to-have but rarely decisive.
- **We don't need perfect long-term forecasts.** By the time T+4h arrives, we'll have re-evaluated 8-48 times with fresh data.
- **The model's job is comparative, not absolute.** We don't need "the temperature will be 72.3°F" — we need "setting the thermostat to 71 keeps it more comfortable than 73 does, given the afternoon sun coming."

## Current State (Feb 2026)

### What works
- **Thermal dynamics baseline** (5 months hourly data): 0.44°F RMSE at 1h, excellent for passive prediction
- **Full model** (10 days 5-min data): 0.59°F RMSE at 1h — competitive, and catches up at longer horizons
- **Solar elevation** emerges as top feature in the full model, confirming the solar gain hypothesis
- **Setpoint counterfactuals** show the model learned the binary heating threshold correctly (heating on vs off)

### What doesn't work yet
- **Extrapolation beyond training distribution**: thermostats have been set to 71-73°F almost exclusively. The model can't predict what happens at 76°F because it's never seen it.
- **Long-horizon convergence**: the model doesn't understand that temperature converges to the setpoint given enough time, because it's a regression model, not a physics simulation.
- **HVAC feature importance is low** in the 10-day model — not enough data yet for the model to reliably distinguish HVAC effects from thermal inertia.

## Paths Forward

### 1. More Data (Ongoing, Passive)

**Effort:** None (just keep the collector running)
**Timeline:** Weeks to months
**Impact:** High

The 10-day full-feature dataset will grow daily. As it does:
- HVAC features will gain importance as the model sees more heating/cooling cycles
- Seasonal variation (spring sun angles, warmer outdoor temps) will broaden the model's range
- More varied setpoint data will naturally occur (manual overrides, schedule changes)

**Action items:**
- Start the collector (`just collect`) and keep it running
- Retrain weekly (`just retrain`)
- Occasionally set thermostats to unusual values (e.g., 68°F or 75°F for a few hours) to broaden the training distribution — this is essentially "exploration" in the RL sense

### 2. Hybrid Physics + ML Model

**Effort:** Medium
**Timeline:** Days to implement
**Impact:** High for long-horizon, moderate for short

The key insight: short-horizon prediction is dominated by thermal inertia (the ML model handles this well), but long-horizon prediction needs physics (exponential decay toward equilibrium).

**Approach:**
- For T+1h to T+2h: use the ML model directly (it's excellent here)
- For T+4h+: blend ML prediction with a physics prior

The physics model for a heated zone is approximately:

```
T(t) = T_eq - (T_eq - T_now) * exp(-t / tau)
```

Where:
- `T_eq` is the equilibrium temperature (≈ setpoint when heating, ≈ some function of outdoor temp when not)
- `tau` is the thermal time constant (estimated from observed heating/cooling rates)
- `t` is time into the future

We can estimate `tau` from data: when the thermostat turns on, how fast does the temperature rise? When it turns off, how fast does it fall? Typical values for a well-insulated house: tau ≈ 4-8 hours.

**Blending:** weight the ML prediction and physics prediction by horizon:
```
weight_ml = exp(-horizon_hours / crossover_hours)
prediction = weight_ml * ml_pred + (1 - weight_ml) * physics_pred
```

With `crossover_hours ≈ 3-4`, this gives ML ~75% weight at 1h, ~50% at 3h, ~25% at 5h, ~5% at 12h.

### 3. Physics-Informed Features

**Effort:** Low-Medium
**Timeline:** Days
**Impact:** Moderate

Instead of a separate physics model, encode physical relationships as features:
- `heating_rate_1h`: observed dT/dt over the last hour (°F/hr). Captures current thermal momentum.
- `estimated_time_to_setpoint`: `(setpoint - current) / heating_rate`. How long until we hit the target at current rate? Infinite if cooling.
- `thermal_deficit`: integral of `(setpoint - actual)` over recent history. Captures accumulated under/over-heating.
- `solar_gain_rate`: dT/dt attributable to solar (e.g., temp change minus expected HVAC contribution)

These features would help the ML model learn control-relevant dynamics without needing a full physics simulation.

### 4. Reinforcement Learning / Model Predictive Control

**Effort:** High
**Timeline:** Months
**Impact:** Very high (this is the end goal)

The natural evolution: use the ML model as a **world model** inside an optimization loop.

**Model Predictive Control (MPC):**
1. At each control step, enumerate candidate HVAC configurations (setpoint values, blower settings, mini split modes)
2. For each configuration, predict temperatures over the next N hours using the ML model
3. Score each trajectory against a comfort+efficiency objective
4. Execute the best configuration
5. Repeat at next control step

**Objective function** (to be tuned):
```
cost = w_comfort * comfort_penalty + w_energy * energy_estimate
```

Where:
- `comfort_penalty` = sum of deviations from desired range (e.g., penalty for being below 70°F or above 74°F)
- `energy_estimate` = some proxy for energy use (e.g., minutes of heating, blower runtime)

This doesn't require RL training — it's optimization over the ML model's predictions. The ML model is the learned dynamics, and MPC finds the best action sequence.

**Prerequisites:**
- Reliable ML model (more data needed)
- Confidence in model predictions (need uncertainty estimates)
- Safe fallback (if model is wrong, don't freeze the house)

## Recommended Sequence

| Phase | What | Why | When |
|-------|------|-----|------|
| **Now** | Start collector, retrain weekly | Data is everything; every day matters | Immediately |
| **Week 1-2** | Physics-informed features | Low effort, improves model without new architecture | Soon |
| **Week 2-4** | Hybrid physics+ML for long horizons | Fixes the extrapolation problem for setpoint exploration | After baseline data grows |
| **Month 2-3** | Simple MPC loop | First closed-loop control, conservative settings | After model is trusted |
| **Month 3+** | Tune MPC objective, add efficiency | Full system with comfort/efficiency tradeoff | Ongoing |

## Control Loop Architecture (Target)

```
Every 5-30 minutes:
  1. Fetch current state from HA
  2. For each candidate HVAC configuration:
     a. Override setpoints/blower/split features
     b. Predict temperatures at T+1h, T+2h, T+4h
     c. Score the trajectory (comfort + efficiency)
  3. Pick the best configuration
  4. If it differs from current: apply changes via HA
  5. Log prediction vs reality for model improvement
```

The ML model serves as the "what if" engine. The control logic is just argmin over predicted costs. Re-evaluating frequently (receding horizon) means we self-correct as predictions drift.

## Key Design Decisions Still Open

1. **Control granularity**: How often to re-evaluate? 5 min is probably too aggressive (HVAC can't respond that fast). 15-30 min is more realistic for hydronic heat.
2. **Exploration vs exploitation**: Should the controller occasionally try suboptimal settings to generate diverse training data? Classical explore-exploit tradeoff.
3. **Comfort definition**: Is "stay between 70-74°F" the right objective? Or "minimize deviation from 72°F"? Or asymmetric (being cold is worse than being warm)?
4. **Constraint handling**: Never let any zone drop below X°F. Never heat when outdoor > Y°F. Respect quiet hours (don't cycle equipment at 3am).
5. **Multi-zone coordination**: Upstairs heat is ceiling heat for downstairs. The controller should know this — heating upstairs is partially heating downstairs too.
