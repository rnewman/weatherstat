# Modeling Strategy

## The Problem

This is a **control system**, not a forecasting system. The question is:

> **"What should I set all HVAC devices to right now to maximize comfort
> over the next few hours?"**

We re-ask every 15 minutes (receding horizon). This means:

- **Short-horizon accuracy matters most.** T+1h and T+2h drive decisions.
- **The model's job is comparative, not absolute.** We need "setting A keeps it
  more comfortable than setting B" — not necessarily "the temperature will be
  72.3°F."
- **Counterfactual reasoning is essential.** The model must answer "what if I
  turn off the heat?" reliably, even when the training data mostly shows heat
  ON during cold weather.

## Current Architecture (Mar 2026)

### Prediction: LightGBM + physics features

8 rooms x 5 horizons = 40 LightGBM models trained on collector snapshots.
Physics is encoded as features (Newton cooling predictions, dT/dt, heating
power interactions, retrospective HVAC runtime/duty cycle), not as a separate
model layer.

**Strengths:** Good short-horizon accuracy (~3°F RMSE). Captures solar gain,
inter-room coupling, and HVAC effects implicitly. Fast enough for real-time
sweep (30ms for 180 scenarios).

**Weaknesses:** Observational confounding limits counterfactual reliability.
The model learned "heat is ON when cold" (correlation), not "heat ON causes
warming" (causation). Guardrails (Newton floor/ceiling, cold-room overrides)
compensate but indicate the wrong architecture for causal questions.

### Control: sweep + score

The controller enumerates candidate HVAC configurations, predicts outcomes for
each using the ML model, and scores against comfort schedules + energy costs.
Physical constraints (hold times, setpoint clamps, blower-zone coupling,
cold-room override) are enforced post-sweep.

This framework is sound and carries forward regardless of the prediction engine.

### System identification: `sysid.py`

Two-stage fitting from all collector data:
1. **Tau fitting** from nighttime HVAC-off periods (weighted median across
   12-14 segments per sensor, sealed and ventilated separately)
2. **Effector gain regression** on Newton residuals with lagged activity bins
   and hour-of-day solar indicators

Produces the full effector x sensor coupling matrix, delay estimates, and solar
gain profiles. This is the parameter set for a future grey-box forward simulator.

## Next Step: Grey-Box Forward Simulator

Replace ML predictions in the sweep with physics-based forward simulation using
sysid parameters. See `docs/FUTURE.md` for details.

The key payoff: reliable counterfactual reasoning without observational
confounding. "What if heat is off?" becomes a physics question with known
parameters, not a statistical extrapolation.

## Design Principles

- **Physics first, ML second.** Use physics for what we understand (envelope
  loss, heating input); ML for what we can't easily model (solar gain patterns,
  occupancy effects).
- **Modular capabilities.** Forecast, HVAC features, Newton integration, ML,
  and sysid are independent modules.
- **No residual targets.** ML predicts absolute temps with physics as features.
  Training on Newton residuals risks overfitting to Newton's errors.
- **Configuration-driven.** Adding a sensor or device = YAML edit + rerun.

## What We Tried and Learned

See `docs/ARCHITECTURE.md` (Retrospective section) and `docs/EXPERIMENTS.md`
for the full history, including the hybrid_physics blending experiment (failed:
uncalibrated physics made predictions worse) and physics_v1 features (succeeded:
rate-of-change features improved 12h predictions for downstairs rooms).
