# Weatherstat Roadmap

## Done

### Stage 1: Pipeline & Data Infrastructure
- HA WebSocket + REST extraction (hourly statistics, 5-min raw history)
- Collector: 5-min snapshots to SQLite, durable runner with health monitoring
- LightGBM training: baseline (5mo hourly) and full (5-min multi-source)
- Full-model training merges historical Parquet + collector SQLite automatically
- Feature engineering: time, solar position, weather, HVAC state, lag/rolling
- Multi-horizon prediction: T+1h through T+12h per zone
- Evaluation framework comparing baseline vs full model

### Stage 2: Control Loop
- Setpoint sweep over 81 upstairs/downstairs pairs
- Comfort schedules (per-room, time-of-day, asymmetric penalties)
- Safety rails: setpoint clamps, hold time, staleness check, sanity limits
- Dry-run mode (default) + live execution via HA services
- Counterfactual analysis ("what if setpoints were X?")

## In Progress

### Data Accumulation
- Collector running continuously (started Feb 2026)
- All current data is winter — seasonal coverage needs months
- HVAC feature importance should increase as dataset grows
- Retrain weekly to incorporate new collector data

### Dry-Run Control Validation
- Monitoring control loop decisions before going live
- Verifying setpoint recommendations match physical intuition

## Near-Term

### Per-Room Prediction
- Extend models to predict office and bedroom temps (currently upstairs/downstairs only)
- Requires enough blower/mini-split operational data to learn their effects

### Blower Control
- Add blower fan speed to the control sweep (currently thermostats only)
- Blower family room has ~6 weeks of data; blower office only ~2 weeks
- Need per-room models before blowers can be usefully optimized

### Physics-Informed Features
- Heating/cooling rate (dT/dt), thermal deficit, estimated time-to-setpoint
- Helps the model learn control-relevant dynamics without a physics simulation

## Medium-Term

### Mini Split Control
- Include mini split modes/targets in the control sweep
- Needs more varied mini split operational data (current data is mostly static settings)

### Hybrid Physics + ML
- Blend ML predictions (good at short horizons) with exponential-decay physics model (long horizons)
- Fixes extrapolation issues when setpoints move outside training distribution
- See `docs/modeling-strategy.md` for the math

### Automated Retraining
- Cron job or launchd plist to retrain weekly
- Track model metrics over time to detect drift

## Long-Term

### Advisory Mode
- Surface control recommendations in HA (notifications, dashboard cards)
- "The model suggests lowering the thermostat — afternoon sun will overshoot"

### Full MPC
- Model Predictive Control: optimize over multi-step HVAC trajectories
- Exploration vs exploitation strategy for broadening training distribution
- Multi-zone coordination (upstairs heat is ceiling heat for downstairs)

### HA Integration
- Package as a Home Assistant add-on or custom integration
- Replace WebSocket/file-based communication with HA internal API
