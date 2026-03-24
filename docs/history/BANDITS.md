# Contextual Bandits for HVAC Optimization

Analysis of applying Vowpal Wabbit's CATS (Continuous Actions with Tree-based
Smoothing) or similar contextual bandit approaches to weatherstat.

## The Mapping

| CATS Concept | Weatherstat Equivalent |
|---|---|
| **Context** | Current temps, weather, time, thermal mass state, occupancy |
| **Continuous Action** | Boiler setpoint (100–160°F), fan speeds (0–100%), mini-split setpoints |
| **Cost** | Discomfort (deviation from comfort schedule) + energy usage |
| **Bandwidth** | Smoothing around promising setpoints |

## Two Paths

### Path 1: Direct Policy (CATS replaces predict-then-sweep)

The current architecture is **predict-then-optimize**: train 40 LightGBM
models to forecast per-room temperatures, sweep ~180 HVAC combos, score by
comfort+energy, pick the best.

A direct bandit policy would collapse this into **context → action**, learning
which HVAC settings minimize a combined comfort+energy cost signal.

**Advantages:**
- The action space really is continuous — boiler temp, fan speed, mini-split
  setpoints are all continuous knobs we currently discretize for the sweep.
- Joint actions scale better — the current sweep is combinatorial
  (boiler × 8 zones × fan speeds). A continuous policy over the joint action
  vector avoids exponential blowup.
- Adapts to changing conditions (seasonal shifts, insulation upgrades)
  without retraining on stale labels.
- No compounding prediction errors — learns directly from outcomes.

**Challenges:**
- **Sample efficiency.** One decision every 15 minutes → 96/day. CATS
  tutorials converge over thousands of iterations. Even with perfect feedback,
  convergence takes weeks — and we only have winter data. The current ML
  approach leverages every 5-minute snapshot as a supervised training example.
- **Delayed, noisy reward.** Hydronic floor heat has 30–90 minute thermal lag.
  The cost of an action at 6am isn't observable until 7:30am. CATS assumes
  immediate cost feedback. Needs temporal credit assignment on top.
- **Exploration cost is real discomfort.** Epsilon-greedy exploration means
  sometimes trying suboptimal settings. "Explore a boiler setpoint of 105°F"
  means cold floors for two hours. Bandwidth helps but the tension remains.
- **Loss of interpretability.** Current system produces "living room will be
  68.2°F in 2 hours" — useful for debugging and trust. A bandit is a black
  box: "set boiler to 142°F" with no explanation of predicted outcome.

### Path 2: Bandit Meta-Optimizer (layered on top of improved ML)

Keep the ML temperature models for prediction and interpretability. Use CATS
to learn the **comfort-cost tradeoff parameters** that the sweep's scoring
function currently hard-codes:

- Comfort weight vs. energy weight
- Hysteresis thresholds
- Cold-room override trigger point (currently 1°F below comfort min)
- Anticipation horizon weighting (how much to weight 6h vs 1h predictions)
- Newton floor/ceiling aggressiveness
- Per-room comfort asymmetry (how much worse is too-cold vs too-warm?)

The bandit's "action" becomes these **policy parameters** (continuous, ~6–10
dimensions), not raw HVAC setpoints. The sweep still runs underneath with ML
predictions, but the scoring function adapts over time.

**Advantages:**
- Much smaller action space → faster convergence (days, not weeks).
- Exploration in parameter space is far less risky than in setpoint space —
  a slightly different comfort weight won't freeze anyone.
- Preserves interpretability: ML still predicts temperatures, sweep still
  produces explanations.
- Degrades gracefully: worst case, the bandit picks parameters close to
  current hand-tuned values.

**Prerequisite:** The underlying ML predictor needs to be accurate enough that
the sweep's decisions are meaningful. If the predictor is off by 3°F, no
amount of parameter tuning will fix it. This path requires solving the
prediction accuracy problem first.

## Experimentation Framework Fit

### What Works Today

The existing experiment infrastructure can accommodate CATS exploration:

1. **Git worktree isolation.** `just worktree bandits_v1` creates an isolated
   branch and working directory. All code changes (new model type, training
   loop, VW integration) happen there without touching production.

2. **Shared data.** Worktrees symlink `data/` from the main repo. The bandit
   experiment trains and evaluates on the same collector snapshots and
   historical data as production.

3. **Separate model directory.** Experiment models write to
   `data/models/{experiment_name}/`, never touching production models.
   The production control loop continues running LightGBM.

4. **Comparison infrastructure.** `just experiment-compare bandits_v1` can
   evaluate against production — but needs extension for bandit-specific
   metrics (see below).

5. **Backtest.** The overnight cooling fixture
   (`tests/fixtures/overnight_cooling_20260214.parquet`) tests passive
   prediction. A bandit experiment would need its own evaluation fixtures
   focused on decision quality, not prediction accuracy.

6. **Metrics tracking.** `data/metrics/` already logs per-run RMSE/MAE with
   timestamps and git hashes. Bandit experiments would add regret/reward
   metrics in the same format.

### What Needs Extension

**For Path 1 (direct policy):**

- **Model interface abstraction.** Training and inference currently call
  `lgb.LGBMRegressor` and `lgb.Booster` directly. To plug in VW:
  - Extract a `TemperatureModel` protocol (fit/predict/save/load)
  - LightGBM becomes one implementation; VW/bandit becomes another
  - Control sweep's `_batch_predict()` already just calls `model.predict(X)`,
    so inference is nearly model-agnostic
  - Training loop needs refactoring: VW is online/streaming, not batch

- **Online training loop.** VW learns incrementally from each observation.
  The current batch pipeline (load all data → train → save) doesn't fit.
  Need either:
  - A replay-from-collector mode that streams SQLite rows through VW
  - An online hook in the control loop that feeds outcomes back to VW

- **Reward computation.** The control loop currently doesn't compute a reward
  signal from outcomes. Need a feedback loop:
  1. At time T, record the action taken and context
  2. At time T+Δ (lag), observe the resulting temperatures
  3. Compute cost (comfort deviation + energy)
  4. Feed (context, action, cost) to VW

- **Experiment comparison.** Current comparison is RMSE of temperature
  predictions. For a bandit, comparison should be:
  - Cumulative regret vs. oracle (requires knowing optimal action — hard)
  - Average comfort violation and energy usage over evaluation period
  - Decision agreement rate with production (how often does the bandit
    make the same choice as predict-then-sweep?)

**For Path 2 (meta-optimizer):**

- **Parameterized scoring function.** The current `comfort_cost` and
  `energy_cost` in `control.py` use hard-coded weights. Refactor to accept
  a `SweepParams` dataclass that the bandit tunes.

- **Outcome tracking.** After each control cycle, log the actual temperature
  trajectory and compute a retroactive cost. Compare the cost from the
  bandit's chosen parameters against a fixed baseline.

- **This is a much smaller change.** The existing experiment framework handles
  it well — it's just a new experiment that modifies `control.py`'s scoring,
  not the model type.

## Recommended Sequence

1. **Fix prediction accuracy first.** The current ~3°F RMSE at 1–6h makes
   sweep decisions unreliable regardless of optimization approach. Priority
   improvements:
   - Weather forecast integration (outdoor temp wrong by 10–15°F at 6h)
   - Retrospective HVAC features (slab thermal charge)
   - More training data (time cures many ills)

2. **Add reward tracking now.** Regardless of bandit plans, logging
   (context, action, outcome, cost) after each control cycle is cheap and
   creates the dataset needed for any future bandit work.

3. **Experiment with Path 2 first.** Once prediction accuracy improves:
   - Parameterize the sweep scoring function
   - Run offline VW experiments on logged (context, params, cost) tuples
   - Compare bandit-tuned parameters vs. hand-tuned defaults

4. **Path 1 as a longer-term research direction.** Requires:
   - Model interface abstraction (valuable regardless)
   - Substantial online learning infrastructure
   - Solution for delayed rewards and exploration safety
   - Much more data (multiple seasons)

## VW Integration Notes

VW would be added as an optional dependency in `ml/pyproject.toml`. Key
parameters for CATS:

```python
# Example: learning boiler setpoint (continuous, 100-160°F)
vw = vowpalwabbit.Workspace(
    "--cats 16"           # 16 discretization buckets
    " --bandwidth 2"      # ±2°F smoothing
    " --min_value 100"
    " --max_value 160"
    " --epsilon 0.1"      # 10% exploration
)
```

For Path 2 (meta-optimizer), the action space would be the scoring function
parameters, with VW learning which parameter settings minimize retroactive
comfort+energy cost.
