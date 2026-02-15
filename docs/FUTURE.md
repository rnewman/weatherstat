# Sweep Scalability

The control loop evaluates HVAC scenarios and window combinations by brute-force
Cartesian product. This works at current scale but won't survive device expansion.

## Current cost

| Component | Combos | Model calls | Wall time |
|-----------|--------|-------------|-----------|
| HVAC sweep | 180 (pruned from 324) | 180 × 40 = 7,200 | ~2.4s |
| Window advisory | 2^7 = 128 | 128 × 40 = 5,120 | ~2.0s |

## Projected cost at 6 blowers + 10 windows

| Component | Combos | Model calls |
|-----------|--------|-------------|
| HVAC sweep | ~26,000 (4 × 3^6 × 9 before pruning) | ~1,000,000 |
| Window advisory | 2^10 = 1,024 | 40,960 |

## Optimization approaches (in implementation order)

### 1. Batch prediction

**Effort:** small (refactor inner loop). **Impact:** 10–50× on current code. **Risk:** none.

The current inner loop calls `model.predict(X)` on a single row per scenario per
model target — 7,200 individual calls for the HVAC sweep. LightGBM is optimized
for batch prediction. Stack all scenario feature rows into one DataFrame and call
`model.predict(X_batch)` once per model target. This turns 7,200 calls into 40
calls, each predicting 180 rows. Eliminates Python call overhead and improves CPU
cache utilization.

Applies identically to the window advisory sweep.

### 2. Window decomposition

**Effort:** small (replace 2^N loop). **Impact:** O(N) instead of O(2^N). **Risk:** low.

Windows barely interact with each other in the model — opening the bedroom window
doesn't change what the kitchen window does to the kitchen temperature. Replace the
full 2^N sweep with:

1. Compute baseline (current window state).
2. For each window independently, compute the cost of toggling it.
3. Toggle all windows where toggling improves cost.
4. One full-model verification pass on the combined result.

O(N) evaluations instead of O(2^N): 10 instead of 1,024 at 10 windows.

The failure mode is correlated windows (e.g., multiple upstairs windows creating
cross-breeze). The verification pass catches this, and the model is unlikely to have
learned multi-window interactions from the current training data.

### 3. Greedy coordinate descent (HVAC)

**Effort:** medium (new sweep logic). **Impact:** ~300× at 6 blowers. **Risk:** misses some device interactions.

Replace the Cartesian product with iterative single-device optimization:

1. Start from all-off (or current state).
2. For each device, evaluate all its settings while holding others fixed.
3. Apply the single-device change with the best cost reduction.
4. Repeat until no single-device change improves by more than threshold.

Cost per iteration: Σ(levels per device) = 6×3 + 2×2 + 2×3 = 28 evaluations.
Converges in 2–3 passes → ~84 evaluations vs 26,244.

The existing zone-blower constraint already assumes approximate separability. To
handle known interactions, group correlated devices: sweep (zone_heat, zone_blowers)
jointly, then mini-splits independently. Still much smaller than the full product.

### 4. Marginal contribution screening (two-stage)

**Effort:** medium (two-stage pipeline). **Impact:** best of both worlds. **Risk:** slightly more complex.

Exploit approximate additivity for a fast screening pass:

1. Compute baseline (all-off) predictions once.
2. For each device at each level, compute its marginal delta on each room prediction.
3. Approximate any combination as `baseline + Σ(marginal deltas)` — free linear
   combination, no model calls.
4. Score all combinations using the linear approximation.
5. Run full LightGBM predictions only for the top-K candidates (10–20).

Screening cost is O(Σ levels_i) model calls (same as one coordinate descent pass).
Full eval is O(K). The linear approximation isn't perfect — LightGBM captures
interactions — but it's excellent for ranking. The top candidates from the linear
screen almost always include the true optimum.

### 5. Spatial decomposition

**Effort:** medium (device-room influence map). **Impact:** 3–5× per evaluation. **Risk:** needs maintenance as devices change.

Most devices primarily affect one or two rooms:
- `blower_office` → office (maybe downstairs aggregate)
- `mini_split_bedroom` → bedroom
- `thermostat_upstairs` → upstairs, bedroom, kitchen, piano, bathroom

Formalize a device → affected rooms map (derivable from YAML `rooms.zone` + window
`rooms`). When evaluating a single-device change, only re-run room models for
affected rooms. Reduces per-evaluation cost from 40 model calls to 8–12 for a
single-device change.

Combines well with coordinate descent: each greedy step only re-predicts the rooms
that the candidate device influences.

## Interaction with Unified Action Framework

When HVAC and window actions merge into a single optimization (see PLAN.md,
"Unified Action Framework"), the combined search space is the product of electronic
and advisory actions. The approaches above compose naturally:

- Batch prediction speeds up any sweep strategy.
- Window decomposition applies to the advisory subset of the unified action space.
- Coordinate descent generalizes to all action types — evaluate one action at a time.
- Marginal screening provides fast approximate scoring for the full unified space.
- Spatial decomposition prunes re-prediction per action regardless of action type.

The likely end state is marginal screening (stage 4) over the full unified action
space, with spatial decomposition (stage 5) making each screening evaluation cheap,
and full-model verification on the top-K candidates.
