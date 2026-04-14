# Sysid Gain Deconfounding Experiments

## Problem

The optimizer ran the living room mini split in cool mode to pull
`thermostat_upstairs_temp` into its nighttime preferred band — overcooling piano_temp to
near its hard rail for a marginal comfort benefit on a single sensor.

Initial hypothesis: the cross-room gain `mini_split_living_room → thermostat_upstairs_temp`
(+0.608°F/hr, t=8.28) was inflated by time-of-day confounding. We designed three
experiments to reduce confounded gains while preserving real ones.

## Experiment harness

Built `scripts/experiment_sysid.py` — runs `fit_sysid()`, extracts structured metrics
(per-sensor R²/DW/holdout, per-effector direct/cross-room gain classification, aggregate
cross/direct ratio), saves JSON, and does side-by-side comparison against a baseline.

```bash
just experiment-sysid --save-baseline
just experiment-sysid --compare experiments/baseline.json --variant <name>
```

Baseline saved at `experiments/baseline.json` (13,904 snapshots, 2026-02-01 to 2026-04-13).

## Baseline metrics

| Metric | Value |
|--------|-------|
| Mean R² | 0.0851 |
| Mean DW | 0.2753 |
| Mean direct gain | 0.5412 °F/hr |
| Mean cross-room gain | 0.2080 °F/hr |
| Cross/direct ratio | 0.384 |
| Target: ms_lr → tu_temp | +0.608 °F/hr |

`living_room_climate_temp` has grade F (holdout 32% degradation, R²=0.014) — its regression
is unreliable. Piano_temp (+0.873, t=14.3) is the most robust mini split gain.

## Experiment 1: Time-of-day Fourier features

### Hypothesis

sin/cos harmonics of local hour absorb diurnal internal-heat patterns (occupancy, cooking,
stored wall heat) that confound cross-room gains.

### Implementation

4 features in `_fit_sensor_model()` after weather features: `sin(2πh/24)`, `cos(2πh/24)`,
`sin(4πh/24)`, `cos(4πh/24)` using fractional local hour. Not selectively standardized
(natural scale [-1, 1]).

### Results (2026-04-13)

| Metric | Baseline | Fourier | Change |
|--------|----------|---------|--------|
| Mean R² | 0.0851 | 0.0895 | **+5.2%** |
| Mean DW | 0.2753 | 0.2760 | +0.3% |
| Mean direct gain | 0.5412 | 0.5266 | -2.7% |
| Mean cross-room gain | 0.2080 | 0.2018 | -3.0% |
| Cross/direct ratio | 0.384 | 0.383 | -0.3% |
| Target: ms_lr → tu_temp | +0.608 | +0.592 | -2.5% |

R² improved across all 9 sensors (kitchen upgraded B→A). Cross-room gains decreased
consistently but only ~3%. Time-of-day is a minor confounder — the Fourier features capture
real diurnal structure (hence the universal R² improvement) but this structure wasn't
strongly correlated with HVAC activity.

### Verdict

**Land it.** The R² improvement is clean and universal. The gain reduction is small but
consistently in the right direction. No regressions. 20 lines of code.

### To land

In `_fit_sensor_model()`, after the weather features block (after line ~666), before the
environment/advisory section (line ~668), add:

```python
# Time-of-day Fourier features — absorb diurnal patterns in internal
# heat (occupancy, cooking, stored wall heat) that correlate with HVAC
# activity but aren't caused by it.
if "_ts" in df.columns:
    _tz = _CFG.location.timezone
    ts_local = df["_ts"].dt.tz_convert(_tz)
    hour_frac = (ts_local.dt.hour + ts_local.dt.minute / 60.0).values
    hour_rad = hour_frac * (2 * np.pi / 24)
    feature_names.extend(["_tod_sin1", "_tod_cos1", "_tod_sin2", "_tod_cos2"])
    feature_cols.extend([
        np.sin(hour_rad),
        np.cos(hour_rad),
        np.sin(2 * hour_rad),
        np.cos(2 * hour_rad),
    ])
```

Update feature count expectations in tests if any assert on `n_features`.

## Experiment 2: Transition-weighted regression

### Hypothesis

Upweighting observations near effector state transitions isolates causal signal from
steady-state confounding.

### Results: symmetric kernel (2026-04-13)

Initial test with symmetric Gaussian kernel (σ=30 min, 5× boost): all gains *increased*
(ms→tu: 0.608→0.700, +15%). The symmetric kernel includes pre-transition observations
where the effector hasn't acted yet — these are pure confounding, inflating estimates.

### Results: forward-only kernel sweep (2026-04-13)

Swept σ from 10 to 120 minutes with forward-only kernel (zero weight before transition),
plus per-effector σ (based on `max_lag_minutes`) and symmetric comparison:

```
Config             ms→tu    ms→piano  ms→direct  ms_xroom  mean_R²  xd_ratio
─────────────────  ──────   ────────  ──────────  ────────  ───────  ────────
baseline           +0.608   +0.873    +1.091      0.232    0.0851    0.213
fwd_σ=10m          +0.688   +1.059    +1.173      0.255    0.0825    0.217
fwd_σ=20m          +0.713   +1.099    +1.172      0.263    0.0829    0.224
fwd_σ=30m          +0.726   +1.113    +1.157      0.269    0.0831    0.232
fwd_σ=60m          +0.733   +1.097    +1.146      0.278    0.0835    0.242
fwd_σ=90m          +0.724   +1.065    +1.160      0.279    0.0839    0.240
fwd_σ=120m         +0.714   +1.042    +1.170      0.277    0.0841    0.237
fwd_per-eff        +0.712   +1.057    +1.156      0.270    0.0837    0.234
sym_σ=30m          +0.700   +1.051    +1.147      0.267    0.0834    0.233
```

### Key findings

1. **The ms→tu gain is real.** Every kernel configuration *increases* it from the OLS
   baseline. If it were confounded, focusing on causal variation should shrink it. The
   steady-state OLS estimate (0.608) is actually *underestimating* the true coupling.

2. **Gains stabilize at σ=30-60m**, then slightly decline at wider windows — the
   convergence behavior expected from a real physical effect peaking at the propagation
   timescale.

3. **Piano gain also increases** (0.873→1.11 at σ=30m), confirming strong physical
   coupling between the mini split and the adjacent piano room.

4. **The direct gain is rock-solid** at 1.09-1.17 across all configurations.

5. **R² drops slightly** (0.0851→0.0825-0.0841) — expected since WLS optimizes for
   transition windows, not overall MSE.

### Verdict

**Do not land.** The transition weighting amplifies all gains — it doesn't selectively
reduce confounded ones. The target gain is real, not confounded. The approach is valid
as a diagnostic (it confirmed the gain is causal) but not useful as a production change.

## Experiment 3: Conditional subsample validation

Not run. The transition weighting sweep already answered the causality question for the
target gain. Subsample validation remains available as a diagnostic for other gains if
needed in the future.

## Conclusion

**The gains are real.** The mini split genuinely affects thermostat_upstairs_temp at
~0.6-0.7°F/hr through the open-plan layout. The original problem — unnecessary cooling —
is a **scoring/weighting issue**, not a sysid issue.

The optimizer ran the mini split because `thermostat_upstairs_temp` was 1.7°F above its
nighttime preferred_hi (70.8°F after Away + MRT), and the comfort cost from that single
sensor outweighed the overcooling cost on piano (which has low `cold_penalty: 0.2`). The
fix is adjusting the comfort weights so that the cost of overcooling piano outweighs the
marginal benefit of cooling thermostat_upstairs slightly faster than natural decay:

- Raise `cold_penalty` on piano (already done: bumped from 0.2)
- Widen the nighttime thermostat_upstairs preferred range (already done: 69–72)
- More generally: review `cold_penalty` / `hot_penalty` weights across sensors to ensure
  they reflect actual occupant discomfort, not just symmetric defaults

The existing scoring mechanism (asymmetric cold/hot penalties per sensor per time-of-day)
is the right tool. No new scoring features are needed.

## Deliverables

| Item | Status |
|------|--------|
| Experiment harness (`scripts/experiment_sysid.py`) | Done, on main |
| Baseline metrics (`experiments/baseline.json`) | Done, on main |
| `just experiment-sysid` task | Done, on main |
| Fourier features in sysid | **Landed** |
| Transition weighting | Explored, not landing (diagnostic value only) |
| Constraint weight tuning | User-side config change (already applied) |
