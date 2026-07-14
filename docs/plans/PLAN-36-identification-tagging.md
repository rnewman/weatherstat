# PLAN-36: Causal identification tagging for behavior-driven couplings

## Problem

Environment couplings (shades, windows, doors — user-operated) are fitted
from observational habit data: the "treatment" is when the user feels like
toggling the device, which correlates with time of day, season, and
weather. Measured consequences (July 2026, `scripts/shade_transition_test.py`):

- `moss_garden_shade → piano_temp` (−0.39, t −2.2 in batch) is **mostly
  habit correlation**: under transition-weighted identification it sheds
  ~40% of its magnitude and all significance. Both shades are
  `advisory: true`, so the sweep was crediting a ~2×-overstated benefit
  when counterfactually toggling the shade for the piano room.
- `moss_garden_shade → bedroom_temp` is **causal and batch-attenuated 3×**
  (−0.27 → −0.82 as identification tightens): observational confounding
  can *dilute* real couplings, not just invent fake ones. Prune-only guard
  rails cannot correct in that direction.

Effector gains don't have this problem — the controller flips them
quasi-experimentally thousands of times (stage-31 transition test confirmed
them causal). The gap is specifically human-operated devices.

Declarative scoping (an `affects:` map) was rejected: it is anti-thesis to
"learned over configured" — it would have vetoed the learned
family-room→living-room floor coupling, and dismissed possibly-real paths
(hallway airspace) by human prior rather than by data.

## Design

**Split coupling *use* by identification strength, not existence.**
Correlational couplings genuinely improve conditional prediction (habit
predicts, in distribution); they only lie under intervention. So: every
kept coupling serves observed-state prediction; only couplings that pass a
causal-identification bar inform the advisory sweep's counterfactual
toggles.

### Stage 2.5 (sysid, per kept advisory-device coupling)

For each env×tau and env×solar beta on an `advisory: true` device, refit
with Gaussian sample weights around the device's state transitions
(σ ∈ {8h, 4h, 2h}; `identification.py`, shared with the standalone test
script). Within a window, slow confounders are ~constant while the state
flips — within-window contrast identification. Classify:

- **causal** — at every σ: same sign as batch, |t| ≥ 1.5, and ≥50% of the
  batch magnitude retained (the retention bar catches high-SNR couplings
  that stay "significant" while collapsing severalfold).
- **correlational** — powered but fails persistence.
- **underpowered** — weighted signal energy below `INFO_MIN` at some σ
  (e.g., night-only transitions): no verdict, treated as causal-by-default
  since the batch |t| ≥ 2 bar was already passed (conservative =
  behavior-preserving).

Results land in `thermal_params.json` as `coupling_identification`
(family → device → sensor → {tag, beta_batch, by_sigma}).

### Enforcement (simulator)

`SimParams.correlational_couplings` (family → {(device, sensor)}). At the
two advisory-timeline application sites (per-step tau modulation, solar
modulation), a correlational pair is **pinned to the observed state**: the
scenario's counterfactual timeline for that device earns no credit for that
sensor, while the observed-state contribution (conditional prediction)
remains. advisory.py has no independent predict path (post-stage-25 all
counterfactuals flow through the unified sweep), so these two sites are the
complete enforcement surface.

## Validation

- Synthetic: causal effect holds; habit-correlation (effect absent near
  transitions) collapses; zero/night-only transitions → underpowered;
  enforcement semantics (toggle credits causal, not correlational;
  observed state still predicts both). `tests/test_identification.py`.
- Live acceptance: automatic tags reproduce the manual verdicts —
  moss→piano correlational, moss→bedroom causal, blinds→bathroom causal;
  all 16 env×tau couplings (doors/windows) causal with transition t up to
  28 (balcony door). 19 classified, 1 correlational. Sysid runtime impact
  negligible (~5.3s total).

## Future

- **Correction, not just gating:** replacing batch magnitudes with
  transition-identified values (e.g., recovering the bedroom attenuation)
  needs a converged estimator (the σ-sweep hasn't plateaued) and a
  counterfactual-quality gate — advisory-followed transitions accumulating
  in the decision log are that evidence. Tag now, correct when the
  interventional data arrives.
- Advisory-followed transitions are quasi-interventions with system-chosen
  timing: identification improves fastest for exactly the couplings the
  system acts on.
