# Plan: Advisory Effectors in Trajectory Sweep

## Problem

Windows are currently treated as fixed environmental state for the entire 6-hour prediction horizon. When windows are open, the simulator predicts aggressive cooling, triggering panic heating. The system fights its own advisories.

But windows aren't the only user-operated device. Space heaters, manual blinds, vent fans — these all affect the physics and the system can only advise on them, not control them directly. The fix isn't "add windows to the sweep" — it's a general **advisory effector** concept.

## Core Principles

1. **Advisory effectors are user-operated devices the system can observe, model, and advise on — but not directly control.** They participate in the trajectory sweep alongside HVAC effectors. The executor sends notifications, not commands.

2. **Fit all effects, don't declare them.** Sysid includes every advisory effector state as a feature in every regression (tau, gains, solar). A window might modify tau AND have a small draft-cooling gain. A space heater might be primarily a gain source but also slightly modify tau. The system discovers the physics from data; gain filtering (t-stat, magnitude cap) prunes insignificant effects. No `effect_type` in config.

3. **Three planning layers** from a single sweep, gated by three-tier comfort bounds:
   - **Proactive advice**: sweep all advisory effectors including state changes from default → recommendations ("open piano window for 2h to save energy")
   - **Reasonable plan**: when advisory effectors are non-default, sweep return-to-default timings, assume user cooperates at optimal time → **this drives HVAC action selection** (prevents panic heating)
   - **Worst-case hedge**: evaluate "user does nothing" scenario; if temperatures breach `backup` bounds, take defensive HVAC action NOW and explain why

4. **Explicit declaration only.** Advisory effectors must be declared in `advisory_effectors` config. State sensors with `encoding: window` that aren't declared continue to work as passive fixed state (affect sysid/simulator but aren't swept). No auto-synthesis — the user controls what gets advised on.

5. **Compute always, present selectively.** Generate the full advisory opportunity set every cycle. A separate presentation layer decides what to surface and when (notification frequency, suppression, grouping). Decouple computation from notification policy.

## Three-Tier Comfort Bounds

Replace the current two-tier model (`preferred` + `min`/`max`) with three tiers that map to the three planning layers:

```yaml
constraint_schedules:
  living_room_climate_temp:
    daytime:
      hours: [8, 22]
      preferred: [70, 72]    # Dead band: zero cost inside this range
      acceptable: [68, 74]   # Normal comfort-cost-weighted HVAC action
      backup: [65, 78]       # Worst-case hedge: defensive action if breached
```

| Tier | Cost model | Planning layer |
|------|-----------|----------------|
| `preferred` | Zero cost (dead band) | — |
| `acceptable` | Normal quadratic comfort cost | Reasonable plan (L2): drives HVAC action and advisory scoring |
| `backup` | Steep penalty (10× current hard-rail penalty) | Worst-case hedge (L3): triggers defensive HVAC when "user does nothing" breaches this |

How the tiers interact with advisory planning:

- **Reasonable plan scoring** uses `acceptable` bounds. When the optimizer sweeps advisory return-to-default timings × HVAC options, it scores against `acceptable`. This means the system won't aggressively heat to get back inside `preferred` when windows are open — it'll plan HVAC to keep things within `acceptable` assuming the user closes at the optimal time.

- **Worst-case hedge scoring** uses `backup` bounds. The "user does nothing" scenario is scored against `backup`. Only when this wider range is breached does the system override with defensive HVAC and warn the user. This prevents panic: windows open on a 60°F day might push living room to 69°F (outside `preferred`, inside `acceptable`, well inside `backup`) — no defensive action needed.

- **Proactive advice scoring** uses `acceptable` bounds. The benefit of "open this window" is measured as cost reduction within `acceptable`. Small benefits (temperature moves within `preferred` → `acceptable`) are computed but the presentation layer may choose not to surface them.

### Migration from current bounds

Current config has `preferred` (point or range) and `min`/`max` (hard rails). Migration:
- `preferred` → `preferred` (unchanged)
- `min`/`max` → `acceptable` (the current hard-rail penalties now apply at `acceptable`)
- `backup` → new, defaults to `acceptable` widened by a configurable margin (default 3°F / 1.7°C each side)

This is backward compatible: existing configs without `backup` get a reasonable default. Existing penalty behavior at `min`/`max` is preserved as `acceptable`.

## Config

New YAML section. Advisory effectors reference their state sensor (which still lives in `sensors.state`) and declare a default state (what the system considers "normal" / what the user should return to).

```yaml
advisory_effectors:
  piano_window:
    state_sensor: piano_window       # refs sensors.state entry
    default_state: closed
  living_room_window:
    state_sensor: living_room_window
    default_state: closed
  office_heater:
    state_sensor: office_space_heater
    default_state: "off"
```

That's it. No `effect_type`, no gain declarations, no physical model config. Sysid learns what each device does.

State sensors with `encoding: window` (or any encoding) that aren't listed in `advisory_effectors` remain passive: sysid and the simulator use their current state, but the sweep holds them fixed. No auto-synthesis, no unsolicited advice. A migration script (`scripts/migrate-advisory-effectors.py`) helps users add existing window sensors to the new section.

## Sysid Changes

### Current (window-specific)
- Tau regression: `1/tau = 1/tau_base + Σ(beta_w × is_open_w) + Σ(beta_pair × is_open_w1 × is_open_w2)`
- Gain regression: HVAC effectors only
- Solar regression: no state interactions

### Generalized
- **Tau regression**: include all advisory effector states as features. `window_betas` → `advisory_tau_betas` keyed by device name. `cross_breeze_betas` → `advisory_interaction_betas` for any pairwise combination (not just windows).
- **Gain regression**: include advisory effector states as additional columns alongside HVAC effectors. A space heater gets a gain coefficient like a thermostat does. A window might get a small draft-cooling gain.
- **Solar regression**: include advisory effector states as interaction terms with solar elevation. Blinds closing reduces solar gain; the regression discovers this.
- **Gain filtering** applies uniformly to all advisory effects: t-stat ≥ 1.5, magnitude cap, mode-direction clamp where applicable.
- **Pairwise interaction betas**: fit for all advisory effector pairs, let gain filtering prune noise. Prototype and measure — if cross-type interactions (window × heater) are always noise, restrict later.

### Output (`thermal_params.json`)

```json
{
  "tau": { "living_room_climate_temp": 55.2, ... },
  "advisory_tau_betas": {
    "piano_window": { "living_room_climate_temp": 0.003, "piano_temp": 0.004 },
    "living_room_window": { "living_room_climate_temp": 0.005 }
  },
  "advisory_interaction_betas": {
    "piano_window+living_room_window": { "living_room_climate_temp": 0.001 }
  },
  "gains": {
    "thermostat_upstairs": { ... },
    "office_heater": { "office_temp": 1.5, "hallway_temp": 0.2 }
  },
  "advisory_solar_betas": {
    "living_room_blinds": { "living_room_climate_temp": -0.3 }
  },
  "solar_elevation_gains": { ... }
}
```

Advisory effects that don't survive gain filtering are simply absent. A window with no significant gain effect has no entry in `gains`. A space heater with no significant tau effect has no entry in `advisory_tau_betas`.

## Data Structures (`types.py`)

### `AdvisoryEffectorConfig`

```python
@dataclass(frozen=True)
class AdvisoryEffectorConfig:
    name: str
    state_sensor: str       # key in sensors.state
    default_state: str      # "closed", "off", etc.
```

### `AdvisoryDecision`

```python
@dataclass(frozen=True)
class AdvisoryDecision:
    name: str
    action: str             # "hold", "close", "open", "turn_off", "turn_on", etc.
    transition_step: int    # 5-min step at which transition occurs (0 = now)
```

### `Scenario` expansion

```python
@dataclass(frozen=True)
class Scenario:
    effectors: dict[str, EffectorDecision]
    advisories: dict[str, AdvisoryDecision] = field(default_factory=dict)
```

Empty `advisories` = "hold all at current state" = current behavior (backward compat).

### Three-tier comfort schedule

```python
@dataclass(frozen=True)
class ComfortBounds:
    preferred_lo: float
    preferred_hi: float
    acceptable_lo: float
    acceptable_hi: float
    backup_lo: float
    backup_hi: float
```

### Planning result

```python
@dataclass(frozen=True)
class PlanningResult:
    hvac_plan: Scenario                        # HVAC actions to execute (from reasonable plan)
    advisory_recommendations: list[AdvisoryDecision]  # all computed recommendations
    worst_case_warnings: list[str]             # safety warnings if user does nothing
    advisory_scores: dict[str, float]          # per-advisory cost delta (for presentation layer)
```

## Simulator Changes

### Per-step advisory state timelines

`_build_advisory_timelines(scenarios, advisory_states, n_future)` → `dict[str, np.ndarray]` where each array is `(n_scenarios, n_future)` float (0.0 or 1.0, or intermediate for proportional effects later).

A timeline encodes the advisory effector's state at each step based on its `AdvisoryDecision`:
- `hold`: constant at current state
- transition: current state until `transition_step`, then flip

### Per-step tau computation

`compute_tau_matrix(tau_model, advisory_timelines, n_scenarios, n_future)` → `(n_scenarios, n_future)` float.

```python
inv_tau = 1/tau_base
for device, timeline in advisory_timelines.items():
    if device in advisory_tau_betas:
        inv_tau += advisory_tau_betas[device][sensor] * timeline  # (n_scenarios, n_future)
for pair, beta in advisory_interaction_betas.items():
    d1, d2 = pair.split("+")
    inv_tau += beta[sensor] * advisory_timelines[d1] * advisory_timelines[d2]
tau_matrix = 1.0 / inv_tau
```

### Per-step gains

Advisory effectors with gain effects contribute to the gain sum at each step, modulated by their timeline:

```python
for device, timeline in advisory_timelines.items():
    if device in gains:
        gain_contribution += gains[device][sensor] * timeline[:, step]
```

### Per-step solar modulation

Advisory effectors with solar betas modulate the solar forcing:

```python
solar_factor = 1.0
for device, timeline in advisory_timelines.items():
    if device in advisory_solar_betas:
        solar_factor += advisory_solar_betas[device][sensor] * timeline[:, step]
solar_forcing *= solar_factor
```

### Euler loop

All three effects apply per-step:
```python
dTdt = (outdoor_vec[step_idx] - T) / tau_matrix[:, step_idx] \
     + gain_sum[:, step_idx] \
     + solar_forcing[:, step_idx]
```

### Fast path

If no scenarios have non-hold advisory decisions (empty `advisories` dict on all scenarios), fall back to scalar tau and current gains. Preserves performance for the common case (no advisory effectors, or all at default state).

## Scenario Generation (`control.py`)

### `_advisory_sweep_options(advisory_configs, advisory_states, thermal_params)`

For each advisory effector:

**Currently non-default** (window open, heater on):
- Hold (= worst case scenario)
- Return to default at T+0, T+1h, T+2h, T+3h

**Currently default** (window closed, heater off):
- Hold
- Activate at T+0 with return at T+1h, T+2h, T+3h (for proactive advice)

**Relevance filter**: only advisory effectors with surviving effects in `thermal_params` (any of: tau_betas, gains, solar_betas on a constrained sensor) get sweep options. Others are held.

### Sweep structure

```
HVAC scenarios × advisory combos = total scenarios
```

Combinatorics management:
- 0 active advisories → 1× (no change)
- 2 active × 4 return options → 16× (~3K total)
- 4 active × 4 return options → 256× (~51K total)
- Hard cap: if >50K, coarsen advisory grid to [0, T+2h] only
- Fallback: if still >100K, hold all advisories (degrade to current behavior)

### Three-layer extraction from sweep results

All layers come from the same sweep:

1. **Reasonable plan** (Layer 2): among scenarios where non-default advisories return to default within the horizon, score against `acceptable` bounds. Lowest-cost scenario's HVAC component = commands to execute. Advisory component = optimal close/off timing to recommend.

2. **Worst-case hedge** (Layer 3): among scenarios where non-default advisories are held (user does nothing), score against `backup` bounds. If any sensor breaches `backup`, override HVAC toward defensive action and generate warning message explaining why.

3. **Proactive advice** (Layer 1): among scenarios where default-state advisories are activated, score against `acceptable` bounds. Compute cost delta vs best hold-all scenario. All recommendations generated; presentation layer decides what to surface.

## Advisory Output

### Command JSON

```json
{
  "effectors": { ... },
  "advisoryRecommendations": [
    {
      "device": "piano_window",
      "action": "close",
      "inMinutes": 60,
      "reason": "Room will reach optimal temp; closing saves energy",
      "costDelta": -0.42,
      "layer": "reasonable"
    }
  ],
  "advisoryWarnings": [
    {
      "message": "Starting heat: piano window open 2h, room projected to hit 64°F by 6PM if left open (backup minimum is 65°F)",
      "layer": "worst_case"
    }
  ],
  "proactiveAdvice": [
    {
      "device": "piano_window",
      "action": "open",
      "durationMinutes": 120,
      "costDelta": -0.31,
      "layer": "proactive"
    }
  ]
}
```

Executor ignores advisory fields — notifications only.

### Presentation layer (separate concern)

The control loop generates the full advisory set every cycle. A presentation component decides:
- Which recommendations to surface as notifications
- Notification frequency / cooldown
- Grouping ("open piano + living room windows for 2h")
- Dismissal lifecycle (advisory state changes, recommendation expires)
- Benefit threshold for surfacing (small `costDelta` → suppress)

This separation means the optimizer doesn't need to think about notification UX, and notification policy can evolve independently.

## Removing Window-Specific Code

| Current (window-specific) | Generalized |
|--------------------------|-------------|
| `window_betas` in TauModel | `advisory_tau_betas` in thermal params |
| `cross_breeze_betas` | `advisory_interaction_betas` |
| `adjust_schedules_for_windows()` | Removed — sweep handles it |
| `evaluate_window_opportunities()` + re-sweeps | `extract_advisory_recommendations()` from sweep results |
| `process_opportunities()` with window-specific logic | Generic advisory notification lifecycle |
| `WindowState` enum if any | Advisory effector state from sensor |

## Implementation Phases

### Phase 1: Three-tier comfort bounds
- Add `backup` tier to `ComfortBounds` / constraint schedule parsing
- New three-tier config: `preferred`, `acceptable`, `backup`
- `backup` defaults from `acceptable` ± configurable margin when not specified
- Migration script: `scripts/migrate-comfort-tiers.py` converts old `min`/`max` → `acceptable`, adds `backup`
- Plumb `backup` through scoring (not yet used in sweep — just data structures)
- Tests: parse three-tier config, default derivation

### Phase 2: Sysid generalization
- Rename window-specific params to advisory params in sysid output
- Include advisory effector states (from config) in all regressions (tau, gains, solar)
- Pairwise interaction betas for all advisory pairs (prototype, measure noise)
- Gain filtering on advisory effects
- Migration script: `scripts/migrate-thermal-params.py` renames `window_betas` → `advisory_tau_betas` etc. in existing `thermal_params.json`
- Tests: synthetic data with space-heater-like gain source

### Phase 3: Simulator — per-step effects
- `AdvisoryDecision` type, `advisories` on `Scenario`
- `_build_advisory_timelines()`
- Per-step tau, gains, and solar in Euler loop
- Fast path for no-advisory scenarios
- Tests: advisory close at step 12 produces different predictions than hold

### Phase 4: Scenario generation + three layers
- `_advisory_sweep_options()` with relevance filter
- HVAC × advisory product in `generate_trajectory_scenarios()`
- Layer extraction: reasonable plan (score vs `acceptable`), worst-case hedge (score vs `backup`), proactive advice
- Combinatorics management (coarsening, caps)
- Tests: scenario count, layer extraction logic

### Phase 5: Control loop integration
- Remove `adjust_schedules_for_windows()` from control path
- `extract_advisory_recommendations()` replaces re-sweep advisory system
- Full advisory output in command JSON every cycle
- Worst-case warnings trigger defensive HVAC + explanation
- Tests: end-to-end dry-run

### Phase 6: Config + presentation + TUI
- `advisory_effectors` YAML section, `AdvisoryEffectorConfig`
- Presentation layer: notification policy, cooldown, benefit threshold
- TUI displays advisory recommendations with timing and cost deltas
- Prune all dead window-specific code paths (no backward-compat loaders)
- Migration script: `scripts/migrate-advisory-effectors.py` adds `advisory_effectors` section from existing window state sensors

### Phase 7: Documentation + archive
- Update `docs/FUTURE.md` to reflect advisory effectors as done, note follow-on work
- Update `docs/ARCHITECTURE.md` with advisory effector model, three-tier bounds, three planning layers
- Update `README.md` with advisory effector config example
- Update `docs/operations.md` with migration steps and new config fields
- Archive this plan to `docs/plans/advisory-effectors.md`
- Update `CLAUDE.md` with new stage entry

## Files Modified

| File | Changes |
|------|---------|
| `types.py` | `AdvisoryEffectorConfig`, `AdvisoryDecision`, `ComfortBounds` with backup tier, `advisories` on `Scenario`, `PlanningResult` |
| `yaml_config.py` | Parse `advisory_effectors` section, three-tier comfort bounds |
| `sysid.py` | Generalize window features → advisory features in all regressions |
| `simulator.py` | `_build_advisory_timelines()`, per-step tau/gains/solar, fast path |
| `control.py` | `_advisory_sweep_options()`, advisory combos in sweep, three-layer extraction, remove `adjust_schedules_for_windows` |
| `advisory.py` | Replace `evaluate_window_opportunities()` with `extract_advisory_recommendations()` + presentation layer |
| `tui/app.py` | Advisory recommendations display, three-tier comfort in status |
| `tui/widgets.py` | Timing + cost delta in opportunity panel |
| `scripts/migrate-comfort-tiers.py` | Migrate old min/max → acceptable + backup |
| `scripts/migrate-thermal-params.py` | Rename window_betas → advisory_tau_betas in thermal_params.json |
| `scripts/migrate-advisory-effectors.py` | Add advisory_effectors section from existing window state sensors |
| `tests/` | New tests for each phase |
| `docs/FUTURE.md` | Advisory effectors done, follow-on work |
| `docs/ARCHITECTURE.md` | Advisory effector model, three-tier bounds, three planning layers |
| `README.md` | Advisory effector config example |
| `docs/operations.md` | Migration steps and new config fields |
| `docs/plans/advisory-effectors.md` | Archived plan |

## Verification

1. `just test` passes at each phase
2. Three-tier bounds: parse correctly, `backup` defaults from `acceptable` ± margin
3. Sysid: synthetic data with advisory effector produces tau_betas AND gains
4. Simulator: advisory transition mid-horizon produces different trajectory than hold
5. Control: winning scenario includes advisory timing; HVAC action reflects cooperative assumption (scored vs `acceptable`)
6. Worst-case: open window with cold forecast, trajectory breaches `backup` → defensive heating + warning
7. Worst-case: open window, trajectory stays inside `backup` → no defensive action, just recommendation
8. Proactive: warm day with closed windows → "open for free cooling" in full advisory set
9. Presentation: small-benefit recommendations computed but not surfaced
10. Sweep time stays under 5s with 2-4 active advisories
11. No window-specific code paths remain (no backward-compat loaders)
12. Migration scripts: each converts existing config/data cleanly, idempotent
13. Docs: FUTURE.md, ARCHITECTURE.md, README.md, operations.md all reflect new model
14. Plan archived to `docs/plans/advisory-effectors.md`

## Open Questions

1. **Default `backup` margin**: when `backup` isn't specified, derive from `acceptable` ± how much? Suggested default: 3°F (1.7°C) each side. Configurable in `defaults:` section.

2. **Proactive sweep combinatorics**: Layer 1 sweeps currently-default advisories with activate+return options. With many advisory effectors this could explode. Mitigation: only sweep advisories with effects on sensors currently outside `preferred` (no point advising "open window" when already comfortable). Further mitigation: proactive and reactive scenarios are independent sets, so proactive doesn't multiply reactive.

3. **Advisory interaction betas**: pairwise interactions across heterogeneous device types (window × heater) may be noise. Plan: fit them all, let gain filtering prune. Measure empirically after Phase 2.
