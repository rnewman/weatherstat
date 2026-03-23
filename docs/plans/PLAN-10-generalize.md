# Plan: Unified Effector Model

## Context

The control loop treats thermostats, mini-splits, and blowers as fundamentally different types with separate code paths, hardcoded zone names ("upstairs"/"downstairs"), and type-specific decision dataclasses. But they're all effectors that act on sensors — the differences (lag magnitude, mode control, dependencies) are properties, not categories. The current architecture can't accommodate new effector types (space heaters, blinds, radiators) without adding more code paths.

Goal: a single effector decision type, config-driven scenario generation, and no hardcoded device names anywhere.

## Architecture

The simulator is already nearly general — it dispatches on `device_type` from sysid params and applies multiplicative dependency gates (state_gate for boiler, zone_active for blowers). The hardcoding is concentrated in:

1. **Type system** — three decision dataclasses (`ThermostatTrajectory`, `BlowerDecision`, `MiniSplitDecision`) and `TrajectoryScenario` with typed fields per category
2. **Scenario generation** — three separate option-building paths, nested 2-zone for loops, hardcoded blower zone constraint
3. **Control loop** — hardcoded zone names in sweep params, command JSON, display, counterfactuals
4. **TS executor** — hardcoded device arrays, typed Prediction interface

## Phase 1: Unified Decision Types (types.py)

Replace three decision types with one:

```python
@dataclass(frozen=True)
class EffectorDecision:
    """Decision for a single effector in a scenario."""
    name: str                          # "thermostat_upstairs", "blower_office"
    mode: str = "off"                  # "off", "heating", "heat", "cool", "low", "high"
    target: float | None = None        # regulating effectors (mini-split target temp)
    delay_steps: int = 0              # trajectory effectors (0 = start now)
    duration_steps: int | None = None  # trajectory effectors (None = full horizon)
```

Replace `TrajectoryScenario` with:

```python
@dataclass(frozen=True)
class Scenario:
    effectors: dict[str, EffectorDecision]  # effector_name -> decision
```

Replace `ControlDecision` zone-specific fields with:

```python
@dataclass(frozen=True)
class ControlDecision:
    timestamp: str
    effectors: tuple[EffectorDecision, ...]
    command_targets: dict[str, float]  # effector_name -> setpoint for command JSON
    total_cost: float = 0.0
    comfort_cost: float = 0.0
    energy_cost: float = 0.0
    predictions: dict[str, dict[str, float]] = field(default_factory=dict)
    trajectory_info: dict[str, dict[str, int | None]] = field(default_factory=dict)
    dry_run: bool = True
```

Replace `ControlState` zone-specific fields with:

```python
@dataclass(frozen=True)
class ControlState:
    last_decision_time: str
    setpoints: dict[str, float] = field(default_factory=dict)  # effector_name -> setpoint
    modes: dict[str, str] = field(default_factory=dict)         # effector_name -> mode
    mode_times: dict[str, str] = field(default_factory=dict)    # effector_name -> ISO timestamp
```

Delete `ThermostatTrajectory`, `BlowerDecision`, `MiniSplitDecision`.

**Files**: `ml/src/weatherstat/types.py`

## Phase 2: Unified Effector Config (config.py)

Add a single config class. Key design choices:

- **No `category` field.** Thermostats and mini-splits are both `climate` entities. Blowers are `fan` entities. The behavioral differences are captured by properties, not categories.
- **`control_type`** drives sweep strategy: `"trajectory"` (slow-twitch: on/off timing search), `"regulating"` (proportional band with target), `"binary"` (discrete modes).
- **`mode_control`**: `"manual"` (human controls mode — sweep skips mode, only does timing/target) vs `"automatic"` (system controls mode — sweep includes mode in search).
- **`supported_modes`**: what the device can do — `("heat",)` for heat-only thermostats, `("heat", "cool")` for mini-splits, `("off", "low", "high")` for blowers. Combined with `mode_control`, determines sweep space.
- **`depends_on`**: references another effector by name — `"thermostat_downstairs"`, not a zone. The dependency means "this effector only produces useful output when the named effector is actively delivering." For blowers, this is really about heat in the lines (thermostat calling AND boiler firing), which the simulator models via the multiplicative state_gate. The scenario-generation constraint is a pruning optimization.
- **`target_derives_from`**: `"comfort"` (regulating — target IS the comfort setpoint) vs `"control"` (trajectory — target is a derived control lever from `_cautious_setpoint`). This captures the distinction between mini-split targets (we want 72°F) and thermostat targets (we set 73°F to make it call for heat).

```python
@dataclass(frozen=True)
class EffectorConfig:
    name: str                          # "thermostat_upstairs", "mini_split_bedroom"
    entity_id: str
    control_type: str                  # "trajectory", "regulating", "binary"
    mode_control: str                  # "manual" (human controls mode) or "automatic"
    supported_modes: tuple[str, ...]   # ("heat",), ("heat", "cool"), ("off", "low", "high")
    command_keys: dict[str, str]       # purpose -> camelCase: {"target": "thermostatUpstairsTarget"}
    depends_on: str | None = None      # effector name this depends on (e.g., "thermostat_downstairs")
    state_device: str | None = None    # state sensor confirming delivery
    proportional_band: float = 1.0     # regulating: activity ramp width in °F
    mode_hold_window: tuple[int, int] | None = None  # quiet hours for mode changes
    mode_encoding: dict[str, float] = field(default_factory=dict)  # for energy cost / sysid
    temp_col: str = ""                 # sensor column for current temp (climate entities)
```

Build `EFFECTORS: tuple[EffectorConfig, ...]` from YAML, iterating all effector sections:

```python
EFFECTORS = tuple(itertools.chain(
    # thermostats → control_type="trajectory", mode_control="manual",
    #               supported_modes=("heat",), command_keys={"target": ...}
    # mini_splits → control_type="regulating", mode_control="automatic",
    #               supported_modes=("heat", "cool"), command_keys={"mode": ..., "target": ...}
    # blowers → control_type="binary", mode_control="automatic",
    #           supported_modes=("off", "low", "high"), depends_on="thermostat_downstairs"
))
```

Remove `BlowerConfig`, `MiniSplitConfig`, `BLOWERS`, `MINI_SPLITS`.

**Files**: `ml/src/weatherstat/config.py`

## Phase 3: Generic Scenario Generation (control.py)

Replace three separate option-building paths with one loop over `EFFECTORS`:

```python
per_effector_options: dict[str, list[EffectorDecision]] = {}

for eff in EFFECTORS:
    if eff.name in ineligible:
        per_effector_options[eff.name] = [EffectorDecision(eff.name)]  # off
        continue

    if eff.control_type == "trajectory":
        options = [EffectorDecision(eff.name)]  # off
        for delay, dur in trajectory_grid:
            options.append(EffectorDecision(eff.name, mode="heating",
                                           delay_steps=delay, duration_steps=dur))
        per_effector_options[eff.name] = options

    elif eff.control_type == "regulating":
        # [off] + [heat targets] + [cool targets] from comfort schedule
        ...

    elif eff.control_type == "binary":
        per_effector_options[eff.name] = [
            EffectorDecision(eff.name, mode=m) for m in ("off",) + eff.levels
        ]
```

### Dependency constraints

Effectors with `depends_on` only produce useful output when their dependency is actively delivering. The `depends_on` field names another effector directly (e.g., `depends_on: "thermostat_downstairs"`). The scenario-gen constraint is a pruning optimization — the simulator would correctly zero out the dependent effector's contribution via multiplicative activity, but we avoid generating thousands of useless scenarios.

```python
# Two-tier product: independent effectors first, then dependent
independent = {n: opts for n, opts in per_effector_options.items()
               if EFFECTOR_MAP[n].depends_on is None}
dependent = {n: opts for n, opts in per_effector_options.items()
             if EFFECTOR_MAP[n].depends_on is not None}

for indep_combo in product(*independent.values()):
    indep_dict = dict(zip(independent.keys(), indep_combo))
    # Resolve dependent options based on parent effector state
    dep_options = {}
    for dep_name, dep_opts in dependent.items():
        parent_name = EFFECTOR_MAP[dep_name].depends_on
        parent_dec = indep_dict.get(parent_name)
        # Parent must be actively delivering NOW (not delayed)
        if parent_dec and parent_dec.mode != "off" and parent_dec.delay_steps == 0:
            dep_options[dep_name] = dep_opts  # full sweep
        else:
            dep_options[dep_name] = [EffectorDecision(dep_name)]  # off only
    for dep_combo in product(*dep_options.values()):
        effectors = {**indep_dict, **dict(zip(dep_options.keys(), dep_combo))}
        scenarios.append(Scenario(effectors))
```

### Scenario count

Same as today for 2 thermostats: `|trajectory_grid+1|^2 × |blower_levels|^2 × |split_options|^2 ≈ 7400`. Grows with more effectors but remains tractable for the vectorized simulator.

**Files**: `ml/src/weatherstat/control.py` (generate_trajectory_scenarios, sweep_scenarios_physics)

## Phase 4: Update Simulator (simulator.py)

Update `_build_activity_matrices()` to use `Scenario.effectors` dict instead of typed field access.

### Trajectory active masks

Replace hardcoded `up_active`/`dn_active` with a dict built by iterating trajectory effectors:

```python
trajectory_active: dict[str, np.ndarray] = {}  # effector_name -> (n_scenarios, n_future) bool mask
for eff in params.effectors:
    if eff["device_type"] == "thermostat":
        name = eff["name"]
        decisions = [s.effectors[name] for s in scenarios]
        heating = np.array([d.mode != "off" for d in decisions])
        delay = np.array([d.delay_steps for d in decisions])
        dur = np.array([d.duration_steps or (n_future - d.delay_steps) for d in decisions])
        trajectory_active[name] = heating[:, None] & (steps >= delay[:, None]) & (steps < (delay + dur)[:, None])
```

### Thermostat dispatch

```python
if dtype == "thermostat":
    active = trajectory_active[name]
    future = active.astype(np.float64) * encoding.get("heating", 1.0)
```

### Dependent effector dispatch

```python
elif dtype == "blower":
    eff_cfg = EFFECTOR_MAP.get(name)
    dep_active = trajectory_active.get(eff_cfg.depends_on, np.ones(...)) if eff_cfg else np.ones(...)
    enc_vals = np.array([
        encoding.get(s.effectors[name].mode, 0.0) for s in scenarios
    ])
    future = dep_active.astype(np.float64) * enc_vals[:, None]
```

### Mini-split and regulating dispatch

Extract mode/target from `s.effectors[name]` instead of `_find_split()`.

**Files**: `ml/src/weatherstat/simulator.py`

## Phase 5: Update Control Loop (control.py)

### sweep_scenarios_physics

- Replace `up_current`/`dn_current` params with `current_temps: dict[str, float]` (keyed by effector name or zone name, derived from `EFFECTORS[*].temp_col`)
- Comfort max check: iterate trajectory effectors from config
- Cold-room override: iterate sensors below comfort min, find their coupled trajectory effector from sysid coupling matrix, check it's active in the scenario
- ControlDecision construction: `command_targets` dict built from config command_keys
- Energy cost: iterate `scenario.effectors.values()` with category-based cost lookup

### write_command_json

```python
for eff in EFFECTORS:
    decision = cd_effectors_map.get(eff.name)
    if not decision or eff.name in _ineligible:
        continue
    for purpose, key in eff.command_keys.items():
        if purpose == "mode":
            command[key] = decision.mode
        elif purpose == "target":
            command[key] = command_targets.get(eff.name)
```

### Counterfactuals

```python
for name, dec in winning.effectors.items():
    if dec.mode != "off":
        cf_effectors = dict(winning.effectors)
        cf_effectors[name] = EffectorDecision(name)  # off
        counterfactuals.append(Scenario(cf_effectors))
        cf_keys.append(name)
```

### Display

```python
for eff in EFFECTORS:
    dec = decision_map.get(eff.name)
    if eff.control_type == "trajectory":
        label = "ON" if dec and dec.mode != "off" else "OFF"
        target = command_targets.get(eff.name, "?")
        print(f"  {eff.name}: {label} → setpoint {target}°F")
    ...
```

### State persistence

`ControlState.setpoints` and `.modes` are dicts keyed by effector name. JSON round-trips naturally.

**Files**: `ml/src/weatherstat/control.py`

## Phase 6: Update Safety, Advisory, Decision Log

### safety.py

Iterate configured climate effectors with `mode_control: "manual"` (thermostats — mode is human-controlled, so if it's off we should alert):

```python
for eff in EFFECTORS:
    if eff.mode_control != "manual":
        continue
    heating = decision_effectors.get(eff.name, EffectorDecision(eff.name)).mode != "off"
    action = str(latest.get(f"{eff.name}_action", ""))
    if heating and action == "off":
        alerts.append(SafetyAlert(key=f"{eff.name}_off", ...))
```

### advisory.py

Replace `up_temp`/`dn_temp` with dict built from config:
```python
effector_temps = {
    eff.name: current_temps.get(eff.temp_col.removesuffix("_temp"), 71.0)
    for eff in EFFECTORS if eff.temp_col
}
```

### decision_log.py

Replace `upstairs_heating`/`downstairs_heating`/`upstairs_setpoint`/`downstairs_setpoint` columns with:
- `effector_decisions TEXT` — JSON of effector decisions list
- `command_targets TEXT` — JSON of command targets dict

Schema migration: `ALTER TABLE decisions ADD COLUMN effector_decisions TEXT`. Old columns left as-is (nullable).

**Files**: `safety.py`, `advisory.py`, `decision_log.py`

## Phase 7: Generalize TS Executor

### types.ts

```typescript
interface Prediction {
  timestamp: string;
  confidence: number;
  [key: string]: unknown;
}
```

### executor.ts

Replace hardcoded device arrays with config iteration:

```typescript
for (const [name, cfg] of Object.entries(config.thermostats)) {
  const key = snakeToPascal(`thermostat_${name}_target`);
  const target = prediction[key] as number | undefined;
  if (target === undefined) continue;  // ineligible
  await applyThermostat(client, cfg.entityId, target);
}
// Same pattern for mini-splits and blowers
```

### entities.ts

Remove individual device constant exports. Executor uses `config.*` directly.

**Files**: `ha-client/src/types.ts`, `ha-client/src/executor.ts`, `ha-client/src/entities.ts`

## Phase 8: Test Updates

All tests constructing `TrajectoryScenario(up_traj, dn_traj, blowers, splits)` → `Scenario({"thermostat_upstairs": ..., "thermostat_downstairs": ..., "blower_family_room": ..., ...})`.

All tests accessing `decision.upstairs_heating` → `decision.effectors_map["thermostat_upstairs"].mode != "off"`.

**Files**: `test_control.py`, `test_advisory.py`, `test_safety.py`, `conftest.py`

## Transition Safety

Command JSON keys are derived from config via `_snake_to_camel()`. For "upstairs" this produces `thermostatUpstairsTarget` — identical to today. The JSON format is unchanged, so Python and TS sides can be updated independently.

## What This Enables

- Adding a new climate entity (thermostat, mini-split, radiator valve) = YAML edit (+ sysid refit)
- Adding a new fan entity (blower) = YAML edit
- Adding a new effector type (space heater, electric blanket, blinds) = YAML edit + config parser + sysid picks up the gains
- Dependency backpropagation: sweep discovers that blower + thermostat outperforms either alone
- Future: trajectory search for any slow-twitch effector, not just thermostats
- Future: a new climate entity with mode_control="automatic" and control_type="trajectory" (slow-response central HVAC) works without code changes

## Key Files

- `ml/src/weatherstat/types.py` — EffectorDecision, Scenario, ControlDecision, ControlState
- `ml/src/weatherstat/config.py` — EffectorConfig, EFFECTORS tuple
- `ml/src/weatherstat/control.py` — scenario gen, sweep, command JSON, counterfactuals, display (~30 sites)
- `ml/src/weatherstat/simulator.py` — trajectory_active dict, dispatch by device_type
- `ml/src/weatherstat/safety.py` — iterate config thermostats
- `ml/src/weatherstat/advisory.py` — zone temp dict from config
- `ml/src/weatherstat/decision_log.py` — JSON columns
- `ha-client/src/executor.ts` — config iteration
- `ha-client/src/types.ts` — dynamic Prediction
- Tests: `test_control.py`, `test_advisory.py`, `test_safety.py`, `conftest.py`

## Verification

1. `just test` — all tests pass
2. `just lint` — clean
3. `just typecheck` — clean
4. `just control` — output shows effector names from config, scenario count unchanged
5. `just execute` — executor iterates config, applies commands correctly
6. Manually verify: add a fake thermostat zone to YAML, confirm control generates scenarios for it
