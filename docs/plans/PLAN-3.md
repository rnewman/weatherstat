# Blower & Mini-Split Control Integration

## Context

The control system currently only sweeps 4 binary thermostat combinations (up on/off x dn on/off). Blowers and mini-splits are passed through from current HA state. The models already include blower/mini-split features — we just need to vary them in the control sweep.

**Goal**: Actively control all HVAC devices — thermostats (binary), blowers (off/low/high), and mini-splits (mode + target temp) — in a unified sweep. Make the architecture extensible for the 4 additional blowers planned.

## Design Decisions

### Mini-split modes: heat/off/cool (not target-swept)
- **3 modes: off, heat, cool.** The target temperature only controls how quickly the split approaches the thermal asymptote — the fundamental decision is whether to add heat, remove heat, or do nothing.
- **Skip "auto"**: cedes the heat/cool decision to the split's own thermostat, defeating our controller's purpose.
- **Skip "fan_only"/"dry"**: not temperature control.
- For the model override during sweep, use a fixed representative target (72°F). This gives the model a realistic `target_delta` without multiplying scenarios.
- The actual command target is computed AFTER the sweep from the room's comfort schedule midpoint.

### Blower modes
- All 3: off, low, high (matching model encoding 0/1/2).

### Sweep space (2 blowers, 2 mini-splits)
- Thermostats: 4 | Blowers: 3^2 = 9 | Mini-splits: 3^2 = 9
- **Total: 324 combos** (~4s at 13ms/combo). Fast.

### Extensibility for 6 blowers
- 3^6 = 729 blower combos → 4 × 729 × 9 = 26,244 (too many). Options when the time comes:
  - Binary (off/high): 2^6 = 64 → 4 × 64 × 9 = 2,304 (~30s)
  - Hierarchical sweep (thermostats first → best 2 → sweep blowers → best 3 → sweep splits)
- For now, config-driven device lists so adding blowers is a config change.

### Energy cost hierarchy
- Gas (Navien via thermostat): 0.010 per zone — highest cost
- Electric (mini-split heat pump): 0.005 per unit — efficient but uses electricity
- Fan (blower): 0.001/0.002 (low/high) — negligible
- These are tiebreakers when comfort is equal.

## Implementation

### 1. Device config (`config.py`)

Add frozen dataclasses `BlowerConfig` and `MiniSplitConfig` carrying all column names, entity IDs, and encoding info. Add `BLOWERS` and `MINI_SPLITS` tuples — the single source of truth for device lists.

```python
BLOWERS: tuple[BlowerConfig, ...] = (
    BlowerConfig(name="family_room", feature_col="blower_family_room_mode_enc",
                 command_key="blowerFamilyRoomMode"),
    BlowerConfig(name="office", feature_col="blower_office_mode_enc",
                 command_key="blowerOfficeMode"),
)

MINI_SPLITS: tuple[MiniSplitConfig, ...] = (
    MiniSplitConfig(name="bedroom", mode_feature_col="mini_split_bedroom_mode_enc",
                    target_feature_col="mini_split_bedroom_target",
                    delta_feature_col="bedroom_target_delta",
                    temp_col="mini_split_bedroom_temp",
                    command_mode_key="miniSplitBedroomMode",
                    command_target_key="miniSplitBedroomTarget"),
    MiniSplitConfig(name="living_room", ...),
)

MINI_SPLIT_SWEEP_MODES = ("off", "heat", "cool")
MINI_SPLIT_SWEEP_TARGET = 72.0  # Representative target for model override
```

Adding a blower = adding one `BlowerConfig`. `BlowerConfig.levels` can be shortened to `("off", "high")` to reduce sweep space.

### 2. New types (`types.py`)

```python
@dataclass(frozen=True)
class BlowerDecision:
    name: str        # "family_room"
    mode: str        # "off", "low", "high"

@dataclass(frozen=True)
class MiniSplitDecision:
    name: str        # "bedroom"
    mode: str        # "off", "heat", "cool"
    target: float    # command target (derived from comfort schedule after sweep)

@dataclass(frozen=True)
class HVACScenario:
    upstairs_heating: bool
    downstairs_heating: bool
    blowers: tuple[BlowerDecision, ...]
    mini_splits: tuple[MiniSplitDecision, ...]
```

Update `ControlDecision`: add `blowers` and `mini_splits` fields.
Update `ControlState`: add `blower_modes`, `mini_split_modes`, `mini_split_targets` dicts.

### 3. Override builder (`inference.py`)

Replace `_build_heating_overrides(up, dn)` with `build_hvac_overrides(scenario, current_split_temps)`:
- Thermostat action encodings (same as before)
- Navien mode (fires when either thermostat on, same as before)
- Blower mode encodings from scenario (no longer follows downstairs thermostat)
- Mini-split mode encodings (off=0, heat=1, cool=-1)
- Mini-split target override: fixed 72°F when heat/cool, unchanged when off
- Delta override: 72 - current_split_temp when heat/cool, unchanged when off
- The fixed target gives the model a realistic delta without needing to sweep targets

### 4. Scenario generation & sweep (`control.py`)

`generate_scenarios(season="winter")` — cartesian product of all device options from config.

`sweep_scenarios()` replaces `sweep_heating()`:
- Same structure: iterate scenarios, apply overrides, predict, compute cost, track best
- Passes `current_temps` dict (mini-split sensor temps) for delta computation
- Returns `ControlDecision` with all device decisions

### 5. Energy cost (`control.py`)

`compute_energy_cost(scenario: HVACScenario)` — tiered: gas zones > mini-split active modes > blower fans.

### 6. Command JSON (`control.py`)

`write_command_json(decision)` — no more `current_state` parameter. All device values come from the decision. The JSON format is unchanged (same keys the executor expects).

### 7. Output & display (`control.py`)

- Current state section: show blower modes and mini-split mode/targets
- Decision section: show blower and mini-split decisions
- Prediction table: unchanged (already per-room)

### 8. State persistence (`control.py`)

`save_control_state`/`load_control_state` — add blower/mini-split state. Old state files load cleanly via `.get()` defaults.

### 9. Counterfactual (`inference.py`)

Update `predict_counterfactual()` with named scenarios (all_off, thermo_only, thermo+blowers, thermo+splits@72, everything_on) rather than full 900-combo sweep.

## Files Modified

| File | Changes |
|------|---------|
| `ml/src/weatherstat/config.py` | Add `BlowerConfig`, `MiniSplitConfig`, device tuples, target constants |
| `ml/src/weatherstat/types.py` | Add `BlowerDecision`, `MiniSplitDecision`, `HVACScenario`; update `ControlDecision`, `ControlState` |
| `ml/src/weatherstat/inference.py` | Replace `_build_heating_overrides` → `build_hvac_overrides`; update counterfactual |
| `ml/src/weatherstat/control.py` | Add `generate_scenarios`, replace `sweep_heating` → `sweep_scenarios`, update energy cost, command JSON, display, persistence |

**No changes needed**: features.py, training pipeline, executor.ts, ha-client types, collector.

## Verification

1. `just lint` — clean
2. `just control` — shows 324-combo sweep with blower + mini-split decisions
3. Verify sweep time is ~4s
4. `just control --live` — command JSON includes all device decisions
5. `just counterfactual` — shows named HVAC scenarios
6. Sanity: compare with previous thermostat-only decisions (should produce similar thermostat choices when blowers/splits add no benefit)
