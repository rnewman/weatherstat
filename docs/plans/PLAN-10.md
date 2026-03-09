# PLAN-10: Mini Split Target Temperature Control

## Context

The control loop treats mini splits as binary on/off effectors — sweeping
`["off", "heat", "cool"]` every 15 minutes and picking the best mode. This
causes the mini split mode to cycle (heat→off→heat), which moves the vent
louvers and starts/stops the compressor. At night this is noisy and disruptive.

Mini splits are fundamentally different from zone thermostats. A zone
thermostat triggers a boiler that heats a slab — binary on/off with massive
thermal lag. A mini split is a PID controller: set a target temperature and
it regulates itself. The correct control variable is the **target temperature**,
not the mode. Mode (heat/cool/off) should be stable for hours.

---

## Effector Type Model

Three control types emerge from the hardware:

| Type | Examples | Control variable | Activity model |
|------|----------|-----------------|----------------|
| **Trajectory** | Thermostats | On/off schedule (delay × duration) | Binary, pre-computed |
| **Discrete** | Blowers | Level (off/low/high) | Constant, pre-computed |
| **Regulating** | Mini splits | Target temperature | Proportional to `(target - T)`, computed per Euler step |

These types are already implicit in the code. The plan makes them explicit
in config and uses them to drive different simulation and sweep strategies.

---

## Design

### 1. Config: `control_type` and `proportional_band`

Add to mini split YAML definitions:

```yaml
mini_splits:
  bedroom:
    entity_id: climate.m5nanoc6_bed_split_bedroom_split
    control_type: regulating
    proportional_band: 3.0  # °F — full activity when room is this far from target
    sweep_targets: [comfort_min, comfort_mid, comfort_max]  # grid derived from schedule
    sweep_modes: [heat, cool]  # seasonal modes (swept rarely, not per-cycle)
    mode_hold_window: [22, 7]  # hours — no mode changes during this window
    command_encoding: { ... }  # unchanged
    state_encoding: { ... }    # unchanged
```

- `control_type: regulating` — tells the simulator and sweep to use
  target-based control.
- `proportional_band: 3.0` — when `|target - current| >= 3°F`, activity
  saturates at 1.0. Linear ramp below. Configurable, could later be learned.
- `sweep_targets` — which comfort-derived targets to evaluate. Replaces
  the old mode sweep.
- `sweep_modes` — which modes this device supports. Used for mode
  transitions, not per-cycle sweep.
- `mode_hold_window` — hour range during which mode changes are forbidden.
  Only target temperature adjustments are allowed.

Thermostats and blowers don't need changes — their control types are
implicit in their existing config structure.

### 2. Sweep: Target Grid Instead of Mode Sweep

**Current:** `MINI_SPLIT_SWEEP_MODES = ["off", "heat", "cool"]` → 3 options
per split → 9 combos for 2 splits.

**New:** Per-split options derived from comfort schedule at current hour:

```python
def _mini_split_sweep_options(
    split_name: str, schedules: list[ComfortSchedule], base_hour: int,
) -> list[MiniSplitDecision]:
    """Generate sweep options: off + target grid from comfort bounds."""
    options = [MiniSplitDecision(split_name, "off", 0.0)]

    for sched in schedules:
        if sched.room != split_name:
            continue
        comfort = sched.comfort_at(base_hour)
        if comfort is None:
            continue

        mode = "heat"  # TODO: derive from season/outdoor temp
        for target in [comfort.min_temp, (comfort.min_temp + comfort.max_temp) / 2, comfort.max_temp]:
            options.append(MiniSplitDecision(split_name, mode, target))
        break

    return options
```

This gives 4 options per split (off + 3 targets) → 16 combos for 2 splits.
Modest increase from 9 to 16, well within performance budget.

The `mode` field in `MiniSplitDecision` is now derived from context:
- Winter: `"heat"` for all non-off options
- Summer: `"cool"` for all non-off options
- Shoulder season: could use `"heat"` or `"cool"` based on outdoor temp
  vs target (or sweep both → 7 options per split)

### 3. Simulator: Regulating Effector Activity

**Current (pre-computed, constant activity):**
```python
# In _build_activity_matrices():
enc_vals = np.array([encoding.get(sd.mode, 0.0) for ...])
future = np.broadcast_to(enc_vals[:, None], (n, n_future))  # constant!
```

**New (dynamic activity, computed inside Euler loop):**

Skip regulating effectors in `_build_activity_matrices()`. Instead, collect
their parameters and compute activity per Euler step:

```python
# Before the Euler loop, collect regulating effector info per sensor:
regulating_effs: list[tuple[float, float, np.ndarray]] = []
# Each: (gain, lag_steps, target_temps_per_scenario)

# Inside the Euler loop:
for step_idx in range(max_horizon):
    step = step_idx + 1
    dTdt = (outdoor_vec[step_idx] - T) / tau + total_eff[:, step] + solar_vec[step_idx]

    # Add regulating effector contributions (depends on current T)
    for gain, lag_s, targets in regulating_effs:
        if step - lag_s < 1:
            continue
        # Proportional activity: 0 when at target, 1 when proportional_band away
        if gain > 0:  # heating
            activity = np.clip((targets - T) / proportional_band, 0.0, 1.0)
        else:  # cooling
            activity = np.clip((T - targets) / proportional_band, 0.0, 1.0)
        dTdt += abs(gain) * activity

    T = T + _DT_HOURS * dTdt
```

**Performance:** This adds 2 numpy operations per regulating effector per
timestep. With 2 mini splits × 72 steps × 2 ops = 288 numpy calls. Each
is `(n_scenarios,)` shaped — tiny. No measurable performance impact.

**Key property:** Activity naturally drops to zero as room reaches target.
This means the simulator predicts the mini split *stabilizing* the room
at the target temperature, not overshooting. Different targets produce
meaningfully different predictions.

### 4. Mode Stability: Configurable Mode Hold Window

Mode changes (heat→off, off→heat, heat→cool) cause mechanical noise —
louver repositioning, compressor start/stop. Target changes are silent.
The system needs two independent constraints on mode changes:

#### 4a. Time-of-day hold window (configurable per device)

A YAML-configurable `mode_hold_window` defines hours during which mode
changes are forbidden. Only target temperature adjustments are allowed.

```yaml
mini_splits:
  bedroom:
    mode_hold_window: [22, 7]  # no mode changes 10pm–7am
  living_room:
    mode_hold_window: [23, 6]  # different window per device
```

**In the sweep:** During the hold window, constrain options to the current
mode. If the device is in "heat", only "heat" options (with varying targets)
and "off" are available. If the device is "off", it stays "off" until the
window ends.

```python
def _in_hold_window(hour: int, window: tuple[int, int]) -> bool:
    """Check if hour falls within [start, end) window, wrapping midnight."""
    start, end = window
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # wraps midnight

# Before generating mini split options:
for split_name, split_cfg in mini_splits.items():
    if split_cfg.mode_hold_window and _in_hold_window(now.hour, split_cfg.mode_hold_window):
        prev_mode = prev_state.mini_split_modes.get(split_name, "off")
        options = [o for o in options if o.mode == prev_mode]
```

#### 4b. Minimum hold time (global)

Outside the hold window, mode changes are still rate-limited to prevent
rapid cycling:

```python
MODE_HOLD_SECONDS = 2 * 3600  # 2 hours — mode changes are infrequent

# If not in hold window, still enforce minimum hold time:
prev_time = prev_state.mini_split_mode_times.get(split_name)
if prev_time:
    elapsed = (now - parse(prev_time)).total_seconds()
    if elapsed < MODE_HOLD_SECONDS:
        prev_mode = prev_state.mini_split_modes.get(split_name, "off")
        options = [o for o in options if o.mode == prev_mode]
```

**Combined effect:** During quiet hours (10pm–7am), mode is locked — no
compressor starts/stops, only silent target adjustments. Outside quiet
hours, mode can change but no more than once every 2 hours.

**In control state (`ControlState`):**

```python
mini_split_modes: dict[str, str]          # name -> last mode
mini_split_mode_times: dict[str, str]     # name -> ISO timestamp of last mode change
mini_split_targets: dict[str, float]      # name -> last target
```

### 5. Energy Cost

**Current:** Flat `ENERGY_COST_MINI_SPLIT = 0.005` per device when on.

**New:** Proportional to expected activity. Higher target (relative to
outdoor temp) → more energy:

```python
# Approximate average activity over horizon
if sd.mode == "off":
    energy = 0.0
else:
    avg_delta = max(0, sd.target - outdoor_temp) if sd.mode == "heat" else max(0, outdoor_temp - sd.target)
    avg_activity = min(1.0, avg_delta / proportional_band)
    energy = ENERGY_COST_MINI_SPLIT * avg_activity
```

This makes the optimizer prefer lower targets (less energy) when comfort
allows it, which is the correct physical trade-off.

### 6. Auto Mode Considerations

Mini splits support an "auto" mode (`heat_cool`) that automatically heats
or cools to reach the target. This seems attractive but conflicts with the
comfort model:

**Problem with auto mode:** The mini split will actively heat *and* cool
to maintain the exact target temperature. Within a comfort band (e.g.,
68–72°F), we'd typically prefer to let the room temperature wander naturally
rather than spending energy to pin it at one value. Auto mode treats any
deviation from target as an error to correct, which wastes energy on
corrections that aren't needed for comfort.

**Example:** Comfort band is 68–72°F, target is 70°F. With "heat" mode,
if the room drifts to 71°F the mini split idles (good — it's in band).
With "auto" mode, it would start cooling to pull the room back to 70°F
(wasteful — 71°F is comfortable).

**When auto mode could be useful:**
- Very tight comfort bands where any deviation matters (e.g., sleeping with
  min=68, max=69 — effectively pinning temperature)
- Shoulder seasons where outdoor temp is near the comfort range and the
  room might need heating in the morning and cooling in the afternoon

**Design decision:** For now, auto mode is excluded from the sweep. The
mode is determined by season/context (heat in winter, cool in summer), and
the proportional activity model handles the regulation. The `command_encoding`
already includes `auto: 0.5` and `heat_cool: 0.5` for backward compatibility
with sysid training data, but the sweep does not generate auto-mode options.

**Future extension:** If shoulder-season data shows that mode needs to flip
within a day (morning heat, afternoon cool), we could:
1. Add auto mode to `sweep_modes` for specific seasons
2. Model auto mode in the simulator as bidirectional proportional activity
3. Add an energy penalty multiplier for auto mode (since it corrects in
   both directions)

This is deferred until we have spring/summer data to validate against.

---

## Implementation Order

### Step 1: Config changes (low risk)
- Add `control_type: regulating`, `proportional_band: 3.0`, and
  `mode_hold_window: [22, 7]` to mini split YAML definitions.
- Parse in `yaml_config.py` — add fields to `MiniSplitYamlConfig`.
- Add `sweep_targets` config (or derive from comfort schedule).
- No behavioral change yet.

### Step 2: Sweep target grid (medium risk)
- Replace `MINI_SPLIT_SWEEP_MODES` with target-grid generation.
- `MiniSplitDecision.target` becomes meaningful during sweep (not post-hoc).
- `MiniSplitDecision.mode` derived from season/context.
- Remove post-sweep target derivation (`final_splits` logic).
- Tests: verify scenario count, verify all-off baseline unchanged.

### Step 3: Simulator regulating model (medium risk)
- Skip regulating effectors in `_build_activity_matrices()`.
- Collect regulating effector params (gain, lag, proportional_band, targets).
- Compute proportional activity inside the Euler loop.
- Tests: verify predictions change with different targets, verify
  temperature approaches target and stabilizes (doesn't overshoot).

### Step 4: Mode hold window + minimum hold time (low risk)
- Add `mini_split_mode_times` to `ControlState`.
- Implement `_in_hold_window()` check using per-device config.
- Constrain sweep options when in hold window or within minimum hold time.
- Target changes still happen every 15 minutes.
- Tests: verify mode doesn't change during hold window, verify minimum
  hold time is respected outside the window.

### Step 5: Energy cost refinement (low risk)
- Make mini split energy cost proportional to expected activity.
- Tests: verify optimizer prefers lower targets when comfort allows.

---

## Files Changed

| File | Step | Change |
|------|------|--------|
| `weatherstat.yaml` | 1 | `control_type`, `proportional_band`, `mode_hold_window` on mini splits |
| `ml/src/weatherstat/yaml_config.py` | 1 | Parse new fields |
| `ml/src/weatherstat/config.py` | 2 | Replace `MINI_SPLIT_SWEEP_MODES` with target grid |
| `ml/src/weatherstat/control.py` | 2,4,5 | Target sweep, mode hold window, energy cost |
| `ml/src/weatherstat/simulator.py` | 3 | Regulating activity in Euler loop |
| `ml/src/weatherstat/types.py` | 4 | `ControlState.mini_split_mode_times` |
| `ml/tests/test_simulator.py` | 3 | Test target-based predictions |
| `ml/tests/test_control.py` | 2,4 | Test target sweep, mode hold window |

---

## Verification

1. **Unit tests:** New tests for regulating activity model (temperature
   approaches target, stabilizes, different targets give different predictions).
2. **Scenario count:** Verify ~16 mini split combos (vs previous 9), total
   scenarios still < 15K, sweep < 100ms.
3. **Control loop:** `just control` — verify mini split decisions show
   target temperatures, mode is stable across consecutive runs.
4. **Sysid compatibility:** Existing `thermal_params.json` gains work
   unchanged — the proportional model is compatible with binary-learned gains.
5. **Live test:** Run `just control-live` overnight, verify no mode cycling.
   Target should adjust quietly every 15 min. Mode locked during hold window.

---

## What This Doesn't Change

- Thermostat control (trajectory search) — unchanged.
- Blower control (discrete levels) — unchanged.
- Sysid gain learning — the gains represent °F/hr at full activity (1.0),
  which is exactly what the proportional model uses at saturation.
- Advisory system — unchanged (evaluates window toggles, not split targets).
- Executor structure — `applyMiniSplit()` already sets mode + target.
  The change is that mode changes become rare, target changes become the norm.
