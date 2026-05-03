# Mean Radiant Temperature (MRT) Correction

## The Problem

A thermometer reads air temperature, but humans feel **operative
temperature** — a blend of air temperature and the temperature of
surrounding surfaces (walls, windows, floors, ceiling). The standard
approximation for typical indoor air speeds is:

```
T_operative ≈ (T_air + T_mrt) / 2
```

where T_mrt is the **mean radiant temperature**: the area-weighted
average of all surface temperatures visible to the occupant. When
exterior surfaces are cold, T_mrt drops below T_air, and the space
feels colder than the thermometer says. The reverse happens in summer.

This matters for comfort control. A thermostat reading 71°F on a cold
rainy day doesn't feel the same as 71°F in August. The difference is
typically 1–3°F of perceived shift.

## Why Surfaces Differ from Air

A wall's inner surface temperature depends on:
- **Outdoor temperature** (drives heat loss through the envelope)
- **Wall R-value** (insulation reduces the difference)
- **Window area** (windows have much lower R-value than walls)

For a typical insulated wall (R-13), the inner surface sits about
0.05 × (T_indoor - T_outdoor) below indoor air temperature. At
T_indoor = 71°F and T_outdoor = 35°F, that's ≈1.8°F below air temp.

Windows are worse. A double-pane window might have an inner surface
at roughly T_air - 0.3 × (T_air - T_outdoor). At 35°F outside, a
window's inner surface could be 50–55°F — dramatically pulling down
the MRT for anyone near it.

The floor matters too: hydronic floor heat produces warm floor
surfaces (a comfort bonus), while an unheated slab is cold.

## Putting It Together

For a room with 25% exterior walls, 15% windows, 30% floor, and 30%
ceiling/interior:

| Surface       | Area | Temp (35°F outside) | Temp (80°F outside) |
|--------------|------|-------------------|-------------------|
| Exterior walls | 25%  | 69°F               | 72°F               |
| Windows        | 15%  | 52°F               | 77°F               |
| Floor (heated) | 30%  | 73°F               | 71°F               |
| Interior       | 30%  | 71°F               | 71°F               |
| **MRT**        |      | **68°F**           | **72.4°F**         |
| **Operative**  |      | **69.5°F**         | **71.7°F**         |

The same 71°F air temperature produces 69.5°F operative on a cold day
vs 71.7°F on a hot day — a 2.2°F perceived difference.

## The Correction

MRT correction converts predicted air temperatures into **feels-like
(operative) temperatures** before comparing to comfort bounds. Comfort
bounds stay at user-configured values — only predictions are adjusted.

### Sign Convention

```
offset = clamp(alpha × (reference_temp - effective_outdoor), -max, +max) × mrt_weight
feels_like = air_temp - offset
```

- **Positive offset** (cold outside): air feels colder than measured →
  `feels_like < air_temp`. The system heats more aggressively.
- **Negative offset** (hot outside): air feels warmer than measured →
  `feels_like > air_temp`. The system cools more aggressively.
- **Zero offset** (at reference temp): configured comfort bounds are
  exactly right.

### Math Equivalence

The previous implementation shifted comfort bounds up by the offset.
The current approach shifts predictions down by the same offset. These
are mathematically equivalent:

```
pred > (bound + offset)   ↔   (pred - offset) > bound
```

The prediction-correction approach has two advantages: (1) comfort
bounds in logs and display match user-configured values (no confusion
about shifted numbers), and (2) the offset can vary per prediction
horizon.

### Sun-Aware Effective Outdoor Temperature

The correction is **sun-aware**: current solar forcing raises the
effective outdoor temperature per sensor, reducing cold-wall correction
on sunny days while preserving it on cloudy days and at night.

Per sensor, the effective outdoor temperature is:

```
effective_outdoor = outdoor_temp + β_solar(sensor) × sin⁺(elev) × weather_fraction × solar_response
```

where:
- `β_solar(sensor)` is the sysid-fitted solar elevation gain for this
  sensor (higher for rooms with large south/west windows)
- `sin⁺(elev)` is max(0, sin(solar_elevation)) — 0 at night, ~0.7 at
  noon in Seattle
- `weather_fraction` is the condition-derived solar fraction (1.0 sunny,
  0.15 overcast)
- `solar_response` is a tunable scale factor (default 2.0) mapping the
  air-temperature solar gain to wall-surface MRT effect

Effects by condition:
- **Cold cloudy day**: effective_outdoor ≈ outdoor (no solar warming) →
  full cold-wall correction
- **Cold sunny day**: effective_outdoor > outdoor → reduced correction
  (sun heats interior surfaces through windows)
- **Night**: sin⁺(elev) = 0 → effective_outdoor = outdoor → same as
  temperature-only correction
- **Reference temp**: zero correction — this is the outdoor temperature
  at which the comfort schedule values were tuned to feel right

### Per-Horizon Correction

Each prediction horizon uses that horizon's forecast outdoor temperature,
solar elevation, and solar fraction to compute the MRT offset. This
captures temporal transitions that a static correction would miss:

- **Afternoon → evening**: Solar MRT relief fades as the sun sets. The
  1h-out offset may include some solar benefit; the 4h-out offset after
  sunset does not.
- **Overnight cooling**: Outdoor temp drops through the night. Later
  horizons have larger cold-wall offsets, making the system more
  aggressive about pre-heating.
- **Morning warmup**: As the sun rises and outdoor temp increases, later
  horizons have smaller offsets — less correction needed.

Implementation: `compute_horizon_mrt_offsets()` returns a nested dict
`{sensor_col: {horizon_step: offset}}` mapping each sensor and horizon
to its specific MRT offset. The comfort scorer (`_batch_comfort_cost`)
applies per-entry offsets to predicted temperatures before comparing
to uncorrected bounds.

### Regulating Effector Targets

Regulating effectors (mini splits) sweep target temperatures from
comfort schedule bounds. These targets are **air temperatures** — the
physical setpoint the device aims for. But the user's comfort preference
is about **operative (perceived) temperature**.

If the user wants to feel 72°F and MRT says cold walls make 72°F air
feel like 70.5°F, the device must target 73.5°F air to achieve 72°F
perceived:

```
target_air = preferred_operative + mrt_offset
```

Without this adjustment, the sweep can only offer "off" and "heat@72"
as candidates. Scoring penalizes 72°F air (which feels like 70.5°F)
for being below preferred, but no better option exists — the optimizer
can't discover 73.5°F if it's not in the candidate set. The result:
the mini split heats to 72°F, the room feels cold, and the optimizer
reports comfort cost it can't reduce.

Regulating targets use the **current-moment** MRT offset (not
per-horizon), because the target temperature is set immediately and
held for the sweep duration.

### Parameters

| Parameter        | Default | Meaning |
|-----------------|---------|---------|
| `alpha`          | 0.1     | °F of comfort shift per °F of outdoor deviation |
| `reference_temp` | 50      | Outdoor temp where current targets feel right (°F) |
| `max_offset`     | 3.0     | Maximum correction magnitude (°F) |
| `solar_response` | 2.0     | Scale factor: solar air-temp gain → MRT surface effect |

### Tuning

- **`alpha`**: Start at 0.1. Increase if rooms still feel cold on cold
  days despite hitting target temps. Decrease if the correction
  overshoots (too warm on cold days).
- **`reference_temp`**: The outdoor temperature at which your current
  comfort schedules feel right. For Seattle (where schedules were tuned
  in winter at ~45–55°F), 50°F is a reasonable starting point.
- **`max_offset`**: Caps extreme corrections. At alpha=0.1, this caps
  at ±30°F outdoor deviation from reference. 3°F is conservative.
- **`solar_response`**: How much the sysid solar gain translates to
  MRT surface warming. At 2.0, a sensor with β_solar=3.0 on a sunny
  noon (sin⁺≈0.7, SF=1.0) gets +4.2°F added to effective outdoor temp.
  Increase if sunny cold days still feel too cold despite the correction.

## Design Choices

**Why correct predictions, not bounds?** Two reasons. First, comfort
bounds in logs and display match user-configured values — when a user
sets backup_hi to 77°F, messages and plots show 77°F, not 76.5°F.
Second, prediction correction naturally supports per-horizon offsets:
each prediction step can use that step's forecast conditions, capturing
transitions like sunset or overnight cooling that a static bound shift
would miss.

**Why sun-aware?** A sunny 35°F day and a cloudy 35°F day produce
very different interior surface temperatures. Sun streaming through
windows heats walls, floors, and furniture — raising MRT well above
what outdoor air temperature alone would predict. The greenhouse
effect through glass is the dominant factor in surface temperature
variation at a given outdoor temp. Without the solar adjustment, the
system would over-correct (heat too aggressively) on sunny cold days.

**Per-room differentiation:** The correction is naturally per-sensor
because `β_solar` varies by sensor — rooms with large south-facing
windows (high sysid solar gain) get more solar MRT relief than
interior rooms (low or zero solar gain). Additionally, each sensor
can have a manual `mrt_weight` multiplier in the YAML constraint
schedule (default 1.0) for non-solar adjustments.

## Data Flow

```
Forecast data (outdoor_temp, solar_elevation, weather_fraction per horizon)
  ↓
compute_horizon_mrt_offsets()  →  {sensor: {horizon: offset}}
compute_current_mrt_offsets()  →  {sensor: offset}
  ↓
Comfort schedules: UNCORRECTED (user-configured values)
  ↓
Regulating sweep targets: preferred + current_mrt_offset[sensor]
Cold-room override: (current_temp - offset) vs acceptable_lo
  ↓
Scoring: _ComfortSpec.build(schedules, mrt_offsets=horizon_offsets)
         → _batch_comfort_cost: feels_like = temps - mrt_offsets
         → compare feels_like to uncorrected bounds
  ↓
Backup breaches: (pred - horizon_offset) vs backup bounds
  ↓
Display: raw air temp + offset shown; bounds at configured values
```

## Implementation

Three functions in `control.py`:

- `compute_mrt_offset(outdoor_temp, mrt_config, ...)` — single-sensor,
  single-moment offset.
- `compute_current_mrt_offsets(sensors, outdoor_temp, mrt_config, ...)`
  — per-sensor offsets at current conditions. Used for regulating
  targets, cold-room override, and display.
- `compute_horizon_mrt_offsets(sensors, mrt_config, outdoor_temp,
  forecast_temps, solar_elevations, solar_fractions, ...)` — per-sensor,
  per-horizon offsets using forecast data. Used for comfort scoring.

## References

- ASHRAE Standard 55 — Thermal Environmental Conditions for Human
  Occupancy (operative temperature, PMV/PPD model)
- ISO 7730 — Ergonomics of the thermal environment
- Fanger, P.O. (1970) — Thermal Comfort: Analysis and Applications
  in Environmental Engineering
