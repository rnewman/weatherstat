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

We use outdoor temperature as a proxy for the MRT offset, since we
don't have surface temperature sensors. The correction is **sun-aware**:
current solar forcing raises the effective outdoor temperature per sensor,
reducing cold-wall correction on sunny days while preserving it on cloudy
days and at night.

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

Then the MRT offset is:

```
offset = clamp(alpha × (reference_temp - effective_outdoor), -max_offset, +max_offset) × mrt_weight
```

This shifts all comfort targets (preferred, min, max) uniformly:
- **Cold cloudy day**: effective_outdoor ≈ outdoor (no solar warming) →
  full cold-wall correction
- **Cold sunny day**: effective_outdoor > outdoor → reduced correction
  (sun heats interior surfaces through windows)
- **Night**: sin⁺(elev) = 0 → effective_outdoor = outdoor → same as
  temperature-only correction
- **Reference temp**: zero correction — this is the outdoor temperature
  at which the comfort schedule values were tuned to feel right

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

**Why static, not per-horizon?** Wall surfaces have high thermal mass.
Even if the forecast says it'll warm from 35°F to 55°F in 6 hours, the
walls won't catch up for much longer. The current outdoor temp is more
representative of current wall state than any forecast.

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

**Why shift all targets uniformly?** The min/max bounds are comfort
limits, not safety limits. If 71°F feels like 69.5°F on a cold day,
the acceptable range should shift too, not just the target.

## References

- ASHRAE Standard 55 — Thermal Environmental Conditions for Human
  Occupancy (operative temperature, PMV/PPD model)
- ISO 7730 — Ergonomics of the thermal environment
- Fanger, P.O. (1970) — Thermal Comfort: Analysis and Applications
  in Environmental Engineering
