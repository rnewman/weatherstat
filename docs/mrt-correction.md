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
don't have surface temperature sensors:

```
offset = clamp(alpha × (reference_temp - outdoor_temp), -max_offset, +max_offset)
```

This shifts all comfort targets (preferred, min, max) uniformly:
- **Cold day**: offset is positive → targets rise → the system heats
  more aggressively to compensate for cold walls
- **Hot day**: offset is negative → targets drop → less heating needed
  because warm walls contribute to comfort
- **Reference temp**: zero correction — this is the outdoor temperature
  at which the comfort schedule values were tuned to feel right

### Parameters

| Parameter        | Default | Meaning |
|-----------------|---------|---------|
| `alpha`          | 0.1     | °F of comfort shift per °F of outdoor deviation |
| `reference_temp` | 50      | Outdoor temp where current targets feel right (°F) |
| `max_offset`     | 3.0     | Maximum correction magnitude (°F) |

### Tuning

- **`alpha`**: Start at 0.1. Increase if rooms still feel cold on cold
  days despite hitting target temps. Decrease if the correction
  overshoots (too warm on cold days).
- **`reference_temp`**: The outdoor temperature at which your current
  comfort schedules feel right. For Seattle (where schedules were tuned
  in winter at ~45–55°F), 50°F is a reasonable starting point.
- **`max_offset`**: Caps extreme corrections. At alpha=0.1, this caps
  at ±30°F outdoor deviation from reference. 3°F is conservative.

## Design Choices

**Why static, not per-horizon?** Wall surfaces have high thermal mass.
Even if the forecast says it'll warm from 35°F to 55°F in 6 hours, the
walls won't catch up for much longer. The current outdoor temp is more
representative of current wall state than any forecast.

**Why not per-room?** Actually, it is per-room now. Each sensor gets a
`mrt_weight` multiplier that scales the global offset. Two sources:

1. **Manual** (`mrt_weight` in YAML constraint schedule): explicit
   override for rooms where you know the solar exposure. Default 1.0.
2. **Derived** (from sysid solar gain profiles): sensors with high
   daytime solar gains get lower weight (sun warms surfaces, reducing
   MRT correction need), sensors with zero solar gains get higher
   weight. Stored in `thermal_params.json` as `mrt_weights`.

Priority: manual weight wins if set != 1.0, otherwise derived weight
is used. If neither is set, weight defaults to 1.0.

**Why shift all targets uniformly?** The min/max bounds are comfort
limits, not safety limits. If 71°F feels like 69.5°F on a cold day,
the acceptable range should shift too, not just the target.

## References

- ASHRAE Standard 55 — Thermal Environmental Conditions for Human
  Occupancy (operative temperature, PMV/PPD model)
- ISO 7730 — Ergonomics of the thermal environment
- Fanger, P.O. (1970) — Thermal Comfort: Analysis and Applications
  in Environmental Engineering
