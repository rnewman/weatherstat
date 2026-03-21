# PLAN-16: MRT (Mean Radiant Temperature) Comfort Correction

## Context

Measured air temperature doesn't capture thermal comfort. Cold exterior
walls and windows absorb occupant IR radiation, making a room feel colder
than the thermometer reads. The reverse happens in summer when sun-heated
walls re-emit IR. 71°F in the office on a cold rainy day feels very
different to 71°F in August.

The standard metric is **operative temperature** ≈ (T_air + T_mrt) / 2.
MRT depends on wall surface temps, which track outdoor conditions. For
typical insulated walls + windows, the net effect is 1–3°F of perceived
temperature shift between winter and summer at the same air reading.

## Approach

Use outdoor temperature as a proxy for MRT offset. Apply a correction to
comfort schedule targets: raise them when it's cold outside (compensate
for cold walls), lower them when it's hot (walls radiate warmth):

```
offset = clamp(alpha × (reference_temp - outdoor_temp), -max_offset, +max_offset)
```

This fits naturally into the existing comfort pipeline as another schedule
adjustment, alongside `apply_comfort_profile()` and
`adjust_schedules_for_windows()`.

**Pipeline order:**
```
base schedules → profile offsets → MRT correction → window adjustments
```

## YAML Config

Under `constraints:`, after `window_open_offset`:

```yaml
  mrt_correction:
    alpha: 0.1           # °F comfort shift per °F outdoor deviation
    reference_temp: 50   # outdoor temp where current comfort targets feel right
    max_offset: 3.0      # cap correction magnitude (°F)
```

Optional — if absent, no correction is applied.

**Example effects** (alpha=0.1, ref=50):
- 35°F outside → +1.5°F (raise targets, compensate for cold walls)
- 50°F outside → 0.0°F (reference point, targets as configured)
- 80°F outside → -3.0°F (capped, lower targets, walls already warm)

## Design Decisions

**Static correction, not per-horizon**: Wall surface temps have high
thermal mass — current outdoor temp better represents current wall state
than forecast temps hours ahead. One offset per control cycle is correct.

**Uniform offset to preferred/min/max**: Same shift to all three. The
alpha is small enough (typical 1–2°F) that this doesn't cause problems
with bounds. One knob to tune.

**Global alpha, not per-sensor**: Different rooms have different window
ratios, but per-room alpha is premature. Single global alpha to start.

## Implementation

### `yaml_config.py`

`MrtCorrectionConfig` frozen dataclass with `alpha`, `reference_temp`,
`max_offset`. Optional field on `WeatherstatConfig` (None if unconfigured).
Parsed from `constraints.mrt_correction` in YAML.

### `control.py`

`apply_mrt_correction(schedules, outdoor_temp, config)` computes clamped
offset, shifts all comfort entries' preferred/min/max uniformly. Wired
into `run_control_cycle()` between profile offsets and window adjustments.
Uses the current sensor outdoor temp (met.no or side sensor) — appropriate
because wall surfaces track current conditions, not forecast.

### Tests

6 unit tests: cold day (+1.5°F), warm day (clamped -3°F), at reference
(no change), extreme cold (clamped +3°F), None config (no-op), penalty
preservation.

## Files Changed

| File | Change |
|---|---|
| `ml/src/weatherstat/yaml_config.py` | `MrtCorrectionConfig` dataclass, field, parsing |
| `ml/src/weatherstat/control.py` | `apply_mrt_correction()`, wiring in control loop |
| `weatherstat.yaml.example` | `mrt_correction` config under constraints |
| `~/.weatherstat/weatherstat.yaml` | Same |
| `ml/tests/test_control.py` | `TestMrtCorrection` class (6 tests) |
