# PLAN-37: Comfort overlays — "bedroom is too cold"

## Problem

The configured schedules and profiles express what the house should be like
in general. They say nothing about *right now*: a guest in the bedroom, a
cold evening on the sofa, a workout in the gym. The only way to express
"this room is wrong at the moment" was to edit `weatherstat.yaml` (and
remember to undo it) or to override a device by hand, which the executor
treats as a manual override and the system can only wait out.

Both lose information. A request like "too cold" is exactly the feedback a
learned comfort model needs, and it was being thrown away.

## Design

### Overlay = per-sensor signed delta with an expiry

```
base schedules → profile offsets → overlays → (MRT applied to predictions)
```

A `ComfortOverlay(sensor, delta, created_at, expires_at, source, note)`
shifts **all three tiers** of that sensor's comfort uniformly for the hours
it is active:

- Shifting only `preferred` would clip against the unchanged `acceptable`
  rail: a +2 on a `[71,73] / 70–74` schedule becomes `[73,75]` with a steep
  penalty above 74, and the optimizer parks the room at 74. Shifting the
  rails too makes the request unambiguous: *move the room*.
- Shifting `backup` keeps the worst-case hedge consistent with the request.

Positive = warmer ("too cold"), negative = cooler. |delta| is capped
(`defaults.overlay_max_delta`, default 5°F); duration defaults to
`defaults.overlay_duration_hours` (3h). Repeated bumps **accumulate** and
refresh the expiry; `=±N` sets an absolute delta; `clear` removes.

### Expiry-aware at hour granularity

Every consumer reads comfort through `ComfortSchedule.comfort_at(hour)`, so
hours are the schedule's native resolution. `apply_overlays` computes the
local clock hours touched by `[now, expires_at)` and `shift_schedule_hours`
rebuilds the schedule from a 24-slot table: shifted slots for those hours,
the original comfort elsewhere, coalesced back into entries (merging across
midnight into a wrapping entry). The sweep therefore sees the bounds revert
mid-horizon — a 2-hour bump does not make the optimizer plan a 6-hour heat.
A bump of a day or more shifts every hour.

Because it is applied to the schedules themselves, nothing downstream had
to change: comfort scoring, regulating-effector targets
(`target_air = preferred + mrt_offset`), the cold-room override, per-sensor
cost breakdowns, the decision log's `comfort_bounds`, and the TUI bars all
follow.

### Recording

- `~/.weatherstat/overlays.json` — current overlays (one per sensor).
- `overlay_events` table in `decision_log.db` — one row per
  set / bump / clear / expire: resulting delta, the increment requested,
  expiry, source (`cli` / `tui` / `web`), free-text note, and a context
  JSON with the room temperature, outdoor temperature, local hour and
  weekday at request time (and the active profile when known). Indexed by
  timestamp and (sensor, timestamp). **Never pruned by retention.**
- `decisions.overlays` — `{label: delta}` active when each decision was
  made, so any decision can be joined to the request that shaped it.

### Entry points

| Surface | How | Notes |
|---|---|---|
| CLI | `just overlay bedroom +2`, `just overlay bedroom -1 --hours 4 --note "guest"`, `just overlay bedroom clear`, `just overlay --history` | Picked up on the next control cycle (≤ 5 min in `--loop`). |
| TUI | `O` opens a prompt: `bedroom +2`, `office -1 90m`, `bedroom =+3 4h`, `bedroom clear` | Kicks a control cycle immediately. Temperature panel shows `+2.0 2h40m` in cyan. |
| Web | `−` / `+` per room (one step, `defaults.overlay_step`), `×` to clear; `POST /action/overlay sensor=… delta=… [hours=…]` | Phone-first: you are in the room that is wrong. Also a one-line HA `rest_command` target. |

Room names resolve from the constraint label, the sensor column, or a unique
prefix (`bed` → bedroom). Bare numbers are rejected (`bedroom 72` is
ambiguous) — a sign is required.

### Observing

`just debug overlays [N]` prints active overlays, the last N events with
their context, and a per-room **habit summary**: request count, how many
warmer vs cooler, mean delta, and which hour bands (night / morning / day /
evening) the requests fall in. That table is the first thing a schedule
proposal would be built from.

## What this is not

- Not a setpoint override. The user never talks to a device; the request is
  about a room and the optimizer decides which effectors to use.
- Not a schedule edit. Overlays expire; the configured schedule is still the
  statement of intent. Learning from overlays should *propose* schedule
  changes (FUTURE.md § "Learning From Comfort Overlays"), not make them.
- Not an HA entity (yet). Voice / automation entry can come through the web
  endpoint; a native `input_number` per room was deferred until the request
  patterns are known.

## Verification

- `tests/test_overlay.py` (38 tests): parsing, resolution, store semantics
  (accumulate, set, clamp, net-zero clear, persistence, prune + expire
  events, event context), hour windows across midnight and time zones,
  lossless schedule round-trip, tier shifting only in active hours, entry
  coalescing with a wrapping entry, label serialization.
- `tests/test_tui_controller.py::TestOverlays`: bump → snapshot bounds shift
  for that room only, command + clear, error path, cycle trigger.
- `tests/test_web.py`: `POST /action/overlay` bump / hours / clear / bad
  input; buttons rendered per room; badge + `×` + JSON when active.
