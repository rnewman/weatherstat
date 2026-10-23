# PLAN-38: Weatherstat as a Home Assistant device

## Problem

Stage 37 made "bedroom is too cold" a recorded request, reachable from the
CLI, the TUI and the phone web page. The place people actually look at the
house is the Home Assistant dashboard, and nothing there could express the
request or show what Weatherstat knows: comfort bands, feels-like
temperature, where each room is headed, and why the controller is doing
what it does.

## Options considered

| Approach | Why not |
|---|---|
| `input_number` helpers polled by Weatherstat | Two sources of truth with a sync protocol (expiry, restarts, who wins); latency of the poll; overlays lose their recorded context. |
| `rest_command` → web endpoint | YAML-only in HA; HA must reach the Weatherstat host; the web server has no auth. |
| Custom integration (`custom_components/`) | Code to deploy into HA's Python environment and an HA restart per change; HA polls Weatherstat. |
| **MQTT discovery** | Chosen. Weatherstat owns the entity definitions; push in both directions; no HA restart; nothing installed on the HA box; the standard pattern for external services. |

## Design

`mqtt_bridge.py` connects to the broker (default: the host of `HA_URL`, i.e.
the Mosquitto app), with a last will on `weatherstat/status` so every entity
goes unavailable when Weatherstat stops.

**Comfort controls** (TUI and headless): per constrained sensor a `number`
(the overlay) and a timestamp `sensor` (its expiry). A number value is
absolute: setting +2 is `OverlayCommand(set, +2)`, 0 is a clear. Commands are
echoed immediately (HA updates at once) and applied after a per-room debounce
(`mqtt.debounce_seconds`, 2 s), so three taps of + are one recorded "set +3",
one event and one control cycle. Every change goes through
`overlay.apply_overlay_command` (the single write path, now also used by the
TUI controller), so HA requests are recorded with source `ha` and the same
context as any other entry point. State is republished every 30 s from
`overlays.json`, which carries CLI changes and expiries to HA without extra
plumbing.

**Status** (TUI only): the TUI's `StatusSnapshot` is published on every
refresh as one `sensor.weatherstat_<room>` per room (state: feels-like;
attributes: bands, overlay, predictions, cost, status, source sensor) and
`sensor.weatherstat_status` (state: live / dry run; attributes: effector plan
with rationale, costs, requests, advisories, health). Unchanged payloads are
skipped, ignoring timestamps.

**Wiring.** The TUI starts the bridge on mount when `mqtt:` is configured,
applies HA commands through the controller (which kicks a cycle and notifies
the bridge through an overlay listener), and publishes each snapshot. The
headless loop applies commands directly and wakes its sleep for an immediate
cycle.

**Cards.** `ha/weatherstat-cards.js` (room, rooms list, status) reads only
these entities and needs no configuration beyond a room label. It is
installed as an inline `data:` module resource by `just ha-cards`, so a new
version is a new URL (no stale cache) and nothing is copied to the HA box.
The room card re-bases the published bands on the current overlay so a −/+
moves the band before Weatherstat's next snapshot, and it caches attributes
so an offline room still shows its HA temperature.

## Verification

- `tests/test_mqtt_bridge.py` (26 tests): topics, discovery payloads and
  unique IDs, payload parsing and clamping, overlay/room/status payloads,
  dedupe ignoring volatile fields, debounce (last press wins, rooms
  independent), bad commands, apply failure resyncing HA, snapshot sent on
  (re)connect, config parsing.
- `tests/test_tui_controller.py::TestOverlayListeners`.
- Live: discovery and state messages generated from a real snapshot and
  published through HA's `mqtt.publish` created the expected entities
  (HA 2026.10, `default_entity_id`); cards rendered against them on the
  Living and Connect dashboards.
