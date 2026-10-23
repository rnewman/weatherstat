# Weatherstat in Home Assistant

Weatherstat publishes itself to Home Assistant as an MQTT-discovered device,
and `weatherstat-cards.js` renders it. Nothing is installed on the HA box: the
entities come from MQTT discovery and the cards are an inline dashboard
resource.

## Setup

1. **Broker.** Install the Mosquitto broker app and the MQTT integration in HA.
   Create a Home Assistant user for Weatherstat (Settings → People → Users;
   a non-admin user is enough, since Mosquitto accepts HA users).
2. **Credentials.** Add to `~/.weatherstat/.env`:

   ```
   MQTT_USERNAME=weatherstat
   MQTT_PASSWORD=...
   ```

3. **Enable the bridge.** Add `mqtt: {}` to `weatherstat.yaml` (the broker host
   defaults to the host of `HA_URL`; see `weatherstat.yaml.example` for options).
   Restart the TUI (`just tui --live --web`) or the headless loop
   (`just control --live --loop`). Both connect on startup.
4. **Cards.** `just ha-cards` registers `weatherstat-cards.js` as a dashboard
   resource (re-run it after editing the file). Reload the dashboard.

## Entities

For every constrained sensor (`<room>` is its label from `weatherstat.yaml`):

| Entity | What it is |
|---|---|
| `number.weatherstat_<room>_comfort` | The room's comfort overlay, −max…+max. Setting it applies an absolute overlay for `overlay_duration_hours`; 0 clears it. −/+ presses are debounced (2 s) into one recorded request. |
| `sensor.weatherstat_<room>_comfort_until` | When that overlay expires (timestamp; unknown when none). |
| `sensor.weatherstat_<room>` | Feels-like temperature (air minus MRT offset). Attributes: comfort band (preferred/acceptable, overlay included), overlay, predictions by horizon, comfort cost vs all-off, status (cold/cool/ok/warm/hot), and the HA sensor it reads. |
| `sensor.weatherstat_status` | `live` / `dry run`. Attributes: profile, next cycle, effector plan with per-device rationale, costs, comfort requests, advisories, warnings, data and model age. |

The room and status sensors come from the TUI's status snapshot, so the TUI
publishes them; the headless loop publishes only the comfort controls. All
entities go unavailable when Weatherstat stops (MQTT last will).

The number entities work anywhere HA does: voice ("set bedroom comfort to
2"), automations, or a plain tile card with the `numeric-input` feature.

## Cards

```yaml
type: custom:weatherstat-room-card      # one room
room: bedroom
name: Bedroom                           # optional
source_entity: sensor.bedroom_temp      # optional: shows the HA temperature even while Weatherstat is offline
horizon: 2h                             # optional: which prediction to draw (1h, 2h, 4h, 6h, 12h)
```

```yaml
type: custom:weatherstat-rooms-card     # every room in one compact list
title: Rooms                            # optional
rooms:                                  # optional; default: all Weatherstat rooms, alphabetical
  - bedroom
  - room: living_room_climate
    name: Living room
```

```yaml
type: custom:weatherstat-status-card    # plan, rationale, requests, advisories, health
```

The room bar shows the acceptable range (light), the preferred band (solid),
the feels-like temperature (dot, coloured by status), the air temperature (thin
tick, when the MRT offset matters), and where the room is headed (arrow at the
chosen horizon). With an overlay active, a dashed outline marks the band the
schedule would otherwise ask for. Tapping the room opens the HA sensor's
history.

Colours follow the HA theme; override with `--weatherstat-{cold,cool,ok,warm,hot}-color`.
