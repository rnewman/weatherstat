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
| `sensor.weatherstat_<effector>_input` | Per regulating effector (mini split): the temperature it should regulate on, a weighted mean of the rooms it affects (weights from its learned gains). Published every minute; unavailable when Weatherstat stops, when no update arrives for 3 minutes, or when too few of those rooms report. Attributes: weights, member readings, dropped members, coverage. See [Mini split inputs](#mini-split-inputs). |
| `sensor.weatherstat_status` | `live` / `dry run`. Attributes: profile, next cycle, effector plan with per-device rationale, costs, comfort requests, advisories, warnings, data and model age. |

The room and status sensors come from the TUI's status snapshot, so the TUI
publishes them; the headless loop publishes only the comfort controls. All
entities go unavailable when Weatherstat stops (MQTT last will).

The number entities work anywhere HA does: voice ("set bedroom comfort to
2"), automations, or a plain tile card with the `numeric-input` feature.

## Mini split inputs

A mini split runs its own thermostat on whatever temperature it reads, so
Weatherstat publishes `sensor.weatherstat_<effector>_input` for each regulating
effector (enabled with the `mqtt:` block). Pointing the unit at it makes the unit
regulate on the rooms it actually heats and cools, weighted by its learned
effect on each. Weatherstat plans its setpoints on that input's scale.

1. **Check the input.** `just debug regulation` shows each unit's weights, the
   live input, and the unit's own reading.
2. **Point the unit at it.** For a Mitsubishi unit on an ESPHome
   [CN105](https://github.com/echavet/MitsubishiCN105ESPHome) board:

   ```yaml
   sensor:
     - platform: homeassistant
       id: remote_temp
       name: "Remote Temperature Sensor"
       entity_id: sensor.weatherstat_mini_split_bedroom_input
       unit_of_measurement: "°C"
       filters:
         - lambda: return (x - 32) * (5.0/9.0);   # only if HA reports °F
       on_value:
         then:
           - lambda: |-
               if (std::isfinite(x) && x >= 1 && x <= 40) id(hp).set_remote_temperature(x);
               else id(hp).set_remote_temperature(0);  // unavailable: the unit's own thermistor

   interval:
     - interval: 60s
       then:
         - if:
             condition:
               api.connected:
             then:
               - lambda: |-
                   const float t = id(remote_temp).state;
                   if (std::isfinite(t) && t >= 1 && t <= 40) id(hp).set_remote_temperature(t);
   ```

   - **The interval is required.** HA sends a subscribed entity to the board
     only when its state changes (`force_update` does not change that), and
     the CN105 watchdog (`remote_temperature_timeout`) resets only when
     `set_remote_temperature()` is called. Without the interval, a steady
     input drops the unit to its thermistor after the timeout.
   - **Don't use `clamp` or `throttle` on this sensor.** `clamp` with
     `ignore_out_of_range` discards the NaN an unavailable entity arrives
     as, so the unit would keep a stale value until the timeout; `throttle`
     can swallow the update that says the input went away. The `on_value`
     lambda above falls back to the thermistor at once instead.
3. **Turn on the follow check.** Add the effector to `defaults.regulation_live`
   in `weatherstat.yaml` and restart. Weatherstat then alerts when the unit's
   reported temperature stays more than 1.5°F from the input for 10 minutes
   (the remote reading timed out, the board reads another entity, or the unit
   fell back to its thermistor).

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
