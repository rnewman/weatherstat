#!/usr/bin/env python3
"""Discover Home Assistant entities and generate a starter weatherstat.yaml.

Connects to your HA instance, finds climate devices, temperature/humidity
sensors, fans, window/door sensors, and weather entities, then writes a
commented YAML config you can customize.

Usage:
    just discover                           # uses HA_URL/HA_TOKEN from env or .env
    just discover --output my-config.yaml   # write to specific file
    just discover --url https://ha.local --token YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from the weatherstat data directory (best-effort)."""
    data_dir = os.environ.get("WEATHERSTAT_DATA_DIR", str(Path.home() / ".weatherstat"))
    env_file = Path(data_dir) / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredEntity:
    entity_id: str
    friendly_name: str
    state: str
    domain: str
    device_class: str
    unit: str
    attributes: dict


@dataclass
class Classified:
    """Entities grouped by weatherstat role."""

    thermostats: list[DiscoveredEntity] = field(default_factory=list)
    mini_splits: list[DiscoveredEntity] = field(default_factory=list)
    fans: list[DiscoveredEntity] = field(default_factory=list)
    temp_sensors: list[DiscoveredEntity] = field(default_factory=list)
    humidity_sensors: list[DiscoveredEntity] = field(default_factory=list)
    power_sensors: list[DiscoveredEntity] = field(default_factory=list)
    state_sensors: list[DiscoveredEntity] = field(default_factory=list)
    windows: list[DiscoveredEntity] = field(default_factory=list)
    doors: list[DiscoveredEntity] = field(default_factory=list)
    weather: list[DiscoveredEntity] = field(default_factory=list)
    location: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HA API helpers
# ---------------------------------------------------------------------------


def fetch_ha_config(url: str, token: str) -> dict:
    """Fetch HA instance configuration (location, timezone, etc.)."""
    import requests

    resp = requests.get(
        f"{url}/api/config",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_entities(url: str, token: str) -> list[dict]:
    """Fetch all entity states from HA REST API."""
    import requests

    resp = requests.get(
        f"{url}/api/states",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _parse_entity(raw: dict) -> DiscoveredEntity:
    attrs = raw.get("attributes", {})
    return DiscoveredEntity(
        entity_id=raw["entity_id"],
        friendly_name=attrs.get("friendly_name", raw["entity_id"]),
        state=raw.get("state", "unknown"),
        domain=raw["entity_id"].split(".")[0],
        device_class=attrs.get("device_class", ""),
        unit=attrs.get("unit_of_measurement", ""),
        attributes=attrs,
    )


def _is_thermostat(e: DiscoveredEntity) -> bool:
    """Heuristic: heat-only climate entity → thermostat."""
    modes = set(e.attributes.get("hvac_modes", []))
    # Has heat but not cool/fan_only → likely a thermostat
    return "heat" in modes and "cool" not in modes and "fan_only" not in modes


def classify(entities: list[dict], ha_config: dict) -> Classified:
    c = Classified()
    c.location = {
        "latitude": ha_config.get("latitude", 0),
        "longitude": ha_config.get("longitude", 0),
        "elevation": ha_config.get("elevation", 0),
        "timezone": ha_config.get("time_zone", "UTC"),
    }

    for raw in entities:
        e = _parse_entity(raw)
        if e.state in ("unavailable", "unknown"):
            continue

        if e.domain == "climate":
            if _is_thermostat(e):
                c.thermostats.append(e)
            else:
                c.mini_splits.append(e)

        elif e.domain == "fan":
            c.fans.append(e)

        elif e.domain == "sensor":
            if e.device_class == "temperature" or e.unit in ("°F", "°C"):
                c.temp_sensors.append(e)
            elif e.device_class == "humidity" or e.unit == "%":
                # Only humidity-class sensors, not battery % etc.
                if e.device_class == "humidity":
                    c.humidity_sensors.append(e)
            elif e.device_class in ("power", "energy"):
                c.power_sensors.append(e)

        elif e.domain == "binary_sensor":
            if e.device_class in ("window", "opening"):
                c.windows.append(e)
            elif e.device_class == "door":
                c.doors.append(e)

        elif e.domain == "weather":
            c.weather.append(e)

    # Sort each list by entity_id for stable output
    for lst in [
        c.thermostats, c.mini_splits, c.fans, c.temp_sensors,
        c.humidity_sensors, c.power_sensors, c.windows, c.doors, c.weather,
    ]:
        lst.sort(key=lambda x: x.entity_id)

    return c


# ---------------------------------------------------------------------------
# Name generation
# ---------------------------------------------------------------------------


def _clean_name(entity_id: str) -> str:
    """Strip domain prefix."""
    return entity_id.split(".", 1)[1]


def _temp_name(entity_id: str) -> str:
    """Generate a config name for a temperature sensor."""
    name = _clean_name(entity_id)
    for suffix in ("_air_temperature", "_temperature"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{name}_temp"


def _humidity_name(entity_id: str) -> str:
    name = _clean_name(entity_id)
    if not name.endswith("_humidity"):
        name = f"{name}_humidity"
    return name


def _window_name(entity_id: str) -> str:
    name = _clean_name(entity_id)
    # Strip common suffixes
    for suffix in ("_intrusion", "_is_open", "_window_door_is_open", "_contact"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # Strip common prefixes
    for prefix in ("window_", "door_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_summary(c: Classified) -> None:
    """Print a human-readable summary of discovered entities."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              Home Assistant Entity Discovery                ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    loc = c.location
    print(f"  Location: {loc['latitude']:.4f}°N, {loc['longitude']:.4f}°E, "
          f"{loc['elevation']}m — {loc['timezone']}\n")

    sections = [
        ("Thermostats (climate, heat-only)", c.thermostats),
        ("Mini-splits / heat pumps (climate, heat+cool)", c.mini_splits),
        ("Fans", c.fans),
        ("Temperature sensors", c.temp_sensors),
        ("Humidity sensors", c.humidity_sensors),
        ("Power/energy sensors", c.power_sensors),
        ("Window sensors", c.windows),
        ("Door sensors", c.doors),
        ("Weather entities", c.weather),
    ]

    for label, entities in sections:
        print(f"  {label}: {len(entities)}")
        for e in entities:
            state_info = f" = {e.state}"
            if e.unit:
                state_info += f" {e.unit}"
            modes = e.attributes.get("hvac_modes")
            if modes:
                state_info += f"  modes: {modes}"
            presets = e.attributes.get("preset_modes")
            if presets:
                state_info += f"  presets: {presets}"
            print(f"    {e.entity_id}{state_info}")
            print(f"      {e.friendly_name}")
        print()


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


def _is_plausible_room_temp(e: DiscoveredEntity) -> bool:
    """Check if the sensor's current reading is in a plausible indoor range."""
    try:
        val = float(e.state)
        # Room temps should be roughly 40-100°F / 5-38°C
        if e.unit == "°C":
            return 5 <= val <= 38
        return 40 <= val <= 100  # °F or unknown
    except (ValueError, TypeError):
        return True  # Can't parse — include it and let the user decide


def _indent(text: str, level: int = 2) -> str:
    prefix = " " * level
    return "\n".join(prefix + line if line.strip() else "" for line in text.splitlines())


def _quote_if_needed(s: str) -> str:
    """Quote YAML values that look like keywords."""
    if s in ("off", "on", "true", "false", "yes", "no", "null"):
        return f'"{s}"'
    return s


def generate_yaml(c: Classified) -> str:
    """Generate a starter weatherstat.yaml with good comments."""
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    loc = c.location

    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    # Header
    w(f"# Weatherstat configuration — generated by discover.py on {now}")
    w("# Review and customize, then copy to ~/.weatherstat/weatherstat.yaml")
    w("#")
    w("# Docs: docs/onboarding.md, docs/overview.md")
    w()

    # Location
    w("location:")
    w(f"  latitude: {loc['latitude']}")
    w(f"  longitude: {loc['longitude']}")
    w(f"  elevation: {loc['elevation']}")
    w(f"  timezone: {loc['timezone']}")
    w()

    # ── Sensors ──
    w("# ── Sensors: observable quantities ────────────────────────────────────────")
    w("# key = snake_case column name used in snapshots and features.")
    w("# One temperature sensor MUST have 'role: outdoor' for the physics model.")
    w()
    w("sensors:")

    # Temperature
    w("  temperature:")
    if not c.temp_sensors:
        w("    # No temperature sensors found — add manually.")
    else:
        outdoor_candidates = [
            e for e in c.temp_sensors
            if any(kw in e.entity_id.lower() or kw in e.friendly_name.lower()
                   for kw in ("outdoor", "outside", "exterior", "side"))
        ]
        for e in c.temp_sensors:
            name = _temp_name(e.entity_id)
            is_outdoor = e in outdoor_candidates
            w(f"    {name}:")
            w(f"      entity_id: {e.entity_id}")
            if is_outdoor:
                w("      role: outdoor")
            w(f"      # {e.friendly_name} — currently {e.state}{e.unit}")

        if not outdoor_candidates:
            w("    # WARNING: no outdoor temperature sensor detected.")
            w("    # Add one with 'role: outdoor' — required for the physics model.")
            w("    # outdoor_temp:")
            w("    #   entity_id: sensor.your_outdoor_sensor")
            w("    #   role: outdoor")

    # Humidity
    w("  humidity:")
    if not c.humidity_sensors:
        w("    # No humidity sensors found (optional).")
    else:
        for e in c.humidity_sensors:
            name = _humidity_name(e.entity_id)
            w(f"    {name}:")
            w(f"      entity_id: {e.entity_id}")
            w(f"      # {e.friendly_name}")

    # State sensors (user adds manually — boiler, etc.)
    w("  # state:")
    w("  #   # State sensors track categorical device states (e.g., boiler heating mode).")
    w("  #   # These are sensors that report what the device is DOING, not effectors.")
    w("  #   boiler_heating:")
    w("  #     entity_id: sensor.your_boiler_mode")
    w('  #     encoding: { "Heating": 1, "Idle": 0 }')
    w("  # power:")
    w("  #   # Power sensors for energy monitoring (optional).")
    w("  #   boiler_gas:")
    w("  #     entity_id: sensor.your_boiler_power")
    w()

    # ── Effectors ──
    w("# ── Effectors: actuatable devices ─────────────────────────────────────────")
    w("#")
    w("# control_type:")
    w("#   trajectory  — slow-response (hydronic heat, radiant panels): on/off timing search")
    w("#   regulating  — self-regulating with target temp (mini-splits, radiator valves)")
    w("#   binary      — discrete modes (fans, dampers): mode sweep")
    w("#")
    w("# mode_control:")
    w("#   manual    — human controls the mode (you flip the thermostat to heat/off)")
    w("#   automatic — system controls the mode (it decides heat/cool/off)")
    w("#")
    w("# depends_on: name(s) of parent effector(s) — child only useful when ALL parents active.")
    w("#   Example: a duct fan only helps when the thermostat is calling for heat AND")
    w("#   the boiler is actually firing. List all required parents.")
    w()
    w("effectors:")

    if not c.thermostats and not c.mini_splits and not c.fans:
        w("  # No HVAC devices found — add manually.")
    else:
        for e in c.thermostats:
            name = _clean_name(e.entity_id)
            modes = [m for m in e.attributes.get("hvac_modes", []) if m != "off"]
            w(f"  {name}:")
            w(f"    entity_id: {e.entity_id}")
            w("    control_type: trajectory         # slow-response heating")
            w("    mode_control: manual             # you control heat/off")
            w(f"    supported_modes: [{', '.join(modes) or 'heat'}]")
            w("    # state_device: ???              # state sensor confirming delivery (e.g., boiler mode)")
            w("    # state_encoding: { heating: 1, idle: 0, \"off\": 0 }")
            w("    max_lag_minutes: 60              # hydronic: 60-90, forced air: 15-30")
            w("    energy_cost: 0.010               # relative cost per heating hour")
            w(f"    # {e.friendly_name}")

        for e in c.mini_splits:
            name = _clean_name(e.entity_id)
            modes = [m for m in e.attributes.get("hvac_modes", []) if m not in ("off", "auto")]
            # Pick the useful modes
            useful_modes = [m for m in modes if m in ("heat", "cool")]
            if not useful_modes:
                useful_modes = modes[:2] if modes else ["heat", "cool"]
            all_modes_str = ", ".join(e.attributes.get("hvac_modes", []))
            w(f"  {name}:")
            w(f"    entity_id: {e.entity_id}")
            w("    control_type: regulating         # proportional target-temperature control")
            w("    mode_control: automatic          # system chooses heat/cool/off")
            w(f"    supported_modes: [{', '.join(useful_modes)}]")
            w("    proportional_band: 1.0           # activity ramps 0→1 within this many °F of target")
            w("    mode_hold_window: [22, 7]        # no mode changes during these hours (quiet)")
            # Generate encodings for all known modes
            enc_parts = []
            for m in e.attributes.get("hvac_modes", []):
                if m == "off":
                    enc_parts.append('"off": 0')
                elif m == "heat":
                    enc_parts.append("heat: 1")
                elif m == "cool":
                    enc_parts.append("cool: -1")
                elif m == "fan_only":
                    enc_parts.append("fan_only: 0.5")
                elif m == "dry":
                    enc_parts.append("dry: 0.25")
                elif m in ("auto", "heat_cool"):
                    enc_parts.append(f"{m}: 0.5")
                else:
                    enc_parts.append(f"{m}: 0")
            w(f"    command_encoding: {{ {', '.join(enc_parts)} }}")
            w("    state_encoding: { heating: 1, cooling: -1, drying: 0.25, idle: 0, \"off\": 0 }")
            w("    max_lag_minutes: 15")
            w("    energy_cost: 0.005")
            w(f"    # {e.friendly_name} — modes: [{all_modes_str}]")

        for e in c.fans:
            name = _clean_name(e.entity_id)
            presets = [p for p in e.attributes.get("preset_modes", []) if p.lower() != "off"]
            # Build supported modes from preset_modes or defaults
            if presets:
                fan_modes = ['"off"'] + [_quote_if_needed(p) for p in presets]
                enc_parts = ['"off": 0']
                for i, p in enumerate(presets, 1):
                    enc_parts.append(f'{_quote_if_needed(p)}: {i}')
            else:
                fan_modes = ['"off"', "low", "high"]
                enc_parts = ['"off": 0', "low: 1", "high: 2"]
            w(f"  {name}:")
            w(f"    entity_id: {e.entity_id}")
            w("    control_type: binary             # discrete modes")
            w("    mode_control: automatic")
            w(f"    supported_modes: [{', '.join(fan_modes)}]")
            w("    # depends_on: ???                # parent effector name (e.g., a thermostat)")
            w(f"    state_encoding: {{ {', '.join(enc_parts)} }}")
            w("    max_lag_minutes: 5")
            energy_parts = ['"off": 0.0']
            for i, p in enumerate(presets or ["low", "high"], 1):
                energy_parts.append(f"{_quote_if_needed(p)}: {0.001 * i:.3f}")
            w(f"    energy_cost: {{ {', '.join(energy_parts)} }}")
            w(f"    # {e.friendly_name}")

    w()

    # ── Health checks ──
    w("# ── Health checks: device-level alerts (optional) ─────────────────────────")
    w("# Monitor critical infrastructure. Alerts bypass quiet hours.")
    w()
    w("# health:")
    w("#   boiler_connection:")
    w("#     entity: binary_sensor.your_boiler_connection")
    w('#     expected_state: "on"')
    w("#     severity: critical")
    w('#     message: "Boiler connection lost"')
    w()

    # ── Windows ──
    w("# ── Windows: environmental modifiers ──────────────────────────────────────")
    w("# Window/door sensors. Sysid learns the thermal effect of each window —")
    w("# you don't need to configure which rooms they affect.")
    w()
    w("windows:")
    all_openings = c.windows + c.doors
    if not all_openings:
        w("  # No window/door sensors found (optional but valuable).")
    else:
        seen_names: set[str] = set()
        for e in all_openings:
            name = _window_name(e.entity_id)
            # Disambiguate collisions (e.g., bedroom window + bedroom door)
            if name in seen_names:
                is_door = e.device_class == "door" or "door" in e.entity_id.lower()
                suffix = "door" if is_door else "window"
                name = f"{name}_{suffix}"
            # Still collides? Append entity suffix
            while name in seen_names:
                name = f"{name}_2"
            seen_names.add(name)
            w(f"  {name}:")
            w(f"    entity_id: {e.entity_id}")
            w(f"    # {e.friendly_name}")
    w()

    # ── Weather ──
    w("weather:")
    if c.weather:
        w(f"  entity_id: {c.weather[0].entity_id}")
        if len(c.weather) > 1:
            w("  # Other weather entities found:")
            for e in c.weather[1:]:
                w(f"  #   {e.entity_id} — {e.friendly_name}")
    else:
        w("  entity_id: weather.forecast_home  # default met.no integration")
    w()

    # ── Constraints ──
    w("# ── Constraints: comfort objectives ───────────────────────────────────────")
    w("# Define what 'comfortable' means for each sensor you want to optimize.")
    w("# The system minimizes cost = comfort_deviation + energy_cost.")
    w("#")
    w("# preferred: ideal temperature (quadratic cost for any deviation)")
    w("# min/max: hard rails with steep 10× penalty beyond these bounds")
    w("# cold_penalty/hot_penalty: asymmetric weights (default 1.0)")
    w("# hours: [start, end] in 24h format (wraps past midnight)")
    w("#")
    w("# Tip: start with a single 24h schedule per sensor. Add time-of-day")
    w("# schedules later once you see how the system behaves.")
    w()
    w("constraints:")
    w("  # comfort_entity: input_select.thermostat_mode  # HA entity for Home/Away")
    w("  # profiles:")
    w("#    Home: {}  # base schedules unchanged")
    w("#    Away:")
    w("#      preferred_offset: -3")
    w("#      min_offset: -3")
    w("#      max_offset: 2")
    w("  # mrt_correction:           # Mean radiant temperature correction (optional)")
    w("  #   alpha: 0.1              # °F shift per °F outdoor deviation from reference")
    w("  #   reference_temp: 50      # outdoor temp where targets feel right")
    w("  #   max_offset: 3.0         # cap correction magnitude")
    w("  schedules:")

    # Generate a constraint for each plausible indoor temperature sensor
    _outdoor_kw = ("outdoor", "outside", "exterior", "side")
    _device_kw = ("esp_temp", "esp_temperature", "inlet", "outlet", "return", "target_temp",
                   "blink_", "motion_", "apollo_", "bluetooth_", "satellite")
    indoor_temps = [
        e for e in c.temp_sensors
        if not any(kw in e.entity_id.lower() or kw in e.friendly_name.lower()
                   for kw in _outdoor_kw)
        and not any(kw in e.entity_id.lower() for kw in _device_kw)
        # Skip sensors with implausible room temperatures (device internals, etc.)
        and _is_plausible_room_temp(e)
    ]
    if indoor_temps:
        for e in indoor_temps[:8]:  # Cap at 8 to avoid overwhelming
            name = _temp_name(e.entity_id)
            w(f"    - sensor: {name}")
            w("      schedule:")
            w("        - { hours: [0, 24], preferred: 71, min: 69, max: 75 }")
            w(f"      # {e.friendly_name}")
        if len(indoor_temps) > 8:
            w(f"    # ... and {len(indoor_temps) - 8} more temperature sensors")
            w("    # Add constraints for the ones you care about.")
    else:
        w("    # Add a constraint for each sensor you want to keep comfortable:")
        w("    # - sensor: living_room_temp")
        w("    #   schedule:")
        w("    #     - { hours: [0, 24], preferred: 71, min: 69, max: 75 }")
    w()

    # ── Notifications ──
    w("# notifications:")
    w("#   target: notify.mobile_app_your_phone  # HA notify service for alerts")
    w()

    # ── Advisory ──
    w("# advisory:")
    w("#   opportunity_threshold: 0.3   # minimum benefit to track window opportunity")
    w("#   notification_threshold: 1.5  # minimum benefit to push notification")
    w("#   quiet_hours: [22, 7]")
    w("#   cooldowns:")
    w("#     free_cooling: 14400        # seconds between open-window notifications")
    w("#     close_windows: 3600        # seconds between close-window notifications")
    w()

    # ── Safety ──
    w("# safety:")
    w("#   cooldowns:")
    w("#     thermostat_off: 3600")
    w("#     device_fault: 1800")
    w()

    # ── Defaults ──
    w("defaults:")
    w("  tau: 45.0  # hours — envelope time constant (used before sysid runs)")
    w("             # Well-insulated: 30-60h. Poorly insulated: 10-20h.")
    w("             # Sysid will fit this from your data; this is just the starting guess.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover HA entities and generate weatherstat.yaml")
    parser.add_argument("--url", help="HA URL (default: HA_URL env var)")
    parser.add_argument("--token", help="HA long-lived access token (default: HA_TOKEN env var)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    _load_dotenv()

    url = args.url or os.environ.get("HA_URL", "")
    token = args.token or os.environ.get("HA_TOKEN", "")

    if not url or not token:
        print("Error: HA_URL and HA_TOKEN required.", file=sys.stderr)
        print("  Set via --url/--token flags, environment variables, or", file=sys.stderr)
        print("  ~/.weatherstat/.env (HA_URL=... HA_TOKEN=...)", file=sys.stderr)
        sys.exit(1)

    # Strip trailing slash
    url = url.rstrip("/")

    print(f"Connecting to {url}...", file=sys.stderr)

    try:
        import importlib.util

        if importlib.util.find_spec("requests") is None:
            raise ImportError
    except ImportError:
        print("Error: 'requests' package required. Run: pip install requests", file=sys.stderr)
        sys.exit(1)

    try:
        ha_config = fetch_ha_config(url, token)
    except Exception as e:
        print(f"Error fetching HA config: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        raw_entities = fetch_all_entities(url, token)
    except Exception as e:
        print(f"Error fetching entities: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(raw_entities)} entities.", file=sys.stderr)

    classified = classify(raw_entities, ha_config)
    print_summary(classified)

    yaml_content = generate_yaml(classified)

    if args.output:
        Path(args.output).write_text(yaml_content)
        print(f"\nConfig written to {args.output}", file=sys.stderr)
        print("Next steps:", file=sys.stderr)
        print(f"  1. Review and edit {args.output}", file=sys.stderr)
        print("  2. Copy to ~/.weatherstat/weatherstat.yaml", file=sys.stderr)
        print("  3. Run: just collect  (start collecting data)", file=sys.stderr)
        print("  4. Wait a few days for data to accumulate", file=sys.stderr)
        print("  5. Run: just sysid   (fit thermal parameters)", file=sys.stderr)
        print("  6. Run: just control (first control cycle)", file=sys.stderr)
    else:
        print("\n# ── Generated YAML below ─────────────────────────────────────\n",
              file=sys.stderr)
        print(yaml_content)


if __name__ == "__main__":
    main()
