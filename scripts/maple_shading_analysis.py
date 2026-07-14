"""Deciduous shading (maple) analysis: does the SW maple change per-room solar gain?

A large deciduous maple stands SW of the house. If it meaningfully shades
glass, the temperature response to *west-plane* irradiance should be larger
in the bare season (leaf-out is ~mid-April in Seattle) than in the leafed
season — while the *east*-plane response (no maple) should be seasonally
stable. A static per-sensor solar coefficient would average the two regimes
and be wrong in both.

Method: per sensor, regress sysid-style Newton residuals (smoothed dT/dt
minus envelope loss) on analytic plane-irradiance features (horizontal,
east wall, west wall — each x cloud fraction), with recent HVAC activity
as controls. Fit separately for bare (< Apr 8) and leafed (> May 10)
periods; compare west slopes. The analytic plane geometry is validated
against the measured forecast.solar plane series first.

Usage: uv run python scripts/maple_shading_analysis.py
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from weatherstat.config import DATA_DIR, SNAPSHOTS_DB
from weatherstat.extract import load_collector_snapshots
from weatherstat.weather import condition_to_solar_fraction, solar_elevation
from weatherstat.yaml_config import load_config

BARE_END = pd.Timestamp("2026-04-08", tz="UTC")
LEAFED_START = pd.Timestamp("2026-05-10", tz="UTC")
SMOOTH_W = 3  # sysid's smoothed-derivative half-window (steps)
DT_HOURS = 5.0 / 60.0
# Autocorrelation from the 35-min smoothing kernel reduces effective sample
# size ~3x (docs/debugging-notes.md); SEs are inflated by sqrt(3).
SE_INFLATION = math.sqrt(3.0)

ZONE_ACTION = {
    "thermostat_upstairs": "thermostat_upstairs_action",
    "thermostat_downstairs": "thermostat_downstairs_action",
}
OWN_SPLIT_ACTION = {  # rooms with their own mini split (stage-22 naming trap: hardcoded)
    "bedroom_temp": "mini_split_bedroom_action",
    "living_room_climate_temp": "mini_split_living_room_action",
}


def solar_azimuth(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """Solar azimuth in degrees from north (clockwise), matching weather.py's
    declination and hour-angle conventions (Spencer-era simple formula)."""
    day_of_year = dt.timetuple().tm_yday
    declination = 23.45 * math.sin(math.radians(360.0 / 365.0 * (day_of_year - 81)))
    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_angle = 15.0 * (utc_hour + lon_deg / 15.0 - 12.0)
    lat_r, dec_r, ha_r = map(math.radians, (lat_deg, declination, hour_angle))
    # Azimuth measured from south, positive toward west; convert to from-north
    az_south = math.atan2(math.sin(ha_r), math.cos(ha_r) * math.sin(lat_r) - math.tan(dec_r) * math.cos(lat_r))
    return (math.degrees(az_south) + 180.0) % 360.0


def plane_feature(elev_deg: float, az_deg: float, plane_az_deg: float | None) -> float:
    """Clamped incidence factor: horizontal if plane_az is None, else vertical wall."""
    if elev_deg <= 0:
        return 0.0
    if plane_az_deg is None:
        return math.sin(math.radians(elev_deg))
    cos_inc = math.cos(math.radians(elev_deg)) * math.cos(math.radians(az_deg - plane_az_deg))
    return max(0.0, cos_inc)


def validate_geometry(feat: pd.DataFrame) -> None:
    """Correlate analytic plane features with measured forecast.solar series."""
    conn = sqlite3.connect(str(SNAPSHOTS_DB))
    meas = pd.read_sql(
        "SELECT timestamp, name, value FROM readings WHERE name LIKE 'solar_power_%'", conn)
    conn.close()
    meas["ts"] = pd.to_datetime(meas["timestamp"], format="ISO8601", utc=True)
    meas["value"] = pd.to_numeric(meas["value"], errors="coerce")
    wide = meas.pivot_table(index="ts", columns="name", values="value")
    joined = feat.join(wide, how="inner").dropna()
    print("── Geometry validation: analytic feature vs measured forecast.solar plane ──")
    for analytic, measured in [
        ("f_horiz", "solar_power_horizontal"),
        ("f_east", "solar_power_east"),
        ("f_west", "solar_power_west"),
        ("f_south", "solar_power_south"),
    ]:
        if measured in joined.columns:
            r = joined[analytic].corr(joined[measured])
            print(f"  {analytic:<8} vs {measured:<24} r = {r:+.3f}  (n={len(joined)})")
    print()


def sensor_taus_and_zones() -> tuple[dict[str, float], dict[str, str]]:
    with open(DATA_DIR / "thermal_params.json") as f:
        params = json.load(f)
    taus = {ft["sensor"]: ft["tau_base"] for ft in params["fitted_taus"]}
    best: dict[str, tuple[str, float]] = {}
    for g in params["effector_sensor_gains"]:
        if g["negligible"] or g["gain_f_per_hour"] <= 0 or not g["effector"].startswith("thermostat_"):
            continue
        if g["gain_f_per_hour"] > best.get(g["sensor"], ("", 0.0))[1]:
            best[g["sensor"]] = (g["effector"], g["gain_f_per_hour"])
    zones = {sensor: eff for sensor, (eff, _) in best.items()}
    return taus, zones


def fit_season(
    df: pd.DataFrame, sensor: str, tau: float, zone_action: str, split_action: str | None,
) -> dict[str, tuple[float, float, int]] | None:
    """OLS of Newton residual on plane features + controls. Returns coef -> (b, se, n)."""
    temps = df[sensor].values.astype(float)
    smooth = pd.Series(temps).rolling(2 * SMOOTH_W + 1, center=True, min_periods=SMOOTH_W + 1).mean().values
    dT = np.full_like(temps, np.nan)
    if len(temps) > 2 * SMOOTH_W:
        dT[SMOOTH_W:-SMOOTH_W] = (smooth[2 * SMOOTH_W:] - smooth[:-2 * SMOOTH_W]) / (2 * SMOOTH_W * DT_HOURS)
    outdoor = df["_outdoor"].values.astype(float)
    resid = dT - (outdoor - temps) / tau

    heat_recent = df[zone_action].eq("heating").rolling(12, min_periods=1).mean().values
    win_open = df["_windows_open"].values.astype(float)
    split_recent = np.zeros(len(df))
    if split_action and split_action in df.columns:
        s = df[split_action]
        split_recent = (s.eq("heating").astype(float) - s.eq("cooling").astype(float)).rolling(
            12, min_periods=1).mean().values

    X_cols = {
        "f_horiz": df["f_horiz"].values,
        "f_east": df["f_east"].values,
        "f_west": df["f_west"].values,
        "heat": heat_recent,
        "split": split_recent,
        "windows": win_open,
        "dT_out": (outdoor - temps),
    }
    # Drop zero-variance columns (e.g., split activity for rooms without one)
    X_cols = {k: v for k, v in X_cols.items() if np.nanstd(v) > 1e-9}
    X = np.column_stack([np.ones(len(df))] + list(X_cols.values()))
    y = resid
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (df["_elev"].values > 0)
    X, y = X[mask], y[mask]
    n = len(y)
    if n < 300:
        return None
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid_ols = y - X @ beta
    dof = max(n - X.shape[1], 1)
    sigma2 = float(resid_ols @ resid_ols) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    ses = np.sqrt(np.diag(cov)) * SE_INFLATION
    names = ["const"] + list(X_cols.keys())
    return {name: (float(b), float(se), n) for name, b, se in zip(names, beta, ses, strict=True)}


def main() -> None:
    cfg = load_config()
    lat, lon = cfg.location.latitude, cfg.location.longitude

    df = load_collector_snapshots()
    df["_ts"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("_ts").reset_index(drop=True)

    # Outdoor: weather-entity temp (side sensor is solar-contaminated)
    df["_outdoor"] = df.get("met_outdoor_temp")

    # Analytic solar features x cloud fraction
    times = df["_ts"].dt.to_pydatetime()
    df["_elev"] = [solar_elevation(lat, lon, t) for t in times]
    az = [solar_azimuth(lat, lon, t) for t in times]
    cond = df.get("weather_condition", pd.Series("", index=df.index))
    sf = [condition_to_solar_fraction(str(w)) for w in cond]
    for name, plane_az in [("f_horiz", None), ("f_east", 90.0), ("f_south", 180.0), ("f_west", 270.0)]:
        df[name] = [plane_feature(e, a, plane_az) * f for e, a, f in zip(df["_elev"], az, sf, strict=True)]

    validate_geometry(df.set_index("_ts")[["f_horiz", "f_east", "f_south", "f_west"]])

    # Windows/doors open count (ventilation confound: summer afternoons)
    win_cols = [e.column for e in cfg.environment.values() if e.kind in ("window", "door") and e.column in df.columns]
    df["_windows_open"] = df[win_cols].fillna(False).astype(bool).sum(axis=1) if win_cols else 0.0

    taus, zones = sensor_taus_and_zones()
    bare = df[df["_ts"] < BARE_END]
    leafed = df[df["_ts"] >= LEAFED_START]
    print(f"bare rows: {len(bare)} (through {BARE_END.date()}), leafed rows: {len(leafed)} (from {LEAFED_START.date()})\n")

    print("── West-plane slope (°F/hr per unit feature): bare vs leafed ──")
    hdr = f"{'sensor':<28}{'W bare':>14}{'W leafed':>14}{'ratio':>7}{'t_diff':>7}   {'E bare':>12}{'E leafed':>12}{'E ratio':>8}"
    print(hdr)
    print("-" * len(hdr))
    for sensor, zone in sorted(zones.items()):
        tau = taus.get(sensor, 45.0)
        action_col = ZONE_ACTION.get(zone)
        if action_col is None or sensor not in df.columns:
            continue
        split_col = OWN_SPLIT_ACTION.get(sensor)
        fb = fit_season(bare, sensor, tau, action_col, split_col)
        fl = fit_season(leafed, sensor, tau, action_col, split_col)
        if fb is None or fl is None:
            print(f"{sensor:<28}{'insufficient data':>28}")
            continue
        wb, wb_se, _ = fb["f_west"]
        wl, wl_se, _ = fl["f_west"]
        eb, _, _ = fb["f_east"]
        el, _, _ = fl["f_east"]
        ratio = wb / wl if abs(wl) > 1e-6 else float("inf")
        e_ratio = eb / el if abs(el) > 1e-6 else float("inf")
        t_diff = (wb - wl) / math.sqrt(wb_se**2 + wl_se**2)
        print(
            f"{sensor:<28}{wb:+.2f} ±{wb_se:.2f}{wl:+.2f} ±{wl_se:.2f}{ratio:>7.2f}{t_diff:>7.1f}"
            f"   {eb:>+12.2f}{el:>+12.2f}{e_ratio:>8.2f}"
        )

    print("\nInterpretation: W ratio >> 1 with |t_diff| >= 2 and E ratio ~ 1 -> maple")
    print("shading confirmed for that room. E ratio far from 1 too -> seasonal")
    print("geometry/data artifact, not the tree.")


if __name__ == "__main__":
    main()
