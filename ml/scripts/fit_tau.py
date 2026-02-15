"""Fit thermal time constants (τ) per room from the overnight cooling fixture.

Fits Newton's law of cooling:  T(t) = T_outdoor + (T_0 - T_outdoor) * exp(-t/τ)

Uses the 10pm-6am window (PST) from Feb 14-15 2026 where all HVAC was off.
For rooms with window sensors, fits separate sealed (closed) and ventilated (open)
τ values when window state changes during the overnight period.

Prints fitted τ values and a YAML snippet for weatherstat.yaml.

Usage:
  uv run python scripts/fit_tau.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from weatherstat.config import PREDICTION_ROOMS
from weatherstat.features import ROOM_TEMP_COLUMNS
from weatherstat.yaml_config import load_config

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "overnight_cooling_20260214.parquet"

# Overnight window (PST): 10pm Feb 14 → 6am Feb 15
# In UTC: 06:00 → 14:00 Feb 15
OVERNIGHT_START_UTC = pd.Timestamp("2026-02-15 06:00:00", tz="UTC")
OVERNIGHT_END_UTC = pd.Timestamp("2026-02-15 14:00:00", tz="UTC")

_CFG = load_config()

# Room -> window column mapping (from YAML)
ROOM_WINDOW_COLS: dict[str, str] = {}
for win_name, win_cfg in _CFG.windows.items():
    for room in win_cfg.rooms:
        ROOM_WINDOW_COLS[room] = f"window_{win_name}_open"


def _fit_tau(t_hours: np.ndarray, temps: np.ndarray, t_outdoor: float) -> float | None:
    """Fit τ from time series. Returns None on failure."""
    t_0 = temps[0]
    if abs(t_0 - t_outdoor) < 1.0:
        return None
    normalized = (temps - t_outdoor) / (t_0 - t_outdoor)
    try:
        popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau), t_hours, normalized, p0=[40.0], bounds=(1.0, 500.0))
        return float(popt[0])
    except RuntimeError:
        return None


def main() -> None:
    if not FIXTURE_PATH.exists():
        print(f"Fixture not found: {FIXTURE_PATH}")
        print("Run: uv run python scripts/save_overnight_fixture.py")
        return

    raw = pd.read_parquet(FIXTURE_PATH)
    raw["_ts"] = pd.to_datetime(raw["timestamp"], format="ISO8601", utc=True)

    overnight = raw[(raw["_ts"] >= OVERNIGHT_START_UTC) & (raw["_ts"] <= OVERNIGHT_END_UTC)].copy()
    print(f"Overnight window: {len(overnight)} rows")
    print(f"  {overnight['_ts'].iloc[0]} to {overnight['_ts'].iloc[-1]}")

    t_outdoor_mean = overnight["outdoor_temp"].mean()
    print(f"  Outdoor temp (mean): {t_outdoor_mean:.1f}°F")
    print()

    fitted_sealed: dict[str, float] = {}
    fitted_vent: dict[str, float] = {}

    for room in PREDICTION_ROOMS:
        temp_col = ROOM_TEMP_COLUMNS.get(room)
        if temp_col is None or temp_col not in overnight.columns:
            print(f"  {room}: no data")
            continue

        window_col = ROOM_WINDOW_COLS.get(room)
        has_window_data = window_col is not None and window_col in overnight.columns
        window_changed = has_window_data and overnight[window_col].nunique() > 1

        if window_changed:
            # Use contiguous segments to avoid contamination
            # (e.g. post-close rows are cold from the open period, not from sealed decay)
            win_states = overnight[window_col].values
            segments: list[tuple[str, pd.DataFrame]] = []
            seg_start = 0
            for i in range(1, len(win_states)):
                if win_states[i] != win_states[seg_start]:
                    label = "ventilated" if win_states[seg_start] else "sealed"
                    segments.append((label, overnight.iloc[seg_start:i].copy()))
                    seg_start = i
            label = "ventilated" if win_states[seg_start] else "sealed"
            segments.append((label, overnight.iloc[seg_start:].copy()))

            # Fit the first contiguous segment of each type
            fitted_labels: set[str] = set()
            for label, phase in segments:
                if label in fitted_labels or len(phase) < 5:
                    continue
                t_hours = (phase["_ts"] - phase["_ts"].iloc[0]).dt.total_seconds().values / 3600.0
                temps = phase[temp_col].values
                tau = _fit_tau(t_hours, temps, t_outdoor_mean)
                if tau is None:
                    print(f"  {room:15s} {label}: fit failed")
                    continue
                t_pred = t_outdoor_mean + (temps[0] - t_outdoor_mean) * np.exp(-t_hours[-1] / tau)
                print(
                    f"  {room:15s} {label:10s}: τ = {tau:5.1f}h"
                    f"  (T: {temps[0]:.1f} → {temps[-1]:.1f}°F, pred: {t_pred:.1f}°F)"
                )
                target_dict = fitted_sealed if label == "sealed" else fitted_vent
                target_dict[room] = round(tau, 1)
                fitted_labels.add(label)
        else:
            # Single fit (all windows closed or no window sensor)
            t_hours = (overnight["_ts"] - overnight["_ts"].iloc[0]).dt.total_seconds().values / 3600.0
            temps = overnight[temp_col].values
            tau = _fit_tau(t_hours, temps, t_outdoor_mean)
            if tau is None:
                print(f"  {room:15s}: T_0 ≈ T_outdoor, skipping")
                continue
            t_pred = t_outdoor_mean + (temps[0] - t_outdoor_mean) * np.exp(-t_hours[-1] / tau)
            print(
                f"  {room:15s} sealed    : τ = {tau:5.1f}h"
                f"  (T: {temps[0]:.1f} → {temps[-1]:.1f}°F, pred: {t_pred:.1f}°F)"
            )
            fitted_sealed[room] = round(tau, 1)

    # Estimate ventilated τ for rooms without measured data
    # Use bedroom's measured ratio as a starting point
    if "bedroom" in fitted_sealed and "bedroom" in fitted_vent:
        ratio = fitted_vent["bedroom"] / fitted_sealed["bedroom"]
        print(f"\n  Bedroom ventilated/sealed ratio: {ratio:.2f}")
    else:
        ratio = 0.44  # fallback
        print(f"\n  Using default ventilated/sealed ratio: {ratio:.2f}")

    for room in fitted_sealed:
        if room not in fitted_vent:
            est = round(fitted_sealed[room] * ratio, 1)
            fitted_vent[room] = est
            print(f"  {room:15s} ventilated: τ ≈ {est:5.1f}h (estimated)")

    # Print YAML snippet
    print("\n# YAML snippet for weatherstat.yaml:")
    print("thermal:")
    print("  tau_sealed:")
    for room in PREDICTION_ROOMS:
        if room in fitted_sealed:
            print(f"    {room}: {fitted_sealed[room]}")
    print("  tau_ventilated:")
    for room in PREDICTION_ROOMS:
        if room in fitted_vent:
            print(f"    {room}: {fitted_vent[room]}")
    print("  default_tau_sealed: 45.0")
    print("  default_tau_ventilated: 20.0")


if __name__ == "__main__":
    main()
