"""Fit thermal time constants (τ) per room from the overnight cooling fixture.

Fits Newton's law of cooling:  T(t) = T_outdoor + (T_0 - T_outdoor) * exp(-t/τ)

Uses the 10pm-6am window (PST) from Feb 14-15 2026 where all HVAC was off.
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

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "overnight_cooling_20260214.parquet"

# Overnight window (PST): 10pm Feb 14 → 6am Feb 15
# In UTC: 06:00 → 14:00 Feb 15
OVERNIGHT_START_UTC = pd.Timestamp("2026-02-15 06:00:00", tz="UTC")
OVERNIGHT_END_UTC = pd.Timestamp("2026-02-15 14:00:00", tz="UTC")


def newton_cooling(t: np.ndarray, tau: float) -> np.ndarray:
    """Newton's law of cooling with tau as the only free parameter.

    T_outdoor and T_0 are baked in via closure in the fitting loop.
    """
    return np.exp(-t / tau)


def main() -> None:
    if not FIXTURE_PATH.exists():
        print(f"Fixture not found: {FIXTURE_PATH}")
        print("Run: uv run python scripts/save_overnight_fixture.py")
        return

    raw = pd.read_parquet(FIXTURE_PATH)
    raw["_ts"] = pd.to_datetime(raw["timestamp"], format="ISO8601", utc=True)

    # Filter to overnight window
    overnight = raw[(raw["_ts"] >= OVERNIGHT_START_UTC) & (raw["_ts"] <= OVERNIGHT_END_UTC)].copy()
    print(f"Overnight window: {len(overnight)} rows")
    print(f"  {overnight['_ts'].iloc[0]} to {overnight['_ts'].iloc[-1]}")

    # Time in hours from start
    t0 = overnight["_ts"].iloc[0]
    t_hours = (overnight["_ts"] - t0).dt.total_seconds().values / 3600.0

    # Average outdoor temp over the window (relatively stable overnight)
    t_outdoor_mean = overnight["outdoor_temp"].mean()
    print(f"  Outdoor temp (mean): {t_outdoor_mean:.1f}°F")
    print()

    fitted: dict[str, float] = {}

    for room in PREDICTION_ROOMS:
        temp_col = ROOM_TEMP_COLUMNS.get(room)
        if temp_col is None or temp_col not in overnight.columns:
            print(f"  {room}: no data")
            continue

        temps = overnight[temp_col].values
        t_0 = temps[0]

        if abs(t_0 - t_outdoor_mean) < 1.0:
            print(f"  {room}: T_0 ≈ T_outdoor, skipping (no meaningful decay)")
            continue

        # Normalized: (T - T_outdoor) / (T_0 - T_outdoor) should decay as exp(-t/τ)
        normalized = (temps - t_outdoor_mean) / (t_0 - t_outdoor_mean)

        try:
            popt, _ = curve_fit(
                newton_cooling,
                t_hours,
                normalized,
                p0=[40.0],
                bounds=(1.0, 500.0),
            )
            tau = popt[0]

            # Predicted final temp
            t_pred_final = t_outdoor_mean + (t_0 - t_outdoor_mean) * np.exp(-t_hours[-1] / tau)
            t_actual_final = temps[-1]

            print(f"  {room:15s}: τ = {tau:5.1f}h  (T: {t_0:.1f} → {t_actual_final:.1f}°F, pred: {t_pred_final:.1f}°F)")
            fitted[room] = round(tau, 1)
        except RuntimeError as e:
            print(f"  {room}: fit failed — {e}")

    # Print YAML snippet
    print("\n# YAML snippet for weatherstat.yaml:")
    print("thermal:")
    print("  tau:")
    for room, tau in fitted.items():
        print(f"    {room}: {tau}")
    print("  default_tau: 45.0")


if __name__ == "__main__":
    main()
