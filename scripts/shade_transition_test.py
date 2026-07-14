"""Transition-weighted causality test for shade × solar couplings.

The batch fit contains cross-room shade couplings (e.g., moss_garden_shade →
piano_temp −0.39 °F/hr per unit solar feature) fitted from observational
user behavior — the weakest identification in the regression. This test
refits each sensor's Stage-2 regression with sample weights concentrated in
Gaussian windows around the shade's *state transitions*: within a few hours
of a transition, slow confounders (season, weather regime, occupancy
rhythm) are approximately constant while the shade state flips, so the
coefficient is identified by within-window contrast rather than by habit
correlation.

Reading the output: a coefficient that HOLDS as σ shrinks is causal (the
stage-31 result for effector gains); one that COLLAPSES toward zero was
diurnal/seasonal correlation. Power check: "info" is the weighted signal
energy Σw·F² for the feature — verdicts with tiny info are no verdicts.

Controls:
  bedroom_temp  — positive control (the shade is physically on that room)
  family_room   — null control (no fitted shade beta; should stay ~0)

Usage: uv run python scripts/shade_transition_test.py --params <thermal_params_with_regression_models>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from weatherstat.extract import load_collector_snapshots
from weatherstat.identification import transition_weights, weighted_ridge
from weatherstat.sysid import (
    _enumerate_effectors,
    _enumerate_sensors,
    _preprocess,
    build_design_matrix,
    newton_residual,
)

SIGMAS_HOURS = [None, 8.0, 4.0, 2.0, 1.0]  # None = uniform weights (batch replication)
DEVICES = ["moss_garden_shade", "bedroom_upper_blinds"]
SENSORS = ["piano_temp", "bedroom_temp", "bathroom_temp", "family_room_temp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    args = parser.parse_args()

    params = json.loads(args.params.read_text())
    models = params.get("regression_models", {})

    effectors = _enumerate_effectors()
    all_sensors = {s.name: s for s in _enumerate_sensors()}

    print("Loading and preprocessing full history...")
    df = load_collector_snapshots()
    df = _preprocess(df, effectors, list(all_sensors.values()))
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    ts_hours = ((ts - ts.iloc[0]).dt.total_seconds() / 3600.0).values

    for device in DEVICES:
        col = f"{device}_active"
        if col not in df.columns:
            print(f"\n{device}: no state column, skipped")
            continue
        state = (pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0) > 0.5).values
        n_trans = int((state[1:] != state[:-1]).sum())
        feature = f"_adv_solar_{device}"
        print(f"\n══ {device}: {n_trans} transitions ══")
        header = f"{'sensor':<18}" + "".join(
            f"{'uniform' if s is None else f'σ={s:.0f}h':>20}" for s in SIGMAS_HOURS
        )
        print(header)
        print(" " * 18 + "".join(f"{'β     t   info':>20}" for _ in SIGMAS_HOURS))

        for sensor_name in SENSORS:
            sensor = all_sensors.get(sensor_name)
            model = models.get(sensor_name)
            if sensor is None or model is None:
                continue
            dm = build_design_matrix(df, sensor, effectors)
            y = newton_residual(df, sensor, model["tau_base"])
            if dm is None or y is None or feature not in dm.feature_names:
                print(f"{sensor_name:<18}  (no {feature} in design — insufficient active rows)")
                continue
            j = dm.feature_names.index(feature)
            valid = ~np.isnan(y)
            for k in range(dm.X.shape[1]):
                valid &= ~np.isnan(dm.X[:, k])
            X, yv, th = dm.X[valid], y[valid], ts_hours[valid]
            state_v = state[valid]

            row = f"{sensor_name:<18}"
            for sigma in SIGMAS_HOURS:
                w = np.ones(len(yv)) if sigma is None else transition_weights(state_v, th, sigma)
                info = float((w * X[:, j] ** 2).sum())
                if info < 5.0:
                    row += f"{'(info<5)':>20}"
                    continue
                beta, t = weighted_ridge(X, yv, w, dm.feature_names)
                row += f"{beta[j]:>+8.3f}{t[j]:>6.1f}{info:>6.0f}"
            print(row)

    print("\nHOLDS as σ shrinks → causal (airspace); COLLAPSES → habit correlation.")
    print("Verdicts require info well above the gate at that σ.")


if __name__ == "__main__":
    main()
