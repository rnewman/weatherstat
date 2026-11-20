"""PLAN-41 offline check: does the regulating-effector model predict room temperatures
at least as well as the model it replaces?

Replays history at points where every effector's command and state held constant for
the whole window (the simulator takes one constant command per effector, and the
controller re-plans every 5 minutes, so a plan is not what happened — a constant
window is). The same scenario is predicted and compared with what the sensors read.

Run the same command in two checkouts (old vs new simulator) and compare the JSON:

    uv run python scripts/backtest_regulation.py --out new.json
    (cd <old worktree> && uv run python <this script> --out old.json)

With ``--input UNIT=SENSOR[,SENSOR…]`` (new simulator only, repeatable) a unit's
input V is the equal-weight mean of those sensors (whatever the unit actually read in
the period, e.g. a hand-built HA average) instead of the gain-weighted V, and the
response curves are fitted from history, as sysid would. The period defaults to the
last 60 days of snapshots; pick one before any change to what the units read.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd

from weatherstat.config import EFFECTORS, PREDICTION_SENSORS
from weatherstat.extract import load_collector_snapshots
from weatherstat.simulator import HouseState, extract_recent_history, load_sim_params, predict
from weatherstat.types import EffectorDecision, Scenario
from weatherstat.weather import SOLAR_PLANES, effective_solar_fraction, plane_incidence, solar_sin_elevation
from weatherstat.yaml_config import load_config

_CFG = load_config()
HORIZONS = [12, 24]  # 1h, 2h


def _f(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="UTC start (default: --days before --end)")
    ap.add_argument("--end", help="UTC end (default: the newest snapshot)")
    ap.add_argument("--days", type=float, default=60.0)
    ap.add_argument("--stride", type=int, default=6, help="snapshots between candidate points")
    ap.add_argument("--input", action="append", default=[], metavar="UNIT=SENSOR[,SENSOR...]",
                    help="what a unit read in the period (equal weights); repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        df = load_collector_snapshots()
        params = load_sim_params()
    if args.input:
        from weatherstat.regulation import RegulationModel, fit_responses

        inputs: dict[str, dict[str, float]] = {}
        for spec in args.input:
            unit, _, sensors = spec.partition("=")
            members = [s for s in sensors.split(",") if s]
            if not members:
                ap.error(f"--input {spec!r}: expected UNIT=SENSOR[,SENSOR...]")
            inputs[unit] = {s: 1.0 / len(members) for s in members}
        responses = fit_responses(df)
        params = replace(params, regulation={
            name: RegulationModel(name, w, responses[name]) for name, w in inputs.items() if name in responses
        })
        print("responses:", {k: (v.offset, v.band) for k, v in responses.items()})

    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)
    lat, lon = _CFG.location.latitude, _CFG.location.longitude
    tz = _CFG.location.timezone

    traj = [e for e in EFFECTORS if e.control_type == "trajectory"]
    binary = [e for e in EFFECTORS if e.control_type == "binary"]
    regulating = [e for e in EFFECTORS if e.control_type == "regulating"]
    command_cols = [f"{e.name}_action" for e in traj] + [f"{e.name}_mode" for e in binary]
    command_cols += [c for e in regulating for c in (f"{e.name}_mode", f"{e.name}_target")]
    command_cols = [c for c in command_cols if c in df.columns]
    df[command_cols] = df[command_cols].ffill()
    env_cols = [c.column for c in _CFG.environment.values() if c.column in df.columns]
    df[env_cols] = df[env_cols].ffill()

    end = pd.Timestamp(args.end, tz="UTC") if args.end else df["_ts"].iloc[-1]
    start = pd.Timestamp(args.start, tz="UTC") if args.start else end - pd.Timedelta(days=args.days)
    lo = df["_ts"].searchsorted(start)
    hi = df["_ts"].searchsorted(end)
    H = max(HORIZONS)
    errors: dict[str, dict[int, list[float]]] = {s: {h: [] for h in HORIZONS} for s in PREDICTION_SENSORS}
    errors_split: dict[str, dict[int, list[float]]] = {s: {h: [] for h in HORIZONS} for s in PREDICTION_SENSORS}
    n_windows = n_split = 0

    for i in range(lo, min(hi, len(df) - H), args.stride):
        win = df.iloc[i:i + H + 1]
        t0 = win["_ts"].iloc[0]
        if win["_ts"].iloc[-1] - t0 != timedelta(minutes=5 * H):
            continue  # gap in the record
        if any(win[c].astype(str).nunique() > 1 for c in command_cols):
            continue  # a command or state changed inside the window
        row = win.iloc[0]
        temps = {s: v for s in PREDICTION_SENSORS if s in df.columns and (v := _f(row[s])) is not None}
        if len(temps) < len(PREDICTION_SENSORS) - 1:
            continue

        effs: dict[str, EffectorDecision] = {}
        for e in traj:
            if str(row.get(f"{e.name}_action")) == "heating":
                effs[e.name] = EffectorDecision(e.name, mode="heating")
        for e in binary:
            mode = str(row.get(f"{e.name}_mode", "off"))
            if mode in e.supported_modes and mode != "off":
                effs[e.name] = EffectorDecision(e.name, mode=mode)
        split_active = False
        for e in regulating:
            mode = str(row.get(f"{e.name}_mode", "off"))
            tgt = _f(row.get(f"{e.name}_target"))
            if mode in ("heat", "cool") and tgt is not None:
                effs[e.name] = EffectorDecision(e.name, mode=mode, target=tgt)
                split_active = True

        forecast = [v for h in range(1, 13) if (v := _f(row.get(f"forecast_temp_{h}h"))) is not None]
        outdoor = forecast[0] if forecast else (_f(row.get("met_outdoor_temp")) or 50.0)
        sf = [effective_solar_fraction(_f(row.get("cloud_coverage")), str(row.get("weather_condition", "unknown")))]
        for h in range(1, len(forecast) + 1):
            sf.append(effective_solar_fraction(_f(row.get(f"forecast_cloud_{h}h")),
                                               str(row.get(f"forecast_condition_{h}h", "unknown"))))
        steps = [t0.to_pydatetime() + timedelta(minutes=5 * k) for k in range(1, H + 1)]
        local = t0.tz_convert(tz)
        state = HouseState(
            current_temps=temps, outdoor_temp=outdoor, forecast_temps=forecast,
            environment_states={n: str(row.get(c.column, "0")) in ("1", "1.0", "True") for n, c in _CFG.environment.items()},
            hour_of_day=local.hour + local.minute / 60.0,
            recent_history=extract_recent_history(df.iloc[max(0, i - 40):i + 1], params),
            solar_fractions=sf,
            solar_elevations=[solar_sin_elevation(lat, lon, t) for t in steps],
            solar_planes={p: [plane_incidence(lat, lon, t, az) for t in steps] for p, az in SOLAR_PLANES.items()},
        )
        with contextlib.redirect_stdout(io.StringIO()):
            names, pred = predict(state, [Scenario(effectors=effs)], params, HORIZONS)
        n_windows += 1
        n_split += split_active
        for j, name in enumerate(names):
            sensor, h = name.rsplit("_t+", 1)
            actual = _f(df.iloc[i + int(h)][sensor])
            if actual is None:
                continue
            err = float(pred[0, j]) - actual
            errors[sensor][int(h)].append(err)
            if split_active:
                errors_split[sensor][int(h)].append(err)

    def _summ(e: dict[str, dict[int, list[float]]]) -> dict:
        return {
            s: {str(h): {"n": len(v), "rmse": float(np.sqrt(np.mean(np.square(v)))) if v else None,
                         "bias": float(np.mean(v)) if v else None} for h, v in hs.items()}
            for s, hs in e.items()
        }

    out = {"n_windows": n_windows, "n_split_active": n_split, "all": _summ(errors), "split_active": _summ(errors_split)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{n_windows} windows ({n_split} with a split running) → {args.out}")


if __name__ == "__main__":
    main()
