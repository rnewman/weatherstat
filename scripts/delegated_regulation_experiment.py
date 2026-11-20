"""PLAN-41 experiment: do we need delivery data, or is "the unit converges on its target" enough?

For each variant, rebuild the mini splits' lagged regression features from a different
activity signal and refit sysid on the same data. Tau segment selection is unchanged
(it reads _eff_*, which we leave alone), so only the regression differs.

Variants:
  none                 split activity = 0 (how much do the splits explain at all?)
  action               current: hvac_action encoding (heating=1, cooling=-1)
  ramp(o,b)            delegated: sign · clip((sign·(target − V) + o) / b, 0, 1)
  sigmoid(o,s)         delegated: sign · σ((sign·(target − V) + o) / s)
where V is the unit's reported temperature (what it regulates on) and sign is +1 for heat,
−1 for cool, 0 otherwise. "o" shifts the switch point (o > 0 runs before reaching
target − V = 0); b / s set how fast it saturates at max power.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time

import numpy as np
import pandas as pd

import weatherstat.sysid as sysid
from weatherstat.config import EFFECTORS
from weatherstat.extract import load_collector_snapshots

UNITS = tuple(e.name for e in EFFECTORS if e.control_type == "regulating")
_orig_preprocess = sysid._preprocess


def _sign(mode: pd.Series) -> np.ndarray:
    return np.select([mode.eq("heat"), mode.eq("cool")], [1.0, -1.0], 0.0)


def activity(df: pd.DataFrame, unit: str, kind: str, o: float, b: float) -> pd.Series:
    if kind == "none":
        return pd.Series(0.0, index=df.index)
    if kind == "action":
        return df[f"_eff_{unit}"]
    v = pd.to_numeric(df[f"{unit}_temp"], errors="coerce").ffill()
    tgt = pd.to_numeric(df[f"{unit}_target"], errors="coerce").ffill()
    sign = _sign(df[f"{unit}_mode"].ffill().astype(str))
    err = sign * (tgt - v).to_numpy()
    if kind == "ramp":
        level = np.clip((err + o) / b, 0.0, 1.0)
    elif kind == "sigmoid":
        level = 1.0 / (1.0 + np.exp(-(err + o) / b))
    else:
        raise ValueError(kind)
    out = sign * np.nan_to_num(level)
    return pd.Series(out, index=df.index)


def make_preprocess(kind: str, o: float, b: float):
    def _pre(df, effectors, sensors):
        df = _orig_preprocess(df, effectors, sensors)
        for eff in effectors:
            if eff.name not in UNITS:
                continue
            a = activity(df, eff.name, kind, o, b)
            for (start, end), label in sysid._lag_bins(eff.max_lag_minutes):
                acc = sum(a.shift(lag).fillna(0.0) for lag in range(start, end))
                df[f"_lag_{eff.name}_{label}"] = acc / max(end - start, 1)
        return df
    return _pre


def run(kind: str, o: float, b: float, snaps: pd.DataFrame) -> dict:
    sysid._preprocess = make_preprocess(kind, o, b)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result, diags = sysid.fit_sysid(snapshots=snaps.copy())
    finally:
        sysid._preprocess = _orig_preprocess
    gains = {(g.effector, g.sensor): (g.gain_f_per_hour, g.t_statistic) for g in result.effector_sensor_gains
             if g.effector in UNITS and not g.negligible}
    return {"r2": {s: d.r_squared for s, d in diags.items()},
            "holdout": {s: d.holdout_degradation for s, d in diags.items()},
            "gains": gains}


def main() -> None:
    variants: list[tuple[str, float, float]] = [("none", 0, 1), ("action", 0, 1)]
    for o in (-0.5, 0.0, 0.5, 1.0):
        for b in (0.5, 1.0, 2.0):
            variants.append(("ramp", o, b))
    for o, s in ((0.5, 0.25), (0.5, 0.5), (1.0, 0.5)):
        variants.append(("sigmoid", o, s))
    if len(sys.argv) > 1:
        variants = [v for v in variants if v[0] in sys.argv[1:]]

    with contextlib.redirect_stdout(io.StringIO()):
        snaps = load_collector_snapshots(quarantine=True)
    print(f"{len(snaps)} snapshots")

    rows = []
    for kind, o, b in variants:
        t0 = time.monotonic()
        r = run(kind, o, b, snaps)
        label = kind if kind in ("none", "action") else f"{kind}(o={o:+.1f},{'b' if kind == 'ramp' else 's'}={b})"
        rows.append((label, r))
        print(f"  {label:28s} {time.monotonic() - t0:5.1f}s", flush=True)

    sensors = sorted(rows[0][1]["r2"])
    base = dict(rows)["none"]["r2"]
    print("\nΔR² vs no-split model, ×1000 (higher = split activity explains more of dT/dt)")
    short = [s.removesuffix("_temp").replace("thermostat_", "t_")[:9] for s in sensors]
    print(f"{'variant':28s} " + " ".join(f"{n:>9s}" for n in short) + "      sum")
    for label, r in rows:
        d = [1000 * (r["r2"][s] - base[s]) for s in sensors]
        print(f"{label:28s} " + " ".join(f"{x:9.2f}" for x in d) + f" {sum(d):8.2f}")

    print("\nKept split gains (°F/hr per unit activity, t) — action vs best delegated variant")
    best = max((x for x in rows if x[0] not in ("none", "action")),
               key=lambda x: sum(x[1]["r2"][s] - base[s] for s in sensors))
    for label, r in (("action", dict(rows)["action"]), best):
        print(f"  {label}")
        for (eff, sen), (g, t) in sorted(r["gains"].items()):
            print(f"    {eff:24s} → {sen:28s} {g:+.3f}  t={t:6.1f}")


if __name__ == "__main__":
    main()
