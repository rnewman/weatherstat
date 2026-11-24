"""Virtual regulation sensors: the input each regulating effector regulates on (PLAN-41).

A regulating effector (mini split) is *delegated*: it runs its own thermostat,
comparing its target to the temperature it reads, and an inverter varies its
output. Sysid learns the effect side — when the unit runs at activity *a*,
sensor s changes at gₛ·*a* °F/hr. This module supplies the cause side:

**What the unit reads.** Weatherstat provides it. For each regulating effector,
V = Σ wₛ·Tₛ over the constrained sensors it warms, with wₛ ∝ its learned
gain gₛ (positive, after the simulator's load-time filters). With a quadratic
comfort cost, the cost changes with activity in proportion to Σ gₛ·(prefₛ − Tₛ),
so (target − V) is the comfort gradient along the one direction the unit can push
the house: between planning cycles the unit's fast loop works on the right
signal. V is published every minute (mqtt_bridge) and the ESP feeds it to the
unit as its remote temperature. The simulator evaluates the same formula on
simulated temperatures, so the unit's input is known by construction —
nothing about it is fitted, and nothing about which rooms it serves is
configured.

Frozen or unavailable members are dropped and the rest renormalized; below
``regulation_min_coverage`` of the weight there is no value (the unit falls back
to its internal thermistor, which beats a wrong number).

**Stability.** Weights are recomputed after each sysid refit but adopted only
when membership changes or a weight moves more than
``regulation_weight_tolerance``: refit noise would otherwise step V and cycle the
unit. Adopted sets live in ``regulation_state.json``; every adoption is appended
to ``regulation_history.jsonl`` so any published V can be reproduced.

**How the unit responds.** Learned per unit and per mode from history: the
probability the unit is running (``hvac_action`` heating/cooling in the matching
mode) as a saturating ramp in its error,
level = clip((sign·(target − V) + offset) / band, 0, 1). Fitting it to observed
action keeps it on the same scale as the gains, which are fitted against action.
Units do not stop at their setpoint: one with a built-in heating offset holds
2–3° past it, and the same unit cools to a different point, so heat and cool
are fitted separately (a pooled fit splits the difference and is right for
neither). The per-mode curve says which setpoint makes the unit hold a level
(planner, ``setpoint_for``). The configured ``proportional_band`` is only the
fallback before a fit exists.

**Hysteresis.** A unit's thermostat has a deadband: once running it carries on
past the point where an idle unit would start, and switching it into a mode
starts it if it is anywhere short of where it would stop. One curve averages
those cases, so it credits a setpoint the unit would ignore with a little
activity. Each mode therefore also has three transition curves, fitted on
consecutive 5-minute rows: ``start`` (idle in the mode a row earlier),
``stay`` (running) and ``enter`` (just switched into the mode). The simulator
runs them as a two-state chain from what the unit is doing now
(``HouseState.unit_states``): the running probability is
a·stay(err) + (1 − a)·start(err) while the plan keeps the unit in its mode, and
enter(err) on the step it changes mode. A setpoint past the far side of the
deadband simulates as exactly idle, and the planner folds it into off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weatherstat.config import DATA_DIR, EFFECTORS, PREDICTION_SENSORS
from weatherstat.yaml_config import load_config

_CFG = load_config()

REGULATION_STATE_FILE = DATA_DIR / "regulation_state.json"
REGULATION_HISTORY_FILE = DATA_DIR / "regulation_history.jsonl"

MIN_COVERAGE: float = _CFG.regulation_min_coverage
WEIGHT_TOLERANCE: float = _CFG.regulation_weight_tolerance

# Response fit, on rows where the unit is in heat or cool mode: a grid over where the
# unit is half-on (° past its setpoint, in the configured unit; + = it runs on beyond
# the setpoint) and the width of the ramp around that point. The grid reaches well
# past the setpoint because units really do regulate there. Trailing window so a
# change in what the unit reads (e.g. switching from an HA aggregate to V) is followed.
RESPONSE_WINDOW_DAYS = 60
RESPONSE_MIN_ROWS = 500  # per fit: a mode with fewer rows uses the pooled curve
TRANSITION_MIN_ROWS = 100  # per transition fit: fewer rows and that transition uses the mode's curve
TRANSITIONS = ("start", "stay", "enter")
_CONTIGUOUS = pd.Timedelta(minutes=7)  # consecutive collector rows (5 min apart, with slack)
_HALF_ON = np.arange(-2.0, 4.01, 0.25)
_BANDS = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
RESPONSE_MODES = {"heat": 1.0, "cool": -1.0}


# ── Types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResponseCurve:
    """How a delegated unit's activity follows its error: clip((err + offset) / band, 0, 1)."""

    offset: float
    band: float
    n_rows: int = 0
    brier: float | None = None  # mean squared error vs observed running (0/1); None = not fitted
    # Per-mode fits ("heat", "cool"). This curve itself is the pooled one, used for a
    # mode without its own fit.
    modes: dict[str, ResponseCurve] = field(default_factory=dict)
    # Per-mode curves only: P(running) given what the unit was doing a row earlier
    # (``TRANSITIONS``). A missing transition falls back to this curve.
    transitions: dict[str, ResponseCurve] = field(default_factory=dict)

    @property
    def half_on(self) -> float:
        """How far past its setpoint the unit is half-on (negative: before reaching it)."""
        return self.offset - self.band / 2.0

    def for_mode(self, mode: str) -> ResponseCurve:
        return self.modes.get(mode, self)

    def transition(self, kind: str) -> ResponseCurve:
        return self.transitions.get(kind, self)

    @property
    def has_transitions(self) -> bool:
        return bool(self.transitions) or any(m.transitions for m in self.modes.values())

    def level(self, err: np.ndarray | float) -> np.ndarray:
        return np.clip((np.asarray(err, dtype=float) + self.offset) / self.band, 0.0, 1.0)

    def level_for(self, err: np.ndarray, sign: np.ndarray, kind: str | None = None) -> np.ndarray:
        """``level`` per scenario: the heat curve where ``sign`` > 0, the cool curve where < 0.

        ``kind`` (one of ``TRANSITIONS``) selects that transition curve of each mode.
        """

        def curve(c: ResponseCurve) -> ResponseCurve:
            return c.transition(kind) if kind else c

        if not self.modes:
            return curve(self).level(err)
        heat, cool = curve(self.for_mode("heat")).level(err), curve(self.for_mode("cool")).level(err)
        return np.where(np.asarray(sign) >= 0, heat, cool)

    def setpoint_for(self, hold: float, sign: float) -> float:
        """The setpoint that makes the unit hold its input near ``hold``.

        The unit is half-on when its input reads ``hold``: above it (heating,
        ``sign`` +1) it backs off and idles, below it it works harder. A unit
        that keeps heating until it reads 1.25° over its setpoint is told 1.1°
        less than the level wanted.
        """
        return hold + sign * (self.band / 2.0 - self.offset)

    def holds_at(self, setpoint: float, sign: float) -> float:
        """Inverse of ``setpoint_for``: the input level a setpoint holds."""
        return setpoint - sign * (self.band / 2.0 - self.offset)


@dataclass(frozen=True)
class RegulationModel:
    """Everything the simulator and planner need about one regulating effector."""

    effector: str
    weights: dict[str, float]  # sensor -> weight (sums to 1 over all members)
    response: ResponseCurve
    adopted_at: str | None = None  # None = computed from current gains, not yet adopted


@dataclass(frozen=True)
class VirtualReading:
    """One evaluation of V from current readings."""

    value: float | None  # None when coverage is below MIN_COVERAGE
    used: dict[str, float]  # sensor -> renormalized weight actually applied
    dropped: dict[str, str] = field(default_factory=dict)  # sensor -> reason
    coverage: float = 0.0  # share of total weight available


# ── Weights ───────────────────────────────────────────────────────────────


def regulating_effectors() -> list[str]:
    return [e.name for e in EFFECTORS if e.control_type == "regulating"]


def compute_weights(
    gains: dict[tuple[str, str], tuple[float, float]],
    effector: str,
    sensors: list[str] | None = None,
) -> dict[str, float]:
    """wₛ ∝ positive gain of ``effector`` on each constrained sensor; {} if none."""
    allowed = set(sensors if sensors is not None else PREDICTION_SENSORS)
    raw = {s: g for (e, s), (g, _lag) in gains.items() if e == effector and s in allowed and g > 0}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {s: round(g / total, 4) for s, g in sorted(raw.items(), key=lambda kv: -kv[1])}


def weights_differ(old: dict[str, float], new: dict[str, float], tolerance: float = WEIGHT_TOLERANCE) -> bool:
    if set(old) != set(new):
        return True
    return any(abs(old[s] - new[s]) > tolerance for s in new)


def load_adopted() -> dict[str, dict]:
    """effector -> {"weights": {...}, "adopted_at": iso}; {} when no file."""
    try:
        data = json.loads(REGULATION_STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {str(k): v for k, v in data.get("effectors", {}).items()}


def adopt_weights(
    gains: dict[tuple[str, str], tuple[float, float]],
    sensors: list[str] | None = None,
    *,
    now: datetime | None = None,
    tolerance: float = WEIGHT_TOLERANCE,
    log=print,
) -> list[str]:
    """Adopt freshly computed weights where they differ enough. Returns effectors changed.

    Call after a sysid refit with the simulator's filtered gains
    (``load_sim_params().gains``) so the published V and the simulation agree.
    """
    now = now or datetime.now(UTC)
    adopted = load_adopted()
    changed: list[str] = []
    for name in regulating_effectors():
        new = compute_weights(gains, name, sensors)
        old = adopted.get(name, {}).get("weights")
        if not new:
            continue  # no positive gains (yet): keep whatever was adopted
        if old is not None and not weights_differ(old, new, tolerance):
            continue
        adopted[name] = {"weights": new, "adopted_at": now.isoformat()}
        changed.append(name)
        with open(REGULATION_HISTORY_FILE, "a") as f:
            f.write(json.dumps({"ts": now.isoformat(), "effector": name, "weights": new, "previous": old}) + "\n")
        log(f"  [regulation] {name}: adopted weights " + ", ".join(f"{s}={w:.2f}" for s, w in new.items()))
    if changed:
        tmp = REGULATION_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"effectors": adopted}, indent=2))
        tmp.replace(REGULATION_STATE_FILE)
    return changed


def build_models(
    gains: dict[tuple[str, str], tuple[float, float]],
    responses: dict[str, ResponseCurve],
    sensors: list[str] | None = None,
    adopted: dict[str, dict] | None = None,
) -> dict[str, RegulationModel]:
    """Regulation models for the simulator: adopted weights when present, else computed."""
    from weatherstat.config import EFFECTOR_MAP

    adopted = load_adopted() if adopted is None else adopted
    allowed = set(sensors) if sensors is not None else None
    models: dict[str, RegulationModel] = {}
    for name in regulating_effectors():
        entry = adopted.get(name)
        weights = dict(entry["weights"]) if entry else compute_weights(gains, name, sensors)
        if allowed is not None:
            weights = {s: w for s, w in weights.items() if s in allowed}
        if not weights:
            continue
        eff = EFFECTOR_MAP[name]
        response = responses.get(name) or ResponseCurve(offset=0.0, band=eff.proportional_band)
        models[name] = RegulationModel(
            effector=name, weights=weights, response=response,
            adopted_at=entry.get("adopted_at") if entry else None,
        )
    return models


# ── The virtual input ─────────────────────────────────────────────────────


def virtual_reading(
    weights: dict[str, float],
    temps: dict[str, float],
    *,
    unavailable: dict[str, str] | None = None,
    min_coverage: float = MIN_COVERAGE,
) -> VirtualReading:
    """V over the available members, renormalized. ``unavailable``: sensor -> reason (e.g. frozen)."""
    unavailable = unavailable or {}
    dropped: dict[str, str] = {}
    present: dict[str, float] = {}
    for s, w in weights.items():
        if s in unavailable:
            dropped[s] = unavailable[s]
        elif s not in temps or temps[s] is None or not np.isfinite(temps[s]):
            dropped[s] = "no reading"
        else:
            present[s] = w
    total = sum(weights.values())
    coverage = sum(present.values()) / total if total > 0 else 0.0
    if not present or coverage < min_coverage:
        return VirtualReading(value=None, used={}, dropped=dropped, coverage=round(coverage, 3))
    norm = sum(present.values())
    used = {s: w / norm for s, w in present.items()}
    value = sum(used[s] * float(temps[s]) for s in used)
    return VirtualReading(value=round(value, 2), used=used, dropped=dropped, coverage=round(coverage, 3))


# ── Learned response ──────────────────────────────────────────────────────


def _fit_curve(err: np.ndarray, running: np.ndarray, min_rows: int = RESPONSE_MIN_ROWS) -> ResponseCurve | None:
    """Best ramp for P(running) vs error, by mean squared error over the grid."""
    if len(err) < min_rows:
        return None
    scores = np.array([
        [float(np.mean((np.clip((err + h + b / 2.0) / b, 0.0, 1.0) - running) ** 2)) for b in _BANDS]
        for h in _HALF_ON
    ])
    # A unit that reports whole degrees leaves its switch point undetermined between
    # two readings: every half-on value in that gap scores the same. Take the middle of
    # the tie (and the widest ramp there), not whichever edge the grid reaches first.
    tied = np.argwhere(scores <= scores.min() + 1e-9)
    halves = sorted({int(i) for i, _ in tied})
    i_h = halves[len(halves) // 2]
    i_b = max(int(j) for i, j in tied if int(i) == i_h)
    half_on, band = float(_HALF_ON[i_h]), float(_BANDS[i_b])
    return ResponseCurve(
        offset=half_on + band / 2.0, band=band, n_rows=int(len(err)), brier=round(float(scores[i_h, i_b]), 4),
    )


def fit_response(df: pd.DataFrame, effector: str, window_days: float = RESPONSE_WINDOW_DAYS) -> ResponseCurve | None:
    """Fit a unit's response from its history: P(running) vs its own error, per mode.

    Uses the unit's reported temperature (``<name>_temp``: whatever it regulates
    on — an HA aggregate before switch-over, V after), its target and mode, and
    ``hvac_action`` as the running signal. Rows outside heat/cool mode carry no
    information about the response and are skipped.

    Returns the pooled curve with ``modes`` holding a separate fit for each mode
    that has enough rows, each with its ``transitions``: the same fit on rows whose
    previous row (the snapshot 5 minutes earlier) was idle in the mode (``start``),
    running in it (``stay``) or in another mode (``enter``). None when even the
    pooled fit lacks data.
    """
    cols = [f"{effector}_temp", f"{effector}_target", f"{effector}_mode", f"{effector}_action"]
    if any(c not in df.columns for c in cols):
        return None
    d = df[["timestamp", *cols]].copy()
    d["_ts"] = pd.to_datetime(d["timestamp"], format="ISO8601", utc=True)
    d = d[d["_ts"] >= d["_ts"].max() - pd.Timedelta(days=window_days)].sort_values("_ts")
    ts = d["_ts"]
    mode = d[f"{effector}_mode"].astype(str)
    sign = np.select([mode.eq("heat"), mode.eq("cool")], [1.0, -1.0], 0.0)
    v = pd.to_numeric(d[f"{effector}_temp"], errors="coerce").to_numpy(dtype=float)
    tgt = pd.to_numeric(d[f"{effector}_target"], errors="coerce").to_numpy(dtype=float)
    action = d[f"{effector}_action"].astype(str).to_numpy()
    running = np.where(sign > 0, action == "heating", action == "cooling").astype(float)
    ok = (sign != 0) & np.isfinite(v) & np.isfinite(tgt)
    if int(ok.sum()) < RESPONSE_MIN_ROWS:
        return None
    err_all = sign * (tgt - v)
    err = err_all[ok]
    y = running[ok]
    pooled = _fit_curve(err, y)
    if pooled is None:
        return None
    # The row before each row, when it is the previous collector snapshot.
    contiguous = np.zeros(len(d), dtype=bool)
    contiguous[1:] = (ts.iloc[1:].to_numpy() - ts.iloc[:-1].to_numpy()) <= _CONTIGUOUS.to_timedelta64()
    prev_sign = np.concatenate([[0.0], sign[:-1]])
    prev_running = np.concatenate([[0.0], running[:-1]])
    modes: dict[str, ResponseCurve] = {}
    for name, s in RESPONSE_MODES.items():
        sel = sign[ok] == s
        curve = _fit_curve(err[sel], y[sel])
        if curve is None:
            continue
        in_mode = ok & (sign == s)
        stayed = in_mode & contiguous & (prev_sign == s)
        rows = {
            "start": stayed & (prev_running == 0.0),
            "stay": stayed & (prev_running == 1.0),
            "enter": in_mode & contiguous & (prev_sign != s),
        }
        transitions = {
            kind: fit for kind, m in rows.items()
            if (fit := _fit_curve(err_all[m], running[m], min_rows=TRANSITION_MIN_ROWS)) is not None
        }
        modes[name] = replace(curve, transitions=transitions)
    return replace(pooled, modes=modes)


def fit_responses(df: pd.DataFrame) -> dict[str, ResponseCurve]:
    out: dict[str, ResponseCurve] = {}
    for name in regulating_effectors():
        curve = fit_response(df, name)
        if curve is not None:
            out[name] = curve
    return out


def _curve_to_json(r: ResponseCurve) -> dict:
    out: dict = {"offset": r.offset, "band": r.band, "n_rows": r.n_rows, "brier": r.brier}
    if r.modes:
        out["modes"] = {m: _curve_to_json(c) for m, c in r.modes.items()}
    if r.transitions:
        out["transitions"] = {k: _curve_to_json(c) for k, c in r.transitions.items()}
    return out


def _curve_from_json(v: dict) -> ResponseCurve:
    return ResponseCurve(
        offset=float(v["offset"]), band=float(v["band"]),
        n_rows=int(v.get("n_rows", 0)), brier=v.get("brier"),
        modes={str(m): _curve_from_json(c) for m, c in (v.get("modes") or {}).items()},
        transitions={str(k): _curve_from_json(c) for k, c in (v.get("transitions") or {}).items()},
    )


def responses_to_json(responses: dict[str, ResponseCurve]) -> dict[str, dict]:
    """``regulation_responses`` in thermal_params.json: the pooled curve, with per-mode fits
    (each with its ``transitions``) under ``modes``."""
    return {k: _curve_to_json(r) for k, r in responses.items()}


def responses_from_json(data: dict) -> dict[str, ResponseCurve]:
    """Inverse of ``responses_to_json``; older files have no ``modes`` or ``transitions``."""
    return {str(k): _curve_from_json(v) for k, v in (data or {}).items()}


def state_path() -> Path:
    return REGULATION_STATE_FILE
