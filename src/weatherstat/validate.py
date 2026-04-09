"""Automated validation for sysid results and simulator predictions.

Two layers:
1. Sysid diagnostics — statistical quality checks on fitted parameters
   (VIF, holdout RMSE, R², Durbin-Watson, bootstrap stability, coefficient
   magnitude bounds). Run post-fit, before saving to thermal_params.json.
2. Prediction envelope — physical plausibility checks on simulator output
   (temperature range, all-off convergence). Run after every predict() call.

Both return lists of ValidationIssue. Errors block saving/execution;
warnings are logged. The sysid quality gate in save_sysid_result() and
the TUI's periodic refit both call validate_sysid_result().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from weatherstat.config import abs_temp, delta_temp

if TYPE_CHECKING:

    from weatherstat.sysid import SysIdResult


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class SensorHealthGrade(StrEnum):
    """Per-sensor health grade. A = healthy, B = minor issues, C = marginal, F = broken."""
    A = "A"
    B = "B"
    C = "C"
    F = "F"


@dataclass(frozen=True)
class ValidationIssue:
    check: str  # e.g., "vif", "holdout_rmse", "prediction_range"
    sensor: str  # which sensor, or "" for global
    message: str
    severity: Severity = Severity.WARNING
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionDiagnostics:
    """Per-sensor diagnostics from the gain regression."""
    r_squared: float
    durbin_watson: float
    n_rows: int
    n_features: int
    holdout_degradation: float | None  # (out-in)/in, or None if too few rows
    bootstrap_cvs: dict[str, float]  # feature_name → CV (adv/solar features only)
    issues: list[ValidationIssue]


@dataclass(frozen=True)
class SensorHealth:
    """Rolled-up health assessment for one sensor."""
    sensor: str
    grade: SensorHealthGrade
    r_squared: float
    durbin_watson: float
    n_segments: int
    n_gains: int  # significant effector gains
    n_effectors: int  # total effectors
    n_advisory_betas: int  # kept advisory betas
    n_unstable_kept: int  # kept features with bootstrap CV > 1
    holdout_degradation: float | None
    notes: list[str]  # short issue descriptions


# ── Sysid validation ─────────────────────────────────────────────────────

# Thresholds
_VIF_WARNING = 10.0  # standard multicollinearity threshold
_VIF_ERROR = 50.0  # severe — coefficients are meaningless
_ENV_TAU_BETA_MAX_FRAC = 2.0  # beta < 2.0 / tau_base → eff_tau > tau_base/3
_ENV_SOLAR_BETA_MAX_RATIO = 2.0  # |β_solar_env| < 2 × base_solar_gain
_HOLDOUT_DEGRADATION = 0.20  # holdout RMSE > 20% worse than in-sample → warning
_BOOTSTRAP_CV_WARNING = 1.0  # coefficient of variation (std/|mean|) > 1 → unstable
_BOOTSTRAP_N = 50  # number of bootstrap resamples

# Health grade thresholds.
# R² is low by design: the dependent variable is the Newton cooling residual,
# and effectors are a small perturbation on top of sensor noise and thermal
# mass effects. R² of 0.08-0.20 indicates real signal.
# DW is low by design: the smoothed derivative (15-min half-window rolling mean)
# introduces structural autocorrelation. DW ≈ 0.25-0.40 is normal for this data;
# DW < 0.10 would indicate something broken beyond smoothing effects.
_GRADE_A_R2 = 0.08
_GRADE_A_DW = 0.20
_GRADE_A_SEGMENTS = 5
_GRADE_A_HOLDOUT = 0.20
_GRADE_B_R2 = 0.04
_GRADE_B_DW = 0.10
_GRADE_B_SEGMENTS = 3
_GRADE_B_HOLDOUT = 0.35
_GRADE_F_R2 = 0.02


def compute_vif(X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R²_j) where R²_j is from regressing feature j on all others.
    Uses ridge-stabilized regression to avoid singular matrices.
    """
    n_features = X.shape[1]
    vifs: dict[str, float] = {}
    for j in range(n_features):
        y_j = X[:, j]
        X_other = np.delete(X, j, axis=1)
        # Ridge-stabilized to handle collinear features without blowing up
        lam = 0.001 * len(y_j)
        XtX = X_other.T @ X_other + lam * np.eye(X_other.shape[1])
        try:
            beta = np.linalg.solve(XtX, X_other.T @ y_j)
            y_hat = X_other @ beta
            ss_res = np.sum((y_j - y_hat) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[feature_names[j]] = 1.0 / (1.0 - min(r_sq, 0.9999))
        except np.linalg.LinAlgError:
            vifs[feature_names[j]] = float("inf")
    return vifs


def compute_holdout_rmse(
    X: np.ndarray,
    y: np.ndarray,
    scale: np.ndarray,
    lam: float,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> tuple[float, float]:
    """Fit on (1-holdout_frac) of data, compute RMSE on holdout.

    Returns (in_sample_rmse, holdout_rmse).
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - holdout_frac))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Same ridge as sysid
    X_train_s = X_train / scale
    XsXs = X_train_s.T @ X_train_s + lam * np.eye(X_train_s.shape[1])
    try:
        beta_s = np.linalg.solve(XsXs, X_train_s.T @ y_train)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")

    beta = beta_s / scale
    in_rmse = float(np.sqrt(np.mean((y_train - X_train @ beta) ** 2)))
    out_rmse = float(np.sqrt(np.mean((y_test - X_test @ beta) ** 2)))
    return in_rmse, out_rmse


def compute_durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson statistic for residual autocorrelation.

    DW ≈ 2: no autocorrelation (good — model captures temporal structure).
    DW < 1.5: positive autocorrelation (systematic missing physics).
    DW > 2.5: negative autocorrelation (unusual, possible over-differencing).
    """
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / max(np.sum(residuals ** 2), 1e-30))


def compute_r_squared(y: np.ndarray, residuals: np.ndarray) -> float:
    """R² — fraction of variance in y explained by the model.

    For gain regression, y is the Newton residual (unexplained by pure
    cooling) and the model is effectors + solar + environment. R² = 0.3
    means those features explain 30% of what Newton cooling couldn't.
    """
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_bootstrap_stability(
    X: np.ndarray,
    y: np.ndarray,
    scale: np.ndarray,
    lam: float,
    n_boot: int = _BOOTSTRAP_N,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap coefficient stability: CV = std(β) / |mean(β)| per feature.

    Returns array of CVs (one per feature). CV > 1 means the coefficient
    changes sign or magnitude across resamples — it's not identified.
    """
    rng = np.random.RandomState(seed)
    n, p = X.shape
    betas = np.zeros((n_boot, p))
    X_s = X / scale
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        X_boot = X_s[idx]
        y_boot = y[idx]
        XtX = X_boot.T @ X_boot + lam * np.eye(p)
        try:
            beta_s = np.linalg.solve(XtX, X_boot.T @ y_boot)
            betas[i] = beta_s / scale
        except np.linalg.LinAlgError:
            betas[i] = np.nan

    means = np.nanmean(betas, axis=0)
    stds = np.nanstd(betas, axis=0)
    # CV = std / |mean|; undefined when mean is 0, treat as stable (coefficient is zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        cvs = np.where(np.abs(means) > 1e-10, stds / np.abs(means), 0.0)
    return cvs


def validate_sysid_result(result: SysIdResult) -> list[ValidationIssue]:
    """Validate a SysIdResult for statistical and physical plausibility.

    Checks:
    1. Environment tau beta magnitude (eff tau can't drop below tau_base/3)
    2. Environment solar beta magnitude (< 2× base solar gain)
    3. Overall fit quality (n_taus, n_gains)
    """
    issues: list[ValidationIssue] = []

    # Check 1: Environment tau beta magnitude
    for ft in result.fitted_taus:
        tau_base = ft.tau_base
        max_beta = _ENV_TAU_BETA_MAX_FRAC / tau_base if tau_base > 0 else float("inf")
        for dev, beta in ft.environment_tau_betas.items():
            if beta > max_beta:
                eff_tau = 1.0 / (1.0 / tau_base + beta) if beta > 0 else tau_base
                seg_note = f", {ft.n_segments} seg" if ft.n_segments < _GRADE_A_SEGMENTS else ""
                issues.append(ValidationIssue(
                    check="env_tau_beta_magnitude",
                    sensor=ft.sensor,
                    message=(
                        f"Environment tau beta for {dev} is {beta:.4f} "
                        f"(eff tau={eff_tau:.1f}h, base={tau_base:.1f}h{seg_note}). "
                        f"Effective tau < tau_base/3."
                    ),
                    severity=Severity.ERROR if eff_tau < 1.0 else Severity.WARNING,
                    detail={"device": dev, "beta": beta, "eff_tau": eff_tau, "tau_base": tau_base},
                ))

    # Check 2: Environment solar beta magnitude vs base solar gain
    for dev, sensor_betas in result.environment_solar_betas.items():
        for sensor_name, beta in sensor_betas.items():
            base_gain = result.solar_elevation_gains.get(sensor_name, 0.0)
            limit = max(abs(base_gain) * _ENV_SOLAR_BETA_MAX_RATIO, delta_temp(1.0))
            if abs(beta) > limit:
                issues.append(ValidationIssue(
                    check="env_solar_beta_magnitude",
                    sensor=sensor_name,
                    message=(
                        f"Environment solar beta for {dev}→{sensor_name} is {beta:+.4f}, "
                        f"but base solar gain is only {base_gain:+.4f}. "
                        f"Max allowed: ±{limit:.4f}."
                    ),
                    severity=Severity.ERROR,
                    detail={"device": dev, "beta": beta, "base_gain": base_gain},
                ))

    # Check 3: Zero-tau or zero-gain (existing quality gate, formalized)
    n_taus = len(result.fitted_taus)
    n_gains = sum(1 for g in result.effector_sensor_gains if not g.negligible)
    if n_taus == 0:
        issues.append(ValidationIssue(
            check="no_taus",
            sensor="",
            message="No sensors fitted — sysid produced no tau estimates.",
            severity=Severity.ERROR,
        ))
    if n_gains == 0:
        issues.append(ValidationIssue(
            check="no_gains",
            sensor="",
            message="No significant effector gains found.",
            severity=Severity.ERROR,
        ))

    return issues


def validate_sysid_regression(
    sensor_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    scale: np.ndarray,
    lam: float,
    beta: np.ndarray,
    residuals: np.ndarray,
) -> RegressionDiagnostics:
    """Per-sensor regression diagnostics. Called from _fit_sensor_model.

    Computes:
    1. R² — fraction of Newton residual explained by effectors + solar + environment
    2. Durbin-Watson — residual autocorrelation (missing physics detector)
    3. Bootstrap coefficient stability for advisory/solar features
    4. VIF for environment and solar feature groups
    5. Holdout RMSE vs in-sample RMSE

    Returns RegressionDiagnostics with issues and per-feature bootstrap CVs.
    Bootstrap CVs are returned for all adv/solar features but warnings are NOT
    emitted here — the caller (sysid.py) decides which features are kept and
    warns only for those with high CV.
    """
    issues: list[ValidationIssue] = []

    # R² and Durbin-Watson
    r_squared = compute_r_squared(y, residuals)
    durbin_watson = compute_durbin_watson(residuals)

    # Bootstrap coefficient stability for advisory/solar features
    # Computed for all adv/solar features; caller decides which matter.
    bootstrap_cvs: dict[str, float] = {}
    adv_solar_indices = [i for i, n in enumerate(feature_names) if n.startswith("_adv_") or n.startswith("_solar_")]
    if adv_solar_indices and len(y) >= 500:
        cvs = compute_bootstrap_stability(X, y, scale, lam)
        for idx in adv_solar_indices:
            bootstrap_cvs[feature_names[idx]] = float(cvs[idx])

    # VIF for advisory/solar features (skip effector lag features — too many)
    if adv_solar_indices:
        X_s = X / scale
        vifs = compute_vif(X_s, feature_names)
        for idx in adv_solar_indices:
            name = feature_names[idx]
            vif = vifs.get(name, 1.0)
            if vif > _VIF_ERROR:
                issues.append(ValidationIssue(
                    check="vif",
                    sensor=sensor_name,
                    message=f"Feature '{name}' has VIF={vif:.1f} (>{_VIF_ERROR}). Severely collinear — coefficient is meaningless.",
                    severity=Severity.ERROR,
                    detail={"feature": name, "vif": vif},
                ))
            elif vif > _VIF_WARNING:
                issues.append(ValidationIssue(
                    check="vif",
                    sensor=sensor_name,
                    message=f"Feature '{name}' has VIF={vif:.1f} (>{_VIF_WARNING}). Collinear — coefficient may be unstable.",
                    severity=Severity.WARNING,
                    detail={"feature": name, "vif": vif},
                ))

    # Holdout RMSE
    holdout_degradation: float | None = None
    if len(y) >= 1000:
        in_rmse, out_rmse = compute_holdout_rmse(X, y, scale, lam)
        if out_rmse > 0 and in_rmse > 0:
            holdout_degradation = (out_rmse - in_rmse) / in_rmse
            if holdout_degradation > _HOLDOUT_DEGRADATION:
                issues.append(ValidationIssue(
                    check="holdout_rmse",
                    sensor=sensor_name,
                    message=(
                        f"Holdout RMSE ({out_rmse:.4f}) is {holdout_degradation:.0%} worse than "
                        f"in-sample ({in_rmse:.4f}). Model may be overfitting."
                    ),
                    severity=Severity.WARNING,
                    detail={"in_sample_rmse": in_rmse, "holdout_rmse": out_rmse, "degradation": holdout_degradation},
                ))

    return RegressionDiagnostics(
        r_squared=r_squared,
        durbin_watson=durbin_watson,
        n_rows=len(y),
        n_features=X.shape[1],
        holdout_degradation=holdout_degradation,
        bootstrap_cvs=bootstrap_cvs,
        issues=issues,
    )


# ── Health grading ────────────────────────────────────────────────────────


def compute_sensor_health(
    sensor: str,
    *,
    r_squared: float,
    durbin_watson: float,
    n_segments: int,
    n_gains: int,
    n_effectors: int,
    n_advisory_betas: int,
    n_unstable_kept: int,
    holdout_degradation: float | None,
    has_validation_errors: bool,
) -> SensorHealth:
    """Grade a sensor's sysid health from its diagnostics."""
    notes: list[str] = []

    # F: broken — no data or catastrophic
    if n_segments == 0 or has_validation_errors:
        reason = "no segments" if n_segments == 0 else "validation errors"
        notes.append(reason)
        return SensorHealth(
            sensor=sensor, grade=SensorHealthGrade.F,
            r_squared=r_squared, durbin_watson=durbin_watson,
            n_segments=n_segments, n_gains=n_gains, n_effectors=n_effectors,
            n_advisory_betas=n_advisory_betas, n_unstable_kept=n_unstable_kept,
            holdout_degradation=holdout_degradation, notes=notes,
        )
    if n_gains == 0:
        notes.append("no significant gains")
    if r_squared < _GRADE_F_R2:
        notes.append(f"R²={r_squared:.2f}")

    if n_gains == 0 or r_squared < _GRADE_F_R2:
        return SensorHealth(
            sensor=sensor, grade=SensorHealthGrade.F,
            r_squared=r_squared, durbin_watson=durbin_watson,
            n_segments=n_segments, n_gains=n_gains, n_effectors=n_effectors,
            n_advisory_betas=n_advisory_betas, n_unstable_kept=n_unstable_kept,
            holdout_degradation=holdout_degradation, notes=notes,
        )

    # Check A thresholds
    is_a = (
        r_squared >= _GRADE_A_R2
        and durbin_watson >= _GRADE_A_DW
        and n_segments >= _GRADE_A_SEGMENTS
        and n_unstable_kept == 0
        and (holdout_degradation is None or holdout_degradation <= _GRADE_A_HOLDOUT)
    )
    if is_a:
        return SensorHealth(
            sensor=sensor, grade=SensorHealthGrade.A,
            r_squared=r_squared, durbin_watson=durbin_watson,
            n_segments=n_segments, n_gains=n_gains, n_effectors=n_effectors,
            n_advisory_betas=n_advisory_betas, n_unstable_kept=n_unstable_kept,
            holdout_degradation=holdout_degradation, notes=notes,
        )

    # Check B thresholds
    is_b = (
        r_squared >= _GRADE_B_R2
        and durbin_watson >= _GRADE_B_DW
        and n_segments >= _GRADE_B_SEGMENTS
        and (holdout_degradation is None or holdout_degradation <= _GRADE_B_HOLDOUT)
    )

    # Build notes for non-A sensors
    if r_squared < _GRADE_A_R2:
        notes.append(f"low R² ({r_squared:.2f})")
    if durbin_watson < _GRADE_A_DW:
        notes.append(f"low DW ({durbin_watson:.2f})")
    if n_segments < _GRADE_A_SEGMENTS:
        notes.append(f"{n_segments} segments")
    if n_unstable_kept > 0:
        notes.append(f"{n_unstable_kept} unstable kept")
    if holdout_degradation is not None and holdout_degradation > _GRADE_A_HOLDOUT:
        notes.append(f"holdout +{holdout_degradation:.0%}")

    grade = SensorHealthGrade.B if is_b else SensorHealthGrade.C
    return SensorHealth(
        sensor=sensor, grade=grade,
        r_squared=r_squared, durbin_watson=durbin_watson,
        n_segments=n_segments, n_gains=n_gains, n_effectors=n_effectors,
        n_advisory_betas=n_advisory_betas, n_unstable_kept=n_unstable_kept,
        holdout_degradation=holdout_degradation, notes=notes,
    )


def format_health_summary(
    healths: list[SensorHealth],
    gain_changes: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> str:
    """Format a scannable health summary table.

    gain_changes: optional dict of (effector, sensor) → (old_gain, new_gain) for
    gains that changed significantly since last refit.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("SYSID HEALTH SUMMARY")
    lines.append("=" * 80)

    hdr = (
        f"  {'Sensor':<30s} {'Grade':>5s} {'R²':>5s} {'DW':>5s}"
        f" {'Seg':>4s} {'Gains':>6s} {'Adv β':>5s}  Notes"
    )
    lines.append(hdr)
    lines.append("  " + "─" * 76)

    grade_counts: dict[str, int] = {}
    problem_sensors: list[str] = []

    for h in healths:
        grade_counts[h.grade] = grade_counts.get(h.grade, 0) + 1
        if h.grade not in ("A", "B"):
            problem_sensors.append(h.sensor)

        r2_str = f".{int(h.r_squared * 100):02d}" if h.r_squared < 1.0 else "1.00"
        dw_str = f"{h.durbin_watson:.2f}"
        gains_str = f"{h.n_gains}/{h.n_effectors}"
        adv_str = str(h.n_advisory_betas) if h.n_advisory_betas > 0 else "–"
        notes_str = ", ".join(h.notes) if h.notes else ""

        lines.append(
            f"  {h.sensor:<30s} {h.grade:>5s} {r2_str:>5s} {dw_str:>5s}"
            f" {h.n_segments:4d} {gains_str:>6s} {adv_str:>5s}  {notes_str}"
        )

    lines.append("  " + "─" * 76)

    # Gain stability
    if gain_changes:
        lines.append("")
        lines.append("  Gain drift since last refit:")
        for (eff, sensor), (old_g, new_g) in sorted(gain_changes.items()):
            pct = ((new_g - old_g) / abs(old_g) * 100) if abs(old_g) > 1e-6 else 0
            lines.append(f"    {eff} → {sensor}: {old_g:+.3f} → {new_g:+.3f} ({pct:+.0f}%)")

    # Rollup
    grade_parts = []
    for g in ("A", "B", "C", "F"):
        if grade_counts.get(g, 0) > 0:
            grade_parts.append(f"{grade_counts[g]}{g}")
    rollup = " ".join(grade_parts)

    if problem_sensors:
        short = [s.removesuffix("_temp") for s in problem_sensors]
        lines.append(f"\n  Overall: {rollup} — attention needed: {', '.join(short)}")
    else:
        lines.append(f"\n  Overall: {rollup} — all sensors healthy")

    return "\n".join(lines)


# ── Prediction validation ─────────────────────────────────────────────────

# Physical bounds for indoor temperature predictions
_INDOOR_MIN = abs_temp(35.0)  # nothing indoors should go below 35°F
_INDOOR_MAX = abs_temp(100.0)  # nothing indoors should go above 100°F
_MAX_RISE_PER_HOUR = delta_temp(8.0)  # max temperature rise per hour (effectors + solar)
_MAX_DROP_PER_HOUR = delta_temp(8.0)  # max temperature drop per hour


def validate_predictions(
    predictions: np.ndarray,
    target_names: list[str],
    current_temps: dict[str, float],
    outdoor_temp: float,
    horizons: list[int],
) -> list[ValidationIssue]:
    """Validate prediction array for physical plausibility.

    Args:
        predictions: shape (n_scenarios, n_targets) from predict()
        target_names: e.g. ["bedroom_temp_t+12", "bedroom_temp_t+24", ...]
        current_temps: sensor -> current temperature
        outdoor_temp: current outdoor temperature
        horizons: prediction horizons in 5-min steps

    Returns list of issues. Errors indicate predictions that are physically
    impossible and suggest model parameter problems.
    """
    issues: list[ValidationIssue] = []

    # Parse target names to get (sensor, horizon_steps) pairs
    targets: list[tuple[str, int]] = []
    for name in target_names:
        parts = name.rsplit("_t+", 1)
        if len(parts) == 2:
            sensor = parts[0]
            try:
                steps = int(parts[1])
                targets.append((sensor, steps))
            except ValueError:
                continue

    n_scenarios = predictions.shape[0]

    for col_idx, (sensor, steps) in enumerate(targets):
        hours = steps / 12.0  # 5-min steps to hours
        current = current_temps.get(sensor)
        if current is None:
            continue

        preds = predictions[:, col_idx]

        # Check 1: Absolute temperature bounds
        pred_min = float(np.nanmin(preds))
        pred_max = float(np.nanmax(preds))

        if pred_max > _INDOOR_MAX:
            n_violating = int(np.sum(preds > _INDOOR_MAX))
            issues.append(ValidationIssue(
                check="prediction_range_high",
                sensor=sensor,
                message=(
                    f"{sensor} at {hours:.0f}h: {n_violating}/{n_scenarios} scenarios "
                    f"predict >{_INDOOR_MAX:.0f}°F (max={pred_max:.1f}°F, current={current:.1f}°F)."
                ),
                severity=Severity.ERROR,
                detail={"horizon_hours": hours, "pred_max": pred_max, "current": current, "n_violating": n_violating},
            ))

        if pred_min < _INDOOR_MIN:
            n_violating = int(np.sum(preds < _INDOOR_MIN))
            issues.append(ValidationIssue(
                check="prediction_range_low",
                sensor=sensor,
                message=(
                    f"{sensor} at {hours:.0f}h: {n_violating}/{n_scenarios} scenarios "
                    f"predict <{_INDOOR_MIN:.0f}°F (min={pred_min:.1f}°F, current={current:.1f}°F)."
                ),
                severity=Severity.ERROR,
                detail={"horizon_hours": hours, "pred_min": pred_min, "current": current, "n_violating": n_violating},
            ))

        # Check 2: Rate of change — max plausible temperature change
        max_change = max(_MAX_RISE_PER_HOUR, _MAX_DROP_PER_HOUR) * hours
        median_pred = float(np.nanmedian(preds))
        if abs(median_pred - current) > max_change:
            issues.append(ValidationIssue(
                check="prediction_rate",
                sensor=sensor,
                message=(
                    f"{sensor} at {hours:.0f}h: median prediction {median_pred:.1f}°F is "
                    f"{abs(median_pred - current):.1f}°F from current {current:.1f}°F "
                    f"(max plausible: {max_change:.1f}°F in {hours:.0f}h)."
                ),
                severity=Severity.WARNING,
                detail={"horizon_hours": hours, "median_pred": median_pred, "current": current, "max_change": max_change},
            ))

        # Check 3: Scenario spread — if predictions vary wildly, model is unstable
        pred_spread = pred_max - pred_min
        if pred_spread > delta_temp(30.0):
            issues.append(ValidationIssue(
                check="prediction_spread",
                sensor=sensor,
                message=(
                    f"{sensor} at {hours:.0f}h: prediction spread is {pred_spread:.1f}°F "
                    f"across {n_scenarios} scenarios (min={pred_min:.1f}, max={pred_max:.1f})."
                ),
                severity=Severity.WARNING,
                detail={"horizon_hours": hours, "spread": pred_spread, "pred_min": pred_min, "pred_max": pred_max},
            ))

    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    """Return True if any issue has ERROR severity."""
    return any(i.severity == Severity.ERROR for i in issues)


def format_issues(issues: list[ValidationIssue]) -> str:
    """Format issues for display."""
    if not issues:
        return "  All checks passed."
    lines: list[str] = []
    for issue in issues:
        prefix = "ERROR" if issue.severity == Severity.ERROR else "WARNING"
        sensor_part = f" [{issue.sensor}]" if issue.sensor else ""
        lines.append(f"  {prefix}{sensor_part}: {issue.message}")
    return "\n".join(lines)
