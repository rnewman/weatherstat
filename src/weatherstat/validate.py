"""Automated validation for sysid results and simulator predictions.

Two layers:
1. Sysid diagnostics — statistical quality checks on fitted parameters
   (VIF, holdout RMSE, coefficient magnitude bounds). Run post-fit,
   before saving to thermal_params.json.
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


@dataclass(frozen=True)
class ValidationIssue:
    check: str  # e.g., "vif", "holdout_rmse", "prediction_range"
    sensor: str  # which sensor, or "" for global
    message: str
    severity: Severity = Severity.WARNING
    detail: dict = field(default_factory=dict)


# ── Sysid validation ─────────────────────────────────────────────────────

# Thresholds
_VIF_WARNING = 10.0  # standard multicollinearity threshold
_VIF_ERROR = 50.0  # severe — coefficients are meaningless
_ENV_TAU_BETA_MAX_FRAC = 0.5  # beta < 0.5 / tau_base (eff_tau > 2h for 40h base)
_ENV_SOLAR_BETA_MAX_RATIO = 2.0  # |β_solar_env| < 2 × base_solar_gain
_HOLDOUT_DEGRADATION = 0.20  # holdout RMSE > 20% worse than in-sample → warning
_CONDITION_NUMBER_WARNING = 1e5
_CONDITION_NUMBER_ERROR = 1e7


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


def validate_sysid_result(result: SysIdResult) -> list[ValidationIssue]:
    """Validate a SysIdResult for statistical and physical plausibility.

    Checks:
    1. Environment tau beta magnitude (can't make effective tau < 2h)
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
                issues.append(ValidationIssue(
                    check="env_tau_beta_magnitude",
                    sensor=ft.sensor,
                    message=(
                        f"Environment tau beta for {dev} is {beta:.4f} "
                        f"(eff tau={eff_tau:.1f}h, base={tau_base:.1f}h). "
                        f"Max allowed: {max_beta:.4f}."
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
    cond: float,
    beta: np.ndarray,
) -> list[ValidationIssue]:
    """Per-sensor regression diagnostics. Called from _fit_sensor_model.

    Checks:
    1. Condition number of design matrix
    2. VIF for environment and solar feature groups
    3. Holdout RMSE vs in-sample RMSE
    """
    issues: list[ValidationIssue] = []

    # Check 1: Condition number
    if cond > _CONDITION_NUMBER_ERROR:
        issues.append(ValidationIssue(
            check="condition_number",
            sensor=sensor_name,
            message=f"Design matrix condition number {cond:.0e} exceeds {_CONDITION_NUMBER_ERROR:.0e}. Coefficients are unreliable.",
            severity=Severity.ERROR,
            detail={"condition_number": float(cond)},
        ))
    elif cond > _CONDITION_NUMBER_WARNING:
        issues.append(ValidationIssue(
            check="condition_number",
            sensor=sensor_name,
            message=f"Design matrix condition number {cond:.0e} is high (>{_CONDITION_NUMBER_WARNING:.0e}). Coefficients may be unstable.",
            severity=Severity.WARNING,
            detail={"condition_number": float(cond)},
        ))

    # Check 2: VIF for advisory/solar features (skip effector lag features — too many)
    adv_solar_indices = [i for i, n in enumerate(feature_names) if n.startswith("_adv_") or n.startswith("_solar_")]
    if adv_solar_indices:
        # Compute VIF on the subset that includes these features + the base features
        # they might be collinear with. Use scaled X for consistency with regression.
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

    # Check 3: Holdout RMSE
    if len(y) >= 1000:  # only meaningful with enough data
        in_rmse, out_rmse = compute_holdout_rmse(X, y, scale, lam)
        if out_rmse > 0 and in_rmse > 0:
            degradation = (out_rmse - in_rmse) / in_rmse
            if degradation > _HOLDOUT_DEGRADATION:
                issues.append(ValidationIssue(
                    check="holdout_rmse",
                    sensor=sensor_name,
                    message=(
                        f"Holdout RMSE ({out_rmse:.4f}) is {degradation:.0%} worse than "
                        f"in-sample ({in_rmse:.4f}). Model may be overfitting."
                    ),
                    severity=Severity.WARNING,
                    detail={"in_sample_rmse": in_rmse, "holdout_rmse": out_rmse, "degradation": degradation},
                ))

    return issues


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
