"""Tests for validation module — sysid diagnostics and prediction envelope."""

from __future__ import annotations

import numpy as np

from weatherstat.validate import (
    Severity,
    ValidationIssue,
    compute_holdout_rmse,
    compute_vif,
    format_issues,
    has_errors,
    validate_predictions,
    validate_sysid_regression,
    validate_sysid_result,
)

# ── VIF computation ──────────────────────────────────────────────────────


class TestComputeVIF:
    def test_independent_features_low_vif(self) -> None:
        """Uncorrelated features should have VIF close to 1."""
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 3)
        vifs = compute_vif(X, ["a", "b", "c"])
        for name, vif in vifs.items():
            assert vif < 2.0, f"{name} VIF={vif:.1f} should be ~1.0"

    def test_collinear_features_high_vif(self) -> None:
        """Highly correlated features should have high VIF."""
        rng = np.random.RandomState(42)
        x1 = rng.randn(1000)
        x2 = x1 + rng.randn(1000) * 0.01  # nearly identical
        x3 = rng.randn(1000)
        X = np.column_stack([x1, x2, x3])
        vifs = compute_vif(X, ["x1", "x2", "x3"])
        assert vifs["x1"] > 10.0, f"x1 VIF={vifs['x1']:.1f} should be high"
        assert vifs["x2"] > 10.0, f"x2 VIF={vifs['x2']:.1f} should be high"
        assert vifs["x3"] < 3.0, f"x3 VIF={vifs['x3']:.1f} should be low"

    def test_perfect_collinearity(self) -> None:
        """Perfectly collinear features should have very high VIF."""
        rng = np.random.RandomState(42)
        x1 = rng.randn(1000)
        x2 = x1 * 2.0  # perfect collinearity
        X = np.column_stack([x1, x2])
        vifs = compute_vif(X, ["x1", "x2"])
        # Ridge stabilization prevents inf but VIF should be very high
        assert vifs["x1"] > 100.0


# ── Holdout RMSE ─────────────────────────────────────────────────────────


class TestHoldoutRMSE:
    def test_well_specified_model(self) -> None:
        """A well-specified model should have similar in-sample and holdout RMSE."""
        rng = np.random.RandomState(42)
        n = 2000
        X = rng.randn(n, 3)
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + rng.randn(n) * 0.1
        scale = np.ones(3)
        lam = 0.01 * n

        in_rmse, out_rmse = compute_holdout_rmse(X, y, scale, lam)
        # Holdout should be close to in-sample for a well-specified model
        assert out_rmse < in_rmse * 1.3, f"holdout {out_rmse:.4f} >> in-sample {in_rmse:.4f}"

    def test_overfitting_model(self) -> None:
        """A model with many noise features should show holdout degradation."""
        rng = np.random.RandomState(42)
        n = 200
        # 3 real features + 50 noise features → overfit
        X_real = rng.randn(n, 3)
        X_noise = rng.randn(n, 50)
        X = np.column_stack([X_real, X_noise])
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X_real @ beta_true + rng.randn(n) * 0.5
        scale = np.ones(53)
        lam = 0.001 * n  # weak regularization → overfit

        in_rmse, out_rmse = compute_holdout_rmse(X, y, scale, lam)
        # Holdout should be meaningfully worse
        assert out_rmse > in_rmse * 1.1, f"expected holdout degradation: in={in_rmse:.4f} out={out_rmse:.4f}"


# ── Sysid regression validation ─────────────────────────────────────────


class TestValidateSysidRegression:
    def test_clean_regression_no_issues(self) -> None:
        """Well-conditioned regression with independent features produces no issues."""
        rng = np.random.RandomState(42)
        n = 2000
        X = rng.randn(n, 5)
        beta_true = np.array([0.5, -0.3, 0.1, 0.0, 0.2])
        y = X @ beta_true + rng.randn(n) * 0.1
        feature_names = ["eff_a", "eff_b", "_adv_tau_win", "_solar_elev", "_adv_solar_shade"]
        scale = np.ones(5)
        lam = 0.01 * n
        cond = np.linalg.cond(X / scale)

        issues = validate_sysid_regression("test_sensor", X, y, feature_names, scale, lam, cond, beta_true)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Expected no errors: {[i.message for i in errors]}"

    def test_high_vif_detected(self) -> None:
        """Collinear advisory features should trigger VIF warning/error."""
        rng = np.random.RandomState(42)
        n = 2000
        solar = rng.randn(n)
        window_open = (rng.rand(n) > 0.5).astype(float)
        adv_solar = window_open * solar  # correlated with solar when window is open a lot
        # Make them highly correlated by biasing window_open toward high solar
        solar = np.abs(solar)
        window_open = (solar > 0.5).astype(float)  # window open when sunny!
        adv_solar = window_open * solar

        X = np.column_stack([rng.randn(n), solar, adv_solar])
        feature_names = ["eff_a", "_solar_elev", "_adv_solar_office"]
        y = rng.randn(n)
        scale = np.ones(3)
        for j in [1, 2]:
            s = np.std(X[:, j])
            if s > 0:
                scale[j] = s
        lam = 0.01 * n
        cond = np.linalg.cond(X / scale)

        issues = validate_sysid_regression("test_sensor", X, y, feature_names, scale, lam, cond, np.zeros(3))
        vif_issues = [i for i in issues if i.check == "vif"]
        assert len(vif_issues) > 0, "Expected VIF warning for collinear solar features"

    def test_high_condition_number_detected(self) -> None:
        """Near-singular design matrix should trigger condition number warning."""
        rng = np.random.RandomState(42)
        n = 2000
        x1 = rng.randn(n)
        x2 = x1 + rng.randn(n) * 1e-6  # nearly identical
        X = np.column_stack([x1, x2])
        feature_names = ["_solar_elev", "_adv_solar_shade"]
        y = rng.randn(n)
        scale = np.ones(2)
        cond = np.linalg.cond(X / scale)

        issues = validate_sysid_regression("test_sensor", X, y, feature_names, scale, 0.01 * n, cond, np.zeros(2))
        cond_issues = [i for i in issues if i.check == "condition_number"]
        assert len(cond_issues) > 0, f"Expected condition number issue, cond={cond:.0e}"


# ── Sysid result validation ─────────────────────────────────────────────


class TestValidateSysidResult:
    @staticmethod
    def _make_result(**overrides):
        """Build a minimal SysIdResult-like object for testing."""
        from weatherstat.sysid import (
            EffectorSensorGain,
            FittedTau,
            SensorSpec,
            SysIdResult,
        )

        defaults = dict(
            timestamp="2026-04-09T00:00:00Z",
            data_start="2026-02-01",
            data_end="2026-04-09",
            n_snapshots=10000,
            effectors=[],
            sensors=[SensorSpec(name="bedroom_temp", temp_column="bedroom_temp", yaml_tau_base=45.0)],
            fitted_taus=[FittedTau(sensor="bedroom_temp", tau_base=45.0, n_segments=10)],
            effector_sensor_gains=[
                EffectorSensorGain(
                    effector="thermostat_upstairs", sensor="bedroom_temp",
                    gain_f_per_hour=0.4, best_lag_minutes=60, t_statistic=2.0, negligible=False,
                ),
            ],
            solar_gains=[],
            solar_elevation_gains={"bedroom_temp": 2.0},
            state_gates={},
            environment_solar_betas={},
        )
        defaults.update(overrides)
        return SysIdResult(**defaults)

    def test_clean_result_passes(self) -> None:
        """A well-formed result should produce no errors."""
        result = self._make_result()
        issues = validate_sysid_result(result)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_extreme_tau_beta_flagged(self) -> None:
        """An environment tau beta that makes effective tau < 2h should be flagged."""
        from weatherstat.sysid import FittedTau

        result = self._make_result(
            fitted_taus=[FittedTau(
                sensor="bedroom_temp", tau_base=45.0, n_segments=10,
                environment_tau_betas={"door_bedroom": 0.5},  # eff_tau = 1/(1/45+0.5) ≈ 1.96h
            )],
        )
        issues = validate_sysid_result(result)
        tau_issues = [i for i in issues if i.check == "env_tau_beta_magnitude"]
        assert len(tau_issues) > 0, "Expected tau beta magnitude issue"

    def test_extreme_solar_beta_flagged(self) -> None:
        """A solar beta much larger than base solar gain should be flagged."""
        result = self._make_result(
            solar_elevation_gains={"bedroom_temp": 2.0},
            environment_solar_betas={"blinds": {"bedroom_temp": 25.0}},  # 12.5× base gain
        )
        issues = validate_sysid_result(result)
        solar_issues = [i for i in issues if i.check == "env_solar_beta_magnitude"]
        assert len(solar_issues) > 0, "Expected solar beta magnitude issue"
        assert solar_issues[0].severity == Severity.ERROR

    def test_reasonable_solar_beta_passes(self) -> None:
        """A solar beta within 2× base gain should not be flagged."""
        result = self._make_result(
            solar_elevation_gains={"bedroom_temp": 2.0},
            environment_solar_betas={"blinds": {"bedroom_temp": -1.5}},  # 0.75× base, reasonable for blinds
        )
        issues = validate_sysid_result(result)
        solar_issues = [i for i in issues if i.check == "env_solar_beta_magnitude"]
        assert len(solar_issues) == 0

    def test_no_taus_is_error(self) -> None:
        result = self._make_result(fitted_taus=[])
        issues = validate_sysid_result(result)
        assert any(i.check == "no_taus" and i.severity == Severity.ERROR for i in issues)

    def test_no_gains_is_error(self) -> None:
        from weatherstat.sysid import EffectorSensorGain

        result = self._make_result(
            effector_sensor_gains=[
                EffectorSensorGain(
                    effector="thermostat_upstairs", sensor="bedroom_temp",
                    gain_f_per_hour=0.0, best_lag_minutes=0, t_statistic=0.0, negligible=True,
                ),
            ],
        )
        issues = validate_sysid_result(result)
        assert any(i.check == "no_gains" and i.severity == Severity.ERROR for i in issues)


# ── Prediction validation ────────────────────────────────────────────────


class TestValidatePredictions:
    def test_reasonable_predictions_pass(self) -> None:
        """Normal predictions within bounds should produce no issues."""
        predictions = np.array([[72.5, 73.0, 74.0, 75.0]])  # 1 scenario, 4 horizons
        target_names = ["bedroom_temp_t+12", "bedroom_temp_t+24", "bedroom_temp_t+48", "bedroom_temp_t+72"]
        current_temps = {"bedroom_temp": 72.0}
        issues = validate_predictions(predictions, target_names, current_temps, 45.0, [12, 24, 48, 72])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_absurd_high_prediction_flagged(self) -> None:
        """A prediction of 109°F should be flagged as ERROR."""
        predictions = np.array([[75.0, 85.0, 95.0, 109.5]])
        target_names = ["piano_temp_t+12", "piano_temp_t+24", "piano_temp_t+48", "piano_temp_t+72"]
        current_temps = {"piano_temp": 72.0}
        issues = validate_predictions(predictions, target_names, current_temps, 45.0, [12, 24, 48, 72])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) > 0, "Expected error for 109.5°F prediction"
        assert any("piano_temp" in e.message for e in errors)

    def test_absurd_low_prediction_flagged(self) -> None:
        """A prediction below 35°F should be flagged."""
        predictions = np.array([[60.0, 50.0, 40.0, 30.0]])
        target_names = ["bedroom_temp_t+12", "bedroom_temp_t+24", "bedroom_temp_t+48", "bedroom_temp_t+72"]
        current_temps = {"bedroom_temp": 65.0}
        issues = validate_predictions(predictions, target_names, current_temps, 20.0, [12, 24, 48, 72])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) > 0, "Expected error for <35°F prediction"

    def test_excessive_rate_flagged(self) -> None:
        """A 20°F rise in 1 hour should be flagged."""
        predictions = np.array([[92.0]])
        target_names = ["piano_temp_t+12"]
        current_temps = {"piano_temp": 72.0}
        issues = validate_predictions(predictions, target_names, current_temps, 45.0, [12])
        rate_issues = [i for i in issues if i.check == "prediction_rate"]
        assert len(rate_issues) > 0, "Expected rate warning for 20°F/hr change"

    def test_wide_scenario_spread_flagged(self) -> None:
        """A 40°F spread across scenarios should be flagged."""
        predictions = np.array([[60.0], [70.0], [95.0]])  # 3 scenarios, spread = 35°F
        target_names = ["bedroom_temp_t+72"]
        current_temps = {"bedroom_temp": 70.0}
        issues = validate_predictions(predictions, target_names, current_temps, 45.0, [72])
        spread_issues = [i for i in issues if i.check == "prediction_spread"]
        assert len(spread_issues) > 0, "Expected spread warning for 35°F spread"

    def test_multiple_sensors_checked(self) -> None:
        """All sensors in the prediction should be validated."""
        # bedroom fine, piano absurd
        predictions = np.array([[73.0, 109.0]])
        target_names = ["bedroom_temp_t+72", "piano_temp_t+72"]
        current_temps = {"bedroom_temp": 72.0, "piano_temp": 72.0}
        issues = validate_predictions(predictions, target_names, current_temps, 45.0, [72])
        piano_errors = [i for i in issues if i.severity == Severity.ERROR and "piano_temp" in i.message]
        bedroom_errors = [i for i in issues if i.severity == Severity.ERROR and "bedroom_temp" in i.message]
        assert len(piano_errors) > 0
        assert len(bedroom_errors) == 0


# ── Utility functions ────────────────────────────────────────────────────


class TestUtilities:
    def test_has_errors_true(self) -> None:
        issues = [ValidationIssue(check="test", sensor="", message="bad", severity=Severity.ERROR)]
        assert has_errors(issues)

    def test_has_errors_false(self) -> None:
        issues = [ValidationIssue(check="test", sensor="", message="meh", severity=Severity.WARNING)]
        assert not has_errors(issues)

    def test_format_issues_empty(self) -> None:
        assert "passed" in format_issues([])

    def test_format_issues_with_items(self) -> None:
        issues = [
            ValidationIssue(check="test", sensor="bedroom_temp", message="too high", severity=Severity.ERROR),
            ValidationIssue(check="test2", sensor="", message="something", severity=Severity.WARNING),
        ]
        output = format_issues(issues)
        assert "ERROR" in output
        assert "WARNING" in output
        assert "bedroom_temp" in output
