"""Tests for validation module — sysid diagnostics and prediction envelope."""

from __future__ import annotations

import numpy as np

from weatherstat.validate import (
    RegressionDiagnostics,
    SensorHealthGrade,
    Severity,
    ValidationIssue,
    compute_bootstrap_stability,
    compute_durbin_watson,
    compute_holdout_rmse,
    compute_r_squared,
    compute_sensor_health,
    compute_vif,
    format_health_summary,
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


# ── Durbin-Watson ──────────────────────────────────────────────────────


class TestDurbinWatson:
    def test_white_noise_near_two(self) -> None:
        """White noise residuals should have DW ≈ 2."""
        rng = np.random.RandomState(42)
        residuals = rng.randn(1000)
        dw = compute_durbin_watson(residuals)
        assert 1.8 < dw < 2.2, f"DW={dw:.2f} should be ~2.0 for white noise"

    def test_positive_autocorrelation_low_dw(self) -> None:
        """Positively autocorrelated residuals should have DW < 1.5."""
        rng = np.random.RandomState(42)
        # AR(1) process with strong positive autocorrelation
        n = 1000
        residuals = np.zeros(n)
        residuals[0] = rng.randn()
        for i in range(1, n):
            residuals[i] = 0.9 * residuals[i - 1] + rng.randn() * 0.3
        dw = compute_durbin_watson(residuals)
        assert dw < 1.0, f"DW={dw:.2f} should be <1.0 for strong AR(1)"


# ── R-squared ──────────────────────────────────────────────────────────


class TestRSquared:
    def test_perfect_fit(self) -> None:
        """Zero residuals should give R² = 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.zeros(5)
        assert compute_r_squared(y, residuals) == 1.0

    def test_no_explanatory_power(self) -> None:
        """Residuals equal to y should give R² ≈ 0."""
        rng = np.random.RandomState(42)
        y = rng.randn(100)
        residuals = y - np.mean(y)  # model predicts the mean
        r2 = compute_r_squared(y, residuals)
        assert abs(r2) < 0.01, f"R²={r2:.3f} should be ~0"

    def test_partial_fit(self) -> None:
        """Partial fit should give 0 < R² < 1."""
        rng = np.random.RandomState(42)
        y = rng.randn(100) + 2.0
        residuals = rng.randn(100) * 0.5  # model explains most variance
        r2 = compute_r_squared(y, residuals)
        assert 0.5 < r2 < 1.0, f"R²={r2:.3f} should be between 0.5 and 1"


# ── Bootstrap stability ─────────────────────────────────────────────────


class TestBootstrapStability:
    def test_stable_coefficients_low_cv(self) -> None:
        """Well-identified coefficients should have low bootstrap CV."""
        rng = np.random.RandomState(42)
        n = 2000
        X = rng.randn(n, 3)
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + rng.randn(n) * 0.1
        scale = np.ones(3)
        lam = 0.01 * n
        cvs = compute_bootstrap_stability(X, y, scale, lam, n_boot=30)
        for j in range(3):
            assert cvs[j] < 0.5, f"Feature {j} CV={cvs[j]:.2f} should be low for well-identified coeff"

    def test_unidentified_coefficient_high_cv(self) -> None:
        """A noise feature uncorrelated with y should have high CV (if nonzero)."""
        rng = np.random.RandomState(42)
        n = 500
        x_signal = rng.randn(n)
        x_noise = rng.randn(n)
        X = np.column_stack([x_signal, x_noise])
        y = x_signal * 1.0 + rng.randn(n) * 0.5
        scale = np.ones(2)
        lam = 0.001 * n  # weak regularization so noise feature gets nonzero β
        cvs = compute_bootstrap_stability(X, y, scale, lam, n_boot=50)
        # Signal feature should be stable, noise feature less so
        assert cvs[0] < cvs[1], f"Signal CV={cvs[0]:.2f} should be lower than noise CV={cvs[1]:.2f}"


# ── Sysid regression validation ─────────────────────────────────────────


class TestValidateSysidRegression:
    def test_clean_regression_returns_diagnostics(self) -> None:
        """Well-conditioned regression should return clean diagnostics."""
        rng = np.random.RandomState(42)
        n = 2000
        X = rng.randn(n, 5)
        beta_true = np.array([0.5, -0.3, 0.1, 0.0, 0.2])
        y = X @ beta_true + rng.randn(n) * 0.1
        residuals = y - X @ beta_true
        feature_names = ["eff_a", "eff_b", "_adv_tau_win", "_solar_elev", "_adv_solar_shade"]
        scale = np.ones(5)
        lam = 0.01 * n

        diag = validate_sysid_regression("test_sensor", X, y, feature_names, scale, lam, beta_true, residuals)
        assert isinstance(diag, RegressionDiagnostics)
        assert diag.r_squared > 0.9  # near-perfect fit
        assert 1.5 < diag.durbin_watson < 2.5  # white noise residuals
        assert diag.n_rows == n
        assert diag.n_features == 5
        errors = [i for i in diag.issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Expected no errors: {[i.message for i in errors]}"
        # Bootstrap CVs should be returned for advisory/solar features
        assert "_adv_tau_win" in diag.bootstrap_cvs
        assert "_solar_elev" in diag.bootstrap_cvs

    def test_high_vif_detected(self) -> None:
        """Collinear advisory features should trigger VIF warning/error."""
        rng = np.random.RandomState(42)
        n = 2000
        solar = rng.randn(n)
        # Make them highly correlated by biasing window_open toward high solar
        solar = np.abs(solar)
        window_open = (solar > 0.5).astype(float)  # window open when sunny!
        adv_solar = window_open * solar

        X = np.column_stack([rng.randn(n), solar, adv_solar])
        feature_names = ["eff_a", "_solar_elev", "_adv_solar_office"]
        y = rng.randn(n)
        residuals = y.copy()
        scale = np.ones(3)
        for j in [1, 2]:
            s = np.std(X[:, j])
            if s > 0:
                scale[j] = s
        lam = 0.01 * n

        diag = validate_sysid_regression("test_sensor", X, y, feature_names, scale, lam, np.zeros(3), residuals)
        vif_issues = [i for i in diag.issues if i.check == "vif"]
        assert len(vif_issues) > 0, "Expected VIF warning for collinear solar features"

    def test_bootstrap_cvs_not_warned(self) -> None:
        """Bootstrap CVs are returned but NOT warned about (caller decides)."""
        rng = np.random.RandomState(42)
        n = 2000
        x_signal = rng.randn(n)
        x_noise = rng.randn(n)
        X = np.column_stack([x_signal, x_noise])
        feature_names = ["_adv_tau_win", "_adv_solar_shade"]
        y = x_signal * 0.5 + rng.randn(n) * 2.0  # noise feature poorly identified
        residuals = y - x_signal * 0.5
        scale = np.ones(2)
        lam = 0.001 * n

        diag = validate_sysid_regression("test_sensor", X, y, feature_names, scale, lam, np.array([0.5, 0.01]), residuals)
        # CVs are returned
        assert len(diag.bootstrap_cvs) == 2
        # No bootstrap warnings emitted (caller handles this)
        bootstrap_issues = [i for i in diag.issues if i.check == "bootstrap_stability"]
        assert len(bootstrap_issues) == 0, "Bootstrap warnings should not be emitted by validate_sysid_regression"


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
        """An environment tau beta that makes eff tau < tau_base/3 should be flagged."""
        from weatherstat.sysid import FittedTau

        # beta = 0.5 → eff_tau = 1/(1/45+0.5) ≈ 1.96h, which is well below 45/3=15h
        result = self._make_result(
            fitted_taus=[FittedTau(
                sensor="bedroom_temp", tau_base=45.0, n_segments=10,
                environment_tau_betas={"door_bedroom": 0.5},
            )],
        )
        issues = validate_sysid_result(result)
        tau_issues = [i for i in issues if i.check == "env_tau_beta_magnitude"]
        assert len(tau_issues) > 0, "Expected tau beta magnitude issue"

    def test_moderate_tau_beta_passes(self) -> None:
        """A tau beta giving eff tau > tau_base/3 should NOT be flagged."""
        from weatherstat.sysid import FittedTau

        # beta = 0.02 → eff_tau = 1/(1/45+0.02) ≈ 22.5h, above 45/3=15h
        result = self._make_result(
            fitted_taus=[FittedTau(
                sensor="bedroom_temp", tau_base=45.0, n_segments=10,
                environment_tau_betas={"window_bedroom": 0.02},
            )],
        )
        issues = validate_sysid_result(result)
        tau_issues = [i for i in issues if i.check == "env_tau_beta_magnitude"]
        assert len(tau_issues) == 0, f"Moderate beta should pass: {[i.message for i in tau_issues]}"

    def test_tau_beta_includes_segment_count(self) -> None:
        """Tau beta warning should mention segment count when low."""
        from weatherstat.sysid import FittedTau

        result = self._make_result(
            fitted_taus=[FittedTau(
                sensor="bedroom_temp", tau_base=45.0, n_segments=3,
                environment_tau_betas={"door_bedroom": 0.5},
            )],
        )
        issues = validate_sysid_result(result)
        tau_issues = [i for i in issues if i.check == "env_tau_beta_magnitude"]
        assert len(tau_issues) > 0
        assert "3 seg" in tau_issues[0].message

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


# ── Health grading ──────────────────────────────────────────────────────


class TestSensorHealth:
    def test_grade_a(self) -> None:
        """Good diagnostics should produce grade A."""
        health = compute_sensor_health(
            "bedroom_temp",
            r_squared=0.12, durbin_watson=0.30,
            n_segments=13, n_gains=6, n_effectors=7,
            n_advisory_betas=2, n_unstable_kept=0,
            holdout_degradation=0.10, has_validation_errors=False,
        )
        assert health.grade == SensorHealthGrade.A
        assert len(health.notes) == 0

    def test_grade_b_holdout(self) -> None:
        """Moderate holdout degradation should produce grade B."""
        health = compute_sensor_health(
            "kitchen_temp",
            r_squared=0.06, durbin_watson=0.30,
            n_segments=13, n_gains=4, n_effectors=7,
            n_advisory_betas=2, n_unstable_kept=0,
            holdout_degradation=0.28, has_validation_errors=False,
        )
        assert health.grade == SensorHealthGrade.B
        assert any("holdout" in n for n in health.notes)

    def test_grade_c_low_segments_and_holdout(self) -> None:
        """Very few segments + high holdout degradation should produce grade C."""
        health = compute_sensor_health(
            "living_room_climate_temp",
            r_squared=0.06, durbin_watson=0.25,
            n_segments=2, n_gains=3, n_effectors=7,
            n_advisory_betas=1, n_unstable_kept=0,
            holdout_degradation=0.40, has_validation_errors=False,
        )
        assert health.grade == SensorHealthGrade.C
        assert any("segment" in n for n in health.notes)
        assert any("holdout" in n for n in health.notes)

    def test_grade_f_no_gains(self) -> None:
        """No significant gains should produce grade F."""
        health = compute_sensor_health(
            "broken_sensor",
            r_squared=0.08, durbin_watson=0.25,
            n_segments=5, n_gains=0, n_effectors=7,
            n_advisory_betas=0, n_unstable_kept=0,
            holdout_degradation=None, has_validation_errors=False,
        )
        assert health.grade == SensorHealthGrade.F
        assert any("no significant gains" in n for n in health.notes)

    def test_grade_f_validation_errors(self) -> None:
        """Validation errors should produce grade F regardless of other metrics."""
        health = compute_sensor_health(
            "bad_sensor",
            r_squared=0.20, durbin_watson=0.35,
            n_segments=13, n_gains=6, n_effectors=7,
            n_advisory_betas=2, n_unstable_kept=0,
            holdout_degradation=0.05, has_validation_errors=True,
        )
        assert health.grade == SensorHealthGrade.F

    def test_unstable_kept_blocks_grade_a(self) -> None:
        """Unstable kept features should prevent grade A."""
        health = compute_sensor_health(
            "sensor",
            r_squared=0.15, durbin_watson=0.30,
            n_segments=13, n_gains=6, n_effectors=7,
            n_advisory_betas=2, n_unstable_kept=1,
            holdout_degradation=0.10, has_validation_errors=False,
        )
        assert health.grade != SensorHealthGrade.A
        assert any("unstable" in n for n in health.notes)


class TestHealthSummary:
    def test_format_all_healthy(self) -> None:
        """All A grades should produce 'all sensors healthy' rollup."""
        healths = [
            compute_sensor_health(
                "sensor_a", r_squared=0.12, durbin_watson=0.30,
                n_segments=10, n_gains=5, n_effectors=7,
                n_advisory_betas=2, n_unstable_kept=0,
                holdout_degradation=0.10, has_validation_errors=False,
            ),
            compute_sensor_health(
                "sensor_b", r_squared=0.09, durbin_watson=0.25,
                n_segments=8, n_gains=4, n_effectors=7,
                n_advisory_betas=1, n_unstable_kept=0,
                holdout_degradation=0.15, has_validation_errors=False,
            ),
        ]
        output = format_health_summary(healths)
        assert "all sensors healthy" in output
        assert "2A" in output

    def test_format_with_problems(self) -> None:
        """Mixed grades should list problem sensors."""
        healths = [
            compute_sensor_health(
                "good_temp", r_squared=0.12, durbin_watson=0.30,
                n_segments=10, n_gains=5, n_effectors=7,
                n_advisory_betas=2, n_unstable_kept=0,
                holdout_degradation=None, has_validation_errors=False,
            ),
            compute_sensor_health(
                "bad_temp", r_squared=0.01, durbin_watson=0.05,
                n_segments=2, n_gains=0, n_effectors=7,
                n_advisory_betas=0, n_unstable_kept=0,
                holdout_degradation=None, has_validation_errors=False,
            ),
        ]
        output = format_health_summary(healths)
        assert "attention needed" in output
        assert "bad" in output

    def test_format_with_gain_drift(self) -> None:
        """Gain drift should be listed in the summary."""
        healths = [
            compute_sensor_health(
                "sensor_temp", r_squared=0.12, durbin_watson=0.30,
                n_segments=10, n_gains=5, n_effectors=7,
                n_advisory_betas=2, n_unstable_kept=0,
                holdout_degradation=None, has_validation_errors=False,
            ),
        ]
        changes = {("thermostat", "sensor_temp"): (0.50, 0.80)}
        output = format_health_summary(healths, changes)
        assert "Gain drift" in output
        assert "thermostat" in output


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
