"""Tests for wrtds.regression module."""

import numpy as np
import pytest
from scipy import optimize, stats

from wrtds.regression import (
    compute_weights,
    jitter_sample,
    neg_log_likelihood_with_grad,
    predict,
    run_surv_reg,
    tricube,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_sample():
    """Synthetic sample data with known properties for regression tests.

    Generates data from: ln(c) = 1.0 + 0.01*t - 0.5*logQ + 0.1*sin + 0.05*cos + N(0, 0.3)
    """
    rng = np.random.default_rng(123)
    n = 200
    dec_year = 1990.0 + rng.uniform(0, 20, n)
    logq = rng.normal(2.0, 0.8, n)
    sin_dy = np.sin(2 * np.pi * dec_year)
    cos_dy = np.cos(2 * np.pi * dec_year)

    true_beta = np.array([1.0, 0.01, -0.5, 0.1, 0.05])
    sigma_true = 0.3
    X = np.column_stack([np.ones(n), dec_year, logq, sin_dy, cos_dy])
    y_true = X @ true_beta + rng.normal(0, sigma_true, n)
    conc = np.exp(y_true)

    # All uncensored
    return {
        'DecYear': dec_year,
        'LogQ': logq,
        'SinDY': sin_dy,
        'CosDY': cos_dy,
        'ConcLow': conc,
        'ConcHigh': conc,
        'Uncen': np.ones(n, dtype=bool),
    }, true_beta, sigma_true


@pytest.fixture
def censored_sample():
    """Synthetic sample with ~20% censored observations."""
    rng = np.random.default_rng(456)
    n = 200
    dec_year = 1990.0 + rng.uniform(0, 20, n)
    logq = rng.normal(2.0, 0.8, n)
    sin_dy = np.sin(2 * np.pi * dec_year)
    cos_dy = np.cos(2 * np.pi * dec_year)

    true_beta = np.array([1.0, 0.01, -0.5, 0.1, 0.05])
    sigma_true = 0.3
    X = np.column_stack([np.ones(n), dec_year, logq, sin_dy, cos_dy])
    y_true = X @ true_beta + rng.normal(0, sigma_true, n)
    conc = np.exp(y_true)

    # Censor the lowest ~20% at a detection limit
    detection_limit = np.percentile(conc, 20)
    censored = conc < detection_limit
    conc_low = np.where(censored, np.nan, conc)
    conc_high = np.where(censored, detection_limit, conc)
    uncen = ~censored

    return {
        'DecYear': dec_year,
        'LogQ': logq,
        'SinDY': sin_dy,
        'CosDY': cos_dy,
        'ConcLow': conc_low,
        'ConcHigh': conc_high,
        'Uncen': uncen,
    }, true_beta, sigma_true


# ---------------------------------------------------------------------------
# tricube
# ---------------------------------------------------------------------------

class TestTricube:
    def test_at_center(self):
        assert tricube(0.0, 5.0) == pytest.approx(1.0)

    def test_at_boundary(self):
        assert tricube(5.0, 5.0) == pytest.approx(0.0)

    def test_outside_window(self):
        assert tricube(6.0, 5.0) == pytest.approx(0.0)

    def test_half_window(self):
        h = 5.0
        d = 2.5
        expected = (1.0 - (0.5)**3) ** 3
        assert tricube(d, h) == pytest.approx(expected)

    def test_negative_distance(self):
        assert tricube(-2.5, 5.0) == tricube(2.5, 5.0)

    def test_array_input(self):
        d = np.array([-1.0, 0.0, 1.0, 5.0, 6.0])
        result = tricube(d, 5.0)
        assert result.shape == (5,)
        assert result[1] == pytest.approx(1.0)  # center
        assert result[3] == pytest.approx(0.0)  # boundary
        assert result[4] == pytest.approx(0.0)  # outside

    def test_monotone_decreasing(self):
        d = np.linspace(0, 5.0, 100)
        result = tricube(d, 5.0)
        assert np.all(np.diff(result) <= 0)


# ---------------------------------------------------------------------------
# compute_weights
# ---------------------------------------------------------------------------

class TestComputeWeights:
    def test_product_of_three(self):
        """Weight should be the product of time, discharge, and season weights."""
        dec_year = np.array([2000.0, 2001.0, 2002.0])
        logq = np.array([1.0, 2.0, 3.0])
        uncen = np.array([True, True, True])

        w = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2001.0, target_logq=2.0,
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=1, min_num_uncen=1,
        )

        # Manual check for the center observation (index 1)
        # All distances are 0 -> all tricube weights are 1 -> product is 1
        assert w[1] == pytest.approx(1.0)
        assert len(w) == 3

    def test_all_positive_for_close_obs(self):
        """Observations close to the target should get positive weight."""
        dec_year = np.array([2000.5])
        logq = np.array([2.0])
        uncen = np.array([True])

        w = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2000.5, target_logq=2.0,
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=0, min_num_uncen=0,
        )
        assert w[0] == pytest.approx(1.0)

    def test_zero_for_distant_obs(self):
        """Observation far from target should get zero weight."""
        dec_year = np.array([2020.0])
        logq = np.array([2.0])
        uncen = np.array([True])

        w = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2000.0, target_logq=2.0,
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=0, min_num_uncen=0,
        )
        assert w[0] == pytest.approx(0.0)

    def test_circular_season_distance(self):
        """Observations 1 year apart should have season distance ~0 (same season)."""
        dec_year = np.array([2000.0, 2001.0])
        logq = np.array([2.0, 2.0])
        uncen = np.array([True, True])

        w = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2000.0, target_logq=2.0,
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=0, min_num_uncen=0,
        )
        # 1 year apart -> season_dist = |1 - round(1)| = 0 -> season weight = 1
        # time_dist = 1 -> time weight = tricube(1, 7) > 0
        assert w[1] > 0

    def test_half_year_season_distance(self):
        """Observations 0.5 years apart have maximum season distance."""
        dec_year = np.array([2000.0, 2000.5])
        logq = np.array([2.0, 2.0])
        uncen = np.array([True, True])

        w = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2000.0, target_logq=2.0,
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=0, min_num_uncen=0,
        )
        # season_dist = |0.5 - round(0.5)| = 0.5 if round(0.5) = 0
        # But np.round(0.5) = 0 (banker's rounding), so season_dist = 0.5
        # tricube(0.5, 0.5) = (1 - 1^3)^3 = 0
        assert w[1] == pytest.approx(0.0)

    def test_edge_adjustment(self):
        """Near record start, time window should expand."""
        n = 200
        rng = np.random.default_rng(42)
        dec_year = 2000.0 + rng.uniform(0, 20, n)
        logq = rng.normal(2.0, 0.5, n)
        uncen = np.ones(n, dtype=bool)

        # Target near start of record
        w_edge = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2001.0, target_logq=2.0,
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=1, min_num_uncen=1,
            edge_adjust=True,
            record_start=2000.0, record_end=2020.0,
        )
        w_no_edge = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2001.0, target_logq=2.0,
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=1, min_num_uncen=1,
            edge_adjust=False,
        )
        # Edge adjustment should include more observations
        assert np.sum(w_edge > 0) >= np.sum(w_no_edge > 0)

    def test_window_expansion(self):
        """Windows expand when too few observations have nonzero weight."""
        # Only 5 observations — can't meet min_num_obs=100 without expansion
        dec_year = np.array([2000.0, 2001.0, 2002.0, 2003.0, 2004.0])
        logq = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        uncen = np.array([True, True, True, True, True])

        w = compute_weights(
            dec_year, logq, uncen,
            target_dec_year=2002.0, target_logq=2.0,
            window_y=1, window_q=0.5, window_s=0.5,
            min_num_obs=5, min_num_uncen=5,
        )
        # After expansion, all 5 should have nonzero weight
        assert np.sum(w > 0) == 5


# ---------------------------------------------------------------------------
# neg_log_likelihood_with_grad
# ---------------------------------------------------------------------------

class TestNegLogLikelihoodWithGrad:
    def _make_simple_problem(self):
        """Create a small test problem for NLL testing."""
        rng = np.random.default_rng(99)
        n = 20
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
        beta_true = np.array([1.0, 0.5, -0.3, 0.1, 0.2])
        sigma_true = 0.5
        y = X @ beta_true + rng.normal(0, sigma_true, n)

        # First 15 uncensored, last 5 censored
        uncen_mask = np.ones(n, dtype=bool)
        uncen_mask[15:] = False

        y_uncen = y.copy()
        y_cen_high = y.copy()
        # For censored obs, set y_cen_high to something above predicted
        y_cen_high[15:] = X[15:] @ beta_true + 0.5

        weights = np.ones(n)
        theta = np.concatenate([beta_true, [np.log(sigma_true)]])

        return theta, X, y_uncen, y_cen_high, uncen_mask, weights

    def test_cost_finite(self):
        theta, X, y_uncen, y_cen_high, uncen_mask, weights = self._make_simple_problem()
        cost, grad = neg_log_likelihood_with_grad(theta, X, y_uncen, y_cen_high, uncen_mask, weights)
        assert np.isfinite(cost)
        assert np.all(np.isfinite(grad))

    def test_cost_all_uncensored(self):
        """For all-uncensored data, NLL should equal weighted Gaussian NLL."""
        rng = np.random.default_rng(42)
        n = 10
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
        beta = np.array([1.0, 0.5, -0.3, 0.1, 0.2])
        sigma = 0.4
        y = X @ beta + rng.normal(0, sigma, n)

        uncen_mask = np.ones(n, dtype=bool)
        weights = np.ones(n)
        theta = np.concatenate([beta, [np.log(sigma)]])

        cost, _ = neg_log_likelihood_with_grad(theta, X, y, y, uncen_mask, weights)

        # Manual computation
        resid = y - X @ beta
        expected = np.sum(0.5 * (resid / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi))
        assert cost == pytest.approx(expected, rel=1e-10)

    def test_gradient_matches_numerical(self):
        """Analytical gradient must match finite-difference approximation."""
        theta, X, y_uncen, y_cen_high, uncen_mask, weights = self._make_simple_problem()

        def cost_only(t):
            c, _ = neg_log_likelihood_with_grad(t, X, y_uncen, y_cen_high, uncen_mask, weights)
            return c

        _, analytical_grad = neg_log_likelihood_with_grad(
            theta, X, y_uncen, y_cen_high, uncen_mask, weights
        )
        numerical_grad = optimize.approx_fprime(theta, cost_only, 1e-7)

        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)

    def test_gradient_all_uncensored(self):
        """Gradient check with no censored observations."""
        rng = np.random.default_rng(42)
        n = 15
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
        y = rng.normal(1.0, 0.5, n)
        uncen_mask = np.ones(n, dtype=bool)
        weights = rng.uniform(0.5, 1.5, n)
        theta = np.array([0.8, 0.1, -0.2, 0.05, 0.1, np.log(0.6)])

        def cost_only(t):
            c, _ = neg_log_likelihood_with_grad(t, X, y, y, uncen_mask, weights)
            return c

        _, analytical = neg_log_likelihood_with_grad(theta, X, y, y, uncen_mask, weights)
        numerical = optimize.approx_fprime(theta, cost_only, 1e-7)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_gradient_all_censored(self):
        """Gradient check with all observations censored."""
        rng = np.random.default_rng(42)
        n = 10
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
        y_cen_high = rng.normal(1.0, 0.5, n)
        y_uncen = np.full(n, np.nan)
        uncen_mask = np.zeros(n, dtype=bool)
        weights = np.ones(n)
        theta = np.array([0.5, 0.1, -0.1, 0.05, 0.0, np.log(0.5)])

        def cost_only(t):
            c, _ = neg_log_likelihood_with_grad(t, X, y_uncen, y_cen_high, uncen_mask, weights)
            return c

        _, analytical = neg_log_likelihood_with_grad(theta, X, y_uncen, y_cen_high, uncen_mask, weights)
        numerical = optimize.approx_fprime(theta, cost_only, 1e-7)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_gradient_with_varying_weights(self):
        """Gradient check with non-uniform weights and mixed censoring."""
        rng = np.random.default_rng(77)
        n = 25
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
        y = rng.normal(1.0, 0.5, n)
        y_cen_high = y + 0.3

        uncen_mask = np.ones(n, dtype=bool)
        uncen_mask[::3] = False  # every 3rd is censored

        weights = rng.uniform(0.1, 2.0, n)
        theta = np.array([1.0, -0.1, 0.3, 0.0, 0.1, np.log(0.4)])

        def cost_only(t):
            c, _ = neg_log_likelihood_with_grad(t, X, y, y_cen_high, uncen_mask, weights)
            return c

        _, analytical = neg_log_likelihood_with_grad(theta, X, y, y_cen_high, uncen_mask, weights)
        numerical = optimize.approx_fprime(theta, cost_only, 1e-7)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# run_surv_reg
# ---------------------------------------------------------------------------

class TestRunSurvReg:
    def test_recovers_parameters_uncensored(self, synthetic_sample):
        """MLE should approximately recover the true parameters on large uncensored data."""
        sample_data, true_beta, true_sigma = synthetic_sample
        weights = np.ones(len(sample_data['DecYear']))

        beta, sigma = run_surv_reg(sample_data, weights)

        # Intercept (beta[0]) is hard to pin down due to collinearity with
        # the time covariate (~2000), so check slope coefficients and sigma.
        np.testing.assert_allclose(beta[1:], true_beta[1:], atol=0.15)
        assert sigma == pytest.approx(true_sigma, abs=0.1)

        # Also verify predictions are close at a representative point
        yHat_est, _, _ = predict(beta, sigma, 2000.0, 2.0)
        yHat_true, _, _ = predict(true_beta, true_sigma, 2000.0, 2.0)
        assert float(yHat_est) == pytest.approx(float(yHat_true), abs=0.15)

    def test_converges_censored(self, censored_sample):
        """MLE should converge and return finite parameters with censored data."""
        sample_data, true_beta, true_sigma = censored_sample
        weights = np.ones(len(sample_data['DecYear']))

        beta, sigma = run_surv_reg(sample_data, weights)

        assert np.all(np.isfinite(beta))
        assert np.isfinite(sigma)
        assert sigma > 0

    def test_weighted_regression(self, synthetic_sample):
        """Weights should influence the regression — observations with higher
        weight should have more influence on the fit."""
        sample_data, _, _ = synthetic_sample
        n = len(sample_data['DecYear'])

        # Uniform weights
        w_uniform = np.ones(n)
        beta_uniform, _ = run_surv_reg(sample_data, w_uniform)

        # Weight only the first half heavily
        w_skewed = np.ones(n)
        w_skewed[:n // 2] = 10.0
        w_skewed[n // 2:] = 0.01
        beta_skewed, _ = run_surv_reg(sample_data, w_skewed)

        # Coefficients should differ
        assert not np.allclose(beta_uniform, beta_skewed, atol=1e-3)

    def test_returns_correct_shapes(self, synthetic_sample):
        sample_data, _, _ = synthetic_sample
        weights = np.ones(len(sample_data['DecYear']))
        beta, sigma = run_surv_reg(sample_data, weights)

        assert beta.shape == (5,)
        assert isinstance(sigma, float) or sigma.ndim == 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_scalar_input(self):
        beta = np.array([1.0, 0.01, -0.5, 0.1, 0.05])
        sigma = 0.3
        yHat, SE, ConcHat = predict(beta, sigma, 2000.0, 2.0)

        expected_yHat = (1.0 + 0.01 * 2000.0 - 0.5 * 2.0
                         + 0.1 * np.sin(2 * np.pi * 2000.0)
                         + 0.05 * np.cos(2 * np.pi * 2000.0))
        assert float(yHat) == pytest.approx(expected_yHat)
        assert float(SE) == pytest.approx(sigma)

    def test_bias_correction(self):
        beta = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        sigma = 0.5
        yHat, SE, ConcHat = predict(beta, sigma, 2000.0, 0.0)

        assert float(ConcHat) == pytest.approx(np.exp(1.0 + 0.5**2 / 2))

    def test_array_input(self):
        beta = np.array([1.0, 0.01, -0.5, 0.1, 0.05])
        sigma = 0.3
        dec_year = np.array([2000.0, 2005.0, 2010.0])
        logq = np.array([1.0, 2.0, 3.0])

        yHat, SE, ConcHat = predict(beta, sigma, dec_year, logq)
        assert yHat.shape == (3,)
        assert SE.shape == (3,)
        assert ConcHat.shape == (3,)
        np.testing.assert_allclose(SE, sigma)
        np.testing.assert_allclose(ConcHat, np.exp(yHat + sigma**2 / 2))

    def test_higher_sigma_higher_conchat(self):
        """Larger sigma -> larger bias correction -> higher ConcHat."""
        beta = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        _, _, conc_low_sigma = predict(beta, 0.1, 2000.0, 0.0)
        _, _, conc_high_sigma = predict(beta, 1.0, 2000.0, 0.0)
        assert float(conc_high_sigma) > float(conc_low_sigma)


# ---------------------------------------------------------------------------
# jitter_sample
# ---------------------------------------------------------------------------

class TestJitterSample:
    def test_output_shape(self):
        conc_low = np.array([1.0, np.nan, 3.0])
        conc_high = np.array([1.0, 2.0, 3.0])
        cl_j, ch_j = jitter_sample(conc_low, conc_high, rng=np.random.default_rng(42))
        assert cl_j.shape == conc_low.shape
        assert ch_j.shape == conc_high.shape

    def test_nan_preserved(self):
        conc_low = np.array([1.0, np.nan, 3.0])
        conc_high = np.array([1.0, 2.0, 3.0])
        cl_j, _ = jitter_sample(conc_low, conc_high, rng=np.random.default_rng(42))
        assert np.isnan(cl_j[1])
        assert not np.isnan(cl_j[0])

    def test_values_close(self):
        rng = np.random.default_rng(42)
        conc_low = np.array([5.0, 10.0, 15.0])
        conc_high = np.array([5.0, 10.0, 15.0])
        cl_j, ch_j = jitter_sample(conc_low, conc_high, scale=0.01, rng=rng)

        # With scale=0.01, jittered values should be within a few percent
        np.testing.assert_allclose(cl_j, conc_low, rtol=0.05)
        np.testing.assert_allclose(ch_j, conc_high, rtol=0.05)

    def test_reproducible_with_rng(self):
        conc_low = np.array([1.0, 2.0, 3.0])
        conc_high = np.array([1.0, 2.0, 3.0])
        cl1, ch1 = jitter_sample(conc_low, conc_high, rng=np.random.default_rng(42))
        cl2, ch2 = jitter_sample(conc_low, conc_high, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(cl1, cl2)
        np.testing.assert_array_equal(ch1, ch2)
