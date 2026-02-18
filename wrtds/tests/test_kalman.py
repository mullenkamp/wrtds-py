"""Tests for wrtds.kalman module."""

import numpy as np
import pandas as pd
import pytest

from wrtds.data_prep import populate_daily, populate_sample
from wrtds.cross_val import cross_validate
from wrtds.flow_norm import bin_qs, estimate_daily, flow_normalize
from wrtds.kalman import ar1_conditional_draw, make_augmented_sample, wrtds_kalman
from wrtds.surfaces import compute_surface_index, estimate_surfaces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fitted_data():
    """Build a fully fitted daily+sample+surfaces for kalman tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': q}))

    sample_dates = dates[::15]
    n = len(sample_dates)
    conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
    conc_low = conc.copy()
    conc_high = conc.copy()
    # Make first 3 observations censored
    conc_low[:3] = np.nan
    conc_high[:3] = conc[:3] * 1.5

    sample = populate_sample(
        pd.DataFrame({'Date': sample_dates, 'ConcLow': conc_low, 'ConcHigh': conc_high}),
        daily,
    )

    # Cross-validation
    sample_cv = cross_validate(
        sample,
        window_y=7, window_q=2, window_s=0.5,
        min_num_obs=10, min_num_uncen=5,
    )

    # Surfaces
    si = compute_surface_index(sample_cv)
    surf = estimate_surfaces(
        sample_cv, si,
        window_y=7, window_q=2, window_s=0.5,
        min_num_obs=10, min_num_uncen=5,
    )

    # Daily estimation + flow normalization
    daily_est = estimate_daily(daily, surf, si)
    q_bins = bin_qs(daily_est)
    daily_fn = flow_normalize(daily_est, surf, si, q_bins)

    return daily_fn, sample_cv, surf, si


# ---------------------------------------------------------------------------
# make_augmented_sample
# ---------------------------------------------------------------------------

class TestMakeAugmentedSample:
    def test_uncensored_unchanged(self, fitted_data):
        """Uncensored observations should return ConcAve directly."""
        _, sample, _, _ = fitted_data
        rng = np.random.default_rng(123)
        r_obs = make_augmented_sample(sample, rng=rng)

        uncen = sample['Uncen'].values.astype(bool)
        np.testing.assert_array_equal(r_obs[uncen], sample['ConcAve'].values[uncen])

    def test_censored_below_detection(self, fitted_data):
        """Censored draws should be at most ConcHigh."""
        _, sample, _, _ = fitted_data
        cen = ~sample['Uncen'].values.astype(bool)
        if cen.sum() == 0:
            pytest.skip('No censored observations in fixture')

        rng = np.random.default_rng(456)
        for _ in range(20):
            r_obs = make_augmented_sample(sample, rng=rng)
            assert np.all(r_obs[cen] <= sample['ConcHigh'].values[cen] * (1 + 1e-10))

    def test_censored_positive(self, fitted_data):
        """Censored draws should be positive concentrations."""
        _, sample, _, _ = fitted_data
        rng = np.random.default_rng(789)
        r_obs = make_augmented_sample(sample, rng=rng)
        assert np.all(r_obs > 0)

    def test_length_matches_sample(self, fitted_data):
        _, sample, _, _ = fitted_data
        r_obs = make_augmented_sample(sample)
        assert len(r_obs) == len(sample)

    def test_reproducible_with_seed(self, fitted_data):
        _, sample, _, _ = fitted_data
        r1 = make_augmented_sample(sample, rng=np.random.default_rng(42))
        r2 = make_augmented_sample(sample, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_varies_without_seed(self, fitted_data):
        """Different RNG states should produce different draws for censored obs."""
        _, sample, _, _ = fitted_data
        cen = ~sample['Uncen'].values.astype(bool)
        if cen.sum() == 0:
            pytest.skip('No censored observations')

        r1 = make_augmented_sample(sample, rng=np.random.default_rng(1))
        r2 = make_augmented_sample(sample, rng=np.random.default_rng(2))
        # At least one censored value should differ
        assert not np.allclose(r1[cen], r2[cen])


# ---------------------------------------------------------------------------
# ar1_conditional_draw
# ---------------------------------------------------------------------------

class TestAR1ConditionalDraw:
    def test_empty_gap(self):
        """n_gap=0 returns empty array."""
        rng = np.random.default_rng(42)
        result = ar1_conditional_draw(0.9, 0, 1.0, -1.0, rng)
        assert len(result) == 0

    def test_correct_length(self):
        rng = np.random.default_rng(42)
        for n_gap in [1, 5, 10, 50]:
            result = ar1_conditional_draw(0.9, n_gap, 0.5, -0.5, rng)
            assert len(result) == n_gap

    def test_single_gap_interpolates(self):
        """With rho close to 1 and a single gap, the draw should be near the mean of endpoints."""
        rng = np.random.default_rng(42)
        draws = []
        for _ in range(500):
            d = ar1_conditional_draw(0.99, 1, 1.0, 1.0, rng)
            draws.append(d[0])
        mean_draw = np.mean(draws)
        # With same endpoints and high rho, mean should be close to 1.0
        assert abs(mean_draw - 1.0) < 0.15

    def test_high_rho_smooth(self):
        """With rho near 1, draws should be smooth (close to linear interpolation)."""
        rng = np.random.default_rng(42)
        result = ar1_conditional_draw(0.999, 10, 0.0, 1.0, rng)
        # Should roughly follow a linear trend
        linear = np.linspace(0, 1, 12)[1:-1]
        assert np.max(np.abs(result - linear)) < 1.0

    def test_low_rho_more_variable(self):
        """With low rho, draws should be more variable."""
        rng_low = np.random.default_rng(42)
        rng_high = np.random.default_rng(42)

        vars_low = []
        vars_high = []
        for _ in range(200):
            d_low = ar1_conditional_draw(0.1, 5, 0.0, 0.0, rng_low)
            d_high = ar1_conditional_draw(0.99, 5, 0.0, 0.0, rng_high)
            vars_low.append(np.var(d_low))
            vars_high.append(np.var(d_high))

        # Low rho should produce higher variance on average
        assert np.mean(vars_low) > np.mean(vars_high)

    def test_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        d1 = ar1_conditional_draw(0.9, 5, 1.0, -1.0, rng1)
        d2 = ar1_conditional_draw(0.9, 5, 1.0, -1.0, rng2)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# wrtds_kalman (integration)
# ---------------------------------------------------------------------------

class TestWRTDSKalman:
    def test_adds_genconc_genflux(self, fitted_data):
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        assert 'GenConc' in result.columns
        assert 'GenFlux' in result.columns

    def test_does_not_mutate_input(self, fitted_data):
        daily, sample, surf, si = fitted_data
        daily_copy = daily.copy()
        wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        assert 'GenConc' not in daily.columns

    def test_genconc_positive(self, fitted_data):
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        assert (result['GenConc'] > 0).all()

    def test_genflux_positive(self, fitted_data):
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        assert (result['GenFlux'] > 0).all()

    def test_genflux_formula(self, fitted_data):
        """GenConc * Q * 86.4 should equal GenFlux."""
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        expected = result['GenConc'] * result['Q'] * 86.4
        np.testing.assert_allclose(result['GenFlux'], expected, rtol=1e-12)

    def test_same_length_as_daily(self, fitted_data):
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        assert len(result) == len(daily)

    def test_preserves_existing_columns(self, fitted_data):
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        for col in daily.columns:
            assert col in result.columns

    def test_reproducible_with_seed(self, fitted_data):
        daily, sample, surf, si = fitted_data
        r1 = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        r2 = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        np.testing.assert_array_equal(r1['GenConc'].values, r2['GenConc'].values)
        np.testing.assert_array_equal(r1['GenFlux'].values, r2['GenFlux'].values)

    def test_more_iterations_converges(self, fitted_data):
        """More iterations should reduce Monte Carlo noise (std of repeated runs)."""
        daily, sample, surf, si = fitted_data

        # Run with few iterations, different seeds
        results_10 = []
        results_50 = []
        for seed in range(5):
            r10 = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=seed)
            r50 = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=50, seed=seed)
            results_10.append(r10['GenFlux'].mean())
            results_50.append(r50['GenFlux'].mean())

        std_10 = np.std(results_10)
        std_50 = np.std(results_50)
        # 50 iterations should have lower cross-seed variance than 10
        assert std_50 < std_10

    def test_all_values_finite(self, fitted_data):
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=10, seed=42)
        assert np.all(np.isfinite(result['GenConc'].values))
        assert np.all(np.isfinite(result['GenFlux'].values))

    def test_genconc_reasonable_range(self, fitted_data):
        """GenConc should be in a reasonable range relative to ConcDay."""
        daily, sample, surf, si = fitted_data
        result = wrtds_kalman(daily, sample, surf, si, rho=0.9, n_iter=50, seed=42)

        conc_day_median = result['ConcDay'].median()
        gen_conc_median = result['GenConc'].median()

        # GenConc median should be within an order of magnitude of ConcDay
        ratio = gen_conc_median / conc_day_median
        assert 0.1 < ratio < 10.0


# ---------------------------------------------------------------------------
# WRTDS class integration
# ---------------------------------------------------------------------------

class TestWRTDSKalmanIntegration:
    def test_kalman_requires_fit(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = pd.DataFrame({'Date': dates, 'Q': q})

        sample_dates = dates[::15]
        n = len(sample_dates)
        conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
        sample = pd.DataFrame({'Date': sample_dates, 'ConcLow': conc, 'ConcHigh': conc})

        from wrtds import WRTDS
        m = WRTDS(daily, sample)
        with pytest.raises(RuntimeError, match='fit'):
            m.kalman()

    def test_kalman_after_fit(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = pd.DataFrame({'Date': dates, 'Q': q})

        sample_dates = dates[::15]
        n = len(sample_dates)
        conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
        sample = pd.DataFrame({'Date': sample_dates, 'ConcLow': conc, 'ConcHigh': conc})

        from wrtds import WRTDS
        m = WRTDS(daily, sample).fit(
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
        )
        result = m.kalman(rho=0.9, n_iter=10, seed=42)
        assert result is m
        assert 'GenConc' in m.daily.columns
        assert 'GenFlux' in m.daily.columns

    def test_chaining(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = pd.DataFrame({'Date': dates, 'Q': q})

        sample_dates = dates[::15]
        n = len(sample_dates)
        conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
        sample = pd.DataFrame({'Date': sample_dates, 'ConcLow': conc, 'ConcHigh': conc})

        from wrtds import WRTDS
        m = WRTDS(daily, sample).fit(
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
        ).kalman(rho=0.9, n_iter=10, seed=42)

        assert isinstance(m, WRTDS)
        assert 'GenConc' in m.daily.columns
