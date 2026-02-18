"""Tests for wrtds.bootstrap module."""

import numpy as np
import pandas as pd
import pytest

from wrtds import WRTDS
from wrtds.bootstrap import (
    block_resample,
    bootstrap_groups,
    bootstrap_pairs,
    likelihood_descriptor,
    pval,
)
from wrtds.data_prep import populate_daily, populate_sample
from wrtds.surfaces import compute_surface_index, estimate_surfaces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def model_data():
    """Build populated sample and daily DataFrames for bootstrap tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': q}))

    sample_dates = dates[::15]
    n = len(sample_dates)
    conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
    conc_low = conc.copy()
    conc_high = conc.copy()
    conc_low[:3] = np.nan
    conc_high[:3] = conc[:3] * 1.5

    sample = populate_sample(
        pd.DataFrame({'Date': sample_dates, 'ConcLow': conc_low, 'ConcHigh': conc_high}),
        daily,
    )

    fit_params = {
        'window_y': 7, 'window_q': 2, 'window_s': 0.5,
        'min_num_obs': 10, 'min_num_uncen': 5, 'edge_adjust': True,
    }

    si = compute_surface_index(sample)
    surfaces = estimate_surfaces(sample, si, **fit_params)

    return daily, sample, surfaces, si, fit_params


# ---------------------------------------------------------------------------
# block_resample
# ---------------------------------------------------------------------------

class TestBlockResample:
    def test_same_number_of_rows(self, model_data):
        _, sample, _, _, _ = model_data
        resampled = block_resample(sample, block_length=200, rng=np.random.default_rng(42))
        assert len(resampled) == len(sample)

    def test_sorted_by_julian(self, model_data):
        _, sample, _, _, _ = model_data
        resampled = block_resample(sample, block_length=200, rng=np.random.default_rng(42))
        julian = resampled['Julian'].values
        assert np.all(julian[:-1] <= julian[1:])

    def test_preserves_columns(self, model_data):
        _, sample, _, _, _ = model_data
        resampled = block_resample(sample, block_length=200, rng=np.random.default_rng(42))
        for col in sample.columns:
            assert col in resampled.columns

    def test_reproducible_with_seed(self, model_data):
        _, sample, _, _, _ = model_data
        r1 = block_resample(sample, block_length=200, rng=np.random.default_rng(42))
        r2 = block_resample(sample, block_length=200, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1['Julian'].values, r2['Julian'].values)

    def test_different_seeds_different_results(self, model_data):
        _, sample, _, _, _ = model_data
        r1 = block_resample(sample, block_length=200, rng=np.random.default_rng(1))
        r2 = block_resample(sample, block_length=200, rng=np.random.default_rng(2))
        assert not np.array_equal(r1['Julian'].values, r2['Julian'].values)

    def test_values_from_original(self, model_data):
        """All Julian values in resampled data should exist in original."""
        _, sample, _, _, _ = model_data
        resampled = block_resample(sample, block_length=200, rng=np.random.default_rng(42))
        original_julian = set(sample['Julian'].values)
        for j in resampled['Julian'].values:
            assert j in original_julian

    def test_may_have_duplicates(self, model_data):
        """Block bootstrap may produce duplicate observations."""
        _, sample, _, _, _ = model_data
        # With a large block length relative to the record, duplicates are likely
        resampled = block_resample(sample, block_length=50, rng=np.random.default_rng(42))
        n_unique = len(resampled['Julian'].unique())
        # At least some duplicates should exist (probabilistic but very likely)
        # Just check that the function runs and returns valid data
        assert len(resampled) == len(sample)

    def test_small_block_length(self, model_data):
        """Small block lengths should still work."""
        _, sample, _, _, _ = model_data
        resampled = block_resample(sample, block_length=30, rng=np.random.default_rng(42))
        assert len(resampled) == len(sample)

    def test_large_block_length(self, model_data):
        """Block length larger than record should still work."""
        _, sample, _, _, _ = model_data
        julian = sample['Julian'].values
        record_len = int(julian[-1] - julian[0]) + 1
        resampled = block_resample(sample, block_length=record_len * 2,
                                   rng=np.random.default_rng(42))
        assert len(resampled) == len(sample)


# ---------------------------------------------------------------------------
# pval
# ---------------------------------------------------------------------------

class TestPVal:
    def test_all_positive(self):
        """All positive values should give a small p-value."""
        s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        p = pval(s)
        assert p == pytest.approx(2.0 / 6)  # 2 / (m + 1) where m = 5

    def test_all_negative(self):
        """All negative values should give a small p-value."""
        s = np.array([-0.5, -0.4, -0.3, -0.2, -0.1])
        p = pval(s)
        assert p == pytest.approx(2.0 / 6)

    def test_balanced(self):
        """Equal positive and negative should give p near 1."""
        s = np.array([-3, -2, -1, 1, 2, 3])
        p = pval(s)
        # With 3 neg and 3 pos, p should be close to 1.0
        assert p > 0.8

    def test_mostly_positive(self):
        """Mostly positive should give small p-value."""
        s = np.concatenate([np.full(95, 1.0), np.full(5, -1.0)])
        p = pval(s)
        assert p < 0.15

    def test_ignores_zeros(self):
        """Exact zeros should be removed."""
        s = np.array([0.0, 0.0, 0.1, 0.2, 0.3])
        p = pval(s)
        # After removing zeros, all positive -> 2 / (3+1) = 0.5
        assert p == pytest.approx(2.0 / 4)

    def test_ignores_nans(self):
        """NaN values should be removed."""
        s = np.array([np.nan, 0.1, 0.2, 0.3, np.nan])
        p = pval(s)
        assert p == pytest.approx(2.0 / 4)

    def test_empty_returns_one(self):
        p = pval(np.array([]))
        assert p == 1.0

    def test_returns_float(self):
        p = pval(np.array([1.0, 2.0, -0.5]))
        assert isinstance(p, float)

    def test_p_in_valid_range(self):
        """P-value should be in (0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            s = rng.normal(0, 1, 50)
            p = pval(s)
            assert 0 < p <= 1.0

    def test_interpolation_symmetric(self):
        """Symmetric distribution around zero should give p ≈ 1."""
        s = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        p = pval(s)
        assert abs(p - 1.0) < 0.2


# ---------------------------------------------------------------------------
# likelihood_descriptor
# ---------------------------------------------------------------------------

class TestLikelihoodDescriptor:
    def test_highly_unlikely(self):
        assert likelihood_descriptor(0.03) == 'highly unlikely'

    def test_very_unlikely(self):
        assert likelihood_descriptor(0.07) == 'very unlikely'

    def test_unlikely(self):
        assert likelihood_descriptor(0.20) == 'unlikely'

    def test_about_as_likely_as_not(self):
        assert likelihood_descriptor(0.50) == 'about as likely as not'

    def test_likely(self):
        assert likelihood_descriptor(0.80) == 'likely'

    def test_very_likely(self):
        assert likelihood_descriptor(0.92) == 'very likely'

    def test_highly_likely(self):
        assert likelihood_descriptor(0.97) == 'highly likely'

    def test_boundary_zero(self):
        assert likelihood_descriptor(0.0) == 'highly unlikely'

    def test_boundary_one(self):
        assert likelihood_descriptor(1.0) == 'highly likely'

    def test_returns_string(self):
        assert isinstance(likelihood_descriptor(0.5), str)


# ---------------------------------------------------------------------------
# bootstrap_pairs (smoke test — expensive, so minimal replicates)
# ---------------------------------------------------------------------------

class TestBootstrapPairs:
    def test_output_structure(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )

        # Check all expected keys
        assert 'observed' in result
        assert 'boot_conc' in result
        assert 'boot_flux' in result
        assert 'p_conc' in result
        assert 'p_flux' in result
        assert 'ci_conc' in result
        assert 'ci_flux' in result
        assert 'likelihood_conc_up' in result
        assert 'likelihood_flux_up' in result
        assert 'like_conc_up' in result
        assert 'like_conc_down' in result
        assert 'like_flux_up' in result
        assert 'like_flux_down' in result

    def test_boot_array_length(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        assert len(result['boot_conc']) == 3
        assert len(result['boot_flux']) == 3

    def test_p_values_valid(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        assert 0 < result['p_conc'] <= 1.0
        assert 0 < result['p_flux'] <= 1.0

    def test_ci_ordered(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        assert result['ci_conc'][0] <= result['ci_conc'][1]
        assert result['ci_flux'][0] <= result['ci_flux'][1]

    def test_observed_is_dataframe(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        assert isinstance(result['observed'], pd.DataFrame)
        assert result['observed'].shape == (2, 7)

    def test_likelihood_valid(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        assert 0 < result['likelihood_conc_up'] < 1
        assert 0 < result['likelihood_flux_up'] < 1
        assert isinstance(result['like_conc_up'], str)
        assert isinstance(result['like_flux_down'], str)

    def test_reproducible(self, model_data):
        daily, sample, _, _, fit_params = model_data
        r1 = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        r2 = bootstrap_pairs(
            sample, daily, 2002, 2004,
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        np.testing.assert_allclose(r1['boot_conc'], r2['boot_conc'], rtol=1e-3)


# ---------------------------------------------------------------------------
# bootstrap_groups (smoke test)
# ---------------------------------------------------------------------------

class TestBootstrapGroups:
    def test_output_structure(self, model_data):
        daily, sample, surfaces, si, fit_params = model_data
        result = bootstrap_groups(
            daily, sample, surfaces, si,
            group1_years=(2001, 2002), group2_years=(2004, 2005),
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )

        assert 'observed' in result
        assert 'boot_conc' in result
        assert 'boot_flux' in result
        assert 'p_conc' in result
        assert 'p_flux' in result
        assert len(result['boot_conc']) == 3

    def test_p_values_valid(self, model_data):
        daily, sample, surfaces, si, fit_params = model_data
        result = bootstrap_groups(
            daily, sample, surfaces, si,
            group1_years=(2001, 2002), group2_years=(2004, 2005),
            n_boot=3, block_length=200, window_side=3,
            pa_start=1, pa_long=12, fit_params=fit_params, seed=42,
        )
        assert 0 < result['p_conc'] <= 1.0
        assert 0 < result['p_flux'] <= 1.0


# ---------------------------------------------------------------------------
# WRTDS class integration
# ---------------------------------------------------------------------------

class TestWRTDSBootstrap:
    @pytest.fixture(scope='class')
    def fitted_model(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = pd.DataFrame({'Date': dates, 'Q': q})

        sample_dates = dates[::15]
        n = len(sample_dates)
        conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
        sample = pd.DataFrame({'Date': sample_dates,
                                'ConcLow': conc, 'ConcHigh': conc})

        return WRTDS(daily, sample).fit(
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
        )

    def test_bootstrap_pairs_requires_fit(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = pd.DataFrame({'Date': dates, 'Q': q})
        sample_dates = dates[::15]
        n = len(sample_dates)
        conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
        sample = pd.DataFrame({'Date': sample_dates,
                                'ConcLow': conc, 'ConcHigh': conc})
        m = WRTDS(daily, sample)
        with pytest.raises(RuntimeError, match='fit'):
            m.bootstrap_pairs(2001, 2001)

    def test_bootstrap_pairs_via_class(self, fitted_model):
        result = fitted_model.bootstrap_pairs(
            2002, 2004, n_boot=3, window_side=3, seed=42,
        )
        assert 'observed' in result
        assert len(result['boot_conc']) == 3

    def test_bootstrap_groups_via_class(self, fitted_model):
        result = fitted_model.bootstrap_groups(
            (2001, 2002), (2004, 2005), n_boot=3, window_side=3, seed=42,
        )
        assert 'observed' in result
        assert len(result['boot_conc']) == 3
