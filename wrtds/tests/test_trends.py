"""Tests for wrtds.trends module."""

import numpy as np
import pandas as pd
import pytest

from wrtds import WRTDS
from wrtds.data_prep import populate_daily, populate_sample
from wrtds.flow_norm import bin_qs
from wrtds.surfaces import compute_surface_index, estimate_surfaces
from wrtds.trends import (
    _annual_fn_mean,
    _build_result_df,
    _compute_surface_index_narrow,
    _filter_daily_to_year,
    _water_year_dates,
    bin_qs_windowed,
    run_groups,
    run_pairs,
    run_series,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def model_data():
    """Build a fitted WRTDS model for trend tests.

    Uses a 6-year daily record (2000-2005) so there are enough years
    for meaningful group comparisons and series.
    """
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

    # Fit with relaxed thresholds for the small dataset
    fit_params = {
        'window_y': 7, 'window_q': 2, 'window_s': 0.5,
        'min_num_obs': 10, 'min_num_uncen': 5, 'edge_adjust': True,
    }

    si = compute_surface_index(sample)
    surfaces = estimate_surfaces(sample, si, **fit_params)

    return daily, sample, surfaces, si, fit_params


# ---------------------------------------------------------------------------
# _water_year_dates
# ---------------------------------------------------------------------------

class TestWaterYearDates:
    def test_water_year_default(self):
        """pa_start=10: year 2001 → Oct 2000 to Sep 2001."""
        start, end = _water_year_dates(2001, pa_start=10, pa_long=12)
        assert start == pd.Timestamp('2000-10-01')
        assert end == pd.Timestamp('2001-09-30')

    def test_calendar_year(self):
        """pa_start=1: year 2001 → Jan 2001 to Dec 2001."""
        start, end = _water_year_dates(2001, pa_start=1, pa_long=12)
        assert start == pd.Timestamp('2001-01-01')
        assert end == pd.Timestamp('2001-12-31')

    def test_partial_year(self):
        """pa_start=4, pa_long=6: year 2001 → Apr 2001 to Sep 2001."""
        start, end = _water_year_dates(2001, pa_start=4, pa_long=6)
        assert start == pd.Timestamp('2001-04-01')
        assert end == pd.Timestamp('2001-09-30')

    def test_wrapping_partial(self):
        """pa_start=11, pa_long=4: year 2001 → Nov 2000 to Feb 2001."""
        start, end = _water_year_dates(2001, pa_start=11, pa_long=4)
        assert start == pd.Timestamp('2000-11-01')
        assert end == pd.Timestamp('2001-02-28')


# ---------------------------------------------------------------------------
# _filter_daily_to_year
# ---------------------------------------------------------------------------

class TestFilterDailyToYear:
    def test_correct_date_range(self, model_data):
        daily = model_data[0]
        filtered = _filter_daily_to_year(daily, 2002, pa_start=1, pa_long=12)
        assert filtered['Date'].min() >= pd.Timestamp('2002-01-01')
        assert filtered['Date'].max() <= pd.Timestamp('2002-12-31')

    def test_water_year_range(self, model_data):
        daily = model_data[0]
        filtered = _filter_daily_to_year(daily, 2002, pa_start=10, pa_long=12)
        assert filtered['Date'].min() >= pd.Timestamp('2001-10-01')
        assert filtered['Date'].max() <= pd.Timestamp('2002-09-30')

    def test_approximately_365_days(self, model_data):
        daily = model_data[0]
        filtered = _filter_daily_to_year(daily, 2003, pa_start=1, pa_long=12)
        assert 365 <= len(filtered) <= 366


# ---------------------------------------------------------------------------
# _compute_surface_index_narrow
# ---------------------------------------------------------------------------

class TestComputeSurfaceIndexNarrow:
    def test_year_range(self, model_data):
        _, sample, _, _, _ = model_data
        si = _compute_surface_index_narrow(sample, 2001.0, 2002.0)
        assert si['bottom_year'] == 2001.0
        assert si['top_year'] == 2002.0

    def test_logq_range_matches_full(self, model_data):
        _, sample, _, si_full, _ = model_data
        si_narrow = _compute_surface_index_narrow(sample, 2001.0, 2002.0)
        assert si_narrow['bottom_logq'] == si_full['bottom_logq']
        assert si_narrow['top_logq'] == si_full['top_logq']
        assert si_narrow['n_logq'] == 14

    def test_fewer_year_grid_points(self, model_data):
        _, sample, _, si_full, _ = model_data
        si_narrow = _compute_surface_index_narrow(sample, 2001.0, 2002.0)
        assert si_narrow['n_year'] < si_full['n_year']


# ---------------------------------------------------------------------------
# bin_qs_windowed
# ---------------------------------------------------------------------------

class TestBinQsWindowed:
    def test_fewer_obs_than_full(self, model_data):
        daily = model_data[0]
        full_bins = bin_qs(daily)
        windowed_bins = bin_qs_windowed(daily, 2003, window_side=1)

        total_full = sum(len(v) for v in full_bins.values())
        total_windowed = sum(len(v) for v in windowed_bins.values())
        assert total_windowed < total_full

    def test_window_side_zero_is_single_year(self, model_data):
        daily = model_data[0]
        bins = bin_qs_windowed(daily, 2003, window_side=0.5)
        total = sum(len(v) for v in bins.values())
        # Should have roughly 1 year of data (~365 obs)
        assert 300 < total < 500

    def test_large_window_equals_full(self, model_data):
        daily = model_data[0]
        full_bins = bin_qs(daily)
        # Window larger than the record should include everything
        windowed_bins = bin_qs_windowed(daily, 2003, window_side=100)

        total_full = sum(len(v) for v in full_bins.values())
        total_windowed = sum(len(v) for v in windowed_bins.values())
        assert total_windowed == total_full

    def test_keys_range(self, model_data):
        daily = model_data[0]
        bins = bin_qs_windowed(daily, 2003, window_side=2)
        for key in bins:
            assert 1 <= key <= 365


# ---------------------------------------------------------------------------
# _annual_fn_mean
# ---------------------------------------------------------------------------

class TestAnnualFnMean:
    def test_returns_positive_values(self, model_data):
        daily, _, surfaces, si, _ = model_data
        q_bins = bin_qs(daily)
        daily_year = _filter_daily_to_year(daily, 2003, pa_start=1, pa_long=12)
        c, f = _annual_fn_mean(daily_year, surfaces, si, q_bins)
        assert c > 0
        assert f > 0

    def test_returns_finite(self, model_data):
        daily, _, surfaces, si, _ = model_data
        q_bins = bin_qs(daily)
        daily_year = _filter_daily_to_year(daily, 2003, pa_start=1, pa_long=12)
        c, f = _annual_fn_mean(daily_year, surfaces, si, q_bins)
        assert np.isfinite(c)
        assert np.isfinite(f)


# ---------------------------------------------------------------------------
# _build_result_df
# ---------------------------------------------------------------------------

class TestBuildResultDf:
    def test_shape_and_index(self):
        df = _build_result_df(1.0, 100.0, 1.1, 110.0, 0.9, 90.0, 0.95, 95.0)
        assert list(df.index) == ['Conc', 'Flux']
        assert list(df.columns) == ['TotalChange', 'CQTC', 'QTC', 'x10', 'x11', 'x20', 'x22']

    def test_decomposition_identity(self):
        """TotalChange = CQTC + QTC for both Conc and Flux."""
        df = _build_result_df(1.0, 100.0, 1.1, 110.0, 0.9, 90.0, 0.95, 95.0)
        for row in ['Conc', 'Flux']:
            np.testing.assert_allclose(
                df.loc[row, 'TotalChange'],
                df.loc[row, 'CQTC'] + df.loc[row, 'QTC'],
                rtol=1e-12,
            )

    def test_total_change_formula(self):
        """TotalChange = x22 - x11."""
        df = _build_result_df(1.0, 100.0, 1.1, 110.0, 0.9, 90.0, 0.95, 95.0)
        np.testing.assert_allclose(
            df.loc['Conc', 'TotalChange'],
            df.loc['Conc', 'x22'] - df.loc['Conc', 'x11'],
            rtol=1e-12,
        )

    def test_cqtc_formula(self):
        """CQTC = x20 - x10."""
        df = _build_result_df(1.0, 100.0, 1.1, 110.0, 0.9, 90.0, 0.95, 95.0)
        np.testing.assert_allclose(
            df.loc['Conc', 'CQTC'],
            df.loc['Conc', 'x20'] - df.loc['Conc', 'x10'],
            rtol=1e-12,
        )

    def test_flux_conversion(self):
        """Flux values should be converted to 10^6 kg/year."""
        df = _build_result_df(1.0, 1e6 / 365.25, 1.0, 1e6 / 365.25,
                              1.0, 1e6 / 365.25, 1.0, 1e6 / 365.25)
        # f10 = 1e6/365.25 kg/day → 1.0 × 10^6 kg/year
        np.testing.assert_allclose(df.loc['Flux', 'x10'], 1.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# run_pairs
# ---------------------------------------------------------------------------

class TestRunPairs:
    def test_output_shape(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = run_pairs(sample, daily, 2002, 2004,
                           window_side=3, pa_start=1, pa_long=12,
                           fit_params=fit_params)
        assert result.shape == (2, 7)
        assert list(result.index) == ['Conc', 'Flux']

    def test_decomposition_identity(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = run_pairs(sample, daily, 2002, 2004,
                           window_side=3, pa_start=1, pa_long=12,
                           fit_params=fit_params)
        for row in ['Conc', 'Flux']:
            np.testing.assert_allclose(
                result.loc[row, 'TotalChange'],
                result.loc[row, 'CQTC'] + result.loc[row, 'QTC'],
                rtol=1e-10,
            )

    def test_values_finite(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = run_pairs(sample, daily, 2002, 2004,
                           window_side=3, pa_start=1, pa_long=12,
                           fit_params=fit_params)
        assert np.all(np.isfinite(result.values))

    def test_x_values_positive(self, model_data):
        daily, sample, _, _, fit_params = model_data
        result = run_pairs(sample, daily, 2002, 2004,
                           window_side=3, pa_start=1, pa_long=12,
                           fit_params=fit_params)
        for col in ['x10', 'x11', 'x20', 'x22']:
            assert (result[col] > 0).all(), f'{col} should be positive'

    def test_window_side_zero(self, model_data):
        """With window_side=0, x11==x10 and x22==x20 (stationary FN for both)."""
        daily, sample, _, _, fit_params = model_data
        result = run_pairs(sample, daily, 2002, 2004,
                           window_side=0, pa_start=1, pa_long=12,
                           fit_params=fit_params)
        np.testing.assert_allclose(
            result.loc['Conc', 'x11'], result.loc['Conc', 'x10'], rtol=1e-10)
        np.testing.assert_allclose(
            result.loc['Conc', 'x22'], result.loc['Conc', 'x20'], rtol=1e-10)
        # QTC should be zero when using stationary FN
        np.testing.assert_allclose(
            result.loc['Conc', 'QTC'], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# run_groups
# ---------------------------------------------------------------------------

class TestRunGroups:
    def test_output_shape(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_groups(daily, surfaces, si,
                            group1_years=(2001, 2002),
                            group2_years=(2004, 2005),
                            window_side=3, pa_start=1, pa_long=12)
        assert result.shape == (2, 7)
        assert list(result.index) == ['Conc', 'Flux']

    def test_decomposition_identity(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_groups(daily, surfaces, si,
                            group1_years=(2001, 2002),
                            group2_years=(2004, 2005),
                            window_side=3, pa_start=1, pa_long=12)
        for row in ['Conc', 'Flux']:
            np.testing.assert_allclose(
                result.loc[row, 'TotalChange'],
                result.loc[row, 'CQTC'] + result.loc[row, 'QTC'],
                rtol=1e-10,
            )

    def test_values_finite(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_groups(daily, surfaces, si,
                            group1_years=(2001, 2002),
                            group2_years=(2004, 2005),
                            window_side=3, pa_start=1, pa_long=12)
        assert np.all(np.isfinite(result.values))

    def test_x_values_positive(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_groups(daily, surfaces, si,
                            group1_years=(2001, 2002),
                            group2_years=(2004, 2005),
                            window_side=3, pa_start=1, pa_long=12)
        for col in ['x10', 'x11', 'x20', 'x22']:
            assert (result[col] > 0).all(), f'{col} should be positive'

    def test_single_year_groups(self, model_data):
        """Single-year groups should still work."""
        daily, _, surfaces, si, _ = model_data
        result = run_groups(daily, surfaces, si,
                            group1_years=(2002, 2002),
                            group2_years=(2004, 2004),
                            window_side=3, pa_start=1, pa_long=12)
        assert np.all(np.isfinite(result.values))

    def test_window_side_zero(self, model_data):
        """With window_side=0, QTC should be zero."""
        daily, _, surfaces, si, _ = model_data
        result = run_groups(daily, surfaces, si,
                            group1_years=(2001, 2002),
                            group2_years=(2004, 2005),
                            window_side=0, pa_start=1, pa_long=12)
        np.testing.assert_allclose(
            result.loc['Conc', 'QTC'], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# run_series
# ---------------------------------------------------------------------------

class TestRunSeries:
    def test_adds_fn_columns(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_series(daily, surfaces, si,
                            window_side=3, pa_start=1, pa_long=12)
        assert 'FNConc' in result.columns
        assert 'FNFlux' in result.columns

    def test_does_not_mutate_input(self, model_data):
        daily, _, surfaces, si, _ = model_data
        daily_orig = daily.copy()
        run_series(daily, surfaces, si, window_side=3, pa_start=1, pa_long=12)
        # Original should not be modified
        if 'FNConc' in daily_orig.columns:
            np.testing.assert_array_equal(
                daily_orig['FNConc'].values, daily['FNConc'].values)

    def test_fnconc_positive_where_valid(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_series(daily, surfaces, si,
                            window_side=3, pa_start=1, pa_long=12)
        valid = result['FNConc'].notna()
        assert (result.loc[valid, 'FNConc'] > 0).all()

    def test_fnflux_positive_where_valid(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_series(daily, surfaces, si,
                            window_side=3, pa_start=1, pa_long=12)
        valid = result['FNFlux'].notna()
        assert (result.loc[valid, 'FNFlux'] > 0).all()

    def test_same_length_as_input(self, model_data):
        daily, _, surfaces, si, _ = model_data
        result = run_series(daily, surfaces, si,
                            window_side=3, pa_start=1, pa_long=12)
        assert len(result) == len(daily)

    def test_most_days_covered(self, model_data):
        """Most days in complete years should have FNConc values."""
        daily, _, surfaces, si, _ = model_data
        result = run_series(daily, surfaces, si,
                            window_side=3, pa_start=1, pa_long=12)
        valid_frac = result['FNConc'].notna().mean()
        # At least 80% of days should be covered
        assert valid_frac > 0.8

    def test_window_side_zero_stationary(self, model_data):
        """window_side=0 should produce standard (stationary) flow normalization."""
        daily, _, surfaces, si, _ = model_data
        result = run_series(daily, surfaces, si,
                            window_side=0, pa_start=1, pa_long=12)
        valid = result['FNConc'].notna()
        assert valid.sum() > 0
        assert (result.loc[valid, 'FNConc'] > 0).all()


# ---------------------------------------------------------------------------
# WRTDS class integration
# ---------------------------------------------------------------------------

class TestWRTDSTrends:
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

    def test_run_pairs_requires_fit(self):
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
            m.run_pairs(2001, 2001)

    def test_run_pairs_via_class(self, fitted_model):
        result = fitted_model.run_pairs(2002, 2004, window_side=3)
        assert result.shape == (2, 7)
        assert np.all(np.isfinite(result.values))

    def test_run_groups_via_class(self, fitted_model):
        result = fitted_model.run_groups((2001, 2002), (2004, 2005), window_side=3)
        assert result.shape == (2, 7)
        assert np.all(np.isfinite(result.values))

    def test_run_series_via_class(self, fitted_model):
        result = fitted_model.run_series(window_side=3)
        assert isinstance(result, WRTDS)
        assert result is fitted_model
        assert fitted_model.daily['FNConc'].notna().sum() > 0
