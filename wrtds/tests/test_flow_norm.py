"""Tests for wrtds.flow_norm module."""

import numpy as np
import pandas as pd
import pytest

from wrtds.data_prep import populate_daily, populate_sample
from wrtds.flow_norm import bin_qs, estimate_daily, flow_normalize
from wrtds.surfaces import compute_surface_index, estimate_surfaces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fitted_model():
    """Build a small but complete model: daily + sample + surfaces.

    Uses a 2-year record so estimate_surfaces finishes quickly.
    scope='module' so the expensive surface estimation runs once.
    """
    rng = np.random.default_rng(42)

    dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
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

    surface_index = compute_surface_index(sample)
    surfaces = estimate_surfaces(
        sample, surface_index,
        window_y=7, window_q=2, window_s=0.5,
        min_num_obs=10, min_num_uncen=5,
    )
    return daily, sample, surfaces, surface_index


# ---------------------------------------------------------------------------
# estimate_daily
# ---------------------------------------------------------------------------

class TestEstimateDaily:
    def test_adds_columns(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        result = estimate_daily(daily, surfaces, surface_index)
        for col in ('yHat', 'SE', 'ConcDay', 'FluxDay'):
            assert col in result.columns

    def test_does_not_mutate_input(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        original_cols = set(daily.columns)
        estimate_daily(daily, surfaces, surface_index)
        assert set(daily.columns) == original_cols

    def test_concday_positive(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        result = estimate_daily(daily, surfaces, surface_index)
        assert (result['ConcDay'] > 0).all()

    def test_fluxday_formula(self, fitted_model):
        """FluxDay = ConcDay * Q * 86.4"""
        daily, _, surfaces, surface_index = fitted_model
        result = estimate_daily(daily, surfaces, surface_index)
        expected = result['ConcDay'] * result['Q'] * 86.4
        np.testing.assert_allclose(result['FluxDay'], expected, rtol=1e-12)

    def test_se_positive(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        result = estimate_daily(daily, surfaces, surface_index)
        assert (result['SE'] > 0).all()

    def test_values_finite(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        result = estimate_daily(daily, surfaces, surface_index)
        for col in ('yHat', 'SE', 'ConcDay', 'FluxDay'):
            assert result[col].notna().all()
            assert np.all(np.isfinite(result[col].values))


# ---------------------------------------------------------------------------
# bin_qs
# ---------------------------------------------------------------------------

class TestBinQs:
    def test_keys_range(self, fitted_model):
        daily = fitted_model[0]
        bins = bin_qs(daily)
        assert all(1 <= k <= 365 for k in bins)

    def test_no_day_366(self, fitted_model):
        """Leap-adjusted day numbering should never exceed 365."""
        daily = fitted_model[0]
        bins = bin_qs(daily)
        assert 366 not in bins

    def test_total_observations(self, fitted_model):
        """Total binned observations should equal number of daily rows."""
        daily = fitted_model[0]
        bins = bin_qs(daily)
        total = sum(len(v) for v in bins.values())
        assert total == len(daily)

    def test_leap_day_merged(self):
        """Feb 29 observations should be merged into the Feb 28 bin."""
        dates = pd.to_datetime(['2000-02-28', '2000-02-29', '2000-03-01'])
        daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': [1.0, 2.0, 3.0]}))
        bins = bin_qs(daily)

        # Day 59 = Feb 28 in non-leap OR Feb 28+29 in leap
        assert 59 in bins
        assert len(bins[59]) == 2  # both Feb 28 and Feb 29

        # Day 60 should be March 1 (not Feb 29)
        assert 60 in bins
        assert len(bins[60]) == 1

    def test_values_are_logq(self, fitted_model):
        daily = fitted_model[0]
        bins = bin_qs(daily)
        all_binned = np.concatenate(list(bins.values()))
        # All binned values should be present in daily['LogQ']
        np.testing.assert_allclose(
            np.sort(all_binned),
            np.sort(daily['LogQ'].values),
        )

    def test_nonleap_year_coverage(self):
        """A non-leap year should cover days 1-365."""
        dates = pd.date_range('2001-01-01', '2001-12-31', freq='D')
        daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': np.ones(365)}))
        bins = bin_qs(daily)
        assert set(bins.keys()) == set(range(1, 366))


# ---------------------------------------------------------------------------
# flow_normalize
# ---------------------------------------------------------------------------

class TestFlowNormalize:
    def test_adds_columns(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        daily_est = estimate_daily(daily, surfaces, surface_index)
        q_bins = bin_qs(daily)
        result = flow_normalize(daily_est, surfaces, surface_index, q_bins)
        assert 'FNConc' in result.columns
        assert 'FNFlux' in result.columns

    def test_does_not_mutate_input(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        daily_est = estimate_daily(daily, surfaces, surface_index)
        q_bins = bin_qs(daily)
        original_cols = set(daily_est.columns)
        flow_normalize(daily_est, surfaces, surface_index, q_bins)
        assert set(daily_est.columns) == original_cols

    def test_fnconc_positive(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        daily_est = estimate_daily(daily, surfaces, surface_index)
        q_bins = bin_qs(daily)
        result = flow_normalize(daily_est, surfaces, surface_index, q_bins)
        valid = result['FNConc'].notna()
        assert (result.loc[valid, 'FNConc'] > 0).all()

    def test_fnflux_positive(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        daily_est = estimate_daily(daily, surfaces, surface_index)
        q_bins = bin_qs(daily)
        result = flow_normalize(daily_est, surfaces, surface_index, q_bins)
        valid = result['FNFlux'].notna()
        assert (result.loc[valid, 'FNFlux'] > 0).all()

    def test_fnconc_smoother_than_concday(self, fitted_model):
        """Flow-normalised concentration should vary less than raw ConcDay
        because the discharge variability is averaged out."""
        daily, _, surfaces, surface_index = fitted_model
        daily_est = estimate_daily(daily, surfaces, surface_index)
        q_bins = bin_qs(daily)
        result = flow_normalize(daily_est, surfaces, surface_index, q_bins)
        valid = result['FNConc'].notna()
        assert result.loc[valid, 'FNConc'].std() < result.loc[valid, 'ConcDay'].std()

    def test_fnconc_manual_spot_check(self, fitted_model):
        """Verify one day's FNConc equals the mean of ConcHat across that
        day's historical Q distribution."""
        daily, _, surfaces, surface_index = fitted_model
        daily_est = estimate_daily(daily, surfaces, surface_index)
        q_bins = bin_qs(daily)
        result = flow_normalize(daily_est, surfaces, surface_index, q_bins)

        # Pick an interior day
        from wrtds.surfaces import interpolate_surface

        idx = 100
        row = result.iloc[idx]

        # Leap-adjust the day
        d = int(row['Day'])
        is_lp = pd.Timestamp(row['Date']).is_leap_year
        if d == 60 and is_lp:
            d = 59
        elif d > 60 and is_lp:
            d -= 1

        hist_logq = q_bins[d]
        dec_yr = row['DecYear']
        conc_vals = interpolate_surface(
            surfaces, surface_index,
            hist_logq, np.full(len(hist_logq), dec_yr), layer=2,
        )
        expected_fnconc = conc_vals.mean()
        assert row['FNConc'] == pytest.approx(expected_fnconc, rel=1e-10)

    def test_values_finite(self, fitted_model):
        daily, _, surfaces, surface_index = fitted_model
        daily_est = estimate_daily(daily, surfaces, surface_index)
        q_bins = bin_qs(daily)
        result = flow_normalize(daily_est, surfaces, surface_index, q_bins)
        valid = result['FNConc'].notna()
        assert np.all(np.isfinite(result.loc[valid, 'FNConc'].values))
        assert np.all(np.isfinite(result.loc[valid, 'FNFlux'].values))
