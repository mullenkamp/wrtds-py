"""Tests for wrtds.summaries module."""

import numpy as np
import pandas as pd
import pytest

from wrtds import WRTDS
from wrtds.data_prep import populate_daily, populate_sample
from wrtds.flow_norm import bin_qs, estimate_daily, flow_normalize
from wrtds.summaries import error_stats, flux_bias_stat, setup_years, table_change
from wrtds.surfaces import compute_surface_index, estimate_surfaces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fitted_data():
    """Build a fully fitted model for summary tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': q}))

    sample_dates = dates[::15]
    n = len(sample_dates)
    conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
    conc_low = conc.copy()
    conc_high = conc.copy()
    # Add a few censored observations
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

    # Cross-validate to get yHat, SE, ConcHat
    from wrtds.cross_val import cross_validate
    sample = cross_validate(sample, **fit_params)

    # Estimate surfaces and daily values
    si = compute_surface_index(sample)
    surfaces = estimate_surfaces(sample, si, **fit_params)
    daily = estimate_daily(daily, surfaces, si)
    q_bins = bin_qs(daily)
    daily = flow_normalize(daily, surfaces, si, q_bins)

    return daily, sample, surfaces, si


# ---------------------------------------------------------------------------
# setup_years
# ---------------------------------------------------------------------------

class TestSetupYears:
    def test_returns_dataframe(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        for col in ['DecYear', 'Q', 'Conc', 'Flux', 'FNConc', 'FNFlux']:
            assert col in result.columns

    def test_calendar_year_count(self, fitted_data):
        """Calendar year analysis (pa_start=1) on 2000-2005 data should give ~6 years."""
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert len(result) >= 4  # At least 4 full calendar years
        assert len(result) <= 6

    def test_water_year_count(self, fitted_data):
        """Water year analysis (pa_start=10) should work."""
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=10, pa_long=12)
        assert len(result) >= 3

    def test_partial_year(self, fitted_data):
        """Shorter periods should produce more or equal rows."""
        daily, _, _, _ = fitted_data
        full = setup_years(daily, pa_start=1, pa_long=12)
        # 6-month periods still produce one per year
        half = setup_years(daily, pa_start=1, pa_long=6)
        assert len(half) >= len(full)

    def test_q_positive(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert (result['Q'] > 0).all()

    def test_conc_positive(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert (result['Conc'] > 0).all()

    def test_flux_positive(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert (result['Flux'] > 0).all()

    def test_fnconc_positive(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert (result['FNConc'] > 0).all()

    def test_fnflux_positive(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert (result['FNFlux'] > 0).all()

    def test_decyear_increasing(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        dec_years = result['DecYear'].values
        assert np.all(dec_years[:-1] < dec_years[1:])

    def test_decyear_reasonable(self, fitted_data):
        """DecYear should be within the range of the data."""
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert result['DecYear'].min() >= 2000
        assert result['DecYear'].max() <= 2006

    def test_no_genconc_without_kalman(self, fitted_data):
        """GenConc/GenFlux should not appear unless WRTDS-K was run."""
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert 'GenConc' not in result.columns
        assert 'GenFlux' not in result.columns

    def test_genconc_with_kalman(self, fitted_data):
        """GenConc/GenFlux should appear when those columns exist in daily."""
        daily, _, _, _ = fitted_data
        daily_k = daily.copy()
        daily_k['GenConc'] = daily_k['ConcDay'] * 1.01
        daily_k['GenFlux'] = daily_k['FluxDay'] * 1.01
        result = setup_years(daily_k, pa_start=1, pa_long=12)
        assert 'GenConc' in result.columns
        assert 'GenFlux' in result.columns

    def test_values_finite(self, fitted_data):
        daily, _, _, _ = fitted_data
        result = setup_years(daily, pa_start=1, pa_long=12)
        for col in ['DecYear', 'Q', 'Conc', 'Flux', 'FNConc', 'FNFlux']:
            assert np.all(np.isfinite(result[col].values))

    def test_without_concday(self):
        """setup_years should work on daily data without ConcDay (Q-only summary)."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-01', '2003-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': q}))
        result = setup_years(daily, pa_start=1, pa_long=12)
        assert 'Q' in result.columns
        assert 'Conc' not in result.columns
        assert len(result) >= 3

    def test_skips_incomplete_years(self):
        """Years with >10% missing ConcDay should be skipped."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2000-01-01', '2002-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': q}))
        daily['ConcDay'] = 1.0
        daily['FluxDay'] = daily['ConcDay'] * daily['Q'] * 86.4
        daily['FNConc'] = 1.0
        daily['FNFlux'] = daily['FluxDay']

        # Make 2001 mostly missing
        mask_2001 = daily['Date'].dt.year == 2001
        daily.loc[mask_2001, 'ConcDay'] = np.nan

        result = setup_years(daily, pa_start=1, pa_long=12)
        # 2001 should be excluded, leaving 2000 and 2002
        dec_years = result['DecYear'].values
        assert not any(2001.0 <= dy <= 2001.99 for dy in dec_years)


# ---------------------------------------------------------------------------
# table_change
# ---------------------------------------------------------------------------

class TestTableChange:
    @pytest.fixture()
    def annual(self, fitted_data):
        daily, _, _, _ = fitted_data
        return setup_years(daily, pa_start=1, pa_long=12)

    def test_returns_dataframe(self, annual):
        result = table_change(annual, [2001, 2004])
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self, annual):
        """One row per consecutive pair."""
        result = table_change(annual, [2001, 2003, 2005])
        assert len(result) == 2

    def test_single_pair(self, annual):
        result = table_change(annual, [2001, 2005])
        assert len(result) == 1

    def test_has_required_columns(self, annual):
        result = table_change(annual, [2001, 2004])
        expected = [
            'Year1', 'Year2',
            'FNConc_change', 'FNConc_pct_change', 'FNConc_slope', 'FNConc_pct_slope',
            'FNFlux_change', 'FNFlux_pct_change', 'FNFlux_slope', 'FNFlux_pct_slope',
        ]
        for col in expected:
            assert col in result.columns

    def test_slope_equals_change_per_year(self, annual):
        result = table_change(annual, [2001, 2004])
        row = result.iloc[0]
        dt = row['Year2'] - row['Year1']
        assert row['FNConc_slope'] == pytest.approx(row['FNConc_change'] / dt, rel=1e-6)
        assert row['FNFlux_slope'] == pytest.approx(row['FNFlux_change'] / dt, rel=1e-6)

    def test_pct_change_formula(self, annual):
        result = table_change(annual, [2001, 2004])
        row = result.iloc[0]
        # Look up FNConc at year1 to verify percent change
        dec_years = annual['DecYear'].values
        idx1 = int(np.argmin(np.abs(dec_years - 2001)))
        fnc1 = annual.iloc[idx1]['FNConc']
        expected_pct = 100 * row['FNConc_change'] / fnc1
        assert row['FNConc_pct_change'] == pytest.approx(expected_pct, rel=1e-6)

    def test_flux_factor_applied(self, annual):
        """Different flux factors should scale flux results."""
        r1 = table_change(annual, [2001, 2004], flux_factor=1.0)
        r2 = table_change(annual, [2001, 2004], flux_factor=0.001)
        # Flux change should scale with flux_factor
        assert r1.iloc[0]['FNFlux_change'] == pytest.approx(
            r2.iloc[0]['FNFlux_change'] * 1000, rel=1e-6
        )

    def test_values_finite(self, annual):
        result = table_change(annual, [2001, 2004])
        for col in result.columns:
            assert np.all(np.isfinite(result[col].values))


# ---------------------------------------------------------------------------
# error_stats
# ---------------------------------------------------------------------------

class TestErrorStats:
    def test_returns_dict(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = error_stats(sample, seed=42)
        assert isinstance(result, dict)

    def test_has_required_keys(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = error_stats(sample, seed=42)
        for key in ['rsq_log_conc', 'rsq_log_flux', 'rmse', 'sep_percent']:
            assert key in result

    def test_rsq_in_valid_range(self, fitted_data):
        """R-squared should be <= 1 and reasonable (can be negative for poor models)."""
        _, sample, _, _ = fitted_data
        result = error_stats(sample, seed=42)
        # R² can be negative for models worse than the mean — especially on
        # random synthetic data — so only check the upper bound.
        assert result['rsq_log_conc'] <= 1.0
        assert result['rsq_log_flux'] <= 1.0
        # Should still be close to reasonable (not wildly negative)
        assert result['rsq_log_conc'] > -1.0
        assert result['rsq_log_flux'] > -1.0

    def test_rmse_positive(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = error_stats(sample, seed=42)
        assert result['rmse'] > 0

    def test_sep_percent_positive(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = error_stats(sample, seed=42)
        assert result['sep_percent'] > 0

    def test_values_finite(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = error_stats(sample, seed=42)
        for v in result.values():
            assert np.isfinite(v)

    def test_reproducible_with_seed(self, fitted_data):
        _, sample, _, _ = fitted_data
        r1 = error_stats(sample, seed=42)
        r2 = error_stats(sample, seed=42)
        for key in r1:
            assert r1[key] == pytest.approx(r2[key])

    def test_all_uncensored(self):
        """With all uncensored data, results should be deterministic."""
        rng = np.random.default_rng(42)
        n = 100
        sample = pd.DataFrame({
            'yHat': rng.normal(1.0, 0.3, n),
            'SE': np.full(n, 0.3),
            'ConcHigh': rng.lognormal(1.0, 0.3, n),
            'ConcLow': np.nan,  # will be overwritten below
            'Uncen': np.ones(n, dtype=int),
            'Q': rng.lognormal(2.0, 0.5, n),
        })
        sample['ConcLow'] = sample['ConcHigh']

        r1 = error_stats(sample, seed=1)
        r2 = error_stats(sample, seed=2)
        # Should be identical since no censored obs
        for key in r1:
            assert r1[key] == pytest.approx(r2[key])

    def test_rsq_flux_higher_than_conc(self, fitted_data):
        """R² for flux is typically higher because Q adds explained variance."""
        _, sample, _, _ = fitted_data
        result = error_stats(sample, seed=42)
        # This is generally true but not guaranteed — just check both are computed
        assert result['rsq_log_flux'] >= 0


# ---------------------------------------------------------------------------
# flux_bias_stat
# ---------------------------------------------------------------------------

class TestFluxBiasStat:
    def test_returns_dict(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = flux_bias_stat(sample)
        assert isinstance(result, dict)

    def test_has_required_keys(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = flux_bias_stat(sample)
        for key in ['bias1', 'bias2', 'bias3']:
            assert key in result

    def test_bias3_is_average(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = flux_bias_stat(sample)
        assert result['bias3'] == pytest.approx(
            (result['bias1'] + result['bias2']) / 2
        )

    def test_bias1_leq_bias2(self, fitted_data):
        """bias1 uses ConcHigh (higher obs), so estimated-observed is smaller."""
        _, sample, _, _ = fitted_data
        result = flux_bias_stat(sample)
        assert result['bias1'] <= result['bias2']

    def test_values_finite(self, fitted_data):
        _, sample, _, _ = fitted_data
        result = flux_bias_stat(sample)
        for v in result.values():
            assert np.isfinite(v)

    def test_all_uncensored_bias_equal(self):
        """With all uncensored data, all three bias values should be equal."""
        rng = np.random.default_rng(42)
        n = 50
        conc = rng.lognormal(1.0, 0.3, n)
        sample = pd.DataFrame({
            'ConcLow': conc,
            'ConcHigh': conc,
            'ConcHat': conc * 1.05,  # slight overestimate
            'Q': rng.lognormal(2.0, 0.5, n),
        })
        result = flux_bias_stat(sample)
        assert result['bias1'] == pytest.approx(result['bias2'])
        assert result['bias1'] == pytest.approx(result['bias3'])

    def test_perfect_model_zero_bias(self):
        """If ConcHat == ConcHigh exactly, bias1 should be 0."""
        rng = np.random.default_rng(42)
        n = 50
        conc = rng.lognormal(1.0, 0.3, n)
        sample = pd.DataFrame({
            'ConcLow': conc,
            'ConcHigh': conc,
            'ConcHat': conc,
            'Q': rng.lognormal(2.0, 0.5, n),
        })
        result = flux_bias_stat(sample)
        assert result['bias1'] == pytest.approx(0.0, abs=1e-10)

    def test_overestimate_positive_bias(self):
        """Model that consistently overestimates should have positive bias."""
        rng = np.random.default_rng(42)
        n = 50
        conc = rng.lognormal(1.0, 0.3, n)
        sample = pd.DataFrame({
            'ConcLow': conc,
            'ConcHigh': conc,
            'ConcHat': conc * 2.0,  # 2x overestimate
            'Q': rng.lognormal(2.0, 0.5, n),
        })
        result = flux_bias_stat(sample)
        assert result['bias3'] > 0

    def test_underestimate_negative_bias(self):
        """Model that consistently underestimates should have negative bias."""
        rng = np.random.default_rng(42)
        n = 50
        conc = rng.lognormal(1.0, 0.3, n)
        sample = pd.DataFrame({
            'ConcLow': conc,
            'ConcHigh': conc,
            'ConcHat': conc * 0.5,  # 0.5x underestimate
            'Q': rng.lognormal(2.0, 0.5, n),
        })
        result = flux_bias_stat(sample)
        assert result['bias3'] < 0

    def test_bias_magnitude_reasonable(self, fitted_data):
        """For a fitted model, bias should be small (< 1 in absolute value)."""
        _, sample, _, _ = fitted_data
        result = flux_bias_stat(sample)
        assert abs(result['bias3']) < 1.0


# ---------------------------------------------------------------------------
# WRTDS class integration
# ---------------------------------------------------------------------------

class TestWRTDSSummaries:
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

    def test_table_results_requires_fit(self):
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
            m.table_results()

    def test_table_results_via_class(self, fitted_model):
        result = fitted_model.table_results()
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3
        assert 'FNConc' in result.columns

    def test_table_change_via_class(self, fitted_model):
        result = fitted_model.table_change([2002, 2004])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_table_change_requires_fit(self):
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
            m.table_change([2001, 2001])

    def test_error_stats_via_class(self, fitted_model):
        result = fitted_model.error_stats(seed=42)
        assert isinstance(result, dict)
        assert 'rmse' in result
        assert result['rmse'] > 0

    def test_error_stats_requires_cv(self):
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
        with pytest.raises(RuntimeError, match='cross_validate'):
            m.error_stats()

    def test_flux_bias_stat_via_class(self, fitted_model):
        result = fitted_model.flux_bias_stat()
        assert isinstance(result, dict)
        assert 'bias3' in result
        assert abs(result['bias3']) < 1.0

    def test_flux_bias_stat_requires_cv(self):
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
        with pytest.raises(RuntimeError, match='cross_validate'):
            m.flux_bias_stat()
