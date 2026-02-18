"""Tests for wrtds.core — WRTDS class integration."""

import numpy as np
import pandas as pd
import pytest

from wrtds import WRTDS
from wrtds.data_prep import DEFAULT_INFO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_daily():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    return pd.DataFrame({'Date': dates, 'Q': q})


@pytest.fixture
def raw_sample(raw_daily):
    rng = np.random.default_rng(42)
    dates = raw_daily['Date'].iloc[::15]
    n = len(dates)
    conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
    conc_low = conc.copy()
    conc_high = conc.copy()
    conc_low[:3] = np.nan
    conc_high[:3] = conc[:3] * 1.5
    return pd.DataFrame({'Date': dates.values, 'ConcLow': conc_low, 'ConcHigh': conc_high})


@pytest.fixture
def model(raw_daily, raw_sample):
    return WRTDS(raw_daily, raw_sample)


@pytest.fixture(scope='module')
def fitted_model():
    """A fully fitted WRTDS model (expensive — run once per module)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    daily = pd.DataFrame({'Date': dates, 'Q': q})

    sample_dates = dates[::15]
    n = len(sample_dates)
    conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
    conc_low = conc.copy()
    conc_high = conc.copy()
    conc_low[:3] = np.nan
    conc_high[:3] = conc[:3] * 1.5
    sample = pd.DataFrame({'Date': sample_dates, 'ConcLow': conc_low, 'ConcHigh': conc_high})

    return WRTDS(daily, sample).fit(
        window_y=7, window_q=2, window_s=0.5,
        min_num_obs=10, min_num_uncen=5,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestInit:
    def test_daily_populated(self, model):
        assert 'LogQ' in model.daily.columns
        assert 'DecYear' in model.daily.columns

    def test_sample_populated(self, model):
        assert 'SinDY' in model.sample.columns
        assert 'Q' in model.sample.columns
        assert 'Uncen' in model.sample.columns

    def test_info_defaults(self, model):
        for key, value in DEFAULT_INFO.items():
            assert model.info[key] == value

    def test_info_override(self, raw_daily, raw_sample):
        m = WRTDS(raw_daily, raw_sample, info={'station_name': 'Test Site'})
        assert m.info['station_name'] == 'Test Site'
        assert m.info['pa_start'] == 10  # default still present

    def test_surfaces_none_before_fit(self, model):
        assert model.surfaces is None
        assert model.surface_index is None

    def test_conc_remark_format(self, raw_daily):
        """Constructor should accept Conc+Remark format."""
        sample = pd.DataFrame({
            'Date': raw_daily['Date'].iloc[:5].values,
            'Conc': [5.0, 0.5, 3.0, 8.0, 1.0],
            'Remark': ['', '<', '', '<', ''],
        })
        m = WRTDS(raw_daily, sample)
        assert 'ConcLow' in m.sample.columns
        assert 'ConcHigh' in m.sample.columns


# ---------------------------------------------------------------------------
# fit() — full pipeline
# ---------------------------------------------------------------------------

class TestFit:
    def test_returns_self(self, fitted_model):
        assert isinstance(fitted_model, WRTDS)

    def test_chaining(self, raw_daily, raw_sample):
        """fit() should return self so calls can be chained."""
        m = WRTDS(raw_daily, raw_sample).fit(
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
        )
        assert isinstance(m, WRTDS)
        assert m.surfaces is not None

    def test_surfaces_populated(self, fitted_model):
        assert fitted_model.surfaces is not None
        assert fitted_model.surface_index is not None
        assert fitted_model.surfaces.ndim == 3
        assert fitted_model.surfaces.shape[2] == 3

    def test_sample_has_cv_columns(self, fitted_model):
        for col in ('yHat', 'SE', 'ConcHat'):
            assert col in fitted_model.sample.columns

    def test_daily_has_estimation_columns(self, fitted_model):
        for col in ('yHat', 'SE', 'ConcDay', 'FluxDay', 'FNConc', 'FNFlux'):
            assert col in fitted_model.daily.columns

    def test_concday_positive(self, fitted_model):
        assert (fitted_model.daily['ConcDay'] > 0).all()

    def test_fnconc_positive(self, fitted_model):
        valid = fitted_model.daily['FNConc'].notna()
        assert (fitted_model.daily.loc[valid, 'FNConc'] > 0).all()

    def test_fluxday_formula(self, fitted_model):
        d = fitted_model.daily
        expected = d['ConcDay'] * d['Q'] * 86.4
        np.testing.assert_allclose(d['FluxDay'], expected, rtol=1e-12)

    def test_all_daily_values_finite(self, fitted_model):
        for col in ('yHat', 'SE', 'ConcDay', 'FluxDay'):
            assert np.all(np.isfinite(fitted_model.daily[col].values))


# ---------------------------------------------------------------------------
# Individual sub-steps
# ---------------------------------------------------------------------------

class TestSubSteps:
    def test_cross_validate_only(self, raw_daily, raw_sample):
        m = WRTDS(raw_daily, raw_sample)
        result = m.cross_validate(
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
        )
        assert result is m
        assert 'yHat' in m.sample.columns

    def test_estimate_surfaces_only(self, raw_daily, raw_sample):
        m = WRTDS(raw_daily, raw_sample)
        result = m.estimate_surfaces(
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
        )
        assert result is m
        assert m.surfaces is not None
        assert m.surface_index is not None

    def test_estimate_daily_requires_surfaces(self, raw_daily, raw_sample):
        m = WRTDS(raw_daily, raw_sample)
        with pytest.raises(RuntimeError, match='estimate_surfaces'):
            m.estimate_daily()

    def test_estimate_daily_after_surfaces(self, raw_daily, raw_sample):
        m = WRTDS(raw_daily, raw_sample)
        m.estimate_surfaces(
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
        )
        result = m.estimate_daily()
        assert result is m
        assert 'FNConc' in m.daily.columns


# ---------------------------------------------------------------------------
# Import from package root
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_from_wrtds(self):
        from wrtds import WRTDS as W
        assert W is WRTDS
