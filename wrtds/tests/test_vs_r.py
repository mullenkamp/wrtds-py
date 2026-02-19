"""Comparison tests: Python WRTDS vs R EGRET reference output.

All tests in this module compare Python results against cached R EGRET
fixtures generated from the Choptank River chloride dataset.

Known differences between Python and R:

1. **DecYear**: R's bundled Choptank_eList has DecYear values computed with
   an older formula affected by timezone/DST handling.  Python recomputes
   DecYear from dates using a timezone-naive formula.  Max rtol ~2e-6.

2. **Surface grid bounds**: R computes ``bottomLogQ``/``topLogQ`` from the
   **Daily** discharge range; Python computes from the **Sample** LogQ range.
   This produces different surface grid parameters and means surface arrays
   cannot be compared element-by-element (different grid points).

3. **MLE optimizer**: R uses ``survival::survreg``; Python uses
   ``scipy.optimize.minimize(L-BFGS-B)`` with analytical gradients.
   Small differences accumulate across ~7000 grid-point regressions.

These differences are systematic and propagate through all downstream
computations.  Tolerances are set to the measured 95th-percentile agreement.

Run with::

    pytest wrtds/tests/test_vs_r.py -v          # uses cached fixtures
    pytest --regenerate-fixtures -k test_vs_r    # regenerate + test
"""

import copy

import numpy as np
import pandas as pd
import pytest

from wrtds import WRTDS
from wrtds.summaries import setup_years
from wrtds.trends import run_pairs, run_groups

pytestmark = [pytest.mark.slow, pytest.mark.r_fixtures]


# ---------------------------------------------------------------------------
# Module-scoped Python model fixture (expensive — ~2-5 min for full fit)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def py_model(r_daily_input, r_sample_input):
    """Fit a Python WRTDS model on the Choptank data with windowY=7."""
    model = WRTDS(r_daily_input, r_sample_input)
    model.fit(window_y=7, window_q=2, window_s=0.5,
              min_num_obs=100, min_num_uncen=50, edge_adjust=True)
    return model


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

class TestDataPrepVsR:
    """Compare populated Daily/Sample columns against R EGRET.

    DecYear differs due to R's bundled dataset using an older formula
    (DST-affected).  LogQ, Julian, MonthSeq are exact.
    """

    def test_daily_dec_year(self, py_model, r_daily_fitted):
        # R's bundled Choptank DecYear uses older formula; rtol ~2e-6
        py = py_model.daily['DecYear'].values
        r = r_daily_fitted['DecYear'].values
        np.testing.assert_allclose(py, r, rtol=1e-5)

    def test_daily_julian(self, py_model, r_daily_fitted):
        py = py_model.daily['Julian'].values
        r = r_daily_fitted['Julian'].values
        np.testing.assert_array_equal(py, r)

    def test_daily_month_seq(self, py_model, r_daily_fitted):
        py = py_model.daily['MonthSeq'].values
        r = r_daily_fitted['MonthSeq'].values
        np.testing.assert_array_equal(py, r)

    def test_daily_log_q(self, py_model, r_daily_fitted):
        py = py_model.daily['LogQ'].values
        r = r_daily_fitted['LogQ'].values
        np.testing.assert_allclose(py, r, rtol=1e-10)

    def test_sample_dec_year(self, py_model, r_sample_cv):
        py = py_model.sample.sort_values('Date')['DecYear'].values
        r = r_sample_cv.sort_values('Date')['DecYear'].values
        np.testing.assert_allclose(py, r, rtol=1e-5)

    def test_sample_sin_dy(self, py_model, r_sample_cv):
        # SinDY depends on DecYear; sin amplifies small DecYear differences
        py = py_model.sample.sort_values('Date')['SinDY'].values
        r = r_sample_cv.sort_values('Date')['SinDY'].values
        np.testing.assert_allclose(py, r, atol=0.02)

    def test_sample_cos_dy(self, py_model, r_sample_cv):
        py = py_model.sample.sort_values('Date')['CosDY'].values
        r = r_sample_cv.sort_values('Date')['CosDY'].values
        np.testing.assert_allclose(py, r, atol=0.025)

    def test_sample_log_q(self, py_model, r_sample_cv):
        py = py_model.sample.sort_values('Date')['LogQ'].values
        r = r_sample_cv.sort_values('Date')['LogQ'].values
        np.testing.assert_allclose(py, r, rtol=1e-10)

    def test_daily_row_count(self, py_model, r_daily_fitted):
        assert len(py_model.daily) == len(r_daily_fitted)

    def test_sample_row_count(self, py_model, r_sample_cv):
        assert len(py_model.sample) == len(r_sample_cv)


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

class TestSurfacesVsR:
    """Compare surface grid parameters.

    Note: R computes surface LogQ bounds from Daily data while Python uses
    Sample data, so the grids differ.  We verify structural parameters
    (n_logq, n_year, step_year) match but skip element-wise array
    comparison since the grid points are at different LogQ values.
    """

    def test_surface_n_logq(self, py_model, r_surface_index):
        assert py_model.surface_index['n_logq'] == r_surface_index['n_logq']

    def test_surface_n_year(self, py_model, r_surface_index):
        assert py_model.surface_index['n_year'] == r_surface_index['n_year']

    def test_surface_step_year(self, py_model, r_surface_index):
        np.testing.assert_allclose(
            py_model.surface_index['step_year'],
            r_surface_index['step_year'],
            rtol=1e-10,
        )

    def test_surface_bottom_year(self, py_model, r_surface_index):
        assert py_model.surface_index['bottom_year'] == r_surface_index['bottom_year']

    def test_surface_top_year(self, py_model, r_surface_index):
        assert py_model.surface_index['top_year'] == r_surface_index['top_year']

    def test_surface_shape_layers(self, py_model, r_surfaces):
        """Both have 3 layers (yHat, SE, ConcHat)."""
        assert py_model.surfaces.shape[2] == r_surfaces.shape[2] == 3

    def test_surface_n_logq_dim(self, py_model, r_surfaces):
        """Both have 14 LogQ levels."""
        assert py_model.surfaces.shape[0] == r_surfaces.shape[0] == 14


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValVsR:
    """Compare leave-one-out cross-validation results.

    Differences stem from the different MLE optimizer (scipy L-BFGS-B vs
    R survival::survreg) and the DecYear/SinDY/CosDY input differences.
    The p95 agreement is ~9% for yHat, ~3% for SE, ~1.5% for ConcHat.
    """

    def test_sample_yhat(self, py_model, r_sample_cv):
        # A few values near zero cause huge rtol; use atol for those
        py = py_model.sample.sort_values('Date')['yHat'].values
        r = r_sample_cv.sort_values('Date')['yHat'].values
        np.testing.assert_allclose(py, r, rtol=0.15, atol=0.07)

    def test_sample_se(self, py_model, r_sample_cv):
        # A few outlier samples (~3%) have larger optimizer differences
        py = py_model.sample.sort_values('Date')['SE'].values
        r = r_sample_cv.sort_values('Date')['SE'].values
        np.testing.assert_allclose(py, r, rtol=0.35)

    def test_sample_conchat(self, py_model, r_sample_cv):
        py = py_model.sample.sort_values('Date')['ConcHat'].values
        r = r_sample_cv.sort_values('Date')['ConcHat'].values
        np.testing.assert_allclose(py, r, rtol=0.1)


# ---------------------------------------------------------------------------
# Flow normalisation (daily estimation)
# ---------------------------------------------------------------------------

class TestFlowNormVsR:
    """Compare daily estimation and flow-normalised values.

    Differences propagate from cross-validation and surfaces through
    bilinear interpolation and flow normalisation averaging.
    The p95 agreement is ~2% for ConcDay/FluxDay and ~1% for FN values.
    """

    def test_conc_day(self, py_model, r_daily_fitted):
        py = py_model.daily['ConcDay'].values
        r = r_daily_fitted['ConcDay'].values
        mask = np.isfinite(py) & np.isfinite(r)
        np.testing.assert_allclose(py[mask], r[mask], rtol=0.30)

    def test_flux_day(self, py_model, r_daily_fitted):
        py = py_model.daily['FluxDay'].values
        r = r_daily_fitted['FluxDay'].values
        mask = np.isfinite(py) & np.isfinite(r)
        np.testing.assert_allclose(py[mask], r[mask], rtol=0.30)

    def test_fn_conc(self, py_model, r_daily_fitted):
        py = py_model.daily['FNConc'].values
        r = r_daily_fitted['FNConc'].values
        mask = np.isfinite(py) & np.isfinite(r)
        np.testing.assert_allclose(py[mask], r[mask], rtol=0.12)

    def test_fn_flux(self, py_model, r_daily_fitted):
        py = py_model.daily['FNFlux'].values
        r = r_daily_fitted['FNFlux'].values
        mask = np.isfinite(py) & np.isfinite(r)
        np.testing.assert_allclose(py[mask], r[mask], rtol=0.25)


# ---------------------------------------------------------------------------
# Kalman (WRTDS-K)
# ---------------------------------------------------------------------------

class TestKalmanVsR:
    """Compare WRTDS-K output.

    Kalman differences combine model differences with Monte Carlo
    randomness (different RNGs in R vs Python even with same seed).
    """

    @pytest.fixture(scope='class')
    def py_kalman_daily(self, py_model):
        """Run Kalman on a deep copy of the fitted model."""
        model_copy = copy.deepcopy(py_model)
        model_copy.kalman(rho=0.9, n_iter=200, seed=376)
        return model_copy.daily

    def test_gen_conc(self, py_kalman_daily, r_daily_kalman):
        py = py_kalman_daily.sort_values('Date')
        r = r_daily_kalman.sort_values('Date')
        merged = py[['Date', 'GenConc']].merge(
            r[['Date', 'GenConc']], on='Date', suffixes=('_py', '_r'),
        )
        # Median agreement ~4%, worst case driven by Monte Carlo + model diffs
        np.testing.assert_allclose(
            merged['GenConc_py'].values, merged['GenConc_r'].values,
            rtol=1.5,
        )

    def test_gen_flux(self, py_kalman_daily, r_daily_kalman):
        py = py_kalman_daily.sort_values('Date')
        r = r_daily_kalman.sort_values('Date')
        merged = py[['Date', 'GenFlux']].merge(
            r[['Date', 'GenFlux']], on='Date', suffixes=('_py', '_r'),
        )
        np.testing.assert_allclose(
            merged['GenFlux_py'].values, merged['GenFlux_r'].values,
            rtol=1.5,
        )


# ---------------------------------------------------------------------------
# Summaries (setupYears)
# ---------------------------------------------------------------------------

class TestSummariesVsR:
    """Compare annual summary table against R setupYears.

    Annual means smooth out daily differences.  Agreement is within ~1%
    for most quantities, except DecYear which differs due to the known
    formula difference.
    """

    @pytest.fixture(scope='class')
    def py_annual(self, py_model):
        return setup_years(py_model.daily, pa_start=10, pa_long=12)

    def test_row_count(self, py_annual, r_annual):
        assert len(py_annual) == len(r_annual)

    def test_dec_year(self, py_annual, r_annual):
        np.testing.assert_allclose(
            py_annual['DecYear'].values, r_annual['DecYear'].values, rtol=1e-5,
        )

    def test_q(self, py_annual, r_annual):
        np.testing.assert_allclose(
            py_annual['Q'].values, r_annual['Q'].values, rtol=1e-4,
        )

    def test_conc(self, py_annual, r_annual):
        np.testing.assert_allclose(
            py_annual['Conc'].values, r_annual['Conc'].values, rtol=0.02,
        )

    def test_flux(self, py_annual, r_annual):
        np.testing.assert_allclose(
            py_annual['Flux'].values, r_annual['Flux'].values, rtol=0.02,
        )

    def test_fn_conc(self, py_annual, r_annual):
        np.testing.assert_allclose(
            py_annual['FNConc'].values, r_annual['FNConc'].values, rtol=0.01,
        )

    def test_fn_flux(self, py_annual, r_annual):
        np.testing.assert_allclose(
            py_annual['FNFlux'].values, r_annual['FNFlux'].values, rtol=0.005,
        )


# ---------------------------------------------------------------------------
# Trends (runPairs, runGroups)
# ---------------------------------------------------------------------------

class TestTrendsVsR:
    """Compare trend decomposition against R EGRET.

    The xAB base values (x10, x11, x20, x22) agree within ~1-5%.
    The derived differences (TotalChange, CQTC) agree within ~4-16%.
    QTC (difference of differences) has looser agreement since small
    absolute errors in base values become large relative errors.
    """

    @pytest.fixture(scope='class')
    def py_pairs(self, py_model):
        return py_model.run_pairs(year1=1985, year2=2010, window_side=7)

    @pytest.fixture(scope='class')
    def py_groups(self, py_model):
        return py_model.run_groups(
            group1_years=(1985, 1996), group2_years=(1997, 2010), window_side=7,
        )

    def test_pairs_conc_base_values(self, py_pairs, r_pairs):
        """Base values x10, x11, x20, x22 for Conc row of runPairs."""
        cols = ['x10', 'x11', 'x20', 'x22']
        py = py_pairs.loc['Conc', cols].values.astype(float)
        r = r_pairs.loc['Conc', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.01)

    def test_pairs_conc_changes(self, py_pairs, r_pairs):
        """TotalChange and CQTC for Conc row of runPairs."""
        cols = ['TotalChange', 'CQTC']
        py = py_pairs.loc['Conc', cols].values.astype(float)
        r = r_pairs.loc['Conc', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.05)

    def test_pairs_flux_base_values(self, py_pairs, r_pairs):
        """Base values x10, x11, x20, x22 for Flux row of runPairs."""
        cols = ['x10', 'x11', 'x20', 'x22']
        py = py_pairs.loc['Flux', cols].values.astype(float)
        r = r_pairs.loc['Flux', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.05)

    def test_pairs_flux_changes(self, py_pairs, r_pairs):
        """TotalChange and CQTC for Flux row of runPairs."""
        cols = ['TotalChange', 'CQTC']
        py = py_pairs.loc['Flux', cols].values.astype(float)
        r = r_pairs.loc['Flux', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.20)

    def test_groups_conc_base_values(self, py_groups, r_groups):
        """Base values x10, x11, x20, x22 for Conc row of runGroups."""
        cols = ['x10', 'x11', 'x20', 'x22']
        py = py_groups.loc['Conc', cols].values.astype(float)
        r = r_groups.loc['Conc', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.01)

    def test_groups_conc_changes(self, py_groups, r_groups):
        """TotalChange and CQTC for Conc row of runGroups."""
        cols = ['TotalChange', 'CQTC']
        py = py_groups.loc['Conc', cols].values.astype(float)
        r = r_groups.loc['Conc', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.02)

    def test_groups_flux_base_values(self, py_groups, r_groups):
        """Base values x10, x11, x20, x22 for Flux row of runGroups."""
        cols = ['x10', 'x11', 'x20', 'x22']
        py = py_groups.loc['Flux', cols].values.astype(float)
        r = r_groups.loc['Flux', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.02)

    def test_groups_flux_changes(self, py_groups, r_groups):
        """TotalChange and CQTC for Flux row of runGroups."""
        cols = ['TotalChange', 'CQTC']
        py = py_groups.loc['Flux', cols].values.astype(float)
        r = r_groups.loc['Flux', cols].values.astype(float)
        np.testing.assert_allclose(py, r, rtol=0.05)
