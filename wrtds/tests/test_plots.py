"""Smoke tests for wrtds.plots — every function runs and returns a Figure."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from wrtds import WRTDS
from wrtds.summaries import setup_years
from wrtds.plots import (
    box_conc_month,
    box_q_twice,
    box_resid_month,
    flux_bias_multi,
    plot_conc_hist,
    plot_conc_pred,
    plot_conc_q,
    plot_conc_q_smooth,
    plot_conc_time,
    plot_conc_time_smooth,
    plot_contours,
    plot_diff_contours,
    plot_flux_hist,
    plot_flux_pred,
    plot_overview,
    plot_q_time_daily,
    plot_resid_pred,
    plot_resid_q,
    plot_resid_time,
)


# ---------------------------------------------------------------------------
# Module-scoped fitted model (expensive — built once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
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

    model = WRTDS(daily, sample).fit(
        window_y=7, window_q=2, window_s=0.5,
        min_num_obs=10, min_num_uncen=5,
    )
    return model


@pytest.fixture(scope='module')
def annual(fitted):
    return setup_years(fitted.daily)


# ---------------------------------------------------------------------------
# Data overview plots
# ---------------------------------------------------------------------------

class TestDataOverview:
    def test_plot_q_time_daily(self, fitted):
        fig = plot_q_time_daily(fitted.daily)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_conc_time(self, fitted):
        fig = plot_conc_time(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_conc_q(self, fitted):
        fig = plot_conc_q(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_box_conc_month(self, fitted):
        fig = box_conc_month(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_box_q_twice(self, fitted):
        fig = box_q_twice(fitted.daily, fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_overview(self, fitted):
        fig = plot_overview(fitted.daily, fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_plot_conc_pred(self, fitted):
        fig = plot_conc_pred(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_flux_pred(self, fitted):
        fig = plot_flux_pred(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_resid_pred(self, fitted):
        fig = plot_resid_pred(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_resid_q(self, fitted):
        fig = plot_resid_q(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_resid_time(self, fitted):
        fig = plot_resid_time(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_box_resid_month(self, fitted):
        fig = box_resid_month(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_flux_bias_multi(self, fitted):
        fig = flux_bias_multi(fitted.sample)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Result / trend plots
# ---------------------------------------------------------------------------

class TestResults:
    def test_plot_conc_hist(self, annual):
        fig = plot_conc_hist(annual)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_flux_hist(self, annual):
        fig = plot_flux_hist(annual)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_contours(self, fitted):
        fig = plot_contours(fitted.surfaces, fitted.surface_index)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_conc_q_smooth(self, fitted):
        years = [2001, 2003, 2005]
        fig = plot_conc_q_smooth(fitted.surfaces, fitted.surface_index, years)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_conc_time_smooth(self, fitted):
        logq_grid = fitted.surface_index['logq_grid']
        logq_values = [logq_grid[2], logq_grid[7], logq_grid[11]]
        fig = plot_conc_time_smooth(fitted.surfaces, fitted.surface_index, logq_values)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_diff_contours(self, fitted):
        # Compare surface against itself (diff = 0) as a smoke test
        fig = plot_diff_contours(fitted.surfaces, fitted.surfaces, fitted.surface_index)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# WRTDS class plot wrappers
# ---------------------------------------------------------------------------

class TestWRTDSPlotMethods:
    def test_plot_overview(self, fitted):
        fig = fitted.plot_overview()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_conc_hist(self, fitted):
        fig = fitted.plot_conc_hist()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_flux_hist(self, fitted):
        fig = fitted.plot_flux_hist()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_contours(self, fitted):
        fig = fitted.plot_contours()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_conc_pred(self, fitted):
        fig = fitted.plot_conc_pred()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_residuals(self, fitted):
        fig = fitted.plot_residuals()
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
