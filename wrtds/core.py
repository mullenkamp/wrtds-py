"""WRTDS class — main entry point for Weighted Regressions on Time, Discharge, and Season."""

import numpy as np
import pandas as pd

from wrtds.cross_val import cross_validate as _cross_validate
from wrtds.data_prep import DEFAULT_INFO, populate_daily, populate_sample
from wrtds.flow_norm import bin_qs, estimate_daily as _estimate_daily, flow_normalize
from wrtds.kalman import wrtds_kalman as _wrtds_kalman
from wrtds.surfaces import (
    compute_surface_index,
    estimate_surfaces as _estimate_surfaces,
)
from wrtds.bootstrap import (
    bootstrap_groups as _bootstrap_groups,
    bootstrap_pairs as _bootstrap_pairs,
)
from wrtds.summaries import (
    error_stats as _error_stats,
    flux_bias_stat as _flux_bias_stat,
    setup_years as _setup_years,
    table_change as _table_change,
)
from wrtds.trends import (
    run_groups as _run_groups,
    run_pairs as _run_pairs,
    run_series as _run_series,
)


class WRTDS:
    """Weighted Regressions on Time, Discharge, and Season.

    This is the primary user-facing class.  It wraps the lower-level modules
    (``data_prep``, ``regression``, ``surfaces``, ``flow_norm``, ``cross_val``)
    behind a fluent API where mutating methods return ``self`` for chaining::

        model = WRTDS(daily_df, sample_df).fit()
        print(model.daily[['Date', 'ConcDay', 'FluxDay', 'FNConc', 'FNFlux']])

    Attributes:
        daily: Daily discharge DataFrame (populated with derived columns).
        sample: Water-quality sample DataFrame (populated with derived columns).
        info: Site / parameter metadata dict.
        surfaces: 3-D numpy array ``(n_logq, n_year, 3)`` after :meth:`fit`.
        surface_index: Grid parameter dict after :meth:`fit`.
    """

    def __init__(self, daily, sample, info=None):
        """Validate and prepare input DataFrames.

        Args:
            daily: DataFrame with at least ``Date`` and ``Q`` (m³/s).
            sample: DataFrame with ``Date``, ``ConcLow``, ``ConcHigh``
                (or ``Date``, ``Conc``, ``Remark``).
            info: Optional metadata dict.  Missing keys are filled from
                :data:`~wrtds.data_prep.DEFAULT_INFO`.
        """
        self.daily = populate_daily(daily)
        self.sample = populate_sample(sample, self.daily)

        self.info = {**DEFAULT_INFO}
        if info is not None:
            self.info.update(info)

        self.surfaces = None
        self.surface_index = None

        # Store fit parameters so sub-steps can reuse them
        self._fit_params = {}

    # ------------------------------------------------------------------
    # Core WRTDS pipeline
    # ------------------------------------------------------------------

    def fit(
        self,
        window_y=7.0,
        window_q=2.0,
        window_s=0.5,
        min_num_obs=100,
        min_num_uncen=50,
        edge_adjust=True,
    ):
        """Run the full WRTDS estimation pipeline.

        1. Leave-one-out cross-validation → ``sample[yHat, SE, ConcHat]``
        2. Surface estimation → ``self.surfaces``, ``self.surface_index``
        3. Daily estimation + flow normalisation → ``daily[ConcDay, FluxDay, FNConc, FNFlux]``

        All parameters are stored so that individual sub-steps called later
        use the same settings.

        Returns:
            ``self`` for chaining.
        """
        self._fit_params = {
            'window_y': window_y,
            'window_q': window_q,
            'window_s': window_s,
            'min_num_obs': min_num_obs,
            'min_num_uncen': min_num_uncen,
            'edge_adjust': edge_adjust,
        }

        self.cross_validate(**self._fit_params)
        self.estimate_surfaces(**self._fit_params)
        self.estimate_daily()

        return self

    def cross_validate(
        self,
        window_y=7.0,
        window_q=2.0,
        window_s=0.5,
        min_num_obs=100,
        min_num_uncen=50,
        edge_adjust=True,
    ):
        """Leave-one-out cross-validation.

        Populates ``self.sample`` with columns ``yHat``, ``SE``, ``ConcHat``.

        Returns:
            ``self`` for chaining.
        """
        self.sample = _cross_validate(
            self.sample,
            window_y=window_y,
            window_q=window_q,
            window_s=window_s,
            min_num_obs=min_num_obs,
            min_num_uncen=min_num_uncen,
            edge_adjust=edge_adjust,
        )
        return self

    def estimate_surfaces(
        self,
        window_y=7.0,
        window_q=2.0,
        window_s=0.5,
        min_num_obs=100,
        min_num_uncen=50,
        edge_adjust=True,
    ):
        """Estimate the concentration surfaces grid.

        Populates ``self.surfaces`` and ``self.surface_index``.

        Returns:
            ``self`` for chaining.
        """
        self.surface_index = compute_surface_index(self.sample)
        self.surfaces = _estimate_surfaces(
            self.sample,
            self.surface_index,
            window_y=window_y,
            window_q=window_q,
            window_s=window_s,
            min_num_obs=min_num_obs,
            min_num_uncen=min_num_uncen,
            edge_adjust=edge_adjust,
        )
        return self

    def estimate_daily(self):
        """Interpolate daily concentrations/fluxes and flow-normalise.

        Requires :meth:`estimate_surfaces` to have been called first.

        Populates ``self.daily`` with columns ``yHat``, ``SE``, ``ConcDay``,
        ``FluxDay``, ``FNConc``, ``FNFlux``.

        Returns:
            ``self`` for chaining.
        """
        if self.surfaces is None:
            raise RuntimeError('estimate_surfaces() must be called before estimate_daily()')

        self.daily = _estimate_daily(self.daily, self.surfaces, self.surface_index)
        q_bins = bin_qs(self.daily)
        self.daily = flow_normalize(self.daily, self.surfaces, self.surface_index, q_bins)

        return self

    # ------------------------------------------------------------------
    # WRTDS-K
    # ------------------------------------------------------------------

    def kalman(self, rho=0.90, n_iter=200, seed=None):
        """Run WRTDS-K (Kalman-style AR(1) residual interpolation).

        Requires :meth:`fit` (or at least :meth:`cross_validate` and
        :meth:`estimate_daily`) to have been called first so that both
        ``sample`` and ``daily`` have ``yHat`` / ``SE`` columns.

        Populates ``self.daily`` with columns ``GenConc`` and ``GenFlux``.

        Args:
            rho: AR(1) autocorrelation (0.85 for reactive, 0.90 default,
                0.95 for conservative constituents).
            n_iter: Monte Carlo iterations (200 for exploration, 500+ for
                publication).
            seed: Optional integer seed for reproducibility.

        Returns:
            ``self`` for chaining.
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before kalman()')
        if 'yHat' not in self.sample.columns:
            raise RuntimeError('cross_validate() must be called before kalman()')
        if 'yHat' not in self.daily.columns:
            raise RuntimeError('estimate_daily() must be called before kalman()')

        self.daily = _wrtds_kalman(
            self.daily,
            self.sample,
            self.surfaces,
            self.surface_index,
            rho=rho,
            n_iter=n_iter,
            seed=seed,
        )
        return self

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def run_pairs(self, year1, year2, window_side=7, pa_start=None, pa_long=None):
        """Compare flow-normalised values between two specific years.

        Estimates separate 1-year surfaces for each year, then decomposes
        the total change into a CQTC (concentration-discharge trend component)
        and a QTC (discharge trend component).

        Requires :meth:`fit` to have been called first.

        Args:
            year1: First comparison year.
            year2: Second comparison year.
            window_side: Half-window for generalized flow normalisation (years).
            pa_start: Period of analysis start month (default: from info).
            pa_long: Period of analysis length in months (default: from info).

        Returns:
            DataFrame with index ``['Conc', 'Flux']`` and columns
            ``['TotalChange', 'CQTC', 'QTC', 'x10', 'x11', 'x20', 'x22']``.
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before run_pairs()')

        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)

        return _run_pairs(
            self.sample, self.daily, year1, year2,
            window_side=window_side,
            pa_start=pa_start,
            pa_long=pa_long,
            fit_params=self._fit_params or None,
        )

    def run_groups(self, group1_years, group2_years, window_side=7,
                   pa_start=None, pa_long=None):
        """Compare flow-normalised averages across two groups of years.

        Uses the existing full-period surface to avoid re-estimation.

        Args:
            group1_years: ``(first_year, last_year)`` for group 1.
            group2_years: ``(first_year, last_year)`` for group 2.
            window_side: Half-window for generalized flow normalisation.
            pa_start: Period of analysis start month (default: from info).
            pa_long: Period of analysis length in months (default: from info).

        Returns:
            DataFrame with same format as :meth:`run_pairs`.
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before run_groups()')

        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)

        return _run_groups(
            self.daily, self.surfaces, self.surface_index,
            group1_years, group2_years,
            window_side=window_side,
            pa_start=pa_start,
            pa_long=pa_long,
        )

    def run_series(self, window_side=7, pa_start=None, pa_long=None):
        """Compute annual time series of generalized flow-normalised values.

        Updates ``self.daily['FNConc']`` and ``self.daily['FNFlux']`` with
        generalized flow-normalised values using a sliding discharge window.

        Args:
            window_side: Half-window for generalized flow normalisation.
            pa_start: Period of analysis start month (default: from info).
            pa_long: Period of analysis length in months (default: from info).

        Returns:
            ``self`` for chaining.
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before run_series()')

        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)

        self.daily = _run_series(
            self.daily, self.surfaces, self.surface_index,
            window_side=window_side,
            pa_start=pa_start,
            pa_long=pa_long,
        )
        return self

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------

    def bootstrap_pairs(self, year1, year2, n_boot=100, block_length=200,
                        window_side=7, pa_start=None, pa_long=None, seed=None):
        """Block bootstrap CI for pairwise trend comparison.

        Args:
            year1: First comparison year.
            year2: Second comparison year.
            n_boot: Number of bootstrap replicates.
            block_length: Block length in days (default 200).
            window_side: Half-window for generalized flow normalisation.
            pa_start: Period of analysis start month (default: from info).
            pa_long: Period of analysis length in months (default: from info).
            seed: Optional integer seed for reproducibility.

        Returns:
            Dict with keys ``observed``, ``boot_conc``, ``boot_flux``,
            ``p_conc``, ``p_flux``, ``ci_conc``, ``ci_flux``,
            ``likelihood_conc_up``, ``likelihood_flux_up``,
            and likelihood descriptor strings.
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before bootstrap_pairs()')

        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)

        return _bootstrap_pairs(
            self.sample, self.daily, year1, year2,
            n_boot=n_boot,
            block_length=block_length,
            window_side=window_side,
            pa_start=pa_start,
            pa_long=pa_long,
            fit_params=self._fit_params or None,
            seed=seed,
        )

    def bootstrap_groups(self, group1_years, group2_years, n_boot=100,
                         block_length=200, window_side=7,
                         pa_start=None, pa_long=None, seed=None):
        """Block bootstrap CI for group trend comparison.

        Args:
            group1_years: ``(first_year, last_year)`` for group 1.
            group2_years: ``(first_year, last_year)`` for group 2.
            n_boot: Number of bootstrap replicates.
            block_length: Block length in days (default 200).
            window_side: Half-window for generalized flow normalisation.
            pa_start: Period of analysis start month (default: from info).
            pa_long: Period of analysis length in months (default: from info).
            seed: Optional integer seed for reproducibility.

        Returns:
            Dict with same keys as :meth:`bootstrap_pairs`.
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before bootstrap_groups()')

        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)

        return _bootstrap_groups(
            self.daily, self.sample,
            self.surfaces, self.surface_index,
            group1_years, group2_years,
            n_boot=n_boot,
            block_length=block_length,
            window_side=window_side,
            pa_start=pa_start,
            pa_long=pa_long,
            fit_params=self._fit_params or None,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------

    def table_results(self, pa_start=None, pa_long=None):
        """Annual summary table of discharge and water-quality results.

        Requires :meth:`fit` to have been called first.

        Args:
            pa_start: Period of analysis start month (default: from info).
            pa_long: Period of analysis length in months (default: from info).

        Returns:
            DataFrame with columns ``DecYear``, ``Q``, ``Conc``, ``Flux``,
            ``FNConc``, ``FNFlux`` (and ``GenConc``, ``GenFlux`` if
            :meth:`kalman` has been run).  Flux values are rates (kg/day).
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before table_results()')

        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)

        return _setup_years(self.daily, pa_start=pa_start, pa_long=pa_long)

    def table_change(self, year_points, flux_factor=0.00036525,
                     pa_start=None, pa_long=None):
        """Changes in flow-normalised values between specified years.

        Requires :meth:`fit` to have been called first.

        Args:
            year_points: List of years at which to evaluate changes.
            flux_factor: Conversion factor from kg/day to desired flux
                units.  Default ``0.00036525`` converts to 10^6 kg/year.
            pa_start: Period of analysis start month (default: from info).
            pa_long: Period of analysis length in months (default: from info).

        Returns:
            DataFrame with one row per consecutive pair of years and
            columns for absolute change, percent change, slope, and
            percent slope for both ``FNConc`` and ``FNFlux``.
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before table_change()')

        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)

        annual = _setup_years(self.daily, pa_start=pa_start, pa_long=pa_long)
        return _table_change(annual, year_points, flux_factor=flux_factor)

    def error_stats(self, seed=None):
        """Cross-validation error statistics.

        Requires :meth:`cross_validate` (or :meth:`fit`) to have been
        called first.

        Args:
            seed: Optional integer seed for reproducibility of censored
                observation randomisation.

        Returns:
            Dict with keys ``rsq_log_conc``, ``rsq_log_flux``, ``rmse``,
            ``sep_percent``.
        """
        if 'yHat' not in self.sample.columns:
            raise RuntimeError('cross_validate() must be called before error_stats()')

        return _error_stats(self.sample, seed=seed)

    def flux_bias_stat(self):
        """Flux bias statistic.

        Requires :meth:`cross_validate` (or :meth:`fit`) to have been
        called first so that ``ConcHat`` is available on sample.

        Returns:
            Dict with keys ``bias1``, ``bias2``, ``bias3``.
        """
        if 'ConcHat' not in self.sample.columns:
            raise RuntimeError('cross_validate() must be called before flux_bias_stat()')

        return _flux_bias_stat(self.sample)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_overview(self, fig=None):
        """2x2 overview panel: discharge, concentration vs time/Q, monthly.

        Returns:
            ``matplotlib.figure.Figure``
        """
        from wrtds.plots.data_overview import plot_overview
        return plot_overview(self.daily, self.sample, fig=fig)

    def plot_conc_hist(self, pa_start=None, pa_long=None, ax=None):
        """Annual concentration history (bars + FN line).

        Returns:
            ``matplotlib.figure.Figure``
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before plot_conc_hist()')
        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)
        annual = _setup_years(self.daily, pa_start=pa_start, pa_long=pa_long)
        from wrtds.plots.results import plot_conc_hist
        return plot_conc_hist(annual, ax=ax)

    def plot_flux_hist(self, flux_factor=1.0, pa_start=None, pa_long=None, ax=None):
        """Annual flux history (bars + FN line).

        Returns:
            ``matplotlib.figure.Figure``
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before plot_flux_hist()')
        if pa_start is None:
            pa_start = self.info.get('pa_start', 10)
        if pa_long is None:
            pa_long = self.info.get('pa_long', 12)
        annual = _setup_years(self.daily, pa_start=pa_start, pa_long=pa_long)
        from wrtds.plots.results import plot_flux_hist
        return plot_flux_hist(annual, flux_factor=flux_factor, ax=ax)

    def plot_contours(self, layer=2, ax=None):
        """Filled contour plot of a surface layer.

        Returns:
            ``matplotlib.figure.Figure``
        """
        if self.surfaces is None:
            raise RuntimeError('fit() must be called before plot_contours()')
        from wrtds.plots.results import plot_contours
        return plot_contours(self.surfaces, self.surface_index, layer=layer, ax=ax)

    def plot_conc_pred(self, ax=None):
        """Predicted vs observed concentration scatter.

        Returns:
            ``matplotlib.figure.Figure``
        """
        if 'ConcHat' not in self.sample.columns:
            raise RuntimeError('cross_validate() must be called before plot_conc_pred()')
        from wrtds.plots.diagnostics import plot_conc_pred
        return plot_conc_pred(self.sample, ax=ax)

    def plot_residuals(self, fig=None):
        """Multi-panel diagnostic plots (6 subplots).

        Returns:
            ``matplotlib.figure.Figure``
        """
        if 'yHat' not in self.sample.columns:
            raise RuntimeError('cross_validate() must be called before plot_residuals()')
        from wrtds.plots.diagnostics import flux_bias_multi
        return flux_bias_multi(self.sample, fig=fig)
