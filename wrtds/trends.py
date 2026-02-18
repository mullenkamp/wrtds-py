"""Trend analysis: generalized flow normalization and trend decomposition (EGRET 3.0).

Implements ``runPairs``, ``runGroups``, and ``runSeries`` from the R EGRET
package.  These functions decompose water-quality trends into a
**concentration-discharge trend component (CQTC)** — change in the C-Q
relationship itself — and a **discharge trend component (QTC)** — change
driven by shifts in the discharge distribution.

The key idea is *generalized flow normalization*: instead of averaging
across the entire historical discharge distribution (stationary FN), the
discharge distribution is restricted to a sliding window of
``2 * window_side + 1`` years centred on the target year.
"""

import numpy as np
import pandas as pd

from wrtds.data_prep import decimal_date
from wrtds.flow_norm import bin_qs, flow_normalize
from wrtds.surfaces import estimate_surfaces


# Conversion factor: mean daily flux (kg/day) → annual flux (10^6 kg/year)
_FLUX_CONV = 365.25 / 1e6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _water_year_dates(year, pa_start=10, pa_long=12):
    """Return ``(start_date, end_date)`` as Timestamps for a given analysis year.

    For ``pa_start=10, pa_long=12`` (water year) and ``year=2001``:
    ``(Timestamp('2000-10-01'), Timestamp('2001-09-30'))``.
    """
    if pa_start + pa_long > 13:
        start = pd.Timestamp(year=year - 1, month=pa_start, day=1)
    else:
        start = pd.Timestamp(year=year, month=pa_start, day=1)
    end = start + pd.DateOffset(months=pa_long) - pd.Timedelta(days=1)
    return start, end


def _filter_daily_to_year(daily, year, pa_start=10, pa_long=12):
    """Filter daily DataFrame to a single analysis year."""
    start, end = _water_year_dates(year, pa_start, pa_long)
    mask = (daily['Date'] >= start) & (daily['Date'] <= end)
    return daily.loc[mask].copy()


def _compute_surface_index_narrow(sample, year_start_dec, year_end_dec):
    """Compute surface_index with full LogQ range but a restricted year range.

    Used by :func:`run_pairs` to build 1-year surfaces.
    """
    bottom_logq = sample['LogQ'].min() - 0.05
    top_logq = sample['LogQ'].max() + 0.05
    n_logq = 14
    step_logq = (top_logq - bottom_logq) / (n_logq - 1)

    step_year = 1.0 / 16.0
    n_year = round((year_end_dec - year_start_dec) / step_year) + 1

    return {
        'bottom_logq': bottom_logq,
        'top_logq': top_logq,
        'step_logq': step_logq,
        'n_logq': n_logq,
        'bottom_year': year_start_dec,
        'top_year': year_end_dec,
        'step_year': step_year,
        'n_year': n_year,
        'logq_grid': np.linspace(bottom_logq, top_logq, n_logq),
        'year_grid': np.linspace(year_start_dec, year_end_dec, n_year),
    }


def _annual_fn_mean(daily_year, surfaces, surface_index, q_bins):
    """Compute annual mean FNConc and FNFlux for a year-filtered daily DataFrame.

    Returns:
        ``(mean_fn_conc, mean_fn_flux)`` in mg/L and kg/day respectively.
    """
    daily_fn = flow_normalize(daily_year, surfaces, surface_index, q_bins)
    fn_conc = daily_fn['FNConc'].values
    fn_flux = daily_fn['FNFlux'].values
    valid = np.isfinite(fn_conc) & np.isfinite(fn_flux)
    if not valid.any():
        return np.nan, np.nan
    return float(np.mean(fn_conc[valid])), float(np.mean(fn_flux[valid]))


def _build_result_df(c10, f10, c11, f11, c20, f20, c22, f22):
    """Build the standard trend decomposition DataFrame."""
    c_total = c22 - c11
    c_cqtc = c20 - c10
    c_qtc = c_total - c_cqtc

    f_total = f22 - f11
    f_cqtc = f20 - f10
    f_qtc = f_total - f_cqtc

    return pd.DataFrame({
        'TotalChange': [c_total, _FLUX_CONV * f_total],
        'CQTC': [c_cqtc, _FLUX_CONV * f_cqtc],
        'QTC': [c_qtc, _FLUX_CONV * f_qtc],
        'x10': [c10, _FLUX_CONV * f10],
        'x11': [c11, _FLUX_CONV * f11],
        'x20': [c20, _FLUX_CONV * f20],
        'x22': [c22, _FLUX_CONV * f22],
    }, index=['Conc', 'Flux'])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bin_qs_windowed(daily, center_year, window_side):
    """Group historical log-discharge by day-of-year within a time window.

    Like :func:`~wrtds.flow_norm.bin_qs` but restricted to years within
    ``[center_year - window_side, center_year + window_side]``.

    Args:
        daily: Populated daily DataFrame.
        center_year: Centre year of the window (integer or decimal year).
        window_side: Half-width of the window in years.

    Returns:
        Dict mapping ``{day_of_year: np.array of LogQ values}``.
    """
    dec_year = daily['DecYear'].values
    mask = (dec_year >= center_year - window_side) & (dec_year <= center_year + window_side)
    return bin_qs(daily.loc[mask])


def run_pairs(
    sample,
    daily,
    year1,
    year2,
    window_side=7,
    pa_start=10,
    pa_long=12,
    fit_params=None,
):
    """Compare flow-normalised values between two specific years.

    Estimates separate 1-year surfaces for *year1* and *year2*, then
    decomposes the total change into a **CQTC** (concentration-discharge
    trend component) and a **QTC** (discharge trend component).

    The notation ``xAB`` means:

    - **A** = which year's C-Q surface (1 = year1, 2 = year2)
    - **B** = which flow distribution (0 = stationary / full record,
      1 = year1's window, 2 = year2's window)

    Args:
        sample: Populated sample DataFrame.
        daily: Populated daily DataFrame.
        year1: First comparison year.
        year2: Second comparison year.
        window_side: Half-window for generalized flow normalisation (years).
            Use 0 for stationary flow normalisation only.
        pa_start: Period of analysis start month (10 = water year).
        pa_long: Period of analysis length in months.
        fit_params: Dict of regression parameters
            (``window_y``, ``window_q``, ``window_s``, ``min_num_obs``,
            ``min_num_uncen``, ``edge_adjust``).

    Returns:
        DataFrame with index ``['Conc', 'Flux']`` and columns
        ``['TotalChange', 'CQTC', 'QTC', 'x10', 'x11', 'x20', 'x22']``.
        Concentration in mg/L, flux in 10^6 kg/year.
    """
    if fit_params is None:
        fit_params = {
            'window_y': 7, 'window_q': 2, 'window_s': 0.5,
            'min_num_obs': 100, 'min_num_uncen': 50, 'edge_adjust': True,
        }

    # 1. Water year bounds in decimal years for surface grids
    start1, end1 = _water_year_dates(year1, pa_start, pa_long)
    start2, end2 = _water_year_dates(year2, pa_start, pa_long)

    dec_start1 = decimal_date(pd.Series([start1])).iloc[0]
    dec_end1 = decimal_date(pd.Series([end1])).iloc[0]
    dec_start2 = decimal_date(pd.Series([start2])).iloc[0]
    dec_end2 = decimal_date(pd.Series([end2])).iloc[0]

    # 2. Estimate 1-year surfaces for each year (full sample data, narrow grid)
    si1 = _compute_surface_index_narrow(sample, dec_start1, dec_end1)
    si2 = _compute_surface_index_narrow(sample, dec_start2, dec_end2)

    surfaces1 = estimate_surfaces(sample, si1, **fit_params)
    surfaces2 = estimate_surfaces(sample, si2, **fit_params)

    # 3. Filter daily to each analysis year
    daily_year1 = _filter_daily_to_year(daily, year1, pa_start, pa_long)
    daily_year2 = _filter_daily_to_year(daily, year2, pa_start, pa_long)

    # 4. Q bins: stationary (full record) and windowed
    q_bins_all = bin_qs(daily)

    if window_side > 0:
        q_bins_1 = bin_qs_windowed(daily, year1, window_side)
        q_bins_2 = bin_qs_windowed(daily, year2, window_side)
    else:
        q_bins_1 = q_bins_all
        q_bins_2 = q_bins_all

    # 5. Compute x10, x11, x20, x22
    c10, f10 = _annual_fn_mean(daily_year1, surfaces1, si1, q_bins_all)
    c11, f11 = _annual_fn_mean(daily_year1, surfaces1, si1, q_bins_1)
    c20, f20 = _annual_fn_mean(daily_year2, surfaces2, si2, q_bins_all)
    c22, f22 = _annual_fn_mean(daily_year2, surfaces2, si2, q_bins_2)

    return _build_result_df(c10, f10, c11, f11, c20, f20, c22, f22)


def run_groups(
    daily,
    surfaces,
    surface_index,
    group1_years,
    group2_years,
    window_side=7,
    pa_start=10,
    pa_long=12,
):
    """Compare flow-normalised averages across two groups of years.

    Uses the existing full-period surface and averages annual
    flow-normalised values over each year group.

    Args:
        daily: Populated daily DataFrame.
        surfaces: 3-D surfaces array (from full fit).
        surface_index: Grid parameters dict.
        group1_years: ``(first_year, last_year)`` for group 1.
        group2_years: ``(first_year, last_year)`` for group 2.
        window_side: Half-window for generalized flow normalisation.
        pa_start: Period of analysis start month.
        pa_long: Period of analysis length in months.

    Returns:
        DataFrame with same format as :func:`run_pairs`.
    """
    q_bins_all = bin_qs(daily)

    def _group_means(year_range):
        c_flex, f_flex, c_stat, f_stat = [], [], [], []
        for year in range(year_range[0], year_range[1] + 1):
            daily_year = _filter_daily_to_year(daily, year, pa_start, pa_long)
            if len(daily_year) == 0:
                continue

            # Stationary FN (full Q distribution)
            cs, fs = _annual_fn_mean(daily_year, surfaces, surface_index, q_bins_all)
            c_stat.append(cs)
            f_stat.append(fs)

            # Generalized (windowed) FN
            if window_side > 0:
                q_bins_w = bin_qs_windowed(daily, year, window_side)
            else:
                q_bins_w = q_bins_all
            cg, fg = _annual_fn_mean(daily_year, surfaces, surface_index, q_bins_w)
            c_flex.append(cg)
            f_flex.append(fg)

        return (float(np.mean(c_flex)), float(np.mean(f_flex)),
                float(np.mean(c_stat)), float(np.mean(f_stat)))

    c11, f11, c10, f10 = _group_means(group1_years)
    c22, f22, c20, f20 = _group_means(group2_years)

    return _build_result_df(c10, f10, c11, f11, c20, f20, c22, f22)


def run_series(
    daily,
    surfaces,
    surface_index,
    window_side=7,
    pa_start=10,
    pa_long=12,
):
    """Compute annual time series of generalized flow-normalised values.

    For each year in the record, flow normalisation uses the discharge
    distribution from a sliding window of ``2 * window_side + 1`` years
    centred on the target year.

    Args:
        daily: Populated daily DataFrame (must have ``Day``, ``DecYear``,
            ``Date`` columns).
        surfaces: 3-D surfaces array.
        surface_index: Grid parameters dict.
        window_side: Half-window for generalized flow normalisation.
            Use 0 for standard (stationary) flow normalisation.
        pa_start: Period of analysis start month.
        pa_long: Period of analysis length in months.

    Returns:
        Daily DataFrame with updated ``FNConc`` and ``FNFlux`` columns
        computed using generalized flow normalisation.
    """
    daily = daily.copy()

    # Determine year range from daily data
    dec_year = daily['DecYear'].values
    min_dec = float(dec_year.min())
    max_dec = float(dec_year.max())

    # Find first and last complete analysis years
    first_year = int(np.ceil(min_dec))
    last_year = int(np.floor(max_dec))

    daily['FNConc'] = np.nan
    daily['FNFlux'] = np.nan

    for year in range(first_year, last_year + 1):
        start, end = _water_year_dates(year, pa_start, pa_long)
        year_mask = (daily['Date'] >= start) & (daily['Date'] <= end)
        if not year_mask.any():
            continue

        daily_year = daily.loc[year_mask].copy()

        if window_side > 0:
            q_bins = bin_qs_windowed(daily, year, window_side)
        else:
            q_bins = bin_qs(daily)

        daily_fn = flow_normalize(daily_year, surfaces, surface_index, q_bins)

        daily.loc[year_mask, 'FNConc'] = daily_fn['FNConc'].values
        daily.loc[year_mask, 'FNFlux'] = daily_fn['FNFlux'].values

    return daily
