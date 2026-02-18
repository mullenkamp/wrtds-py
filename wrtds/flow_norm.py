"""Daily estimation and flow normalization for WRTDS."""

import numpy as np
import pandas as pd

from wrtds.surfaces import interpolate_surface, make_interpolator


def estimate_daily(daily, surfaces, surface_index):
    """Interpolate concentration and flux from the surfaces grid for each day.

    Args:
        daily: Populated daily DataFrame (must have ``LogQ``, ``DecYear``, ``Q``).
        surfaces: 3-D surfaces array ``(n_logq, n_year, 3)``.
        surface_index: Grid parameters from :func:`~wrtds.surfaces.compute_surface_index`.

    Returns:
        Daily DataFrame with added columns: ``yHat``, ``SE``, ``ConcDay``, ``FluxDay``.
    """
    daily = daily.copy()
    logq = daily['LogQ'].values
    dec_year = daily['DecYear'].values

    daily['yHat'] = interpolate_surface(surfaces, surface_index, logq, dec_year, layer=0)
    daily['SE'] = interpolate_surface(surfaces, surface_index, logq, dec_year, layer=1)
    daily['ConcDay'] = interpolate_surface(surfaces, surface_index, logq, dec_year, layer=2)
    daily['FluxDay'] = daily['ConcDay'] * daily['Q'] * 86.4

    return daily


def bin_qs(daily):
    """Group historical log-discharge values by day-of-year.

    Leap-day handling: Feb 29 (day 60 in leap years) is merged into
    Feb 28 (day 59) so that every bin key is in 1..365.

    Args:
        daily: Populated daily DataFrame (must have ``Day``, ``LogQ``).

    Returns:
        Dict mapping ``{day_of_year: np.array of LogQ values}``.
    """
    day = daily['Day'].values.copy()
    logq = daily['LogQ'].values

    # In leap years, day-of-year 60 is Feb 29.  Merge into day 59 (Feb 28)
    # and shift all subsequent leap-year days down by 1 so that bins align
    # across leap and non-leap years.
    is_leap = daily['Date'].dt.is_leap_year.values
    # Feb 29 in a leap year -> map to day 59
    day[(day == 60) & is_leap] = 59
    # Days after Feb 29 in leap years: shift down by 1 so Mar 1 = day 60
    day[(day > 60) & is_leap] -= 1

    bins = {}
    for d in range(1, 366):
        mask = day == d
        if mask.any():
            bins[d] = logq[mask]

    return bins


def flow_normalize(daily, surfaces, surface_index, q_bins):
    """Compute flow-normalised concentration and flux for each day.

    For every day *t*, the model's predicted concentration is averaged
    across the full historical discharge distribution for that calendar
    day, removing the effect of year-to-year flow variability.

    Uses a single vectorised interpolation call for performance.

    Args:
        daily: Populated daily DataFrame (must have ``Day``, ``DecYear``;
            ``Day`` values use the same leap-adjusted scheme as :func:`bin_qs`).
        surfaces: 3-D surfaces array.
        surface_index: Grid parameters dict.
        q_bins: Output of :func:`bin_qs`.

    Returns:
        Daily DataFrame with added columns: ``FNConc``, ``FNFlux``.
    """
    daily = daily.copy()

    day = daily['Day'].values.copy()
    dec_year = daily['DecYear'].values
    is_leap = daily['Date'].dt.is_leap_year.values
    day[(day == 60) & is_leap] = 59
    day[(day > 60) & is_leap] -= 1

    n_days = len(daily)

    # Build a ConcHat interpolator once (layer 2)
    interp_conc = make_interpolator(surfaces, surface_index, layer=2)
    logq_lo = surface_index['logq_grid'][0]
    logq_hi = surface_index['logq_grid'][-1]
    year_lo = surface_index['year_grid'][0]
    year_hi = surface_index['year_grid'][-1]

    # Pre-build all (logq_j, decyear_t) pairs and a mapping back to each day
    all_logq = []
    all_year = []
    day_indices = []  # which daily row each pair belongs to

    for i in range(n_days):
        d = day[i]
        hist_logq = q_bins.get(d)
        if hist_logq is None:
            continue
        m = len(hist_logq)
        all_logq.append(hist_logq)
        all_year.append(np.full(m, dec_year[i]))
        day_indices.append(np.full(m, i, dtype=np.intp))

    if len(all_logq) == 0:
        daily['FNConc'] = np.nan
        daily['FNFlux'] = np.nan
        return daily

    all_logq = np.concatenate(all_logq)
    all_year = np.concatenate(all_year)
    day_indices = np.concatenate(day_indices)

    # Clamp and interpolate in one vectorised call
    logq_c = np.clip(all_logq, logq_lo, logq_hi)
    year_c = np.clip(all_year, year_lo, year_hi)
    pts = np.column_stack([logq_c, year_c])
    conc_hat = interp_conc(pts)

    # Flux for each (Q_j, t) pair: ConcHat * Q_j * 86.4
    flux_hat = conc_hat * np.exp(all_logq) * 86.4

    # Reduce: mean per daily row
    fn_conc = np.full(n_days, np.nan)
    fn_flux = np.full(n_days, np.nan)

    # np.add.at + counts is faster than a Python loop for large arrays
    conc_sum = np.zeros(n_days)
    flux_sum = np.zeros(n_days)
    counts = np.zeros(n_days, dtype=np.intp)

    np.add.at(conc_sum, day_indices, conc_hat)
    np.add.at(flux_sum, day_indices, flux_hat)
    np.add.at(counts, day_indices, 1)

    valid = counts > 0
    fn_conc[valid] = conc_sum[valid] / counts[valid]
    fn_flux[valid] = flux_sum[valid] / counts[valid]

    daily['FNConc'] = fn_conc
    daily['FNFlux'] = fn_flux

    return daily
