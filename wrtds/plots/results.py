"""Result and trend plots — annual histories, contour surfaces, C-Q curves."""

import numpy as np

from wrtds.plots.utils import _get_ax


def plot_conc_hist(annual_results, ax=None):
    """Annual concentration history: bars for Conc, line for FNConc.

    If ``GenConc`` is present, an additional line is plotted.

    Args:
        annual_results: DataFrame from ``setup_years`` with ``DecYear``,
            ``Conc``, ``FNConc``, and optionally ``GenConc``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    ax.bar(annual_results['DecYear'], annual_results['Conc'],
           width=0.8, color='C0', alpha=0.5, label='Conc')
    ax.plot(annual_results['DecYear'], annual_results['FNConc'],
            color='C3', linewidth=2, label='FN Conc')
    if 'GenConc' in annual_results.columns:
        ax.plot(annual_results['DecYear'], annual_results['GenConc'],
                color='C2', linewidth=2, linestyle='--', label='Gen Conc')
    ax.set_xlabel('Year')
    ax.set_ylabel('Concentration')
    ax.set_title('Annual Concentration')
    ax.legend(fontsize='small')
    return fig


def plot_flux_hist(annual_results, flux_factor=1.0, ax=None):
    """Annual flux history: bars for Flux, line for FNFlux.

    If ``GenFlux`` is present, an additional line is plotted.

    Args:
        annual_results: DataFrame from ``setup_years`` with ``DecYear``,
            ``Flux``, ``FNFlux``, and optionally ``GenFlux``.
        flux_factor: Multiplier to convert flux units (default 1.0 = kg/day).
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    ax.bar(annual_results['DecYear'], annual_results['Flux'] * flux_factor,
           width=0.8, color='C0', alpha=0.5, label='Flux')
    ax.plot(annual_results['DecYear'], annual_results['FNFlux'] * flux_factor,
            color='C3', linewidth=2, label='FN Flux')
    if 'GenFlux' in annual_results.columns:
        ax.plot(annual_results['DecYear'], annual_results['GenFlux'] * flux_factor,
                color='C2', linewidth=2, linestyle='--', label='Gen Flux')
    ax.set_xlabel('Year')
    ax.set_ylabel('Flux')
    ax.set_title('Annual Flux')
    ax.legend(fontsize='small')
    return fig


def plot_contours(surfaces, surface_index, layer=2, ax=None):
    """Filled contour plot of a surface layer.

    Args:
        surfaces: 3-D array ``(n_logq, n_year, 3)``.
        surface_index: Dict with ``year_grid`` and ``logq_grid``.
        layer: Surface layer index (default 2 = ConcHat).
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    year_grid = surface_index['year_grid']
    logq_grid = surface_index['logq_grid']
    z = surfaces[:, :, layer]
    cs = ax.contourf(year_grid, logq_grid, z, levels=20, cmap='viridis')
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Log(Q)')
    ax.set_title('Concentration Surface')
    return fig


def plot_conc_q_smooth(surfaces, surface_index, years, layer=2, ax=None):
    """Concentration-discharge curves at selected years.

    Plots vertical slices through the surface at the nearest year indices.

    Args:
        surfaces: 3-D array ``(n_logq, n_year, 3)``.
        surface_index: Dict with ``year_grid`` and ``logq_grid``.
        years: Sequence of years at which to plot C-Q curves.
        layer: Surface layer index (default 2 = ConcHat).
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    year_grid = surface_index['year_grid']
    logq_grid = surface_index['logq_grid']
    for yr in years:
        idx = int(np.argmin(np.abs(year_grid - yr)))
        conc = surfaces[:, idx, layer]
        ax.plot(logq_grid, conc, label=f'{yr:.0f}')
    ax.set_xlabel('Log(Q)')
    ax.set_ylabel('Concentration')
    ax.set_title('C-Q Curves by Year')
    ax.legend(fontsize='small')
    return fig


def plot_conc_time_smooth(surfaces, surface_index, logq_values, layer=2, ax=None):
    """Concentration-time curves at selected discharges.

    Plots horizontal slices through the surface at the nearest LogQ indices.

    Args:
        surfaces: 3-D array ``(n_logq, n_year, 3)``.
        surface_index: Dict with ``year_grid`` and ``logq_grid``.
        logq_values: Sequence of LogQ values at which to plot time curves.
        layer: Surface layer index (default 2 = ConcHat).
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    year_grid = surface_index['year_grid']
    logq_grid = surface_index['logq_grid']
    for lq in logq_values:
        idx = int(np.argmin(np.abs(logq_grid - lq)))
        conc = surfaces[idx, :, layer]
        ax.plot(year_grid, conc, label=f'LogQ={lq:.2f}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Concentration')
    ax.set_title('Concentration vs Time at Fixed Discharge')
    ax.legend(fontsize='small')
    return fig


def plot_diff_contours(surfaces1, surfaces2, surface_index, layer=2, ax=None):
    """Difference contour plot between two surfaces.

    Args:
        surfaces1: First 3-D surface array.
        surfaces2: Second 3-D surface array.
        surface_index: Dict with ``year_grid`` and ``logq_grid``.
        layer: Surface layer index (default 2 = ConcHat).
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    year_grid = surface_index['year_grid']
    logq_grid = surface_index['logq_grid']
    diff = surfaces2[:, :, layer] - surfaces1[:, :, layer]
    vmax = max(abs(diff.min()), abs(diff.max()))
    cs = ax.contourf(year_grid, logq_grid, diff, levels=20,
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Log(Q)')
    ax.set_title('Surface Difference')
    return fig
