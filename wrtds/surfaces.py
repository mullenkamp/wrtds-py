"""Surface grid estimation and bilinear interpolation for WRTDS."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from wrtds.regression import compute_weights, predict, run_surv_reg


def compute_surface_index(sample):
    """Compute the surface grid parameters from sample data.

    The grid spans the range of observed log-discharge and decimal year,
    with fixed resolution: 14 log-Q levels and time steps of 1/16 year
    (~23 days).

    Args:
        sample: Populated sample DataFrame (must have ``LogQ`` and ``DecYear``).

    Returns:
        Dict with keys: ``bottom_logq``, ``top_logq``, ``step_logq``,
        ``n_logq``, ``bottom_year``, ``top_year``, ``step_year``,
        ``n_year``, ``logq_grid``, ``year_grid``.
    """
    bottom_logq = sample['LogQ'].min() - 0.05
    top_logq = sample['LogQ'].max() + 0.05
    n_logq = 14
    step_logq = (top_logq - bottom_logq) / (n_logq - 1)

    bottom_year = np.floor(sample['DecYear'].min())
    top_year = np.ceil(sample['DecYear'].max())
    step_year = 1.0 / 16.0
    n_year = round((top_year - bottom_year) / step_year) + 1

    return {
        'bottom_logq': bottom_logq,
        'top_logq': top_logq,
        'step_logq': step_logq,
        'n_logq': n_logq,
        'bottom_year': bottom_year,
        'top_year': top_year,
        'step_year': step_year,
        'n_year': n_year,
        'logq_grid': np.linspace(bottom_logq, top_logq, n_logq),
        'year_grid': np.linspace(bottom_year, top_year, n_year),
    }


def estimate_surfaces(
    sample,
    surface_index,
    window_y=7.0,
    window_q=2.0,
    window_s=0.5,
    min_num_obs=100,
    min_num_uncen=50,
    edge_adjust=True,
):
    """Fit the censored regression at every grid point to build the surfaces array.

    This is the main computational bottleneck of WRTDS: one MLE solve per
    grid point, typically ~7 000 total (14 log-Q levels x ~500 time steps).

    Args:
        sample: Populated sample DataFrame with ``DecYear``, ``LogQ``,
            ``SinDY``, ``CosDY``, ``ConcLow``, ``ConcHigh``, ``Uncen``.
        surface_index: Grid parameters from :func:`compute_surface_index`.
        window_y: Time half-window in years.
        window_q: Discharge half-window in log units.
        window_s: Season half-window in fraction of year.
        min_num_obs: Minimum observations with nonzero weight.
        min_num_uncen: Minimum uncensored observations with nonzero weight.
        edge_adjust: Expand time window near record edges.

    Returns:
        3-D numpy array of shape ``(n_logq, n_year, 3)`` where layer 0 is
        ``yHat``, layer 1 is ``SE``, and layer 2 is ``ConcHat``.
    """
    logq_grid = surface_index['logq_grid']
    year_grid = surface_index['year_grid']
    n_logq = len(logq_grid)
    n_year = len(year_grid)

    # Pre-extract sample arrays for efficiency
    sample_data = {
        'DecYear': sample['DecYear'].values,
        'LogQ': sample['LogQ'].values,
        'SinDY': sample['SinDY'].values,
        'CosDY': sample['CosDY'].values,
        'ConcLow': sample['ConcLow'].values,
        'ConcHigh': sample['ConcHigh'].values,
        'Uncen': sample['Uncen'].values.astype(bool),
    }
    dec_year_arr = sample_data['DecYear']
    logq_arr = sample_data['LogQ']
    uncen_arr = sample_data['Uncen']
    record_start = float(dec_year_arr.min())
    record_end = float(dec_year_arr.max())

    surfaces = np.empty((n_logq, n_year, 3))

    for j in range(n_logq):
        for k in range(n_year):
            target_logq = logq_grid[j]
            target_year = year_grid[k]

            w = compute_weights(
                dec_year_arr, logq_arr, uncen_arr,
                target_year, target_logq,
                window_y, window_q, window_s,
                min_num_obs=min_num_obs,
                min_num_uncen=min_num_uncen,
                edge_adjust=edge_adjust,
                record_start=record_start,
                record_end=record_end,
            )

            beta, sigma = run_surv_reg(sample_data, w)
            yHat, SE, ConcHat = predict(beta, sigma, target_year, target_logq)

            surfaces[j, k, 0] = float(yHat)
            surfaces[j, k, 1] = float(SE)
            surfaces[j, k, 2] = float(ConcHat)

    return surfaces


def make_interpolator(surfaces, surface_index, layer=2):
    """Create a ``RegularGridInterpolator`` for one surface layer.

    Args:
        surfaces: 3-D array ``(n_logq, n_year, 3)``.
        surface_index: Grid parameters dict.
        layer: Which layer to interpolate (0=yHat, 1=SE, 2=ConcHat).

    Returns:
        A :class:`scipy.interpolate.RegularGridInterpolator` instance.
        Out-of-range queries are clamped to the nearest grid boundary.
    """
    logq_grid = surface_index['logq_grid']
    year_grid = surface_index['year_grid']

    return RegularGridInterpolator(
        (logq_grid, year_grid),
        surfaces[:, :, layer],
        method='linear',
        bounds_error=False,
        fill_value=None,  # nearest-neighbour extrapolation
    )


def interpolate_surface(surfaces, surface_index, logq, dec_year, layer=2):
    """Vectorised bilinear interpolation on a surface layer.

    Queries are clamped to the grid boundaries before interpolation.

    Args:
        surfaces: 3-D array ``(n_logq, n_year, 3)``.
        surface_index: Grid parameters dict.
        logq: Log-discharge value(s).
        dec_year: Decimal year value(s).
        layer: Which layer (0=yHat, 1=SE, 2=ConcHat).

    Returns:
        Array of interpolated values (same shape as inputs).
    """
    logq = np.asarray(logq, dtype=float)
    dec_year = np.asarray(dec_year, dtype=float)

    # Clamp to grid boundaries
    logq_c = np.clip(logq, surface_index['logq_grid'][0], surface_index['logq_grid'][-1])
    dec_year_c = np.clip(dec_year, surface_index['year_grid'][0], surface_index['year_grid'][-1])

    interp = make_interpolator(surfaces, surface_index, layer)
    pts = np.column_stack([logq_c.ravel(), dec_year_c.ravel()])
    result = interp(pts)

    return result.reshape(logq.shape) if logq.ndim > 0 else result.item()
