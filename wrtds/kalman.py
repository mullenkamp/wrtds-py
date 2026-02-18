"""WRTDS-K: Kalman-filter–style AR(1) residual interpolation."""

import numpy as np
from scipy import stats

from wrtds.surfaces import interpolate_surface


def make_augmented_sample(sample, rng=None):
    """Generate a concentration value for every sample observation.

    Uncensored observations use ``ConcAve`` directly.  For censored
    observations a random draw is taken from the truncated log-normal
    implied by the cross-validation fit (upper bound = ``ConcHigh``).

    Args:
        sample: Populated sample DataFrame with ``Uncen``, ``ConcAve``,
            ``ConcHigh``, ``yHat``, ``SE``.
        rng: Optional :class:`numpy.random.Generator`.

    Returns:
        1-D array of ``rObserved`` concentrations (one per sample row).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(sample)
    r_observed = sample['ConcAve'].values.copy().astype(float)
    uncen = sample['Uncen'].values.astype(bool)
    yHat = sample['yHat'].values
    SE = sample['SE'].values
    conc_high = sample['ConcHigh'].values

    cen_idx = np.where(~uncen)[0]
    if len(cen_idx) == 0:
        return r_observed

    # For censored obs draw from N(yHat, SE²) truncated above at ln(ConcHigh)
    mu = yHat[cen_idx]
    sigma = SE[cen_idx]
    b = (np.log(conc_high[cen_idx]) - mu) / sigma  # upper bound in standard units
    # scipy truncnorm: a=-inf (no lower bound), b=b
    draws = stats.truncnorm.rvs(
        a=-np.inf * np.ones(len(cen_idx)),
        b=b,
        loc=mu,
        scale=sigma,
        random_state=rng,
    )
    r_observed[cen_idx] = np.exp(draws)

    return r_observed


def ar1_conditional_draw(rho, n_gap, e_start, e_end, rng):
    """Draw AR(1) residuals for the interior of a gap, conditioned on endpoints.

    Given standardised residuals at two consecutive sample days separated
    by ``n_gap`` unsampled days, draw plausible residuals for every
    intermediate day using the conditional multivariate normal distribution
    implied by an AR(1) covariance structure.

    Args:
        rho: AR(1) autocorrelation parameter (0 < rho < 1).
        n_gap: Number of interior (unsampled) days in the gap.
        e_start: Standardised residual at the left endpoint.
        e_end: Standardised residual at the right endpoint.
        rng: :class:`numpy.random.Generator`.

    Returns:
        ``(n_gap,)`` array of residuals for the interior days.
        Returns an empty array when ``n_gap == 0``.
    """
    if n_gap == 0:
        return np.empty(0)

    # Full sequence length including both endpoints
    m = n_gap + 2

    # AR(1) correlation matrix: R[i,j] = rho^|i-j|
    idx = np.arange(m)
    R = rho ** np.abs(idx[:, None] - idx[None, :])

    # Partition: observed = {0, m-1}, unobserved = {1, ..., m-2}
    obs = np.array([0, m - 1])
    unobs = np.arange(1, m - 1)

    S11 = R[np.ix_(obs, obs)]
    S12 = R[np.ix_(obs, unobs)]
    S21 = R[np.ix_(unobs, obs)]
    S22 = R[np.ix_(unobs, unobs)]

    x_obs = np.array([e_start, e_end])

    # Conditional distribution: N(mu_cond, Sigma_cond)
    S11_inv = np.linalg.inv(S11)
    mu_cond = S21 @ S11_inv @ x_obs
    Sigma_cond = S22 - S21 @ S11_inv @ S12

    # Symmetrise to avoid tiny floating-point asymmetries before Cholesky
    Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2
    # Small diagonal jitter for numerical stability
    Sigma_cond += np.eye(n_gap) * 1e-12

    L = np.linalg.cholesky(Sigma_cond)
    z = rng.standard_normal(n_gap)

    return mu_cond + L @ z


def wrtds_kalman(
    daily,
    sample,
    surfaces,
    surface_index,
    rho=0.90,
    n_iter=200,
    seed=None,
):
    """Run WRTDS-K (Kalman-style residual interpolation).

    Improves daily flux estimates by exploiting the temporal autocorrelation
    of model residuals between consecutive sample days.

    Args:
        daily: Populated daily DataFrame (must have ``yHat``, ``SE`` from
            :func:`~wrtds.flow_norm.estimate_daily`).
        sample: Populated sample DataFrame (must have ``yHat``, ``SE`` from
            :func:`~wrtds.cross_val.cross_validate`).
        surfaces: 3-D surfaces array.
        surface_index: Grid parameters dict.
        rho: AR(1) autocorrelation parameter.
        n_iter: Number of Monte Carlo iterations.
        seed: Optional integer seed for reproducibility.

    Returns:
        Daily DataFrame with added columns ``GenConc`` and ``GenFlux``.
    """
    rng = np.random.default_rng(seed)
    daily = daily.copy()

    daily_dates = daily['Date'].values
    daily_yhat = daily['yHat'].values.astype(float)
    daily_se = daily['SE'].values.astype(float)
    daily_q = daily['Q'].values.astype(float)
    n_daily = len(daily)

    # Map sample dates to daily row indices
    sample_sorted = sample.sort_values('Date').reset_index(drop=True)
    sample_dates = sample_sorted['Date'].values
    # Use searchsorted for fast date matching
    daily_date_index = {d: i for i, d in enumerate(daily_dates)}
    sample_daily_idx = np.array([daily_date_index.get(d, -1) for d in sample_dates])
    valid_sample = sample_daily_idx >= 0
    sample_sorted = sample_sorted.loc[valid_sample].reset_index(drop=True)
    sample_daily_idx = sample_daily_idx[valid_sample]
    n_sample = len(sample_sorted)

    flux_accum = np.zeros(n_daily)

    for _ in range(n_iter):
        # 1. Draw concentrations for censored observations
        r_observed = make_augmented_sample(sample_sorted, rng=rng)

        # 2. Standardised residuals at sample days
        s_yhat = sample_sorted['yHat'].values
        s_se = sample_sorted['SE'].values
        e_sample = (np.log(r_observed) - s_yhat) / s_se

        # 3. Interpolate residuals to all daily rows
        e_daily = np.zeros(n_daily)

        # Place sample residuals at their daily positions
        e_daily[sample_daily_idx] = e_sample

        # Fill gaps between consecutive samples via AR(1) conditional draws
        for k in range(n_sample - 1):
            i_start = sample_daily_idx[k]
            i_end = sample_daily_idx[k + 1]
            n_gap = i_end - i_start - 1
            if n_gap > 0:
                gap_resid = ar1_conditional_draw(
                    rho, n_gap, e_sample[k], e_sample[k + 1], rng,
                )
                e_daily[i_start + 1: i_end] = gap_resid

        # Days before the first sample: unconditional AR(1) decay
        first_idx = sample_daily_idx[0]
        for j in range(first_idx - 1, -1, -1):
            e_daily[j] = rho * e_daily[j + 1] + np.sqrt(1 - rho**2) * rng.standard_normal()

        # Days after the last sample: unconditional AR(1) decay
        last_idx = sample_daily_idx[-1]
        for j in range(last_idx + 1, n_daily):
            e_daily[j] = rho * e_daily[j - 1] + np.sqrt(1 - rho**2) * rng.standard_normal()

        # 4. Convert residuals back to concentrations and fluxes
        conc_iter = np.exp(e_daily * daily_se + daily_yhat)
        flux_iter = conc_iter * daily_q * 86.4

        # 5. Accumulate
        flux_accum += flux_iter

    # Final averages
    daily['GenFlux'] = flux_accum / n_iter
    daily['GenConc'] = daily['GenFlux'] / (daily_q * 86.4)

    return daily
