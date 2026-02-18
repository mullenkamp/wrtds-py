"""Censored Gaussian MLE regression and tricube weight functions for WRTDS."""

import numpy as np
from scipy import optimize, stats


def tricube(d, h):
    """Tricube kernel weight function.

    Args:
        d: Distance values (scalar or array).
        h: Half-window bandwidth (positive scalar).

    Returns:
        Weights in [0, 1]: ``(1 - |d/h|^3)^3`` when ``|d| < h``, else 0.
    """
    u = np.abs(np.asarray(d, dtype=float) / h)
    return np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)


def compute_weights(
    dec_year,
    logq,
    uncen,
    target_dec_year,
    target_logq,
    window_y,
    window_q,
    window_s,
    min_num_obs=100,
    min_num_uncen=50,
    edge_adjust=True,
    record_start=None,
    record_end=None,
):
    """Compute product of three tricube weights (time, discharge, season).

    Args:
        dec_year: Decimal year for each observation.
        logq: Log-discharge for each observation.
        uncen: Boolean or 0/1 array indicating uncensored observations.
        target_dec_year: Estimation point time coordinate.
        target_logq: Estimation point discharge coordinate.
        window_y: Half-window for time dimension (years).
        window_q: Half-window for discharge dimension (log units).
        window_s: Half-window for season dimension (fraction of year).
        min_num_obs: Minimum total observations with nonzero weight.
        min_num_uncen: Minimum uncensored observations with nonzero weight.
        edge_adjust: If True, expand time window near record edges.
        record_start: Earliest DecYear in record (for edge adjustment).
        record_end: Latest DecYear in record (for edge adjustment).

    Returns:
        Array of combined tricube weights, one per observation.
    """
    dec_year = np.asarray(dec_year, dtype=float)
    logq = np.asarray(logq, dtype=float)
    uncen = np.asarray(uncen, dtype=bool)

    # Default record bounds
    if record_start is None:
        record_start = dec_year.min()
    if record_end is None:
        record_end = dec_year.max()

    # Edge adjustment: expand time window near record boundaries
    wy = window_y
    if edge_adjust:
        dist_to_start = target_dec_year - record_start
        dist_to_end = record_end - target_dec_year
        dist_to_edge = min(dist_to_start, dist_to_end)
        if 0 <= dist_to_edge < wy:
            wy = 2 * window_y - dist_to_edge

    wq = window_q
    ws = window_s

    # Iteratively expand windows until minimum sample requirements are met
    for _ in range(100):
        w_time = tricube(dec_year - target_dec_year, wy)
        w_q = tricube(logq - target_logq, wq)

        # Circular season distance: shortest path around the annual cycle
        diff = dec_year - target_dec_year
        season_dist = np.abs(diff - np.round(diff))
        w_season = tricube(season_dist, ws)

        w = w_time * w_q * w_season

        nonzero = w > 0
        if np.sum(nonzero) >= min_num_obs and np.sum(nonzero & uncen) >= min_num_uncen:
            return w

        # Expand all three windows by 10%
        wy *= 1.1
        wq *= 1.1
        ws *= 1.1

    return w


def neg_log_likelihood_with_grad(theta, X, y_uncen, y_cen_high, uncen_mask, weights):
    """Negative log-likelihood and analytical gradient for weighted censored Gaussian.

    The function returns ``(cost, grad)`` in a single call so that
    ``scipy.optimize.minimize(..., jac=True)`` can use the exact gradient,
    avoiding ~6 finite-difference evaluations per iteration.

    Args:
        theta: Parameter vector ``[B0, B1, B2, B3, B4, log_sigma]``.
        X: Design matrix ``(n, 5)`` — ``[1, DecYear, LogQ, SinDY, CosDY]``.
        y_uncen: ``log(ConcAve)`` for every observation (only indexed where
            ``uncen_mask`` is True; values at censored positions are ignored).
        y_cen_high: ``log(ConcHigh)`` for every observation.
        uncen_mask: Boolean array, True where observation is uncensored.
        weights: Tricube weights ``(n,)``.

    Returns:
        Tuple ``(cost, grad)`` where *cost* is a scalar NLL and *grad* is the
        ``(6,)`` gradient vector.
    """
    beta = theta[:-1]
    log_sigma = theta[-1]
    sigma = np.exp(log_sigma)

    cost = 0.0
    grad = np.zeros_like(theta)

    # --- Uncensored observations ---
    if uncen_mask.any():
        X_u = X[uncen_mask]
        w_u = weights[uncen_mask]
        resid_u = y_uncen[uncen_mask] - X_u @ beta
        z_u = resid_u / sigma

        cost += np.sum(w_u * (0.5 * z_u**2 + log_sigma + 0.5 * np.log(2 * np.pi)))

        grad[:-1] += np.sum(-w_u[:, None] * (resid_u[:, None] / sigma**2) * X_u, axis=0)
        grad[-1] += np.sum(w_u * (1.0 - z_u**2))

    # --- Censored observations ---
    cen_mask = ~uncen_mask
    if cen_mask.any():
        X_c = X[cen_mask]
        w_c = weights[cen_mask]
        z_c = (y_cen_high[cen_mask] - X_c @ beta) / sigma

        log_cdf_c = stats.norm.logcdf(z_c)
        cost += np.sum(w_c * (-log_cdf_c))

        # Inverse Mills ratio via log-space for numerical stability
        log_pdf_c = stats.norm.logpdf(z_c)
        mills = np.exp(log_pdf_c - log_cdf_c)

        grad[:-1] += np.sum(w_c[:, None] * mills[:, None] * (X_c / sigma), axis=0)
        grad[-1] += np.sum(w_c * mills * z_c)

    return cost, grad


def run_surv_reg(sample_data, weights, max_retries=3):
    """Run one weighted censored Gaussian regression.

    Replaces R's ``survival::survreg(..., dist="gaussian")``.

    Args:
        sample_data: Dict of arrays with keys ``DecYear``, ``LogQ``,
            ``SinDY``, ``CosDY``, ``ConcLow``, ``ConcHigh``, ``Uncen``.
        weights: Tricube weight for each observation.
        max_retries: Number of jitter-and-retry attempts on convergence failure.

    Returns:
        Tuple ``(beta, sigma)`` where *beta* is ``(5,)`` coefficient vector
        and *sigma* is the positive scale parameter.
    """
    dec_year = np.asarray(sample_data['DecYear'], dtype=float)
    logq = np.asarray(sample_data['LogQ'], dtype=float)
    sin_dy = np.asarray(sample_data['SinDY'], dtype=float)
    cos_dy = np.asarray(sample_data['CosDY'], dtype=float)
    conc_low = np.asarray(sample_data['ConcLow'], dtype=float)
    conc_high = np.asarray(sample_data['ConcHigh'], dtype=float)
    uncen = np.asarray(sample_data['Uncen'], dtype=bool)
    weights = np.asarray(weights, dtype=float)

    n = len(dec_year)
    X = np.column_stack([np.ones(n), dec_year, logq, sin_dy, cos_dy])

    conc_ave = np.where(np.isnan(conc_low), conc_high / 2, (conc_low + conc_high) / 2)
    y_uncen = np.log(conc_ave)
    y_cen_high = np.log(conc_high)
    uncen_mask = uncen.astype(bool)

    # --- Starting values: weighted OLS on uncensored observations ---
    beta_init, sigma_init = _ols_start(X, y_uncen, uncen_mask, weights)
    theta0 = np.concatenate([beta_init, [np.log(sigma_init)]])

    # --- Optimize with jitter retries ---
    for attempt in range(max_retries + 1):
        result = optimize.minimize(
            neg_log_likelihood_with_grad,
            theta0,
            args=(X, y_uncen, y_cen_high, uncen_mask, weights),
            method='L-BFGS-B',
            jac=True,
        )
        if result.success:
            break
        if attempt < max_retries:
            conc_low, conc_high = jitter_sample(conc_low, conc_high)
            conc_ave = np.where(np.isnan(conc_low), conc_high / 2, (conc_low + conc_high) / 2)
            y_uncen = np.log(conc_ave)
            y_cen_high = np.log(conc_high)

    beta = result.x[:-1]
    sigma = np.exp(result.x[-1])
    return beta, sigma


def predict(beta, sigma, dec_year, logq):
    """Predict log-concentration, SE, and bias-corrected concentration.

    Args:
        beta: Regression coefficients ``(5,)``.
        sigma: Scale parameter (positive scalar).
        dec_year: Decimal year(s) at which to predict.
        logq: Log-discharge(s) at which to predict.

    Returns:
        Tuple ``(yHat, SE, ConcHat)`` where each has the same shape as the
        inputs.  ``ConcHat = exp(yHat + SE² / 2)`` applies the lognormal
        bias correction.
    """
    dec_year = np.asarray(dec_year, dtype=float)
    logq = np.asarray(logq, dtype=float)

    sin_dy = np.sin(2.0 * np.pi * dec_year)
    cos_dy = np.cos(2.0 * np.pi * dec_year)

    yHat = beta[0] + beta[1] * dec_year + beta[2] * logq + beta[3] * sin_dy + beta[4] * cos_dy
    SE = np.full_like(yHat, sigma, dtype=float)
    ConcHat = np.exp(yHat + SE**2 / 2.0)

    return yHat, SE, ConcHat


def jitter_sample(conc_low, conc_high, scale=0.01, rng=None):
    """Add small multiplicative jitter to concentration bounds.

    Matches R's ``jitterSam`` — used when MLE fails to converge.

    Args:
        conc_low: Lower concentration bounds (NaN for left-censored).
        conc_high: Upper concentration bounds.
        scale: Standard deviation of the log-normal jitter.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Tuple ``(conc_low_j, conc_high_j)`` of jittered arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    conc_low = np.asarray(conc_low, dtype=float)
    conc_high = np.asarray(conc_high, dtype=float)

    jitter = np.exp(rng.normal(0, scale, size=len(conc_high)))
    conc_high_j = conc_high * jitter

    conc_low_j = conc_low.copy()
    valid = ~np.isnan(conc_low)
    conc_low_j[valid] = conc_low[valid] * jitter[valid]

    return conc_low_j, conc_high_j


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols_start(X, y_uncen, uncen_mask, weights):
    """Compute weighted OLS starting values for the MLE.

    Falls back to unweighted OLS or default values if there is insufficient
    uncensored data.

    Returns:
        Tuple ``(beta_init, sigma_init)``.
    """
    X_u = X[uncen_mask]
    y_u = y_uncen[uncen_mask]
    w_u = weights[uncen_mask]

    if len(y_u) >= 5 and np.sum(w_u > 0) >= 5:
        # Weighted least squares via transformed problem
        sw = np.sqrt(w_u)
        Xw = X_u * sw[:, None]
        yw = y_u * sw
        try:
            beta_init = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            resid = y_u - X_u @ beta_init
            sigma_init = np.sqrt(np.sum(w_u * resid**2) / max(np.sum(w_u), 1e-10))
        except np.linalg.LinAlgError:
            beta_init = np.zeros(5)
            sigma_init = 1.0
    elif len(y_u) >= 5:
        # Unweighted OLS fallback
        try:
            beta_init = np.linalg.lstsq(X_u, y_u, rcond=None)[0]
            resid = y_u - X_u @ beta_init
            sigma_init = np.std(resid) if len(resid) > 1 else 1.0
        except np.linalg.LinAlgError:
            beta_init = np.zeros(5)
            sigma_init = 1.0
    else:
        beta_init = np.zeros(5)
        sigma_init = 1.0

    sigma_init = max(sigma_init, 0.01)
    return beta_init, sigma_init
