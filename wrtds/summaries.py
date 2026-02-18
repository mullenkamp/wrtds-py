"""Summary tables and error statistics for WRTDS results.

Implements ``setupYears``, ``tableChange``, ``errorStats``, and
``fluxBiasStat`` from the R EGRET package.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# setup_years — annual results table
# ---------------------------------------------------------------------------

def setup_years(daily, pa_start=10, pa_long=12):
    """Create annual results DataFrame.

    Aggregates daily results into annual (or sub-annual period) summary
    values.  The period of analysis is defined by ``pa_start`` (starting
    month) and ``pa_long`` (number of months).

    Years where more than 10 % of days have missing ``ConcDay`` are
    excluded.

    Args:
        daily: Populated daily DataFrame with at least ``Date``, ``Q``,
            ``DecYear``, ``MonthSeq``.  After :meth:`~wrtds.core.WRTDS.fit`,
            also has ``ConcDay``, ``FluxDay``, ``FNConc``, ``FNFlux``.
        pa_start: Starting month of the analysis period (1-12).
            Default 10 (October = water year).
        pa_long: Length of the analysis period in months (1-12).
            Default 12 (full year).

    Returns:
        DataFrame with columns: ``DecYear``, ``Q``, ``Conc``, ``Flux``,
        ``FNConc``, ``FNFlux`` (and ``GenConc``, ``GenFlux`` if present).
        Flux values are rates in kg/day, not annual totals.
    """
    month_seq = daily['MonthSeq'].values
    first_ms = int(month_seq.min())
    last_ms = int(month_seq.max())

    # Generate all possible period starts (every 12 months at pa_start)
    # MonthSeq = (year - 1850) * 12 + month
    # Find the first occurrence of pa_start in the data range
    first_year_1850 = (first_ms - 1) // 12  # zero-indexed year since 1850
    first_pa_ms = first_year_1850 * 12 + pa_start
    if first_pa_ms < first_ms:
        first_pa_ms += 12

    starts = np.arange(first_pa_ms, last_ms + 1, 12)

    has_conc = 'ConcDay' in daily.columns
    has_gen = 'GenConc' in daily.columns and 'GenFlux' in daily.columns

    rows = []
    for s in starts:
        e = s + pa_long - 1
        if e > last_ms:
            break

        mask = (month_seq >= s) & (month_seq <= e)
        chunk = daily.loc[mask]
        if len(chunk) == 0:
            continue

        dec_year = float(chunk['DecYear'].mean())
        q_mean = float(chunk['Q'].mean())

        row = {'DecYear': dec_year, 'Q': q_mean}

        if has_conc:
            conc_vals = chunk['ConcDay'].values
            n_valid = int(np.sum(np.isfinite(conc_vals)))
            n_total = len(conc_vals)

            # Skip years with > 10% missing ConcDay
            if n_valid / n_total < 0.9:
                continue

            row['Conc'] = float(np.nanmean(chunk['ConcDay'].values))
            row['Flux'] = float(np.nanmean(chunk['FluxDay'].values))
            row['FNConc'] = float(np.nanmean(chunk['FNConc'].values))
            row['FNFlux'] = float(np.nanmean(chunk['FNFlux'].values))

            if has_gen:
                row['GenConc'] = float(np.nanmean(chunk['GenConc'].values))
                row['GenFlux'] = float(np.nanmean(chunk['GenFlux'].values))

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# table_change — changes between year points
# ---------------------------------------------------------------------------

def table_change(annual_results, year_points, flux_factor=0.00036525):
    """Compute changes in flow-normalised values between specified years.

    For each consecutive pair of years in *year_points*, compute absolute
    change, percent change, slope, and percent slope for both ``FNConc``
    and ``FNFlux``.

    Args:
        annual_results: DataFrame from :func:`setup_years` (must have
            ``DecYear``, ``FNConc``, ``FNFlux``).
        year_points: List of years (as integers or floats) at which to
            evaluate changes.  Must be in chronological order.
        flux_factor: Conversion factor from kg/day to desired flux units.
            Default ``0.00036525`` converts kg/day to 10^6 kg/year.

    Returns:
        DataFrame with one row per consecutive pair and columns:
        ``Year1``, ``Year2``, ``FNConc_change``, ``FNConc_pct_change``,
        ``FNConc_slope``, ``FNConc_pct_slope``, ``FNFlux_change``,
        ``FNFlux_pct_change``, ``FNFlux_slope``, ``FNFlux_pct_slope``.
        Flux values are in the units determined by *flux_factor*.
    """
    dec_years = annual_results['DecYear'].values

    rows = []
    for i in range(len(year_points) - 1):
        y1 = year_points[i]
        y2 = year_points[i + 1]

        idx1 = int(np.argmin(np.abs(dec_years - y1)))
        idx2 = int(np.argmin(np.abs(dec_years - y2)))

        fnc1 = annual_results.iloc[idx1]['FNConc']
        fnc2 = annual_results.iloc[idx2]['FNConc']
        fnf1 = annual_results.iloc[idx1]['FNFlux'] * flux_factor
        fnf2 = annual_results.iloc[idx2]['FNFlux'] * flux_factor

        actual_y1 = dec_years[idx1]
        actual_y2 = dec_years[idx2]
        dt = actual_y2 - actual_y1

        c_change = fnc2 - fnc1
        f_change = fnf2 - fnf1

        row = {
            'Year1': actual_y1,
            'Year2': actual_y2,
            'FNConc_change': c_change,
            'FNConc_pct_change': 100 * c_change / fnc1 if fnc1 != 0 else np.nan,
            'FNConc_slope': c_change / dt if dt != 0 else np.nan,
            'FNConc_pct_slope': 100 * c_change / fnc1 / dt if (fnc1 != 0 and dt != 0) else np.nan,
            'FNFlux_change': f_change,
            'FNFlux_pct_change': 100 * f_change / fnf1 if fnf1 != 0 else np.nan,
            'FNFlux_slope': f_change / dt if dt != 0 else np.nan,
            'FNFlux_pct_slope': 100 * f_change / fnf1 / dt if (fnf1 != 0 and dt != 0) else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# error_stats — cross-validation error statistics
# ---------------------------------------------------------------------------

def error_stats(sample, seed=None):
    """Cross-validation error statistics.

    Computes R-squared for log-concentration and log-flux, RMSE in log
    units, and standard error of prediction in percent.  Matches R
    EGRET's ``errorStats`` function.

    For censored observations, a random draw from the truncated
    log-normal distribution is used (matching R's ``makeAugmentedSample``).

    Args:
        sample: Sample DataFrame with ``yHat``, ``SE``, ``ConcHigh``,
            ``ConcLow``, ``Uncen``, ``Q`` columns (from cross-validation).
        seed: Optional integer seed for reproducibility of censored
            observation randomisation.

    Returns:
        Dict with keys ``rsq_log_conc``, ``rsq_log_flux``, ``rmse``,
        ``sep_percent``.
    """
    rng = np.random.default_rng(seed)

    yhat = sample['yHat'].values.astype(float)
    se = sample['SE'].values.astype(float)
    conc_high = sample['ConcHigh'].values.astype(float)
    conc_low = sample['ConcLow'].values.astype(float)
    uncen = sample['Uncen'].values.astype(bool)
    q = sample['Q'].values.astype(float)

    n = len(sample)
    r_resid = np.empty(n, dtype=float)
    r_observed = np.empty(n, dtype=float)

    # Uncensored: residual = log(ConcHigh) - yHat
    unc_idx = np.where(uncen)[0]
    r_resid[unc_idx] = np.log(conc_high[unc_idx]) - yhat[unc_idx]
    r_observed[unc_idx] = conc_high[unc_idx]

    # Censored: draw from truncated normal
    cen_idx = np.where(~uncen)[0]
    if len(cen_idx) > 0:
        mu = yhat[cen_idx]
        sigma = se[cen_idx]
        b = (np.log(conc_high[cen_idx]) - mu) / sigma

        # Lower bound: log(ConcLow) if available, else -inf
        a = np.full(len(cen_idx), -np.inf)
        finite_low = np.isfinite(conc_low[cen_idx]) & (conc_low[cen_idx] > 0)
        if finite_low.any():
            a[finite_low] = (np.log(conc_low[cen_idx][finite_low]) - mu[finite_low]) / sigma[finite_low]

        draws = stats.truncnorm.rvs(a=a, b=b, loc=mu, scale=sigma, random_state=rng)
        r_resid[cen_idx] = draws - mu
        r_observed[cen_idx] = np.exp(draws)

    # Compute statistics in log space
    log_obs = np.log(r_observed)
    var_resid = float(np.var(r_resid, ddof=0))
    var_log_conc = float(np.var(log_obs, ddof=0))

    true_flux = r_observed * q * 86.4
    log_flux = np.log(true_flux)
    var_log_flux = float(np.var(log_flux, ddof=0))

    rsq_log_conc = (var_log_conc - var_resid) / var_log_conc if var_log_conc > 0 else 0.0
    rsq_log_flux = (var_log_flux - var_resid) / var_log_flux if var_log_flux > 0 else 0.0
    rmse = float(np.sqrt(var_resid))
    sep_percent = float(100 * np.sqrt(np.exp(var_resid) - 1))

    return {
        'rsq_log_conc': rsq_log_conc,
        'rsq_log_flux': rsq_log_flux,
        'rmse': rmse,
        'sep_percent': sep_percent,
    }


# ---------------------------------------------------------------------------
# flux_bias_stat — flux prediction bias
# ---------------------------------------------------------------------------

def flux_bias_stat(sample):
    """Flux bias statistic.

    Measures systematic bias in the model's flux predictions at sampled
    days.  Matches R EGRET's ``fluxBiasStat``.

    Three variants are computed to handle censored data:

    - ``bias1``: uses ``ConcHigh`` as the observed value (conservative for
      censored data, since the detection limit is an upper bound).
    - ``bias2``: uses ``ConcLow`` as the observed value (uses 0 for
      left-censored where ``ConcLow`` is ``NaN``).
    - ``bias3``: average of ``bias1`` and ``bias2`` (recommended).

    The formula is ``(estimated - observed) / estimated``, where both
    estimated and observed are flux-weighted sums (concentration * Q).
    Positive values indicate overestimation; negative values indicate
    underestimation.  Values near zero indicate good performance.

    Args:
        sample: Sample DataFrame with ``ConcLow``, ``ConcHigh``,
            ``ConcHat``, ``Q`` columns.

    Returns:
        Dict with keys ``bias1``, ``bias2``, ``bias3``.
    """
    q = sample['Q'].values.astype(float)
    conc_hat = sample['ConcHat'].values.astype(float)
    conc_high = sample['ConcHigh'].values.astype(float)
    conc_low = sample['ConcLow'].values.copy().astype(float)

    # ConcLow is NaN for left-censored; treat as 0
    conc_low = np.nan_to_num(conc_low, nan=0.0)

    sum_est = float(np.nansum(conc_hat * q))
    sum_high = float(np.nansum(conc_high * q))
    sum_low = float(np.nansum(conc_low * q))

    if sum_est == 0:
        return {'bias1': np.nan, 'bias2': np.nan, 'bias3': np.nan}

    bias1 = (sum_est - sum_high) / sum_est
    bias2 = (sum_est - sum_low) / sum_est
    bias3 = (bias1 + bias2) / 2

    return {'bias1': float(bias1), 'bias2': float(bias2), 'bias3': float(bias3)}
