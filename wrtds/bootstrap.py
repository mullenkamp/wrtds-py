"""Block bootstrap confidence intervals for WRTDS trend analysis.

Implements the bootstrap uncertainty estimation from the R EGRETci
package.  The key idea is to block-resample the water-quality sample
DataFrame (preserving temporal autocorrelation), re-fit WRTDS on each
replicate, and build an empirical distribution of the trend estimates.

Bias correction follows the classical bootstrap formula::

    corrected = 2 * original_estimate - bootstrap_replicate

P-values use two-sided linear interpolation at zero.  Confidence
intervals use the Weibull plotting-position quantile (type 6 in R).
"""

import numpy as np
import pandas as pd

from wrtds.surfaces import compute_surface_index, estimate_surfaces
from wrtds.trends import run_groups as _run_groups, run_pairs as _run_pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def block_resample(sample, block_length=200, rng=None):
    """Block bootstrap resample of the sample DataFrame.

    Blocks are defined by Julian days.  A block of *block_length* days
    captures all samples within that contiguous window.  Blocks are
    drawn with replacement until the resampled dataset has at least as
    many rows as the original, then trimmed to match.

    Args:
        sample: Populated sample DataFrame (must have ``Julian`` column,
            sorted by date).
        block_length: Block length in days (default 200).
        rng: Optional :class:`numpy.random.Generator`.

    Returns:
        Resampled DataFrame with the same number of rows as input,
        sorted by Julian date.  May contain duplicate observations.
    """
    if rng is None:
        rng = np.random.default_rng()

    julian = sample['Julian'].values
    n = len(sample)
    day_one = int(julian[0])
    last_julian = int(julian[-1])

    # Possible block start dates — allows blocks that start before the
    # first sample date but overlap with the data range.
    first_start = day_one - block_length + 1
    possible_starts = np.arange(first_start, last_julian + 1)

    indices = []
    while len(indices) < n:
        random_date = rng.choice(possible_starts)
        block_start = max(random_date, day_one)
        block_end = min(last_julian, random_date + block_length - 1)

        # Half-open interval: >= start and < end (matches R convention)
        mask = (julian >= block_start) & (julian < block_end)
        block_idx = np.where(mask)[0]
        if len(block_idx) > 0:
            indices.extend(block_idx.tolist())

    # Trim to exactly n rows (keep the last n, matching R's prepend-then-trim)
    indices = indices[-n:]

    result = sample.iloc[indices].copy().reset_index(drop=True)
    result = result.sort_values('Julian').reset_index(drop=True)
    return result


def pval(s):
    """Compute two-sided p-value from a bootstrap distribution.

    Uses linear interpolation between the largest negative and smallest
    positive bootstrap replicates to estimate where zero falls in the
    empirical distribution.  Matches the ``pVal`` function in R EGRETci.

    Args:
        s: 1-D array of bootstrap replicate values.

    Returns:
        Two-sided p-value (float).
    """
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    s = s[np.abs(s) > 0]
    s = np.sort(s)
    m = len(s)

    if m == 0:
        return 1.0

    x = int(np.sum(s > 0))  # number of positive values

    # Special case: all values have the same sign
    if x == 0 or x == m:
        return 2.0 / (m + 1)

    # Order statistics around the zero crossing (1-indexed in formulas)
    kp = m - x + 1  # rank of smallest positive value
    kn = kp - 1     # rank of largest negative value

    # Linear interpolation to find where zero falls (0-indexed arrays)
    b1 = (kn - kp) / (s[kn - 1] - s[kp - 1])
    k0 = kp - b1 * s[kp - 1]
    p = k0 / (m + 1)  # Weibull plotting position

    return float(2.0 * min(p, 1.0 - p))


def likelihood_descriptor(likelihood):
    """Map a likelihood value to a descriptive word.

    Uses the EGRETci convention.

    Args:
        likelihood: Probability in [0, 1].

    Returns:
        String descriptor.
    """
    if likelihood <= 0.05:
        return 'highly unlikely'
    elif likelihood <= 0.10:
        return 'very unlikely'
    elif likelihood <= 0.33:
        return 'unlikely'
    elif likelihood <= 0.67:
        return 'about as likely as not'
    elif likelihood <= 0.90:
        return 'likely'
    elif likelihood <= 0.95:
        return 'very likely'
    else:
        return 'highly likely'


def _summarise_bootstrap(boot_conc, boot_flux, observed):
    """Build the result dict from bootstrap replicates.

    Args:
        boot_conc: 1-D array of bias-corrected concentration changes.
        boot_flux: 1-D array of bias-corrected flux changes.
        observed: DataFrame from ``run_pairs`` or ``run_groups``.

    Returns:
        Dict with keys: ``observed``, ``boot_conc``, ``boot_flux``,
        ``p_conc``, ``p_flux``, ``ci_conc``, ``ci_flux``,
        ``likelihood_conc_up``, ``likelihood_flux_up``,
        ``like_conc_up``, ``like_conc_down``,
        ``like_flux_up``, ``like_flux_down``.
    """
    p_conc = pval(boot_conc)
    p_flux = pval(boot_flux)

    ci_conc = tuple(np.quantile(boot_conc, [0.025, 0.975], method='weibull'))
    ci_flux = tuple(np.quantile(boot_flux, [0.025, 0.975], method='weibull'))

    n_good = len(boot_conc)
    pos_conc = int(np.sum(boot_conc > 0))
    pos_flux = int(np.sum(boot_flux > 0))
    like_conc_up = (pos_conc + 0.5) / (n_good + 1)
    like_flux_up = (pos_flux + 0.5) / (n_good + 1)

    return {
        'observed': observed,
        'boot_conc': boot_conc,
        'boot_flux': boot_flux,
        'p_conc': p_conc,
        'p_flux': p_flux,
        'ci_conc': ci_conc,
        'ci_flux': ci_flux,
        'likelihood_conc_up': like_conc_up,
        'likelihood_flux_up': like_flux_up,
        'like_conc_up': likelihood_descriptor(like_conc_up),
        'like_conc_down': likelihood_descriptor(1 - like_conc_up),
        'like_flux_up': likelihood_descriptor(like_flux_up),
        'like_flux_down': likelihood_descriptor(1 - like_flux_up),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bootstrap_pairs(
    sample,
    daily,
    year1,
    year2,
    n_boot=100,
    block_length=200,
    window_side=7,
    pa_start=10,
    pa_long=12,
    fit_params=None,
    seed=None,
):
    """Block bootstrap CI for pairwise trend comparison.

    For each bootstrap replicate:

    1. Block-resample the sample DataFrame.
    2. Estimate two 1-year surfaces and run trend decomposition.
    3. Apply bias correction: ``corrected = 2 * original - bootstrap``.

    Args:
        sample: Populated sample DataFrame.
        daily: Populated daily DataFrame.
        year1: First comparison year.
        year2: Second comparison year.
        n_boot: Number of bootstrap replicates.
        block_length: Block length in days for resampling (default 200).
        window_side: Half-window for generalized flow normalisation.
        pa_start: Period of analysis start month.
        pa_long: Period of analysis length in months.
        fit_params: Dict of regression parameters
            (``window_y``, ``window_q``, etc.).
        seed: Optional integer seed for reproducibility.

    Returns:
        Dict with keys:

        - ``observed``: DataFrame from :func:`~wrtds.trends.run_pairs`
        - ``boot_conc``: 1-D array of bias-corrected concentration changes
        - ``boot_flux``: 1-D array of bias-corrected flux changes
        - ``p_conc``: two-sided p-value for concentration change
        - ``p_flux``: two-sided p-value for flux change
        - ``ci_conc``: ``(lower, upper)`` 95 % CI for concentration change
        - ``ci_flux``: ``(lower, upper)`` 95 % CI for flux change
        - ``likelihood_conc_up``: probability of upward concentration trend
        - ``likelihood_flux_up``: probability of upward flux trend
        - ``like_conc_up``, ``like_conc_down``,
          ``like_flux_up``, ``like_flux_down``: descriptive strings
    """
    rng = np.random.default_rng(seed)

    # 1. Observed trend
    observed = _run_pairs(
        sample, daily, year1, year2,
        window_side=window_side, pa_start=pa_start, pa_long=pa_long,
        fit_params=fit_params,
    )
    obs_conc = observed.loc['Conc', 'TotalChange']
    obs_flux = observed.loc['Flux', 'TotalChange']

    # 2. Bootstrap loop
    boot_conc = []
    boot_flux = []
    n_success = 0
    max_attempts = 2 * n_boot

    for _ in range(max_attempts):
        if n_success >= n_boot:
            break

        try:
            boot_sample = block_resample(sample, block_length, rng=rng)
            boot_result = _run_pairs(
                boot_sample, daily, year1, year2,
                window_side=window_side, pa_start=pa_start, pa_long=pa_long,
                fit_params=fit_params,
            )

            # Bias correction
            boot_conc.append(2 * obs_conc - boot_result.loc['Conc', 'TotalChange'])
            boot_flux.append(2 * obs_flux - boot_result.loc['Flux', 'TotalChange'])
            n_success += 1
        except Exception:
            continue

    boot_conc = np.array(boot_conc)
    boot_flux = np.array(boot_flux)

    return _summarise_bootstrap(boot_conc, boot_flux, observed)


def bootstrap_groups(
    daily,
    sample,
    surfaces,
    surface_index,
    group1_years,
    group2_years,
    n_boot=100,
    block_length=200,
    window_side=7,
    pa_start=10,
    pa_long=12,
    fit_params=None,
    seed=None,
):
    """Block bootstrap CI for group trend comparison.

    For each bootstrap replicate:

    1. Block-resample the sample DataFrame.
    2. Re-estimate the full surface from the resampled data.
    3. Run group trend decomposition.
    4. Apply bias correction.

    Args:
        daily: Populated daily DataFrame.
        sample: Populated sample DataFrame.
        surfaces: 3-D surfaces array (original).
        surface_index: Grid parameters dict (original).
        group1_years: ``(first_year, last_year)`` for group 1.
        group2_years: ``(first_year, last_year)`` for group 2.
        n_boot: Number of bootstrap replicates.
        block_length: Block length in days (default 200).
        window_side: Half-window for generalized flow normalisation.
        pa_start: Period of analysis start month.
        pa_long: Period of analysis length in months.
        fit_params: Dict of regression parameters.
        seed: Optional integer seed for reproducibility.

    Returns:
        Dict with same keys as :func:`bootstrap_pairs`.
    """
    rng = np.random.default_rng(seed)

    if fit_params is None:
        fit_params = {
            'window_y': 7, 'window_q': 2, 'window_s': 0.5,
            'min_num_obs': 100, 'min_num_uncen': 50, 'edge_adjust': True,
        }

    # 1. Observed trend
    observed = _run_groups(
        daily, surfaces, surface_index,
        group1_years, group2_years,
        window_side=window_side, pa_start=pa_start, pa_long=pa_long,
    )
    obs_conc = observed.loc['Conc', 'TotalChange']
    obs_flux = observed.loc['Flux', 'TotalChange']

    # 2. Bootstrap loop
    boot_conc = []
    boot_flux = []
    n_success = 0
    max_attempts = 2 * n_boot

    for _ in range(max_attempts):
        if n_success >= n_boot:
            break

        try:
            boot_sample = block_resample(sample, block_length, rng=rng)

            # Re-estimate full surface from resampled data
            boot_si = compute_surface_index(boot_sample)
            boot_surfaces = estimate_surfaces(boot_sample, boot_si, **fit_params)

            boot_result = _run_groups(
                daily, boot_surfaces, boot_si,
                group1_years, group2_years,
                window_side=window_side, pa_start=pa_start, pa_long=pa_long,
            )

            boot_conc.append(2 * obs_conc - boot_result.loc['Conc', 'TotalChange'])
            boot_flux.append(2 * obs_flux - boot_result.loc['Flux', 'TotalChange'])
            n_success += 1
        except Exception:
            continue

    boot_conc = np.array(boot_conc)
    boot_flux = np.array(boot_flux)

    return _summarise_bootstrap(boot_conc, boot_flux, observed)
