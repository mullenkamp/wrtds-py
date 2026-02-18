"""Leave-one-out cross-validation for WRTDS."""

import numpy as np

from wrtds.regression import compute_weights, predict, run_surv_reg


def cross_validate(
    sample,
    window_y=7.0,
    window_q=2.0,
    window_s=0.5,
    min_num_obs=100,
    min_num_uncen=50,
    edge_adjust=True,
):
    """Leave-one-out jack-knife cross-validation.

    For each sample observation *i*, the model is fitted on all other
    observations and used to predict at the held-out point.  This gives
    an honest measure of prediction error.

    Args:
        sample: Populated sample DataFrame with ``DecYear``, ``LogQ``,
            ``SinDY``, ``CosDY``, ``ConcLow``, ``ConcHigh``, ``Uncen``.
        window_y: Time half-window in years.
        window_q: Discharge half-window in log units.
        window_s: Season half-window in fraction of year.
        min_num_obs: Minimum observations with nonzero weight.
        min_num_uncen: Minimum uncensored observations with nonzero weight.
        edge_adjust: Expand time window near record edges.

    Returns:
        Sample DataFrame with added columns ``yHat``, ``SE``, ``ConcHat``.
        Total MLE solves: *n* (one per sample observation).
    """
    sample = sample.copy()
    n = len(sample)

    # Pre-extract arrays for efficiency
    dec_year = sample['DecYear'].values
    logq = sample['LogQ'].values
    sin_dy = sample['SinDY'].values
    cos_dy = sample['CosDY'].values
    conc_low = sample['ConcLow'].values
    conc_high = sample['ConcHigh'].values
    uncen = sample['Uncen'].values.astype(bool)

    record_start = float(dec_year.min())
    record_end = float(dec_year.max())

    yHat_out = np.empty(n)
    SE_out = np.empty(n)
    ConcHat_out = np.empty(n)

    # Boolean index for leave-one-out masking
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        # Exclude observation i
        mask[i] = False

        loo_data = {
            'DecYear': dec_year[mask],
            'LogQ': logq[mask],
            'SinDY': sin_dy[mask],
            'CosDY': cos_dy[mask],
            'ConcLow': conc_low[mask],
            'ConcHigh': conc_high[mask],
            'Uncen': uncen[mask],
        }

        w = compute_weights(
            loo_data['DecYear'],
            loo_data['LogQ'],
            loo_data['Uncen'],
            target_dec_year=dec_year[i],
            target_logq=logq[i],
            window_y=window_y,
            window_q=window_q,
            window_s=window_s,
            min_num_obs=min_num_obs,
            min_num_uncen=min_num_uncen,
            edge_adjust=edge_adjust,
            record_start=record_start,
            record_end=record_end,
        )

        beta, sigma = run_surv_reg(loo_data, w)
        yh, se, ch = predict(beta, sigma, dec_year[i], logq[i])

        yHat_out[i] = float(yh)
        SE_out[i] = float(se)
        ConcHat_out[i] = float(ch)

        # Restore mask for next iteration
        mask[i] = True

    sample['yHat'] = yHat_out
    sample['SE'] = SE_out
    sample['ConcHat'] = ConcHat_out

    return sample
