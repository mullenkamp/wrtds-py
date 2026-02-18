"""Tests for wrtds.cross_val module."""

import numpy as np
import pandas as pd
import pytest

from wrtds.cross_val import cross_validate
from wrtds.data_prep import populate_daily, populate_sample
from wrtds.regression import compute_weights, predict, run_surv_reg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def sample_and_cv():
    """Build a small sample and run cross-validation once for the module."""
    rng = np.random.default_rng(42)

    dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': q}))

    sample_dates = dates[::15]
    n = len(sample_dates)
    conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
    conc_low = conc.copy()
    conc_high = conc.copy()
    conc_low[:3] = np.nan
    conc_high[:3] = conc[:3] * 1.5

    sample = populate_sample(
        pd.DataFrame({'Date': sample_dates, 'ConcLow': conc_low, 'ConcHigh': conc_high}),
        daily,
    )

    cv_result = cross_validate(
        sample,
        window_y=7, window_q=2, window_s=0.5,
        min_num_obs=10, min_num_uncen=5,
    )
    return sample, cv_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrossValidate:
    def test_adds_columns(self, sample_and_cv):
        _, cv = sample_and_cv
        for col in ('yHat', 'SE', 'ConcHat'):
            assert col in cv.columns

    def test_does_not_mutate_input(self, sample_and_cv):
        sample, _ = sample_and_cv
        assert 'yHat' not in sample.columns

    def test_same_length(self, sample_and_cv):
        sample, cv = sample_and_cv
        assert len(cv) == len(sample)

    def test_yhat_finite(self, sample_and_cv):
        _, cv = sample_and_cv
        assert np.all(np.isfinite(cv['yHat'].values))

    def test_se_positive(self, sample_and_cv):
        _, cv = sample_and_cv
        assert (cv['SE'] > 0).all()

    def test_conchat_positive(self, sample_and_cv):
        _, cv = sample_and_cv
        assert (cv['ConcHat'] > 0).all()

    def test_bias_correction(self, sample_and_cv):
        """ConcHat should equal exp(yHat + SE²/2)."""
        _, cv = sample_and_cv
        expected = np.exp(cv['yHat'].values + cv['SE'].values ** 2 / 2)
        np.testing.assert_allclose(cv['ConcHat'].values, expected, rtol=1e-10)

    def test_no_data_leakage(self):
        """Verify that observation i is truly excluded when predicting i.

        Manually run LOO for one observation and confirm we get the same
        result as cross_validate, proving the held-out obs is excluded.
        """
        rng = np.random.default_rng(99)
        dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
        daily = populate_daily(pd.DataFrame({'Date': dates, 'Q': q}))

        sample_dates = dates[::20]
        n = len(sample_dates)
        conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
        sample = populate_sample(
            pd.DataFrame({'Date': sample_dates, 'ConcLow': conc, 'ConcHigh': conc}),
            daily,
        )

        cv = cross_validate(sample, window_y=7, window_q=2, window_s=0.5,
                            min_num_obs=10, min_num_uncen=5)

        # Manually reproduce LOO for observation 5
        i = 5
        dec_year = sample['DecYear'].values
        logq = sample['LogQ'].values
        uncen = sample['Uncen'].values.astype(bool)

        mask = np.ones(n, dtype=bool)
        mask[i] = False

        loo_data = {
            'DecYear': dec_year[mask],
            'LogQ': logq[mask],
            'SinDY': sample['SinDY'].values[mask],
            'CosDY': sample['CosDY'].values[mask],
            'ConcLow': sample['ConcLow'].values[mask],
            'ConcHigh': sample['ConcHigh'].values[mask],
            'Uncen': uncen[mask],
        }

        w = compute_weights(
            loo_data['DecYear'], loo_data['LogQ'], loo_data['Uncen'],
            dec_year[i], logq[i],
            window_y=7, window_q=2, window_s=0.5,
            min_num_obs=10, min_num_uncen=5,
            record_start=float(dec_year.min()),
            record_end=float(dec_year.max()),
        )

        beta, sigma = run_surv_reg(loo_data, w)
        yh, se, ch = predict(beta, sigma, dec_year[i], logq[i])

        assert cv['yHat'].iloc[i] == pytest.approx(float(yh), rel=1e-10)
        assert cv['SE'].iloc[i] == pytest.approx(float(se), rel=1e-10)
        assert cv['ConcHat'].iloc[i] == pytest.approx(float(ch), rel=1e-10)

    def test_predictions_reasonable(self, sample_and_cv):
        """Predicted concentrations should be in the same ballpark as observed."""
        _, cv = sample_and_cv
        uncen = cv['Uncen'] == 1
        if uncen.sum() < 5:
            pytest.skip('Too few uncensored observations')

        log_pred = cv.loc[uncen, 'yHat'].values
        log_obs = np.log(cv.loc[uncen, 'ConcAve'].values)
        # RMSE in log-space should be reasonable (< 2 for this small dataset)
        rmse = np.sqrt(np.mean((log_pred - log_obs) ** 2))
        assert rmse < 2.0

    def test_preserves_original_columns(self, sample_and_cv):
        """Original sample columns should still be present."""
        sample, cv = sample_and_cv
        for col in sample.columns:
            assert col in cv.columns
