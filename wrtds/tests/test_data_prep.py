"""Tests for wrtds.data_prep module."""

import numpy as np
import pandas as pd
import pytest

from wrtds.data_prep import compress_data, decimal_date, populate_daily, populate_sample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_df():
    """Daily discharge DataFrame covering all of year 2000 (leap year)."""
    dates = pd.date_range('2000-01-01', '2000-12-31', freq='D')
    rng = np.random.default_rng(42)
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    return pd.DataFrame({'Date': dates, 'Q': q})


@pytest.fixture
def populated_daily(daily_df):
    """Pre-populated daily DataFrame."""
    return populate_daily(daily_df)


@pytest.fixture
def sample_conc_remark():
    """Sample DataFrame in Conc+Remark format."""
    return pd.DataFrame({
        'Date': pd.to_datetime(['2000-03-15', '2000-06-01', '2000-09-10', '2000-11-20']),
        'Conc': [5.2, 0.5, 3.1, 8.0],
        'Remark': ['', '<', '', '<'],
    })


@pytest.fixture
def sample_conclowhigh():
    """Sample DataFrame in ConcLow/ConcHigh format."""
    return pd.DataFrame({
        'Date': pd.to_datetime(['2000-03-15', '2000-06-01', '2000-09-10', '2000-11-20']),
        'ConcLow': [5.2, np.nan, 3.1, np.nan],
        'ConcHigh': [5.2, 0.5, 3.1, 8.0],
    })


# ---------------------------------------------------------------------------
# decimal_date
# ---------------------------------------------------------------------------

class TestDecimalDate:
    def test_jan_first(self):
        result = decimal_date(pd.Series(pd.to_datetime(['2000-01-01'])))
        assert result.iloc[0] == pytest.approx(2000.0)

    def test_midyear_leap(self):
        # 2000 is a leap year (366 days). July 2 = 183 days after Jan 1.
        result = decimal_date(pd.Series(pd.to_datetime(['2000-07-02'])))
        assert result.iloc[0] == pytest.approx(2000.0 + 183.0 / 366.0)

    def test_midyear_nonleap(self):
        # 2001 is not a leap year (365 days). July 2 = 182 days after Jan 1.
        result = decimal_date(pd.Series(pd.to_datetime(['2001-07-02'])))
        assert result.iloc[0] == pytest.approx(2001.0 + 182.0 / 365.0)

    def test_dec_31_nonleap(self):
        # Dec 31 of 2001 = 364 days after Jan 1.
        result = decimal_date(pd.Series(pd.to_datetime(['2001-12-31'])))
        assert result.iloc[0] == pytest.approx(2001.0 + 364.0 / 365.0)

    def test_feb_29_leap(self):
        # Feb 29 of 2000 = 59 days after Jan 1.
        result = decimal_date(pd.Series(pd.to_datetime(['2000-02-29'])))
        assert result.iloc[0] == pytest.approx(2000.0 + 59.0 / 366.0)

    def test_multiple_dates(self):
        dates = pd.Series(pd.to_datetime(['2000-01-01', '2000-07-02', '2001-01-01']))
        result = decimal_date(dates)
        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(2000.0)
        assert result.iloc[2] == pytest.approx(2001.0)

    def test_string_input(self):
        result = decimal_date(pd.Series(['2000-01-01']))
        assert result.iloc[0] == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# populate_daily
# ---------------------------------------------------------------------------

class TestPopulateDaily:
    def test_required_columns_present(self, daily_df):
        result = populate_daily(daily_df)
        expected_cols = {'Date', 'Q', 'LogQ', 'Julian', 'DecYear', 'Month', 'Day', 'MonthSeq'}
        assert expected_cols.issubset(set(result.columns))

    def test_missing_date_column(self):
        df = pd.DataFrame({'Q': [1.0, 2.0]})
        with pytest.raises(ValueError, match='missing required columns'):
            populate_daily(df)

    def test_missing_q_column(self):
        df = pd.DataFrame({'Date': ['2000-01-01', '2000-01-02']})
        with pytest.raises(ValueError, match='missing required columns'):
            populate_daily(df)

    def test_negative_q(self):
        df = pd.DataFrame({'Date': ['2000-01-01'], 'Q': [-1.0]})
        with pytest.raises(ValueError, match='positive'):
            populate_daily(df)

    def test_zero_q(self):
        df = pd.DataFrame({'Date': ['2000-01-01'], 'Q': [0.0]})
        with pytest.raises(ValueError, match='positive'):
            populate_daily(df)

    def test_logq(self, daily_df):
        result = populate_daily(daily_df)
        np.testing.assert_allclose(result['LogQ'], np.log(result['Q']))

    def test_julian(self):
        df = pd.DataFrame({
            'Date': ['1850-01-01', '1850-01-02', '2000-01-01'],
            'Q': [1.0, 2.0, 3.0],
        })
        result = populate_daily(df)
        assert result.loc[result['Date'] == pd.Timestamp('1850-01-01'), 'Julian'].iloc[0] == 0
        assert result.loc[result['Date'] == pd.Timestamp('1850-01-02'), 'Julian'].iloc[0] == 1
        # 2000-01-01: 150 years, counting leap years
        julian_2000 = (pd.Timestamp('2000-01-01') - pd.Timestamp('1850-01-01')).days
        assert result.loc[result['Date'] == pd.Timestamp('2000-01-01'), 'Julian'].iloc[0] == julian_2000

    def test_decyear(self):
        df = pd.DataFrame({'Date': ['2000-01-01', '2001-01-01'], 'Q': [1.0, 2.0]})
        result = populate_daily(df)
        assert result['DecYear'].iloc[0] == pytest.approx(2000.0)
        assert result['DecYear'].iloc[1] == pytest.approx(2001.0)

    def test_month(self, daily_df):
        result = populate_daily(daily_df)
        assert result['Month'].min() == 1
        assert result['Month'].max() == 12

    def test_day_of_year(self):
        df = pd.DataFrame({
            'Date': ['2000-01-01', '2000-12-31'],
            'Q': [1.0, 2.0],
        })
        result = populate_daily(df)
        assert result['Day'].iloc[0] == 1
        assert result['Day'].iloc[1] == 366  # 2000 is a leap year

    def test_monthseq(self):
        df = pd.DataFrame({
            'Date': ['1850-01-15', '1850-02-15', '2000-10-15'],
            'Q': [1.0, 2.0, 3.0],
        })
        result = populate_daily(df)
        assert result.loc[result['Date'] == pd.Timestamp('1850-01-15'), 'MonthSeq'].iloc[0] == 1
        assert result.loc[result['Date'] == pd.Timestamp('1850-02-15'), 'MonthSeq'].iloc[0] == 2
        # 2000-10: (2000-1850)*12 + 10 = 1800 + 10 = 1810
        assert result.loc[result['Date'] == pd.Timestamp('2000-10-15'), 'MonthSeq'].iloc[0] == 1810

    def test_sorted_by_date(self):
        df = pd.DataFrame({
            'Date': ['2000-03-01', '2000-01-01', '2000-02-01'],
            'Q': [3.0, 1.0, 2.0],
        })
        result = populate_daily(df)
        assert result['Date'].is_monotonic_increasing

    def test_string_dates(self):
        df = pd.DataFrame({'Date': ['2000-01-01', '2000-01-02'], 'Q': [1.0, 2.0]})
        result = populate_daily(df)
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])


# ---------------------------------------------------------------------------
# compress_data
# ---------------------------------------------------------------------------

class TestCompressData:
    def test_basic(self, sample_conc_remark):
        result = compress_data(sample_conc_remark)
        assert 'ConcLow' in result.columns
        assert 'ConcHigh' in result.columns
        assert 'Conc' not in result.columns
        assert 'Remark' not in result.columns

    def test_uncensored_values(self, sample_conc_remark):
        result = compress_data(sample_conc_remark)
        # Rows 0 and 2 are uncensored (Remark='')
        assert result['ConcLow'].iloc[0] == 5.2
        assert result['ConcHigh'].iloc[0] == 5.2
        assert result['ConcLow'].iloc[2] == 3.1
        assert result['ConcHigh'].iloc[2] == 3.1

    def test_censored_values(self, sample_conc_remark):
        result = compress_data(sample_conc_remark)
        # Rows 1 and 3 are censored (Remark='<')
        assert np.isnan(result['ConcLow'].iloc[1])
        assert result['ConcHigh'].iloc[1] == 0.5
        assert np.isnan(result['ConcLow'].iloc[3])
        assert result['ConcHigh'].iloc[3] == 8.0

    def test_missing_remark_column(self):
        df = pd.DataFrame({
            'Date': ['2000-01-01'],
            'Conc': [5.0],
        })
        result = compress_data(df)
        # No Remark -> all uncensored
        assert result['ConcLow'].iloc[0] == 5.0
        assert result['ConcHigh'].iloc[0] == 5.0

    def test_nan_remark(self):
        df = pd.DataFrame({
            'Date': ['2000-01-01', '2000-02-01'],
            'Conc': [5.0, 3.0],
            'Remark': [None, '<'],
        })
        result = compress_data(df)
        assert result['ConcLow'].iloc[0] == 5.0
        assert np.isnan(result['ConcLow'].iloc[1])

    def test_missing_conc_column(self):
        df = pd.DataFrame({'Date': ['2000-01-01'], 'Remark': ['']})
        with pytest.raises(ValueError, match='missing required columns'):
            compress_data(df)


# ---------------------------------------------------------------------------
# populate_sample
# ---------------------------------------------------------------------------

class TestPopulateSample:
    def test_required_columns_present(self, sample_conclowhigh, populated_daily):
        result = populate_sample(sample_conclowhigh, populated_daily)
        expected_cols = {
            'Date', 'ConcLow', 'ConcHigh', 'Uncen', 'ConcAve',
            'Julian', 'DecYear', 'Month', 'Day', 'MonthSeq',
            'SinDY', 'CosDY', 'Q', 'LogQ',
        }
        assert expected_cols.issubset(set(result.columns))

    def test_missing_columns(self, populated_daily):
        df = pd.DataFrame({'Date': ['2000-01-01']})
        with pytest.raises(ValueError, match='missing required columns'):
            populate_sample(df, populated_daily)

    def test_auto_compress(self, sample_conc_remark, populated_daily):
        result = populate_sample(sample_conc_remark, populated_daily)
        assert 'ConcLow' in result.columns
        assert 'ConcHigh' in result.columns
        assert 'Conc' not in result.columns

    def test_uncen_flag(self, sample_conclowhigh, populated_daily):
        result = populate_sample(sample_conclowhigh, populated_daily)
        # Rows where ConcLow == ConcHigh -> Uncen=1
        assert result['Uncen'].iloc[0] == 1  # 5.2 == 5.2
        assert result['Uncen'].iloc[1] == 0  # NaN != 0.5
        assert result['Uncen'].iloc[2] == 1  # 3.1 == 3.1
        assert result['Uncen'].iloc[3] == 0  # NaN != 8.0

    def test_conc_ave_uncensored(self, sample_conclowhigh, populated_daily):
        result = populate_sample(sample_conclowhigh, populated_daily)
        # Uncensored: ConcAve = (ConcLow + ConcHigh) / 2 = ConcLow
        assert result['ConcAve'].iloc[0] == pytest.approx(5.2)
        assert result['ConcAve'].iloc[2] == pytest.approx(3.1)

    def test_conc_ave_censored(self, sample_conclowhigh, populated_daily):
        result = populate_sample(sample_conclowhigh, populated_daily)
        # Censored: ConcAve = (0 + ConcHigh) / 2 = ConcHigh / 2
        assert result['ConcAve'].iloc[1] == pytest.approx(0.25)
        assert result['ConcAve'].iloc[3] == pytest.approx(4.0)

    def test_time_columns(self, sample_conclowhigh, populated_daily):
        result = populate_sample(sample_conclowhigh, populated_daily)
        # March 15, 2000: month=3, day=75 (31+29+15), Julian > 0
        row = result[result['Date'] == pd.Timestamp('2000-03-15')].iloc[0]
        assert row['Month'] == 3
        assert row['Day'] == 75
        assert row['Julian'] > 0
        assert row['DecYear'] == pytest.approx(2000.0 + 74.0 / 366.0)

    def test_seasonal_harmonics(self, sample_conclowhigh, populated_daily):
        result = populate_sample(sample_conclowhigh, populated_daily)
        dec_year = result['DecYear']
        np.testing.assert_allclose(result['SinDY'], np.sin(2 * np.pi * dec_year))
        np.testing.assert_allclose(result['CosDY'], np.cos(2 * np.pi * dec_year))

    def test_q_merge(self, sample_conclowhigh, populated_daily):
        result = populate_sample(sample_conclowhigh, populated_daily)
        # All sample dates are within daily range, so Q should not be NaN
        assert result['Q'].notna().all()
        assert result['LogQ'].notna().all()
        np.testing.assert_allclose(result['LogQ'], np.log(result['Q']))

    def test_q_merge_values(self, populated_daily):
        """Verify merged Q matches the daily Q for that date."""
        sample = pd.DataFrame({
            'Date': [populated_daily['Date'].iloc[10]],
            'ConcLow': [5.0],
            'ConcHigh': [5.0],
        })
        result = populate_sample(sample, populated_daily)
        expected_q = populated_daily['Q'].iloc[10]
        assert result['Q'].iloc[0] == pytest.approx(expected_q)

    def test_sorted_by_date(self, populated_daily):
        sample = pd.DataFrame({
            'Date': pd.to_datetime(['2000-06-01', '2000-01-15', '2000-10-01']),
            'ConcLow': [1.0, 2.0, 3.0],
            'ConcHigh': [1.0, 2.0, 3.0],
        })
        result = populate_sample(sample, populated_daily)
        assert result['Date'].is_monotonic_increasing

    def test_missing_q_warns(self):
        """Sample date outside daily range should produce NaN Q and a warning."""
        daily = pd.DataFrame({
            'Date': pd.date_range('2000-01-01', '2000-12-31'),
            'Q': np.ones(366),
        })
        daily = populate_daily(daily)

        sample = pd.DataFrame({
            'Date': ['1999-01-01'],  # outside daily range
            'ConcLow': [5.0],
            'ConcHigh': [5.0],
        })
        with pytest.warns(UserWarning, match='no matching daily discharge'):
            result = populate_sample(sample, daily)
        assert result['Q'].isna().iloc[0]

    def test_interval_censored(self, populated_daily):
        """Interval-censored: ConcLow < ConcHigh, both non-NaN."""
        sample = pd.DataFrame({
            'Date': [populated_daily['Date'].iloc[50]],
            'ConcLow': [1.0],
            'ConcHigh': [3.0],
        })
        result = populate_sample(sample, populated_daily)
        assert result['Uncen'].iloc[0] == 0
        assert result['ConcAve'].iloc[0] == pytest.approx(2.0)
