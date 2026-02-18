"""DataFrame validation, column computation, and utilities for WRTDS."""

import warnings

import numpy as np
import pandas as pd


# --- Info dict defaults ---

DEFAULT_INFO = {
    'station_name': '',
    'param_name': '',
    'drainage_area_km2': None,
    'pa_start': 10,
    'pa_long': 12,
}


def decimal_date(dates: pd.Series) -> pd.Series:
    """Convert datetime Series to decimal year.

    Formula matches R's EGRET::decimalDate:
        year + (date - Jan1) / (Jan1_next_year - Jan1)

    Args:
        dates: Series of datetime-like values.

    Returns:
        Series of float decimal years.
    """
    dates = pd.to_datetime(dates)
    year = dates.dt.year
    start = pd.to_datetime(year, format='%Y')
    end = pd.to_datetime(year + 1, format='%Y')
    return year + (dates - start).dt.total_seconds() / (end - start).dt.total_seconds()


def populate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and compute derived columns for daily discharge data.

    Args:
        daily: DataFrame with at least columns ``Date`` and ``Q`` (m³/s).

    Returns:
        DataFrame sorted by Date with added columns:
        LogQ, Julian, DecYear, Month, Day, MonthSeq.

    Raises:
        ValueError: If required columns are missing or Q contains non-positive values.
    """
    daily = daily.copy()

    missing = {'Date', 'Q'} - set(daily.columns)
    if missing:
        raise ValueError(f'Daily DataFrame missing required columns: {sorted(missing)}')

    daily['Date'] = pd.to_datetime(daily['Date']).dt.normalize()

    q_valid = daily['Q'].dropna()
    if len(q_valid) == 0:
        raise ValueError('Q column has no non-null values')
    if (q_valid <= 0).any():
        raise ValueError('Q values must be positive (found zero or negative values)')

    daily['LogQ'] = np.log(daily['Q'])

    epoch = pd.Timestamp('1850-01-01')
    daily['Julian'] = (daily['Date'] - epoch).dt.days

    daily['DecYear'] = decimal_date(daily['Date'])
    daily['Month'] = daily['Date'].dt.month
    daily['Day'] = daily['Date'].dt.dayofyear
    daily['MonthSeq'] = (daily['Date'].dt.year - 1850) * 12 + daily['Month']

    return daily.sort_values('Date').reset_index(drop=True)


def populate_sample(sample: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Validate, compute derived columns, and merge Q from daily.

    Accepts either ``ConcLow``/``ConcHigh`` columns or ``Conc``/``Remark``
    columns (auto-converted via :func:`compress_data`).

    Args:
        sample: Water quality sample DataFrame.
        daily: Daily discharge DataFrame (must already be populated via
            :func:`populate_daily`, i.e. must have ``Date``, ``Q``, ``LogQ``).

    Returns:
        DataFrame sorted by Date with added columns:
        Uncen, ConcAve, Julian, DecYear, Month, Day, MonthSeq,
        SinDY, CosDY, Q, LogQ.

    Raises:
        ValueError: If required columns are missing.
    """
    sample = sample.copy()

    # Auto-convert from Conc+Remark format if needed
    if 'ConcLow' not in sample.columns and 'Conc' in sample.columns:
        sample = compress_data(sample)

    missing = {'Date', 'ConcLow', 'ConcHigh'} - set(sample.columns)
    if missing:
        raise ValueError(f'Sample DataFrame missing required columns: {sorted(missing)}')

    sample['Date'] = pd.to_datetime(sample['Date']).dt.normalize()

    # Censoring indicators
    sample['Uncen'] = (sample['ConcLow'] == sample['ConcHigh']).astype(int)
    conc_low_filled = sample['ConcLow'].fillna(0.0)
    sample['ConcAve'] = (conc_low_filled + sample['ConcHigh']) / 2

    # Time columns
    epoch = pd.Timestamp('1850-01-01')
    sample['Julian'] = (sample['Date'] - epoch).dt.days
    sample['DecYear'] = decimal_date(sample['Date'])
    sample['Month'] = sample['Date'].dt.month
    sample['Day'] = sample['Date'].dt.dayofyear
    sample['MonthSeq'] = (sample['Date'].dt.year - 1850) * 12 + sample['Month']

    # Seasonal harmonics
    sample['SinDY'] = np.sin(2 * np.pi * sample['DecYear'])
    sample['CosDY'] = np.cos(2 * np.pi * sample['DecYear'])

    # Merge Q from daily by nearest date
    daily_q = daily[['Date', 'Q', 'LogQ']].drop_duplicates(subset='Date').sort_values('Date')
    sample = sample.sort_values('Date').reset_index(drop=True)

    sample = pd.merge_asof(
        sample,
        daily_q,
        on='Date',
        direction='nearest',
        tolerance=pd.Timedelta('1D'),
    )

    n_missing_q = sample['Q'].isna().sum()
    if n_missing_q > 0:
        warnings.warn(
            f'{n_missing_q} sample dates have no matching daily discharge within 1 day',
            stacklevel=2,
        )

    return sample.sort_values('Date').reset_index(drop=True)


def compress_data(sample: pd.DataFrame) -> pd.DataFrame:
    """Convert Conc + Remark format to ConcLow/ConcHigh.

    Args:
        sample: DataFrame with ``Date``, ``Conc``, and optionally ``Remark``
            columns. ``Remark='<'`` indicates left-censored (below detection).

    Returns:
        DataFrame with ``Conc``/``Remark`` replaced by ``ConcLow``/``ConcHigh``.

    Raises:
        ValueError: If required columns are missing.
    """
    sample = sample.copy()

    missing = {'Date', 'Conc'} - set(sample.columns)
    if missing:
        raise ValueError(f'Sample DataFrame missing required columns: {sorted(missing)}')

    if 'Remark' not in sample.columns:
        sample['Remark'] = ''

    remark = sample['Remark'].fillna('').astype(str).str.strip()
    censored = remark == '<'

    sample['ConcHigh'] = sample['Conc']
    sample['ConcLow'] = sample['Conc'].where(~censored, np.nan)

    sample = sample.drop(columns=['Conc', 'Remark'])

    return sample
