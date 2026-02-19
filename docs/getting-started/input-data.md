# Input Data Format

WRTDS requires two DataFrames and an optional metadata dictionary.

## Daily DataFrame

One row per day of discharge record.

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Date of observation |
| `Q` | float | Daily mean discharge in m^3^/s (must be > 0) |

```python
daily = pd.DataFrame({
    'Date': pd.date_range('1980-01-01', '2020-12-31'),
    'Q': discharge_values,
})
```

During construction, `populate_daily` adds the following derived columns:

| Column | Description |
|--------|-------------|
| `LogQ` | Natural log of Q |
| `Julian` | Days since 1850-01-01 |
| `DecYear` | Decimal year (e.g. 2010.5 = July 2010) |
| `Month` | Month (1-12) |
| `Day` | Day of month |
| `MonthSeq` | Sequential month index from start of record |

## Sample DataFrame

Water-quality observations. Two input formats are supported.

### Format 1: ConcLow / ConcHigh (Preferred)

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Date of observation |
| `ConcLow` | float | Lower concentration bound |
| `ConcHigh` | float | Upper concentration bound |

For **uncensored** observations, set `ConcLow == ConcHigh` (the measured value).

For **left-censored** observations (below detection limit), set `ConcLow = 0`
(or `NaN`) and `ConcHigh` to the detection limit.

### Format 2: Conc / Remark

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Date of observation |
| `Conc` | float | Concentration value |
| `Remark` | str | `'<'` for left-censored, empty/NaN for uncensored |

This format is automatically converted to ConcLow/ConcHigh internally using
`compress_data`.

### Derived Sample Columns

After preparation, the Sample DataFrame gains:

| Column | Description |
|--------|-------------|
| `Uncen` | 1 if uncensored, 0 if censored |
| `ConcAve` | Average of ConcLow and ConcHigh |
| `Q` | Discharge (merged from Daily by nearest date) |
| `LogQ` | Natural log of Q |
| `Julian` | Days since 1850-01-01 |
| `DecYear` | Decimal year |
| `Month`, `Day` | Calendar month and day |
| `MonthSeq` | Sequential month index |
| `SinDY` | sin(2 * pi * DecYear) — seasonal harmonic |
| `CosDY` | cos(2 * pi * DecYear) — seasonal harmonic |

## Info Dictionary

Optional metadata dictionary. Missing keys are filled from `DEFAULT_INFO`:

```python
DEFAULT_INFO = {
    'station_name': '',
    'param_name': '',
    'drainage_area_km2': None,
    'pa_start': 10,       # Period-of-analysis start month (10 = October = water year)
    'pa_long': 12,        # Period-of-analysis length in months
}
```

The `pa_start` and `pa_long` parameters define the **period of analysis**. The default
(`pa_start=10, pa_long=12`) uses the water year (October through September). To use
calendar years, set `pa_start=1, pa_long=12`.

```python
w = WRTDS(daily, sample, info={
    'station_name': 'Choptank River at Greensboro',
    'param_name': 'Nitrate-N',
    'drainage_area_km2': 292.0,
    'pa_start': 10,
    'pa_long': 12,
})
```
