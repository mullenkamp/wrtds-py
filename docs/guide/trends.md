# Trend Analysis

WRTDS provides three approaches to trend analysis, all based on flow normalization
to separate water-quality changes from discharge variability.

## Pairwise Comparison (`run_pairs`)

Compares flow-normalised values between two specific years by estimating separate
one-year surfaces for each year:

```python
pairs = w.run_pairs(year1=1985, year2=2010, window_side=7)
print(pairs)
```

Returns a DataFrame with index `['Conc', 'Flux']` and columns:

| Column | Description |
|--------|-------------|
| `TotalChange` | Total change = CQTC + QTC |
| `CQTC` | Concentration-Q Trend Component — change due to shifts in the C-Q relationship |
| `QTC` | Q Trend Component — change due to shifts in the discharge distribution |
| `x10` | Surface 1, stationary flow distribution |
| `x11` | Surface 1, year 1 flow distribution |
| `x20` | Surface 2, stationary flow distribution |
| `x22` | Surface 2, year 2 flow distribution |

### The xAB Notation

Each `xAB` value represents a flow-normalised estimate where:

- **A** = which year's concentration surface (1 or 2)
- **B** = which year's flow distribution (0 = stationary/full-period, 1 = year 1's window, 2 = year 2's window)

The trend decomposition is:

- `CQTC = x20 - x10` — change holding flow constant
- `QTC = TotalChange - CQTC` — residual change from flow shifts
- `TotalChange = x22 - x11`

### `window_side`

The `window_side` parameter (default 7) defines the half-width of the discharge
distribution window in years. The discharge distribution for each year is taken from
a `2 * window_side + 1` year window centred on that year.

## Group Comparison (`run_groups`)

Compares flow-normalised averages across two multi-year groups, using the existing
full-period surface:

```python
groups = w.run_groups(
    group1_years=(1985, 1996),
    group2_years=(1997, 2010),
    window_side=7,
)
print(groups)
```

Returns the same DataFrame format as `run_pairs`. Group comparison uses the surfaces
estimated during `fit()` rather than re-estimating per-year surfaces.

## Annual Time Series (`run_series`)

Computes a time series of generalized flow-normalised values using a sliding discharge
window:

```python
w.run_series(window_side=7)
```

This updates `w.daily['FNConc']` and `w.daily['FNFlux']` with values computed using
a discharge distribution that evolves over time (sliding window of `2 * window_side + 1`
years), rather than the full-period stationary distribution used by `fit()`.

## Period of Analysis

All trend methods accept `pa_start` and `pa_long` parameters to define the period of
analysis (which months to include). If not specified, they default to the values in
`w.info`:

```python
# Water year (October-September, the default)
pairs = w.run_pairs(1985, 2010, pa_start=10, pa_long=12)

# Calendar year
pairs = w.run_pairs(1985, 2010, pa_start=1, pa_long=12)

# Summer only (June-August)
pairs = w.run_pairs(1985, 2010, pa_start=6, pa_long=3)
```
