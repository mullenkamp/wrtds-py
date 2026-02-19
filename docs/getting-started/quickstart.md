# Quickstart

This tutorial walks through a complete WRTDS analysis: loading data, fitting the model,
running WRTDS-K, computing trends, and generating plots.

## Load Data

WRTDS needs two DataFrames:

- **Daily** — one row per day with columns `Date` and `Q` (discharge in m^3^/s)
- **Sample** — water-quality observations with columns `Date`, `ConcLow`, and `ConcHigh`

```python
import pandas as pd
from wrtds import WRTDS

daily = pd.read_csv('daily.csv', parse_dates=['Date'])
sample = pd.read_csv('sample.csv', parse_dates=['Date'])
```

See [Input Data Format](input-data.md) for details on column requirements and alternative formats.

## Create a WRTDS Object

```python
w = WRTDS(
    daily,
    sample,
    info={
        'station_name': 'Choptank River',
        'param_name': 'Nitrate',
        'drainage_area_km2': 292.0,
    },
)
```

The constructor validates and prepares the data, computing derived columns like
`LogQ`, `DecYear`, `Julian`, and seasonal harmonics.

## Fit the Model

The `fit()` method runs the full pipeline: leave-one-out cross-validation, surface
estimation, daily interpolation, and flow normalization.

```python
w.fit()
```

!!! note
    This is the most computationally expensive step. It solves thousands of weighted
    censored regressions across the time-discharge grid. Expect it to take a few minutes
    for a typical dataset.

After fitting, `w.daily` contains columns like `ConcDay`, `FluxDay`, `FNConc`, and `FNFlux`.

## Run WRTDS-K (Optional)

WRTDS-K uses AR(1) residual interpolation to improve daily estimates between sample dates:

```python
w.kalman(rho=0.90, n_iter=200)
```

This adds `GenConc` and `GenFlux` columns to `w.daily`.

## Explore Results

### Annual Summary Table

```python
annual = w.table_results()
print(annual)
```

### Change Table

```python
changes = w.table_change(year_points=[1985, 1995, 2005, 2010])
print(changes)
```

### Error Statistics

```python
print(w.error_stats())
print(w.flux_bias_stat())
```

## Trend Analysis

### Pairwise Comparison

```python
pairs = w.run_pairs(year1=1985, year2=2010)
print(pairs)
```

### Group Comparison

```python
groups = w.run_groups(
    group1_years=(1985, 1996),
    group2_years=(1997, 2010),
)
print(groups)
```

### Bootstrap Confidence Intervals

```python
boot = w.bootstrap_pairs(year1=1985, year2=2010, n_boot=100, seed=42)
print(f"Conc p-value: {boot['p_conc']:.3f}")
print(f"Flux p-value: {boot['p_flux']:.3f}")
print(f"Conc direction: {boot['like_conc_down']}")
```

## Plotting

```python
# Data overview (2x2 panel)
w.plot_overview()

# Annual concentration history with flow-normalized trend
w.plot_conc_hist()

# Annual flux history
w.plot_flux_hist()

# Surface contour plot
w.plot_contours()

# Diagnostic residual plots (6-panel)
w.plot_residuals()

# Predicted vs observed
w.plot_conc_pred()
```

All plot methods return a `matplotlib.figure.Figure` that you can further customise
or save:

```python
fig = w.plot_conc_hist()
fig.savefig('conc_history.png', dpi=150, bbox_inches='tight')
```
