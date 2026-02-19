# Plotting

All plot methods are available on the `WRTDS` class and return a `matplotlib.figure.Figure`.
The underlying functions in `wrtds.plots` can also be called directly for more control.

## Data Overview Plots

### `plot_overview`

A 2x2 panel showing discharge time series, concentration vs time, concentration vs
discharge, and monthly concentration box plots.

```python
fig = w.plot_overview()
```

The individual panels can be created separately using the functions in
[`wrtds.plots.data_overview`](../reference/plots/data_overview.md):

- `plot_q_time_daily(daily)` — daily discharge time series (log-scale y-axis)
- `plot_conc_time(sample)` — concentration vs time (open circles for censored data)
- `plot_conc_q(sample)` — log-log concentration vs discharge
- `box_conc_month(sample)` — box plots of concentration by month
- `box_q_twice(daily, sample)` — side-by-side discharge distributions for all days vs sample days

## Result Plots

### `plot_conc_hist`

Annual mean concentration as bars with the flow-normalised trend line overlaid.
If WRTDS-K has been run, the generalized concentration line is also shown.

```python
fig = w.plot_conc_hist()
```

### `plot_flux_hist`

Same as above but for flux. The `flux_factor` parameter can convert units.

```python
fig = w.plot_flux_hist(flux_factor=1.0)
```

### `plot_contours`

Filled contour plot of a surface layer. The `layer` parameter selects which surface
to plot:

- `layer=0` — predicted log-concentration (`yHat`)
- `layer=1` — standard error (`SE`)
- `layer=2` — bias-corrected concentration (`ConcHat`, the default)

```python
fig = w.plot_contours(layer=2)
```

Additional functions in [`wrtds.plots.results`](../reference/plots/results.md):

- `plot_conc_q_smooth(surfaces, surface_index, years)` — C-Q curves at selected years
- `plot_conc_time_smooth(surfaces, surface_index, logq_values)` — concentration-time curves at selected discharges
- `plot_diff_contours(surfaces1, surfaces2, surface_index)` — difference contour between two surfaces

## Diagnostic Plots

### `plot_residuals`

A 6-panel diagnostic display:

```python
fig = w.plot_residuals()
```

The six panels are:

1. Predicted vs observed concentration
2. Predicted vs observed flux
3. Residuals vs predicted concentration
4. Residuals vs log-discharge
5. Residuals vs time (with running-mean smooth)
6. Monthly box plots of residuals

### `plot_conc_pred`

Predicted vs observed concentration scatter plot with 1:1 reference line.

```python
fig = w.plot_conc_pred()
```

All individual diagnostic functions are in
[`wrtds.plots.diagnostics`](../reference/plots/diagnostics.md).

## Saving Figures

All plot methods return a `matplotlib.figure.Figure`:

```python
fig = w.plot_conc_hist()
fig.savefig('conc_history.png', dpi=150, bbox_inches='tight')
```

## Using the Low-Level API

For full control, use the functions in `wrtds.plots` directly:

```python
from wrtds.plots import plot_conc_hist, plot_contours

annual = w.table_results()
fig = plot_conc_hist(annual)

fig = plot_contours(w.surfaces, w.surface_index, layer=2)
```
