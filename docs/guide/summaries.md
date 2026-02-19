# Summary Tables

## Annual Results (`table_results`)

Produces an annual summary table by aggregating daily results. Years with more than
10% missing `ConcDay` values are excluded.

```python
annual = w.table_results()
print(annual)
```

| Column | Description |
|--------|-------------|
| `DecYear` | Decimal year (mid-point of the period) |
| `Q` | Mean discharge (m^3^/s) |
| `Conc` | Mean daily concentration (mg/L) |
| `Flux` | Mean daily flux (kg/day) |
| `FNConc` | Flow-normalised concentration |
| `FNFlux` | Flow-normalised flux |
| `GenConc` | Generalized concentration (if `kalman()` was run) |
| `GenFlux` | Generalized flux (if `kalman()` was run) |

The period of analysis can be customised:

```python
# Calendar year
annual = w.table_results(pa_start=1, pa_long=12)

# Water year (default)
annual = w.table_results(pa_start=10, pa_long=12)
```

## Change Table (`table_change`)

Computes changes in flow-normalised values between specified years:

```python
changes = w.table_change(year_points=[1985, 1995, 2005, 2010])
print(changes)
```

Returns one row per consecutive pair of years with columns:

| Column | Description |
|--------|-------------|
| `Year1`, `Year2` | The comparison endpoints |
| `FNConc_change` | Absolute change in FNConc |
| `FNConc_pct_change` | Percent change in FNConc |
| `FNConc_slope` | Annual slope (change / years) |
| `FNConc_pct_slope` | Annual percent slope |
| `FNFlux_change` | Absolute change in FNFlux |
| `FNFlux_pct_change` | Percent change in FNFlux |
| `FNFlux_slope` | Annual slope for flux |
| `FNFlux_pct_slope` | Annual percent slope for flux |

The `flux_factor` parameter (default `0.00036525`) converts flux from kg/day to
10^6^ kg/year.

## Error Statistics (`error_stats`)

Cross-validation error statistics based on leave-one-out predictions:

```python
stats = w.error_stats()
```

| Key | Description |
|-----|-------------|
| `rsq_log_conc` | R-squared for log-concentration |
| `rsq_log_flux` | R-squared for log-flux |
| `rmse` | Root mean square error (log-space) |
| `sep_percent` | Standard error of prediction as a percentage |

Requires `cross_validate()` or `fit()` to have been called first.

## Flux Bias Statistic (`flux_bias_stat`)

Evaluates the bias of flux estimates relative to observed values:

```python
bias = w.flux_bias_stat()
```

| Key | Description |
|-----|-------------|
| `bias1` | Flux bias using `ConcHigh` as the observed value |
| `bias2` | Flux bias using `ConcLow` as the observed value (NaN treated as 0) |
| `bias3` | Average of `bias1` and `bias2` |

The formula is: `bias = (estimated - observed) / estimated`

A value near zero indicates unbiased estimates. Positive values indicate
overestimation; negative values indicate underestimation.
