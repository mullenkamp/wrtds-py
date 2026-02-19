# WRTDS-K (Kalman)

WRTDS-K extends the standard WRTDS model by using AR(1) residual interpolation to
produce improved daily concentration and flux estimates. It is described in:

> Zhang, Q. and Hirsch, R.M. (2019), River Water-Quality Concentration and Flux
> Estimation Can Be Improved by Accounting for Serial Correlation Through an
> Autoregressive Model. *Water Resources Research*, 55, 9705-9723.

## Running WRTDS-K

WRTDS-K requires a fitted model:

```python
w = WRTDS(daily, sample)
w.fit()
w.kalman(rho=0.90, n_iter=200, seed=42)
```

This adds two columns to `w.daily`:

| Column | Description |
|--------|-------------|
| `GenConc` | Generalized concentration (mg/L) |
| `GenFlux` | Generalized flux (kg/day) |

## How It Works

1. For each Monte Carlo iteration:
    - **Augment censored samples** — draw random concentrations from a truncated
      log-normal for left-censored observations
    - **Compute standardised residuals** at sample dates:
      `e = (ln(C_obs) - yHat) / SE`
    - **Interpolate residuals** for gap days between samples using conditional
      AR(1) draws (conditioned on the residuals at both bounding sample dates)
    - **Convert back** to concentration: `GenConc = exp(yHat + SE * e_interp)`
2. Average across all iterations to get the final `GenConc` and `GenFlux`

## Choosing `rho`

The `rho` parameter is the AR(1) autocorrelation coefficient and controls how strongly
residuals persist from day to day:

| `rho` | Behavior | When to use |
|-------|----------|-------------|
| 0.85 | Reactive | Short-memory systems, flashy catchments |
| 0.90 | Default | Most applications |
| 0.95 | Conservative | Long-memory systems, large rivers |

Higher `rho` values produce smoother interpolation between sample dates. The default
of 0.90 is appropriate for most applications.

## Iterations (`n_iter`)

The `n_iter` parameter controls Monte Carlo averaging. More iterations produce more
stable results but take longer. The default of 200 is sufficient for most applications.
Setting `seed` ensures reproducibility.

## Interpreting Results

- `GenConc` and `GenFlux` are generally more accurate than `ConcDay` and `FluxDay`
  for computing annual totals, because they account for serial correlation in residuals
- Near sample dates, `GenConc` is pulled toward the observed value
- Far from sample dates, `GenConc` reverts toward the WRTDS surface prediction
- Annual means of `GenConc` and `GenFlux` appear in `table_results()` and in the
  annual history plots when available
