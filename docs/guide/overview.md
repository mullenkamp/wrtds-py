# Overview

## What is WRTDS?

**Weighted Regressions on Time, Discharge, and Season** (WRTDS) is a statistical method
for estimating long-term trends in river water quality. It was developed by Robert Hirsch
and colleagues at the U.S. Geological Survey (USGS) and is described in:

> Hirsch, R.M., Moyer, D.L., and Archfield, S.A. (2010), Weighted Regressions on Time,
> Discharge, and Season (WRTDS), with an Application to Chesapeake Bay River Inputs.
> *Journal of the American Water Resources Association*, 46(5), 857-880.

The original implementation is the R package
[EGRET](https://github.com/DOI-USGS/EGRET). This Python package (`wrtds-py`)
is a faithful transcription of EGRET's core algorithms using pandas, numpy, scipy,
and matplotlib.

## The WRTDS Model

WRTDS fits a **locally weighted censored regression** at every point on a grid of
time and discharge. The regression model is:

$$
\ln(C) = \beta_0 + \beta_1 t + \beta_2 \ln(Q) + \beta_3 \sin(2\pi t) + \beta_4 \cos(2\pi t) + \varepsilon
$$

where:

- $C$ is concentration
- $t$ is decimal year
- $Q$ is discharge (m^3^/s)
- $\beta_3, \beta_4$ capture seasonal variation
- $\varepsilon \sim N(0, \sigma^2)$

Each observation is weighted by the product of three **tricube kernel** functions
centred on the target point:

- **Time window** (`window_y`, default 7 years) — weights nearby years more heavily
- **Discharge window** (`window_q`, default 2 log units) — weights similar flow conditions
- **Season window** (`window_s`, default 0.5 years) — weights the same time of year

This local weighting allows the model to capture non-linear, non-stationary relationships
between concentration, discharge, and time.

## Key Concepts

### Surfaces

The model is evaluated on a regular grid of (time, log-discharge) to produce a 3-D
**surface** array. Each grid cell stores the predicted log-concentration (`yHat`),
standard error (`SE`), and bias-corrected concentration (`ConcHat`). Daily values are
then interpolated from this surface.

### Flow Normalization

**Flow-normalised** concentration and flux are computed by averaging the predicted
concentration across the full historical distribution of discharge for each calendar day.
This removes the effect of year-to-year flow variability, isolating the underlying
water-quality trend.

### WRTDS-K (Kalman)

WRTDS-K improves daily estimates by interpolating AR(1) residuals between sample dates
using Monte Carlo simulation. This produces `GenConc` and `GenFlux` — generalized
concentration and flux estimates that better capture day-to-day variability.

### Trend Decomposition

Trends can be decomposed into:

- **CQTC** (Concentration-Q Trend Component) — the change attributable to shifts in the
  concentration-discharge relationship
- **QTC** (Q Trend Component) — the change attributable to shifts in the discharge
  distribution itself

## Typical Workflow

```
Load data → WRTDS() → fit() → kalman() → trends → plots
```

1. Load daily discharge and water-quality sample data
2. Create a `WRTDS` object
3. Call `fit()` to run the full model pipeline
4. Optionally call `kalman()` for improved daily estimates
5. Compute trends with `run_pairs()`, `run_groups()`, or `run_series()`
6. Generate plots and summary tables

See the [Quickstart](../getting-started/quickstart.md) for a worked example.
