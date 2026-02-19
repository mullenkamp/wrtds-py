# Fitting a Model

## The `fit()` Pipeline

Calling `w.fit()` runs three steps in sequence:

1. **Cross-validation** — leave-one-out jackknife to add `yHat`, `SE`, `ConcHat` to the Sample
2. **Surface estimation** — fit the censored regression at every grid point
3. **Daily estimation** — interpolate surfaces onto the daily record and flow-normalise

```python
w = WRTDS(daily, sample)
w.fit()
```

All three steps use the same window parameters, which can be customised:

```python
w.fit(
    window_y=7.0,     # Time half-window in years
    window_q=2.0,     # Discharge half-window in log units
    window_s=0.5,     # Season half-window in years
    min_num_obs=100,   # Minimum total observations per regression
    min_num_uncen=50,  # Minimum uncensored observations per regression
    edge_adjust=True,  # Double time window near record boundaries
)
```

## Window Parameters

### Time Window (`window_y`)

Controls how much weight is given to observations from other years. The default of
7 years means observations 7+ years away from the target get zero weight. Larger values
produce smoother trends.

### Discharge Window (`window_q`)

Controls sensitivity to flow conditions. The default of 2 log units means the model
adapts to flow-dependent patterns in concentration. Smaller values make the model more
flow-sensitive.

### Season Window (`window_s`)

Controls seasonal smoothing. The default of 0.5 years gives zero weight to observations
from the opposite season.

### Minimum Observations

If a target grid point does not have enough observations within its windows, all three
windows are expanded by 10% iteratively until `min_num_obs` (total) and `min_num_uncen`
(uncensored) thresholds are met.

### Edge Adjustment

When `edge_adjust=True`, the time window is doubled minus the distance to the nearest
record boundary for target points near the start or end of the record. This reduces
edge bias.

## Calling Sub-Steps Individually

You can run the pipeline steps separately for more control:

```python
w = WRTDS(daily, sample)

# Step 1: Cross-validation only
w.cross_validate()

# Step 2: Surface estimation (independent of cross-validation)
w.estimate_surfaces()

# Step 3: Daily estimation + flow normalization (requires surfaces)
w.estimate_daily()
```

!!! warning
    `estimate_daily()` requires `estimate_surfaces()` to have been called first.
    The `fit()` method handles this ordering automatically.

## What `fit()` Produces

After fitting, the following columns are available:

### On `w.sample` (from cross-validation)

| Column | Description |
|--------|-------------|
| `yHat` | Predicted log-concentration (leave-one-out) |
| `SE` | Standard error of prediction |
| `ConcHat` | Bias-corrected predicted concentration |

### On `w.daily` (from daily estimation + flow normalization)

| Column | Description |
|--------|-------------|
| `yHat` | Predicted log-concentration |
| `SE` | Standard error |
| `ConcDay` | Daily estimated concentration (mg/L) |
| `FluxDay` | Daily estimated flux (kg/day) |
| `FNConc` | Flow-normalised concentration |
| `FNFlux` | Flow-normalised flux |

### On `w` (stored attributes)

| Attribute | Description |
|-----------|-------------|
| `surfaces` | 3-D numpy array of shape `(n_logq, n_year, 3)` |
| `surface_index` | Dict with grid parameters (`bottom_logq`, `top_logq`, `step_logq`, etc.) |

## Performance

Surface estimation is the computational bottleneck — it requires solving one MLE
regression per grid point (typically ~7,000 solves). Cross-validation adds another
n solves (one per sample observation). A dataset with 500 samples and 30 years of
daily data typically takes a few minutes to fit.
