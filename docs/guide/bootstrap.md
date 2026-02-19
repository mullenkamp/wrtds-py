# Bootstrap Confidence Intervals

Bootstrap methods provide confidence intervals and p-values for trend estimates.
The implementation follows:

> Hirsch, R.M., Archfield, S.A., and De Cicco, L.A. (2015), A bootstrap method for
> estimating uncertainty of water quality trends. *Environmental Modelling & Software*,
> 73, 148-166.

## Bootstrap Pairs

Provides confidence intervals for pairwise trend comparisons:

```python
boot = w.bootstrap_pairs(
    year1=1985,
    year2=2010,
    n_boot=100,
    block_length=200,
    seed=42,
)
```

### What It Does

For each bootstrap replicate:

1. **Block resample** the Sample DataFrame (preserving temporal structure)
2. **Estimate two one-year surfaces** (one for each target year)
3. **Run the trend decomposition** (same as `run_pairs`)
4. **Apply bias correction** to account for systematic differences between the
   bootstrap and original surfaces

### Results

The returned dictionary contains:

| Key | Description |
|-----|-------------|
| `observed` | Original `run_pairs` result |
| `boot_conc` | Array of bootstrap concentration trend replicates |
| `boot_flux` | Array of bootstrap flux trend replicates |
| `p_conc` | Two-sided p-value for concentration trend |
| `p_flux` | Two-sided p-value for flux trend |
| `ci_conc` | 90% confidence interval for concentration change |
| `ci_flux` | 90% confidence interval for flux change |
| `likelihood_conc_up` | Probability that concentration increased |
| `likelihood_flux_up` | Probability that flux increased |
| `like_conc_up` | Likelihood descriptor for concentration increase |
| `like_conc_down` | Likelihood descriptor for concentration decrease |
| `like_flux_up` | Likelihood descriptor for flux increase |
| `like_flux_down` | Likelihood descriptor for flux decrease |

## Bootstrap Groups

Same approach applied to group comparisons:

```python
boot = w.bootstrap_groups(
    group1_years=(1985, 1996),
    group2_years=(1997, 2010),
    n_boot=100,
    seed=42,
)
```

For group bootstrap, each replicate re-estimates the **full surface** (not just
per-year surfaces), then runs the group trend decomposition.

## Block Resampling

The `block_length` parameter (default 200 Julian days) defines the size of temporal
blocks drawn with replacement. Block resampling preserves the temporal correlation
structure of the sample data while allowing statistical inference.

## Likelihood Descriptors

The `likelihood_descriptor` function maps a probability to a qualitative description
using EGRETci thresholds:

| Probability Range | Descriptor |
|-------------------|------------|
| 0.0 -- 0.05 | Highly unlikely |
| 0.05 -- 0.1 | Very unlikely |
| 0.1 -- 0.33 | Unlikely |
| 0.33 -- 0.67 | About as likely as not |
| 0.67 -- 0.9 | Likely |
| 0.9 -- 0.95 | Very likely |
| 0.95 -- 1.0 | Highly likely |

## Number of Replicates

The `n_boot` parameter controls how many bootstrap replicates are drawn. More replicates
give more stable p-values and confidence intervals. Typical values:

- **100** — quick exploratory analysis
- **500--1000** — publication-quality results

Each replicate requires re-fitting the surfaces, so computation time scales linearly
with `n_boot`.
