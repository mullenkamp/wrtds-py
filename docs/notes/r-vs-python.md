# R vs Python Differences

This page documents known numerical differences between wrtds-py and the R EGRET package.
These differences arise from implementation choices in date arithmetic, grid construction,
and numerical optimization. **None of these differences are large enough to affect
scientific conclusions.**

## 1. Decimal Year (DecYear)

**Source:** R's bundled Choptank `eList` dataset uses an older DecYear formula that is
affected by daylight saving time (DST) transitions. Python's `decimal_date()` is
timezone-naive and computes:

$$
\text{DecYear} = \text{year} + \frac{\text{date} - \text{Jan 1}}{\text{Jan 1 (next year)} - \text{Jan 1}}
$$

**Impact:** Relative difference of ~2 x 10^-6^ (about 1 minute per year). The seasonal
harmonics (`SinDY`, `CosDY`) amplify this slightly.

| Column | Tolerance | Notes |
|--------|-----------|-------|
| `daily.DecYear` | rtol = 1e-5 | ~2e-6 relative difference |
| `sample.DecYear` | rtol = 1e-5 | Same DST issue |
| `sample.SinDY` | atol = 0.02 | Amplified by sin() |
| `sample.CosDY` | atol = 0.025 | Amplified by cos() |
| `daily.LogQ` | rtol = 1e-10 | Near machine precision |
| `daily.Julian` | exact | Integer days |

## 2. Surface Grid Bounds

**Source:** R computes `bottomLogQ` and `topLogQ` from the **Daily** discharge range
(all daily Q values). Python computes them from the **Sample** LogQ range (only days
with water-quality observations).

**Impact:** The grid boundaries differ, so the surfaces arrays cannot be compared
element-by-element. However, structural parameters match:

| Parameter | Match |
|-----------|-------|
| `n_logq` | Exact (both = 14) |
| `n_year` | Exact |
| `step_year` | rtol = 1e-10 |
| `bottom_year`, `top_year` | Exact |

The downstream effect on daily estimates is small because the interpolation clamps
queries to grid boundaries in both implementations.

## 3. MLE Optimizer

**Source:** R uses `survival::survreg` (Newton-Raphson with profiled likelihood).
Python uses `scipy.optimize.minimize` with the L-BFGS-B method and analytical gradients.

**Impact:** Small coefficient differences that propagate through the pipeline. The
cross-validation step (which runs one MLE per sample observation) shows the largest
per-point differences; these average out in daily and annual summaries.

### Cross-Validation Tolerances

| Column | rtol | atol | Typical p95 Difference |
|--------|------|------|------------------------|
| `yHat` | 0.15 | 0.07 | ~9% |
| `SE` | 0.35 | — | ~3% (few outliers larger) |
| `ConcHat` | 0.10 | — | ~1.5% |

### Daily Estimation Tolerances

| Column | rtol | Typical p95 Difference |
|--------|------|------------------------|
| `ConcDay` | 0.30 | ~2% |
| `FluxDay` | 0.30 | ~2% |
| `FNConc` | 0.12 | ~1% |
| `FNFlux` | 0.25 | ~1% |

### Kalman (WRTDS-K) Tolerances

| Column | rtol | Notes |
|--------|------|-------|
| `GenConc` | 1.5 | Monte Carlo + model differences; median ~4% |
| `GenFlux` | 1.5 | Same |

The wider Kalman tolerances reflect both the MLE differences and the inherent Monte Carlo
variability.

### Annual Summary Tolerances

Annual means smooth out daily differences:

| Column | rtol |
|--------|------|
| `Q` | 1e-4 |
| `Conc` | 0.02 (2%) |
| `Flux` | 0.02 (2%) |
| `FNConc` | 0.01 (1%) |
| `FNFlux` | 0.005 (0.5%) |

### Trend Analysis Tolerances

| Component | Conc rtol | Flux rtol |
|-----------|-----------|-----------|
| Base values (x10, x11, x20, x22) | 1--5% | 2--5% |
| TotalChange, CQTC | 2--5% | 5--20% |
| Groups base values | 1% | 2% |
| Groups TotalChange, CQTC | 2% | 5% |

Flux trend tolerances are wider because flux = concentration x discharge, so relative
errors compound.

## Summary

All differences originate from three root causes (DecYear formula, grid bounds,
optimizer) and propagate predictably. The largest differences occur in per-point
cross-validation estimates (~9% for individual `yHat` values), but these average down
to 1--2% at the daily level and < 1% at the annual level. Trend conclusions are
robust to these differences.

!!! tip
    The full tolerance table is defined programmatically in
    `wrtds/tests/test_vs_r.py`. Run the R comparison tests with
    `uv run pytest -m r_fixtures` (requires R, EGRET, and jsonlite).
