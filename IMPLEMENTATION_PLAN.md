# WRTDS Python Implementation Plan

## Context

The USGS WRTDS (Weighted Regressions on Time, Discharge, and Season) method is implemented in R as the EGRET package. This plan transcribes it to Python with a class-based API, using pandas DataFrames as the core data structure and scipy for the censored regression engine.

**Full scope**: Core WRTDS, WRTDS-K (Kalman), trend analysis (EGRET 3.0), bootstrap CI, and matplotlib plots.

**Design decisions**:
- Class-based API (`WRTDS` class with methods like `.fit()`, `.kalman()`)
- No USGS data retrieval — accepts user-supplied DataFrames only
- Custom censored Gaussian MLE via `scipy.optimize.minimize` (replaces R's `survival::survreg`)
- pandas DataFrames for Daily/Sample, numpy 3D array for surfaces, dict for metadata

---

## WRTDS Algorithm Overview

### Core Regression

The model fits a weighted regression on ln(concentration):

```
ln(c) = B0 + B1*t + B2*ln(Q) + B3*sin(2*pi*t) + B4*cos(2*pi*t) + epsilon
```

Where:
- `c` = concentration (mg/L)
- `Q` = discharge (m³/s)
- `t` = time in decimal years
- `sin/cos` terms = seasonal component (annual harmonic)

**Coefficients are NOT fixed globally.** They are re-estimated at every estimation point using locally weighted regression with **tricube weights** in 3 dimensions:

| Dimension | Distance Measure | Default Half-Window |
|-----------|-----------------|---------------------|
| Time | `\|t_i - t_0\|` in years | 7 years |
| Discharge | `\|ln(Q_i) - ln(Q_0)\|` | 2 log units |
| Season | Circular fractional-year distance | 0.5 years |

Total weight for observation i = `w_time * w_discharge * w_season`

**Tricube function**: `w(d, h) = (1 - |d/h|³)³` when `|d| < h`, else 0

**Censored data**: Uses censored Gaussian MLE (Tobit regression) to handle below-detection-limit observations. For uncensored data, `ConcLow == ConcHigh`. For left-censored, `ConcLow = NaN`, `ConcHigh = detection_limit`.

**Bias correction** (lognormal retransformation): `ConcHat = exp(yHat + SE²/2)`

**Flux**: `FluxDay = ConcDay * Q * 86.4` (kg/day)

### Flow Normalization

For each day `t`: average the model's concentration estimate across ALL historical discharge values observed on that calendar day.

```
FNConc(t) = mean(ConcHat(Q_j, t)) for all historical Q_j on this day-of-year
FNFlux(t) = mean(ConcHat(Q_j, t) * Q_j * 86.4)
```

This removes the influence of year-to-year discharge variability, isolating the water quality trend.

### Surfaces Grid

A 3D numpy array `[14 LogQ levels × nTimeSteps × 3 layers]`:
- Layer 0: `yHat` (predicted ln(concentration))
- Layer 1: `SE` (standard error / scale parameter)
- Layer 2: `ConcHat` (bias-corrected concentration)

Grid spacing: 14 LogQ levels, time steps at 1/16 year (~23 days). Daily values are obtained via bilinear interpolation on this grid.

---

## Module Structure

```
wrtds/
├── __init__.py            # Version, public API exports
├── core.py                # WRTDS class (main entry point)
├── regression.py          # Censored Gaussian MLE, tricube weights
├── surfaces.py            # Surface grid estimation, bilinear interpolation
├── flow_norm.py           # Flow normalization (standard + generalized)
├── cross_val.py           # Leave-one-out cross-validation
├── kalman.py              # WRTDS-K (AR(1) residual interpolation)
├── trends.py              # runPairs, runGroups, runSeries
├── bootstrap.py           # Block bootstrap confidence intervals
├── data_prep.py           # DataFrame validation, column population, utilities
├── plots/
│   ├── __init__.py
│   ├── data_overview.py   # Raw data plots (Q time series, conc vs Q, box plots)
│   ├── diagnostics.py     # Residual plots, flux bias, observed vs predicted
│   ├── results.py         # Trend histories, contours, daily time series
│   └── utils.py           # Shared plot helpers (axis formatting, etc.)
└── tests/
    ├── __init__.py
    ├── conftest.py              # Fixture loading, --regenerate-fixtures flag
    ├── fixtures/
    │   ├── README.md            # Documents fixture generation and when to regenerate
    │   ├── generate_fixtures.py # rpy2 script: runs R EGRET, exports results as parquet/npy
    │   └── *.parquet, *.npy     # Cached R outputs (committed to repo)
    ├── test_regression.py
    ├── test_surfaces.py
    ├── test_flow_norm.py
    ├── test_cross_val.py
    ├── test_kalman.py
    ├── test_trends.py
    ├── test_bootstrap.py
    ├── test_data_prep.py
    └── test_plots.py
```

## Dependencies

```toml
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
]
```

---

## WRTDS Class API (`core.py`)

```python
class WRTDS:
    # --- Attributes ---
    daily: pd.DataFrame       # Daily discharge record
    sample: pd.DataFrame      # Water quality samples
    info: dict                # Site/parameter metadata
    surfaces: np.ndarray      # 3D array [nLogQ x nTime x 3] after fit()
    surface_index: dict       # Grid params (bottom_logq, step_logq, bottom_year, step_year, ...)

    # --- Constructor ---
    def __init__(self, daily, sample, info=None):
        """Validate and prepare DataFrames, merge Q into sample."""

    # --- Core WRTDS ---
    def fit(self, window_y=7, window_q=2, window_s=0.5,
            min_num_obs=100, min_num_uncen=50, edge_adjust=True) -> 'WRTDS':
        """Run full WRTDS: cross-validation, surface estimation, daily estimation."""

    def cross_validate(self, ...) -> 'WRTDS':
        """Leave-one-out cross-validation, populates sample[yHat, SE, ConcHat]."""

    def estimate_surfaces(self, ...) -> 'WRTDS':
        """Fit regression at each grid point, populates self.surfaces."""

    def estimate_daily(self) -> 'WRTDS':
        """Interpolate from surfaces to get daily ConcDay, FluxDay, FNConc, FNFlux."""

    # --- WRTDS-K ---
    def kalman(self, rho=0.90, n_iter=200) -> 'WRTDS':
        """Run WRTDS-K. Populates daily[GenConc, GenFlux]."""

    # --- Trend Analysis ---
    def run_pairs(self, year1, year2, window_side=7) -> pd.DataFrame:
        """Compare flow-normalized values between two years."""

    def run_groups(self, group1_years, group2_years, window_side=7) -> pd.DataFrame:
        """Compare averages across two multi-year groups."""

    def run_series(self, window_side=7, wall=False, sample1_end_date=None) -> 'WRTDS':
        """Annual series of flow-normalized conc and flux."""

    # --- Bootstrap CI ---
    def bootstrap_pairs(self, year1, year2, n_boot=100,
                        block_length=200, window_side=7) -> dict:
        """Block bootstrap CI for pairwise trend comparison."""

    def bootstrap_groups(self, group1_years, group2_years,
                         n_boot=100, block_length=200, window_side=7) -> dict:
        """Block bootstrap CI for group trend comparison."""

    # --- Summaries ---
    def table_results(self, pa_start=10, pa_long=12) -> pd.DataFrame:
        """Annual summary table (Q, Conc, FNConc, Flux, FNFlux)."""

    def monthly_results(self) -> pd.DataFrame:
        """Monthly means."""

    def error_stats(self) -> dict:
        """Cross-validation error statistics."""

    def flux_bias_stat(self) -> float:
        """Flux bias statistic."""

    # --- Plots (delegate to wrtds.plots.*) ---
    def plot_overview(self, ...): ...
    def plot_conc_q(self, ...): ...
    def plot_conc_time(self, ...): ...
    def plot_flux_hist(self, ...): ...
    def plot_conc_hist(self, ...): ...
    def plot_contours(self, ...): ...
    def plot_residuals(self, ...): ...
    def plot_conc_pred(self, ...): ...
    def plot_flux_pred(self, ...): ...
```

All mutating methods return `self` for chaining:
```python
model = WRTDS(daily, sample).fit().kalman()
model.plot_conc_hist()
```

---

## Implementation Phases

---

### Phase 1: Data Preparation (`data_prep.py`)

**Goal**: Validate input DataFrames, compute derived columns, merge discharge into samples.

#### Daily DataFrame

Required input columns: `Date`, `Q` (m³/s).

Computed columns:

| Column | Computation |
|--------|-------------|
| `LogQ` | `np.log(Q)` |
| `Julian` | Days since 1850-01-01 |
| `DecYear` | Decimal year (e.g. 2005.5 = July 1, 2005). Fraction = day-of-year / days-in-year |
| `Month` | 1-12 |
| `Day` | Day of year 1-366 |
| `MonthSeq` | Months since 1850-01-01: `(year - 1850) * 12 + month` |

#### Sample DataFrame

Required input columns: `Date`, `ConcLow`, `ConcHigh`.

Alternative input: `Date`, `Conc`, `Remark` (where `Remark = '<'` means censored). Convert to ConcLow/ConcHigh format.

Computed columns:

| Column | Computation |
|--------|-------------|
| `Uncen` | 1 if ConcLow == ConcHigh, else 0 |
| `ConcAve` | `(ConcLow + ConcHigh) / 2` (treat NaN ConcLow as 0) |
| `Julian`, `DecYear`, `Month`, `Day`, `MonthSeq` | Same as Daily |
| `SinDY` | `sin(2 * pi * DecYear)` |
| `CosDY` | `cos(2 * pi * DecYear)` |
| `Q`, `LogQ` | Merged from Daily by nearest Date |

#### Censoring convention

- **Uncensored**: `ConcLow == ConcHigh` (exact observation)
- **Left-censored** (below detection): `ConcLow = NaN`, `ConcHigh = detection_limit`
- **Interval-censored**: `ConcLow < ConcHigh` (known to be in an interval)

#### Info dict

Optional metadata with defaults:

```python
{
    'station_name': '',
    'param_name': '',
    'drainage_area_km2': None,
    'pa_start': 10,          # Period of analysis start month (10 = October = water year)
    'pa_long': 12,           # Period of analysis length in months
}
```

#### Functions to implement

```python
def populate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, compute derived columns."""

def populate_sample(sample: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Validate, compute derived columns, merge Q from daily."""

def compress_data(sample: pd.DataFrame) -> pd.DataFrame:
    """Convert Conc + Remark format to ConcLow/ConcHigh/Uncen."""

def decimal_date(dates: pd.Series) -> pd.Series:
    """Convert datetime to decimal year."""
```

#### Tests (`test_data_prep.py`)

- Validate that missing required columns raise clear errors
- Verify Julian, DecYear, MonthSeq computations against known values
- Verify Q merge works correctly (nearest date matching)
- Verify censoring convention conversion
- Edge cases: leap years, year boundaries

---

### Phase 2: Regression Engine (`regression.py`)

**Goal**: Implement the weighted censored Gaussian MLE and tricube weight functions.

#### Tricube weights

```python
def tricube(d: np.ndarray, h: float) -> np.ndarray:
    """Tricube weight: (1 - |d/h|³)³ for |d| < h, else 0."""
    u = np.abs(d / h)
    return np.where(u < 1.0, (1.0 - u**3)**3, 0.0)


def compute_weights(
    dec_year: np.ndarray,
    logq: np.ndarray,
    uncen: np.ndarray,
    target_dec_year: float,
    target_logq: float,
    window_y: float,
    window_q: float,
    window_s: float,
    min_num_obs: int = 100,
    min_num_uncen: int = 50,
    edge_adjust: bool = True,
    record_start: float = None,
    record_end: float = None,
) -> np.ndarray:
    """Compute product of three tricube weights (time, discharge, season).

    Time distance: |dec_year_i - target_dec_year|

    Season distance (circular):
        diff = dec_year_i - target_dec_year
        frac = diff - round(diff)
        season_dist = abs(frac)

    Edge adjustment: if target is within window_y of record start/end,
    expand time window: adjusted_window = 2*window_y - dist_to_edge

    Minimum sample safeguard: if fewer than min_num_obs observations
    have nonzero weight, or fewer than min_num_uncen uncensored
    observations have nonzero weight, expand ALL three windows by
    10% and recompute. Repeat until satisfied.
    """
```

#### Censored Gaussian MLE

This replaces R's `survival::survreg(Surv(ConcLow, ConcHigh, type="interval2") ~ DecYear + LogQ + SinDY + CosDY, weights=w, dist="gaussian")`.

Parameters to optimize: `theta = [B0, B1, B2, B3, B4, log_sigma]` (6 parameters).

```python
def neg_log_likelihood_with_grad(theta, X, y_uncen, y_cen_high, uncen_mask, weights):
    """Negative log-likelihood AND analytical gradient for weighted censored Gaussian regression.

    Returning both cost and gradient in a single function avoids numerical
    differentiation. With 6 parameters, this eliminates ~6 extra function
    evaluations per optimizer iteration — a 5-10x speedup across the ~7500
    MLE solves in surface estimation + cross-validation.

    Parameters:
        theta: [B0, B1, B2, B3, B4, log_sigma] (6 parameters)
        X: design matrix (n, 5) — [1, DecYear, LogQ, SinDY, CosDY]
        y_uncen: log(ConcAve) for uncensored obs (NaN for censored)
        y_cen_high: log(ConcHigh) for all obs
        uncen_mask: boolean array, True if uncensored
        weights: tricube weights (n,)

    Returns:
        cost: scalar, negative log-likelihood
        grad: (6,) array, gradient of NLL w.r.t. theta

    Log-likelihood terms:
        Uncensored obs i:
            w_i * [-0.5*((y_i - X_i@beta)/sigma)² - 0.5*log(2*pi) - log(sigma)]

        Left-censored obs i:
            w_i * log(Phi((log(ConcHigh_i) - X_i @ beta) / sigma))

        where phi = normal PDF, Phi = normal CDF

    Gradient derivation (NLL = negative log-likelihood):

        Let z_i = (y_i - X_i @ beta) / sigma for uncensored,
            z_i = (y_high_i - X_i @ beta) / sigma for censored.
        Let lambda(z) = phi(z) / Phi(z)  (inverse Mills ratio).

        Uncensored:
            dNLL/d(beta) = -sum_uncen w_i * (y_i - X_i@beta) / sigma² * X_i
            dNLL/d(ln_sigma) = sum_uncen w_i * (1 - z_i²)

        Censored:
            dNLL/d(beta) = sum_cen w_i * lambda(z_i) * X_i / sigma
            dNLL/d(ln_sigma) = sum_cen w_i * lambda(z_i) * z_i
    """
    beta = theta[:-1]
    log_sigma = theta[-1]
    sigma = np.exp(log_sigma)

    # --- Cost (NLL) ---
    # Uncensored
    resid_uncen = y_uncen[uncen_mask] - X[uncen_mask] @ beta
    z_uncen = resid_uncen / sigma
    nll_uncen = 0.5 * z_uncen**2 + log_sigma + 0.5 * np.log(2 * np.pi)

    # Censored
    z_cen = (y_cen_high[~uncen_mask] - X[~uncen_mask] @ beta) / sigma
    log_cdf_cen = stats.norm.logcdf(z_cen)
    nll_cen = -log_cdf_cen

    cost = np.sum(weights[uncen_mask] * nll_uncen) + np.sum(weights[~uncen_mask] * nll_cen)

    # --- Gradient of NLL ---
    grad = np.zeros_like(theta)

    # Inverse Mills ratio for censored obs: lambda(z) = phi(z)/Phi(z)
    # Computed in log-space for numerical stability
    log_pdf_cen = stats.norm.logpdf(z_cen)
    mills = np.exp(log_pdf_cen - log_cdf_cen)

    # dNLL/d(beta)
    grad_beta_uncen = -weights[uncen_mask, None] * (resid_uncen[:, None] / sigma**2) * X[uncen_mask]
    grad_beta_cen = weights[~uncen_mask, None] * mills[:, None] * (X[~uncen_mask] / sigma)
    grad[:-1] = np.sum(grad_beta_uncen, axis=0) + np.sum(grad_beta_cen, axis=0)

    # dNLL/d(ln_sigma)
    grad_sigma_uncen = weights[uncen_mask] * (1 - z_uncen**2)
    grad_sigma_cen = weights[~uncen_mask] * mills * z_cen
    grad[-1] = np.sum(grad_sigma_uncen) + np.sum(grad_sigma_cen)

    return cost, grad


def run_surv_reg(
    sample_data: dict,  # Arrays: DecYear, LogQ, SinDY, CosDY, ConcLow, ConcHigh, Uncen
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run one weighted censored Gaussian regression.

    1. Build design matrix X = [1, DecYear, LogQ, SinDY, CosDY]
    2. Get starting values: weighted OLS on uncensored observations
       (fallback: unweighted OLS if insufficient weighted uncensored data)
       Initial log_sigma = log(std(residuals))
    3. Minimize using scipy.optimize.minimize(method='L-BFGS-B', jac=True)
       (jac=True tells scipy the function returns (cost, gradient) tuple)
    4. If convergence fails, apply jitter and retry (see jitter_sample below)

    Returns:
        beta: (5,) regression coefficients
        sigma: scalar, scale parameter (exp(log_sigma))
    """


def predict(beta, sigma, dec_year, logq):
    """Predict at a single point.

    yHat = beta[0] + beta[1]*dec_year + beta[2]*logq
           + beta[3]*sin(2*pi*dec_year) + beta[4]*cos(2*pi*dec_year)
    SE = sigma
    ConcHat = exp(yHat + SE² / 2)    # lognormal bias correction

    Returns: yHat, SE, ConcHat
    """


def jitter_sample(conc_low, conc_high, scale=0.01):
    """Add small random jitter to concentration bounds.
    Matches R's jitterSam — used when MLE fails to converge.
    Multiplies concentrations by exp(Normal(0, scale)).
    """
```

#### Tests (`test_regression.py`)

- `tricube`: Verify w(0, h)=1, w(h, h)=0, w(h*0.5, h) matches formula, negative d works
- `compute_weights`: Verify product of three dimensions, edge adjustment behavior, window expansion
- `neg_log_likelihood_with_grad`: Compare cost against hand-computed values for a tiny dataset
- `neg_log_likelihood_with_grad`: Verify analytical gradient matches scipy.optimize.approx_fprime (numerical gradient) within rtol=1e-5
- `run_surv_reg`: Compare coefficients against R `survreg` output for a known weighted dataset
- `predict`: Verify bias correction formula

---

### Phase 3: Surface Estimation (`surfaces.py`)

**Goal**: Fit the model at every grid point to build the 3D surfaces array.

#### Grid definition

```python
def compute_surface_index(sample: pd.DataFrame) -> dict:
    """Compute grid parameters from sample data.

    Returns dict:
        bottom_logq: min(LogQ) - 0.05
        top_logq: max(LogQ) + 0.05
        step_logq: (top_logq - bottom_logq) / 13
        n_logq: 14 (fixed)
        bottom_year: floor(min(DecYear))
        top_year: ceil(max(DecYear))
        step_year: 1/16 (~23 days)
        n_year: round((top_year - bottom_year) / step_year) + 1
        logq_grid: np.linspace(bottom_logq, top_logq, 14)
        year_grid: np.linspace(bottom_year, top_year, n_year)
    """
```

#### Surface estimation

```python
def estimate_surfaces(
    sample: pd.DataFrame,
    surface_index: dict,
    window_y: float = 7,
    window_q: float = 2,
    window_s: float = 0.5,
    min_num_obs: int = 100,
    min_num_uncen: int = 50,
    edge_adjust: bool = True,
) -> np.ndarray:
    """Fit regression at each (logq, year) grid point.

    surfaces shape: (n_logq, n_year, 3)
        [:, :, 0] = yHat
        [:, :, 1] = SE
        [:, :, 2] = ConcHat

    Algorithm:
        for j in range(n_logq):             # 14 discharge levels
            for k in range(n_year):           # ~500 time steps
                logq_j = logq_grid[j]
                year_k = year_grid[k]

                # 1. Compute tricube weights centered at (year_k, logq_j)
                w = compute_weights(sample, year_k, logq_j, ...)

                # 2. Run censored regression with these weights
                beta, sigma = run_surv_reg(sample, w)

                # 3. Predict at (year_k, logq_j)
                yHat, SE, ConcHat = predict(beta, sigma, year_k, logq_j)

                # 4. Store
                surfaces[j, k, :] = [yHat, SE, ConcHat]

    Total: ~7000 MLE solves. This is the computational bottleneck.
    Start single-threaded; add optional parallel execution later.
    """
```

#### Bilinear interpolation

```python
def make_interpolator(surfaces, surface_index, layer=2):
    """Create a RegularGridInterpolator for a surface layer.

    Uses scipy.interpolate.RegularGridInterpolator.
    bounds_error=False, fill_value=None (extrapolate by nearest).
    """


def interpolate_surface(
    surfaces: np.ndarray,
    surface_index: dict,
    logq: np.ndarray,
    dec_year: np.ndarray,
    layer: int = 2,
) -> np.ndarray:
    """Vectorized bilinear interpolation on the surfaces grid.

    Clamps logq and dec_year to grid boundaries before interpolation.
    Returns array of interpolated values.
    """
```

#### Tests (`test_surfaces.py`)

- `compute_surface_index`: Verify grid params for known sample data
- `estimate_surfaces`: Compare a few grid points against R EGRET output
- `interpolate_surface`: Verify interpolation at grid points returns exact values, between grid points matches expected bilinear result

---

### Phase 4: Daily Estimation & Flow Normalization (`flow_norm.py`)

**Goal**: Use the surfaces grid to estimate daily values and compute flow-normalized concentrations/fluxes.

#### Daily estimation

```python
def estimate_daily(
    daily: pd.DataFrame,
    surfaces: np.ndarray,
    surface_index: dict,
) -> pd.DataFrame:
    """Interpolate from surfaces for each day in the daily record.

    For each day:
        yHat = interpolate_surface(surfaces, idx, LogQ, DecYear, layer=0)
        SE = interpolate_surface(surfaces, idx, LogQ, DecYear, layer=1)
        ConcDay = interpolate_surface(surfaces, idx, LogQ, DecYear, layer=2)
        FluxDay = ConcDay * Q * 86.4

    Adds columns: yHat, SE, ConcDay, FluxDay
    """
```

#### Flow normalization

```python
def bin_qs(daily: pd.DataFrame) -> dict:
    """Group all historical LogQ values by day-of-year (1-366).

    Leap day handling: merge day 60 (Feb 29) into day 59 (Feb 28).
    Returns: {day_of_year: np.array of LogQ values}
    """


def flow_normalize(
    daily: pd.DataFrame,
    surfaces: np.ndarray,
    surface_index: dict,
    q_bins: dict,
) -> pd.DataFrame:
    """Compute flow-normalized concentration and flux for each day.

    For each day t in daily:
        1. Get historical Q distribution for this calendar day: q_bins[day_of_year]
        2. For each historical LogQ_j, interpolate ConcHat(LogQ_j, DecYear_t) from surfaces
        3. FNConc(t) = mean(ConcHat across all historical Q_j)
        4. FNFlux(t) = mean(ConcHat(Q_j, t) * exp(LogQ_j) * 86.4)

    Vectorized approach (critical for performance):
        1. Build arrays of all (LogQ_j, DecYear_t) pairs across all days
        2. Single call to RegularGridInterpolator
        3. Reshape and compute means per day using np.add.reduceat or similar

    Adds columns: FNConc, FNFlux
    """
```

#### Tests (`test_flow_norm.py`)

- `bin_qs`: Verify grouping, leap day merge
- `estimate_daily`: Compare ConcDay, FluxDay against R for known dataset
- `flow_normalize`: Compare FNConc, FNFlux against R for known dataset
- Performance: Verify vectorized approach handles 15,000+ days without timeout

---

### Phase 5: Cross-Validation (`cross_val.py`)

**Goal**: Leave-one-out jack-knife cross-validation.

```python
def cross_validate(
    sample: pd.DataFrame,
    window_y: float = 7,
    window_q: float = 2,
    window_s: float = 0.5,
    min_num_obs: int = 100,
    min_num_uncen: int = 50,
    edge_adjust: bool = True,
) -> pd.DataFrame:
    """Leave-one-out cross-validation.

    For each sample i (i = 0, ..., n-1):
        1. Create sample_loo = sample excluding row i
        2. Compute weights centered at (DecYear_i, LogQ_i) using sample_loo
        3. Run censored regression on sample_loo with those weights
        4. Predict at sample i's covariates:
            yHat_i = X_i @ beta
            SE_i = sigma
            ConcHat_i = exp(yHat_i + SE_i² / 2)

    Adds columns to sample: yHat, SE, ConcHat

    Total: n MLE solves (typically 200-1000 samples).
    """
```

#### Tests (`test_cross_val.py`)

- Compare yHat, SE, ConcHat for selected samples against R EGRET output
- Verify that removing each sample actually excludes it (no data leakage)

---

### Phase 6: WRTDS Class Integration (`core.py`)

**Goal**: Wire everything together into the `WRTDS` class with a working `.fit()` method.

```python
class WRTDS:
    def __init__(self, daily: pd.DataFrame, sample: pd.DataFrame, info: dict = None):
        """
        1. Call populate_daily(daily) -> self.daily
        2. Call populate_sample(sample, self.daily) -> self.sample
        3. Set self.info with defaults
        4. Initialize self.surfaces = None, self.surface_index = None
        """

    def fit(self, window_y=7, window_q=2, window_s=0.5,
            min_num_obs=100, min_num_uncen=50, edge_adjust=True) -> 'WRTDS':
        """Full WRTDS estimation pipeline:
        1. self.cross_validate(...)     -> populates self.sample[yHat, SE, ConcHat]
        2. self.estimate_surfaces(...)  -> populates self.surfaces, self.surface_index
        3. self.estimate_daily()        -> populates self.daily[yHat, SE, ConcDay, FluxDay, FNConc, FNFlux]
        Returns self for chaining.
        """

    # Each sub-step is also callable individually:
    def cross_validate(self, ...) -> 'WRTDS': ...
    def estimate_surfaces(self, ...) -> 'WRTDS': ...
    def estimate_daily(self) -> 'WRTDS': ...
```

#### Milestone check

At this point, the core WRTDS should be fully functional:
```python
model = WRTDS(daily_df, sample_df).fit()
print(model.daily[['Date', 'ConcDay', 'FluxDay', 'FNConc', 'FNFlux']])
```

Compare `model.table_results()` against R EGRET `tableResults()` for the Choptank River chloride example dataset.

---

### Phase 7: WRTDS-K — Kalman Extension (`kalman.py`)

**Goal**: AR(1) residual interpolation between sampled days, with Monte Carlo for censored data.

```python
def make_augmented_sample(
    sample: pd.DataFrame,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """Generate random concentration values for censored observations.

    For censored obs (Uncen == 0):
        Draw from truncated normal in log-space:
            mean = yHat, std = SE
            upper bound = ln(ConcHigh)
        Using scipy.stats.truncnorm

    Sets rObserved = drawn value (for this iteration).
    For uncensored obs: rObserved = ConcAve.
    """


def ar1_conditional_draw(
    rho: float,
    n_gap: int,
    e_start: float,
    e_end: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw AR(1) residuals for a gap, conditioned on endpoints.

    1. Build AR(1) correlation matrix: R[i,j] = rho^|i-j|
       Size: (n_gap + 2) x (n_gap + 2) — includes endpoints
    2. Partition into observed (endpoints) and unobserved (gap interior)
    3. Compute conditional mean and covariance:
       mu_cond = mu_2 + Sigma_21 @ Sigma_11^{-1} @ (x_obs - mu_1)
       Sigma_cond = Sigma_22 - Sigma_21 @ Sigma_11^{-1} @ Sigma_12
    4. Cholesky decompose Sigma_cond
    5. Draw: e_gap = mu_cond + L @ z, where z ~ N(0, I)

    Returns: (n_gap,) array of residuals for the gap interior.
    """


def wrtds_kalman(
    daily: pd.DataFrame,
    sample: pd.DataFrame,
    surfaces: np.ndarray,
    surface_index: dict,
    rho: float = 0.90,
    n_iter: int = 200,
    seed: int = None,
) -> pd.DataFrame:
    """Run WRTDS-K.

    For each Monte Carlo iteration (1..n_iter):
        1. Call make_augmented_sample to draw censored values
        2. Compute standardized residuals at each sample day:
            e_i = (ln(rObserved_i) - yHat_i) / SE_i
        3. For each gap between consecutive sample days:
            a. Call ar1_conditional_draw to get gap residuals
            b. For days outside sample range: use unconditional AR(1) decay
        4. For all days: ConcDay = exp(e * SE + yHat)
        5. FluxDay = ConcDay * Q * 86.4
        6. Accumulate FluxDay into running sum

    Final:
        GenFlux = accumulated_sum / n_iter
        GenConc = GenFlux / (Q * 86.4)

    Adds columns to daily: GenConc, GenFlux
    """
```

#### Key parameter guidance

| Constituent type | Recommended rho |
|-----------------|-----------------|
| Nitrate, chloride (conservative) | 0.95 |
| General/default | 0.90 |
| Sediment, phosphorus (reactive) | 0.85 |

`n_iter`: 200 for exploration, 500-1000 for publication.

#### Tests (`test_kalman.py`)

- `make_augmented_sample`: Verify draws are within bounds for censored data
- `ar1_conditional_draw`: Verify endpoints are preserved, covariance structure is correct
- `wrtds_kalman`: Compare GenConc/GenFlux against R with fixed seed
- Verify that uncensored samples produce deterministic residuals

---

### Phase 8: Trend Analysis (`trends.py`)

**Goal**: Generalized flow normalization and trend decomposition (EGRET 3.0 features).

#### Generalized flow normalization

Standard flow normalization uses ALL years' discharge for every estimation point. Generalized flow normalization restricts the discharge distribution to a window:

- **Standard (SFN)**: Q distribution from all years
- **Generalized with window_side**: Q distribution from `[year - window_side, year + window_side]`

```python
def bin_qs_windowed(
    daily: pd.DataFrame,
    center_year: float,
    window_side: float,
) -> dict:
    """Like bin_qs, but only includes Q values from
    years within [center_year - window_side, center_year + window_side].
    """


def flow_normalize_generalized(
    daily: pd.DataFrame,
    surfaces: np.ndarray,
    surface_index: dict,
    window_side: float = 7,
) -> pd.DataFrame:
    """Generalized flow normalization.

    For each year in the record:
        1. Build windowed Q bins (centered on this year)
        2. Compute FNConc, FNFlux using only that window's Q distribution

    Adds/updates columns: FNConc, FNFlux
    """
```

#### run_pairs

```python
def run_pairs(
    wrtds: 'WRTDS',
    year1: int,
    year2: int,
    window_side: float = 7,
) -> pd.DataFrame:
    """Compare flow-normalized values between two specific years.

    Steps:
        1. Compute surfaces for year1 and year2 (subset or full reuse)
        2. For each year, compute concentration/flux under:
           a. Stationary FN (all years' Q) -> x10, x20
           b. Generalized FN (windowed Q) -> x11, x22
        3. Decompose:
           - CQTC = x20 - x10 (concentration-discharge trend component)
           - QTC = TotalChange - CQTC (discharge trend component)
           - TotalChange = x22 - x11

    Returns DataFrame:
        Index: ['Conc', 'Flux']
        Columns: ['x10', 'x11', 'x20', 'x22', 'CQTC', 'QTC', 'Total']
    """
```

#### run_groups

```python
def run_groups(
    wrtds: 'WRTDS',
    group1_years: tuple,  # (first_year, last_year)
    group2_years: tuple,
    window_side: float = 7,
) -> pd.DataFrame:
    """Same decomposition as run_pairs but averaging across year groups.

    For each group, average the annual flow-normalized values across
    all years in the group.
    """
```

#### run_series

```python
def run_series(
    wrtds: 'WRTDS',
    window_side: float = 7,
    wall: bool = False,
    sample1_end_date: str = None,
) -> 'WRTDS':
    """Annual time series of generalized flow-normalized conc and flux.

    If wall=False: single continuous surface with windowed FN.
    If wall=True: fit two separate surfaces (before/after sample1_end_date),
                  stitch them at the boundary. Used for known interventions.

    Updates daily['FNConc'], daily['FNFlux'] with GFN values.
    """
```

#### Tests (`test_trends.py`)

- `run_pairs`: Compare full output table against R for known dataset
- `run_groups`: Compare against R
- `run_series`: Verify annual FNConc/FNFlux series matches R
- Wall/stitch: Verify discontinuity at boundary

---

### Phase 9: Bootstrap CI (`bootstrap.py`)

**Goal**: Block bootstrap for uncertainty estimation on trends.

```python
def block_resample(
    sample: pd.DataFrame,
    block_length: int = 200,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """Block bootstrap resample of the Sample DataFrame.

    1. Divide sample dates into blocks of block_length days
    2. Randomly select blocks with replacement until total >= n_samples
    3. Truncate to n_samples
    4. Return resampled DataFrame (may have repeated observations)

    Block bootstrap preserves temporal autocorrelation structure.
    """


def bootstrap_pairs(
    wrtds: 'WRTDS',
    year1: int,
    year2: int,
    n_boot: int = 100,
    block_length: int = 200,
    window_side: float = 7,
    seed: int = None,
) -> dict:
    """Block bootstrap CI for pairwise trend comparison.

    1. Compute observed trend: run_pairs(wrtds, year1, year2, window_side)
    2. For each bootstrap replicate b = 1..n_boot:
        a. Block-resample the Sample DataFrame
        b. Create new WRTDS with resampled sample and same daily
        c. Fit WRTDS on resampled data
        d. Run run_pairs on the resampled model
        e. Store trend estimates
    3. Compute:
        - p-value: fraction of bootstrap replicates with opposite sign to observed
        - Confidence intervals: percentile method (2.5%, 97.5%)
        - Likelihood descriptor (e.g., "highly likely increase")

    Returns dict:
        observed: DataFrame from run_pairs
        boot_distribution: array of bootstrap trend estimates
        p_conc: p-value for concentration change
        p_flux: p-value for flux change
        ci_conc: (lower, upper) 95% CI for concentration change
        ci_flux: (lower, upper) 95% CI for flux change
    """


def bootstrap_groups(wrtds, group1_years, group2_years, ...):
    """Same but for group comparisons."""
```

#### Likelihood descriptors (from EGRETci convention)

| p-value | Descriptor |
|---------|-----------|
| p <= 0.01 | Highly likely |
| 0.01 < p <= 0.05 | Very likely |
| 0.05 < p <= 0.1 | Likely |
| 0.1 < p <= 0.2 | Somewhat likely |
| p > 0.2 | About as likely as not |

#### Tests (`test_bootstrap.py`)

- `block_resample`: Verify block structure, correct total count
- `bootstrap_pairs`: Smoke test with small n_boot=5, verify output structure
- Verify p-value and CI computation logic with a mock distribution

---

### Phase 10: Summary Tables (`core.py` methods)

**Goal**: Implement annual and monthly summary methods.

```python
def setup_years(
    daily: pd.DataFrame,
    pa_start: int = 10,
    pa_long: int = 12,
) -> pd.DataFrame:
    """Create annual results DataFrame.

    pa_start: starting month (10 = October for water year, 1 = January for calendar year)
    pa_long: number of months (12 = full year, 1 = single month)

    For each "year" (defined by pa_start):
        - Filter daily to the pa_long months starting from pa_start
        - Compute means: Q, ConcDay, FNConc, FluxDay, FNFlux
        - If GenConc/GenFlux exist: include those means too
        - Convert flux to metric tons/year: FluxDay * 365.25 / 1000

    Columns: Year, Q, Conc, FNConc, Flux, FNFlux [, GenConc, GenFlux]
    """


def table_change(
    annual_results: pd.DataFrame,
    year_points: list,
) -> pd.DataFrame:
    """Changes in flow-normalized values between specified years.

    For each consecutive pair of years in year_points:
        - Change in FNConc (absolute and %)
        - Change in FNFlux (absolute and %)
        - Slope (change per year)
    """


def error_stats(sample: pd.DataFrame) -> dict:
    """Cross-validation error statistics.

    Requires sample to have yHat, ConcHat columns (from cross_validate).
    Computes:
        - Bias: mean(ln(ConcHat) - ln(ConcAve))
        - RMSE: sqrt(mean((ln(ConcHat) - ln(ConcAve))²))
        - Nash-Sutcliffe efficiency
        - Percent bias in flux
    """


def flux_bias_stat(sample: pd.DataFrame, daily: pd.DataFrame) -> float:
    """Flux bias statistic.

    (mean estimated flux - mean observed flux) / mean estimated flux
    Values near zero = good performance.
    """
```

---

### Phase 11: Plots (`plots/`)

**Goal**: Matplotlib implementations of key EGRET plots.

All plot functions take `ax=None` parameter — if None, create new figure; if provided, draw on existing axes. All return the matplotlib `Figure` object.

#### Data overview plots (`plots/data_overview.py`)

```python
def plot_q_time_daily(daily, ax=None):
    """Daily discharge time series. Log-scale y-axis."""

def plot_conc_time(sample, ax=None):
    """Observed concentration vs time.
    Open circles for censored, filled for uncensored."""

def plot_conc_q(sample, ax=None):
    """Observed concentration vs discharge (log-log axes)."""

def box_conc_month(sample, ax=None):
    """Concentration box plots by month."""

def box_q_twice(daily, sample, ax=None):
    """Side-by-side box plots: Q on all days vs Q on sample days."""

def plot_overview(daily, sample, fig=None):
    """4-panel overview combining the above plots."""
```

#### Diagnostic plots (`plots/diagnostics.py`)

```python
def plot_conc_pred(sample, ax=None):
    """Observed vs predicted concentration. 1:1 line for reference."""

def plot_flux_pred(sample, daily, ax=None):
    """Observed vs predicted flux."""

def plot_resid_pred(sample, ax=None):
    """Residuals (ln(observed) - yHat) vs predicted."""

def plot_resid_q(sample, ax=None):
    """Residuals vs log(discharge)."""

def plot_resid_time(sample, ax=None):
    """Residuals vs decimal year. LOWESS smooth overlaid."""

def box_resid_month(sample, ax=None):
    """Residual box plots by month."""

def flux_bias_multi(sample, daily, fig=None):
    """8-panel flux bias diagnostic."""
```

#### Result plots (`plots/results.py`)

```python
def plot_conc_hist(daily, annual_results, ax=None):
    """Annual mean concentration (bars) + flow-normalized concentration (line).
    If GenConc exists, show WRTDS-K line too."""

def plot_flux_hist(daily, annual_results, ax=None):
    """Annual flux + flow-normalized flux."""

def plot_contours(surfaces, surface_index, ax=None):
    """Color contour plot of concentration surface.
    X-axis: decimal year, Y-axis: log(Q), Color: concentration.
    Uses matplotlib contourf."""

def plot_conc_q_smooth(surfaces, surface_index, years, ax=None):
    """C-Q relationship curves for selected years (up to 3).
    Slices through the surface at specific times."""

def plot_conc_time_smooth(surfaces, surface_index, logq_values, ax=None):
    """C-time curves at selected discharge levels.
    Slices through the surface at specific LogQ values."""

def plot_diff_contours(surfaces1, surfaces2, surface_index, ax=None):
    """Difference contour between two surfaces (e.g., two time periods)."""
```

#### Tests (`test_plots.py`)

- Smoke tests only: verify each function runs without error and returns a Figure
- Use small synthetic datasets
- No pixel-level comparison

---

## Testing Strategy

### Hybrid approach: cached R fixtures + rpy2 regeneration

Tests compare Python output against R EGRET output. R results are **generated once via rpy2 and cached as parquet files**. Normal test runs load cached fixtures — R is only re-run when test conditions change (new dataset, different parameters, etc.).

This means:
- `uv run pytest` works without R installed (loads cached fixtures)
- `uv run pytest --regenerate-fixtures` re-runs R via rpy2 to rebuild fixture files (requires R + EGRET + rpy2)

### Test dependencies

Add to `[dependency-groups]` in `pyproject.toml`:
```toml
[dependency-groups]
dev = [
    # ... existing deps ...
    "rpy2>=3.5",        # only needed when regenerating fixtures
]
```

System requirements for fixture regeneration:
```bash
sudo apt install r-base
Rscript -e 'install.packages("EGRET")'
Rscript -e 'install.packages("EGRETci")'
```

### Fixture directory structure

```
wrtds/tests/
├── fixtures/
│   ├── README.md                 # Documents how fixtures were generated and when to regenerate
│   ├── generate_fixtures.py      # rpy2 script that runs R EGRET and exports results
│   ├── choptank_daily_input.parquet     # Raw input: Daily (Date, Q only)
│   ├── choptank_sample_input.parquet    # Raw input: Sample (Date, ConcLow, ConcHigh)
│   ├── choptank_info.json               # Raw input: INFO metadata
│   ├── choptank_daily_fitted.parquet    # R output: Daily after modelEstimation
│   ├── choptank_sample_cv.parquet       # R output: Sample after cross-validation
│   ├── choptank_surfaces.npy            # R output: surfaces 3D array
│   ├── choptank_surface_index.json      # R output: grid parameters
│   ├── choptank_annual.parquet          # R output: tableResults
│   ├── choptank_daily_kalman.parquet    # R output: Daily after WRTDSKalman
│   ├── choptank_pairs.parquet           # R output: runPairs result
│   ├── choptank_groups.parquet          # R output: runGroups result
│   └── choptank_monthly.parquet         # R output: calculateMonthlyResults
├── conftest.py
│   # - Defines @pytest.fixture for loading each fixture file
│   # - Registers --regenerate-fixtures CLI flag
│   # - If flag is set, calls generate_fixtures.py before tests run
└── ...
```

### Fixture generation script (`generate_fixtures.py`)

Uses rpy2 to call R EGRET functions and export all intermediate/final results:

```python
"""Generate test fixtures by running R EGRET on the Choptank River example dataset.

Run directly:   python wrtds/tests/fixtures/generate_fixtures.py
Run via pytest:  uv run pytest --regenerate-fixtures

Requires: R, EGRET R package, EGRETci R package, rpy2 Python package.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

FIXTURES_DIR = Path(__file__).parent


def generate():
    # Load built-in Choptank River chloride dataset
    ro.r('library(EGRET)')
    ro.r('eList <- Choptank_eList')

    # --- Raw inputs (before any modeling) ---
    daily_input = ro.r('eList$Daily[, c("Date", "Q")]')
    daily_input.to_parquet(FIXTURES_DIR / 'choptank_daily_input.parquet')

    sample_input = ro.r('eList$Sample[, c("Date", "ConcLow", "ConcHigh", "Uncen")]')
    sample_input.to_parquet(FIXTURES_DIR / 'choptank_sample_input.parquet')

    info = ro.r('as.list(eList$INFO)')
    # Convert to JSON-serializable dict
    with open(FIXTURES_DIR / 'choptank_info.json', 'w') as f:
        json.dump({k: str(v) for k, v in info.items()}, f)

    # --- Run full WRTDS model ---
    ro.r('eList <- modelEstimation(eList)')

    # Daily with all fitted columns
    daily_fitted = ro.r('eList$Daily')
    daily_fitted.to_parquet(FIXTURES_DIR / 'choptank_daily_fitted.parquet')

    # Sample with cross-validation results
    sample_cv = ro.r('eList$Sample')
    sample_cv.to_parquet(FIXTURES_DIR / 'choptank_sample_cv.parquet')

    # Surfaces 3D array
    surfaces = np.array(ro.r('eList$surfaces'))
    np.save(FIXTURES_DIR / 'choptank_surfaces.npy', surfaces)

    # Surface grid parameters
    surface_index = {
        'bottomLogQ': float(ro.r('eList$INFO$bottomLogQ')[0]),
        'stepLogQ': float(ro.r('eList$INFO$stepLogQ')[0]),
        'nVectorLogQ': int(ro.r('eList$INFO$nVectorLogQ')[0]),
        'bottomYear': float(ro.r('eList$INFO$bottomYear')[0]),
        'stepYear': float(ro.r('eList$INFO$stepYear')[0]),
        'nVectorYear': int(ro.r('eList$INFO$nVectorYear')[0]),
    }
    with open(FIXTURES_DIR / 'choptank_surface_index.json', 'w') as f:
        json.dump(surface_index, f)

    # --- Annual results ---
    ro.r('annualResults <- tableResults(eList)')
    annual = ro.r('annualResults')
    annual.to_parquet(FIXTURES_DIR / 'choptank_annual.parquet')

    # --- Monthly results ---
    ro.r('monthlyResults <- calculateMonthlyResults(eList)')
    monthly = ro.r('monthlyResults')
    monthly.to_parquet(FIXTURES_DIR / 'choptank_monthly.parquet')

    # --- WRTDS-K ---
    ro.r('set.seed(42)')
    ro.r('eList <- WRTDSKalman(eList, niter=200)')
    daily_kalman = ro.r('eList$Daily')
    daily_kalman.to_parquet(FIXTURES_DIR / 'choptank_daily_kalman.parquet')

    # --- Trend analysis ---
    ro.r('pairResults <- runPairs(eList, year1=1985, year2=2010, windowSide=7)')
    pairs = ro.r('pairResults')
    # pairs is a matrix — convert to DataFrame
    pairs_df = pd.DataFrame(
        np.array(pairs),
        index=['Conc', 'Flux'],
        columns=['x10', 'x11', 'x20', 'x22', 'CQTC', 'QTC', 'Total'],
    )
    pairs_df.to_parquet(FIXTURES_DIR / 'choptank_pairs.parquet')

    ro.r('''groupResults <- runGroups(eList,
        group1firstYear=1985, group1lastYear=1996,
        group2firstYear=1997, group2lastYear=2010, windowSide=7)''')
    groups = ro.r('groupResults')
    groups_df = pd.DataFrame(
        np.array(groups),
        index=['Conc', 'Flux'],
        columns=['x10', 'x11', 'x20', 'x22', 'CQTC', 'QTC', 'Total'],
    )
    groups_df.to_parquet(FIXTURES_DIR / 'choptank_groups.parquet')

    print(f'Fixtures generated in {FIXTURES_DIR}')


if __name__ == '__main__':
    generate()
```

### conftest.py fixture loading

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def pytest_addoption(parser):
    parser.addoption(
        '--regenerate-fixtures',
        action='store_true',
        default=False,
        help='Regenerate R EGRET test fixtures (requires R + rpy2)',
    )


def pytest_configure(config):
    if config.getoption('--regenerate-fixtures'):
        from wrtds.tests.fixtures.generate_fixtures import generate
        generate()


@pytest.fixture
def choptank_daily_input():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_daily_input.parquet')


@pytest.fixture
def choptank_sample_input():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_sample_input.parquet')


@pytest.fixture
def choptank_daily_fitted():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_daily_fitted.parquet')


@pytest.fixture
def choptank_sample_cv():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_sample_cv.parquet')


@pytest.fixture
def choptank_surfaces():
    return np.load(FIXTURES_DIR / 'choptank_surfaces.npy')


@pytest.fixture
def choptank_surface_index():
    with open(FIXTURES_DIR / 'choptank_surface_index.json') as f:
        return json.load(f)


@pytest.fixture
def choptank_annual():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_annual.parquet')


@pytest.fixture
def choptank_daily_kalman():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_daily_kalman.parquet')


@pytest.fixture
def choptank_pairs():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_pairs.parquet')


@pytest.fixture
def choptank_groups():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_groups.parquet')


@pytest.fixture
def choptank_monthly():
    return pd.read_parquet(FIXTURES_DIR / 'choptank_monthly.parquet')
```

### Example test using fixtures

```python
# test_flow_norm.py

import numpy as np

def test_fn_conc_matches_r(choptank_daily_input, choptank_sample_input, choptank_daily_fitted):
    """Verify flow-normalized concentration matches R EGRET output."""
    from wrtds import WRTDS

    model = WRTDS(choptank_daily_input, choptank_sample_input).fit()

    np.testing.assert_allclose(
        model.daily['FNConc'].values,
        choptank_daily_fitted['FNConc'].values,
        rtol=1e-4,
        err_msg='FNConc does not match R EGRET output',
    )
```

### Validation checkpoints by phase

| Phase | Fixture used | Columns/values compared |
|-------|-------------|------------------------|
| Phase 1 (data_prep) | `choptank_daily_fitted` | DecYear, Julian, MonthSeq, LogQ |
| Phase 1 (data_prep) | `choptank_sample_cv` | DecYear, SinDY, CosDY, Q, LogQ |
| Phase 3 (surfaces) | `choptank_surfaces`, `choptank_surface_index` | Full surfaces array, grid params |
| Phase 4 (flow_norm) | `choptank_daily_fitted` | ConcDay, FluxDay, FNConc, FNFlux |
| Phase 5 (cross_val) | `choptank_sample_cv` | yHat, SE, ConcHat |
| Phase 7 (kalman) | `choptank_daily_kalman` | GenConc, GenFlux |
| Phase 8 (trends) | `choptank_pairs`, `choptank_groups` | Full result tables |
| Phase 10 (summaries) | `choptank_annual`, `choptank_monthly` | All summary columns |

### When to regenerate fixtures

Fixtures are **committed to the repo** and only need regeneration if:
- The test dataset changes (different site, different parameter)
- The test parameters change (different window sizes, different year ranges)
- A new EGRET version changes its output (rare)

They do **not** need regeneration when changing Python code — that's the whole point.

### Tolerance

- `rtol=1e-4` for most numerical comparisons (regression coefficients, concentrations, fluxes)
- `rtol=1e-2` for WRTDS-K (Monte Carlo variance means exact match is not expected)
- Exact match for integer/date columns (Uncen, Month, Day, Julian, Date)

---

## Implementation Order Summary

| Step | What | Files | Done When |
|------|------|-------|-----------|
| 1 | Add dependencies to pyproject.toml | `pyproject.toml` | `uv sync` succeeds |
| 2 | Generate R fixtures | `tests/fixtures/generate_fixtures.py`, `tests/conftest.py` | `uv run pytest --regenerate-fixtures` produces all parquet/npy files |
| 3 | Data preparation | `data_prep.py`, `test_data_prep.py` | Computed columns match R fixtures |
| 4 | Regression engine | `regression.py`, `test_regression.py` | MLE matches R survreg |
| 5 | Surface estimation | `surfaces.py`, `test_surfaces.py` | Surfaces array matches R fixture |
| 6 | Flow normalization | `flow_norm.py`, `test_flow_norm.py` | FNConc/FNFlux match R fixture |
| 7 | Cross-validation | `cross_val.py`, `test_cross_val.py` | CV stats match R fixture |
| 8 | WRTDS class | `core.py` | `WRTDS(daily, sample).fit()` works end-to-end |
| 9 | WRTDS-K | `kalman.py`, `test_kalman.py` | GenConc/GenFlux match R fixture (rtol=1e-2) |
| 10 | Trend analysis | `trends.py`, `test_trends.py` | runPairs/runGroups match R fixtures |
| 11 | Bootstrap CI | `bootstrap.py`, `test_bootstrap.py` | CI structure correct, smoke test passes |
| 12 | Summary tables | Methods in `core.py` | tableResults matches R fixture |
| 13 | Plots | `plots/*.py`, `test_plots.py` | All plots render without error |
