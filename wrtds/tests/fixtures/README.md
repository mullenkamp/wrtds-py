# R Fixture Tests

The `test_vs_r.py` module compares Python WRTDS output against cached R EGRET
reference output generated from the Choptank River chloride dataset. Normal
test runs load pre-generated fixture files (no R required). The
`--regenerate-fixtures` flag re-generates them from scratch.

## Quick start

```bash
# Run R comparison tests using cached fixtures (no R needed)
uv run pytest wrtds/tests/test_vs_r.py -v

# Skip R tests (run only fast unit tests)
uv run pytest -m "not slow"

# Regenerate fixtures and run tests (requires R)
uv run pytest --regenerate-fixtures wrtds/tests/test_vs_r.py -v
```

## Requirements for fixture regeneration

Regenerating fixtures requires R and two R packages. The Python side needs
only the standard dev dependencies (no rpy2).

### R (>= 4.1)

Install R via your system package manager:

```bash
# Ubuntu/Debian
sudo apt install r-base

# macOS (Homebrew)
brew install r

# Fedora/RHEL
sudo dnf install R
```

Verify with:

```bash
Rscript --version
```

### R packages: EGRET and jsonlite

From an R console (`R` or `Rscript -e`):

```r
install.packages(c("EGRET", "jsonlite"))
```

Or from the shell:

```bash
Rscript -e 'install.packages(c("EGRET", "jsonlite"), repos="https://cloud.r-project.org")'
```

Verify with:

```bash
Rscript -e 'library(EGRET); library(jsonlite); cat("OK\n")'
```

### Tested versions

| Component | Version |
|-----------|---------|
| R         | >= 4.1  |
| EGRET     | >= 3.0  |
| jsonlite  | >= 1.8  |

## What the fixtures contain

The R script (`generate_fixtures.R`) loads the built-in `Choptank_eList`
dataset from EGRET, re-runs the full model estimation with `windowY=7`, and
exports 11 files:

| File | Contents |
|------|----------|
| `choptank_daily_input.csv` | Raw daily data (Date, Q) |
| `choptank_sample_input.csv` | Raw sample data (Date, ConcLow, ConcHigh, Uncen) |
| `choptank_info.json` | Site metadata |
| `choptank_daily_fitted.csv` | Daily after modelEstimation |
| `choptank_sample_cv.csv` | Sample after cross-validation |
| `choptank_surfaces.bin` | 3D surfaces array (float64, Fortran order) |
| `choptank_surface_index.json` | Surface grid parameters and shape |
| `choptank_annual.csv` | setupYears output |
| `choptank_daily_kalman.csv` | WRTDSKalman output (GenConc, GenFlux) |
| `choptank_pairs.csv` | runPairs result (1985 vs 2010) |
| `choptank_groups.csv` | runGroups result (1985-1996 vs 1997-2010) |

Regeneration takes about 5 minutes (dominated by R's `modelEstimation`).

## Standalone regeneration

You can also regenerate fixtures without pytest:

```bash
# Via Python
uv run python -m wrtds.tests.fixtures.generate_fixtures

# Via R directly
Rscript wrtds/tests/fixtures/generate_fixtures.R wrtds/tests/fixtures/
```

## Known differences between R and Python

The comparison tests use calibrated tolerances rather than exact matching.
Three systematic sources of difference are documented in `test_vs_r.py`:

1. **DecYear computation** -- R's bundled Choptank dataset has DecYear values
   from an older formula affected by timezone/DST handling. Python recomputes
   from dates using a timezone-naive formula. ~2e-6 relative difference.

2. **Surface grid bounds** -- R computes LogQ grid bounds from the Daily
   discharge range; Python computes from the Sample LogQ range. This produces
   different grid parameters, so surface arrays cannot be compared
   element-by-element.

3. **MLE optimizer** -- R uses `survival::survreg`; Python uses
   `scipy.optimize.minimize(L-BFGS-B)`. Small differences accumulate across
   thousands of grid-point regressions, resulting in ~1-5% differences in
   model outputs.
