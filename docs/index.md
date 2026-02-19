# wrtds-py

A Python implementation of **WRTDS** (Weighted Regressions on Time, Discharge, and Season),
the USGS method for estimating long-term trends in river water quality.

Originally developed as the R package [EGRET](https://github.com/DOI-USGS/EGRET),
wrtds-py brings the full WRTDS workflow to the Python ecosystem using pandas, numpy,
scipy, and matplotlib.

## Features

- **Weighted censored regression** — locally weighted MLE with tricube kernels on time, discharge, and season
- **Flow normalization** — isolate water-quality trends from discharge variability
- **WRTDS-K** — AR(1) residual interpolation for improved daily estimates
- **Trend analysis** — pairwise, group, and time-series decomposition (CQTC/QTC)
- **Bootstrap confidence intervals** — block resampling with bias correction
- **Plotting** — data overview, annual histories, contour surfaces, and diagnostics

## Quick Example

```python
import pandas as pd
from wrtds import WRTDS

daily = pd.read_csv('daily.csv', parse_dates=['Date'])
sample = pd.read_csv('sample.csv', parse_dates=['Date'])

w = WRTDS(daily, sample, info={'station_name': 'Choptank River'})
w.fit()
w.kalman()

# Annual summary
print(w.table_results())

# Trend between two years
print(w.run_pairs(year1=1985, year2=2010))

# Plot concentration history
w.plot_conc_hist()
```

## Install

```bash
pip install wrtds
```

## Documentation

<div class="grid cards" markdown>

- [**Getting Started**](getting-started/installation.md) — installation, quickstart tutorial, input data format
- [**User Guide**](guide/overview.md) — model fitting, WRTDS-K, trends, bootstrap, plotting
- [**API Reference**](reference/core.md) — full module-level documentation
- [**R vs Python Differences**](notes/r-vs-python.md) — known numerical differences from EGRET

</div>

## License

Apache Software License 2.0
