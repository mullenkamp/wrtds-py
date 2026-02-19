# wrtds-py

A Python implementation of **WRTDS** (Weighted Regressions on Time, Discharge, and Season), the USGS method for estimating long-term trends in river water quality.

[![build](https://github.com/mullenkamp/wrtds-py/workflows/Build/badge.svg)](https://github.com/mullenkamp/wrtds-py/actions)
[![codecov](https://codecov.io/gh/mullenkamp/wrtds-py/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/wrtds-py)
[![PyPI version](https://badge.fury.io/py/wrtds.svg)](https://badge.fury.io/py/wrtds)

---

**Documentation**: [https://mullenkamp.github.io/wrtds-py/](https://mullenkamp.github.io/wrtds-py/)

**Source Code**: [https://github.com/mullenkamp/wrtds-py](https://github.com/mullenkamp/wrtds-py)

---

## Overview

This package is a Python transcription of the USGS R package [EGRET](https://github.com/DOI-USGS/EGRET). It uses pandas DataFrames as the base data structure with scipy for optimization and interpolation and matplotlib for plotting.

Key features:

- **Weighted censored regression** — locally weighted MLE with tricube kernels on time, discharge, and season
- **Flow normalization** — isolate water-quality trends from discharge variability
- **WRTDS-K** — AR(1) residual interpolation for improved daily estimates
- **Trend analysis** — pairwise, group, and time-series decomposition (CQTC/QTC)
- **Bootstrap confidence intervals** — block resampling with bias correction
- **Plotting** — data overview, annual histories, contour surfaces, and diagnostics

## Installation

```bash
pip install wrtds
```

Requires Python >= 3.10.

## Quick Example

```python
import pandas as pd
from wrtds import WRTDS

daily = pd.read_csv('daily.csv', parse_dates=['Date'])
sample = pd.read_csv('sample.csv', parse_dates=['Date'])

w = WRTDS(daily, sample, info={'station_name': 'Choptank River'})
w.fit()
w.kalman()

print(w.table_results())
print(w.run_pairs(year1=1985, year2=2010))
w.plot_conc_hist()
```

See the [Quickstart](https://mullenkamp.github.io/wrtds/getting-started/quickstart/) for a full walkthrough.

## Development

We use [uv](https://docs.astral.sh/uv/) to manage the development environment.

```bash
uv sync            # install dependencies
uv run pytest      # run tests
```

## License

This project is licensed under the terms of the Apache Software License 2.0.
