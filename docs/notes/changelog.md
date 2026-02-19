# Changelog

## 0.1.0

Initial release.

- Core WRTDS model: weighted censored regression on time, discharge, and season
- Surface estimation and bilinear interpolation
- Flow normalization (stationary and generalized)
- Leave-one-out cross-validation
- WRTDS-K (Kalman) AR(1) residual interpolation
- Trend analysis: `run_pairs`, `run_groups`, `run_series`
- Block bootstrap confidence intervals
- Summary tables: `table_results`, `table_change`, `error_stats`, `flux_bias_stat`
- Plotting: data overview, result histories, contour plots, diagnostics
