# Full Documentation Site for wrtds-py

## Context

The project has a minimal MkDocs Material setup (landing page that just includes README, a single API reference page). The CI workflow (`documentation.yml`) uses `hatch` but the project has migrated to `uv`. Docs dependencies (`mkdocs-material`, `mkdocstrings`) aren't in uv's dependency groups so `uv run mkdocs build` doesn't work. GitHub Actions versions are outdated.

The goal is a full user guide with: overview, installation, quickstart tutorial, per-module API reference, and a dedicated page documenting the fractional timestamp (DecYear) issue and other R vs Python differences discovered during the fixture testing work.

## Files to Modify

### `pyproject.toml`
Add docs dependency group:
```toml
[dependency-groups]
docs = [
  "mkdocs-material>=9.5",
  "mkdocstrings[python]>=0.24",
]
```

### `mkdocs.yml`
Full rewrite — expand nav, add extensions (`admonition`, `pymdownx.details`, `pymdownx.tabbed`, `pymdownx.arithmatex`, `attr_list`), add cross-reference imports (pandas, numpy, scipy `objects.inv`), set default mkdocstrings options (`show_root_heading: true`, `show_source: true`, `members_order: source`), add Material theme features (`navigation.sections`, `content.code.copy`, `search.suggest`), add MathJax for formulas.

### `.github/workflows/documentation.yml`
Rewrite to use `uv` instead of `hatch`. Update action versions:
- `actions/checkout@v3` → `v4`
- `actions/setup-python@v4` → `v5`
- `actions/upload-pages-artifact@v1` → `v3`
- `actions/deploy-pages@v1` → `v4`
- Add `astral-sh/setup-uv@v5`
- Use `uv sync --group docs` + `uv run mkdocs build`

### `docs/index.md`
Rewrite from snippet-only to a proper landing page: project description, feature highlights, quick code example, install one-liner, navigation links.

### `docs/reference/wrtds.md`
Delete (superseded by per-module reference pages).

## Files to Create

### Getting Started (3 pages)
| File | Content |
|------|---------|
| `docs/getting-started/installation.md` | pip/uv install, Python >=3.11, dependencies |
| `docs/getting-started/quickstart.md` | End-to-end tutorial: load data → fit → kalman → trends → plots |
| `docs/getting-started/input-data.md` | Daily/Sample DataFrame specs, ConcLow/ConcHigh vs Conc/Remark format, `info` dict, `DEFAULT_INFO` |

### User Guide (7 pages)
| File | Content |
|------|---------|
| `docs/guide/overview.md` | What WRTDS is, scientific background, regression formula, flow normalization concept. Link to Hirsch (2010). |
| `docs/guide/fitting.md` | `.fit()` pipeline details: cross-validation → surfaces → daily estimation → flow norm. Window parameters, min obs, edge adjust. Calling sub-steps individually. |
| `docs/guide/kalman.md` | WRTDS-K: AR(1) residual interpolation, choosing rho, n_iter, interpreting GenConc/GenFlux |
| `docs/guide/trends.md` | `run_pairs`, `run_groups`, `run_series`: CQTC/QTC decomposition, xAB notation, generalized flow normalization |
| `docs/guide/bootstrap.md` | Bootstrap CI: block resampling, bias correction, p-values, likelihood descriptors |
| `docs/guide/summaries.md` | `table_results`, `table_change`, `error_stats`, `flux_bias_stat` |
| `docs/guide/plotting.md` | All plot methods with descriptions |

### Notes (2 pages)
| File | Content |
|------|---------|
| `docs/notes/r-vs-python.md` | **The key disclaimer page.** Three known differences: (1) DecYear — R's bundled Choptank uses older DST-affected formula, Python is timezone-naive, ~2e-6 rtol; (2) Surface grid bounds — R uses Daily Q range, Python uses Sample LogQ range; (3) MLE optimizer — R survreg vs Python scipy L-BFGS-B, ~1-5% differences. Include tolerance table from test_vs_r.py. Note that differences are negligible for scientific conclusions. |
| `docs/notes/changelog.md` | Placeholder |

### API Reference (13 pages, one per module)
Each page uses `:::` mkdocstrings directive:
| File | Directive |
|------|-----------|
| `docs/reference/core.md` | `::: wrtds.core` |
| `docs/reference/data_prep.md` | `::: wrtds.data_prep` |
| `docs/reference/regression.md` | `::: wrtds.regression` |
| `docs/reference/surfaces.md` | `::: wrtds.surfaces` |
| `docs/reference/flow_norm.md` | `::: wrtds.flow_norm` |
| `docs/reference/cross_val.md` | `::: wrtds.cross_val` |
| `docs/reference/kalman.md` | `::: wrtds.kalman` |
| `docs/reference/trends.md` | `::: wrtds.trends` |
| `docs/reference/summaries.md` | `::: wrtds.summaries` |
| `docs/reference/bootstrap.md` | `::: wrtds.bootstrap` |
| `docs/reference/plots/data_overview.md` | `::: wrtds.plots.data_overview` |
| `docs/reference/plots/results.md` | `::: wrtds.plots.results` |
| `docs/reference/plots/diagnostics.md` | `::: wrtds.plots.diagnostics` |

## Nav Structure

```
Home
Getting Started/
  Installation
  Quickstart
  Input Data Format
User Guide/
  Overview
  Fitting a Model
  WRTDS-K (Kalman)
  Trend Analysis
  Bootstrap Confidence Intervals
  Summary Tables
  Plotting
Notes/
  R vs Python Differences    ← the disclaimer page
  Changelog
API Reference/
  WRTDS Class
  Data Preparation
  Regression
  Surfaces
  Flow Normalization
  Cross-Validation
  Kalman
  Trends
  Summaries
  Bootstrap
  Plots/
    Overview Plots
    Result Plots
    Diagnostic Plots
```

## Implementation Order

1. **Infrastructure**: `pyproject.toml` (add docs deps) → `uv sync --group docs` → `mkdocs.yml` (full rewrite)
2. **API reference pages**: 13 mechanical files with `:::` directives; delete old `wrtds.md`
3. **Getting started pages**: installation, input-data, quickstart tutorial
4. **User guide pages**: overview, fitting, kalman, trends, bootstrap, summaries, plotting
5. **Notes pages**: r-vs-python (the disclaimer), changelog
6. **Landing page**: rewrite `docs/index.md`
7. **CI workflow**: rewrite `documentation.yml`

## Verification

1. `uv sync --group docs` — installs mkdocs-material and mkdocstrings
2. `uv run mkdocs build --strict` — clean build with no warnings
3. `uv run mkdocs serve` — visual inspection at localhost:8000
4. All nav links work, API reference renders docstrings, R-vs-Python page has the tolerance table
