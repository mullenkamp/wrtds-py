# WRTDS-PY Context for Gemini

## Project Overview

**wrtds-py** is a Python transcription of the USGS R package **WRTDS** (Weighted Regressions on Time, Discharge, and Season), also known as EGRET. It provides methods for analyzing long-term trends in water quality data.

*   **Goal:** Create a 1:1 Python equivalent of the R package, using `pandas` for data structures and `scipy`/`numpy` for the computational engine (replacing R's `survival::survreg`).
*   **Status:** Early development / Scaffold stage.
*   **Core Tech:** Python 3.10+, Pandas, NumPy, SciPy, Matplotlib.

## Architecture

The project follows a flat layout with the source code in `wrtds/`.

*   **`wrtds/core.py` (Planned):** Contains the main `WRTDS` class, serving as the primary API.
*   **`wrtds/regression.py` (Planned):** Implements the Weighted Censored Gaussian MLE and tricube weighting logic.
*   **`wrtds/surfaces.py` (Planned):** Handles the estimation of the 3D regression surface (LogQ x Year x Layers).
*   **`wrtds/tests/`:** Co-located tests.
    *   **`wrtds/tests/fixtures/`:** Critical directory containing cached parquet/npy files generated from the R package. Tests compare Python output against these R fixtures to ensure accuracy.

## Development Workflow

The project uses **uv** for dependency management and task execution.

### Setup
Initialize the environment and install dependencies:
```bash
uv sync
```

### Testing
Run the test suite using `pytest`.
```bash
uv run pytest
```
*   **Fixture Regeneration:** Tests rely on cached R outputs. If you need to regenerate them (requires R and `rpy2` installed):
    ```bash
    uv run pytest --regenerate-fixtures
    ```

### Code Quality
The project enforces strict linting and formatting.
*   **Linting:** `uv run ruff check .`
*   **Formatting:** `uv run black .`
*   **Type Checking:** `uv run mypy src/wrtds` (or appropriate path)

### Documentation
Built with MkDocs Material.
```bash
uv run docs-serve
```

## Key Files to Reference

*   **`IMPLEMENTATION_PLAN.md`:** **CRITICAL**. This file contains the detailed step-by-step roadmap, mathematical formulas, and architectural decisions. **Always consult this plan before implementing new features.**
*   **`pyproject.toml`:** Defines dependencies and tool configurations (Ruff, Black, Hatch).
*   **`wrtds/tests/conftest.py`:** Defines the fixtures used to load the R comparison data.

## Coding Conventions

*   **Style:** Follows `black` (formatting) and `ruff` (linting) rules.
*   **Line Length:** 120 characters.
*   **Imports:** Use absolute imports (e.g., `from wrtds.utils import ...`) rather than relative imports.
*   **Docstrings:** Google style.
*   **Testing:** New features *must* be verified against R package outputs whenever possible.
