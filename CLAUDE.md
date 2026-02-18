# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wrtds-py is a Python transcription of the USGS R package WRTDS (Weighted Regressions on Time, Discharge, and Season). It uses pandas as the base with scipy, scikit-learn, and statsmodels for core statistics. The project is in early development (scaffold stage).

## Commands

### Environment setup
```bash
uv sync
```

### Running tests
```bash
uv run pytest                                          # all tests
uv run pytest wrtds/tests/test_file.py::test_name     # single test
```

### Linting and formatting (via hatch)
```bash
uv run lint:style    # ruff + black --check
uv run lint:fmt      # auto-format with black + ruff --fix
uv run lint:typing   # mypy type checking
uv run lint:all      # style + typing
```

### Documentation
```bash
uv run docs-serve    # local dev server
uv run docs-build    # build static site
```

### Build and publish
```bash
uv build
uv publish
```

## Architecture

- **Package layout:** `wrtds/` (flat layout, no `src/` prefix)
- **Tests:** Co-located at `wrtds/tests/`
- **Build system:** Hatchling
- **Docs:** MkDocs Material with mkdocstrings (Google-style docstrings)
- **Version:** Single source of truth in `wrtds/__init__.py`

## Code Style

- **Line length:** 120 characters
- **Quotes:** Single quotes preferred (black's `skip-string-normalization = true`)
- **Indentation:** 4 spaces
- **Imports:** Absolute only (relative imports banned by ruff)
- **Docstrings:** Google style
