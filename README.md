# wrtds-py

<p align="center">
    <em>A transcription of the usgs r package wrtds to python</em>
</p>

[![build](https://github.com/mullenkamp/wrtds-py/workflows/Build/badge.svg)](https://github.com/mullenkamp/wrtds-py/actions)
[![codecov](https://codecov.io/gh/mullenkamp/wrtds-py/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/wrtds-py)
[![PyPI version](https://badge.fury.io/py/wrtds.svg)](https://badge.fury.io/py/wrtds)

---

**Source Code**: <a href="https://github.com/mullenkamp/wrtds-py" target="_blank">https://github.com/mullenkamp/wrtds-py</a>

---
## Overview
This package is meant to transcribe the usgs r package called wrtds into python. It should use pandas classes as the base and use a combo of scipy, scikit-learn, and statsmodels for the core stats packages.

## Development

### Setup environment

We use [UV](https://docs.astral.sh/uv/) to manage the development environment and production build. 

```bash
uv sync
```

### Run unit tests

You can run all the tests with:

```bash
uv run pytest
```

### Format the code

Execute the following command to apply linting and check typing:

```bash
uv run lint
```

## License

This project is licensed under the terms of the Apache Software License 2.0.
