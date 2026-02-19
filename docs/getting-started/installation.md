# Installation

## Requirements

- Python >= 3.11

## Install from PyPI

=== "pip"

    ```bash
    pip install wrtds
    ```

=== "uv"

    ```bash
    uv add wrtds
    ```

## Dependencies

The following packages are installed automatically:

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| pandas | >= 2.0 | DataFrames for Daily and Sample records |
| numpy | >= 1.24 | Array operations |
| scipy | >= 1.10 | L-BFGS-B optimizer, interpolation, statistics |
| matplotlib | >= 3.7 | Plotting |

## Development Install

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/mullenkamp/wrtds-py.git
cd wrtds-py
uv sync
```

To also install documentation dependencies:

```bash
uv sync --group docs
```

## Verify Installation

```python
import wrtds
print(wrtds.__version__)
```
