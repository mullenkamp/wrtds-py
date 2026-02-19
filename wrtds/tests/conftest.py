"""Pytest configuration: --regenerate-fixtures hook and R fixture loaders."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / 'fixtures'


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        '--regenerate-fixtures',
        action='store_true',
        default=False,
        help='Regenerate R EGRET reference fixtures before running tests.',
    )


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: marks tests as slow (deselect with -m "not slow")')
    config.addinivalue_line('markers', 'r_fixtures: marks tests that require R fixture files')

    if config.getoption('--regenerate-fixtures', default=False):
        from wrtds.tests.fixtures.generate_fixtures import generate
        print('\n=== Regenerating R fixtures ===')
        generate()
        print('=== R fixtures regenerated ===\n')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(name):
    """Load a CSV fixture, skipping if not present."""
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f'Fixture {name} not found. Run pytest --regenerate-fixtures')
    return pd.read_csv(path)


def _load_json(name):
    """Load a JSON fixture, skipping if not present."""
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f'Fixture {name} not found. Run pytest --regenerate-fixtures')
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Raw input fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def r_daily_input():
    """Raw daily input (Date, Q) from Choptank_eList."""
    df = _load_csv('choptank_daily_input.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@pytest.fixture(scope='module')
def r_sample_input():
    """Raw sample input (Date, ConcLow, ConcHigh, Uncen) from Choptank_eList."""
    df = _load_csv('choptank_sample_input.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@pytest.fixture(scope='module')
def r_info():
    """Site metadata from Choptank_eList INFO."""
    return _load_json('choptank_info.json')


# ---------------------------------------------------------------------------
# Fitted model output fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def r_daily_fitted():
    """Daily DataFrame after modelEstimation (windowY=7)."""
    df = _load_csv('choptank_daily_fitted.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@pytest.fixture(scope='module')
def r_sample_cv():
    """Sample DataFrame after cross-validation."""
    df = _load_csv('choptank_sample_cv.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@pytest.fixture(scope='module')
def r_surfaces():
    """3D surfaces array from R EGRET."""
    bin_path = FIXTURE_DIR / 'choptank_surfaces.bin'
    json_path = FIXTURE_DIR / 'choptank_surface_index.json'
    if not bin_path.exists() or not json_path.exists():
        pytest.skip('Surface fixtures not found. Run pytest --regenerate-fixtures')

    with open(json_path) as f:
        index = json.load(f)

    shape = tuple(index['shape'])
    data = np.fromfile(str(bin_path), dtype=np.float64)
    return data.reshape(shape, order='F')


@pytest.fixture(scope='module')
def r_surface_index():
    """Surface grid parameters from R EGRET."""
    return _load_json('choptank_surface_index.json')


# ---------------------------------------------------------------------------
# Summary and Kalman fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def r_annual():
    """setupYears output (DecYear, Q, Conc, Flux, FNConc, FNFlux)."""
    return _load_csv('choptank_annual.csv')


@pytest.fixture(scope='module')
def r_daily_kalman():
    """WRTDSKalman output (Date, GenConc, GenFlux)."""
    df = _load_csv('choptank_daily_kalman.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ---------------------------------------------------------------------------
# Trend analysis fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def r_pairs():
    """runPairs result (1985 vs 2010) with row index ['Conc', 'Flux']."""
    return _load_csv('choptank_pairs.csv').set_index(
        _load_csv('choptank_pairs.csv').columns[0]
    )


@pytest.fixture(scope='module')
def r_groups():
    """runGroups result (1985-1996 vs 1997-2010) with row index ['Conc', 'Flux']."""
    return _load_csv('choptank_groups.csv').set_index(
        _load_csv('choptank_groups.csv').columns[0]
    )
