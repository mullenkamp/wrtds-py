"""Tests for wrtds.surfaces module."""

import numpy as np
import pandas as pd
import pytest

from wrtds.data_prep import populate_daily, populate_sample
from wrtds.surfaces import (
    compute_surface_index,
    estimate_surfaces,
    interpolate_surface,
    make_interpolator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small populated sample DataFrame for surface tests.

    Uses a compact date range and modest sample size so that
    estimate_surfaces runs quickly (~14 * 33 = 462 grid points).
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
    q = rng.lognormal(mean=2.0, sigma=0.5, size=len(dates))
    daily = pd.DataFrame({'Date': dates, 'Q': q})
    daily = populate_daily(daily)

    # ~50 sample dates spread over the 2-year record
    sample_dates = dates[::15]
    n = len(sample_dates)
    conc = rng.lognormal(mean=1.0, sigma=0.3, size=n)
    # Make a few censored
    conc_low = conc.copy()
    conc_high = conc.copy()
    conc_low[:3] = np.nan
    conc_high[:3] = conc[:3] * 1.5  # detection limit

    sample = pd.DataFrame({
        'Date': sample_dates,
        'ConcLow': conc_low,
        'ConcHigh': conc_high,
    })
    sample = populate_sample(sample, daily)
    return sample


@pytest.fixture
def surface_index(sample_df):
    return compute_surface_index(sample_df)


@pytest.fixture
def small_surfaces(sample_df, surface_index):
    """Pre-computed surfaces on the small sample (relaxed constraints)."""
    return estimate_surfaces(
        sample_df, surface_index,
        window_y=7, window_q=2, window_s=0.5,
        min_num_obs=10, min_num_uncen=5,
    )


# ---------------------------------------------------------------------------
# compute_surface_index
# ---------------------------------------------------------------------------

class TestComputeSurfaceIndex:
    def test_n_logq_is_14(self, surface_index):
        assert surface_index['n_logq'] == 14
        assert len(surface_index['logq_grid']) == 14

    def test_logq_range(self, sample_df, surface_index):
        assert surface_index['bottom_logq'] == pytest.approx(sample_df['LogQ'].min() - 0.05)
        assert surface_index['top_logq'] == pytest.approx(sample_df['LogQ'].max() + 0.05)

    def test_logq_grid_endpoints(self, surface_index):
        assert surface_index['logq_grid'][0] == pytest.approx(surface_index['bottom_logq'])
        assert surface_index['logq_grid'][-1] == pytest.approx(surface_index['top_logq'])

    def test_year_bounds(self, sample_df, surface_index):
        assert surface_index['bottom_year'] == np.floor(sample_df['DecYear'].min())
        assert surface_index['top_year'] == np.ceil(sample_df['DecYear'].max())

    def test_step_year(self, surface_index):
        assert surface_index['step_year'] == pytest.approx(1.0 / 16.0)

    def test_year_grid_size(self, surface_index):
        expected_n = round(
            (surface_index['top_year'] - surface_index['bottom_year']) / surface_index['step_year']
        ) + 1
        assert surface_index['n_year'] == expected_n
        assert len(surface_index['year_grid']) == expected_n

    def test_year_grid_endpoints(self, surface_index):
        assert surface_index['year_grid'][0] == pytest.approx(surface_index['bottom_year'])
        assert surface_index['year_grid'][-1] == pytest.approx(surface_index['top_year'])


# ---------------------------------------------------------------------------
# estimate_surfaces
# ---------------------------------------------------------------------------

class TestEstimateSurfaces:
    def test_shape(self, small_surfaces, surface_index):
        assert small_surfaces.shape == (surface_index['n_logq'], surface_index['n_year'], 3)

    def test_finite(self, small_surfaces):
        assert np.all(np.isfinite(small_surfaces))

    def test_conchat_positive(self, small_surfaces):
        """ConcHat (layer 2) must be positive everywhere."""
        assert np.all(small_surfaces[:, :, 2] > 0)

    def test_se_positive(self, small_surfaces):
        """SE (layer 1) must be positive everywhere."""
        assert np.all(small_surfaces[:, :, 1] > 0)

    def test_bias_correction_consistent(self, small_surfaces):
        """ConcHat should equal exp(yHat + SE²/2) at every grid point."""
        yHat = small_surfaces[:, :, 0]
        SE = small_surfaces[:, :, 1]
        ConcHat = small_surfaces[:, :, 2]
        expected = np.exp(yHat + SE**2 / 2.0)
        np.testing.assert_allclose(ConcHat, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# make_interpolator / interpolate_surface
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_at_grid_point(self, small_surfaces, surface_index):
        """Interpolation at an exact grid point should return the stored value."""
        logq_grid = surface_index['logq_grid']
        year_grid = surface_index['year_grid']

        # Pick a point in the interior of the grid
        j, k = 5, 10
        expected = small_surfaces[j, k, 2]
        result = interpolate_surface(
            small_surfaces, surface_index,
            logq_grid[j], year_grid[k], layer=2,
        )
        assert result == pytest.approx(expected, rel=1e-10)

    def test_at_all_grid_points(self, small_surfaces, surface_index):
        """Interpolation at every grid point should match stored values."""
        logq_grid = surface_index['logq_grid']
        year_grid = surface_index['year_grid']

        logq_mesh, year_mesh = np.meshgrid(logq_grid, year_grid, indexing='ij')
        result = interpolate_surface(
            small_surfaces, surface_index,
            logq_mesh.ravel(), year_mesh.ravel(), layer=2,
        )
        expected = small_surfaces[:, :, 2].ravel()
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_midpoint_interpolation(self, small_surfaces, surface_index):
        """Value at the midpoint of four grid cells should be the average."""
        logq_grid = surface_index['logq_grid']
        year_grid = surface_index['year_grid']

        j, k = 3, 8
        mid_logq = (logq_grid[j] + logq_grid[j + 1]) / 2
        mid_year = (year_grid[k] + year_grid[k + 1]) / 2

        # Bilinear interpolation at the exact center of a cell gives the
        # average of the four corners.
        corners = [
            small_surfaces[j, k, 2],
            small_surfaces[j, k + 1, 2],
            small_surfaces[j + 1, k, 2],
            small_surfaces[j + 1, k + 1, 2],
        ]
        expected = np.mean(corners)
        result = interpolate_surface(
            small_surfaces, surface_index,
            mid_logq, mid_year, layer=2,
        )
        assert result == pytest.approx(expected, rel=1e-10)

    def test_clamping_below(self, small_surfaces, surface_index):
        """Query below the grid should clamp to bottom boundary."""
        result = interpolate_surface(
            small_surfaces, surface_index,
            logq=surface_index['logq_grid'][0] - 10.0,
            dec_year=surface_index['year_grid'][0],
            layer=2,
        )
        expected = small_surfaces[0, 0, 2]
        assert result == pytest.approx(expected, rel=1e-10)

    def test_clamping_above(self, small_surfaces, surface_index):
        """Query above the grid should clamp to top boundary."""
        result = interpolate_surface(
            small_surfaces, surface_index,
            logq=surface_index['logq_grid'][-1] + 10.0,
            dec_year=surface_index['year_grid'][-1],
            layer=2,
        )
        expected = small_surfaces[-1, -1, 2]
        assert result == pytest.approx(expected, rel=1e-10)

    def test_all_layers(self, small_surfaces, surface_index):
        """Interpolation should work for all three layers."""
        logq = surface_index['logq_grid'][5]
        year = surface_index['year_grid'][10]

        for layer in range(3):
            result = interpolate_surface(
                small_surfaces, surface_index,
                logq, year, layer=layer,
            )
            expected = small_surfaces[5, 10, layer]
            assert result == pytest.approx(expected, rel=1e-10)

    def test_vector_input(self, small_surfaces, surface_index):
        """Should accept and return arrays."""
        logq = surface_index['logq_grid'][:3]
        year = surface_index['year_grid'][:3]
        result = interpolate_surface(
            small_surfaces, surface_index, logq, year, layer=2,
        )
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_make_interpolator_reusable(self, small_surfaces, surface_index):
        """make_interpolator should return a callable that gives correct results."""
        interp = make_interpolator(small_surfaces, surface_index, layer=2)
        logq = surface_index['logq_grid'][5]
        year = surface_index['year_grid'][10]
        result = interp(np.array([[logq, year]]))[0]
        expected = small_surfaces[5, 10, 2]
        assert result == pytest.approx(expected, rel=1e-10)
