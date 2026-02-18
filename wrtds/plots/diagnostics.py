"""Model diagnostic plots — residuals, predicted vs observed."""

import numpy as np
from scipy.ndimage import uniform_filter1d

from wrtds.plots.utils import _get_ax, _get_fig


def plot_conc_pred(sample, ax=None):
    """Predicted vs observed concentration with 1:1 reference line.

    Args:
        sample: Sample DataFrame with ``ConcHat`` and ``ConcAve``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    ax.scatter(sample['ConcHat'], sample['ConcAve'], s=20, color='C0')
    lo = min(sample['ConcHat'].min(), sample['ConcAve'].min())
    hi = max(sample['ConcHat'].max(), sample['ConcAve'].max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='1:1')
    ax.set_xlabel('Predicted Concentration')
    ax.set_ylabel('Observed Concentration')
    ax.set_title('Predicted vs Observed Concentration')
    ax.legend(fontsize='small')
    return fig


def plot_flux_pred(sample, ax=None):
    """Predicted vs observed flux with 1:1 reference line.

    Flux = Concentration * Q * 86.4 (kg/day).

    Args:
        sample: Sample DataFrame with ``ConcHat``, ``ConcAve``, ``Q``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    pred_flux = sample['ConcHat'] * sample['Q'] * 86.4
    obs_flux = sample['ConcAve'] * sample['Q'] * 86.4
    ax.scatter(pred_flux, obs_flux, s=20, color='C0')
    lo = min(pred_flux.min(), obs_flux.min())
    hi = max(pred_flux.max(), obs_flux.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='1:1')
    ax.set_xlabel('Predicted Flux (kg/day)')
    ax.set_ylabel('Observed Flux (kg/day)')
    ax.set_title('Predicted vs Observed Flux')
    ax.legend(fontsize='small')
    return fig


def _residuals(sample):
    """Compute log-space residuals: log(ConcAve) - yHat."""
    return np.log(sample['ConcAve'].values) - sample['yHat'].values


def plot_resid_pred(sample, ax=None):
    """Residuals vs predicted concentration.

    Args:
        sample: Sample DataFrame with ``yHat`` and ``ConcAve``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    resid = _residuals(sample)
    pred = np.exp(sample['yHat'].values)
    ax.scatter(pred, resid, s=20, color='C0')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Predicted Concentration (exp(yHat))')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Residuals vs Predicted')
    return fig


def plot_resid_q(sample, ax=None):
    """Residuals vs log-discharge.

    Args:
        sample: Sample DataFrame with ``LogQ``, ``yHat``, ``ConcAve``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    resid = _residuals(sample)
    ax.scatter(sample['LogQ'], resid, s=20, color='C0')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Log(Q)')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Residuals vs Discharge')
    return fig


def plot_resid_time(sample, ax=None):
    """Residuals vs time with running-mean smooth.

    Args:
        sample: Sample DataFrame with ``DecYear``, ``yHat``, ``ConcAve``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    resid = _residuals(sample)
    dec_year = sample['DecYear'].values

    # Sort by time for smooth line
    order = np.argsort(dec_year)
    dec_sorted = dec_year[order]
    resid_sorted = resid[order]

    ax.scatter(dec_year, resid, s=20, color='C0', alpha=0.6)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

    # Running mean smooth (window = ~1/5 of data points, minimum 3)
    window = max(3, len(resid_sorted) // 5)
    if window % 2 == 0:
        window += 1
    smooth = uniform_filter1d(resid_sorted.astype(float), size=window)
    ax.plot(dec_sorted, smooth, color='C3', linewidth=1.5, label='Smooth')

    ax.set_xlabel('Decimal Year')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Residuals vs Time')
    ax.legend(fontsize='small')
    return fig


def box_resid_month(sample, ax=None):
    """Box plots of residuals grouped by month.

    Args:
        sample: Sample DataFrame with ``Month``, ``yHat``, ``ConcAve``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    resid = _residuals(sample)
    months = range(1, 13)
    month_vals = sample['Month'].values
    data = [resid[month_vals == m] for m in months]
    # Only include months that have data
    labels = [str(m) for m in months]
    ax.boxplot(data, tick_labels=labels)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Month')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Residuals by Month')
    return fig


def flux_bias_multi(sample, fig=None):
    """Multi-panel diagnostic: 6 scatter plots.

    Panel layout (2x3):
        - Predicted vs Observed Conc
        - Predicted vs Observed Flux
        - Residuals vs Predicted
        - Residuals vs Discharge
        - Residuals vs Time
        - Residuals by Month

    Args:
        sample: Sample DataFrame with cross-validation columns.
        fig: Optional matplotlib figure.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, axes = _get_fig(fig, nrows=2, ncols=3, figsize=(14, 8))
    plot_conc_pred(sample, ax=axes[0])
    plot_flux_pred(sample, ax=axes[1])
    plot_resid_pred(sample, ax=axes[2])
    plot_resid_q(sample, ax=axes[3])
    plot_resid_time(sample, ax=axes[4])
    box_resid_month(sample, ax=axes[5])
    fig.tight_layout()
    return fig
