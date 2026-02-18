"""Raw data overview plots — discharge and concentration time series."""

import numpy as np

from wrtds.plots.utils import _get_ax, _get_fig


def plot_q_time_daily(daily, ax=None):
    """Daily discharge time series.

    Args:
        daily: Populated daily DataFrame with ``Date`` and ``Q``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    ax.plot(daily['Date'], daily['Q'], linewidth=0.5, color='C0')
    ax.set_yscale('log')
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('Daily Discharge')
    return fig


def plot_conc_time(sample, ax=None):
    """Concentration vs time scatter.

    Open circles for censored observations (``Uncen == 0``).

    Args:
        sample: Populated sample DataFrame with ``DecYear``, ``ConcAve``,
            ``Uncen``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    uncen = sample['Uncen'] == 1
    ax.scatter(sample.loc[uncen, 'DecYear'], sample.loc[uncen, 'ConcAve'],
               s=20, color='C0', label='Uncensored')
    if (~uncen).any():
        ax.scatter(sample.loc[~uncen, 'DecYear'], sample.loc[~uncen, 'ConcAve'],
                   s=20, facecolors='none', edgecolors='C0', label='Censored')
    ax.set_xlabel('Decimal Year')
    ax.set_ylabel('Concentration')
    ax.set_title('Concentration vs Time')
    ax.legend(fontsize='small')
    return fig


def plot_conc_q(sample, ax=None):
    """Log-log scatter of concentration vs discharge.

    Args:
        sample: Populated sample DataFrame with ``Q`` and ``ConcAve``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    ax.scatter(sample['Q'], sample['ConcAve'], s=20, color='C0')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Discharge (m³/s)')
    ax.set_ylabel('Concentration')
    ax.set_title('Concentration vs Discharge')
    return fig


def box_conc_month(sample, ax=None):
    """Box plots of concentration grouped by month.

    Args:
        sample: Populated sample DataFrame with ``Month`` and ``ConcAve``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    months = range(1, 13)
    data = [sample.loc[sample['Month'] == m, 'ConcAve'].dropna().values for m in months]
    ax.boxplot(data, tick_labels=[str(m) for m in months])
    ax.set_xlabel('Month')
    ax.set_ylabel('Concentration')
    ax.set_title('Concentration by Month')
    return fig


def box_q_twice(daily, sample, ax=None):
    """Side-by-side box plots of LogQ for all days vs sample days.

    Args:
        daily: Populated daily DataFrame with ``LogQ``.
        sample: Populated sample DataFrame with ``LogQ``.
        ax: Optional matplotlib axes.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, ax = _get_ax(ax)
    data = [daily['LogQ'].dropna().values, sample['LogQ'].dropna().values]
    ax.boxplot(data, tick_labels=['All Days', 'Sample Days'])
    ax.set_ylabel('Log(Q)')
    ax.set_title('Discharge Distribution: All Days vs Sample Days')
    return fig


def plot_overview(daily, sample, fig=None):
    """2x2 panel combining q_time, conc_time, conc_q, box_conc_month.

    Args:
        daily: Populated daily DataFrame.
        sample: Populated sample DataFrame.
        fig: Optional matplotlib figure.

    Returns:
        ``matplotlib.figure.Figure``
    """
    fig, axes = _get_fig(fig, nrows=2, ncols=2, figsize=(10, 8))
    plot_q_time_daily(daily, ax=axes[0])
    plot_conc_time(sample, ax=axes[1])
    plot_conc_q(sample, ax=axes[2])
    box_conc_month(sample, ax=axes[3])
    fig.tight_layout()
    return fig
