"""Shared plotting utilities."""

import matplotlib.pyplot as plt


def _get_ax(ax=None, **fig_kw):
    """If *ax* is None, create a new figure + axes pair.

    Args:
        ax: Optional existing ``matplotlib.axes.Axes``.
        **fig_kw: Keyword arguments forwarded to ``plt.subplots`` when
            creating a new figure.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
    else:
        fig = ax.get_figure()
    return fig, ax


def _get_fig(fig=None, nrows=1, ncols=1, **fig_kw):
    """If *fig* is None, create a new figure with subplots.

    Args:
        fig: Optional existing ``matplotlib.figure.Figure``.
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        **fig_kw: Keyword arguments forwarded to ``plt.subplots``.

    Returns:
        ``(fig, axes)`` tuple where *axes* is a flat array of axes.
    """
    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, **fig_kw)
    else:
        axes = fig.subplots(nrows, ncols)
    import numpy as np
    axes = np.atleast_1d(axes).ravel()
    return fig, axes
