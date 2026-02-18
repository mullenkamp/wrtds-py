"""WRTDS plotting functions."""

from wrtds.plots.data_overview import (
    box_conc_month,
    box_q_twice,
    plot_conc_q,
    plot_conc_time,
    plot_overview,
    plot_q_time_daily,
)
from wrtds.plots.diagnostics import (
    box_resid_month,
    flux_bias_multi,
    plot_conc_pred,
    plot_flux_pred,
    plot_resid_pred,
    plot_resid_q,
    plot_resid_time,
)
from wrtds.plots.results import (
    plot_conc_hist,
    plot_conc_q_smooth,
    plot_conc_time_smooth,
    plot_contours,
    plot_diff_contours,
    plot_flux_hist,
)

__all__ = [
    # data_overview
    'box_conc_month',
    'box_q_twice',
    'plot_conc_q',
    'plot_conc_time',
    'plot_overview',
    'plot_q_time_daily',
    # diagnostics
    'box_resid_month',
    'flux_bias_multi',
    'plot_conc_pred',
    'plot_flux_pred',
    'plot_resid_pred',
    'plot_resid_q',
    'plot_resid_time',
    # results
    'plot_conc_hist',
    'plot_conc_q_smooth',
    'plot_conc_time_smooth',
    'plot_contours',
    'plot_diff_contours',
    'plot_flux_hist',
]
