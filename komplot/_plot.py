# -*- coding: utf-8 -*-
# Copyright (C) 2024-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Point/line plotting."""


from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib as mpl
from matplotlib.axes import Axes

from ._event import FigureEventManager, ZoomEventManager, figure_event_manager
from ._state import ZoomablePlot, figure_and_axes

try:
    import mplcursors as mplcrs
except ImportError:
    HAVE_MPLCRS = False
else:
    HAVE_MPLCRS = True


# kw_only only supported from Python 3.10
KW_ONLY = {"kw_only": True} if "kw_only" in dataclass.__kwdefaults__ else {}


@dataclass(repr=False, **KW_ONLY)
class LinePlot(ZoomablePlot):
    """State of 2d line/point plot.

    Args:
        figure: Plot figure.
        axes: Plot axes.
        line2d: Object returned by call to
           :meth:`~matplotlib.axes.Axes.plot`.
    """

    line2d: mpl.lines.Line2D


def plot(
    *args,
    xlog: bool = False,
    ylog: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend: Optional[Sequence[str]] = None,
    legend_loc: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fignum: Optional[int] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> LinePlot:
    """Plot points or lines in 2D.

    Plot points or lines in 2D. Largely replicates the interface of
    :meth:`~matplotlib.axes.Axes.plot`, with the following execeptions

    - If an axes is not specified (via parameter :code:`ax`), a new
      figure and axes are created, and
      :meth:`~matplotlib.figure.Figure.show` is called after drawing the
      plot.
    - Interactive features provided by :class:`FigureEventManager` and
      :class:`ZoomEventManager` are supported in addition to the standard
      `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        *args: Plot specification in form :code:`[x], y, [fmt],` etc.
            corresponding to the signature of
            :meth:`~matplotlib.axes.Axes.plot`.
        xlog: Set x axis to log scale if ``True``.
        ylog: Set y axis to log scale if ``True``.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Figure title.
        legend: List of legend strings.
        legend_loc: Legend location string as supported by :code:`loc`
            parameter of :meth:`~matplotlib.axes.Axes.legend`.
        figsize: Specify dimensions of figure to be creaed as a tuple
            (`width`, `height`) in inches.
        fignum: Figure number of figure to be created.
        ax: Plot in specified axes instead of creating one.
        **kwargs: All keyword arguments of
            :meth:`~matplotlib.axes.Axes.plot` are supported, such as
            those specifying :class:`~matplotlib.lines.Line2D` properties
            (if not specified, the defaults for line width (:code:`lw`)
            and marker size (:code:`ms`) are 1.5 and 6.0 respectively),
            as well as the following additional keyword arguments.
            When parameters :code:`xlog` or :code:`ylog` are specified,
            additional properties of the log-scale axes may be specified
            via parameters  :code:`base`, :code:`subs`,
            :code:`nonpositive`, :code:`basex`, :code:`subsx`,
            :code:`nonposx`, :code:`basey`, :code:`subsy`, and
            :code:`nonposy`, supported by
            :meth:`~matplotlib.axes.Axes.loglog`.

    Returns:
        Line/point plot state object.
    """
    fig, ax, show = figure_and_axes(ax, figsize=figsize, fignum=fignum)

    # Set defaults for line width and marker size
    if "lw" not in kwargs and "linewidth" not in kwargs:
        kwargs["lw"] = 1.5
    if "ms" not in kwargs and "markersize" not in kwargs:
        kwargs["ms"] = 6.0

    # See implementation of matplotlib.axes.Axes.loglog
    kwxy = {}
    for k in kwargs:
        if k in ["base", "subs", "nonpositive"]:
            kwxy[k] = kwargs.pop(k)
    kwx = {}
    for k in kwargs:
        if k in ["basex", "subsx", "nonposx"]:
            kwx[k] = kwargs.pop(k)
    kwy = {}
    for k in kwargs:
        if k in ["basey", "subsy", "nonposy"]:
            kwy[k] = kwargs.pop(k)
    if xlog:
        ax.set_xscale("log", **{**kwxy, **kwx})
    if ylog:
        ax.set_yscale("log", **{**kwxy, **kwy})

    line2d = ax.plot(*args, **kwargs)

    ax.fmt_xdata = "{: .2f}".format
    ax.fmt_ydata = "{: .2f}".format

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend is not None:
        ax.legend(legend, loc=legend_loc)

    if HAVE_MPLCRS:
        mplcrs.cursor(line2d)

    if show:
        fig.show()

    lineplot = LinePlot(figure=fig, axes=ax, line2d=line2d)

    if not hasattr(fig, "_event_manager"):
        fem = FigureEventManager(fig)  # constructed object attaches itself to fig
    else:
        fem = figure_event_manager(fig)
    if not hasattr(ax, "_event_manager"):
        ZoomEventManager(ax, fem, lineplot)  # constructed object attaches itself to ax

    return lineplot
