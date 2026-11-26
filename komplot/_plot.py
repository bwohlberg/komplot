# -*- coding: utf-8 -*-
# Copyright (C) 2024-2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Point/line plotting."""

from typing import List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
from matplotlib.axes import Axes

from ._state import LogScaleType, Plot, ZoomMixin, figure_and_axes, log_scale_kwargs

try:
    import mplcursors as mplcrs  # type: ignore[import-not-found]
except ImportError:
    HAVE_MPLCRS = False
else:
    HAVE_MPLCRS = True


LegendLocType = Union[
    Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ],
    Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    Tuple[float, float],
    None,
]


class LinePlot(ZoomMixin, Plot):
    """A 2d line/point plot.

    A 2d line/point plot, supporting the interactive features provided by
    :class:`.FigureEventManager` and :class:`.ZoomMixin`. It is usually
    constructed via the :func:`plot` function, which is a shortcut for
    construction of an instance of this class.

    Attributes:
        figure: Plot figure.
        axes: Plot axes.
        line2d: Object returned by call to
           :meth:`~matplotlib.axes.Axes.plot`.
    """

    line2d: List[mpl.lines.Line2D]

    def __init__(
        self,
        *args,
        xlog: LogScaleType = False,
        ylog: LogScaleType = False,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        legend: Optional[Sequence[str]] = None,
        legend_loc: LegendLocType = None,
        figsize: Optional[Tuple[int, int]] = None,
        fignum: Optional[int] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        """
        See :func:`plot` for a description of the parameters.
        """
        fig, axs, new_fig = figure_and_axes(ax, figsize=figsize, fignum=fignum)
        super().__init__(figure=fig, axes=axs, new_figure=new_fig)

        # Set defaults for line width and marker size
        if "lw" not in kwargs and "linewidth" not in kwargs:
            kwargs["lw"] = 1.5
        if "ms" not in kwargs and "markersize" not in kwargs:
            kwargs["ms"] = 6.0

        xlog_kwargs = log_scale_kwargs(xlog)
        if xlog_kwargs is not None:
            axs.set_xscale("log", **xlog_kwargs)
        ylog_kwargs = log_scale_kwargs(ylog)
        if ylog_kwargs is not None:
            axs.set_yscale("log", **ylog_kwargs)

        self.line2d = axs.plot(*args, **kwargs)

        axs.fmt_xdata = "{: .2f}".format
        axs.fmt_ydata = "{: .2f}".format

        if title is not None:
            axs.set_title(title)
        if xlabel is not None:
            axs.set_xlabel(xlabel)
        if ylabel is not None:
            axs.set_ylabel(ylabel)
        if legend is not None:
            axs.legend(legend, loc=legend_loc)

        if HAVE_MPLCRS:
            mplcrs.cursor(self.line2d)

        self.init_zoom()
        self.show_new_figure()


def plot(
    *args,
    xlog: LogScaleType = False,
    ylog: LogScaleType = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend: Optional[Sequence[str]] = None,
    legend_loc: LegendLocType = None,
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
      :class:`ZoomMixin` are supported in addition to the standard
      `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        *args: Plot specification in form :code:`[x], y, [fmt],` etc.
            corresponding to the signature of
            :meth:`~matplotlib.axes.Axes.plot`.
        xlog: Set x axis to log scale if ``True``. Properties of the log
            scale axis, such as :code:`base`, :code:`subs`, and
            :code:`nonpositive`, may be specified by passing a dict of
            keyword arguments for
            :meth:`~matplotlib.axes.Axes.set_xscale` instead of a bool,
            which also implies the use of a log scale.
        ylog: Set y axis to log scale if ``True``. Properties of the log
            scale axis may be specified by passing a dict of keyword
            arguments for :meth:`~matplotlib.axes.Axes.set_yscale`
            instead of a bool, as for parameter :code:`xlog`.
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
            and marker size (:code:`ms`) are 1.5 and 6.0 respectively).

    Returns:
        Line/point plot object.
    """
    return LinePlot(
        *args,
        xlog=xlog,
        ylog=ylog,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend=legend,
        legend_loc=legend_loc,
        figsize=figsize,
        fignum=fignum,
        ax=ax,
        **kwargs,
    )
