# -*- coding: utf-8 -*-
# Copyright (C) 2024-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Contour plot."""


from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._event import ColorbarEventManager, FigureEventManager, figure_event_manager
from ._state import ColorbarPlot, figure_and_axes

# kw_only only supported from Python 3.10
KW_ONLY = {"kw_only": True} if "kw_only" in dataclass.__kwdefaults__ else {}


@dataclass(repr=False, **KW_ONLY)
class ContourPlot(ColorbarPlot):
    """State of contour plot.

    Args:
        figure: Plot figure.
        axes: Plot axes.
        axesimage: The :class:`~matplotlib.image.AxesImage` associated with the
           colorbar.
        divider: The :class:`~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
           used to create axes for the colorbar.
        cbar_axes: The axes of the colorbar.
        qcntset: Object returned by the call to
           :meth:`~matplotlib.axes.Axes.contour`.
    """

    qcntset: mpl.contour.QuadContourSet


def contour(
    z: np.ndarray,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    *,
    xlog: bool = False,
    ylog: bool = False,
    levels: Union[int, Sequence[float]] = 5,
    clabel_inline: bool = True,
    clabel_format: Optional[str] = None,
    clabel_fontsize: Optional[int] = 10,
    cmap: Optional[Union[Colormap, str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xylabel_fontsize: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fignum: Optional[int] = None,
    ax: Optional[Axes] = None,
) -> ContourPlot:
    """Contour plot of a 2D surface.

    Contour plot of a 2D surface together with pseudo-color
    image representation of the surface. Supports the following features:

    - If an axes is not specified (via parameter :code:`ax`), a new
      figure and axes are created, and
      :meth:`~matplotlib.figure.Figure.show` is called after drawing the
      plot.
    - Interactive features provided by :class:`FigureEventManager` and
      :class:`ColorbarEventManager` are supported in addition to the standard
      `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        z: Contour data 2d array to plot.
        x: Values for x-axis of the plot.
        y: Values for y-axis of the plot.
        xlog: Set x-axis to log scale.
        ylog: Set y-axis to log scale.
        levels: An int specifying the number of contours to plot, or a
            sequence specifying the specific contour levels to plot.
        clabel_inline: Value of parameter :code:`inline` of
            :meth:`~matplotlib.axes.Axes.clabel`.
        clabel_format: Format string for contour labels.
        clabel_fontsize: Contour label font size. No contour labels are
            displayed if set to 0 or ``None``.
        cmap: Color map for color mesh drawn under contours. If none
            specifed, defaults to :code:`matplotlib.cm.YlOrRd`.
        vmin: Set lower bound for the color map (see the corresponding
            parameter of :meth:`~matplotlib.axes.Axes.imshow`).
        vmax: Set upper bound for the color map (see the corresponding
            parameter of :meth:`~matplotlib.axes.Axes.imshow`).
        alpha: Underlying image display alpha value.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        xylabel_fontsize: Axis label font size. The default font size is
            used if set to ``None``.
        title: Figure title.
        figsize: Specify dimensions of figure to be creaed as a tuple
            (`width`, `height`) in inches.
        fignum: Figure number of figure to be created.
        ax: Plot in specified axes instead of creating one.

    Returns:
        Contour plot state object.
    """
    fig, ax, show = figure_and_axes(ax, figsize=figsize, fignum=fignum)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    if cmap is None:
        cmap = mpl.cm.YlOrRd  # pylint: disable=E1101

    x = np.arange(z.shape[1]) if x is None else np.array(x)
    y = np.arange(z.shape[0]) if y is None else np.array(y)
    xg, yg = np.meshgrid(x, y)

    qcntset = ax.contour(xg, yg, z, levels, colors="black")
    clabel_kwargs: Dict[str, Union[str, int]] = {}
    if clabel_fontsize is not None and clabel_fontsize > 0:
        clabel_kwargs["fontsize"] = clabel_fontsize
    if clabel_format is not None:
        clabel_kwargs["fmt"] = clabel_format
    if clabel_kwargs:
        ax.clabel(qcntset, inline=clabel_inline, **clabel_kwargs)

    qmesh = ax.pcolormesh(
        xg,
        yg,
        z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        shading="gouraud",
        clim=(vmin, vmax),
    )

    ax.fmt_xdata = "{: .2e}".format if xlog else "{: .2f}".format
    ax.fmt_ydata = "{: .2e}".format if ylog else "{: .2f}".format

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xylabel_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=xylabel_fontsize)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(qmesh, ax=ax, cax=cax)

    if show:
        fig.show()

    cplot = ContourPlot(
        figure=fig,
        axes=ax,
        axesimage=qmesh,
        divider=divider,
        cbar_axes=cax,
        qcntset=qcntset,
    )

    if not hasattr(fig, "_event_manager"):
        fem = FigureEventManager(fig)  # constructed object attaches itself to fig
    else:
        fem = figure_event_manager(fig)
    if not hasattr(ax, "_event_manager"):
        ColorbarEventManager(ax, fem, cplot)  # constructed object attaches itself to ax

    return cplot
