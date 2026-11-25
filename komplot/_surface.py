# -*- coding: utf-8 -*-
# Copyright (C) 2024-2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Surface plot."""

from typing import Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]

from ._state import Plot, figure_and_axes


class SurfacePlot(Plot):
    """A plot of a 2D surface in 3D.

    A plot of a 2D surface in 3D, together with an optional contour plot
    in the `x`-`y` plane below the surface. Supports the interactive
    features provided by :class:`.FigureEventManager`, in addition to the
    rotation and zoom of a 3D axes provided by matplotlib itself. It is
    usually constructed via the :func:`surface` function, which is a
    shortcut for construction of an instance of this class.

    Attributes:
        figure: Plot figure.
        axes: Plot axes.
        poly3dc: Object returned by the call to
           :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`.
        qcntset: Object returned by the call to
           :meth:`~matplotlib.axes.Axes.contour`, or ``None`` if no
           contours were plotted.
    """

    poly3dc: Poly3DCollection
    qcntset: Optional[mpl.contour.QuadContourSet]

    def __init__(
        self,
        z: np.ndarray,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        *,
        elev: Optional[float] = None,
        azim: Optional[float] = None,
        roll: Optional[float] = None,
        alpha: float = 1.0,
        cmap: Optional[Union[Colormap, str]] = None,
        levels: Optional[Union[int, Sequence[int]]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        labelpad: float = 8.0,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fignum: Optional[int] = None,
        ax: Optional[Axes] = None,
    ):
        """
        See :func:`surface` for a description of the parameters.
        """
        fig, axs, new_fig = figure_and_axes(
            ax, figsize=figsize, fignum=fignum, proj3d=True
        )
        super().__init__(figure=fig, axes=axs, new_figure=new_fig)

        if elev is not None or azim is not None or roll is not None:
            axs.view_init(elev=elev, azim=azim, roll=roll)  # type: ignore[attr-defined]

        if cmap is None:
            cmap = mpl.cm.YlOrRd  # type: ignore[attr-defined]  # pylint: disable=E1101

        if x is None:
            x = np.arange(z.shape[1])
        if y is None:
            y = np.arange(z.shape[0])
        xg, yg = np.meshgrid(x, y)
        self.poly3dc = axs.plot_surface(  # type: ignore[attr-defined]
            xg, yg, z, rstride=1, cstride=1, alpha=alpha, cmap=cmap
        )

        if levels is None:
            self.qcntset = None
        else:
            offset = np.around(z.min() - 0.2 * (z.max() - z.min()), 3)
            self.qcntset = axs.contour(
                xg,
                yg,
                z,
                levels,
                cmap=cmap,
                linewidths=2,
                linestyles="solid",
                offset=offset,
            )
            axs.set_zlim(offset, axs.get_zlim()[1])  # type: ignore[attr-defined]

        axs.fmt_xdata = "{: .2f}".format
        axs.fmt_ydata = "{: .2f}".format
        axs.fmt_zdata = "{: .2f}".format  # type: ignore[attr-defined]

        if title is not None:
            axs.set_title(title)
        if xlabel is not None:
            axs.set_xlabel(xlabel, labelpad=labelpad)
        if ylabel is not None:
            axs.set_ylabel(ylabel, labelpad=labelpad)
        if zlabel is not None:
            axs.set_zlabel(zlabel, labelpad=labelpad)  # type: ignore[attr-defined]

        self.show_new_figure()


def surface(
    z: np.ndarray,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    *,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    roll: Optional[float] = None,
    alpha: float = 1.0,
    cmap: Optional[Union[Colormap, str]] = None,
    levels: Optional[Union[int, Sequence[int]]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    labelpad: float = 8.0,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fignum: Optional[int] = None,
    ax: Optional[Axes] = None,
) -> SurfacePlot:
    """Plot a 2D surface in 3D.

    Plot a 2D surface in 3D, together with an optional contour plot in
    the `x`-`y` plane below the surface. Supports the following features:

    - If an axes is not specified (via parameter :code:`ax`), a new
      figure and axes are created, and
      :meth:`~matplotlib.figure.Figure.show` is called after drawing the
      plot.
    - Interactive features provided by :class:`FigureEventManager` are
      supported in addition to the standard
      `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        z: Surface data 2d array to plot.
        x: Values for x-axis of the plot.
        y: Values for y-axis of the plot.
        elev: Elevation angle (in degrees); see corresponding parameter
            of :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.view_init`.
        azim: Azimuth angle (in degrees); see corresponding parameter
            of :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.view_init`.
        roll: Roll angle (in degrees); see corresponding parameter
            of :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.view_init`.
        alpha: Transparency, if specified should be a float between 0.0
            and 1.0.
        cmap: Color map for surface. If none specifed, defaults to
            :code:`matplotlib.cm.YlOrRd`.
        levels: If not ``None``, plot contours of the surface on the lower end
            of the z-axis. An int specifies the number of contours to
            plot, and a sequence specifies the specific contour levels to
            plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        zlabel: Label for z-axis.
        labelpad: Label padding.
        title: Figure title.
        figsize: Specify dimensions of figure to be creaed as a tuple
            (`width`, `height`) in inches.
        fignum: Figure number of figure to be created.
        ax: Plot in specified axes instead of creating one.

    Returns:
        Surface plot object.
    """
    return SurfacePlot(
        z,
        x,
        y,
        elev=elev,
        azim=azim,
        roll=roll,
        alpha=alpha,
        cmap=cmap,
        levels=levels,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        labelpad=labelpad,
        title=title,
        figsize=figsize,
        fignum=fignum,
        ax=ax,
    )
