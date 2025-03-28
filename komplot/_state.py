# -*- coding: utf-8 -*-
# Copyright (C) 2024-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Plot management classes."""


from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

# kw_only only supported from Python 3.10
KW_ONLY = {"kw_only": True} if "kw_only" in dataclass.__kwdefaults__ else {}


def figure_and_axes(
    ax: Optional[Axes],
    figsize: Optional[Tuple[int, int]] = None,
    fignum: Optional[int] = None,
    proj3d: bool = False,
) -> Tuple[Figure, Axes, bool]:
    """Get figure from axes or create new figure and axes.

    Args:
        ax: If ``None`` create a new figure and axes,
            otherwise use the figure associated with
            the specified axes.
        figsize: Specify dimensions of figure to be creaed as a tuple
            (`width`, `height`) in inches.
        fignum: Figure number of figure to be created.
        proj3d: If ``True`` ensure that axes is 3D.

    Returns: A tuple consisting of the figure, axes, and a flag
        indicating whether a new figure was created.
    """
    if ax is None:
        spkw = {"projection": "3d"} if proj3d else {}
        fig, ax = plt.subplots(subplot_kw=spkw, num=fignum, figsize=figsize)
        new_fig = True
    else:
        fig = ax.get_figure()
        if proj3d:
            # See https://stackoverflow.com/a/43563804
            #     https://stackoverflow.com/a/35221116
            if ax.name != "3d":
                ax.remove()
                ax = fig.add_subplot(ax.get_subplotspec(), projection="3d")
        new_fig = False
    return fig, ax, new_fig


@dataclass(repr=False, **KW_ONLY)
class GenericPlot:
    """Generic plot state.

    Args:
        figure: Plot figure.
        axes: Plot axes.
    """

    figure: Figure
    axes: Axes

    def toolbar_message(self, msg: str):
        """Display message in toolbar if present.

        Args:
            msg: Message string.
        """
        if self.axes.figure.canvas.toolbar is not None:
            self.axes.figure.canvas.toolbar.set_message(msg)


@dataclass(repr=False, **KW_ONLY)
class ZoomablePlot(GenericPlot):
    """State for a plot supporting axis zoom.

    Args:
        figure: Plot figure.
        axes: Plot axes.
    """

    xlim_ref: Tuple[float, float] = field(init=False)
    ylim_ref: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        """Initialize axis limit record.

        Initalize :code:`xlim_ref` and :code:`ylim_ref` attributes from
        axes limits on initialization.
        """
        self.xlim_ref = self.axes.get_xlim()
        self.ylim_ref = self.axes.get_ylim()

    def zoom_view(self, xdata: float, ydata: float, scale_factor: float):
        """Zoom the view in this axes.

        Args:
            xdata: Data `x` coordinate of event location.
            ydata: Data `y` coordinate of event location.
            scale_factor: Zoom scale factor (greater than unity
              for zoom out, less than unity for zoom in).
        """
        ax = self.axes
        # Get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        # Get distance from the cursor to the edge of the figure frame
        x_left = xdata - cur_xlim[0]
        x_right = cur_xlim[1] - xdata
        y_top = ydata - cur_ylim[0]
        y_bottom = cur_ylim[1] - ydata

        # Calculate new x and y limits
        new_xlim = (xdata - x_left * scale_factor, xdata + x_right * scale_factor)
        new_ylim = (ydata - y_top * scale_factor, ydata + y_bottom * scale_factor)

        # Ensure that x limit range is no larger than that of the reference
        if np.diff(new_xlim) > np.diff(self.xlim_ref):
            new_xlim *= np.diff(self.xlim_ref) / np.diff(new_xlim)
        # Ensure that lower x limit is not less than that of the reference
        if new_xlim[0] < self.xlim_ref[0]:
            new_xlim += np.array(self.xlim_ref[0] - new_xlim[0])
        # Ensure that upper x limit is not greater than that of the reference
        if new_xlim[1] > self.xlim_ref[1]:
            new_xlim -= np.array(new_xlim[1] - self.xlim_ref[1])

        # Ensure that ylim tuple has the smallest value first
        if self.ylim_ref[1] < self.ylim_ref[0]:
            ylim_ref = self.ylim_ref[::-1]
            new_ylim = new_ylim[::-1]
        else:
            ylim_ref = self.ylim_ref

        # Ensure that y limit range is no larger than that of the reference
        if np.diff(new_ylim) > np.diff(ylim_ref):
            new_ylim *= np.diff(ylim_ref) / np.diff(new_ylim)
        # Ensure that lower y limit is not less than that of the reference
        if new_ylim[0] < ylim_ref[0]:
            new_ylim += np.array(ylim_ref[0] - new_ylim[0])
        # Ensure that upper y limit is not greater than that of the reference
        if new_ylim[1] > ylim_ref[1]:
            new_ylim -= np.array(new_ylim[1] - ylim_ref[1])

        # Return the ylim tuple to its original order
        if self.ylim_ref[1] < self.ylim_ref[0]:
            new_ylim = new_ylim[::-1]

        # Set new x and y limits
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        # Force redraw
        ax.figure.canvas.draw_idle()

    def zoom_toolbar_message(self):
        """Display toolbar message.

        This function is called on axis zoom events."""
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        if ylim[0] > ylim[1]:
            ylim = ylim[::-1]
        msg = f"Zoom to [{ylim[0]:.1f}:{ylim[1]:.1f}, {xlim[0]:.1f}:{xlim[1]:.1f}]"
        self.toolbar_message(msg)


@dataclass(repr=False, **KW_ONLY)
class ColorbarPlot(ZoomablePlot):
    """State of plot supporting axis zoom and a colorbar.

    Args:
        figure: Plot figure.
        axes: Plot axes.
        axesimage: The :class:`~matplotlib.image.AxesImage` associated with the
           colorbar.
        divider: The :class:`~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
           used to create axes for the colorbar.
        cbar_axes: The axes of the colorbar.
    """

    axesimage: mpl.image.AxesImage
    divider: AxesDivider
    cbar_axes: Axes
    vmin_ref: float = field(init=False)
    vmax_ref: float = field(init=False)

    def __post_init__(self):
        """Initialize colormap limits record.

        Initalize :code:`vmin_ref` and :code:`vmax_ref` attributes from
        colormap limits on initialization.
        """
        self.vmin_ref = self.axesimage.norm.vmin
        self.vmax_ref = self.axesimage.norm.vmax
        super().__post_init__()

    def shift_cmap_vmin(self, rel_delta: float):
        """Change colormap :code:`vmin`.

        Args:
            rel_delta: Signed relative change in colormap :code:`vmin`
                value.
        """
        im = self.axesimage
        abs_delta = rel_delta * (self.vmax_ref - self.vmin_ref)
        new_vmin = im.norm.vmin + abs_delta
        if new_vmin < self.vmin_ref:
            new_vmin = self.vmin_ref
        if new_vmin < im.norm.vmax:
            im.norm.vmin = new_vmin
        self.axes.figure.canvas.draw_idle()

        msg = f"Color map vmin set to {im.norm.vmin:.1f}"
        self.toolbar_message(msg)

    def shift_cmap_vmax(self, rel_delta: float):
        """Change colormap :code:`vmax`.

        Args:
            rel_delta: Signed relative change in colormap :code:`vmax`
                value.
        """
        im = self.axesimage
        abs_delta = rel_delta * (self.vmax_ref - self.vmin_ref)
        new_vmax = im.norm.vmax + abs_delta
        if new_vmax > self.vmax_ref:
            new_vmax = self.vmax_ref
        if new_vmax > im.norm.vmin:
            im.norm.vmax = new_vmax
        self.axes.figure.canvas.draw_idle()

        msg = f"Color map vmax set to {im.norm.vmax:.1f}"
        self.toolbar_message(msg)
