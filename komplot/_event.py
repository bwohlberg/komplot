# -*- coding: utf-8 -*-
# Copyright (C) 2024 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Event handling for interactive features."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure

from ._state import ColorbarPlot, GenericPlot, ZoomablePlot


class FigureEventManager:
    """Manager for figure-based events.

    Manage key-press events, and maintain a record of associated
    axes and their manager instances. The key-press events provide
    monitoring of keys used as modifiers for other events (e.g. mouse
    scroll while shift is depressed) and also support the following
    keyboard shortcuts:

    *q*
       Close figure. (This is also a standard keyboard shortcut.)

    *PageUp/PageDown*
       Increase or decrease figure size by a scaling factor.

    Note that key-press detection is not functional in Jupyter notebooks,
    including with the interactive `ipympl <https://matplotlib.org/ipympl/>`__
    matplotlib backend.
    """

    # Since it is not possible to access information on which keys are
    # pressed while handling events, monitor the status of a set of keys
    # that will be used as modifiers for other events (mouse scroll in
    # particular), keeping a record of when they are pressed and released.
    monitored_keys = ["shift", "left", "right"]

    def __init__(self, fig: Figure, fig_scale: float = 1.1):
        """
        Args:
            fig: Figure to which this manager is attached.
            fig_scale: Scaling factor for figure scaling keyboard shortcut.
        """
        self.fig = fig
        self.key_pressed = {k: False for k in self.monitored_keys}
        self.slice_share_axes: List[Axes] = []
        self.cmap_share_axes: List[Axes] = []
        self.axevman_from_ax: Dict[AxesEventManager, Axes] = {}

        def key_press(event: Event):
            """Callback for key press events."""
            if event.key == "q":
                plt.close(fig)
            elif event.key == "pageup":
                fig.set_size_inches(fig_scale * fig.get_size_inches(), forward=True)
            elif event.key == "pagedown":
                fig.set_size_inches(fig.get_size_inches() / fig_scale, forward=True)
            elif event.key in self.monitored_keys:
                self.key_pressed[event.key] = True

        def key_release(event: Event):
            """Callback for key release events.

            If the released key is in the list of monitored keys, update
            its status record."""
            if event.key in self.monitored_keys:
                self.key_pressed[event.key] = False

        def figure_leave(event: Event):
            """Callback for figure leave events.

            Reset the status of all monitored keys when the cursor leaves
            the figure since key release events will no longer be
            registered.
            """
            self.key_pressed = {k: False for k in self.monitored_keys}

        # Attach this event manager to the figure and connect callbacks
        if hasattr(fig, "_event_manager"):
            raise RuntimeError(f"Figure {fig} already has an event manager attached.")
        fig._event_manager = self
        fig.canvas.mpl_connect("key_press_event", key_press)
        fig.canvas.mpl_connect("key_release_event", key_release)
        fig.canvas.mpl_connect("figure_leave_event", figure_leave)

    def register_axevman_for_axes(self, ax: Axes, axem: AxesEventManager):
        """Register the axes event manager for an axes instance.

        Args:
            axem: Axes event manager.
            ax: Axes.
        """
        self.axevman_from_ax[ax] = axem

    def get_axevman_for_axes(self, ax: Axes):
        """Retrieve the axes event manager associated with an axes.

        Args:
            ax: Axes.

        Returns:
            The axes event manager associated with an axes.
        """
        return self.axevman_from_ax.get(ax)

    def set_slice_share(self, axes: Axes):
        """Define a set of volume slice sharing axes.

        Args:
            axes: List of axes.
        """
        if isinstance(axes, np.ndarray):
            axes = axes.ravel().tolist()
        self.slice_share_axes = axes

    def set_cmap_share(self, axes: Axes):
        """Define a set of colormap sharing axes.

        Args:
            axes: List of axes.
        """
        if isinstance(axes, np.ndarray):
            axes = axes.ravel().tolist()
        self.cmap_share_axes = axes

    @staticmethod
    def attached_manager(fig: Figure, error: bool = False):
        """Get the figure manager attached to a figure.

        Args:
            fig: Figure.
            error: If ``True``, raise an exception if no figure manager
                attached.

        Raises:
            RuntimeError: If `error` parameter is ``True`` and no figure
                manager attached.
        """
        if hasattr(fig, "_event_manager"):
            return fig._event_manager  # pylint: disable=W0212
        if error:
            raise RuntimeError(f"Figure {fig} has no attached FigureEventManager.")
        return None


def figure_event_manager(fig: Figure, error: bool = True) -> FigureEventManager:
    """Get the figure event manager attached to a figure.

    Args:
        fig: Figure.
        error: If ``True``, raise an exception if no figure manager
            attached.

    Raises:
        RuntimeError: If `error` parameter is ``True`` and no figure
            manager attached.
    """
    return FigureEventManager.attached_manager(fig, error=error)


class AxesEventManager:
    """Base class for axes-based event managers.

    Base class for axes-based event managers.
    """

    def __init__(
        self, axes: Axes, fig_event_man: FigureEventManager, plot: GenericPlot
    ):
        """
        Args:
            axes: Axes to which this manager is attached.
            fig_event_man: The figure event manage for the figure to
               which :code:`axes` belong.
            plot: A plot state object.
        """
        self.axes = axes
        self.fig_event_man = fig_event_man
        self.plot = plot
        fig_event_man.register_axevman_for_axes(axes, self)

        # See https://github.com/matplotlib/ipympl/issues/240 and
        #     https://github.com/matplotlib/ipympl/pull/235
        fig_event_man.fig.canvas.capture_scroll = True

        if hasattr(axes, "_event_manager"):
            raise RuntimeError(f"Axes {axes} already has an event manager attached.")
        axes._event_manager = self


class ZoomEventManager(AxesEventManager):
    """Manager for axis zoom events.

    Manage axis zoom via mouse wheel scroll.
    """

    plot: ZoomablePlot

    def __init__(
        self,
        axes: Axes,
        fig_event_man: FigureEventManager,
        zplot: ZoomablePlot,
        zoom_scale: float = 2.0,
    ):
        """
        Args:
            axes: Axes to which this manager is attached.
            fig_event_man: The figure event manage for the figure to
               which :code:`axes` belong.
            zplot: A plot state of type :class:`ZoomablePlot`.
            zoom_scale: Scaling factor for mouse wheel zoom.
        """
        super().__init__(axes, fig_event_man, zplot)
        self.zoom_scale = zoom_scale

        self.fig_event_man.fig.canvas.mpl_connect(
            "scroll_event", self.scroll_event_handler
        )
        axes.callbacks.connect("xlim_changed", self.xylim_changed_handler)
        axes.callbacks.connect("ylim_changed", self.xylim_changed_handler)

    def scroll_event_handler(self, event: Event):
        """Calback for mouse scroll events."""
        if event.inaxes == self.axes:
            if not any(self.fig_event_man.key_pressed.values()):  # zoom
                self.zoom_event_handler(event)

    def zoom_event_handler(self, event: Event):
        """Handle axes zoom event."""
        if event.button == "up":  # Deal with zoom in
            scale_factor = 1.0 / self.zoom_scale
        elif event.button == "down":  # Deal with zoom out
            scale_factor = self.zoom_scale
        # Get event location
        xdata = event.xdata
        ydata = event.ydata
        # Ensure cursor is over valid region of plot
        if not (xdata is None or ydata is None):
            self.plot.zoom_view(xdata, ydata, scale_factor)

    def xylim_changed_handler(self, axes: Axes):
        """Callback for changes to axes limits."""
        self.plot.zoom_toolbar_message()


class ColorbarEventManager(ZoomEventManager):
    """Manager for colorbar events.

    Manage colormap :code:`vmin` and :code:`vmax` adjustment via mouse
    scroll in a colorbar. Scrolling in the bottom/left half of the
    colorbar adjusts :code:`vmin`, and scrolling in the top/right half of
    the colorbar adjusts :code:`vmax`. Axis zoom events on mouse scroll
    are also supported via the :class:`ZoomEventManager` base class.
    """

    plot: ColorbarPlot

    def __init__(
        self,
        axes: Axes,
        fig_event_man: FigureEventManager,
        cbplot: ColorbarPlot,
        zoom_scale: float = 2.0,
        cmap_delta: float = 0.02,
    ):
        """
        Args:
            axes: Axes to which this manager is attached.
            fig_event_man: The figure event manage for the figure to
               which :code:`axes` belong.
            cbplot: A plot state of type :class:`ColorbarPlot`.
            zoom_scale: Scaling factor for mouse wheel zoom.
            cmap_delta: Fraction of colormap range for vmin/vmax shifts.
        """
        super().__init__(axes, fig_event_man, cbplot, zoom_scale=zoom_scale)
        self.cmap_delta = cmap_delta

    def scroll_event_handler(self, event: Event):
        """Calback for mouse scroll events."""
        if event.inaxes == self.plot.cbar_axes:  # cmap range change
            rel_pos = self.cbar_event_rel_pos(event)
            if self.fig_event_man.cmap_share_axes:
                for csax in self.fig_event_man.cmap_share_axes:
                    axevman = self.fig_event_man.get_axevman_for_axes(csax)
                    axevman.cmap_vminmax_event_handler(event, rel_pos)
            else:
                self.cmap_vminmax_event_handler(event, rel_pos)
        else:
            super().scroll_event_handler(event)

    def cbar_event_rel_pos(self, event: Event):
        """Determine relative position of event in a colorbar."""
        if self.plot.cbar_axes is None or event.inaxes != self.plot.cbar_axes:
            return None
        box = self.plot.cbar_axes.get_window_extent().bounds
        if (
            self.plot.cbar_axes._colorbar.orientation  # pylint: disable=W0212
            == "vertical"
        ):
            rel_pos = (event.y - box[1]) / box[3]
        else:
            rel_pos = (event.x - box[0]) / box[2]
        return rel_pos

    def cmap_vminmax_event_handler(self, event, rel_pos: float):
        """Colorbar limits adjust callback."""
        sign = 1 if event.button == "up" else -1
        if rel_pos is not None:
            if rel_pos < 0.5:
                self.plot.shift_cmap_vmin(sign * self.cmap_delta)
            elif rel_pos > 0.5:
                self.plot.shift_cmap_vmax(sign * self.cmap_delta)
