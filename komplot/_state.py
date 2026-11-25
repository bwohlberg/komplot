# -*- coding: utf-8 -*-
# Copyright (C) 2024-2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Plot base class and feature mixin classes.

Each plot type is implemented as a class that combines plot state,
construction of the matplotlib artists, and handling of the events
supporting its interactive features. The features themselves are
provided by mixin classes, each of which contributes state,
user-callable methods, and event handlers, so that a plot type declares
the features it supports via its base classes. For example

::

    class VolumeView(SliceMixin, ImageMixin, ColorbarMixin, ZoomMixin, Plot):

declares a plot type supporting slice selection, image display, colormap
range adjustment, and axis zoom. Mouse scroll events are offered to the
handlers of each feature in the order in which the mixins are declared,
the first handler that claims the event terminating the dispatch.

This module defines the plot base class and the feature mixins that are
not specific to a single plot type; :class:`.ImageMixin`, which supports
both :class:`.ImageView` and :class:`.VolumeView`, is defined with the
former.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.axes_divider import (
    AxesDivider,  # type: ignore[import-untyped]
)

from ._event import attach_figure_event_manager

if TYPE_CHECKING:
    from ._share import ShareGroup


LogScaleType = Union[bool, Dict[str, Any]]

ScrollHandlerType = Callable[[Event], bool]


def log_scale_kwargs(log: LogScaleType) -> Optional[Dict[str, Any]]:
    """Convert a log scale specification to axis scale keyword arguments.

    Args:
        log: A bool specifying whether to use a log scale, or a dict of
           keyword arguments for :meth:`~matplotlib.axes.Axes.set_xscale`
           or :meth:`~matplotlib.axes.Axes.set_yscale`, which also
           implies the use of a log scale.

    Returns:
        A dict of axis scale keyword arguments if a log scale is
        required, otherwise ``None``.
    """
    if isinstance(log, dict):
        return log
    return {} if log else None


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
        fig = ax.get_figure()  # type: ignore[assignment]
        assert isinstance(fig, Figure)
        if proj3d:
            # See https://stackoverflow.com/a/43563804
            #     https://stackoverflow.com/a/35221116
            if ax.name != "3d":
                ax.remove()
                ax = fig.add_subplot(ax.get_subplotspec(), projection="3d")
        new_fig = False
    return fig, ax, new_fig


class Plot:
    """Base class for all plot types.

    Base class for all plot types, providing the figure and axes of the
    plot, registration with the event manager of the figure, and dispatch
    of mouse scroll events to the handlers contributed by feature mixin
    classes.

    Attributes:
        figure: Plot figure.
        axes: Plot axes.
        new_figure: ``True`` if the figure was created for this plot.
        event_manager: Event manager of the plot figure.
    """

    def __init__(self, figure: Figure, axes: Axes, new_figure: bool = False):
        """
        Args:
            figure: Plot figure.
            axes: Plot axes.
            new_figure: ``True`` if the figure was created for this plot.
        """
        self.figure = figure
        self.axes = axes
        self.new_figure = new_figure
        self._shown = False
        self.event_manager = attach_figure_event_manager(figure)
        self.event_manager.register_plot(self)

    def show_new_figure(self):
        """Show the figure if it was created for this plot.

        Show the figure if it was created for this plot and has not
        already been shown. It is called at the end of the initializer of
        each plot type, and has no effect when the plot was drawn in an
        axes specified by the user.
        """
        if self.new_figure and not self._shown:
            self._shown = True
            self.figure.show()

    def owns_axes(self, axes: Axes) -> bool:
        """Determine whether an axes belongs to this plot.

        Args:
            axes: Axes to test.

        Returns:
            ``True`` if :code:`axes` belongs to this plot.
        """
        return axes is self.axes

    def toolbar_message(self, msg: str):
        """Display message in toolbar if present.

        Args:
            msg: Message string.
        """
        if self.axes.figure.canvas.toolbar is not None:
            self.axes.figure.canvas.toolbar.set_message(msg)

    def scroll_handlers(self) -> Tuple[ScrollHandlerType, ...]:
        """Get the mouse scroll event handlers of this plot.

        Each feature mixin class extends the tuple returned by this
        method with its own handler, in order of declaration of the mixin
        classes.

        Returns:
            Tuple of scroll event handlers.
        """
        return ()

    def handle_scroll(self, event: Event) -> bool:
        """Offer a mouse scroll event to the handlers of this plot.

        The event is offered to each handler in turn, terminating when
        one of them claims it.

        Args:
            event: Mouse scroll event.

        Returns:
            ``True`` if the event was claimed by a handler.
        """
        for handler in self.scroll_handlers():
            if handler(event):
                return True
        return False


class ZoomMixin:
    """Mixin class providing axis zoom via mouse wheel scroll.

    Mouse wheel scroll in the plot axes, without any modifier key
    depressed, zooms in or out at the cursor location, the axis limits
    being constrained to those recorded on initialization.

    Attributes:
        zoom_scale: Scaling factor for mouse wheel zoom.
        xlim_ref: Reference x-axis limits.
        ylim_ref: Reference y-axis limits.
    """

    axes: Axes
    event_manager: Any
    zoom_scale: float
    xlim_ref: Tuple[float, float]
    ylim_ref: Tuple[float, float]

    def init_zoom(self, zoom_scale: float = 2.0):
        """Initialize zoom support.

        Record the current axis limits as the reference limits, and
        connect the axis limit change callbacks. It should be called
        after the plot content determining the axis limits has been
        drawn.

        Args:
            zoom_scale: Scaling factor for mouse wheel zoom.
        """
        self.zoom_scale = zoom_scale
        self.xlim_ref = self.axes.get_xlim()
        self.ylim_ref = self.axes.get_ylim()
        self.axes.callbacks.connect("xlim_changed", self._xylim_changed_handler)
        self.axes.callbacks.connect("ylim_changed", self._xylim_changed_handler)

    def scroll_handlers(self) -> Tuple[ScrollHandlerType, ...]:
        """Prepend the zoom handler to the scroll event handlers."""
        return (self._scroll_zoom_handler,) + super().scroll_handlers()  # type: ignore[misc]

    def _scroll_zoom_handler(self, event: Event) -> bool:
        """Handle a mouse scroll event as an axis zoom.

        Args:
            event: Mouse scroll event.

        Returns:
            ``True`` if the event was claimed as a zoom.
        """
        assert hasattr(event, "inaxes") and hasattr(event, "button")
        assert hasattr(event, "xdata") and hasattr(event, "ydata")
        if event.inaxes is not self.axes:
            return False
        if any(self.event_manager.key_pressed.values()):  # modifier key depressed
            return False
        if event.button == "up":  # zoom in
            scale_factor = 1.0 / self.zoom_scale
        elif event.button == "down":  # zoom out
            scale_factor = self.zoom_scale
        else:
            return False
        # Ensure cursor is over valid region of plot
        if event.xdata is None or event.ydata is None:
            return False
        self.zoom_view(event.xdata, event.ydata, scale_factor)
        return True

    def _xylim_changed_handler(self, axes: Axes):
        """Callback for changes to axes limits."""
        self.zoom_toolbar_message()

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
            new_xlim *= np.diff(self.xlim_ref) / np.diff(new_xlim)  # type: ignore[assignment]
        # Ensure that lower x limit is not less than that of the reference
        if new_xlim[0] < self.xlim_ref[0]:
            new_xlim += np.array(self.xlim_ref[0] - new_xlim[0])  # type: ignore[assignment]
        # Ensure that upper x limit is not greater than that of the reference
        if new_xlim[1] > self.xlim_ref[1]:
            new_xlim -= np.array(new_xlim[1] - self.xlim_ref[1])  # type: ignore[assignment]

        # Ensure that ylim tuple has the smallest value first
        if self.ylim_ref[1] < self.ylim_ref[0]:
            ylim_ref = self.ylim_ref[::-1]
            new_ylim = new_ylim[::-1]
        else:
            ylim_ref = self.ylim_ref

        # Ensure that y limit range is no larger than that of the reference
        if np.diff(new_ylim) > np.diff(ylim_ref):
            new_ylim *= np.diff(ylim_ref) / np.diff(new_ylim)  # type: ignore[assignment]
        # Ensure that lower y limit is not less than that of the reference
        if new_ylim[0] < ylim_ref[0]:
            new_ylim += np.array(ylim_ref[0] - new_ylim[0])  # type: ignore[assignment]
        # Ensure that upper y limit is not greater than that of the reference
        if new_ylim[1] > ylim_ref[1]:
            new_ylim -= np.array(new_ylim[1] - ylim_ref[1])  # type: ignore[assignment]

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
        self.toolbar_message(msg)  # type: ignore[attr-defined]


class ColorbarMixin:
    """Mixin class providing colormap range adjustment via mouse wheel scroll.

    Mouse wheel scroll in the bottom/left half of the colorbar adjusts
    the colormap :code:`vmin`, and scroll in the top/right half adjusts
    the colormap :code:`vmax`, the limits being constrained to those
    recorded on initialization. If the plot belongs to a colormap share
    group (see :func:`komplot.share_cmap`), the adjustment is applied to
    every member of the group.

    Attributes:
        axesimage: The :class:`~matplotlib.image.AxesImage` associated
           with the colorbar.
        divider: The :class:`~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
           used to create axes for the colorbar.
        cbar_axes: The axes of the colorbar, or ``None`` if there is no
           colorbar.
        cbar_orient: Orientation of the colorbar, "vertical" or
           "horizontal", or ``None`` if there is no colorbar.
        cbar_visible: ``True`` if the colorbar is visible. An invisible
           colorbar, created to maintain consistent image size within a
           figure, does not respond to mouse scroll events.
        cmap_delta: Fraction of colormap range for vmin/vmax shifts.
        vmin_ref: Reference colormap :code:`vmin`.
        vmax_ref: Reference colormap :code:`vmax`.
    """

    axesimage: mpl.image.AxesImage
    divider: Optional[AxesDivider]
    cbar_axes: Optional[Axes]
    cbar_orient: Optional[str] = None
    cbar_visible: bool = False
    cmap_delta: float
    vmin_ref: float
    vmax_ref: float
    _cmap_group: Optional[ShareGroup] = None

    def init_colorbar(
        self,
        axesimage: mpl.image.AxesImage,
        divider: Optional[AxesDivider] = None,
        cbar_axes: Optional[Axes] = None,
        cbar_orient: Optional[str] = None,
        cbar_visible: bool = True,
        cmap_delta: float = 0.02,
    ):
        """Initialize colorbar support.

        Record the colorbar artists and the current colormap limits as
        the reference limits.

        Args:
            axesimage: The :class:`~matplotlib.image.AxesImage`
               associated with the colorbar.
            divider: The axes divider used to create axes for the
               colorbar.
            cbar_axes: The axes of the colorbar.
            cbar_orient: Orientation of the colorbar, "vertical" or
               "horizontal". If :code:`cbar_axes` is specified and this
               is not, the matplotlib default of "vertical" is assumed.
            cbar_visible: Flag indicating whether the colorbar is
               visible.
            cmap_delta: Fraction of colormap range for vmin/vmax shifts.
        """
        self.axesimage = axesimage
        self.divider = divider
        self.cbar_axes = cbar_axes
        if cbar_axes is None:
            self.cbar_orient = None
        else:
            self.cbar_orient = "vertical" if cbar_orient is None else cbar_orient
        self.cbar_visible = cbar_visible and cbar_axes is not None
        self.cmap_delta = cmap_delta
        self.vmin_ref, self.vmax_ref = self.cmap_limits
        self._cmap_group = None

    @property
    def cmap_limits(self) -> Tuple[float, float]:
        """The current colormap limits, as a tuple (`vmin`, `vmax`).

        The limits are returned as floats, whatever the type with which
        matplotlib recorded them: they are integers when matplotlib
        determines them by autoscaling an array of integer type.
        """
        vmin, vmax = self.axesimage.norm.vmin, self.axesimage.norm.vmax
        assert vmin is not None and vmax is not None
        return float(vmin), float(vmax)

    @property
    def cmap_group(self) -> Optional[ShareGroup]:
        """The colormap share group of this plot, if any."""
        return self._cmap_group

    def owns_axes(self, axes: Axes) -> bool:
        """Determine whether an axes belongs to this plot."""
        if self.cbar_axes is not None and axes is self.cbar_axes:
            return True
        return super().owns_axes(axes)  # type: ignore[misc]

    def scroll_handlers(self) -> Tuple[ScrollHandlerType, ...]:
        """Prepend the colormap handler to the scroll event handlers."""
        return (self._scroll_cmap_handler,) + super().scroll_handlers()  # type: ignore[misc]

    def _scroll_cmap_handler(self, event: Event) -> bool:
        """Handle a mouse scroll event as a colormap range adjustment.

        Args:
            event: Mouse scroll event.

        Returns:
            ``True`` if the event was claimed as a colormap adjustment.
        """
        assert hasattr(event, "button")
        rel_pos = self._cbar_event_rel_pos(event)
        if rel_pos is None:
            return False
        if event.button == "up":
            sign = 1
        elif event.button == "down":
            sign = -1
        else:
            return False
        if rel_pos < 0.5:
            self.shift_cmap_vmin(sign * self.cmap_delta)
        elif rel_pos > 0.5:
            self.shift_cmap_vmax(sign * self.cmap_delta)
        return True

    def _cbar_event_rel_pos(self, event: Event) -> Optional[float]:
        """Determine relative position of an event in the colorbar.

        Args:
            event: Mouse event.

        Returns:
            Position of the event within the colorbar, as a fraction of
            its extent, or ``None`` if the event did not occur in a
            visible colorbar.
        """
        assert hasattr(event, "inaxes") and hasattr(event, "x") and hasattr(event, "y")
        if self.cbar_axes is None or not self.cbar_visible:
            return None
        if event.inaxes is not self.cbar_axes:
            return None
        box = self.cbar_axes.get_window_extent().bounds
        if self.cbar_orient == "vertical":
            rel_pos = (event.y - box[1]) / box[3]
        else:
            rel_pos = (event.x - box[0]) / box[2]
        return rel_pos

    def _cmap_apply(self, method: str, *args):
        """Apply a colormap adjustment to this plot or to its share group.

        Args:
            method: Name of the method to apply.
            *args: Arguments of the method.
        """
        if self._cmap_group is None:
            getattr(self, method)(*args)
        else:
            self._cmap_group.broadcast(method, *args)

    def shift_cmap_vmin(self, rel_delta: float):
        """Change colormap :code:`vmin`.

        Change the colormap :code:`vmin` of this plot, clipping it to the
        reference range. If the plot belongs to a colormap share group,
        the change is applied to every member of the group.

        Args:
            rel_delta: Signed relative change in colormap :code:`vmin`
                value.
        """
        self._cmap_apply("apply_cmap_vmin", rel_delta)

    def shift_cmap_vmax(self, rel_delta: float):
        """Change colormap :code:`vmax`.

        Change the colormap :code:`vmax` of this plot, clipping it to the
        reference range. If the plot belongs to a colormap share group,
        the change is applied to every member of the group.

        Args:
            rel_delta: Signed relative change in colormap :code:`vmax`
                value.
        """
        self._cmap_apply("apply_cmap_vmax", rel_delta)

    def apply_cmap_vmin(self, rel_delta: float):
        """Change the colormap :code:`vmin` of this plot only.

        Change the colormap :code:`vmin` of this plot, without applying
        the change to the colormap share group of the plot, if any.

        Args:
            rel_delta: Signed relative change in colormap :code:`vmin`
                value.
        """
        vmin, vmax = self.cmap_limits
        abs_delta = rel_delta * (self.vmax_ref - self.vmin_ref)
        new_vmin = vmin + abs_delta
        if new_vmin < self.vmin_ref:
            new_vmin = self.vmin_ref
        if new_vmin < vmax:
            self.axesimage.set_clim(vmin=new_vmin)
        self.axes.figure.canvas.draw_idle()  # type: ignore[attr-defined]

        msg = f"Color map vmin set to {self.cmap_limits[0]:.1f}"
        self.toolbar_message(msg)  # type: ignore[attr-defined]

    def apply_cmap_vmax(self, rel_delta: float):
        """Change the colormap :code:`vmax` of this plot only.

        Change the colormap :code:`vmax` of this plot, without applying
        the change to the colormap share group of the plot, if any.

        Args:
            rel_delta: Signed relative change in colormap :code:`vmax`
                value.
        """
        vmin, vmax = self.cmap_limits
        abs_delta = rel_delta * (self.vmax_ref - self.vmin_ref)
        new_vmax = vmax + abs_delta
        if new_vmax > self.vmax_ref:
            new_vmax = self.vmax_ref
        if new_vmax > vmin:
            self.axesimage.set_clim(vmax=new_vmax)
        self.axes.figure.canvas.draw_idle()  # type: ignore[attr-defined]

        msg = f"Color map vmax set to {self.cmap_limits[1]:.1f}"
        self.toolbar_message(msg)  # type: ignore[attr-defined]


class SliceMixin:
    """Mixin class providing volume slice selection.

    Mouse wheel scroll in the plot axes with the shift key depressed, or
    manipulation of the slice selection slider widget, changes the
    displayed slice of a volume. If the plot belongs to a slice share
    group (see :func:`komplot.share_slice`), the change is applied to
    every member of the group.

    Attributes:
        volume: The volume array, with the slice axis in the leading
           position.
        slider_axes: The axes of the slice selection slider, or ``None``
           if there is no slider.
        slider: The slice selection slider widget, or ``None`` if there
           is no slider.
    """

    volume: np.ndarray
    slider_axes: Optional[Axes]
    slider: Optional[Slider]
    _slice_index: int = 0
    _slice_group: Optional[ShareGroup] = None

    def init_slice(
        self,
        volume: np.ndarray,
        slider_axes: Optional[Axes] = None,
        slider: Optional[Slider] = None,
    ):
        """Initialize slice selection support.

        Args:
            volume: The volume array, with the slice axis in the leading
               position.
            slider_axes: The axes of the slice selection slider.
            slider: The slice selection slider widget.
        """
        self.volume = volume
        self.slider_axes = slider_axes
        self.slider = slider
        self._slice_index = 0
        self._slice_group = None
        if slider is not None:
            slider.on_changed(self._slider_changed_handler)

    @property
    def num_slices(self) -> int:
        """The number of slices of the volume."""
        return self.volume.shape[0]

    @property
    def slice_index(self) -> int:
        """The index of the displayed volume slice.

        Assignment to this property displays the specified slice,
        clipping the index to the valid range, and applies the change to
        the slice share group of the plot, if any.
        """
        return self._slice_index

    @slice_index.setter
    def slice_index(self, index: int):
        self.set_volume_slice(index)

    @property
    def slice_group(self) -> Optional[ShareGroup]:
        """The slice share group of this plot, if any."""
        return self._slice_group

    def owns_axes(self, axes: Axes) -> bool:
        """Determine whether an axes belongs to this plot."""
        if self.slider_axes is not None and axes is self.slider_axes:
            return True
        return super().owns_axes(axes)  # type: ignore[misc]

    def scroll_handlers(self) -> Tuple[ScrollHandlerType, ...]:
        """Prepend the slice shift handler to the scroll event handlers."""
        return (self._scroll_slice_handler,) + super().scroll_handlers()  # type: ignore[misc]

    def _scroll_slice_handler(self, event: Event) -> bool:
        """Handle a mouse scroll event as a slice shift.

        Args:
            event: Mouse scroll event.

        Returns:
            ``True`` if the event was claimed as a slice shift.
        """
        assert hasattr(event, "inaxes") and hasattr(event, "button")
        if event.inaxes is not self.axes:  # type: ignore[attr-defined]
            return False
        if not self.event_manager.key_pressed["shift"]:  # type: ignore[attr-defined]
            return False
        if event.button == "up":
            index = self._slice_index + 1
        elif event.button == "down":
            index = self._slice_index - 1
        else:
            return False
        self.set_volume_slice(index, update_slider=True, suppress_events=True)
        return True

    def _slider_changed_handler(self, val: float):
        """Callback for slice selection slider changes."""
        self.set_volume_slice(int(val), update_slider=True, suppress_events=True)

    def set_volume_slice(
        self, index: int, update_slider: bool = True, suppress_events: bool = False
    ):
        """Set the volume slice index.

        Set the volume slice index, clipping it to the valid range. If
        the plot belongs to a slice share group, the change is applied to
        every member of the group.

        Args:
            index: Index of volume slice to display.
            update_slider: If ``True`` also update the volume slice
              selection sider widget to the selected index.
            suppress_events: If ``True``, suppress events generated by
              the slider update.
        """
        index = int(min(max(index, 0), self.num_slices - 1))
        if self._slice_group is None:
            self.apply_volume_slice(index, update_slider, suppress_events)
        else:
            self._slice_group.broadcast(
                "apply_volume_slice", index, update_slider, True
            )

    def apply_volume_slice(
        self, index: int, update_slider: bool = True, suppress_events: bool = False
    ):
        """Display a volume slice in this plot only.

        Display the specified volume slice in this plot, without applying
        the change to the slice share group of the plot, if any.

        Args:
            index: Index of volume slice to display.
            update_slider: If ``True`` also update the volume slice
              selection sider widget to the selected index.
            suppress_events: If ``True``, suppress events generated by
              the slider update.
        """
        self._slice_index = index
        if self.slider is not None and update_slider:
            if suppress_events:
                self.slider.eventson = False
            self.slider.set_val(self._slice_index)
            if suppress_events:
                self.slider.eventson = True
        im = self.axesimage  # type: ignore[attr-defined]
        im.set_data(self.volume[self._slice_index])
        self.axes.figure.canvas.draw_idle()  # type: ignore[attr-defined]

        msg = f"Slice index {self._slice_index} of {self.volume.shape[0]}"
        self.toolbar_message(msg)  # type: ignore[attr-defined]
