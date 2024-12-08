# -*- coding: utf-8 -*-
# Copyright (C) 2024 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Image and volume viewer."""


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

from ._event import ColorbarEventManager, FigureEventManager, figure_event_manager
from ._state import ColorbarPlot, figure_and_axes

try:
    import mplcursors as mplcrs
except ImportError:
    HAVE_MPLCRS = False
else:
    HAVE_MPLCRS = True


@dataclass(kw_only=True)
class ImageView(ColorbarPlot):
    """State of imview plot.

    Args:
        figure: Plot figure.
        axes: Plot axes.
        axesimage: The :class:`~matplotlib.image.AxesImage` associated
           with the colorbar.
        divider: The :class:`~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
           used to create axes for the colorbar.
        cbar_axes: The axes of the colorbar.
        volume: The volume array (when a volume slice is being displayed,
           otherwise ``None``).
        slice_index: The index of the volume slice (only meaningful when
           a volume slice is being displayed).
        slider_axes: The axes of the volume slice index selection slider.
        slider: The volume slice index selection slider widget.
    """

    volume: Optional[np.ndarray] = None
    slice_index: int = 0
    slider_axes: Optional[Axes] = None
    slider: Optional[Slider] = None

    def set_volume_slice(self, index: int, update_slider: bool = True):
        """Set the volume slice index.

        Set the volume slice index. May only be used when a slice of a
        3D volume is being displayed.

        Args:
            index: Index of volume slice to display.
            update_slider: If ``True`` also update the volume slice
              selection sider widget to the selected index.
        """
        self.slice_index = index
        if self.slider is not None and update_slider:
            self.slider.set_val(self.slice_index)
        im = self.axesimage
        assert self.volume is not None
        im.set_data(self.volume[self.slice_index])
        self.axes.figure.canvas.draw_idle()

        msg = f"Slice index {self.slice_index} of {self.volume.shape[0]}"
        self.toolbar_message(msg)


class ImageViewEventManager(ColorbarEventManager):
    """Manager for axes-based events.

    Manage mouse scroll and slider widget events. The following
    interactive features are supported:

    *Mouse wheel scroll*
       Zoom in or out at current cursor location.

    *Mouse wheel scroll with shift key depressed*
       Shift the displayed slice when displaying a volume.

    *Click or drag slider widget*
       Change the displayed slice when displaying a volume.

    *Mouse wheel scroll in bottom half of colorbar*
       Increase or decrease colormap :code:`vmin`.

    *Mouse wheel scroll in top half of colorbar*
       Increase or decrease colormap :code:`vmax`.
    """

    plot: ImageView

    def __init__(
        self,
        axes: Axes,
        fig_event_man: FigureEventManager,
        iview: ImageView,
        zoom_scale: float = 2.0,
        cmap_delta: float = 0.02,
    ):
        """
        Args:
            axes: Axes to which this manager is attached.
            fig_event_man: The figure event manage for the figure to
               which :code:`axes` belong.
            iview: A plot state of type :class:`ImageView`.
            zoom_scale: Scaling factor for mouse wheel zoom.
            cmap_delta: Fraction of colormap range for vmin/vmax shifts.
        """
        super().__init__(axes, fig_event_man, iview, zoom_scale=zoom_scale)
        if iview.slider is not None:
            iview.slider.on_changed(lambda val: self.slider_event_handler(val))

    def scroll_event_handler(self, event: Event):
        """Calback for mouse scroll events."""
        if event.inaxes == self.axes and self.fig_event_man.key_pressed["shift"]:
            if self.fig_event_man.slice_share_axes:
                for ssax in self.fig_event_man.slice_share_axes:
                    axevman = self.fig_event_man.get_axevman_for_axes(ssax)
                    axevman.shift_slice_event_handler(event)
            else:
                self.shift_slice_event_handler(event)
        else:
            super().scroll_event_handler(event)

    def shift_slice_event_handler(self, event: Event):
        """Handle shift slice event."""
        index = self.plot.slice_index
        assert self.plot.volume is not None
        if event.button == "up":
            if self.plot.slice_index < self.plot.volume.shape[0] - 1:
                index += 1
        elif event.button == "down":
            if self.plot.slice_index > 0:
                index -= 1
        self.plot.set_volume_slice(index, update_slider=True)

    def slider_event_handler(self, val: int):
        """Calback for slider widget changes."""
        if self.fig_event_man.slice_share_axes:  # Slice display axes are shared
            # Iterate over all slice display axes for this figure
            for scax in self.fig_event_man.slice_share_axes:
                # Change displayed slice for all shared axes
                axevman = self.fig_event_man.get_axevman_for_axes(scax)
                axevman.plot.set_volume_slice(val, update_slider=False)
                # Ensure that slider is updated for axes other than the one on
                # which the slice change was triggered
                if scax != self.axes:
                    axevman.plot.slider.eventson = False
                    axevman.plot.slider.set_val(val)
                    axevman.plot.slider.eventson = True
        else:  # Slice display axes are not shared
            self.plot.set_volume_slice(val, update_slider=False)


def _format_coord(x: float, y: float, image: np.ndarray) -> str:
    """Format data cursor display string."""
    nr, nc = image.shape[0:2]
    col = int(x + 0.5)
    row = int(y + 0.5)
    if 0 <= col < nc and 0 <= row < nr:
        z = image[row, col]
        if image.ndim == 2:
            return f"x={x:6.2f}, y={y:6.2f}, z={z:.2f}"
        return f"x={x:6.2f}, y={y:6.2f}, z=" + ",".join([f"{c:.2f}" for c in z])
    return f"x={x:.2f}, y={y:.2f}"


def _patch_coord_statusbar(fig: Figure):
    """Monkey patch the coordinate status bar message.

    Monkey patch the coordinate status bar message mechanism so that
    `format_coord` controls both cursor location and pixel value
    format.
    """
    if fig.canvas.toolbar is not None:
        # See https://stackoverflow.com/a/47086132
        def mouse_move(self, event):
            if event.inaxes and event.inaxes.get_navigate():
                s = event.inaxes.format_coord(event.xdata, event.ydata)
                self.set_message(s)

        def patch_mouse_move(arg):
            return mouse_move(fig.canvas.toolbar, arg)

        fig.canvas.toolbar._idDrag = fig.canvas.mpl_connect(  # pylint: disable=W0212
            "motion_notify_event", patch_mouse_move
        )


def _create_colorbar(
    ax: Axes,
    axim: mpl.image.AxesImage,
    divider: AxesDivider,
    image: np.ndarray,
    visible: bool = True,
) -> Tuple[Axes, str]:
    """Create a colorbar attached to the displayed image.

    If `visible` is ``False``, ensure the colorbar is invisible, for use
    in maintaining consistent size of image and colorbar region.
    """
    orient = "vertical" if image.shape[0] >= image.shape[1] else "horizontal"
    pos = "right" if orient == "vertical" else "bottom"
    cax = divider.append_axes(pos, size="5%", pad=0.2)
    if visible:
        plt.colorbar(axim, ax=ax, cax=cax, orientation=orient)
    else:
        # See http://chris35wills.github.io/matplotlib_axis
        if hasattr(cax, "set_facecolor"):
            cax.set_facecolor("none")
        else:
            cax.set_axis_bgcolor("none")
        for axis in ["top", "bottom", "left", "right"]:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
    return cax, orient


def _create_slider(
    divider: AxesDivider, volume: np.ndarray, pad: float = 0.1
) -> Tuple[Axes, Slider]:
    """Create a volume slice slider attached to the displayed slice."""
    orient, pos = "horizontal", "bottom"
    sax = divider.append_axes(pos, size="5%", pad=pad)
    slider = Slider(
        ax=sax,
        label="Slice",
        valmin=0,
        valmax=volume.shape[0],
        valstep=range(volume.shape[0]),
        valinit=0,
        orientation=orient,
    )
    return sax, slider


def imview(
    data: np.ndarray,
    *,
    vol_slice_axis: Optional[int] = None,
    interpolation: str = "nearest",
    origin: str = "upper",
    norm: Normalize = None,
    show_cbar: Optional[bool] = False,
    cmap: Optional[Union[Colormap, str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fignum: Optional[int] = None,
    ax: Optional[Axes] = None,
) -> ImageView:
    """Display an image or a slice of a volume.

    Display an image or a slice of a volume. Pixel values are displayed
    when the pointer is over valid image data. Supports the following
    features:

    - If an axes is not specified (via parameter :code:`ax`), a new
      figure and axes are created, and
      :meth:`~matplotlib.figure.Figure.show` is called after drawing the
      plot.
    - Interactive features provided by :class:`FigureEventManager` and
      :class:`ImageViewEventManager` are supported in addition to the
      standard `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        data: Image or volume to display. An image should be two or three
            dimensional, with the third dimension, if present,
            representing color and opacity channels, and having size
            3 or 4. A volume should be three or four dimensional, with
            the final dimension after exclusion of the axis identified by
            :code:`vol_slice_axis` having size 3 or 4.
        vol_slice_axis: The axis of :code:`data`, if any, from which to
            select volume slices for display.
        interpolation: Specify type of interpolation used to display
            image (see :code:`interpolation` parameter of
            :meth:`~matplotlib.axes.Axes.imshow`).
        origin: Specify the origin of the image support. Valid values are
            "upper" and "lower" (see :code:`origin` parameter of
            :meth:`~matplotlib.axes.Axes.imshow`). The location of the
            plot x-ticks indicates which of these options was selected.
        norm: Specify the :class:`~matplotlib.colors.Normalize` instance
            used to scale pixel values for input to the color map.
        show_cbar: Flag indicating whether to display a colorbar. If set
            to ``None``, create an invisible colorbar so that the image
            occupies the same amount of space in a subplot as one with a
            visible colorbar.
        cmap: Color map for image or volume slices. If none specifed,
            defaults to :code:`matplotlib.cm.Greys_r` for monochrome
            image.
        title: Figure title.
        figsize: Specify dimensions of figure to be creaed as a tuple
            (`width`, `height`) in inches.
        fignum: Figure number of figure to be created.
        ax: Plot in specified axes instead of creating one.

    Returns:
        Image view state object.

    Raises:
        ValueError: If the input array is not of the required shape.
    """

    if vol_slice_axis is None:  # image display
        if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] not in (3, 4)):
            raise ValueError(
                f"Argument data shape {data.shape} not appropriate for image display."
            )
        data_shape = data.shape
        image = data
        volume = None
    else:  # volume slice display
        if vol_slice_axis < 0:
            vol_slice_axis = data.ndim + vol_slice_axis
        data_shape = data.shape[0:vol_slice_axis] + data.shape[vol_slice_axis + 1 :]
        if data.ndim not in (3, 4) or (data.ndim == 4 and data_shape[-1] not in (3, 4)):
            raise ValueError(
                f"Argument data shape {data.shape} not appropriate for volume slice "
                f"display with vol_slice_axis={vol_slice_axis}."
            )
        data = np.transpose(
            data,
            (vol_slice_axis,)
            + tuple(range(0, vol_slice_axis))
            + tuple(
                range(vol_slice_axis + 1, data.ndim)
            ),  # move slice axis to position 0
        )
        image = data[0]  # current slice
        volume = data

    fig, ax, show = figure_and_axes(ax, figsize=figsize, fignum=fignum)

    try:
        ax.set_adjustable("box")
    except ValueError:
        ax.set_adjustable("datalim")

    if cmap is None and data.ndim == 2:
        cmap = mpl.cm.Greys_r  # pylint: disable=E1101

    kwargs = (
        {"vmin": data.min(), "vmax": data.max()} if norm is None else {"norm": norm}
    )
    axim = ax.imshow(
        image, cmap=cmap, interpolation=interpolation, origin=origin, **kwargs
    )

    if origin == "upper":
        ax.tick_params(axis="x", top=True, bottom=False)
    else:
        ax.tick_params(axis="x", top=False, bottom=True)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if title is not None:
        ax.set_title(title)

    if vol_slice_axis is not None or show_cbar or show_cbar is None:
        divider = make_axes_locatable(ax)
    else:
        divider = None

    if show_cbar or show_cbar is None:
        cax, orient = _create_colorbar(
            ax, axim, divider, image, visible=show_cbar is not None
        )
    else:
        cax, orient = None, None

    if vol_slice_axis is not None:
        assert volume is not None
        pad = 0.35 if show_cbar and orient == "horizontal" else 0.1
        sax, vol_slider = _create_slider(divider, volume, pad=pad)
    else:
        sax, vol_slider = None, None

    ax.format_coord = lambda x, y: _format_coord(x, y, image)
    _patch_coord_statusbar(fig)

    if HAVE_MPLCRS:
        mplcrs.cursor(axim)

    if show:
        fig.show()

    imvw = ImageView(
        figure=fig,
        axes=ax,
        axesimage=axim,
        divider=divider,
        cbar_axes=cax,
        volume=volume,
        slider_axes=sax,
        slider=vol_slider,
    )

    if not hasattr(fig, "_event_manager"):
        fem = FigureEventManager(fig)  # constructed object attaches itself to fig
    else:
        fem = figure_event_manager(fig)
    if not hasattr(ax, "_event_manager"):
        ImageViewEventManager(ax, fem, imvw)  # constructed object attaches itself to ax

    return imvw
