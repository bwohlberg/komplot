# -*- coding: utf-8 -*-
# Copyright (C) 2024 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Volume viewer."""


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.colors import Colormap, Normalize
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

from ._event import FigureEventManager, figure_event_manager
from ._imview import ImageView, ImageViewEventManager, _create_colorbar, _image_view

try:
    pass
except ImportError:
    HAVE_MPLCRS = False
else:
    HAVE_MPLCRS = True


@dataclass(kw_only=True)
class VolumeView(ImageView):
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


class VolumeViewEventManager(ImageViewEventManager):
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

    plot: VolumeView

    def __init__(
        self,
        axes: Axes,
        fig_event_man: FigureEventManager,
        iview: VolumeView,
        zoom_scale: float = 2.0,
        cmap_delta: float = 0.02,
    ):
        """
        Args:
            axes: Axes to which this manager is attached.
            fig_event_man: The figure event manage for the figure to
               which :code:`axes` belong.
            iview: A plot state of type :class:`VolumeView`.
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


def _create_slider(
    divider: AxesDivider, volume: np.ndarray, orient: str, pad: float = 0.1
) -> Tuple[Axes, Slider]:
    """Create a volume slice slider attached to the displayed slice."""
    pos = "left" if orient == "vertical" else "bottom"
    sax = divider.append_axes(pos, size="5%", pad=pad)
    slider = Slider(
        ax=sax,
        label="Slice",
        valmin=0,
        valmax=volume.shape[0] - 1,
        valstep=range(volume.shape[0]),
        valinit=0,
        orientation=orient,
    )
    return sax, slider


def volview(
    volume: np.ndarray,
    *,
    slice_axis: int = 0,
    interpolation: str = "nearest",
    origin: str = "upper",
    norm: Normalize = None,
    show_cbar: Optional[bool] = False,
    cmap: Optional[Union[Colormap, str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fignum: Optional[int] = None,
    ax: Optional[Axes] = None,
) -> VolumeView:
    """Display an image or a slice of a volume.

    Display an image or a slice of a volume. Pixel values are displayed
    when the pointer is over valid image data. Supports the following
    features:

    - If an axes is not specified (via parameter :code:`ax`), a new
      figure and axes are created, and
      :meth:`~matplotlib.figure.Figure.show` is called after drawing the
      plot.
    - Interactive features provided by :class:`FigureEventManager` and
      :class:`VolumeViewEventManager` are supported in addition to the
      standard `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        volume: Image or volume to display. An image should be two or three
            dimensional, with the third dimension, if present,
            representing color and opacity channels, and having size
            3 or 4. A volume should be three or four dimensional, with
            the final dimension after exclusion of the axis identified by
            :code:`vol_slice_axis` having size 3 or 4.
        slice_axis: The axis of :code:`data`, if any, from which to
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

    if slice_axis < 0:
        slice_axis = volume.ndim + slice_axis
    volume_shape = volume.shape[0:slice_axis] + volume.shape[slice_axis + 1 :]  # type: ignore
    if volume.ndim not in (3, 4) or (
        volume.ndim == 4 and volume_shape[-1] not in (3, 4)
    ):
        raise ValueError(
            f"Argument volume shape {volume.shape} not appropriate for volume slice "
            f"display with slice_axis={slice_axis}."
        )

    assert isinstance(slice_axis, int)
    volume = np.transpose(
        volume,
        (slice_axis,)
        + tuple(range(0, slice_axis))
        + tuple(range(slice_axis + 1, volume.ndim)),  # move slice axis to position 0
    )
    image = volume[0]  # current slice

    kwargs = (
        {"vmin": volume.min(), "vmax": volume.max()} if norm is None else {"norm": norm}
    )

    fig, ax, show, axim = _image_view(
        image,
        interpolation=interpolation,
        origin=origin,
        imshow_kwargs=kwargs,
        show_cbar=show_cbar,
        cmap=cmap,
        title=title,
        figsize=figsize,
        fignum=fignum,
        ax=ax,
    )

    divider = make_axes_locatable(ax)
    if show_cbar or show_cbar is None:
        cbar_orient = "vertical" if image.shape[0] >= image.shape[1] else "horizontal"
        cax = _create_colorbar(
            ax, axim, divider, orient=cbar_orient, visible=show_cbar is not None
        )
    else:
        cbar_orient, cax = None, None

    assert volume is not None
    pad = 0.35 if show_cbar and cbar_orient == "horizontal" else 0.1
    if image.shape[0] >= 2 * image.shape[1]:
        slider_orient = "vertical"
        pad = 0.25
    else:
        slider_orient = "horizontal"
    sax, vol_slider = _create_slider(divider, volume, orient=slider_orient, pad=pad)

    if show:
        fig.show()

    vlvw = VolumeView(
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
        VolumeViewEventManager(
            ax, fem, vlvw
        )  # constructed object attaches itself to ax

    return vlvw
