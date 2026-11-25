# -*- coding: utf-8 -*-
# Copyright (C) 2024-2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Volume viewer."""

from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.axes_divider import (
    AxesDivider,  # type: ignore[import-untyped]
)

from ._imview import ImageMixin, _cmap_limit_kwargs
from ._state import ColorbarMixin, Plot, SliceMixin, ZoomMixin, figure_and_axes


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
        orientation=orient,  # type: ignore[arg-type]
    )
    return sax, slider


def _slice_axis_first(volume: np.ndarray, slice_axis: int) -> np.ndarray:
    """Transpose a volume so that the slice axis is in the leading position.

    Args:
        volume: Volume to transpose.
        slice_axis: Axis of :code:`volume` from which slices are
           selected.

    Returns:
        The transposed volume.

    Raises:
        ValueError: If the volume is not of a shape appropriate for
           volume slice display with the specified slice axis.
    """
    if slice_axis < 0:
        slice_axis = volume.ndim + slice_axis
    slice_shape = volume.shape[0:slice_axis] + volume.shape[slice_axis + 1 :]  # type: ignore
    if volume.ndim not in (3, 4) or (
        volume.ndim == 4 and slice_shape[-1] not in (3, 4)
    ):
        raise ValueError(
            f"Argument volume shape {volume.shape} not appropriate for volume slice "
            f"display with slice_axis={slice_axis}."
        )
    assert isinstance(slice_axis, int)
    return np.transpose(
        volume,
        (slice_axis,)
        + tuple(range(0, slice_axis))
        + tuple(range(slice_axis + 1, volume.ndim)),  # move slice axis to position 0
    )


class VolumeView(SliceMixin, ImageMixin, ColorbarMixin, ZoomMixin, Plot):
    """A display of a slice of a volume.

    A display of a slice of a volume, supporting the interactive features
    provided by :class:`.FigureEventManager`, :class:`.SliceMixin`,
    :class:`.ColorbarMixin`, and :class:`.ZoomMixin`. It is usually
    constructed via the :func:`volview` function, which additionally
    provides convenient specification of the colormap limits.

    Attributes:
        figure: Plot figure.
        axes: Plot axes.
        axesimage: The :class:`~matplotlib.image.AxesImage` of the
           displayed volume slice.
        divider: The :class:`~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
           used to create axes for the colorbar and the slice selection
           slider.
        cbar_axes: The axes of the colorbar, or ``None`` if there is no
           colorbar.
        volume: The volume array, with the slice axis in the leading
           position.
        slider_axes: The axes of the slice selection slider.
        slider: The slice selection slider widget.
    """

    def __init__(
        self,
        volume: np.ndarray,
        *,
        slice_axis: int = 0,
        interpolation: str = "nearest",
        origin: str = "upper",
        imshow_kwargs: Optional[dict] = None,
        show_cbar: Optional[bool] = False,
        cmap: Optional[Union[Colormap, str]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fignum: Optional[int] = None,
        ax: Optional[Axes] = None,
    ):
        """
        Args:
            volume: Volume to display. See :func:`volview`.
            slice_axis: See :func:`volview`.
            interpolation: See :func:`volview`.
            origin: See :func:`volview`.
            imshow_kwargs: Additional keyword arguments for
                :meth:`~matplotlib.axes.Axes.imshow`, such as those
                specifying the colormap limits :code:`vmin` and
                :code:`vmax`.
            show_cbar: See :func:`volview`.
            cmap: See :func:`volview`.
            title: See :func:`volview`.
            figsize: See :func:`volview`.
            fignum: See :func:`volview`.
            ax: See :func:`volview`.

        Raises:
            ValueError: If the input array is not of the required shape.
        """
        volume = _slice_axis_first(volume, slice_axis)
        image = volume[0]  # initial slice

        fig, axs, new_fig = figure_and_axes(ax, figsize=figsize, fignum=fignum)
        super().__init__(figure=fig, axes=axs, new_figure=new_fig)

        self.init_image(
            image,
            interpolation=interpolation,
            origin=origin,
            imshow_kwargs=imshow_kwargs,
            make_divider=True,
            show_cbar=show_cbar,
            cmap=cmap,
            title=title,
        )

        pad = 0.35 if show_cbar and self.cbar_orient == "horizontal" else 0.1
        if image.shape[0] >= 2 * image.shape[1]:
            slider_orient = "vertical"
            pad = 0.25
        else:
            slider_orient = "horizontal"
        sax, vol_slider = _create_slider(
            self.divider, volume, orient=slider_orient, pad=pad
        )

        self.init_slice(volume, slider_axes=sax, slider=vol_slider)
        self.init_zoom()
        self.show_new_figure()


def volview(
    volume: np.ndarray,
    *,
    slice_axis: int = 0,
    interpolation: str = "nearest",
    origin: str = "upper",
    vmin_quantile: float = 0.0,
    norm: Optional[Normalize] = None,
    show_cbar: Optional[bool] = False,
    cmap: Optional[Union[Colormap, str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fignum: Optional[int] = None,
    ax: Optional[Axes] = None,
) -> VolumeView:
    """Display a slice of a volume.

    Display a slice of a volume. Pixel values are displayed when the
    pointer is over valid image data. Supports the following features:

    - If an axes is not specified (via parameter :code:`ax`), a new
      figure and axes are created, and
      :meth:`~matplotlib.figure.Figure.show` is called after drawing the
      plot.
    - Interactive features provided by :class:`FigureEventManager`,
      :class:`SliceMixin`, :class:`ColorbarMixin`, and
      :class:`ZoomMixin` are supported in addition to the standard
      `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        volume: Volume to display. It should be three or four dimensional.
            If four dimensional, the final dimension after exclusion of
            the axis identified by :code:`slice_axis` represents color
            and opacity channels, should have size 3 or 4.
        slice_axis: The axis of :code:`volume`, if any, from which to
            select volume slices for display.
        interpolation: Specify type of interpolation used to display
            image (see :code:`interpolation` parameter of
            :meth:`~matplotlib.axes.Axes.imshow`).
        origin: Specify the origin of the image support. Valid values are
            "upper" and "lower" (see :code:`origin` parameter of
            :meth:`~matplotlib.axes.Axes.imshow`). The location of the
            plot x-ticks indicates which of these options was selected.
        vmin_quantile: Specify color map :code:`vmin` and :code:`vmax`
            based on pixel value quantiles. The default of 0.0
            corresponds to setting :code:`vmin` and :code:`vmax` to the
            minimum and maximum pixel value respectively. If it is
            non-zero, :code:`vmin` and :code:`vmax` are set to the
            :code:`vmin_quantile` quantile and the 1 -
            :code:`vmin_quantile` respectively.
        norm: Specify the :class:`~matplotlib.colors.Normalize` instance
            used to scale pixel values for input to the color map. If not
            ``None``, it is used to define the color map range instead of
            the :code:`vmin` and :code:`vmax` determined by
            :code:`vmin_quantile`.
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
        Volume view object.

    Raises:
        ValueError: If the input array is not of the required shape.
    """
    return VolumeView(
        volume,
        slice_axis=slice_axis,
        interpolation=interpolation,
        origin=origin,
        imshow_kwargs=_cmap_limit_kwargs(
            volume, vmin_quantile=vmin_quantile, norm=norm
        ),
        show_cbar=show_cbar,
        cmap=cmap,
        title=title,
        figsize=figsize,
        fignum=fignum,
        ax=ax,
    )
