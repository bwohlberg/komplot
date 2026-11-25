# -*- coding: utf-8 -*-
# Copyright (C) 2024-2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Image viewer."""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore[import-untyped]
from mpl_toolkits.axes_grid1.axes_divider import (
    AxesDivider,  # type: ignore[import-untyped]
)

from ._state import ColorbarMixin, Plot, ZoomMixin, figure_and_axes

try:
    import mplcursors as mplcrs  # type: ignore[import-not-found]
except ImportError:
    HAVE_MPLCRS = False
else:
    HAVE_MPLCRS = True


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

        fig.canvas.toolbar._idDrag = fig.canvas.mpl_connect(  # type: ignore[attr-defined]  # pylint: disable=W0212
            "motion_notify_event", patch_mouse_move
        )


def _create_colorbar(
    ax: Axes,
    axim: mpl.image.AxesImage,
    divider: AxesDivider,
    orient: str,
    visible: bool = True,
) -> Axes:
    """Create a colorbar attached to the displayed image.

    If `visible` is ``False``, ensure the colorbar is invisible, for use
    in maintaining consistent size of image and colorbar region.
    """
    pos = "right" if orient == "vertical" else "bottom"
    cax = divider.append_axes(pos, size="5%", pad=0.2)
    if visible:
        plt.colorbar(axim, ax=ax, cax=cax, orientation=orient)
    else:
        # See http://chris35wills.github.io/matplotlib_axis
        cax.set_facecolor("none")
        for axis in ["top", "bottom", "left", "right"]:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
    return cax


def _check_image_shape(image: np.ndarray):
    """Check that an array has a shape suitable for image display.

    Args:
        image: Array to check.

    Raises:
        ValueError: If the array is not of the required shape.
    """
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (3, 4)):
        raise ValueError(
            f"Argument image shape {image.shape} not appropriate for image display."
        )


def _cmap_limit_kwargs(
    data: np.ndarray, vmin_quantile: float = 0.0, norm: Optional[Normalize] = None
) -> Dict[str, Any]:
    """Get colormap limit keyword arguments for an image display.

    Args:
        data: Array from which colormap limits are determined.
        vmin_quantile: Quantile determining the colormap limits. If zero,
            the limits are the minimum and maximum values of
            :code:`data`.
        norm: If not ``None``, it is used to define the colormap range
            instead of the limits determined by :code:`vmin_quantile`.

    Returns:
        A dict of keyword arguments for
        :meth:`~matplotlib.axes.Axes.imshow`.
    """
    if norm is not None:
        return {"norm": norm}
    if vmin_quantile == 0.0:
        vmin, vmax = data.min(), data.max()
    else:
        vmin, vmax = np.quantile(data, [vmin_quantile, 1.0 - vmin_quantile])  # type: ignore
    return {"vmin": vmin, "vmax": vmax}


class ImageMixin:
    """Mixin class providing display of an image.

    Mixin class providing display of an image, with pixel values
    displayed when the pointer is over valid image data, and an optional
    colorbar. It requires :class:`.ColorbarMixin`, the initializer of
    which it calls with the image and colorbar artists it creates.

    Attributes:
        cbar_orient: Orientation of the colorbar, or ``None`` if there is
           no colorbar.
    """

    cbar_orient: Optional[str] = None

    def init_image(
        self,
        image: np.ndarray,
        interpolation: str = "nearest",
        origin: str = "upper",
        imshow_kwargs: Optional[dict] = None,
        make_divider: bool = False,
        show_cbar: Optional[bool] = False,
        cmap: Optional[Union[Colormap, str]] = None,
        title: Optional[str] = None,
    ):
        """Initialize image display support.

        Display the image and create the colorbar, if required.

        Args:
            image: Image to display, of a shape suitable for image
                display.
            interpolation: Specify type of interpolation used to display
                image.
            origin: Specify the origin of the image support, either
                "upper" or "lower".
            imshow_kwargs: Additional keyword arguments for
                :meth:`~matplotlib.axes.Axes.imshow`, such as those
                specifying the colormap limits.
            make_divider: If ``True``, create an axes divider even when
                no colorbar is required, for use in attaching other
                auxiliary axes to the image.
            show_cbar: Flag indicating whether to display a colorbar. If
                set to ``None``, create an invisible colorbar.
            cmap: Color map for the image.
            title: Axes title.
        """
        ax: Axes = self.axes  # type: ignore[attr-defined]

        try:
            ax.set_adjustable("box")
        except ValueError:
            ax.set_adjustable("imagelim")  # type: ignore[arg-type]

        if cmap is None and image.ndim == 2:
            cmap = mpl.cm.Greys_r  # type: ignore[attr-defined]  # pylint: disable=E1101

        if imshow_kwargs is None:
            imshow_kwargs = {}
        axim = ax.imshow(
            image,
            cmap=cmap,  # type: ignore[arg-type]
            interpolation=interpolation,
            origin=origin,  # type: ignore[arg-type]
            **imshow_kwargs,
        )

        if origin == "upper":
            ax.tick_params(axis="x", top=True, bottom=False)
        else:
            ax.tick_params(axis="x", top=False, bottom=True)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        if title is not None:
            ax.set_title(title)

        def _fmt_crd(x, y):
            return _format_coord(x, y, axim.get_array().data)

        ax.format_coord = _fmt_crd  # type: ignore[method-assign]
        _patch_coord_statusbar(self.figure)  # type: ignore[attr-defined]

        if HAVE_MPLCRS:
            mplcrs.cursor(axim)

        divider = make_axes_locatable(ax) if make_divider else None
        if show_cbar or show_cbar is None:
            if divider is None:
                divider = make_axes_locatable(ax)
            self.cbar_orient = (
                "vertical" if image.shape[0] >= image.shape[1] else "horizontal"
            )
            cax = _create_colorbar(
                ax,
                axim,
                divider,
                orient=self.cbar_orient,
                visible=show_cbar is not None,
            )
        else:
            self.cbar_orient, cax = None, None

        self.init_colorbar(axim, divider=divider, cbar_axes=cax)  # type: ignore[attr-defined]


class ImageView(ImageMixin, ColorbarMixin, ZoomMixin, Plot):
    """A display of an image.

    A display of an image, supporting the interactive features provided
    by :class:`.FigureEventManager`, :class:`.ColorbarMixin`, and
    :class:`.ZoomMixin`. It is usually constructed via the
    :func:`imview` function, which additionally provides convenient
    specification of the colormap limits.

    Attributes:
        figure: Plot figure.
        axes: Plot axes.
        axesimage: The :class:`~matplotlib.image.AxesImage` of the
           displayed image.
        divider: The :class:`~mpl_toolkits.axes_grid1.axes_divider.AxesDivider`
           used to create axes for the colorbar, or ``None`` if there is
           none.
        cbar_axes: The axes of the colorbar, or ``None`` if there is no
           colorbar.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        interpolation: str = "nearest",
        origin: str = "upper",
        imshow_kwargs: Optional[dict] = None,
        make_divider: bool = False,
        show_cbar: Optional[bool] = False,
        cmap: Optional[Union[Colormap, str]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fignum: Optional[int] = None,
        ax: Optional[Axes] = None,
        show: bool = True,
    ):
        """
        Args:
            image: Image to display. See :func:`imview`.
            interpolation: See :func:`imview`.
            origin: See :func:`imview`.
            imshow_kwargs: Additional keyword arguments for
                :meth:`~matplotlib.axes.Axes.imshow`, such as those
                specifying the colormap limits :code:`vmin` and
                :code:`vmax`.
            make_divider: If ``True``, create an axes divider even when
                no colorbar is required.
            show_cbar: See :func:`imview`.
            cmap: See :func:`imview`.
            title: See :func:`imview`.
            figsize: See :func:`imview`.
            fignum: See :func:`imview`.
            ax: See :func:`imview`.
            show: If ``True``, show a newly created figure once the plot
                has been drawn. It should be set to ``False`` by a
                derived class that draws additional plot components.

        Raises:
            ValueError: If the input array is not of the required shape.
        """
        _check_image_shape(image)
        fig, axs, new_fig = figure_and_axes(ax, figsize=figsize, fignum=fignum)
        super().__init__(figure=fig, axes=axs, new_figure=new_fig)
        self.init_image(
            image,
            interpolation=interpolation,
            origin=origin,
            imshow_kwargs=imshow_kwargs,
            make_divider=make_divider,
            show_cbar=show_cbar,
            cmap=cmap,
            title=title,
        )
        self.init_zoom()
        if show:
            self.show_new_figure()


def imview(
    image: np.ndarray,
    *,
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
) -> ImageView:
    """Display an image.

    Display an image. Pixel values are displayed when the pointer is over
    valid image data. Supports the following features:

    - If an axes is not specified (via parameter :code:`ax`), a new
      figure and axes are created, and
      :meth:`~matplotlib.figure.Figure.show` is called after drawing the
      plot.
    - Interactive features provided by :class:`FigureEventManager`,
      :class:`ColorbarMixin`, and :class:`ZoomMixin` are supported in
      addition to the standard `matplotlib <https://matplotlib.org/>`__
      `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__.

    Args:
        image: Image to display. It should be two or three dimensional,
            with the third dimension, if present, representing color and
            opacity channels, and having size 3 or 4.
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
        Image view object.

    Raises:
        ValueError: If the input array is not of the required shape.
    """
    return ImageView(
        image,
        interpolation=interpolation,
        origin=origin,
        imshow_kwargs=_cmap_limit_kwargs(image, vmin_quantile=vmin_quantile, norm=norm),
        show_cbar=show_cbar,
        cmap=cmap,
        title=title,
        figsize=figsize,
        fignum=fignum,
        ax=ax,
    )
