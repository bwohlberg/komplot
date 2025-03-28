# -*- coding: utf-8 -*-
# Copyright (C) 2024-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Image viewer."""


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
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


# kw_only only supported from Python 3.10
KW_ONLY = {"kw_only": True} if "kw_only" in dataclass.__kwdefaults__ else {}


@dataclass(repr=False, **KW_ONLY)
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
    """


class ImageViewEventManager(ColorbarEventManager):
    """Manager for axes-based events.

    Manage mouse scroll and slider widget events. The following
    interactive features are supported:

    *Mouse wheel scroll*
       Zoom in or out at current cursor location.

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


def _get_axes_width(ax: Axes):
    """Get axes width."""
    return ax.get_tightbbox().bounds[2]


def _get_axes_height(ax: Axes):
    """Get axes height."""
    return ax.get_tightbbox().bounds[3]


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
        if hasattr(cax, "set_facecolor"):
            cax.set_facecolor("none")
        else:
            cax.set_axis_bgcolor("none")
        for axis in ["top", "bottom", "left", "right"]:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
    return cax


def _image_view(
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
) -> Tuple[
    Figure,
    Axes,
    bool,
    mpl.image.AxesImage,
    Optional[AxesDivider],
    Optional[Axes],
    Optional[str],
]:
    """Set up a basic image display.

    Set up an image display with basic features.
    """

    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (3, 4)):
        raise ValueError(
            f"Argument image shape {image.shape} not appropriate for image display."
        )

    fig, ax, show = figure_and_axes(ax, figsize=figsize, fignum=fignum)

    try:
        ax.set_adjustable("box")
    except ValueError:
        ax.set_adjustable("imagelim")

    if cmap is None and image.ndim == 2:
        cmap = mpl.cm.Greys_r  # pylint: disable=E1101

    if imshow_kwargs is None:
        imshow_kwargs = {}
    axim = ax.imshow(
        image, cmap=cmap, interpolation=interpolation, origin=origin, **imshow_kwargs
    )

    if origin == "upper":
        ax.tick_params(axis="x", top=True, bottom=False)
    else:
        ax.tick_params(axis="x", top=False, bottom=True)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if title is not None:
        ax.set_title(title)

    ax.format_coord = lambda x, y: _format_coord(x, y, image)
    _patch_coord_statusbar(fig)

    if HAVE_MPLCRS:
        mplcrs.cursor(axim)

    divider = make_axes_locatable(ax) if make_divider else None
    if show_cbar or show_cbar is None:
        if divider is None:
            divider = make_axes_locatable(ax)
        cbar_orient = "vertical" if image.shape[0] >= image.shape[1] else "horizontal"
        cax = _create_colorbar(
            ax, axim, divider, orient=cbar_orient, visible=show_cbar is not None
        )
    else:
        cbar_orient, cax = None, None

    return fig, ax, show, axim, divider, cax, cbar_orient


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
    - Interactive features provided by :class:`FigureEventManager` and
      :class:`ImageViewEventManager` are supported in addition to the
      standard `matplotlib <https://matplotlib.org/>`__
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
        Image view state object.

    Raises:
        ValueError: If the input array is not of the required shape.
    """

    if norm is None:
        if vmin_quantile == 0.0:
            vmin, vmax = image.min(), image.max()
        else:
            vmin, vmax = np.quantile(image, [vmin_quantile, 1.0 - vmin_quantile])  # type: ignore
        kwargs = {"vmin": vmin, "vmax": vmax}
    else:
        kwargs = {"norm": norm}
    fig, ax, show, axim, divider, cax, cbar_orient = _image_view(
        image,
        interpolation=interpolation,
        origin=origin,
        imshow_kwargs=kwargs,
        make_divider=False,
        show_cbar=show_cbar,
        cmap=cmap,
        title=title,
        figsize=figsize,
        fignum=fignum,
        ax=ax,
    )

    if show:
        fig.show()

    imvw = ImageView(
        figure=fig, axes=ax, axesimage=axim, divider=divider, cbar_axes=cax
    )

    if not hasattr(fig, "_event_manager"):
        fem = FigureEventManager(fig)  # constructed object attaches itself to fig
    else:
        fem = figure_event_manager(fig)
    if not hasattr(ax, "_event_manager"):
        ImageViewEventManager(ax, fem, imvw)  # constructed object attaches itself to ax

    return imvw
