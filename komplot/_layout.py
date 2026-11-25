# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Figure and axes creation with space reserved for auxiliary axes.

A colorbar is usually attached to a plot by taking the space for it from
the axes of the plot itself, as
:func:`~mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable` does,
with the result that a plot with a colorbar is smaller than a plot without
one, and plots within a figure are misaligned. The :func:`subplots`
function defined here instead reserves space for a colorbar, or for the
slice selection slider of a volume view, within the layout of the figure
itself, as additional gridspec tracks, so that the axes of the plots are
of equal size whether or not they have a colorbar, and remain so when the
figure is resized.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._event import plot_for_axes

__all__ = ["subplots"]


#: Type of a specification of space to be reserved for auxiliary axes
ReservationType = Union[bool, str, Sequence, np.ndarray]

#: Type of a specification of sharing of an axis between plots
ShareType = Union[bool, Literal["none", "all", "row", "col"]]

#: Reservation specifications for which an axes is shared by multiple plots
SHARED_MODES = ("shared", "row", "col")

#: Names of the attributes by which a plot axes refers to its reserved axes
RESERVED_NAMES = ("cax", "sax")


def reserved_axes(axes: Axes, name: str = "cax") -> Optional[Axes]:
    """Get an auxiliary axes reserved for a plot axes by :func:`subplots`.

    Args:
        axes: Primary axes of a plot.
        name: Name of the required reserved axes, "cax" for a colorbar or
           "sax" for a slice selection slider.

    Returns:
        The reserved axes, or ``None`` if there is none.
    """
    if name not in RESERVED_NAMES:
        raise ValueError(
            f'Reserved axes name must be one of {RESERVED_NAMES}, not "{name}".'
        )
    reserved = getattr(axes, name, None)
    return reserved if isinstance(reserved, Axes) else None


def claim_reserved_axes(
    axes: Axes, name: str = "cax"
) -> Tuple[Optional[Axes], Optional[Any]]:
    """Get an auxiliary axes reserved for a plot axes, if it is unclaimed.

    An axes reserved by :func:`subplots` may be shared by more than one
    plot, in which case only the first of them draws in it, the others
    deferring to it.

    Args:
        axes: Primary axes of a plot.
        name: Name of the required reserved axes, "cax" for a colorbar or
           "sax" for a slice selection slider.

    Returns:
        A tuple consisting of the reserved axes, or ``None`` if there is
        none or it has already been claimed, and the plot that has
        already claimed it, if there is one.
    """
    reserved = reserved_axes(axes, name)
    if reserved is None:
        return None, None
    owner = plot_for_axes(reserved)
    if owner is not None:
        return None, owner
    return reserved, None


def reserved_orientation(axes: Axes) -> str:
    """Determine the orientation of an auxiliary axes from its shape.

    Args:
        axes: An auxiliary axes, such as one reserved by :func:`subplots`.

    Returns:
        The orientation of the axes, "vertical" if it is taller than it is
        wide, and "horizontal" otherwise.
    """
    box = axes.get_position()
    figure = axes.get_figure(root=True)
    width, height = (1.0, 1.0) if figure is None else figure.get_size_inches()
    return "vertical" if box.height * height >= box.width * width else "horizontal"


def _reserving(mode: ReservationType) -> bool:
    """Determine whether a reservation specification reserves any space.

    Args:
        mode: Reservation specification, as passed to :func:`subplots`.

    Returns:
        ``True`` if any space at all is to be reserved.
    """
    if isinstance(mode, str):
        return True
    return bool(np.any(np.asarray(mode, dtype=bool)))


class _Track:
    """A reservation of figure layout space for auxiliary axes.

    A track is a set of gridspec rows or columns, inserted into the layout
    of a figure, within which auxiliary axes such as colorbars are placed.
    """

    def __init__(
        self,
        name: str,
        mode: ReservationType,
        size: float,
        pad: float,
        location: str,
        shape: Tuple[int, int],
    ):
        """
        Args:
            name: Name of the attribute by which a plot axes refers to its
               reserved axes.
            mode: Reservation specification, as passed to :func:`subplots`.
            size: Size of the reserved track, as a fraction of the size of
               a plot track.
            pad: Space between a plot track and its reserved track, as a
               fraction of the size of a plot track.
            location: Side of the plot axes on which space is reserved,
               either "right" or "bottom".
            shape: Shape (`nrows`, `ncols`) of the grid of plots.

        Raises:
            ValueError: If the specification is not a valid one.
        """
        if location not in ("right", "bottom"):
            raise ValueError(
                f'Reservation location for {name} must be either "right" or "bottom", '
                f'not "{location}".'
            )
        if isinstance(mode, str):
            if mode not in SHARED_MODES:
                raise ValueError(
                    f"Reservation specification for {name} must be a bool, a sequence "
                    f'of bools, or one of {SHARED_MODES}, not "{mode}".'
                )
            if mode == "row" and location != "right":
                raise ValueError(
                    'Reservation specification "row" requires location "right", since '
                    "an axes shared by a row of plots is placed at the end of the row."
                )
            if mode == "col" and location != "bottom":
                raise ValueError(
                    'Reservation specification "col" requires location "bottom", since '
                    "an axes shared by a column of plots is placed below the column."
                )
        if size <= 0.0 or pad < 0.0:
            raise ValueError(
                f"Reservation size for {name} must be positive and pad must be "
                f"non-negative, but size is {size} and pad is {pad}."
            )
        self.name = name
        self.mode = mode
        self.size = size
        self.pad = pad
        self.location = location
        self.shape = shape
        self.flags = self._cell_flags()

    @property
    def vertical(self) -> bool:
        """``True`` if the reserved axes are taller than they are wide."""
        return self.location == "right"

    @property
    def shared(self) -> bool:
        """``True`` if a single reserved axes serves more than one plot."""
        return isinstance(self.mode, str)

    def _cell_flags(self) -> np.ndarray:
        """Determine the cells of the grid of plots that are served.

        Returns:
            Boolean array, of the shape of the grid of plots.

        Raises:
            ValueError: If the specification cannot be broadcast to the
               shape of the grid of plots.
        """
        nrows, ncols = self.shape
        if self.shared:
            return np.ones((nrows, ncols), dtype=bool)
        flags = np.asarray(self.mode, dtype=bool)
        if flags.ndim == 0:
            return np.full((nrows, ncols), bool(flags))
        if flags.size != nrows * ncols:
            raise ValueError(
                f"Reservation specification for {self.name} has {flags.size} entries, "
                f"which cannot be matched to the grid shape {self.shape}."
            )
        return flags.reshape(nrows, ncols)

    def required(self) -> np.ndarray:
        """Determine the plot tracks for which a reserved track is required.

        Since the rows and columns of a gridspec span the full extent of
        the figure, space is reserved alongside a row or column of plots
        if it is required by any of the plots within it.

        Returns:
            Boolean array with one entry per plot column, if the location
            is "right", or per plot row, if the location is "bottom".
        """
        if self.shared:
            # A single track, placed at the end of the direction, serves every plot
            return np.zeros(
                self.shape[1] if self.vertical else self.shape[0], dtype=bool
            )
        return np.atleast_1d(self.flags.any(axis=0 if self.vertical else 1))


class _Direction:
    """The gridspec rows, or columns, of a figure layout.

    Attributes:
        ratios: Size ratios of all of the tracks in this direction.
        plot: Indices of the tracks occupied by plots.
        reserved: Dict mapping the name of each reservation to the indices
           of the tracks reserved for it, with ``None`` for plot tracks
           that have no reservation.
        shared: Dict mapping the name of each shared reservation to the
           index of the single track reserved for it.
        inserted: ``True`` if any tracks were inserted into the layout.
    """

    def __init__(
        self,
        ratios: List[float],
        plot: List[int],
        reserved: Dict[str, List[Optional[int]]],
        shared: Dict[str, int],
        inserted: bool,
    ):
        self.ratios = ratios
        self.plot = plot
        self.reserved = reserved
        self.shared = shared
        self.inserted = inserted


def _build_direction(
    plot_ratios: Sequence[float], tracks: List[_Track], gap: float
) -> _Direction:
    """Construct the layout of the rows, or columns, of a figure.

    Args:
        plot_ratios: Size ratios of the tracks occupied by plots.
        tracks: Reservations to be inserted into this direction, in the
           order in which they are to be placed.
        gap: Size ratio of the space between adjacent plots.

    Returns:
        The layout of this direction.
    """
    if not tracks:
        return _Direction(
            list(plot_ratios), list(range(len(plot_ratios))), {}, {}, False
        )

    required = {track.name: track.required() for track in tracks}
    ratios: List[float] = []
    plot: List[int] = []
    reserved: Dict[str, List[Optional[int]]] = {track.name: [] for track in tracks}
    shared: Dict[str, int] = {}
    for n, ratio in enumerate(plot_ratios):
        if n:
            ratios.append(gap)
        plot.append(len(ratios))
        ratios.append(ratio)
        for track in tracks:
            if track.shared or not required[track.name][n]:
                reserved[track.name].append(None)
                continue
            ratios.append(track.pad * ratio)
            reserved[track.name].append(len(ratios))
            ratios.append(track.size * ratio)
    mean_ratio = float(np.mean(plot_ratios))
    for track in tracks:
        if track.shared:
            ratios.append(track.pad * mean_ratio)
            shared[track.name] = len(ratios)
            ratios.append(track.size * mean_ratio)
    return _Direction(ratios, plot, reserved, shared, True)


def _share_target(share: ShareType, row: int, col: int) -> Optional[Tuple[int, int]]:
    """Determine the plot with which a plot shares an axis.

    Args:
        share: Axis sharing specification, as passed to :func:`subplots`.
        row: Row of the plot within the grid of plots.
        col: Column of the plot within the grid of plots.

    Returns:
        Position within the grid of plots of the plot with which the axis
        is shared, or ``None`` if it is not shared.

    Raises:
        ValueError: If the specification is not a valid one.
    """
    if share is False or share == "none":
        return None
    if share is True or share == "all":
        target = (0, 0)
    elif share == "row":
        target = (row, 0)
    elif share == "col":
        target = (0, col)
    else:
        raise ValueError(
            'Axis sharing specification must be a bool or one of "all", "row", "col" '
            f'or "none", not "{share}".'
        )
    return None if target == (row, col) else target


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    cbar: ReservationType = False,
    slider: ReservationType = False,
    cbar_size: float = 0.05,
    cbar_pad: float = 0.02,
    cbar_location: str = "right",
    slider_size: float = 0.05,
    slider_pad: float = 0.02,
    slider_location: str = "bottom",
    sharex: ShareType = False,
    sharey: ShareType = False,
    squeeze: bool = True,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
    subplot_kw: Optional[Dict[str, Any]] = None,
    gridspec_kw: Optional[Dict[str, Any]] = None,
    **fig_kw: Any,
) -> Tuple[Figure, Any]:
    """Create a figure and a grid of axes, with space for auxiliary axes.

    This is a drop-in replacement for :func:`matplotlib.pyplot.subplots`,
    to which it delegates when neither `cbar` nor `slider` requests that
    space be reserved. When space is requested, it is reserved within the
    layout of the figure, so that the axes of the plots themselves are of
    equal size whether or not they have a colorbar or a slider, and remain
    so when the figure is resized. Each axes of the grid refers to the
    axes reserved for it by attributes :code:`cax`, for a colorbar, and
    :code:`sax`, for a slider, either of which is ``None`` if no such
    space was reserved. A plot created in an axes for which space has been
    reserved uses it in preference to taking space from the axes itself
    (see :func:`~komplot.imview`).

    Reservations are specified by `cbar` and `slider`, each of which may be

    ``False``
       Reserve no space, the default.

    ``True``
       Reserve space for every plot in the grid.

    a sequence of bools
       Reserve space for the corresponding plots of the grid, in row-major
       order. Since the rows and columns of a gridspec span the full
       extent of the figure, space is reserved alongside a row or column
       of plots if it is required by any of the plots within it.

    "shared"
       Reserve space for a single axes, spanning and shared by all of the
       plots in the grid.

    "row"
       Reserve space for a single axes per row of plots, placed at the end
       of the row and shared by the plots within it. Requires a location
       of "right".

    "col"
       Reserve space for a single axes per column of plots, placed below
       the column and shared by the plots within it. Requires a location
       of "bottom".

    A single colorbar shared by multiple plots only represents the data of
    all of those plots if they have the same colormap range, which may be
    arranged by specifying the same :code:`vmin` and :code:`vmax` for each
    of them, and maintained under interactive adjustment by
    :func:`~komplot.share_cmap`.

    Args:
        nrows: Number of rows of plots.
        ncols: Number of columns of plots.
        cbar: Specification of space to reserve for colorbars.
        slider: Specification of space to reserve for slice selection
           sliders.
        cbar_size: Size of a colorbar, as a fraction of the size of a plot.
        cbar_pad: Space between a plot and its colorbar, as a fraction of
           the size of a plot.
        cbar_location: Side of a plot on which to reserve space for a
           colorbar, either "right" or "bottom".
        slider_size: Size of a slider, as a fraction of the size of a plot.
        slider_pad: Space between a plot and its slider, as a fraction of
           the size of a plot.
        slider_location: Side of a plot on which to reserve space for a
           slider, either "right" or "bottom".
        sharex: Share the x axis of the plots. Valid values are ``True``
           or "all", ``False`` or "none", "row", and "col".
        sharey: Share the y axis of the plots, as for `sharex`.
        squeeze: If ``True``, axes of unit length are removed from the
           array of axes that is returned.
        width_ratios: Size ratios of the columns of plots.
        height_ratios: Size ratios of the rows of plots.
        subplot_kw: Keyword arguments passed to the creation of each axes.
        gridspec_kw: Keyword arguments passed to the creation of the
           gridspec. Entries :code:`wspace` and :code:`hspace` determine
           the space between plots, as for
           :func:`matplotlib.pyplot.subplots`.
        **fig_kw: Keyword arguments passed to the creation of the figure.
           Note that a tight layout (e.g. :code:`layout="tight"`) is not
           compatible with reserved space, since the spacing of the
           tracks is specified by the gridspec itself, which matplotlib
           declines to adjust, warning that it has done so; a constrained
           layout (:code:`layout="constrained"`) may be used instead.

    Returns:
        A tuple consisting of the figure and the array of axes. The array
        has shape (`nrows`, `ncols`), reduced as specified by `squeeze`.
    """
    if not _reserving(cbar) and not _reserving(slider):
        return plt.subplots(
            nrows,
            ncols,
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            **fig_kw,
        )

    shape = (nrows, ncols)
    tracks = []
    if _reserving(cbar):
        tracks.append(_Track("cax", cbar, cbar_size, cbar_pad, cbar_location, shape))
    if _reserving(slider):
        tracks.append(
            _Track("sax", slider, slider_size, slider_pad, slider_location, shape)
        )

    gridspec_kw = {} if gridspec_kw is None else dict(gridspec_kw)
    wratios = gridspec_kw.pop("width_ratios", width_ratios)
    hratios = gridspec_kw.pop("height_ratios", height_ratios)
    wratios = [1.0] * ncols if wratios is None else [float(r) for r in wratios]
    hratios = [1.0] * nrows if hratios is None else [float(r) for r in hratios]
    if len(wratios) != ncols or len(hratios) != nrows:
        raise ValueError(
            f"Size ratios of lengths {len(wratios)} and {len(hratios)} do not match "
            f"grid shape {shape}."
        )
    wspace = gridspec_kw.pop("wspace", mpl.rcParams["figure.subplot.wspace"])
    hspace = gridspec_kw.pop("hspace", mpl.rcParams["figure.subplot.hspace"])

    # Space between plots is represented by tracks of the gridspec when
    # reservation tracks have been inserted, since the spacing provided by
    # matplotlib is specified relative to the mean size of all tracks, and
    # would therefore be distorted by the inserted tracks
    cols = _build_direction(
        wratios, [t for t in tracks if t.location == "right"], wspace
    )
    rows = _build_direction(
        hratios, [t for t in tracks if t.location == "bottom"], hspace
    )

    fig = plt.figure(**fig_kw)
    gs = fig.add_gridspec(
        len(rows.ratios),
        len(cols.ratios),
        width_ratios=cols.ratios,
        height_ratios=rows.ratios,
        wspace=0.0 if cols.inserted else wspace,
        hspace=0.0 if rows.inserted else hspace,
        **gridspec_kw,
    )

    subplot_kw = {} if subplot_kw is None else dict(subplot_kw)
    axes = np.empty(shape, dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            kwargs = dict(subplot_kw)
            xtarget = _share_target(sharex, row, col)
            ytarget = _share_target(sharey, row, col)
            if xtarget is not None:
                kwargs["sharex"] = axes[xtarget]
            if ytarget is not None:
                kwargs["sharey"] = axes[ytarget]
            axes[row, col] = fig.add_subplot(
                gs[rows.plot[row], cols.plot[col]], **kwargs
            )
    _hide_inner_labels(axes, sharex, sharey)

    for track in tracks:
        _create_reserved_axes(fig, gs, axes, track, rows, cols)

    if squeeze:
        return fig, axes.item() if axes.size == 1 else axes.squeeze()
    return fig, axes


def _hide_inner_labels(axes: np.ndarray, sharex: ShareType, sharey: ShareType):
    """Hide the tick labels of plots that share an axis with another plot.

    Tick labels are hidden for all but the bottom row of plots when the x
    axis is shared within columns, and for all but the first column of
    plots when the y axis is shared within rows, as is the behaviour of
    :func:`matplotlib.pyplot.subplots`. The
    :meth:`~matplotlib.axes.Axes.label_outer` method cannot be used for
    this purpose since it determines the position of an axes from its
    position within the gridspec, which includes the inserted reservation
    tracks.

    Args:
        axes: Array of the axes of the plots.
        sharex: Sharing specification for the x axis.
        sharey: Sharing specification for the y axis.
    """
    nrows, ncols = axes.shape
    for (row, col), axs in np.ndenumerate(axes):
        if sharex in (True, "all", "col") and row != nrows - 1:
            if axs.xaxis.get_label_position() == "bottom":
                axs.set_xlabel("")
            axs.xaxis.set_tick_params(which="both", labelbottom=False)
            axs.xaxis.offsetText.set_visible(False)
        if sharey in (True, "all", "row") and col != 0:
            if axs.yaxis.get_label_position() == "left":
                axs.set_ylabel("")
            axs.yaxis.set_tick_params(which="both", labelleft=False)
            axs.yaxis.offsetText.set_visible(False)


def _create_reserved_axes(
    fig: Figure,
    gs: Any,
    axes: np.ndarray,
    track: _Track,
    rows: _Direction,
    cols: _Direction,
):
    """Create the auxiliary axes of a reservation and attach them to the plot axes.

    Args:
        fig: Figure within which to create the axes.
        gs: Gridspec of the figure.
        axes: Array of the axes of the plots.
        track: Reservation for which axes are to be created.
        rows: Layout of the rows of the figure.
        cols: Layout of the columns of the figure.
    """
    nrows, ncols = axes.shape
    reserved = np.empty(axes.shape, dtype=object)
    if track.mode == "shared":
        if track.vertical:
            span = gs[rows.plot[0] : rows.plot[-1] + 1, cols.shared[track.name]]
        else:
            span = gs[rows.shared[track.name], cols.plot[0] : cols.plot[-1] + 1]
        reserved[:] = fig.add_subplot(span)
    elif track.mode == "row":
        for row in range(nrows):
            reserved[row, :] = fig.add_subplot(
                gs[rows.plot[row], cols.shared[track.name]]
            )
    elif track.mode == "col":
        for col in range(ncols):
            reserved[:, col] = fig.add_subplot(
                gs[rows.shared[track.name], cols.plot[col]]
            )
    else:
        for row in range(nrows):
            for col in range(ncols):
                if not track.flags[row, col]:
                    continue
                if track.vertical:
                    position = gs[rows.plot[row], cols.reserved[track.name][col]]
                else:
                    position = gs[rows.reserved[track.name][row], cols.plot[col]]
                reserved[row, col] = fig.add_subplot(position)

    for index in np.ndindex(axes.shape):
        axs = reserved[index]
        if isinstance(axs, Axes):
            # A reserved axes is shown only once a plot has drawn in it
            axs.set_visible(False)
        setattr(axes[index], track.name, axs)
