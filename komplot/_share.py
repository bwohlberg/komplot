# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Sharing of plot properties between plots."""

from __future__ import annotations

import weakref
from typing import Iterable, Iterator, List, Union

import numpy as np

from ._state import ColorbarMixin, Plot, SliceMixin

PlotsType = Union[Plot, Iterable[Plot], np.ndarray]


class ShareGroup:
    """A group of plots that share a property.

    A group of plots that share a property, such as colormap limits or
    volume slice index. A change to the shared property of any member of
    the group is applied to every member. Members are held via weak
    references, so that membership of a group does not prevent a plot
    from being garbage collected when its figure is closed.

    Groups are created by the sharing functions, such as
    :func:`share_cmap`, rather than directly.

    Attributes:
        key: Identifier of the shared property, e.g. "cmap" or "slice".
    """

    def __init__(self, key: str):
        """
        Args:
            key: Identifier of the shared property. The corresponding
               plot attribute recording group membership is
               :code:`"_" + key + "_group"`.
        """
        self.key = key
        self._refs: List[weakref.ReferenceType] = []
        self._broadcasting = False

    @property
    def attribute(self) -> str:
        """Name of the plot attribute recording membership of this group."""
        return f"_{self.key}_group"

    def members(self) -> List[Plot]:
        """Get the live members of the group.

        Returns:
            List of members, in the order in which they joined the group.
        """
        members = []
        live_refs = []
        for ref in self._refs:
            plot = ref()
            if plot is not None:
                live_refs.append(ref)
                members.append(plot)
        self._refs = live_refs
        return members

    def __iter__(self) -> Iterator[Plot]:
        """Iterate over the live members of the group."""
        return iter(self.members())

    def __len__(self) -> int:
        """Get the number of live members of the group."""
        return len(self.members())

    def __contains__(self, plot: Plot) -> bool:
        """Determine whether a plot is a member of the group."""
        return any(member is plot for member in self.members())

    def add(self, plot: Plot):
        """Add a plot to the group.

        The plot leaves any other group for the same property before
        joining this one.

        Args:
            plot: Plot to add.
        """
        if plot in self:
            return
        current = getattr(plot, self.attribute, None)
        if current is not None:
            current.remove(plot)
        self._refs.append(weakref.ref(plot))
        setattr(plot, self.attribute, self)

    def remove(self, plot: Plot):
        """Remove a plot from the group.

        A group with fewer than two members after the removal is
        dissolved, since a plot sharing a property with no other plot is
        indistinguishable from one that is not shared at all.

        Args:
            plot: Plot to remove.
        """
        self._refs = [
            ref for ref in self._refs if ref() is not plot and ref() is not None
        ]
        if getattr(plot, self.attribute, None) is self:
            setattr(plot, self.attribute, None)
        if len(self._refs) < 2:
            self.dissolve()

    def dissolve(self):
        """Dissolve the group, removing all of its members."""
        for plot in self.members():
            setattr(plot, self.attribute, None)
        self._refs = []

    def broadcast(self, method: str, *args, **kwargs):
        """Apply a method to every live member of the group.

        Calls made while a broadcast is in progress are discarded, so
        that a change applied to a member cannot trigger a further
        broadcast to the group.

        Args:
            method: Name of the method to be called on each member.
            *args: Positional arguments of the method.
            **kwargs: Keyword arguments of the method.
        """
        if self._broadcasting:
            return
        self._broadcasting = True
        try:
            for plot in self.members():
                getattr(plot, method)(*args, **kwargs)
        finally:
            self._broadcasting = False


def _plot_list(plots: PlotsType) -> List[Plot]:
    """Convert a specification of a set of plots into a list of plots.

    Args:
        plots: A single plot, or an iterable of plots, such as a list, an
           array, or a :class:`ShareGroup`.

    Returns:
        List of distinct plots, in the order specified.
    """
    if isinstance(plots, Plot):
        return [plots]
    if isinstance(plots, np.ndarray):
        plots = plots.ravel().tolist()
    plot_list: List[Plot] = []
    for plot in plots:
        if not any(member is plot for member in plot_list):
            plot_list.append(plot)
    return plot_list


def _share_group(plots: PlotsType, key: str, feature: type, label: str) -> ShareGroup:
    """Construct a share group from a specification of a set of plots.

    Args:
        plots: An iterable of plots.
        key: Identifier of the shared property.
        feature: Mixin class required of each member.
        label: Description of the required feature, for error messages.

    Returns:
        The share group.

    Raises:
        ValueError: If fewer than two distinct plots are specified, or if
           any of them does not support the shared property.
    """
    plot_list = _plot_list(plots)
    if len(plot_list) < 2:
        raise ValueError(
            f"Sharing of {key} requires at least two distinct plots, "
            f"{len(plot_list)} specified."
        )
    for plot in plot_list:
        if not isinstance(plot, feature):
            raise ValueError(
                f"Sharing of {key} requires plots {label}, but a "
                f"{type(plot).__name__} was specified."
            )
    group = ShareGroup(key)
    for plot in plot_list:
        group.add(plot)
    return group


def share_cmap(plots: PlotsType) -> ShareGroup:
    """Share colormap limits between plots.

    The specified plots form a colormap share group: an adjustment of the
    colormap :code:`vmin` or :code:`vmax` of any member, whether via the
    colorbar of that member or by a call to
    :meth:`~komplot.ColorbarMixin.shift_cmap_vmin` or
    :meth:`~komplot.ColorbarMixin.shift_cmap_vmax`, is applied to every
    member. Since the adjustment is relative to the colormap range of
    each member, members are not required to have equal colormap ranges.

    Each call defines a complete group. A plot that was a member of
    another colormap share group leaves it, and a group left with fewer
    than two members is dissolved.

    Args:
        plots: An iterable of plots supporting a colorbar, such as a list
           or an array of plots, or a :class:`ShareGroup`.

    Returns:
        The colormap share group.

    Raises:
        ValueError: If fewer than two distinct plots are specified, or if
           any of them does not support a colorbar.
    """
    return _share_group(plots, "cmap", ColorbarMixin, "supporting a colorbar")


def share_slice(plots: PlotsType) -> ShareGroup:
    """Share volume slice index between plots.

    The specified plots form a slice share group: a change of the
    displayed slice of any member, whether via mouse scroll, the slice
    selection slider, or assignment to
    :attr:`~komplot.SliceMixin.slice_index`, is applied to every member.
    On formation of the group, all members are set to the slice index of
    the first member.

    Each call defines a complete group. A plot that was a member of
    another slice share group leaves it, and a group left with fewer than
    two members is dissolved.

    Args:
        plots: An iterable of plots supporting slice selection, such as a
           list or an array of plots, or a :class:`ShareGroup`.

    Returns:
        The slice share group.

    Raises:
        ValueError: If fewer than two distinct plots are specified, if
           any of them does not support slice selection, or if they do
           not all have the same number of slices.
    """
    plot_list = _plot_list(plots)
    group = _share_group(plot_list, "slice", SliceMixin, "supporting slice selection")
    num_slices = {plot.num_slices for plot in plot_list}  # type: ignore[attr-defined]
    if len(num_slices) > 1:
        shapes = ", ".join(str(plot.volume.shape) for plot in plot_list)  # type: ignore[attr-defined]
        group.dissolve()
        raise ValueError(
            "Sharing of slice index requires plots with equal numbers of slices, "
            f"but volume shapes {shapes} were specified."
        )
    group.broadcast("apply_volume_slice", plot_list[0].slice_index, True, True)  # type: ignore[attr-defined]
    return group


def unshare_cmap(plots: PlotsType):
    """Remove plots from their colormap share groups.

    A group left with fewer than two members is dissolved, so passing a
    :class:`ShareGroup` removes every member of that group.

    Args:
        plots: A plot, or an iterable of plots, such as a list, an array,
           or a :class:`ShareGroup`.
    """
    for plot in _plot_list(plots):
        group = getattr(plot, "_cmap_group", None)
        if group is not None:
            group.remove(plot)


def unshare_slice(plots: PlotsType):
    """Remove plots from their volume slice share groups.

    A group left with fewer than two members is dissolved, so passing a
    :class:`ShareGroup` removes every member of that group.

    Args:
        plots: A plot, or an iterable of plots, such as a list, an array,
           or a :class:`ShareGroup`.
    """
    for plot in _plot_list(plots):
        group = getattr(plot, "_slice_group", None)
        if group is not None:
            group.remove(plot)
