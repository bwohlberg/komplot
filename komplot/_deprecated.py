# -*- coding: utf-8 -*-
# Copyright (C) 2024-2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Compatibility support for the API deprecated in version 0.0.5.

Every public name removed by the restructuring of the plot class
hierarchy and of the sharing API in version 0.0.5 remains usable, each
use emitting a :class:`DeprecationWarning` and delegating to the name
that replaces it. Nothing in the package refers to this module other
than the final block of :mod:`komplot`, so the layer may be removed, at
the end of the deprecation cycle, by deleting

* this module,
* the block at the end of ``komplot/__init__.py`` that imports it,
* ``test/test_deprecated.py``,
* the deprecation notes in ``CHANGES.rst`` and ``docs/overview.rst``.

Two parts of the removed API are not supported, since they are neither
documented nor used in the examples of the version in which they were
removed, and cannot be reached other than via an object of a class that
no longer exists: the event handler methods of the removed event manager
classes, and attributes :code:`cmap_share_axes` and
:code:`slice_share_axes` of :class:`.FigureEventManager`, which have no
counterpart in the current sharing API, in which a figure may have any
number of share groups.

A deprecated sharing call also requires the plots sharing the property
to exist, since it is translated into a call that acts on plots rather
than on axes, while the removed methods recorded a set of axes that was
only consulted when an event occurred.
"""

import warnings
from typing import Any, Dict, List, Sequence, Union

import numpy as np
from matplotlib.axes import Axes

from ._event import FigureEventManager
from ._imview import ImageMixin
from ._share import ShareGroup, share_cmap, share_slice
from ._state import ColorbarMixin, Plot, SliceMixin, ZoomMixin

#: Version of the package in which the deprecated API is to be removed
REMOVED_IN = "0.0.7"

#: Deprecated public names of the package, and the names that replace
#: them. The plot state classes and the event manager classes of the
#: removed hierarchies are replaced by the plot base class and the
#: feature mixin classes from which each plot class is composed.
DEPRECATED_NAMES: Dict[str, Any] = {
    "GenericPlot": Plot,
    "ZoomablePlot": ZoomMixin,
    "ColorbarPlot": ColorbarMixin,
    "AxesEventManager": Plot,
    "ZoomEventManager": ZoomMixin,
    "ColorbarEventManager": ColorbarMixin,
    "ImageViewEventManager": ImageMixin,
    "VolumeViewEventManager": SliceMixin,
}

AxesSequenceType = Union[Sequence[Axes], np.ndarray]


def warn_deprecated(name: str, replacement: str, stacklevel: int = 3):
    """Emit a warning that a part of the API is deprecated.

    Args:
        name: Name of the deprecated part of the API.
        replacement: Description of the replacement for it, as a clause
          following the statement that the name is deprecated.
        stacklevel: Stack level at which the warning is to be
          reported, counted from this function, the default of 3
          reporting it at the caller of a deprecated method or property
          of a retained class.
    """
    warnings.warn(
        f"{name} is deprecated and will be removed in komplot version "
        f"{REMOVED_IN}; {replacement}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated_attribute(name: str) -> Any:
    """Get the replacement for a deprecated attribute of the package.

    Intended for use by the module level :code:`__getattr__` of
    :mod:`komplot`, which is called only for a name that the package
    does not define.

    Args:
        name: Name of the required attribute.

    Returns:
        The current name replacing deprecated name :code:`name`.

    Raises:
        AttributeError: If :code:`name` is neither a current nor a
           deprecated attribute of the package.
    """
    if name not in DEPRECATED_NAMES:
        raise AttributeError(f"module 'komplot' has no attribute '{name}'")
    replacement = DEPRECATED_NAMES[name]
    warn_deprecated(
        f"komplot.{name}", f"use komplot.{replacement.__name__} instead", stacklevel=4
    )
    return replacement


def _plots_for_axes(
    manager: FigureEventManager, axes: AxesSequenceType, function: str
) -> List[Plot]:
    """Get the plots owning a set of axes.

    Args:
        manager: Figure event manager of the figure of the axes.
        axes: Axes of the required plots.
        function: Name of the function replacing the deprecated method
          requiring the plots, for inclusion in an error message.

    Returns:
        List of the plots owning :code:`axes`.

    Raises:
        ValueError: If any of :code:`axes` has no plot.
    """
    axlist = axes.ravel().tolist() if isinstance(axes, np.ndarray) else list(axes)
    plots = [manager.plot_for_axes(axs) for axs in axlist]
    if any(plot is None for plot in plots):
        raise ValueError(
            f"Deprecated sharing methods are translated into calls of {function}, "
            "which requires the plots that are to share the property: "
            f"{sum(plot is None for plot in plots)} of the {len(plots)} specified "
            "axes have no plot. Create the plots before sharing."
        )
    return [plot for plot in plots if plot is not None]


def _get_axevman_for_axes(self: FigureEventManager, ax: Axes) -> Any:
    """Deprecated. Get the plot that owns an axes.

    Args:
        ax: Axes.

    Returns:
        The plot owning :code:`ax`, or ``None`` if there is none.
    """
    warn_deprecated(
        "FigureEventManager.get_axevman_for_axes",
        "use FigureEventManager.plot_for_axes or komplot.plot_for_axes, which "
        "return the plot itself",
    )
    return self.plot_for_axes(ax)


def _register_axevman_for_axes(self: FigureEventManager, ax: Axes, axem: Plot):
    """Deprecated. Add a plot to the record of plots of the figure.

    Args:
        ax: Axes of the plot, which is ignored, the plot determining its
          own axes.
        axem: Plot to register.
    """
    warn_deprecated(
        "FigureEventManager.register_axevman_for_axes",
        "use FigureEventManager.register_plot, which takes only the plot",
    )
    self.register_plot(axem)


def _set_cmap_share(self: FigureEventManager, axes: AxesSequenceType) -> ShareGroup:
    """Deprecated. Share the colormap range of the plots of a set of axes.

    Args:
        axes: Axes of the plots that are to share their colormap range.

    Returns:
        The colormap share group of the plots.
    """
    warn_deprecated(
        "FigureEventManager.set_cmap_share",
        "use komplot.share_cmap, which takes plots rather than axes, and which "
        "supports sharing between plots in different figures",
    )
    return share_cmap(_plots_for_axes(self, axes, "komplot.share_cmap"))


def _set_slice_share(self: FigureEventManager, axes: AxesSequenceType) -> ShareGroup:
    """Deprecated. Share the slice index of the plots of a set of axes.

    Args:
        axes: Axes of the plots that are to share their slice index.

    Returns:
        The slice share group of the plots.
    """
    warn_deprecated(
        "FigureEventManager.set_slice_share",
        "use komplot.share_slice, which takes plots rather than axes, and which "
        "supports sharing between plots in different figures",
    )
    return share_slice(_plots_for_axes(self, axes, "komplot.share_slice"))


def _axevman_from_ax(self: FigureEventManager) -> Dict[Axes, Plot]:
    """Deprecated. Get a mapping from axes to the plot that owns them.

    Returns:
        Dict mapping the primary axes of each plot of the figure to the
        plot itself.
    """
    warn_deprecated(
        "FigureEventManager.axevman_from_ax",
        "use the plots attribute, or komplot.plot_for_axes, the plots themselves "
        "replacing the axes event managers",
    )
    return {plot.axes: plot for plot in self.plots}


def _plot(self: Plot) -> Plot:
    """Deprecated. Get the plot itself.

    Provided for compatibility with the removed axes event managers,
    which held the plot state in this attribute.

    Returns:
        The plot itself.
    """
    warn_deprecated(
        "The plot attribute of an axes event manager",
        "plot state and event handling are now combined in the plot, which is "
        "returned by komplot.plot_for_axes",
    )
    return self


def _fig_event_man(self: Plot) -> FigureEventManager:
    """Deprecated. Get the figure event manager of the figure of the plot.

    Returns:
        The figure event manager of the figure of the plot.
    """
    warn_deprecated(
        "The fig_event_man attribute of an axes event manager",
        "use the event_manager attribute of the plot",
    )
    return self.event_manager


#: Deprecated members of retained classes, as (class, name, member)
#: triples, each member replacing one removed in version 0.0.5
DEPRECATED_MEMBERS = (
    (FigureEventManager, "get_axevman_for_axes", _get_axevman_for_axes),
    (FigureEventManager, "register_axevman_for_axes", _register_axevman_for_axes),
    (FigureEventManager, "set_cmap_share", _set_cmap_share),
    (FigureEventManager, "set_slice_share", _set_slice_share),
    (FigureEventManager, "axevman_from_ax", property(_axevman_from_ax)),
    (Plot, "plot", property(_plot)),
    (Plot, "fig_event_man", property(_fig_event_man)),
)


def _installed_by_layer(member: Any) -> bool:
    """Determine whether a class member is one defined by this module.

    Args:
        member: Member of a class, which may be a property.

    Returns:
        ``True`` if :code:`member` is defined by this module.
    """
    function = member.fget if isinstance(member, property) else member
    return getattr(function, "__module__", None) == __name__


def install():
    """Attach the deprecated members to the classes that retain them.

    A member that is already attached is left as it is, so that a reload
    of the package, as performed by the autoreload extension of
    :mod:`IPython`, is not an error.

    Raises:
        RuntimeError: If a class already has an attribute of the same
           name as one of the deprecated members, and that attribute is
           not one attached by this module, which would indicate that a
           name removed in version 0.0.5 has been reintroduced with a
           different meaning.
    """
    for cls, name, member in DEPRECATED_MEMBERS:
        current = getattr(cls, name, None)
        if current is not None:
            if _installed_by_layer(current):
                continue
            raise RuntimeError(
                f"Deprecated name {cls.__name__}.{name} is in current use; the "
                "corresponding entry of komplot._deprecated.DEPRECATED_MEMBERS "
                "must be removed."
            )
        setattr(cls, name, member)
