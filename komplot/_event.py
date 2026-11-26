# -*- coding: utf-8 -*-
# Copyright (C) 2024-2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Figure-level event handling for interactive features."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from mpl_help import HelpManager

if TYPE_CHECKING:
    from ._state import Plot

from ._help import HELP_KEY, attach_help


class FigureEventManager:
    """Manager for figure-based events.

    Manage key-press events, dispatch mouse scroll events to the plot to
    which they are directed, and maintain a record of the plots attached
    to the figure. The key-press events provide monitoring of keys used
    as modifiers for other events (e.g. mouse scroll while shift is
    depressed) and also support the following keyboard shortcuts:

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

        Raises:
            RuntimeError: If the figure already has a manager attached.
        """
        if hasattr(fig, "_event_manager"):
            raise RuntimeError(f"Figure {fig} already has an event manager attached.")

        self.fig = fig
        self.fig_scale = fig_scale
        self.key_pressed = {k: False for k in self.monitored_keys}
        self.plots: List[Plot] = []
        self._help_manager: Optional[HelpManager] = None

        # Attach this event manager to the figure and connect callbacks
        fig._event_manager = self  # type: ignore[attr-defined]
        # See https://github.com/matplotlib/ipympl/issues/240 and
        #     https://github.com/matplotlib/ipympl/pull/235
        fig.canvas.capture_scroll = True  # type: ignore[attr-defined]
        fig.canvas.mpl_connect("key_press_event", self.key_press_handler)
        fig.canvas.mpl_connect("key_release_event", self.key_release_handler)
        fig.canvas.mpl_connect("figure_leave_event", self.figure_leave_handler)
        fig.canvas.mpl_connect("scroll_event", self.scroll_event_handler)

    def key_press_handler(self, event: Event):
        """Callback for key press events."""
        assert hasattr(event, "key")
        if event.key == "q":
            plt.close(self.fig)
        elif event.key == "pageup":
            self.fig.set_size_inches(
                self.fig_scale * self.fig.get_size_inches(), forward=True  # type: ignore[arg-type]
            )
        elif event.key == "pagedown":
            self.fig.set_size_inches(
                self.fig.get_size_inches() / self.fig_scale, forward=True  # type: ignore[arg-type]
            )
        elif event.key == HELP_KEY and self._help_manager is None:
            # The help display is created when it is first required, since it
            # describes the plots of the figure, which are not all constructed
            # when this manager is; once created, it handles the key itself
            self.show_help()
        elif event.key in self.monitored_keys:
            self.key_pressed[event.key] = True

    def key_release_handler(self, event: Event):
        """Callback for key release events.

        If the released key is in the list of monitored keys, update its
        status record.
        """
        assert hasattr(event, "key")
        if event.key in self.monitored_keys:
            self.key_pressed[event.key] = False

    def figure_leave_handler(self, event: Event):
        """Callback for figure leave events.

        Reset the status of all monitored keys when the cursor leaves the
        figure since key release events will no longer be registered.
        """
        self.key_pressed = {k: False for k in self.monitored_keys}

    def scroll_event_handler(self, event: Event):
        """Callback for mouse scroll events.

        Dispatch the event to the plot that owns the axes in which it
        occurred, if any.
        """
        assert hasattr(event, "inaxes")
        if event.inaxes is None:
            return
        plot = self.plot_for_axes(event.inaxes)
        if plot is not None:
            plot.handle_scroll(event)

    def register_plot(self, plot: Plot):
        """Add a plot to the record of plots attached to the figure.

        Args:
            plot: Plot to register.
        """
        self.plots.append(plot)
        self._discard_help()

    def show_help(self):
        """Display the help for the interactive features of the figure.

        The help is also displayed, and hidden, by pressing the
        :data:`komplot._help.HELP_KEY` key within the figure.
        """
        if self._help_manager is None:
            self._help_manager = attach_help(self.fig, self.plots)
        self._help_manager.show()

    def _discard_help(self):
        """Discard the help display for the figure.

        The help describes the interactive features of the plots of the
        figure, and is therefore discarded, to be constructed again when
        it is next required, whenever a plot is added to the figure. A
        help display that is open when it is discarded is closed, since
        it would otherwise remain out of date.
        """
        if self._help_manager is not None:
            self._help_manager.detach()
            self._help_manager = None

    def plot_for_axes(self, axes: Axes) -> Optional[Plot]:
        """Get the plot that owns an axes.

        An axes is owned by a plot if it is the primary axes of the plot,
        or an auxiliary axes of the plot, such as the axes of a colorbar
        or of a slice selection slider.

        Args:
            axes: Axes for which the owning plot is required.

        Returns:
            The plot owning :code:`axes`, or ``None`` if there is none.
        """
        for plot in self.plots:
            if plot.owns_axes(axes):
                return plot
        return None

    @staticmethod
    def attached_manager(fig: Figure, error: bool = False):
        """Get the figure manager attached to a figure.

        Args:
            fig: Figure.
            error: If ``True``, raise an exception if no figure manager
                attached.

        Returns:
            The attached figure event manager, or ``None`` if there is
            none and `error` is ``False``.

        Raises:
            RuntimeError: If `error` parameter is ``True`` and no figure
                manager attached.
        """
        if hasattr(fig, "_event_manager"):
            return fig._event_manager  # type: ignore[attr-defined]  # pylint: disable=W0212
        if error:
            raise RuntimeError(f"Figure {fig} has no attached FigureEventManager.")
        return None


def figure_event_manager(fig: Figure, error: bool = True) -> FigureEventManager:
    """Get the figure event manager attached to a figure.

    Args:
        fig: Figure.
        error: If ``True``, raise an exception if no figure manager
            attached.

    Returns:
        The attached figure event manager.

    Raises:
        RuntimeError: If `error` parameter is ``True`` and no figure
            manager attached.
    """
    return FigureEventManager.attached_manager(fig, error=error)


def attach_figure_event_manager(fig: Figure) -> FigureEventManager:
    """Get the figure event manager attached to a figure, creating one if required.

    Args:
        fig: Figure.

    Returns:
        The figure event manager attached to :code:`fig`.
    """
    fem = FigureEventManager.attached_manager(fig)
    if fem is None:
        fem = FigureEventManager(fig)  # constructed object attaches itself to fig
    return fem


def plots(fig: Figure) -> List[Plot]:
    """Get the plots attached to a figure.

    Args:
        fig: Figure.

    Returns:
        List of the plots attached to :code:`fig`, in the order in which
        they were created.
    """
    fem = FigureEventManager.attached_manager(fig)
    return [] if fem is None else list(fem.plots)


def plot_for_axes(axes: Axes) -> Optional[Plot]:
    """Get the plot that owns an axes.

    An axes is owned by a plot if it is the primary axes of the plot, or
    an auxiliary axes of the plot, such as the axes of a colorbar or of a
    slice selection slider. If more than one plot was created in the same
    axes, the first of them owns it.

    Args:
        axes: Axes for which the owning plot is required.

    Returns:
        The plot owning :code:`axes`, or ``None`` if there is none.
    """
    fig = axes.get_figure()
    if not isinstance(fig, Figure):
        return None
    fem = FigureEventManager.attached_manager(fig)
    return None if fem is None else fem.plot_for_axes(axes)


def show_help(fig: Optional[Figure] = None):
    """Display the help for the interactive features of a figure.

    The help describes the features of the plots of the figure as well as
    those of the figure itself. It is also displayed, and hidden, by
    pressing the "?" key within the figure, which is not possible in a
    Jupyter notebook with the `ipympl <https://matplotlib.org/ipympl/>`__
    backend, in which keyboard shortcuts are not functional.

    Args:
        fig: Figure for which the help is required. Defaults to the
          current figure.

    Raises:
        RuntimeError: If the figure has no attached event manager, as it
           has not if no plot has been created in it.
    """
    if fig is None:
        fig = plt.gcf()
    figure_event_manager(fig, error=True).show_help()
