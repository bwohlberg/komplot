# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Help display for the interactive features of a figure.

The help for a figure is assembled from the classes of which its plots
are composed, so that it describes the features that are actually
available in that figure: each class defining a :code:`help_section`
method contributes the section that the method returns, and the sections
contributed by the plots of the figure are merged. A feature that is
provided by a mixin class is therefore described by that mixin, next to
the implementation of the feature itself, and a figure containing, for
example, a line plot and a volume view offers the help for each of them
in a single display.

The display itself is provided by the :mod:`mpl_help` package.
"""

import re
from typing import TYPE_CHECKING, Dict, List, Sequence, Type

from matplotlib.figure import Figure
from mpl_help import HelpManager, HelpSection, KeyBinding

if TYPE_CHECKING:
    from ._state import Plot

#: Key that displays or hides the help for a figure
HELP_KEY = "?"

#: Title displayed at the head of the help display
HELP_TITLE = "KomPlot Controls"

#: Colour theme of the help display, "light" or "dark"
HELP_THEME = "light"

#: Note displayed with the help for the features of a figure as a whole
FIGURE_HELP_NOTE = (
    "The standard matplotlib shortcuts also apply, including s to save the "
    "figure and f1 for the matplotlib help."
)


def figure_help_section() -> HelpSection:
    """Get the help for the features of a figure as a whole.

    Returns:
        Help section describing the keys handled for every figure
        managed by a :class:`~komplot.FigureEventManager`.
    """
    return HelpSection(
        title="Figure",
        accent="blue",
        bindings=[
            KeyBinding(HELP_KEY, "Display or hide this help"),
            KeyBinding("q", "Close the figure"),
            KeyBinding("PageUp / PageDown", "Increase or decrease the figure size"),
        ],
        note=FIGURE_HELP_NOTE,
    )


def plot_type_name(plot_class: Type) -> str:
    """Get a readable name for a plot class.

    Args:
        plot_class: Class of a plot.

    Returns:
        The name of the class with its words separated and in lower
        case, e.g. "volume view" for :class:`~komplot.VolumeView`.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", " ", plot_class.__name__).lower()


def _merge_bindings(section: HelpSection, other: HelpSection):
    """Add the bindings of one help section to another.

    A binding with a key that the first section already describes is
    omitted, so that a feature common to several plots is described
    once, while a binding that only some of them provide, such as the
    slice selection slider of a volume view, is included.

    Args:
        section: Section to which bindings are to be added.
        other: Section providing the bindings.
    """
    keys = [binding.key for binding in section.bindings]
    for binding in other.bindings:
        if binding.key not in keys:
            section.bindings.append(binding)


def _applies_to_note(names: Sequence[str]) -> str:
    """Get a note stating the plots to which a help section applies.

    Args:
        names: Names of the plot types to which the section applies.

    Returns:
        Note text.
    """
    return f"Applies to the {' and '.join(names)} in this figure."


def plot_help_sections(plots: Sequence["Plot"]) -> List[HelpSection]:
    """Get the help for the features of a set of plots.

    Each class of which a plot is composed that defines a
    :code:`help_section` method contributes the section returned by that
    method, or nothing if it returns ``None``, as it does when the
    feature is not available in the plot, such as colormap range
    adjustment in a plot without a visible colorbar. Sections are
    ordered from the most general class to the most specific, and the
    sections contributed by more than one plot are merged. A section
    that does not apply to every plot of the figure is annotated with
    the plot types to which it does apply.

    Args:
        plots: Plots for which the help is required.

    Returns:
        List of help sections.
    """
    sections: Dict[Type, HelpSection] = {}
    contributors: Dict[Type, List[str]] = {}
    names: List[str] = []
    for plot in plots:
        name = plot_type_name(type(plot))
        if name not in names:
            names.append(name)
        for plot_class in reversed(type(plot).__mro__):
            method = plot_class.__dict__.get("help_section")
            if method is None:
                continue
            section = method(plot)
            if section is None:
                continue
            if plot_class in sections:
                _merge_bindings(sections[plot_class], section)
            else:
                sections[plot_class] = section
                contributors[plot_class] = []
            if name not in contributors[plot_class]:
                contributors[plot_class].append(name)

    for plot_class, section in sections.items():
        if len(names) > 1 and len(contributors[plot_class]) < len(names):
            note = _applies_to_note(contributors[plot_class])
            section.note = note if section.note is None else f"{section.note} {note}"
    return list(sections.values())


def attach_help(figure: Figure, plots: Sequence["Plot"]) -> HelpManager:
    """Attach a help display to a figure.

    The help display is opened and closed by the :data:`HELP_KEY` key,
    and closes with the figure to which it is attached.

    Args:
        figure: Figure to which the help display is to be attached.
        plots: Plots of the figure, the features of which are to be
          described in addition to those of the figure itself.

    Returns:
        The attached help manager.
    """
    manager = HelpManager(
        figure,
        title=HELP_TITLE,
        trigger_key=HELP_KEY,
        theme=HELP_THEME,
    )
    manager.add_section(figure_help_section())
    for section in plot_help_sections(plots):
        manager.add_section(section)
    return manager.attach()
