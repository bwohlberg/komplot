# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._help.

The help display closes with the figure to which it is attached, which
is not tested here because the close event on which that depends is
emitted by the interactive backends rather than by pyplot itself, and so
never occurs under the non-interactive backend used for the tests.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import press_key
from matplotlib.text import Text

import komplot as kplt
from komplot import Plot, SliceMixin
from komplot._help import (
    HELP_KEY,
    HELP_TITLE,
    figure_help_section,
    plot_help_sections,
    plot_type_name,
)


class SliceTestPlot(SliceMixin, Plot):
    """A minimal plot type supporting slice selection, without a slider."""

    def __init__(self, volume):
        """Display a slice of a volume, with no slice selection slider."""
        fig, ax = plt.subplots()
        super().__init__(figure=fig, axes=ax)
        self.axesimage = ax.imshow(volume[0])
        self.init_slice(volume)


def titles(plots):
    """Titles of the help sections of a set of plots."""
    return [section.title for section in plot_help_sections(plots)]


def section(plots, title):
    """The help section with a given title, or None if there is none."""
    found = [sec for sec in plot_help_sections(plots) if sec.title == title]
    return found[0] if found else None


def help_figure(fig):
    """Open the help for a figure and return the figure displaying it."""
    numbers = set(plt.get_fignums())
    kplt.show_help(fig)
    new = set(plt.get_fignums()) - numbers
    return plt.figure(new.pop()) if new else None


def help_text(fig):
    """Text displayed in a help figure, with white space normalized."""
    return [
        " ".join(text.get_text().split())
        for text in fig.findobj(Text)
        if text.get_text()
    ]


class TestPlotSections:
    """Tests for the help contributed by each type of plot."""

    def test_line_plot(self):
        """A line plot should offer zoom and nothing more."""
        plot = kplt.plot(np.arange(10.0))
        assert titles([plot]) == ["Zoom"]

    def test_image_view(self, image):
        """An image view with a colorbar should offer all of its features."""
        iview = kplt.imview(image, show_cbar=True)
        assert titles([iview]) == ["Zoom", "Colormap Range", "Image"]

    def test_image_view_without_colorbar(self, image):
        """Colormap adjustment should be offered only with a colorbar."""
        iview = kplt.imview(image, show_cbar=False)
        assert titles([iview]) == ["Zoom", "Image"]

    def test_image_view_with_reserved_colorbar(self, image):
        """A colorbar in reserved space should offer colormap adjustment.

        A plot in an axes for which space has been reserved displays a
        colorbar without being asked to.
        """
        _, ax = kplt.subplots(nrows=1, ncols=2, cbar=[False, True])
        views = [kplt.imview(image, ax=axs) for axs in ax]
        assert titles([views[0]]) == ["Zoom", "Image"]
        assert titles([views[1]]) == ["Zoom", "Colormap Range", "Image"]

    def test_shared_colorbar(self, image):
        """A shared colorbar should be described once, by the plot drawing it.

        A plot that defers to a colorbar shared with other plots has no
        colorbar of its own in which to scroll.
        """
        _, ax = kplt.subplots(nrows=1, ncols=2, cbar="shared")
        views = [kplt.imview(image, ax=axs) for axs in ax]
        assert titles([views[1]]) == ["Zoom", "Image"]
        assert titles(views).count("Colormap Range") == 1

    def test_share_group_noted(self, image):
        """Adjustment of a shared colormap range should be noted as such."""
        _, ax = kplt.subplots(nrows=1, ncols=2, cbar="shared")
        views = [kplt.imview(image, ax=axs) for axs in ax]
        assert section(views, "Colormap Range").note is None
        kplt.share_cmap(views)
        assert section(views, "Colormap Range").note == (
            "An adjustment is applied to every plot of the colormap share group."
        )

    def test_reserved_slider(self, volume):
        """A slider in reserved space should be described as a slider."""
        _, ax = kplt.subplots(nrows=1, ncols=2, slider=True)
        vview = kplt.volview(volume, ax=ax[0])
        keys = [binding.key for binding in section([vview], "Volume Slice").bindings]
        assert keys == ["Shift + Mouse wheel", "Slider click or drag"]

    @pytest.mark.parametrize(
        "shape,halves", [((8, 6), ("lower", "upper")), ((4, 12), ("left", "right"))]
    )
    def test_colorbar_orientation(self, shape, halves):
        """The halves of the colorbar should be named for its orientation."""
        iview = kplt.imview(np.zeros(shape), show_cbar=True)
        keys = [binding.key for binding in section([iview], "Colormap Range").bindings]
        assert [half in key for half, key in zip(halves, keys)] == [True, True]

    def test_volume_view(self, volume):
        """A volume view should offer slice selection and the slider."""
        vview = kplt.volview(volume, show_cbar=True)
        assert titles([vview]) == [
            "Zoom",
            "Colormap Range",
            "Image",
            "Volume Slice",
        ]
        keys = [binding.key for binding in section([vview], "Volume Slice").bindings]
        assert keys == ["Shift + Mouse wheel", "Slider click or drag"]

    def test_slice_without_slider(self, volume):
        """Slice selection without a slider should not offer the slider."""
        plot = SliceTestPlot(volume)
        keys = [binding.key for binding in section([plot], "Volume Slice").bindings]
        assert keys == ["Shift + Mouse wheel"]

    def test_contour(self):
        """A contour plot should offer zoom and colormap adjustment."""
        plot = kplt.contour(np.zeros((8, 8)))
        assert titles([plot]) == ["Zoom", "Colormap Range"]

    def test_surface(self):
        """A surface plot, which supports no zoom, should offer nothing."""
        plot = kplt.surface(np.zeros((8, 8)))
        assert titles([plot]) == []

    def test_figure_section(self):
        """The help for a figure should describe the keys it handles."""
        keys = [binding.key for binding in figure_help_section().bindings]
        assert keys == [HELP_KEY, "q", "PageUp / PageDown"]

    def test_no_plots(self):
        """A figure without plots should contribute no plot sections."""
        assert plot_help_sections([]) == []


class TestAssembly:
    """Tests for the assembly of the help for several plots."""

    def test_same_type_merged(self, image):
        """Plots of the same type should be described once, without a note."""
        _, ax = plt.subplots(nrows=1, ncols=2)
        views = [kplt.imview(image, show_cbar=True, ax=axs) for axs in ax]
        sections = plot_help_sections(views)
        assert [sec.title for sec in sections] == [
            "Zoom",
            "Colormap Range",
            "Image",
        ]
        assert all(sec.note is None for sec in sections)
        assert len(sections[0].bindings) == 1

    def test_mixed_types_annotated(self, image, volume):
        """A feature of only some of the plots should name those plots."""
        _, ax = plt.subplots(nrows=1, ncols=2)
        plots = [
            kplt.plot(np.arange(10.0), ax=ax[0]),
            kplt.volview(volume, show_cbar=True, ax=ax[1]),
        ]
        sections = {sec.title: sec for sec in plot_help_sections(plots)}
        assert sections["Zoom"].note is None
        assert sections["Volume Slice"].note == (
            "Applies to the volume view in this figure."
        )
        assert "volume view" in sections["Colormap Range"].note

    def test_merged_bindings(self, volume):
        """A binding offered by only one plot should be included once."""
        _, ax = plt.subplots(nrows=1, ncols=2)
        plots = [
            SliceTestPlot(volume),
            kplt.volview(volume, ax=ax[1]),
        ]
        keys = [binding.key for binding in section(plots, "Volume Slice").bindings]
        assert keys == ["Shift + Mouse wheel", "Slider click or drag"]

    def test_order_is_general_to_specific(self, volume):
        """Sections should be ordered from the most general feature."""
        vview = kplt.volview(volume, show_cbar=True)
        assert titles([vview]) == [
            "Zoom",
            "Colormap Range",
            "Image",
            "Volume Slice",
        ]

    def test_plot_type_name(self):
        """A plot class name should be rendered as separate words."""
        assert plot_type_name(kplt.VolumeView) == "volume view"
        assert plot_type_name(kplt.LinePlot) == "line plot"


class TestDisplay:
    """Tests for the display of the help for a figure."""

    def test_created_on_demand(self, image):
        """The help display should be constructed when first required."""
        iview = kplt.imview(image, show_cbar=True)
        manager = kplt.figure_event_manager(iview.figure)
        assert manager._help_manager is None
        kplt.show_help(iview.figure)
        assert manager._help_manager is not None

    def test_key_displays_help(self, image):
        """The help key should display the help in a new figure."""
        iview = kplt.imview(image, show_cbar=True)
        numbers = set(plt.get_fignums())
        press_key(iview.figure, HELP_KEY)
        assert len(set(plt.get_fignums()) - numbers) == 1

    def test_key_hides_help(self, image):
        """The help key should hide a help display that is open."""
        iview = kplt.imview(image, show_cbar=True)
        numbers = set(plt.get_fignums())
        press_key(iview.figure, HELP_KEY)
        press_key(iview.figure, HELP_KEY)
        assert set(plt.get_fignums()) == numbers
        press_key(iview.figure, HELP_KEY)
        assert len(set(plt.get_fignums()) - numbers) == 1

    def test_content(self, volume):
        """The help should describe the features of the plots displayed."""
        vview = kplt.volview(volume, show_cbar=True)
        text = help_text(help_figure(vview.figure))
        assert HELP_TITLE in text
        for description in (
            "Display or hide this help",
            "Close the figure",
            "Zoom in or out at the current cursor location",
            "Increase or decrease the colormap vmin",
            "Display the next or previous volume slice",
        ):
            assert description in text

    def test_content_excludes_absent_features(self):
        """The help should not describe features that are not available."""
        plot = kplt.plot(np.arange(10.0))
        text = " ".join(help_text(help_figure(plot.figure)))
        assert "Zoom in or out" in text
        assert "colormap" not in text
        assert "volume slice" not in text

    def test_discarded_when_a_plot_is_added(self, image, volume):
        """The help should be rebuilt after a plot is added to the figure."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        kplt.imview(image, show_cbar=True, ax=ax[0])
        kplt.show_help(fig)
        manager = kplt.figure_event_manager(fig)
        assert manager._help_manager is not None
        kplt.volview(volume, show_cbar=True, ax=ax[1])
        assert manager._help_manager is None
        text = " ".join(help_text(help_figure(fig)))
        assert "Display the next or previous volume slice" in text

    def test_current_figure(self, image):
        """The help should be displayed for the current figure by default."""
        iview = kplt.imview(image, show_cbar=True)
        plt.figure(iview.figure)
        numbers = set(plt.get_fignums())
        kplt.show_help()
        assert len(set(plt.get_fignums()) - numbers) == 1

    def test_figure_without_plots(self):
        """A figure in which no plot has been created should raise."""
        fig = plt.figure()
        with pytest.raises(RuntimeError, match="no attached FigureEventManager"):
            kplt.show_help(fig)

    def test_figure_event_manager_method(self, image):
        """The figure event manager should display the help for its figure."""
        iview = kplt.imview(image, show_cbar=True)
        numbers = set(plt.get_fignums())
        kplt.figure_event_manager(iview.figure).show_help()
        assert len(set(plt.get_fignums()) - numbers) == 1

    def test_independent_figures(self, image, volume):
        """Each figure should have its own help display."""
        iview = kplt.imview(image, show_cbar=True)
        vview = kplt.volview(volume, show_cbar=True)
        first = help_figure(iview.figure)
        second = help_figure(vview.figure)
        assert first is not second
        assert "Display the next or previous volume slice" not in help_text(first)
        assert "Display the next or previous volume slice" in help_text(second)
