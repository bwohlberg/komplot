# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._event."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import (
    leave_figure,
    press_key,
    release_key,
    scroll_at_data,
    scroll_in_axes,
    scroll_outside_axes,
)

import komplot as kplt
from komplot import FigureEventManager, figure_event_manager
from komplot._event import attach_figure_event_manager


@pytest.fixture
def line_plot():
    """A line plot with an attached event manager."""
    x = np.linspace(0.0, 10.0, 32)
    return kplt.plot(x, x**2)


@pytest.fixture
def image_view(image):
    """An image view with a visible colorbar."""
    return kplt.imview(image, show_cbar=True)


@pytest.fixture
def wide_image_view(wide_image):
    """An image view with a horizontal colorbar."""
    return kplt.imview(wide_image, show_cbar=True)


@pytest.fixture
def volume_view(volume):
    """A volume view with a visible colorbar."""
    return kplt.volview(volume, show_cbar=True)


class TestFigureEventManager:
    """Tests for class FigureEventManager."""

    def test_attach(self):
        """A constructed manager should attach itself to the figure."""
        fig = plt.figure()
        fem = FigureEventManager(fig)
        assert fem.fig is fig
        assert figure_event_manager(fig) is fem

    def test_duplicate_attach(self):
        """Attaching a second manager to a figure should raise an exception."""
        fig = plt.figure()
        FigureEventManager(fig)
        with pytest.raises(RuntimeError, match="already has an event manager"):
            FigureEventManager(fig)

    def test_attached_manager_absent(self):
        """A figure without a manager should be detected."""
        fig = plt.figure()
        assert FigureEventManager.attached_manager(fig) is None
        assert figure_event_manager(fig, error=False) is None

    def test_attached_manager_absent_error(self):
        """An absent manager should raise an exception when so specified."""
        fig = plt.figure()
        with pytest.raises(RuntimeError, match="no attached FigureEventManager"):
            FigureEventManager.attached_manager(fig, error=True)
        with pytest.raises(RuntimeError, match="no attached FigureEventManager"):
            figure_event_manager(fig)

    def test_attach_figure_event_manager(self):
        """A manager should be created only if the figure has none."""
        fig = plt.figure()
        fem = attach_figure_event_manager(fig)
        assert figure_event_manager(fig) is fem
        assert attach_figure_event_manager(fig) is fem

    def test_scroll_capture_enabled(self, line_plot):
        """Scroll capture should be enabled for the canvas."""
        assert line_plot.figure.canvas.capture_scroll

    def test_initial_key_state(self, line_plot):
        """No monitored key should initially be recorded as pressed."""
        fem = figure_event_manager(line_plot.figure)
        assert set(fem.key_pressed) == set(FigureEventManager.monitored_keys)
        assert not any(fem.key_pressed.values())

    @pytest.mark.parametrize("key", FigureEventManager.monitored_keys)
    def test_monitored_key_press_release(self, line_plot, key):
        """Monitored key presses and releases should be recorded."""
        fem = figure_event_manager(line_plot.figure)
        press_key(line_plot.figure, key)
        assert fem.key_pressed[key]
        release_key(line_plot.figure, key)
        assert not fem.key_pressed[key]

    def test_unmonitored_key(self, line_plot):
        """An unmonitored key press should not change the key record."""
        fem = figure_event_manager(line_plot.figure)
        press_key(line_plot.figure, "a")
        release_key(line_plot.figure, "a")
        assert not any(fem.key_pressed.values())
        assert "a" not in fem.key_pressed

    def test_figure_leave(self, line_plot):
        """Monitored key state should be reset on figure leave."""
        fem = figure_event_manager(line_plot.figure)
        press_key(line_plot.figure, "shift")
        press_key(line_plot.figure, "left")
        leave_figure(line_plot.figure)
        assert not any(fem.key_pressed.values())

    def test_close_figure_key(self, line_plot):
        """Key "q" should close the figure."""
        press_key(line_plot.figure, "q")
        assert plt.get_fignums() == []

    def test_figure_scale_keys(self, line_plot):
        """Keys "pageup" and "pagedown" should scale the figure."""
        fig = line_plot.figure
        size = fig.get_size_inches().copy()
        press_key(fig, "pageup")
        np.testing.assert_allclose(fig.get_size_inches(), 1.1 * size)
        press_key(fig, "pagedown")
        np.testing.assert_allclose(fig.get_size_inches(), size)

    def test_figure_scale_factor(self):
        """The figure scaling factor should be as specified."""
        fig = plt.figure()
        FigureEventManager(fig, fig_scale=2.0)
        size = fig.get_size_inches().copy()
        press_key(fig, "pageup")
        np.testing.assert_allclose(fig.get_size_inches(), 2.0 * size)


class TestPlotRegistry:
    """Tests for the record of plots attached to a figure."""

    def test_initial_plot_record(self):
        """A new manager should have no registered plots."""
        fem = FigureEventManager(plt.figure())
        assert fem.plots == []

    def test_plot_registration(self, image):
        """Plots should be registered in order of construction."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, ax=ax[0])
        iview1 = kplt.imview(image, ax=ax[1])
        assert figure_event_manager(fig).plots == [iview0, iview1]

    def test_plot_for_axes(self, image):
        """The plot owning an axes should be retrievable."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, ax=ax[0])
        iview1 = kplt.imview(image, ax=ax[1])
        fem = figure_event_manager(fig)
        assert fem.plot_for_axes(ax[0]) is iview0
        assert fem.plot_for_axes(ax[1]) is iview1

    def test_plot_for_colorbar_axes(self, image_view):
        """A colorbar axes should be owned by its plot."""
        fem = figure_event_manager(image_view.figure)
        assert fem.plot_for_axes(image_view.cbar_axes) is image_view

    def test_plot_for_slider_axes(self, volume_view):
        """A slice slider axes should be owned by its plot."""
        fem = figure_event_manager(volume_view.figure)
        assert fem.plot_for_axes(volume_view.slider_axes) is volume_view

    def test_plot_for_axes_absent(self, line_plot):
        """An axes with no plot should have no owning plot."""
        fem = figure_event_manager(line_plot.figure)
        other_ax = line_plot.figure.add_subplot(2, 1, 2)
        assert fem.plot_for_axes(other_ax) is None

    def test_repeated_plot_in_axes(self, line_plot):
        """The first plot constructed in an axes should own it."""
        fem = figure_event_manager(line_plot.figure)
        other = kplt.plot(np.arange(10), ax=line_plot.axes)
        assert fem.plots == [line_plot, other]
        assert fem.plot_for_axes(line_plot.axes) is line_plot

    def test_plots_function(self, image):
        """The plots attached to a figure should be retrievable."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iviews = [kplt.imview(image, ax=a) for a in ax]
        assert kplt.plots(fig) == iviews

    def test_plots_function_no_manager(self):
        """A figure with no event manager should have no plots."""
        assert kplt.plots(plt.figure()) == []

    def test_plot_for_axes_function(self, volume_view):
        """The plot owning an axes should be retrievable."""
        assert kplt.plot_for_axes(volume_view.axes) is volume_view
        assert kplt.plot_for_axes(volume_view.slider_axes) is volume_view

    def test_plot_for_axes_function_no_manager(self):
        """An axes in a figure with no event manager should have no plot."""
        _, ax = plt.subplots()
        assert kplt.plot_for_axes(ax) is None


class TestScrollDispatch:
    """Tests for dispatch of mouse scroll events to plots."""

    def test_dispatch_to_owning_plot(self, image):
        """A scroll should be dispatched to the plot owning the axes."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, show_cbar=True, ax=ax[0])
        iview1 = kplt.imview(image, show_cbar=True, ax=ax[1])
        scroll_in_axes(iview1.cbar_axes, 0.5, 0.25, button="up")
        assert iview1.axesimage.norm.vmin > iview1.vmin_ref
        assert iview0.axesimage.norm.vmin == iview0.vmin_ref

    def test_dispatch_outside_axes(self, line_plot):
        """A scroll outside any axes should be discarded."""
        xlim_ref = line_plot.axes.get_xlim()
        scroll_outside_axes(line_plot.figure, button="up")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    def test_dispatch_in_axes_without_plot(self, line_plot):
        """A scroll in an axes with no plot should be discarded."""
        xlim_ref = line_plot.axes.get_xlim()
        other_ax = line_plot.figure.add_subplot(2, 1, 2)
        scroll_in_axes(other_ax, button="up")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    def test_zoom_scroll(self, line_plot):
        """Scroll in a line plot should zoom in about the event location."""
        xlim_ref = line_plot.axes.get_xlim()
        scroll_at_data(line_plot.axes, 5.0, 25.0, button="up")
        assert np.diff(line_plot.axes.get_xlim()) == pytest.approx(
            np.diff(xlim_ref) / 2.0
        )
        scroll_at_data(line_plot.axes, 5.0, 25.0, button="down")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    @pytest.mark.parametrize("key", FigureEventManager.monitored_keys)
    def test_zoom_scroll_with_modifier_key(self, line_plot, key):
        """A scroll with a modifier key depressed should not zoom."""
        xlim_ref = line_plot.axes.get_xlim()
        press_key(line_plot.figure, key)
        scroll_at_data(line_plot.axes, 5.0, 25.0, button="up")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    def test_zoom_toolbar_message(self, recording_toolbar):
        """An axis limit change should generate a toolbar message."""
        fig, ax = plt.subplots()
        fig.canvas.toolbar = recording_toolbar
        lplot = kplt.plot(np.arange(10), ax=ax)
        lplot.axes.set_xlim(2.0, 4.0)
        assert recording_toolbar.messages[-1].startswith("Zoom to")

    def test_colorbar_scroll_lower(self, image_view):
        """Scroll in the lower half of a vertical colorbar should set vmin."""
        vmin_ref, vmax_ref = image_view.vmin_ref, image_view.vmax_ref
        scroll_in_axes(image_view.cbar_axes, 0.5, 0.25, button="up")
        assert image_view.axesimage.norm.vmin == pytest.approx(
            vmin_ref + 0.02 * (vmax_ref - vmin_ref)
        )
        assert image_view.axesimage.norm.vmax == vmax_ref

    def test_colorbar_scroll_upper(self, image_view):
        """Scroll in the upper half of a vertical colorbar should set vmax."""
        vmin_ref, vmax_ref = image_view.vmin_ref, image_view.vmax_ref
        scroll_in_axes(image_view.cbar_axes, 0.5, 0.75, button="down")
        assert image_view.axesimage.norm.vmax == pytest.approx(
            vmax_ref - 0.02 * (vmax_ref - vmin_ref)
        )
        assert image_view.axesimage.norm.vmin == vmin_ref

    def test_horizontal_colorbar_scroll_lower(self, wide_image_view):
        """Scroll in the left half of a horizontal colorbar should set vmin."""
        iview = wide_image_view
        assert iview.cbar_axes._colorbar.orientation == "horizontal"
        scroll_in_axes(iview.cbar_axes, 0.25, 0.5, button="up")
        assert iview.axesimage.norm.vmin > iview.vmin_ref
        assert iview.axesimage.norm.vmax == iview.vmax_ref

    def test_horizontal_colorbar_scroll_upper(self, wide_image_view):
        """Scroll in the right half of a horizontal colorbar should set vmax."""
        iview = wide_image_view
        scroll_in_axes(iview.cbar_axes, 0.75, 0.5, button="down")
        assert iview.axesimage.norm.vmax < iview.vmax_ref
        assert iview.axesimage.norm.vmin == iview.vmin_ref

    @pytest.mark.parametrize("button", ["left", "right"])
    def test_colorbar_scroll_other_button(self, image_view, button):
        """A colorbar event for another button should not change the limits."""
        scroll_in_axes(image_view.cbar_axes, 0.5, 0.25, button=button)
        assert image_view.axesimage.norm.vmin == image_view.vmin_ref
        assert image_view.axesimage.norm.vmax == image_view.vmax_ref

    def test_colorbar_centre_scroll(self, image_view, monkeypatch):
        """A scroll at the exact centre of the colorbar should change nothing."""
        monkeypatch.setattr(
            type(image_view), "_cbar_event_rel_pos", lambda self, event: 0.5
        )
        event = type("Event", (), {"button": "up"})()
        assert image_view._scroll_cmap_handler(event)
        assert image_view.axesimage.norm.vmin == image_view.vmin_ref
        assert image_view.axesimage.norm.vmax == image_view.vmax_ref

    def test_scroll_outside_colorbar_not_claimed(self, image_view):
        """A scroll outside the colorbar should not be claimed as a shift."""
        event = type("Event", (), {"inaxes": None, "x": 0, "y": 0, "button": "up"})()
        assert not image_view._scroll_cmap_handler(event)

    def test_scroll_in_image_zooms(self, image_view):
        """Scroll in the image should zoom rather than change the limits."""
        xlim_ref = image_view.axes.get_xlim()
        scroll_in_axes(image_view.axes, 0.5, 0.5, button="up")
        assert np.diff(image_view.axes.get_xlim()) < np.diff(xlim_ref)
        assert image_view.axesimage.norm.vmin == image_view.vmin_ref

    def test_relative_position_outside_colorbar(self, image_view):
        """The relative position should be None for a non-colorbar event."""
        event = type("Event", (), {"inaxes": image_view.axes, "x": 0, "y": 0})()
        assert image_view._cbar_event_rel_pos(event) is None

    def test_relative_position_no_colorbar(self, image):
        """The relative position should be None when there is no colorbar."""
        iview = kplt.imview(image)
        event = type("Event", (), {"inaxes": iview.axes, "x": 0, "y": 0})()
        assert iview._cbar_event_rel_pos(event) is None

    def test_cmap_delta_default(self, image_view):
        """The default colormap shift fraction should be 0.02."""
        assert image_view.cmap_delta == 0.02

    def test_shift_scroll_increment_slice(self, volume_view, volume):
        """Scroll up with shift depressed should increment the slice index."""
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="up")
        assert volume_view.slice_index == 1
        np.testing.assert_allclose(volume_view.axesimage.get_array(), volume[1])
        assert volume_view.slider.val == 1

    def test_shift_scroll_decrement_slice(self, volume_view):
        """Scroll down with shift depressed should decrement the slice index."""
        volume_view.slice_index = 3
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="down")
        assert volume_view.slice_index == 2

    def test_shift_scroll_clipped_at_final_slice(self, volume_view, volume):
        """The slice index should not be incremented past the final slice."""
        volume_view.slice_index = volume.shape[0] - 1
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="up")
        assert volume_view.slice_index == volume.shape[0] - 1

    def test_shift_scroll_clipped_at_first_slice(self, volume_view):
        """The slice index should not be decremented below the first slice."""
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="down")
        assert volume_view.slice_index == 0

    @pytest.mark.parametrize("button", ["left", "right"])
    def test_shift_scroll_other_button(self, volume_view, button):
        """A shift scroll for another button should not change the slice."""
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button=button)
        assert volume_view.slice_index == 0

    def test_scroll_without_shift_zooms(self, volume_view):
        """Scroll without shift depressed should zoom."""
        xlim_ref = volume_view.axes.get_xlim()
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="up")
        assert volume_view.slice_index == 0
        assert np.diff(volume_view.axes.get_xlim()) < np.diff(xlim_ref)

    def test_volume_colorbar_scroll(self, volume_view):
        """Colorbar scroll support should be provided for a volume view."""
        scroll_in_axes(volume_view.cbar_axes, 0.5, 0.25, button="up")
        assert volume_view.axesimage.norm.vmin > volume_view.vmin_ref
