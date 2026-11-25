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
from komplot import (
    AxesEventManager,
    ColorbarEventManager,
    FigureEventManager,
    VolumeViewEventManager,
    ZoomEventManager,
    figure_event_manager,
)


@pytest.fixture
def line_plot():
    """A line plot with attached event managers."""
    x = np.linspace(0.0, 10.0, 32)
    return kplt.plot(x, x**2)


@pytest.fixture
def image_view(image):
    """An image view with a visible colorbar and attached event managers."""
    return kplt.imview(image, show_cbar=True)


@pytest.fixture
def wide_image_view(wide_image):
    """An image view with a horizontal colorbar."""
    return kplt.imview(wide_image, show_cbar=True)


@pytest.fixture
def volume_view(volume):
    """A volume view with a visible colorbar and attached event managers."""
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

    def test_axes_manager_registry(self, line_plot):
        """Axes event managers should be retrievable by axes."""
        fem = figure_event_manager(line_plot.figure)
        axem = fem.get_axevman_for_axes(line_plot.axes)
        assert isinstance(axem, ZoomEventManager)
        assert fem.axevman_from_ax[line_plot.axes] is axem

    def test_axes_manager_registry_absent(self, line_plot):
        """An unregistered axes should have no axes event manager."""
        fem = figure_event_manager(line_plot.figure)
        other_ax = line_plot.figure.add_subplot(2, 1, 2)
        assert fem.get_axevman_for_axes(other_ax) is None

    def test_initial_share_state(self, line_plot):
        """No axes should initially be shared."""
        fem = figure_event_manager(line_plot.figure)
        assert fem.slice_share_axes == []
        assert fem.cmap_share_axes == []

    def test_set_share_list(self):
        """Shared axes should be settable from a list."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        fem = FigureEventManager(fig)
        fem.set_slice_share(list(ax))
        fem.set_cmap_share(list(ax))
        assert fem.slice_share_axes == list(ax)
        assert fem.cmap_share_axes == list(ax)

    def test_set_share_array(self):
        """Shared axes should be settable from an array of axes."""
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fem = FigureEventManager(fig)
        fem.set_slice_share(ax)
        fem.set_cmap_share(ax)
        assert fem.slice_share_axes == ax.ravel().tolist()
        assert fem.cmap_share_axes == ax.ravel().tolist()


class TestAxesEventManager:
    """Tests for class AxesEventManager."""

    def test_attach(self, line_plot):
        """A constructed manager should attach itself to the axes."""
        fig = line_plot.figure
        fem = figure_event_manager(fig)
        ax = fig.add_subplot(2, 1, 2)
        axem = AxesEventManager(ax, fem, line_plot)
        assert axem.axes is ax
        assert axem.fig_event_man is fem
        assert axem.plot is line_plot
        assert fem.get_axevman_for_axes(ax) is axem

    def test_duplicate_attach(self, line_plot):
        """Attaching a second manager to an axes should raise an exception."""
        fem = figure_event_manager(line_plot.figure)
        with pytest.raises(RuntimeError, match="already has an event manager"):
            AxesEventManager(line_plot.axes, fem, line_plot)

    def test_scroll_capture_enabled(self, line_plot):
        """Scroll capture should be enabled for the canvas."""
        assert line_plot.figure.canvas.capture_scroll


class TestZoomEventManager:
    """Tests for class ZoomEventManager."""

    def test_scroll_zoom_in(self, line_plot):
        """Scroll up should zoom in about the event location."""
        xlim_ref = line_plot.axes.get_xlim()
        scroll_at_data(line_plot.axes, 5.0, 25.0, button="up")
        xlim = line_plot.axes.get_xlim()
        assert np.diff(xlim) == pytest.approx(np.diff(xlim_ref) / 2.0)

    def test_scroll_zoom_out(self, line_plot):
        """Scroll down should zoom out towards the reference limits."""
        xlim_ref = line_plot.axes.get_xlim()
        scroll_at_data(line_plot.axes, 5.0, 25.0, button="up")
        scroll_at_data(line_plot.axes, 5.0, 25.0, button="down")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    def test_zoom_scale(self, line_plot):
        """The zoom scaling factor should be as specified."""
        fig = line_plot.figure
        fem = figure_event_manager(fig)
        ax = fig.add_subplot(2, 1, 2)
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 10.0)
        zplot = kplt.ZoomablePlot(figure=fig, axes=ax)
        ZoomEventManager(ax, fem, zplot, zoom_scale=4.0)
        scroll_at_data(ax, 5.0, 5.0, button="up")
        assert np.diff(ax.get_xlim()) == pytest.approx(2.5)

    def test_scroll_outside_axes(self, line_plot):
        """A scroll outside the axes should not zoom."""
        xlim_ref = line_plot.axes.get_xlim()
        scroll_outside_axes(line_plot.figure, button="up")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    @pytest.mark.parametrize("key", FigureEventManager.monitored_keys)
    def test_scroll_with_modifier_key(self, line_plot, key):
        """A scroll with a modifier key depressed should not zoom."""
        xlim_ref = line_plot.axes.get_xlim()
        press_key(line_plot.figure, key)
        scroll_at_data(line_plot.axes, 5.0, 25.0, button="up")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    def test_scroll_in_other_axes(self, line_plot):
        """A scroll in another axes should not zoom this one."""
        fig = line_plot.figure
        xlim_ref = line_plot.axes.get_xlim()
        other_ax = fig.add_subplot(2, 1, 2)
        scroll_in_axes(other_ax, button="up")
        assert line_plot.axes.get_xlim() == pytest.approx(xlim_ref)

    def test_axis_limit_change_message(self, image, recording_toolbar):
        """An axis limit change should generate a toolbar message."""
        fig, ax = plt.subplots()
        fig.canvas.toolbar = recording_toolbar
        lplot = kplt.plot(np.arange(10), ax=ax)
        lplot.axes.set_xlim(2.0, 4.0)
        assert recording_toolbar.messages[-1].startswith("Zoom to")


class TestColorbarEventManager:
    """Tests for class ColorbarEventManager."""

    def test_manager_type(self, image_view):
        """An image view should have a colorbar event manager."""
        fem = figure_event_manager(image_view.figure)
        assert isinstance(
            fem.get_axevman_for_axes(image_view.axes), ColorbarEventManager
        )

    def test_scroll_vertical_colorbar_lower(self, image_view):
        """Scroll in the lower half of a vertical colorbar should set vmin."""
        vmin_ref, vmax_ref = image_view.vmin_ref, image_view.vmax_ref
        scroll_in_axes(image_view.cbar_axes, 0.5, 0.25, button="up")
        assert image_view.axesimage.norm.vmin == pytest.approx(
            vmin_ref + 0.02 * (vmax_ref - vmin_ref)
        )
        assert image_view.axesimage.norm.vmax == vmax_ref

    def test_scroll_vertical_colorbar_upper(self, image_view):
        """Scroll in the upper half of a vertical colorbar should set vmax."""
        vmin_ref, vmax_ref = image_view.vmin_ref, image_view.vmax_ref
        scroll_in_axes(image_view.cbar_axes, 0.5, 0.75, button="down")
        assert image_view.axesimage.norm.vmax == pytest.approx(
            vmax_ref - 0.02 * (vmax_ref - vmin_ref)
        )
        assert image_view.axesimage.norm.vmin == vmin_ref

    def test_scroll_horizontal_colorbar_lower(self, wide_image_view):
        """Scroll in the left half of a horizontal colorbar should set vmin."""
        iview = wide_image_view
        assert iview.cbar_axes._colorbar.orientation == "horizontal"
        scroll_in_axes(iview.cbar_axes, 0.25, 0.5, button="up")
        assert iview.axesimage.norm.vmin > iview.vmin_ref
        assert iview.axesimage.norm.vmax == iview.vmax_ref

    def test_scroll_horizontal_colorbar_upper(self, wide_image_view):
        """Scroll in the right half of a horizontal colorbar should set vmax."""
        iview = wide_image_view
        scroll_in_axes(iview.cbar_axes, 0.75, 0.5, button="down")
        assert iview.axesimage.norm.vmax < iview.vmax_ref
        assert iview.axesimage.norm.vmin == iview.vmin_ref

    @pytest.mark.parametrize("rel_pos", [0.5, None])
    def test_colorbar_centre_or_invalid_position(self, image_view, rel_pos):
        """The colormap limits should be unchanged at the colorbar centre."""
        fem = figure_event_manager(image_view.figure)
        axem = fem.get_axevman_for_axes(image_view.axes)
        event = type("Event", (), {"button": "up"})()
        axem.cmap_vminmax_event_handler(event, rel_pos)
        assert image_view.axesimage.norm.vmin == image_view.vmin_ref
        assert image_view.axesimage.norm.vmax == image_view.vmax_ref

    def test_scroll_in_image_zooms(self, image_view):
        """Scroll in the image should zoom rather than change the limits."""
        xlim_ref = image_view.axes.get_xlim()
        scroll_in_axes(image_view.axes, 0.5, 0.5, button="up")
        assert np.diff(image_view.axes.get_xlim()) < np.diff(xlim_ref)
        assert image_view.axesimage.norm.vmin == image_view.vmin_ref

    def test_relative_position_outside_colorbar(self, image_view):
        """The relative position should be None for a non-colorbar event."""
        fem = figure_event_manager(image_view.figure)
        axem = fem.get_axevman_for_axes(image_view.axes)
        event = type("Event", (), {"inaxes": image_view.axes, "x": 0, "y": 0})()
        assert axem.cbar_event_rel_pos(event) is None

    def test_relative_position_no_colorbar(self, image):
        """The relative position should be None when there is no colorbar."""
        iview = kplt.imview(image)
        fem = figure_event_manager(iview.figure)
        axem = fem.get_axevman_for_axes(iview.axes)
        event = type("Event", (), {"inaxes": None, "x": 0, "y": 0})()
        assert axem.cbar_event_rel_pos(event) is None

    def test_cmap_share(self, image):
        """Shared colormap axes should all respond to a colorbar scroll."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, show_cbar=True, ax=ax[0])
        iview1 = kplt.imview(2.0 * image, show_cbar=True, ax=ax[1])
        figure_event_manager(fig).set_cmap_share(ax)
        scroll_in_axes(iview0.cbar_axes, 0.5, 0.25, button="up")
        for iview in (iview0, iview1):
            assert iview.axesimage.norm.vmin == pytest.approx(
                iview.vmin_ref + 0.02 * (iview.vmax_ref - iview.vmin_ref)
            )

    @pytest.mark.parametrize(
        "manager_class,plot_name",
        [
            (ColorbarEventManager, "image_view"),
            (VolumeViewEventManager, "volume_view"),
        ],
    )
    def test_cmap_delta(self, request, manager_class, plot_name):
        """The colormap shift fraction should be as specified."""
        plot = request.getfixturevalue(plot_name)
        fem = figure_event_manager(plot.figure)
        ax = plot.figure.add_subplot(2, 1, 2)
        axem = manager_class(ax, fem, plot, cmap_delta=0.1)
        assert axem.cmap_delta == 0.1

    def test_cmap_delta_default(self, image_view):
        """The default colormap shift fraction should be 0.02."""
        fem = figure_event_manager(image_view.figure)
        axem = fem.get_axevman_for_axes(image_view.axes)
        assert axem.cmap_delta == 0.02


class TestVolumeViewEventManager:
    """Tests for class VolumeViewEventManager."""

    def test_manager_type(self, volume_view):
        """A volume view should have a volume view event manager."""
        fem = figure_event_manager(volume_view.figure)
        assert isinstance(
            fem.get_axevman_for_axes(volume_view.axes), VolumeViewEventManager
        )

    def test_shift_scroll_increment_slice(self, volume_view, volume):
        """Scroll up with shift depressed should increment the slice index."""
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="up")
        assert volume_view.slice_index == 1
        np.testing.assert_allclose(volume_view.axesimage.get_array(), volume[1])
        assert volume_view.slider.val == 1

    def test_shift_scroll_decrement_slice(self, volume_view):
        """Scroll down with shift depressed should decrement the slice index."""
        volume_view.set_volume_slice(3)
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="down")
        assert volume_view.slice_index == 2

    def test_shift_scroll_clipped_at_final_slice(self, volume_view, volume):
        """The slice index should not be incremented past the final slice."""
        volume_view.set_volume_slice(volume.shape[0] - 1)
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="up")
        assert volume_view.slice_index == volume.shape[0] - 1

    def test_shift_scroll_clipped_at_first_slice(self, volume_view):
        """The slice index should not be decremented below the first slice."""
        press_key(volume_view.figure, "shift")
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="down")
        assert volume_view.slice_index == 0

    def test_scroll_without_shift_zooms(self, volume_view):
        """Scroll without shift depressed should zoom."""
        xlim_ref = volume_view.axes.get_xlim()
        scroll_in_axes(volume_view.axes, 0.5, 0.5, button="up")
        assert volume_view.slice_index == 0
        assert np.diff(volume_view.axes.get_xlim()) < np.diff(xlim_ref)

    def test_slider_change(self, volume_view, volume):
        """A slider change should change the displayed slice."""
        volume_view.slider.set_val(2)
        assert volume_view.slice_index == 2
        np.testing.assert_allclose(volume_view.axesimage.get_array(), volume[2])

    def test_colorbar_scroll(self, volume_view):
        """Colorbar scroll support should be inherited."""
        scroll_in_axes(volume_view.cbar_axes, 0.5, 0.25, button="up")
        assert volume_view.axesimage.norm.vmin > volume_view.vmin_ref

    def test_slice_share_shift_scroll(self, volume):
        """Shared slice axes should all respond to a shift scroll."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        vview0 = kplt.volview(volume, ax=ax[0])
        vview1 = kplt.volview(volume, ax=ax[1])
        figure_event_manager(fig).set_slice_share(ax)
        press_key(fig, "shift")
        scroll_in_axes(vview0.axes, 0.5, 0.5, button="up")
        assert vview0.slice_index == 1
        assert vview1.slice_index == 1

    def test_slice_share_slider(self, volume):
        """A slider change should be propagated to shared slice axes."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        vview0 = kplt.volview(volume, ax=ax[0])
        vview1 = kplt.volview(volume, ax=ax[1])
        figure_event_manager(fig).set_slice_share(ax)
        vview0.slider.set_val(3)
        assert vview0.slice_index == 3
        assert vview1.slice_index == 3
        assert vview1.slider.val == 3
