# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._state."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import scroll_at_data, scroll_in_axes
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import komplot as kplt
from komplot import ColorbarMixin, Plot, ZoomMixin, figure_event_manager
from komplot._state import figure_and_axes, log_scale_kwargs


class ZoomTestPlot(ZoomMixin, Plot):
    """A minimal plot type supporting axis zoom, for testing the mixin."""

    def __init__(self, xlim=(0.0, 10.0), ylim=(0.0, 10.0), zoom_scale=2.0):
        """Create a figure and axes with the specified axis limits."""
        fig, ax = plt.subplots()
        super().__init__(figure=fig, axes=ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        self.init_zoom(zoom_scale=zoom_scale)


class ColorbarTestPlot(ColorbarMixin, ZoomMixin, Plot):
    """A minimal plot type supporting a colorbar, for testing the mixin."""

    def __init__(self, image, cmap_delta=0.02, **kwargs):
        """Display an image with a colorbar."""
        fig, ax = plt.subplots()
        super().__init__(figure=fig, axes=ax)
        axesimage = ax.imshow(image, vmin=image.min(), vmax=image.max(), **kwargs)
        divider = make_axes_locatable(ax)
        cbar_axes = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(axesimage, ax=ax, cax=cbar_axes)
        self.init_colorbar(
            axesimage,
            divider=divider,
            cbar_axes=cbar_axes,
            cmap_delta=cmap_delta,
        )
        self.init_zoom()


class AutoscaledColorbarTestPlot(ColorbarMixin, ZoomMixin, Plot):
    """A plot type with colormap limits determined by matplotlib.

    The colormap limits are determined by matplotlib autoscaling, which
    records them as integers when the displayed array is of integer
    type.
    """

    def __init__(self, image):
        """Display an image with a colorbar, without specifying limits."""
        fig, ax = plt.subplots()
        super().__init__(figure=fig, axes=ax)
        axesimage = ax.imshow(image)
        divider = make_axes_locatable(ax)
        cbar_axes = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(axesimage, ax=ax, cax=cbar_axes)
        self.init_colorbar(
            axesimage, divider=divider, cbar_axes=cbar_axes, cbar_orient="vertical"
        )
        self.init_zoom()


class TestLogScaleKwargs:
    """Tests for function log_scale_kwargs."""

    @pytest.mark.parametrize("log", [False, None])
    def test_no_log_scale(self, log):
        """A false specification should indicate no log scale."""
        assert log_scale_kwargs(log) is None

    def test_log_scale(self):
        """A true specification should give empty keyword arguments."""
        assert log_scale_kwargs(True) == {}

    @pytest.mark.parametrize("log", [{}, {"base": 2}])
    def test_log_scale_kwargs(self, log):
        """A dict specification should be used as the keyword arguments."""
        assert log_scale_kwargs(log) == log


class TestFigureAndAxes:
    """Tests for function figure_and_axes."""

    def test_new_figure(self):
        """A new figure and axes should be created when no axes specified."""
        fig, ax, new_fig = figure_and_axes(None)
        assert new_fig
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_figure() is fig

    def test_new_figure_attributes(self):
        """Figure size and number should be as specified."""
        fig, ax, new_fig = figure_and_axes(None, figsize=(4, 3), fignum=7)
        assert new_fig
        assert fig.number == 7
        assert tuple(fig.get_size_inches()) == (4.0, 3.0)

    def test_existing_axes(self):
        """The figure of a specified axes should be returned."""
        ext_fig, ext_ax = plt.subplots()
        fig, ax, new_fig = figure_and_axes(ext_ax)
        assert not new_fig
        assert fig is ext_fig
        assert ax is ext_ax

    def test_new_figure_proj3d(self):
        """A new 3d axes should be created when requested."""
        _, ax, new_fig = figure_and_axes(None, proj3d=True)
        assert new_fig
        assert ax.name == "3d"

    def test_existing_axes_proj3d(self):
        """An existing 2d axes should be replaced by a 3d axes."""
        ext_fig, ext_ax = plt.subplots()
        fig, ax, new_fig = figure_and_axes(ext_ax, proj3d=True)
        assert not new_fig
        assert fig is ext_fig
        assert ax is not ext_ax
        assert ax.name == "3d"
        assert ext_ax not in fig.axes

    def test_existing_3d_axes_proj3d(self):
        """An existing 3d axes should be left unchanged."""
        _, ext_ax = plt.subplots(subplot_kw={"projection": "3d"})
        _, ax, _ = figure_and_axes(ext_ax, proj3d=True)
        assert ax is ext_ax


class TestPlot:
    """Tests for class Plot."""

    def test_attributes(self):
        """Figure and axes attributes should be as specified."""
        fig, ax = plt.subplots()
        plot = Plot(figure=fig, axes=ax)
        assert plot.figure is fig
        assert plot.axes is ax
        assert not plot.new_figure

    def test_event_manager_registration(self):
        """A plot should register with the event manager of its figure."""
        fig, ax = plt.subplots()
        plot = Plot(figure=fig, axes=ax)
        fem = figure_event_manager(fig)
        assert plot.event_manager is fem
        assert fem.plots == [plot]
        assert fem.plot_for_axes(ax) is plot

    def test_owns_axes(self):
        """A plot should own its own axes only."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plot = Plot(figure=fig, axes=ax[0])
        assert plot.owns_axes(ax[0])
        assert not plot.owns_axes(ax[1])

    def test_no_scroll_handlers(self):
        """A plot with no feature mixins should have no scroll handlers."""
        fig, ax = plt.subplots()
        plot = Plot(figure=fig, axes=ax)
        assert plot.scroll_handlers() == ()
        assert not plot.handle_scroll(object())

    def test_show_new_figure(self, monkeypatch):
        """A figure created for the plot should be shown once only."""
        shown = []
        monkeypatch.setattr(Figure, "show", lambda self: shown.append(self))
        fig, ax = plt.subplots()
        plot = Plot(figure=fig, axes=ax, new_figure=True)
        plot.show_new_figure()
        plot.show_new_figure()
        assert shown == [fig]

    def test_show_existing_figure(self, monkeypatch):
        """A figure not created for the plot should not be shown."""
        shown = []
        monkeypatch.setattr(Figure, "show", lambda self: shown.append(self))
        fig, ax = plt.subplots()
        Plot(figure=fig, axes=ax).show_new_figure()
        assert shown == []

    def test_toolbar_message_without_toolbar(self):
        """A toolbar message should be discarded if there is no toolbar."""
        fig, ax = plt.subplots()
        assert fig.canvas.toolbar is None
        Plot(figure=fig, axes=ax).toolbar_message("message")

    def test_toolbar_message(self, toolbar_axes):
        """A toolbar message should be displayed if there is a toolbar."""
        fig, ax, toolbar = toolbar_axes
        Plot(figure=fig, axes=ax).toolbar_message("message")
        assert toolbar.messages == ["message"]


class TestScrollHandlerComposition:
    """Tests for the composition of scroll event handlers by the mixins."""

    @pytest.mark.parametrize(
        "plot_type,handlers",
        [
            (kplt.SurfacePlot, []),
            (kplt.LinePlot, ["zoom"]),
            (kplt.ContourPlot, ["cmap", "zoom"]),
            (kplt.ImageView, ["cmap", "zoom"]),
            (kplt.VolumeView, ["slice", "cmap", "zoom"]),
        ],
    )
    def test_handler_order(self, plot_type, handlers, image, volume, surface_data):
        """Handlers should be ordered as the mixins are declared."""
        data = {
            kplt.SurfacePlot: surface_data,
            kplt.LinePlot: image[0],
            kplt.ContourPlot: surface_data,
            kplt.ImageView: image,
            kplt.VolumeView: volume,
        }[plot_type]
        plot = plot_type(data)
        expected = [f"_scroll_{name}_handler" for name in handlers]
        assert [h.__name__ for h in plot.scroll_handlers()] == expected

    def test_custom_composition(self):
        """A plot type may be composed from any set of feature mixins."""
        plot = ZoomTestPlot()
        assert [h.__name__ for h in plot.scroll_handlers()] == ["_scroll_zoom_handler"]


class TestZoomMixin:
    """Tests for class ZoomMixin."""

    def test_limit_reference(self):
        """Reference limits should be initialized from the axes limits."""
        zplot = ZoomTestPlot(xlim=(1.0, 3.0), ylim=(-1.0, 5.0))
        assert zplot.xlim_ref == (1.0, 3.0)
        assert zplot.ylim_ref == (-1.0, 5.0)

    def test_zoom_in(self):
        """Zoom in at the centre should scale the limits about the centre."""
        zplot = ZoomTestPlot()
        zplot.zoom_view(5.0, 5.0, 0.5)
        assert zplot.axes.get_xlim() == pytest.approx((2.5, 7.5))
        assert zplot.axes.get_ylim() == pytest.approx((2.5, 7.5))

    def test_zoom_in_off_centre(self):
        """Zoom in should be centred on the event location."""
        zplot = ZoomTestPlot()
        zplot.zoom_view(2.0, 8.0, 0.5)
        assert zplot.axes.get_xlim() == pytest.approx((1.0, 6.0))
        assert zplot.axes.get_ylim() == pytest.approx((4.0, 9.0))

    def test_zoom_out_clipped_to_reference(self):
        """Zoom out should not exceed the reference limits."""
        zplot = ZoomTestPlot()
        zplot.zoom_view(5.0, 5.0, 0.5)
        zplot.zoom_view(5.0, 5.0, 8.0)
        assert zplot.axes.get_xlim() == pytest.approx((0.0, 10.0))
        assert zplot.axes.get_ylim() == pytest.approx((0.0, 10.0))

    def test_zoom_out_off_centre_clipped_to_reference(self):
        """Zoom out at an off-centre location should be shifted into range."""
        zplot = ZoomTestPlot()
        zplot.zoom_view(1.0, 9.0, 4.0)
        assert zplot.axes.get_xlim() == pytest.approx((0.0, 10.0))
        assert zplot.axes.get_ylim() == pytest.approx((0.0, 10.0))

    def test_zoom_out_clipped_to_upper_reference(self):
        """Zoom out should not exceed the upper reference limits."""
        zplot = ZoomTestPlot()
        zplot.axes.set_xlim(8.0, 10.0)
        zplot.axes.set_ylim(8.0, 10.0)
        zplot.zoom_view(9.5, 9.5, 2.0)
        assert zplot.axes.get_xlim() == pytest.approx((6.0, 10.0))
        assert zplot.axes.get_ylim() == pytest.approx((6.0, 10.0))

    def test_zoom_with_inverted_y_axis(self):
        """Zoom should respect an inverted y axis, as used for images."""
        zplot = ZoomTestPlot(ylim=(10.0, 0.0))
        zplot.zoom_view(5.0, 5.0, 0.5)
        ylim = zplot.axes.get_ylim()
        assert ylim[0] > ylim[1]
        assert ylim == pytest.approx((7.5, 2.5))
        zplot.zoom_view(5.0, 5.0, 8.0)
        assert zplot.axes.get_ylim() == pytest.approx((10.0, 0.0))

    def test_scroll_zoom_in(self):
        """Scroll up in the plot axes should zoom in."""
        zplot = ZoomTestPlot()
        scroll_at_data(zplot.axes, 5.0, 5.0, button="up")
        assert zplot.axes.get_xlim() == pytest.approx((2.5, 7.5))

    def test_scroll_zoom_out(self):
        """Scroll down in the plot axes should zoom out."""
        zplot = ZoomTestPlot()
        zplot.zoom_view(5.0, 5.0, 0.5)
        scroll_at_data(zplot.axes, 5.0, 5.0, button="down")
        assert zplot.axes.get_xlim() == pytest.approx((0.0, 10.0))

    def test_zoom_scale(self):
        """The zoom scaling factor should be as specified."""
        zplot = ZoomTestPlot(zoom_scale=4.0)
        scroll_at_data(zplot.axes, 5.0, 5.0, button="up")
        assert np.diff(zplot.axes.get_xlim()) == pytest.approx(2.5)

    def test_zoom_toolbar_message(self, toolbar_axes):
        """The zoom toolbar message should display the current limits."""
        fig, ax, toolbar = toolbar_axes
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 10.0)
        zplot = kplt.plot(np.arange(2), ax=ax)
        zplot.zoom_view(5.0, 5.0, 0.5)
        zplot.zoom_toolbar_message()
        assert toolbar.messages[-1].startswith("Zoom to")

    def test_zoom_toolbar_message_inverted_y_axis(self, toolbar_axes):
        """The zoom toolbar message should order inverted y limits."""
        fig, ax, toolbar = toolbar_axes
        plot = kplt.plot(np.arange(2), ax=ax)
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(10.0, 0.0)
        plot.zoom_toolbar_message()
        assert toolbar.messages[-1] == "Zoom to [0.0:10.0, 0.0:10.0]"


class TestColorbarMixin:
    """Tests for class ColorbarMixin."""

    def test_colormap_limit_reference(self, image):
        """Reference limits should be initialized from the colormap limits."""
        cbplot = ColorbarTestPlot(image)
        assert cbplot.vmin_ref == image.min()
        assert cbplot.vmax_ref == image.max()
        assert cbplot.xlim_ref == cbplot.axes.get_xlim()

    def test_no_share_group(self, image):
        """A plot should not initially belong to a colormap share group."""
        assert ColorbarTestPlot(image).cmap_group is None

    def test_owns_colorbar_axes(self, image):
        """The colorbar axes should be owned by the plot."""
        cbplot = ColorbarTestPlot(image)
        assert cbplot.owns_axes(cbplot.cbar_axes)
        assert cbplot.owns_axes(cbplot.axes)

    def test_shift_vmin(self, image):
        """A vmin shift should be relative to the reference range."""
        cbplot = ColorbarTestPlot(image)
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        cbplot.shift_cmap_vmin(0.25)
        assert cbplot.axesimage.norm.vmin == pytest.approx(
            cbplot.vmin_ref + 0.25 * vrange
        )
        assert cbplot.axesimage.norm.vmax == cbplot.vmax_ref

    def test_shift_vmax(self, image):
        """A vmax shift should be relative to the reference range."""
        cbplot = ColorbarTestPlot(image)
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        cbplot.shift_cmap_vmax(-0.25)
        assert cbplot.axesimage.norm.vmax == pytest.approx(
            cbplot.vmax_ref - 0.25 * vrange
        )
        assert cbplot.axesimage.norm.vmin == cbplot.vmin_ref

    def test_shift_vmin_clipped_to_reference(self, image):
        """A vmin shift should not reduce vmin below the reference value."""
        cbplot = ColorbarTestPlot(image)
        cbplot.shift_cmap_vmin(-1.0)
        assert cbplot.axesimage.norm.vmin == cbplot.vmin_ref

    def test_shift_vmax_clipped_to_reference(self, image):
        """A vmax shift should not increase vmax above the reference value."""
        cbplot = ColorbarTestPlot(image)
        cbplot.shift_cmap_vmax(1.0)
        assert cbplot.axesimage.norm.vmax == cbplot.vmax_ref

    def test_shift_vmin_not_above_vmax(self, image):
        """A vmin shift should not raise vmin above vmax."""
        cbplot = ColorbarTestPlot(image)
        for _ in range(100):
            cbplot.shift_cmap_vmin(0.1)
        assert cbplot.axesimage.norm.vmin < cbplot.axesimage.norm.vmax

    def test_shift_vmax_not_below_vmin(self, image):
        """A vmax shift should not lower vmax below vmin."""
        cbplot = ColorbarTestPlot(image)
        for _ in range(100):
            cbplot.shift_cmap_vmax(-0.1)
        assert cbplot.axesimage.norm.vmax > cbplot.axesimage.norm.vmin

    def test_shift_toolbar_messages(self, image, recording_toolbar):
        """Colormap limit shifts should generate toolbar messages."""
        cbplot = ColorbarTestPlot(image)
        cbplot.figure.canvas.toolbar = recording_toolbar
        cbplot.shift_cmap_vmin(0.1)
        cbplot.shift_cmap_vmax(-0.1)
        assert recording_toolbar.messages[0].startswith("Color map vmin set to")
        assert recording_toolbar.messages[1].startswith("Color map vmax set to")

    def test_scroll_colorbar_lower(self, image):
        """Scroll in the lower half of the colorbar should shift vmin."""
        cbplot = ColorbarTestPlot(image)
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.25, button="up")
        assert cbplot.axesimage.norm.vmin == pytest.approx(
            cbplot.vmin_ref + 0.02 * (cbplot.vmax_ref - cbplot.vmin_ref)
        )

    def test_scroll_colorbar_upper(self, image):
        """Scroll in the upper half of the colorbar should shift vmax."""
        cbplot = ColorbarTestPlot(image)
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.75, button="down")
        assert cbplot.axesimage.norm.vmax == pytest.approx(
            cbplot.vmax_ref - 0.02 * (cbplot.vmax_ref - cbplot.vmin_ref)
        )

    def test_cmap_delta(self, image):
        """The colormap shift fraction should be as specified."""
        cbplot = ColorbarTestPlot(image, cmap_delta=0.1)
        assert cbplot.cmap_delta == 0.1
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.25, button="up")
        assert cbplot.axesimage.norm.vmin == pytest.approx(
            cbplot.vmin_ref + 0.1 * (cbplot.vmax_ref - cbplot.vmin_ref)
        )

    def test_apply_cmap_vmin(self, image):
        """A vmin change may be applied without sharing the change."""
        cbplot = ColorbarTestPlot(image)
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        cbplot.apply_cmap_vmin(0.25)
        assert cbplot.axesimage.norm.vmin == pytest.approx(
            cbplot.vmin_ref + 0.25 * vrange
        )

    def test_apply_cmap_vmax(self, image):
        """A vmax change may be applied without sharing the change."""
        cbplot = ColorbarTestPlot(image)
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        cbplot.apply_cmap_vmax(-0.25)
        assert cbplot.axesimage.norm.vmax == pytest.approx(
            cbplot.vmax_ref - 0.25 * vrange
        )

    def test_colorbar_orientation_and_visibility(self, image):
        """The colorbar orientation and visibility should be recorded."""
        cbplot = ColorbarTestPlot(image)
        assert cbplot.cbar_orient == "vertical"
        assert cbplot.cbar_visible

    def test_invisible_colorbar_not_adjusted(self, image):
        """An invisible colorbar should not respond to a scroll event."""
        cbplot = ColorbarTestPlot(image)
        cbplot.cbar_visible = False
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.25, button="up")
        assert cbplot.axesimage.norm.vmin == cbplot.vmin_ref
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.75, button="down")
        assert cbplot.axesimage.norm.vmax == cbplot.vmax_ref

    @pytest.mark.parametrize("dtype", ["uint8", "int32", "float32", "float64"])
    def test_cmap_limits_type(self, image, dtype):
        """The colormap limits should be floats for any array type."""
        cbplot = AutoscaledColorbarTestPlot((16.0 * image).astype(dtype))
        assert all(isinstance(limit, float) for limit in cbplot.cmap_limits)
        assert cbplot.cmap_limits == (cbplot.vmin_ref, cbplot.vmax_ref)

    @pytest.mark.parametrize("dtype", ["uint8", "int32"])
    def test_shift_autoscaled_integer_limits(self, image, dtype):
        """Shifts should be supported for integer colormap limits.

        Matplotlib records the colormap limits of an array of integer
        type as integers rather than floats when it determines them by
        autoscaling.
        """
        cbplot = AutoscaledColorbarTestPlot((16.0 * image).astype(dtype))
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        cbplot.shift_cmap_vmin(0.25)
        cbplot.shift_cmap_vmax(-0.25)
        assert cbplot.cmap_limits == pytest.approx(
            (cbplot.vmin_ref + 0.25 * vrange, cbplot.vmax_ref - 0.25 * vrange)
        )

    @pytest.mark.parametrize("dtype", ["uint8", "int32"])
    def test_scroll_autoscaled_integer_limits(self, image, dtype):
        """A colorbar scroll should be supported for integer limits."""
        cbplot = AutoscaledColorbarTestPlot((16.0 * image).astype(dtype))
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.25, button="up")
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.75, button="down")
        assert cbplot.cmap_limits == pytest.approx(
            (cbplot.vmin_ref + 0.02 * vrange, cbplot.vmax_ref - 0.02 * vrange)
        )

    def test_zoom_still_supported(self, image):
        """Zoom support should be provided by the ZoomMixin base class."""
        cbplot = ColorbarTestPlot(image)
        xlim_ref = cbplot.xlim_ref
        cbplot.zoom_view(np.mean(xlim_ref), 2.0, 0.5)
        assert np.diff(cbplot.axes.get_xlim()) < np.diff(xlim_ref)
