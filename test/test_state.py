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
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import komplot as kplt
from komplot import ColorbarMixin, Plot, ZoomMixin, figure_event_manager
from komplot._state import data_limits, figure_and_axes, log_scale_kwargs


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


class ClippedColorbarTestPlot(ColorbarMixin, ZoomMixin, Plot):
    """A plot type with a colormap range narrower than its data range."""

    def __init__(self, image, vmin, vmax, **kwargs):
        """Display an image with colormap limits within its data range."""
        fig, ax = plt.subplots()
        super().__init__(figure=fig, axes=ax)
        axesimage = ax.imshow(image, vmin=vmin, vmax=vmax, **kwargs)
        divider = make_axes_locatable(ax)
        cbar_axes = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(axesimage, ax=ax, cax=cbar_axes)
        self.init_colorbar(
            axesimage,
            divider=divider,
            cbar_axes=cbar_axes,
            cbar_orient="vertical",
            cmap_data_limits=data_limits(image),
        )
        self.init_zoom()


class TestDataLimits:
    """Tests for function data_limits."""

    def test_limits(self, image):
        """The limits should be the minimum and maximum array values."""
        assert data_limits(image) == (image.min(), image.max())

    def test_limits_are_floats(self, image):
        """The limits should be floats for an array of any type."""
        assert all(
            isinstance(limit, float)
            for limit in data_limits((16.0 * image).astype("uint8"))
        )

    def test_nan_ignored(self, image):
        """NaN values should be ignored."""
        nan_image = np.concatenate((image.ravel(), [np.nan]))
        assert data_limits(nan_image) == data_limits(image)


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
        """A vmin shift should not reduce vmin below the default bound."""
        cbplot = ColorbarTestPlot(image)
        cbplot.shift_cmap_vmin(-1.0)
        assert cbplot.axesimage.norm.vmin == cbplot.vmin_ref

    def test_shift_vmax_clipped_to_reference(self, image):
        """A vmax shift should not increase vmax above the default bound."""
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

    def test_colorbar_orientation(self, image):
        """The colorbar orientation should be recorded."""
        cbplot = ColorbarTestPlot(image)
        assert cbplot.cbar_orient == "vertical"

    def test_colorbar_absent_not_adjusted(self, image):
        """A plot without a colorbar should not respond to a scroll event."""
        cbplot = ColorbarTestPlot(image)
        cbar_axes, cbplot.cbar_axes = cbplot.cbar_axes, None
        scroll_in_axes(cbar_axes, 0.5, 0.25, button="up")
        assert cbplot.axesimage.norm.vmin == cbplot.vmin_ref
        scroll_in_axes(cbar_axes, 0.5, 0.75, button="down")
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
        assert cbplot.cmap_limits[0] == pytest.approx(cbplot.vmin_ref + 0.25 * vrange)
        span = cbplot.cmap_limits[1] - cbplot.cmap_limits[0]
        cbplot.shift_cmap_vmax(-0.25)
        assert cbplot.cmap_limits[1] == pytest.approx(cbplot.vmax_ref - 0.25 * span)

    @pytest.mark.parametrize("dtype", ["uint8", "int32"])
    def test_scroll_autoscaled_integer_limits(self, image, dtype):
        """A colorbar scroll should be supported for integer limits."""
        cbplot = AutoscaledColorbarTestPlot((16.0 * image).astype(dtype))
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.25, button="up")
        assert cbplot.cmap_limits[0] == pytest.approx(cbplot.vmin_ref + 0.02 * vrange)
        span = cbplot.cmap_limits[1] - cbplot.cmap_limits[0]
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.75, button="down")
        assert cbplot.cmap_limits[1] == pytest.approx(cbplot.vmax_ref - 0.02 * span)

    def test_bounds_default_to_reference_limits(self, image):
        """The adjustment bounds should default to the reference limits."""
        cbplot = ColorbarTestPlot(image)
        assert (cbplot.vmin_bound, cbplot.vmax_bound) == (
            cbplot.vmin_ref,
            cbplot.vmax_ref,
        )

    def test_bounds_from_data_limits(self, image):
        """The adjustment bounds should be the range of the data."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        assert (cbplot.vmin_ref, cbplot.vmax_ref) == (10.0, 20.0)
        assert (cbplot.vmin_bound, cbplot.vmax_bound) == data_limits(image)

    def test_bounds_not_narrower_than_reference_limits(self, image):
        """Data limits within the reference limits should not be bounds."""
        cbplot = ClippedColorbarTestPlot(image, -10.0, 100.0)
        assert (cbplot.vmin_bound, cbplot.vmax_bound) == (-10.0, 100.0)

    def test_bounds_ignored_for_nonlinear_norm(self, image):
        """Data limits should not bound the adjustment of a nonlinear norm.

        A colormap limit valid for the data is not necessarily valid for
        a nonlinear norm, the vmin of a LogNorm of data including
        nonpositive values being an example.
        """
        cbplot = ClippedColorbarTestPlot(image, None, None, norm=LogNorm())
        assert image.min() == 0.0
        assert cbplot.vmin_ref > 0.0
        assert (cbplot.vmin_bound, cbplot.vmax_bound) == (
            cbplot.vmin_ref,
            cbplot.vmax_ref,
        )

    def test_bounds_ignored_for_nonfinite_data_limits(self, image):
        """Nonfinite data limits should not bound the adjustment."""
        image = image.copy()
        image[0, 0] = np.inf
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        assert (cbplot.vmin_bound, cbplot.vmax_bound) == (10.0, 20.0)

    def test_shift_vmin_below_reference(self, image):
        """A vmin shift should be able to widen the colormap range."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.shift_cmap_vmin(-0.5)
        assert cbplot.cmap_limits[0] == pytest.approx(5.0)

    def test_shift_vmax_above_reference(self, image):
        """A vmax shift should be able to widen the colormap range."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.shift_cmap_vmax(0.5)
        assert cbplot.cmap_limits[1] == pytest.approx(25.0)

    def test_shift_vmin_clipped_to_bound(self, image):
        """A vmin shift should not reduce vmin below the lower bound."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.shift_cmap_vmin(-100.0)
        assert cbplot.cmap_limits[0] == cbplot.vmin_bound == image.min()

    def test_shift_vmax_clipped_to_bound(self, image):
        """A vmax shift should not increase vmax above the upper bound."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.shift_cmap_vmax(100.0)
        assert cbplot.cmap_limits[1] == cbplot.vmax_bound == image.max()

    def test_shift_size_relative_to_current_range(self, image):
        """A shift should be relative to the current colormap range."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        scroll_in_axes(cbplot.cbar_axes, 0.5, 0.25, button="down")
        assert cbplot.cmap_limits[0] == pytest.approx(10.0 - 0.02 * 10.0)

    def test_widening_shifts_accelerate(self, image):
        """Successive widening shifts should grow with the range."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.shift_cmap_vmax(0.5)  # range 10.0 -> 15.0
        assert cbplot.cmap_limits == pytest.approx((10.0, 25.0))
        cbplot.shift_cmap_vmax(0.5)  # range 15.0 -> 22.5
        assert cbplot.cmap_limits == pytest.approx((10.0, 32.5))

    def test_narrowing_shifts_decelerate(self, image):
        """Successive narrowing shifts should shrink with the range."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.shift_cmap_vmin(0.5)  # range 10.0 -> 5.0
        assert cbplot.cmap_limits == pytest.approx((15.0, 20.0))
        cbplot.shift_cmap_vmin(0.5)  # range 5.0 -> 2.5
        assert cbplot.cmap_limits == pytest.approx((17.5, 20.0))

    def test_minimum_span_default(self, image):
        """The minimum colormap range should be a fraction of the initial one."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        assert cbplot.cmap_span_min == pytest.approx(1e-3 * 10.0)

    def test_narrowing_limited_to_minimum_span(self, image):
        """Repeated narrowing should not reduce the range below the minimum.

        Since each shift is proportional to the current colormap range,
        repeated narrowing would otherwise drive the range towards zero,
        at which point matplotlib expands it again, with erratic
        results.
        """
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        for _ in range(40):
            cbplot.shift_cmap_vmin(0.5)
            cbplot.shift_cmap_vmax(-0.5)
        vmin, vmax = cbplot.cmap_limits
        assert vmax - vmin == pytest.approx(cbplot.cmap_span_min)
        assert 10.0 < vmin < vmax < 20.0

    def test_narrowing_limited_to_specified_minimum_span(self, image):
        """The minimum colormap range should be settable by the user."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.cmap_span_min = 1.0
        for _ in range(20):
            cbplot.shift_cmap_vmin(0.5)
        assert cbplot.cmap_limits == pytest.approx((19.0, 20.0))

    def test_widening_to_bound_from_minimum_span(self, image):
        """A range narrowed to the minimum should be able to widen again."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        for _ in range(40):
            cbplot.shift_cmap_vmin(0.5)
        for _ in range(40):
            cbplot.shift_cmap_vmin(-0.5)
        assert cbplot.cmap_limits[0] == cbplot.vmin_bound

    def test_bound_reached_in_bounded_number_of_shifts(self, image):
        """Traversal of the adjustment range should not depend on its size.

        Since each shift is proportional to the current colormap range,
        the number of shifts required to widen the range from the
        initial limits to the bounds depends on their ratio rather than
        on their difference.
        """
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        shifts = 0
        while cbplot.cmap_limits != (cbplot.vmin_bound, cbplot.vmax_bound):
            cbplot.shift_cmap_vmin(-cbplot.cmap_delta)
            cbplot.shift_cmap_vmax(cbplot.cmap_delta)
            shifts += 1
            assert shifts < 200
        assert 50 < shifts < 150

    def test_user_specified_bound(self, image):
        """The adjustment bounds should be settable by the user."""
        cbplot = ClippedColorbarTestPlot(image, 10.0, 20.0)
        cbplot.vmin_bound = 8.0
        cbplot.shift_cmap_vmin(-100.0)
        assert cbplot.cmap_limits[0] == 8.0

    def test_zoom_still_supported(self, image):
        """Zoom support should be provided by the ZoomMixin base class."""
        cbplot = ColorbarTestPlot(image)
        xlim_ref = cbplot.xlim_ref
        cbplot.zoom_view(np.mean(xlim_ref), 2.0, 0.5)
        assert np.diff(cbplot.axes.get_xlim()) < np.diff(xlim_ref)
