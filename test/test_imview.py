# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._imview."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import move_mouse, scroll_in_axes
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

import komplot as kplt
from komplot import ImageView, _imview, figure_event_manager
from komplot._imview import _format_coord


class TestFormatCoord:
    """Tests for function _format_coord."""

    def test_monochrome(self, image):
        """Monochrome image pixel values should be displayed."""
        assert _format_coord(2.0, 3.0, image) == "x=  2.00, y=  3.00, z=20.00"

    def test_rounding_to_nearest_pixel(self, image):
        """Coordinates should be rounded to the nearest pixel."""
        assert _format_coord(1.6, 3.4, image) == "x=  1.60, y=  3.40, z=20.00"

    def test_colour(self, color_image):
        """All channels of a colour image pixel should be displayed."""
        pixel = color_image[3, 1]
        expected = "x=  1.00, y=  3.00, z=" + ",".join(f"{c:.2f}" for c in pixel)
        assert _format_coord(1.0, 3.0, color_image) == expected

    @pytest.mark.parametrize("coord", [(-2.0, 1.0), (1.0, -2.0), (100.0, 1.0)])
    def test_outside_image(self, image, coord):
        """No pixel value should be displayed outside the image support."""
        x, y = coord
        assert _format_coord(x, y, image) == f"x={x:.2f}, y={y:.2f}"


class TestImview:
    """Tests for function imview."""

    def test_return_type(self, image):
        """The returned object should be a fully initialized ImageView."""
        iview = kplt.imview(image)
        assert isinstance(iview, ImageView)
        assert iview.figure is iview.axes.get_figure()
        assert iview.axesimage in iview.axes.images

    def test_image_data(self, image):
        """The displayed data should be the specified image."""
        iview = kplt.imview(image)
        np.testing.assert_allclose(iview.axesimage.get_array(), image)

    def test_colour_image_data(self, color_image):
        """A colour image should be displayed without modification."""
        iview = kplt.imview(color_image)
        np.testing.assert_allclose(iview.axesimage.get_array(), color_image)

    @pytest.mark.parametrize("shape", [(8,), (8, 8, 2), (8, 8, 5), (2, 8, 8, 3)])
    def test_invalid_shape(self, shape):
        """An invalid image shape should raise an exception."""
        with pytest.raises(ValueError, match="not appropriate for image display"):
            kplt.imview(np.zeros(shape))

    def test_colormap_limits(self, image):
        """Colormap limits should default to the image value range."""
        iview = kplt.imview(image)
        assert iview.axesimage.norm.vmin == image.min()
        assert iview.axesimage.norm.vmax == image.max()

    def test_colormap_limits_quantile(self):
        """Colormap limits should be set from quantiles when specified."""
        rng = np.random.default_rng(3141)
        image = rng.uniform(size=(16, 16))
        iview = kplt.imview(image, vmin_quantile=0.1)
        vmin, vmax = np.quantile(image, [0.1, 0.9])
        assert iview.axesimage.norm.vmin == pytest.approx(vmin)
        assert iview.axesimage.norm.vmax == pytest.approx(vmax)

    def test_norm(self, image):
        """A specified norm should be used instead of the colormap limits."""
        norm = LogNorm(vmin=1.0, vmax=100.0)
        iview = kplt.imview(image + 1.0, norm=norm)
        assert iview.axesimage.norm is norm
        assert iview.vmin_ref == 1.0
        assert iview.vmax_ref == 100.0

    def test_default_colormap_monochrome(self, image):
        """A monochrome image should have a greyscale colormap by default."""
        assert kplt.imview(image).axesimage.get_cmap().name == "Greys_r"

    def test_specified_colormap(self, image):
        """A specified colormap should be used."""
        assert kplt.imview(image, cmap="viridis").axesimage.get_cmap().name == "viridis"

    def test_interpolation(self, image):
        """A specified interpolation should be used."""
        iview = kplt.imview(image, interpolation="bilinear")
        assert iview.axesimage.get_interpolation() == "bilinear"
        assert kplt.imview(image).axesimage.get_interpolation() == "nearest"

    def test_origin_upper(self, image):
        """Origin "upper" should invert the y axis and use top x ticks."""
        iview = kplt.imview(image, origin="upper")
        iview.figure.canvas.draw()
        ylim = iview.axes.get_ylim()
        assert ylim[0] > ylim[1]
        tick = iview.axes.xaxis.get_major_ticks()[0]
        assert tick.tick2line.get_visible()
        assert not tick.tick1line.get_visible()

    def test_origin_lower(self, image):
        """Origin "lower" should not invert the y axis and use bottom ticks."""
        iview = kplt.imview(image, origin="lower")
        iview.figure.canvas.draw()
        ylim = iview.axes.get_ylim()
        assert ylim[0] < ylim[1]
        tick = iview.axes.xaxis.get_major_ticks()[0]
        assert tick.tick1line.get_visible()
        assert not tick.tick2line.get_visible()

    def test_no_tick_labels(self, image):
        """Axis tick labels should be suppressed."""
        iview = kplt.imview(image)
        iview.figure.canvas.draw()
        assert all(t.get_text() == "" for t in iview.axes.get_xticklabels())
        assert all(t.get_text() == "" for t in iview.axes.get_yticklabels())

    def test_title(self, image):
        """A title should be set when specified."""
        assert kplt.imview(image, title="T").axes.get_title() == "T"

    def test_no_colorbar(self, image):
        """No colorbar should be created by default."""
        iview = kplt.imview(image)
        assert iview.cbar_axes is None
        assert iview.divider is None

    def test_colorbar(self, image):
        """A colorbar should be created when requested."""
        iview = kplt.imview(image, show_cbar=True)
        assert isinstance(iview.cbar_axes, Axes)
        assert isinstance(iview.divider, AxesDivider)
        assert iview.cbar_axes._colorbar.orientation == "vertical"

    def test_colorbar_orientation_wide_image(self, wide_image):
        """A wide image should have a horizontal colorbar."""
        iview = kplt.imview(wide_image, show_cbar=True)
        assert iview.cbar_axes._colorbar.orientation == "horizontal"

    def test_reserved_colorbar(self, image):
        """A colorbar should be created in axes reserved for it."""
        fig, ax = kplt.subplots(nrows=1, ncols=2, cbar=[False, True])
        iview0 = kplt.imview(image, ax=ax[0])
        iview1 = kplt.imview(image, ax=ax[1])
        assert iview0.cbar_axes is None
        assert iview1.cbar_axes is ax[1].cax
        assert iview1.cbar_axes.get_visible()
        # No space is taken from the axes of the image itself
        assert iview1.divider is None

    def test_reserved_colorbar_alignment(self, image, wide_image):
        """Reserved space should keep the plots of a figure aligned."""
        fig, ax = kplt.subplots(nrows=1, ncols=2, cbar=[False, True])
        kplt.imview(image, ax=ax[0])
        kplt.imview(image, ax=ax[1])
        fig.canvas.draw()
        cells = [axs.get_subplotspec().get_position(fig) for axs in ax]
        assert cells[0].width == pytest.approx(cells[1].width)
        assert cells[0].height == pytest.approx(cells[1].height)
        bounds = [axs.get_window_extent().bounds for axs in ax]
        assert bounds[0][2] == pytest.approx(bounds[1][2])
        assert bounds[0][3] == pytest.approx(bounds[1][3])

    def test_reserved_colorbar_resize(self, image):
        """Reserved space should keep the plots aligned when resized."""
        fig, ax = kplt.subplots(nrows=1, ncols=2, cbar=[False, True], figsize=(10, 4))
        kplt.imview(image, ax=ax[0])
        kplt.imview(image, ax=ax[1])
        for size in ((10, 4), (5, 4), (12, 8)):
            fig.set_size_inches(*size, forward=True)
            fig.canvas.draw()
            bounds = [axs.get_window_extent().bounds for axs in ax]
            assert bounds[0][2] == pytest.approx(bounds[1][2])
            assert bounds[0][3] == pytest.approx(bounds[1][3])

    def test_reserved_colorbar_declined(self, image):
        """A colorbar should not be displayed in reserved space if not required."""
        fig, ax = kplt.subplots(nrows=1, ncols=2, cbar=True)
        iview = kplt.imview(image, show_cbar=False, ax=ax[0])
        assert iview.cbar_axes is None
        assert iview.cbar_orient is None
        assert not ax[0].cax.get_visible()

    def test_shared_colorbar(self, image):
        """A colorbar shared by multiple plots should be drawn by the first."""
        fig, ax = kplt.subplots(nrows=1, ncols=2, cbar="shared")
        iview0 = kplt.imview(image, ax=ax[0])
        iview1 = kplt.imview(image, ax=ax[1])
        assert iview0.cbar_axes is ax[0].cax
        assert iview1.cbar_axes is None
        assert iview1.cbar_orient is None
        # The plot that drew the colorbar owns its axes
        assert kplt.plot_for_axes(ax[0].cax) is iview0

    def test_shared_colorbar_range_warning(self, image):
        """A plot sharing a colorbar of a different range should warn."""
        fig, ax = kplt.subplots(nrows=1, ncols=2, cbar="shared")
        kplt.imview(image, ax=ax[0])
        with pytest.warns(UserWarning, match="sharing a colorbar"):
            kplt.imview(2.0 * image, ax=ax[1])

    def test_shared_colorbar_same_range(self, image, recwarn):
        """A plot sharing a colorbar of the same range should not warn."""
        fig, ax = kplt.subplots(nrows=1, ncols=2, cbar="shared")
        kplt.imview(image, ax=ax[0])
        kplt.imview(image, ax=ax[1])
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_data_cursor(self, image, monkeypatch):
        """A data cursor should be attached when mplcursors is available."""
        cursor_args: list = []
        monkeypatch.setattr(
            _imview, "mplcrs", SimpleNamespace(cursor=cursor_args.append), raising=False
        )
        monkeypatch.setattr(_imview, "HAVE_MPLCRS", True)
        iview = kplt.imview(image)
        assert cursor_args == [iview.axesimage]

    def test_coord_format(self, image):
        """The cursor coordinate display should include the pixel value."""
        iview = kplt.imview(image)
        assert iview.axes.format_coord(2.0, 3.0) == _format_coord(2.0, 3.0, image)

    def test_coord_statusbar(self, image, toolbar_axes):
        """Mouse motion should display coordinates and value in the toolbar."""
        fig, ax, toolbar = toolbar_axes
        kplt.imview(image, ax=ax)
        move_mouse(ax, 2.0, 3.0)
        assert toolbar.messages[-1] == _format_coord(2.0, 3.0, image)

    def test_figure_attributes(self, image):
        """Figure size and number should be as specified."""
        iview = kplt.imview(image, figsize=(5, 4), fignum=4)
        assert iview.figure.number == 4
        assert tuple(iview.figure.get_size_inches()) == (5.0, 4.0)

    def test_specified_axes(self, image):
        """Display in a specified axes should not create a figure."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iview = kplt.imview(image, ax=ax[0])
        assert iview.axes is ax[0]
        assert iview.figure is fig
        assert plt.get_fignums() == [fig.number]

    @pytest.mark.parametrize("show_cbar", [True, False, None])
    def test_colorbar_presence(self, image, show_cbar):
        """A colorbar should be displayed only when required."""
        iview = kplt.imview(image, show_cbar=show_cbar)
        # Without space reserved for one, a colorbar is displayed only on request
        assert (iview.cbar_axes is not None) == bool(show_cbar)
        assert (iview.cbar_orient is None) == (not show_cbar)

    @pytest.mark.parametrize("show_cbar", [True, None])
    def test_reserved_colorbar_presence(self, image, show_cbar):
        """A colorbar should be displayed by default in reserved space."""
        fig, ax = kplt.subplots(cbar=True)
        iview = kplt.imview(image, show_cbar=show_cbar, ax=ax)
        assert iview.cbar_axes is ax.cax

    def test_colorbar_orientation(self, image, wide_image):
        """Colorbar orientation should be determined by the image shape."""
        assert kplt.imview(image, show_cbar=True).cbar_orient == "vertical"
        assert kplt.imview(wide_image, show_cbar=True).cbar_orient == "horizontal"

    @pytest.mark.parametrize(
        "location,orient", [("right", "vertical"), ("bottom", "horizontal")]
    )
    def test_reserved_colorbar_orientation(self, image, location, orient):
        """Colorbar orientation should follow the shape of the reserved axes."""
        fig, ax = kplt.subplots(cbar=True, cbar_location=location)
        assert kplt.imview(image, ax=ax).cbar_orient == orient

    def test_cmap_adjustment_bounds(self, image):
        """Colormap adjustment should be bounded by the image range."""
        iview = kplt.imview(image, show_cbar=True, vmin_quantile=0.1)
        assert iview.vmin_ref > image.min()
        assert iview.vmax_ref < image.max()
        assert (iview.vmin_bound, iview.vmax_bound) == (image.min(), image.max())

    def test_cmap_adjustment_bounds_default(self, image):
        """The adjustment bounds should be the image range by default."""
        iview = kplt.imview(image, show_cbar=True)
        assert (iview.vmin_bound, iview.vmax_bound) == (image.min(), image.max())
        assert (iview.vmin_bound, iview.vmax_bound) == (
            iview.vmin_ref,
            iview.vmax_ref,
        )

    def test_cmap_adjustment_beyond_initial_limits(self, image):
        """A colorbar scroll should be able to widen the colormap range."""
        iview = kplt.imview(image, show_cbar=True, vmin_quantile=0.1)
        vrange = iview.vmax_ref - iview.vmin_ref
        scroll_in_axes(iview.cbar_axes, 0.5, 0.25, button="down")
        assert iview.cmap_limits[0] == pytest.approx(iview.vmin_ref - 0.02 * vrange)
        span = iview.cmap_limits[1] - iview.cmap_limits[0]
        scroll_in_axes(iview.cbar_axes, 0.5, 0.75, button="up")
        assert iview.cmap_limits[1] == pytest.approx(iview.vmax_ref + 0.02 * span)

    def test_event_manager(self, image):
        """The plot should be registered with the figure event manager."""
        iview = kplt.imview(image)
        fem = figure_event_manager(iview.figure)
        assert iview.event_manager is fem
        assert fem.plot_for_axes(iview.axes) is iview

    def test_event_manager_shared_figure(self, image):
        """A single figure event manager should serve multiple plots."""
        _, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, ax=ax[0])
        iview1 = kplt.imview(image, ax=ax[1])
        fem = figure_event_manager(iview0.figure)
        assert fem is figure_event_manager(iview1.figure)
        assert fem.plots == [iview0, iview1]

    def test_feature_support(self, image):
        """The plot should support colormap range adjustment and zoom."""
        iview = kplt.imview(image)
        assert isinstance(iview, (kplt.ColorbarMixin, kplt.ZoomMixin))
        assert iview.vmin_ref == iview.axesimage.norm.vmin
        assert iview.xlim_ref == iview.axes.get_xlim()
