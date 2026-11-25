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
from conftest import move_mouse
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

import komplot as kplt
from komplot import ImageView, ImageViewEventManager, _imview, figure_event_manager
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

    def test_invisible_colorbar(self, image):
        """An invisible colorbar should be created when requested."""
        iview = kplt.imview(image, show_cbar=None)
        assert isinstance(iview.cbar_axes, Axes)
        assert not hasattr(iview.cbar_axes, "_colorbar")
        assert list(iview.cbar_axes.get_xticks()) == []
        assert list(iview.cbar_axes.get_yticks()) == []

    def test_invisible_colorbar_size(self, image):
        """An invisible colorbar should occupy the space of a visible one."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, show_cbar=None, ax=ax[0])
        iview1 = kplt.imview(image, show_cbar=True, ax=ax[1])
        fig.canvas.draw()
        bounds0 = iview0.axes.get_window_extent().bounds
        bounds1 = iview1.axes.get_window_extent().bounds
        assert bounds0[2] == pytest.approx(bounds1[2])
        assert bounds0[3] == pytest.approx(bounds1[3])

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

    def test_event_managers(self, image):
        """Figure and axes event managers should be attached."""
        iview = kplt.imview(image)
        fem = figure_event_manager(iview.figure)
        axem = fem.get_axevman_for_axes(iview.axes)
        assert isinstance(axem, ImageViewEventManager)
        assert axem.plot is iview

    def test_event_managers_shared_figure(self, image):
        """A single figure event manager should serve multiple axes."""
        _, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, ax=ax[0])
        iview1 = kplt.imview(image, ax=ax[1])
        fem = figure_event_manager(iview0.figure)
        assert fem is figure_event_manager(iview1.figure)
        assert fem.get_axevman_for_axes(ax[1]).plot is iview1
