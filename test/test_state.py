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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from komplot import ColorbarPlot, GenericPlot, ZoomablePlot
from komplot._state import figure_and_axes, log_scale_kwargs


def make_zoomable(xlim=(0.0, 10.0), ylim=(0.0, 10.0)):
    """Construct a ZoomablePlot with specified axis limits."""
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ZoomablePlot(figure=fig, axes=ax)


def make_colorbar_plot(image, **kwargs):
    """Construct a ColorbarPlot displaying an image."""
    fig, ax = plt.subplots()
    axim = ax.imshow(image, vmin=image.min(), vmax=image.max(), **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(axim, ax=ax, cax=cax)
    return ColorbarPlot(
        figure=fig, axes=ax, axesimage=axim, divider=divider, cbar_axes=cax
    )


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


class TestGenericPlot:
    """Tests for class GenericPlot."""

    def test_attributes(self):
        """Figure and axes attributes should be as specified."""
        fig, ax = plt.subplots()
        gplot = GenericPlot(figure=fig, axes=ax)
        assert gplot.figure is fig
        assert gplot.axes is ax

    def test_positional_args_not_supported(self):
        """Construction should require keyword arguments."""
        fig, ax = plt.subplots()
        with pytest.raises(TypeError):
            GenericPlot(fig, ax)

    def test_toolbar_message_without_toolbar(self):
        """A toolbar message should be discarded if there is no toolbar."""
        fig, ax = plt.subplots()
        assert fig.canvas.toolbar is None
        GenericPlot(figure=fig, axes=ax).toolbar_message("message")

    def test_toolbar_message(self, toolbar_axes):
        """A toolbar message should be displayed if there is a toolbar."""
        fig, ax, toolbar = toolbar_axes
        GenericPlot(figure=fig, axes=ax).toolbar_message("message")
        assert toolbar.messages == ["message"]


class TestZoomablePlot:
    """Tests for class ZoomablePlot."""

    def test_limit_reference(self):
        """Reference limits should be initialized from the axes limits."""
        zplot = make_zoomable(xlim=(1.0, 3.0), ylim=(-1.0, 5.0))
        assert zplot.xlim_ref == (1.0, 3.0)
        assert zplot.ylim_ref == (-1.0, 5.0)

    def test_zoom_in(self):
        """Zoom in at the centre should scale the limits about the centre."""
        zplot = make_zoomable()
        zplot.zoom_view(5.0, 5.0, 0.5)
        assert zplot.axes.get_xlim() == pytest.approx((2.5, 7.5))
        assert zplot.axes.get_ylim() == pytest.approx((2.5, 7.5))

    def test_zoom_in_off_centre(self):
        """Zoom in should be centred on the event location."""
        zplot = make_zoomable()
        zplot.zoom_view(2.0, 8.0, 0.5)
        assert zplot.axes.get_xlim() == pytest.approx((1.0, 6.0))
        assert zplot.axes.get_ylim() == pytest.approx((4.0, 9.0))

    def test_zoom_out_clipped_to_reference(self):
        """Zoom out should not exceed the reference limits."""
        zplot = make_zoomable()
        zplot.zoom_view(5.0, 5.0, 0.5)
        zplot.zoom_view(5.0, 5.0, 8.0)
        assert zplot.axes.get_xlim() == pytest.approx((0.0, 10.0))
        assert zplot.axes.get_ylim() == pytest.approx((0.0, 10.0))

    def test_zoom_out_off_centre_clipped_to_reference(self):
        """Zoom out at an off-centre location should be shifted into range."""
        zplot = make_zoomable()
        zplot.zoom_view(1.0, 9.0, 4.0)
        assert zplot.axes.get_xlim() == pytest.approx((0.0, 10.0))
        assert zplot.axes.get_ylim() == pytest.approx((0.0, 10.0))

    def test_zoom_out_clipped_to_upper_reference(self):
        """Zoom out should not exceed the upper reference limits."""
        zplot = make_zoomable()
        zplot.axes.set_xlim(8.0, 10.0)
        zplot.axes.set_ylim(8.0, 10.0)
        zplot.zoom_view(9.5, 9.5, 2.0)
        assert zplot.axes.get_xlim() == pytest.approx((6.0, 10.0))
        assert zplot.axes.get_ylim() == pytest.approx((6.0, 10.0))

    def test_zoom_with_inverted_y_axis(self):
        """Zoom should respect an inverted y axis, as used for images."""
        zplot = make_zoomable(ylim=(10.0, 0.0))
        zplot.zoom_view(5.0, 5.0, 0.5)
        ylim = zplot.axes.get_ylim()
        assert ylim[0] > ylim[1]
        assert ylim == pytest.approx((7.5, 2.5))
        zplot.zoom_view(5.0, 5.0, 8.0)
        assert zplot.axes.get_ylim() == pytest.approx((10.0, 0.0))

    def test_zoom_toolbar_message(self, toolbar_axes):
        """The zoom toolbar message should display the current limits."""
        fig, ax, toolbar = toolbar_axes
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 10.0)
        zplot = ZoomablePlot(figure=fig, axes=ax)
        zplot.zoom_view(5.0, 5.0, 0.5)
        zplot.zoom_toolbar_message()
        assert toolbar.messages[-1] == "Zoom to [2.5:7.5, 2.5:7.5]"

    def test_zoom_toolbar_message_inverted_y_axis(self, toolbar_axes):
        """The zoom toolbar message should order inverted y limits."""
        fig, ax, toolbar = toolbar_axes
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(10.0, 0.0)
        zplot = ZoomablePlot(figure=fig, axes=ax)
        zplot.zoom_toolbar_message()
        assert toolbar.messages[-1] == "Zoom to [0.0:10.0, 0.0:10.0]"


class TestColorbarPlot:
    """Tests for class ColorbarPlot."""

    def test_colormap_limit_reference(self, image):
        """Reference limits should be initialized from the colormap limits."""
        cbplot = make_colorbar_plot(image)
        assert cbplot.vmin_ref == image.min()
        assert cbplot.vmax_ref == image.max()
        assert cbplot.xlim_ref == cbplot.axes.get_xlim()

    def test_shift_vmin(self, image):
        """A vmin shift should be relative to the reference range."""
        cbplot = make_colorbar_plot(image)
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        cbplot.shift_cmap_vmin(0.25)
        assert cbplot.axesimage.norm.vmin == pytest.approx(
            cbplot.vmin_ref + 0.25 * vrange
        )
        assert cbplot.axesimage.norm.vmax == cbplot.vmax_ref

    def test_shift_vmax(self, image):
        """A vmax shift should be relative to the reference range."""
        cbplot = make_colorbar_plot(image)
        vrange = cbplot.vmax_ref - cbplot.vmin_ref
        cbplot.shift_cmap_vmax(-0.25)
        assert cbplot.axesimage.norm.vmax == pytest.approx(
            cbplot.vmax_ref - 0.25 * vrange
        )
        assert cbplot.axesimage.norm.vmin == cbplot.vmin_ref

    def test_shift_vmin_clipped_to_reference(self, image):
        """A vmin shift should not reduce vmin below the reference value."""
        cbplot = make_colorbar_plot(image)
        cbplot.shift_cmap_vmin(-1.0)
        assert cbplot.axesimage.norm.vmin == cbplot.vmin_ref

    def test_shift_vmax_clipped_to_reference(self, image):
        """A vmax shift should not increase vmax above the reference value."""
        cbplot = make_colorbar_plot(image)
        cbplot.shift_cmap_vmax(1.0)
        assert cbplot.axesimage.norm.vmax == cbplot.vmax_ref

    def test_shift_vmin_not_above_vmax(self, image):
        """A vmin shift should not raise vmin above vmax."""
        cbplot = make_colorbar_plot(image)
        for _ in range(100):
            cbplot.shift_cmap_vmin(0.1)
        assert cbplot.axesimage.norm.vmin < cbplot.axesimage.norm.vmax

    def test_shift_vmax_not_below_vmin(self, image):
        """A vmax shift should not lower vmax below vmin."""
        cbplot = make_colorbar_plot(image)
        for _ in range(100):
            cbplot.shift_cmap_vmax(-0.1)
        assert cbplot.axesimage.norm.vmax > cbplot.axesimage.norm.vmin

    def test_shift_toolbar_messages(self, image, recording_toolbar):
        """Colormap limit shifts should generate toolbar messages."""
        cbplot = make_colorbar_plot(image)
        toolbar = recording_toolbar
        cbplot.figure.canvas.toolbar = toolbar
        cbplot.shift_cmap_vmin(0.1)
        cbplot.shift_cmap_vmax(-0.1)
        assert toolbar.messages[0].startswith("Color map vmin set to")
        assert toolbar.messages[1].startswith("Color map vmax set to")

    def test_zoom_still_supported(self, image):
        """Zoom support should be inherited from ZoomablePlot."""
        cbplot = make_colorbar_plot(image)
        xlim_ref = cbplot.xlim_ref
        cbplot.zoom_view(np.mean(xlim_ref), 2.0, 0.5)
        assert np.diff(cbplot.axes.get_xlim()) < np.diff(xlim_ref)
