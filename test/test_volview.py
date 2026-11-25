# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._volview."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider

import komplot as kplt
from komplot import VolumeView, figure_event_manager


class TestVolview:
    """Tests for function volview."""

    def test_return_type(self, volume):
        """The returned object should be a fully initialized VolumeView."""
        vview = kplt.volview(volume)
        assert isinstance(vview, VolumeView)
        assert vview.figure is vview.axes.get_figure()
        assert vview.axesimage in vview.axes.images
        assert isinstance(vview.slider_axes, Axes)
        assert isinstance(vview.slider, Slider)

    def test_initial_slice(self, volume):
        """The initial slice displayed should be the first slice."""
        vview = kplt.volview(volume)
        assert vview.slice_index == 0
        np.testing.assert_allclose(vview.axesimage.get_array(), volume[0])

    def test_volume_attribute(self, volume):
        """The volume attribute should record the volume array."""
        np.testing.assert_allclose(kplt.volview(volume).volume, volume)

    @pytest.mark.parametrize("slice_axis", [0, 1, 2, -1, -2, -3])
    def test_slice_axis(self, volume, slice_axis):
        """Slices should be selected from the specified axis."""
        vview = kplt.volview(volume, slice_axis=slice_axis)
        expected = np.moveaxis(volume, slice_axis, 0)
        np.testing.assert_allclose(vview.volume, expected)
        np.testing.assert_allclose(vview.axesimage.get_array(), expected[0])
        assert vview.slider.valmax == expected.shape[0] - 1

    @pytest.mark.parametrize("slice_axis", [0, 1, 2])
    def test_slice_axis_colour_volume(self, color_volume, slice_axis):
        """Slices of a colour volume should retain the channel axis."""
        vview = kplt.volview(color_volume, slice_axis=slice_axis)
        expected = np.moveaxis(color_volume, slice_axis, 0)
        np.testing.assert_allclose(vview.volume, expected)
        assert vview.axesimage.get_array().shape == expected.shape[1:]

    @pytest.mark.parametrize(
        "shape,slice_axis",
        [
            ((8, 8), 0),
            ((8,), 0),
            ((2, 4, 8, 8), 0),
            ((2, 8, 8, 5), 0),
            ((2, 8, 8, 3), 3),
        ],
    )
    def test_invalid_shape(self, shape, slice_axis):
        """An invalid volume shape should raise an exception."""
        with pytest.raises(ValueError, match="not appropriate for volume slice"):
            kplt.volview(np.zeros(shape), slice_axis=slice_axis)

    def test_colormap_limits(self, volume):
        """Colormap limits should default to the volume value range."""
        vview = kplt.volview(volume)
        assert vview.axesimage.norm.vmin == volume.min()
        assert vview.axesimage.norm.vmax == volume.max()

    def test_colormap_limits_quantile(self, volume):
        """Colormap limits should be set from quantiles when specified."""
        vview = kplt.volview(volume, vmin_quantile=0.05)
        vmin, vmax = np.quantile(volume, [0.05, 0.95])
        assert vview.axesimage.norm.vmin == pytest.approx(vmin)
        assert vview.axesimage.norm.vmax == pytest.approx(vmax)

    def test_norm(self, volume):
        """A specified norm should be used instead of the colormap limits."""
        norm = Normalize(vmin=-1.0, vmax=2.0)
        vview = kplt.volview(volume, norm=norm)
        assert vview.axesimage.norm is norm
        assert vview.vmin_ref == -1.0
        assert vview.vmax_ref == 2.0

    def test_slider(self, volume):
        """The slider range should correspond to the number of slices."""
        slider = kplt.volview(volume).slider
        assert slider.valmin == 0
        assert slider.valmax == volume.shape[0] - 1
        assert slider.val == 0
        assert slider.orientation == "horizontal"

    def test_slider_orientation_tall_slice(self):
        """A tall volume slice should have a vertical slider."""
        vview = kplt.volview(np.zeros((3, 20, 5)))
        assert vview.slider.orientation == "vertical"

    def test_colorbar(self, volume):
        """A colorbar should be created when requested."""
        vview = kplt.volview(volume, show_cbar=True)
        assert isinstance(vview.cbar_axes, Axes)
        assert kplt.volview(volume).cbar_axes is None

    def test_divider(self, volume):
        """A divider should be created even without a colorbar."""
        assert kplt.volview(volume).divider is not None

    def test_specified_axes(self, volume):
        """Display in a specified axes should not create a figure."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        vview = kplt.volview(volume, ax=ax[1])
        assert vview.axes is ax[1]
        assert vview.figure is fig
        assert plt.get_fignums() == [fig.number]

    def test_event_manager(self, volume):
        """The plot should be registered with the figure event manager."""
        vview = kplt.volview(volume)
        fem = figure_event_manager(vview.figure)
        assert vview.event_manager is fem
        assert fem.plot_for_axes(vview.axes) is vview

    def test_feature_support(self, volume):
        """The plot should support slice selection, colormap, and zoom."""
        vview = kplt.volview(volume)
        assert isinstance(vview, (kplt.SliceMixin, kplt.ColorbarMixin, kplt.ZoomMixin))
        assert vview.num_slices == volume.shape[0]
        assert vview.vmin_ref == vview.axesimage.norm.vmin
        assert vview.xlim_ref == vview.axes.get_xlim()


class TestSetVolumeSlice:
    """Tests for volume slice selection."""

    def test_initial_slice(self, volume):
        """The first slice should initially be displayed."""
        vview = kplt.volview(volume)
        assert vview.slice_index == 0
        np.testing.assert_allclose(vview.axesimage.get_array(), volume[0])

    def test_slice_index_property(self, volume):
        """Assignment to the slice index property should set the slice."""
        vview = kplt.volview(volume)
        vview.slice_index = 2
        assert vview.slice_index == 2
        np.testing.assert_allclose(vview.axesimage.get_array(), volume[2])
        assert vview.slider.val == 2

    @pytest.mark.parametrize("index,expected", [(-2, 0), (100, 4)])
    def test_slice_index_clipped(self, volume, index, expected):
        """An out of range slice index should be clipped to the valid range."""
        vview = kplt.volview(volume)
        vview.slice_index = index
        assert vview.slice_index == expected

    def test_apply_volume_slice(self, volume):
        """A slice may be displayed without sharing the change."""
        vview = kplt.volview(volume)
        vview.apply_volume_slice(2)
        assert vview.slice_index == 2
        np.testing.assert_allclose(vview.axesimage.get_array(), volume[2])

    def test_no_share_group(self, volume):
        """A plot should not initially belong to a slice share group."""
        assert kplt.volview(volume).slice_group is None

    def test_set_slice(self, volume):
        """Setting the slice index should update the displayed slice."""
        vview = kplt.volview(volume)
        vview.set_volume_slice(3)
        assert vview.slice_index == 3
        np.testing.assert_allclose(vview.axesimage.get_array(), volume[3])

    def test_slider_updated(self, volume):
        """Setting the slice index should update the slider by default."""
        vview = kplt.volview(volume)
        vview.set_volume_slice(2)
        assert vview.slider.val == 2

    def test_slider_not_updated(self, volume):
        """The slider should not be updated when so specified."""
        vview = kplt.volview(volume)
        vview.set_volume_slice(2, update_slider=False)
        assert vview.slice_index == 2
        assert vview.slider.val == 0

    def test_slider_events(self, volume):
        """Slider callbacks should be called when events are not suppressed."""
        vview = kplt.volview(volume)
        record = []
        vview.slider.on_changed(record.append)
        vview.set_volume_slice(1)
        assert record == [1]

    def test_slider_events_suppressed(self, volume):
        """Slider callbacks should not be called when events are suppressed."""
        vview = kplt.volview(volume)
        record = []
        vview.slider.on_changed(record.append)
        vview.set_volume_slice(1, suppress_events=True)
        assert record == []
        assert vview.slider.val == 1
        assert vview.slider.eventson

    def test_toolbar_message(self, volume, toolbar_axes):
        """Setting the slice index should generate a toolbar message."""
        _, ax, toolbar = toolbar_axes
        vview = kplt.volview(volume, ax=ax)
        vview.set_volume_slice(2)
        assert toolbar.messages[-1] == f"Slice index 2 of {volume.shape[0]}"
