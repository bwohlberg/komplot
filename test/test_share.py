# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._share."""

import gc

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import press_key, scroll_in_axes

import komplot as kplt
from komplot import ShareGroup, share_cmap, share_slice, unshare_cmap, unshare_slice


@pytest.fixture
def image_views(image):
    """Three image views with colorbars in a single figure."""
    _, ax = plt.subplots(nrows=1, ncols=3)
    return [
        kplt.imview((n + 1.0) * image, show_cbar=True, ax=a) for n, a in enumerate(ax)
    ]


@pytest.fixture
def volume_views(volume):
    """Three volume views in a single figure."""
    _, ax = plt.subplots(nrows=1, ncols=3)
    return [kplt.volview((n + 1.0) * volume, ax=a) for n, a in enumerate(ax)]


def cmap_shift(view):
    """Get the colormap vmin shift of a view, relative to its own range."""
    return (view.axesimage.norm.vmin - view.vmin_ref) / (view.vmax_ref - view.vmin_ref)


class TestShareGroup:
    """Tests for class ShareGroup."""

    def test_attributes(self, image_views):
        """The group key and membership attribute should be as specified."""
        group = share_cmap(image_views)
        assert group.key == "cmap"
        assert group.attribute == "_cmap_group"

    def test_members(self, image_views):
        """Members should be recorded in the order in which they joined."""
        group = share_cmap(image_views)
        assert group.members() == image_views
        assert list(group) == image_views
        assert len(group) == 3
        assert all(view in group for view in image_views)

    def test_add_existing_member(self, image_views):
        """Adding an existing member should not duplicate it."""
        group = share_cmap(image_views)
        group.add(image_views[0])
        assert group.members() == image_views

    def test_add_new_member(self, image, image_views):
        """A plot may be added to an existing group."""
        group = share_cmap(image_views[:2])
        group.add(image_views[2])
        assert group.members() == image_views
        assert image_views[2].cmap_group is group

    def test_remove_member(self, image_views):
        """A removed member should no longer belong to the group."""
        group = share_cmap(image_views)
        group.remove(image_views[1])
        assert group.members() == [image_views[0], image_views[2]]
        assert image_views[1].cmap_group is None

    def test_remove_dissolves_group(self, image_views):
        """A group reduced below two members should be dissolved."""
        group = share_cmap(image_views[:2])
        group.remove(image_views[0])
        assert len(group) == 0
        assert all(view.cmap_group is None for view in image_views[:2])

    def test_dissolve(self, image_views):
        """A dissolved group should have no members."""
        group = share_cmap(image_views)
        group.dissolve()
        assert len(group) == 0
        assert all(view.cmap_group is None for view in image_views)

    def test_broadcast(self, image_views):
        """A broadcast should be applied to every member."""
        group = share_cmap(image_views)
        group.broadcast("apply_cmap_vmin", 0.25)
        assert all(cmap_shift(view) == pytest.approx(0.25) for view in image_views)

    def test_broadcast_reentrancy(self, image_views):
        """A broadcast triggered during a broadcast should be discarded."""
        group = share_cmap(image_views)
        calls = []

        def recursive_shift(self, rel_delta):
            calls.append(self)
            group.broadcast("shift", rel_delta)

        for view in image_views:
            view.shift = recursive_shift.__get__(view)
        group.broadcast("shift", 0.1)
        assert calls == image_views

    def test_weak_membership(self, volume):
        """Membership should not prevent garbage collection of a plot."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        views = [kplt.volview(volume, ax=a) for a in ax]
        group = share_slice(views)
        assert len(group) == 2
        kplt.close(fig)
        del views, ax, fig
        gc.collect()
        assert len(group) == 0


class TestShareCmap:
    """Tests for function share_cmap."""

    def test_group_membership(self, image_views):
        """Every specified plot should belong to the returned group."""
        group = share_cmap(image_views)
        assert isinstance(group, ShareGroup)
        assert all(view.cmap_group is group for view in image_views)

    def test_share_array_of_plots(self, image):
        """An array of plots should be accepted."""
        _, ax = plt.subplots(nrows=2, ncols=2)
        views = np.array([kplt.imview(image, ax=a) for a in ax.ravel()]).reshape(2, 2)
        assert len(share_cmap(views)) == 4

    def test_share_group(self, image_views):
        """An existing group should be accepted as a specification."""
        group = share_cmap(image_views)
        assert share_cmap(group).members() == image_views

    def test_shift_broadcast(self, image_views):
        """A colormap shift should be applied to every member."""
        share_cmap(image_views)
        image_views[1].shift_cmap_vmin(0.25)
        assert all(cmap_shift(view) == pytest.approx(0.25) for view in image_views)

    def test_colorbar_scroll_broadcast(self, image_views):
        """A colorbar scroll should be applied to every member."""
        share_cmap(image_views)
        scroll_in_axes(image_views[2].cbar_axes, 0.5, 0.25, button="up")
        assert all(cmap_shift(view) == pytest.approx(0.02) for view in image_views)

    def test_unshared_plot_unaffected(self, image_views):
        """A plot outside the group should not be affected by a shift."""
        share_cmap(image_views[:2])
        image_views[0].shift_cmap_vmin(0.25)
        assert cmap_shift(image_views[1]) == pytest.approx(0.25)
        assert cmap_shift(image_views[2]) == 0.0

    def test_regroup(self, image_views):
        """A new group should take members from the group they belonged to."""
        group0 = share_cmap(image_views)
        group1 = share_cmap(image_views[1:])
        assert group0.members() == [image_views[0]] or len(group0) == 0
        assert group1.members() == image_views[1:]
        assert image_views[0].cmap_group is None

    def test_regroup_retains_larger_group(self, image, image_views):
        """A group retaining two members should not be dissolved."""
        _, ax = plt.subplots()
        extra = kplt.imview(image, show_cbar=True, ax=ax)
        group0 = share_cmap(image_views)
        group1 = share_cmap([image_views[2], extra])
        assert group0.members() == image_views[:2]
        assert group1.members() == [image_views[2], extra]

    def test_duplicate_plots(self, image_views):
        """Repeated specification of a plot should be ignored."""
        group = share_cmap([image_views[0], image_views[1], image_views[0]])
        assert group.members() == image_views[:2]

    def test_broadcast_autoscaled_integer_limits(self, image):
        """A broadcast should be supported for integer colormap limits.

        Matplotlib records autoscaled colormap limits of an array of
        integer type as integers rather than floats.
        """
        _, ax = plt.subplots(nrows=1, ncols=2)
        views = [
            kplt.ImageView((16.0 * image).astype("uint8"), show_cbar=True, ax=a)
            for a in ax
        ]
        assert not any(isinstance(v.axesimage.norm.vmin, float) for v in views)
        share_cmap(views)
        scroll_in_axes(views[0].cbar_axes, 0.5, 0.25, button="up")
        scroll_in_axes(views[0].cbar_axes, 0.5, 0.75, button="down")
        for view in views:
            assert view.cmap_limits == pytest.approx(
                (
                    view.vmin_ref + 0.02 * (view.vmax_ref - view.vmin_ref),
                    view.vmax_ref - 0.02 * (view.vmax_ref - view.vmin_ref),
                )
            )

    def test_cross_figure_share(self, image):
        """Plots in different figures should be shareable."""
        view0 = kplt.imview(image, show_cbar=True)
        view1 = kplt.imview(2.0 * image, show_cbar=True)
        assert view0.figure is not view1.figure
        share_cmap([view0, view1])
        scroll_in_axes(view0.cbar_axes, 0.5, 0.25, button="up")
        assert cmap_shift(view1) == pytest.approx(0.02)

    @pytest.mark.parametrize("plots", [[], "single"])
    def test_too_few_plots(self, image_views, plots):
        """Fewer than two distinct plots should raise an exception."""
        if plots == "single":
            plots = [image_views[0], image_views[0]]
        with pytest.raises(ValueError, match="at least two distinct plots"):
            share_cmap(plots)

    def test_unsupported_plot_type(self, image_views):
        """A plot not supporting a colorbar should raise an exception."""
        lplot = kplt.plot(np.arange(5))
        with pytest.raises(ValueError, match="supporting a colorbar"):
            share_cmap([image_views[0], lplot])
        assert image_views[0].cmap_group is None


class TestShareSlice:
    """Tests for function share_slice."""

    def test_group_membership(self, volume_views):
        """Every specified plot should belong to the returned group."""
        group = share_slice(volume_views)
        assert isinstance(group, ShareGroup)
        assert group.key == "slice"
        assert all(view.slice_group is group for view in volume_views)

    def test_initial_synchronization(self, volume_views):
        """Members should be set to the slice index of the first member."""
        volume_views[0].slice_index = 2
        volume_views[1].slice_index = 4
        share_slice(volume_views)
        assert [view.slice_index for view in volume_views] == [2, 2, 2]
        assert [view.slider.val for view in volume_views] == [2, 2, 2]

    def test_set_slice_broadcast(self, volume_views, volume):
        """A slice index change should be applied to every member."""
        share_slice(volume_views)
        volume_views[1].slice_index = 3
        assert [view.slice_index for view in volume_views] == [3, 3, 3]
        for n, view in enumerate(volume_views):
            np.testing.assert_allclose(
                view.axesimage.get_array(), (n + 1.0) * volume[3]
            )

    def test_shift_scroll_broadcast(self, volume_views):
        """A shift scroll should change the slice index of every member."""
        share_slice(volume_views)
        press_key(volume_views[0].figure, "shift")
        scroll_in_axes(volume_views[2].axes, 0.5, 0.5, button="up")
        assert [view.slice_index for view in volume_views] == [1, 1, 1]

    def test_slider_broadcast(self, volume_views):
        """A slider change should change the slice index of every member."""
        share_slice(volume_views)
        volume_views[0].slider.set_val(3)
        assert [view.slice_index for view in volume_views] == [3, 3, 3]
        assert [view.slider.val for view in volume_views] == [3, 3, 3]

    def test_unequal_slice_counts(self, volume_views, volume):
        """Unequal numbers of slices should raise an exception."""
        other = kplt.volview(volume[0:-1])
        with pytest.raises(ValueError, match="equal numbers of slices"):
            share_slice([volume_views[0], other])
        assert volume_views[0].slice_group is None
        assert other.slice_group is None

    def test_unsupported_plot_type(self, volume_views, image):
        """A plot not supporting slice selection should raise an exception."""
        iview = kplt.imview(image)
        with pytest.raises(ValueError, match="supporting slice selection"):
            share_slice([volume_views[0], iview])

    def test_independent_of_cmap_share(self, volume_views):
        """Slice and colormap sharing should be independent."""
        slice_group = share_slice(volume_views)
        cmap_group = share_cmap(volume_views[:2])
        assert slice_group.members() == volume_views
        assert volume_views[2].slice_group is slice_group
        assert volume_views[2].cmap_group is None
        assert cmap_group.members() == volume_views[:2]


class TestUnshare:
    """Tests for functions unshare_cmap and unshare_slice."""

    def test_unshare_cmap_plot(self, image_views):
        """A plot should be removable from its colormap share group."""
        group = share_cmap(image_views)
        unshare_cmap(image_views[0])
        assert image_views[0].cmap_group is None
        assert group.members() == image_views[1:]
        image_views[1].shift_cmap_vmin(0.25)
        assert cmap_shift(image_views[0]) == 0.0
        assert cmap_shift(image_views[2]) == pytest.approx(0.25)

    def test_unshare_cmap_group(self, image_views):
        """A group should be removable by passing the group itself."""
        group = share_cmap(image_views)
        unshare_cmap(group)
        assert len(group) == 0
        assert all(view.cmap_group is None for view in image_views)

    def test_unshare_cmap_list(self, image_views):
        """A list of plots should be removable from their group."""
        group = share_cmap(image_views)
        unshare_cmap(image_views[:2])
        assert len(group) == 0
        assert image_views[2].cmap_group is None

    def test_unshare_cmap_unshared_plot(self, image_views):
        """Removing an unshared plot from a group should have no effect."""
        unshare_cmap(image_views[0])
        assert image_views[0].cmap_group is None

    def test_unshare_slice_plot(self, volume_views):
        """A plot should be removable from its slice share group."""
        group = share_slice(volume_views)
        unshare_slice(volume_views[0])
        assert volume_views[0].slice_group is None
        assert group.members() == volume_views[1:]
        volume_views[1].slice_index = 3
        assert [view.slice_index for view in volume_views] == [0, 3, 3]

    def test_unshare_slice_group(self, volume_views):
        """A group should be removable by passing the group itself."""
        group = share_slice(volume_views)
        unshare_slice(group)
        assert len(group) == 0
        assert all(view.slice_group is None for view in volume_views)

    def test_unshare_slice_unshared_plot(self, volume_views):
        """Removing an unshared plot from a group should have no effect."""
        unshare_slice(volume_views[0])
        assert volume_views[0].slice_group is None
