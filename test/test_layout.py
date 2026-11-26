# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._layout."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox

import komplot as kplt
from komplot._layout import (
    DEFAULT_PAD,
    SHARED_PAD,
    claim_reserved_axes,
    reserved_axes,
    reserved_orientation,
)


def cell(fig, axs):
    """Get the extent of the gridspec cell of an axes, in pixels."""
    box = axs.get_subplotspec().get_position(fig)
    return box.width * fig.bbox.width, box.height * fig.bbox.height


def cells_equal(fig, axes):
    """Determine whether the gridspec cells of a set of axes are of equal size."""
    fig.canvas.draw()
    extents = [cell(fig, axs) for axs in np.ravel(axes)]
    return all(extent == pytest.approx(extents[0], abs=1e-6) for extent in extents[1:])


class TestDelegation:
    """Tests for delegation to matplotlib when no space is reserved."""

    def test_default(self):
        """A grid of axes should be created as by matplotlib itself."""
        fig, ax = kplt.subplots(nrows=2, ncols=3)
        assert ax.shape == (2, 3)
        assert all(axs.get_subplotspec() is not None for axs in np.ravel(ax))
        assert all(reserved_axes(axs) is None for axs in np.ravel(ax))

    def test_squeeze(self):
        """Axes of unit length should be removed unless requested otherwise."""
        assert isinstance(kplt.subplots()[1], Axes)
        assert kplt.subplots(nrows=1, ncols=3)[1].shape == (3,)
        assert kplt.subplots(nrows=1, ncols=1, squeeze=False)[1].shape == (1, 1)

    def test_squeeze_with_reservation(self):
        """Reservation should not change the shape of the array of axes."""
        assert isinstance(kplt.subplots(cbar=True)[1], Axes)
        assert kplt.subplots(nrows=1, ncols=3, cbar=True)[1].shape == (3,)
        assert kplt.subplots(nrows=2, ncols=2, cbar=True)[1].shape == (2, 2)
        assert kplt.subplots(nrows=1, ncols=1, cbar=True, squeeze=False)[1].shape == (
            1,
            1,
        )

    @pytest.mark.parametrize("share", [True, False, "all", "row", "col", "none"])
    def test_sharing_matches_matplotlib(self, share):
        """Axis sharing should behave as it does in matplotlib itself."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar=True, sharex=share, sharey=share)
        figref, axref = plt.subplots(nrows=2, ncols=2, sharex=share, sharey=share)
        for axs, axsref in zip(np.ravel(ax), np.ravel(axref)):
            assert (
                axs.xaxis.get_tick_params()["labelbottom"]
                == axsref.xaxis.get_tick_params()["labelbottom"]
            )
            assert (
                axs.yaxis.get_tick_params()["labelleft"]
                == axsref.yaxis.get_tick_params()["labelleft"]
            )
            assert axs.get_shared_x_axes().joined(
                axs, np.ravel(ax)[0]
            ) == axsref.get_shared_x_axes().joined(axsref, np.ravel(axref)[0])


class TestReservation:
    """Tests for the reservation of space for auxiliary axes."""

    @pytest.mark.parametrize("name", ["cax", "sax"])
    def test_reserved_for_every_plot(self, name):
        """Space may be reserved for every plot of a grid."""
        fig, ax = kplt.subplots(
            nrows=2,
            ncols=2,
            **{name.replace("cax", "cbar").replace("sax", "slider"): True},
        )
        assert all(reserved_axes(axs, name) is not None for axs in np.ravel(ax))
        assert cells_equal(fig, ax)

    def test_reserved_per_cell(self):
        """Space may be reserved for individual plots of a grid."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar=[[False, True], [False, True]])
        reserved = [
            [reserved_axes(ax[r, c]) is not None for c in range(2)] for r in range(2)
        ]
        assert reserved == [[False, True], [False, True]]
        assert cells_equal(fig, ax)

    def test_reserved_for_a_column(self):
        """Space is reserved for a column of plots if any plot requires it."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar=[[False, True], [False, False]])
        assert reserved_axes(ax[0, 1]) is not None
        # The reservation is per column, but an axes is created only where required
        assert reserved_axes(ax[1, 1]) is None
        assert cells_equal(fig, ax)

    def test_reserved_axes_hidden(self):
        """A reserved axes should not be displayed until a plot draws in it."""
        fig, ax = kplt.subplots(ncols=2, cbar=True)
        assert not reserved_axes(ax[0]).get_visible()

    def test_unused_reservation_preserves_alignment(self, image):
        """Plots should be aligned whether or not they use reserved space."""
        fig, ax = kplt.subplots(ncols=2, cbar=True)
        kplt.imview(image, ax=ax[0])
        kplt.imview(image, show_cbar=False, ax=ax[1])
        assert cells_equal(fig, ax)

    @pytest.mark.parametrize(
        "location,expected", [("right", "vertical"), ("bottom", "horizontal")]
    )
    def test_location(self, location, expected):
        """A reserved axes should be placed on the specified side of the plot."""
        fig, ax = kplt.subplots(ncols=2, cbar=True, cbar_location=location)
        fig.canvas.draw()
        box = ax[0].get_window_extent()
        cbox = reserved_axes(ax[0]).get_window_extent()
        if location == "right":
            assert cbox.x0 > box.x1
            assert cbox.height == pytest.approx(box.height)
        else:
            assert cbox.y1 < box.y0
            assert cbox.width == pytest.approx(box.width)
        assert reserved_orientation(reserved_axes(ax[0])) == expected
        assert cells_equal(fig, ax)

    def test_size_and_pad(self):
        """The size of a reserved axes should be as specified."""
        fig, ax = kplt.subplots(ncols=2, cbar=True, cbar_size=0.2, cbar_pad=0.1)
        fig.canvas.draw()
        box = ax[0].get_window_extent()
        cbox = reserved_axes(ax[0]).get_window_extent()
        assert cbox.width == pytest.approx(0.2 * box.width)
        assert cbox.x0 - box.x1 == pytest.approx(0.1 * box.width)

    def test_colorbar_and_slider(self):
        """Space may be reserved for a colorbar and a slider at once."""
        fig, ax = kplt.subplots(ncols=2, cbar=True, slider=[True, False])
        assert reserved_axes(ax[0], "sax") is not None
        assert reserved_axes(ax[1], "sax") is None
        assert all(reserved_axes(axs, "cax") is not None for axs in ax)
        assert cells_equal(fig, ax)

    def test_ratios_and_spacing(self):
        """Size ratios and spacing of the plots should be as specified."""
        fig, ax = kplt.subplots(ncols=2, cbar=[False, True], width_ratios=[1, 2])
        fig.canvas.draw()
        box = [axs.get_window_extent() for axs in ax]
        assert box[1].width == pytest.approx(2.0 * box[0].width)
        fig, ax = kplt.subplots(
            ncols=2, cbar=[False, True], gridspec_kw={"wspace": 0.0}
        )
        fig.canvas.draw()
        box = [axs.get_window_extent() for axs in ax]
        assert box[1].x0 == pytest.approx(box[0].x1)

    @pytest.mark.parametrize("size", [(10, 4), (4, 4), (12, 9), (3, 8)])
    def test_resize(self, image, size):
        """Plots should remain aligned when the figure is resized."""
        fig, ax = kplt.subplots(ncols=2, cbar=[False, True], figsize=(10, 4))
        kplt.imview(image, ax=ax[0])
        kplt.imview(image, ax=ax[1])
        fig.set_size_inches(*size, forward=True)
        assert cells_equal(fig, ax)
        box = [axs.get_window_extent() for axs in ax]
        assert box[0].width == pytest.approx(box[1].width)
        assert box[0].height == pytest.approx(box[1].height)

    @pytest.mark.parametrize("layout", ["tight", "constrained"])
    @pytest.mark.filterwarnings("ignore:This figure includes Axes")
    def test_layout_engine(self, image, layout):
        """Plots should remain aligned under a matplotlib layout engine."""
        fig, ax = kplt.subplots(ncols=2, cbar=[False, True], layout=layout)
        kplt.imview(image, ax=ax[0])
        kplt.imview(image, ax=ax[1])
        fig.canvas.draw()
        box = [axs.get_window_extent() for axs in ax]
        assert box[0].width == pytest.approx(box[1].width)
        assert box[0].height == pytest.approx(box[1].height)

    def test_tight_layout_declines(self, image):
        """A tight layout should decline to adjust a reserved layout.

        The spacing of a reserved layout is specified by the gridspec
        itself, which matplotlib considers incompatible with a tight
        layout, warning that it has been left unchanged.
        """
        fig, ax = kplt.subplots(ncols=2, cbar=[False, True], layout="tight")
        kplt.imview(image, ax=ax[0])
        kplt.imview(image, ax=ax[1])
        with pytest.warns(UserWarning, match="not compatible with tight_layout"):
            fig.canvas.draw()


class TestSharedReservation:
    """Tests for the reservation of space shared by multiple plots."""

    def test_shared(self):
        """A single axes may be shared by all of the plots of a grid."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar="shared")
        cax = reserved_axes(ax[0, 0])
        assert all(reserved_axes(axs) is cax for axs in np.ravel(ax))
        assert cells_equal(fig, ax)
        fig.canvas.draw()
        cbox = cax.get_window_extent()
        top, bottom = ax[0, 0].get_window_extent(), ax[1, 0].get_window_extent()
        assert cbox.y1 == pytest.approx(top.y1)
        assert cbox.y0 == pytest.approx(bottom.y0)
        assert cbox.x0 > ax[0, 1].get_window_extent().x1

    def test_shared_bottom(self):
        """A shared axes may be placed below the plots of a grid."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar="shared", cbar_location="bottom")
        cax = reserved_axes(ax[0, 0])
        fig.canvas.draw()
        cbox = cax.get_window_extent()
        assert cbox.y1 < ax[1, 0].get_window_extent().y0
        assert cbox.x0 == pytest.approx(ax[0, 0].get_window_extent().x0)
        assert cbox.x1 == pytest.approx(ax[0, 1].get_window_extent().x1)
        assert cells_equal(fig, ax)

    def test_shared_per_row(self):
        """A single axes may be shared by each row of plots of a grid."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar="row")
        assert reserved_axes(ax[0, 0]) is reserved_axes(ax[0, 1])
        assert reserved_axes(ax[1, 0]) is reserved_axes(ax[1, 1])
        assert reserved_axes(ax[0, 0]) is not reserved_axes(ax[1, 0])
        assert cells_equal(fig, ax)

    def test_shared_per_column(self):
        """A single axes may be shared by each column of plots of a grid."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar="col", cbar_location="bottom")
        assert reserved_axes(ax[0, 0]) is reserved_axes(ax[1, 0])
        assert reserved_axes(ax[0, 1]) is reserved_axes(ax[1, 1])
        assert reserved_axes(ax[0, 0]) is not reserved_axes(ax[0, 1])
        assert cells_equal(fig, ax)

    def test_claim(self, image):
        """A reserved axes should be claimed by the first plot to use it."""
        fig, ax = kplt.subplots(ncols=2, cbar="shared")
        reserved, owner = claim_reserved_axes(ax[0])
        assert reserved is ax[0].cax and owner is None
        iview = kplt.imview(image, ax=ax[0])
        reserved, owner = claim_reserved_axes(ax[1])
        assert reserved is None and owner is iview

    def test_claim_without_reservation(self):
        """An axes with no reserved space should have nothing to claim."""
        fig, ax = kplt.subplots(ncols=2)
        assert claim_reserved_axes(ax[0]) == (None, None)


class TestErrors:
    """Tests for the detection of invalid specifications."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"cbar": "every"}, "must be a bool"),
            ({"cbar": "row", "cbar_location": "bottom"}, '"row" requires'),
            ({"cbar": "col"}, '"col" requires'),
            ({"cbar": [True, False, True]}, "cannot be matched"),
            ({"cbar": True, "cbar_location": "top"}, "must be either"),
            ({"cbar": True, "cbar_size": 0.0}, "must be positive"),
            ({"cbar": True, "cbar_pad": -0.1}, "non-negative"),
            ({"cbar": True, "width_ratios": [1, 2, 3]}, "do not match grid shape"),
            ({"cbar": True, "sharex": "both"}, "sharing specification"),
            ({"slider": "every"}, "must be a bool"),
        ],
    )
    def test_invalid_specification(self, kwargs, match):
        """An invalid specification should raise an exception."""
        with pytest.raises(ValueError, match=match):
            kplt.subplots(nrows=1, ncols=2, **kwargs)

    def test_invalid_reserved_name(self):
        """An invalid reserved axes name should raise an exception."""
        fig, ax = kplt.subplots()
        with pytest.raises(ValueError, match="Reserved axes name"):
            reserved_axes(ax, "zax")


class TestAxesReplacement:
    """Tests for reservation when an axes is replaced by a 3D axes."""

    def test_reservation_preserved(self, surface_data):
        """Reserved space should survive replacement of the axes by a 3D axes."""
        fig, ax = kplt.subplots(ncols=2, cbar=True)
        splot = kplt.surface(surface_data, ax=ax[0])
        assert splot.axes is not ax[0]
        assert reserved_axes(splot.axes) is reserved_axes(ax[0])

    def test_alignment_preserved(self, surface_data, image):
        """A 3D axes should remain aligned with the other plots of a figure."""
        fig, ax = kplt.subplots(ncols=2, cbar=[False, True])
        splot = kplt.surface(surface_data, ax=ax[0])
        kplt.imview(image, ax=ax[1])
        assert cells_equal(fig, [splot.axes, ax[1]])


class TestReservedExtent:
    """Tests for the extent of a reserved axes.

    The axes of a plot with a fixed aspect ratio, such as an image, is
    smaller than the cell of the gridspec within which it is placed, so
    that a reserved axes is positioned from the extent of the plots that
    it serves rather than from that cell.
    """

    @staticmethod
    def extent(plots):
        """Get the union of the extents of the axes of a set of plots."""
        return Bbox.union([plot.axes.get_window_extent() for plot in plots])

    def test_shared_matches_tallest(self, image, wide_image):
        """A shared colorbar should match the extent of the tallest plot."""
        fig, ax = kplt.subplots(ncols=3, cbar="shared", figsize=(12, 4))
        norm = Normalize(vmin=0.0, vmax=48.0)
        plots = [
            kplt.imview(data, norm=norm, ax=axs)
            for data, axs in zip((image, wide_image, image), ax)
        ]
        fig.canvas.draw()
        tallest = max(
            (plot.axes.get_window_extent() for plot in plots),
            key=lambda box: box.height,
        )
        cbox = ax[0].cax.get_window_extent()
        assert cbox.y0 == pytest.approx(tallest.y0)
        assert cbox.y1 == pytest.approx(tallest.y1)

    def test_shared_matches_only_wide_plots(self, wide_image):
        """A shared colorbar should not exceed the plots that it serves."""
        fig, ax = kplt.subplots(ncols=3, cbar="shared", figsize=(12, 4))
        norm = Normalize(vmin=0.0, vmax=48.0)
        plots = [kplt.imview(wide_image, norm=norm, ax=axs) for axs in ax]
        fig.canvas.draw()
        extent = self.extent(plots)
        cbox = ax[0].cax.get_window_extent()
        assert cbox.height == pytest.approx(extent.height)
        # The extent of the plots is well short of that of their cells
        assert extent.height < 0.9 * cell(fig, ax[0])[1]

    def test_shared_spanning_rows(self, image, wide_image):
        """A shared colorbar should span from the topmost to the bottommost plot."""
        fig, ax = kplt.subplots(nrows=2, ncols=2, cbar="shared", figsize=(8, 8))
        norm = Normalize(vmin=0.0, vmax=48.0)
        data = (image, wide_image, wide_image, image)
        plots = [
            kplt.imview(datum, norm=norm, ax=axs)
            for datum, axs in zip(data, np.ravel(ax))
        ]
        fig.canvas.draw()
        extent = self.extent(plots)
        cbox = ax[0, 0].cax.get_window_extent()
        assert cbox.y1 == pytest.approx(extent.y1)
        assert cbox.y0 == pytest.approx(extent.y0)

    def test_shared_below_column(self, image, wide_image):
        """A shared colorbar below a column should match the widest plot."""
        fig, ax = kplt.subplots(
            nrows=2, ncols=2, cbar="col", cbar_location="bottom", figsize=(8, 8)
        )
        norm = Normalize(vmin=0.0, vmax=48.0)
        plots = [
            kplt.imview(data, norm=norm, ax=axs)
            for data, axs in zip((image, wide_image), ax[:, 0])
        ]
        fig.canvas.draw()
        extent = self.extent(plots)
        cbox = ax[0, 0].cax.get_window_extent()
        assert cbox.x0 == pytest.approx(extent.x0)
        assert cbox.x1 == pytest.approx(extent.x1)

    def test_per_plot_matches_its_plot(self, wide_image):
        """A colorbar should match the extent of the plot that it represents."""
        fig, ax = kplt.subplots(ncols=2, cbar=True, cbar_pad=0.05, figsize=(12, 4))
        plot = kplt.imview(wide_image, ax=ax[0])
        kplt.imview(wide_image, ax=ax[1])
        fig.canvas.draw()
        box = plot.axes.get_window_extent()
        cbox = plot.cbar_axes.get_window_extent()
        assert cbox.y0 == pytest.approx(box.y0)
        assert cbox.y1 == pytest.approx(box.y1)
        # It is adjacent to the plot, rather than to the cell of the plot
        assert cbox.x0 - box.x1 == pytest.approx(0.05 * cell(fig, ax[0])[0])

    def test_slider_matches_its_plot(self, volume):
        """A slider should match the extent of the slice that it selects."""
        fig, ax = kplt.subplots(ncols=2, slider=True, figsize=(8, 6))
        vview = kplt.volview(volume, ax=ax[0])
        fig.canvas.draw()
        box = vview.axes.get_window_extent()
        sbox = vview.slider_axes.get_window_extent()
        assert sbox.x0 == pytest.approx(box.x0)
        assert sbox.x1 == pytest.approx(box.x1)

    @pytest.mark.parametrize("size", [(12, 4), (6, 4), (4, 9)])
    def test_extent_tracked_on_resize(self, wide_image, size):
        """A reserved axes should track its plots when the figure is resized."""
        fig, ax = kplt.subplots(ncols=2, cbar="shared", figsize=(12, 4))
        norm = Normalize(vmin=0.0, vmax=48.0)
        plots = [kplt.imview(wide_image, norm=norm, ax=axs) for axs in ax]
        fig.set_size_inches(*size, forward=True)
        fig.canvas.draw()
        extent = self.extent(plots)
        cbox = ax[0].cax.get_window_extent()
        assert cbox.y0 == pytest.approx(extent.y0)
        assert cbox.y1 == pytest.approx(extent.y1)

    def test_cell_used_without_plots(self):
        """A reserved axes should use its cell when it serves no plot."""
        fig, ax = kplt.subplots(ncols=2, cbar=True, figsize=(8, 4))
        fig.canvas.draw()
        box = ax[0].cax.get_window_extent()
        expected = ax[0].cax.get_subplotspec().get_position(fig)
        assert box.y0 == pytest.approx(expected.y0 * fig.bbox.height)
        assert box.y1 == pytest.approx(expected.y1 * fig.bbox.height)

    def test_cell_used_when_plot_replaced(self, surface_data):
        """A reserved axes should use its cell when its plot has been replaced."""
        fig, ax = kplt.subplots(ncols=2, cbar=True, figsize=(8, 4))
        reserved = ax[0].cax
        kplt.surface(surface_data, ax=ax[0])
        fig.canvas.draw()
        box = reserved.get_window_extent()
        expected = reserved.get_subplotspec().get_position(fig)
        assert box.y1 == pytest.approx(expected.y1 * fig.bbox.height)

    def test_locator_subplotspec(self):
        """The locator of a reserved axes should report its cell."""
        fig, ax = kplt.subplots(ncols=2, cbar=True)
        locator = ax[0].cax.get_axes_locator()
        assert locator.get_subplotspec() is ax[0].cax.get_subplotspec()


class TestReservedPad:
    """Tests for the space between a plot and an axes reserved for it."""

    def test_default_pad(self, image):
        """A colorbar representing one plot should be at the default distance."""
        fig, ax = kplt.subplots(ncols=2, cbar=True, figsize=(8, 4))
        plot = kplt.imview(image, ax=ax[0])
        fig.canvas.draw()
        box = plot.axes.get_window_extent()
        cbox = plot.cbar_axes.get_window_extent()
        assert cbox.x0 - box.x1 == pytest.approx(DEFAULT_PAD * cell(fig, ax[0])[0])

    @pytest.mark.parametrize("mode", ["shared", "row"])
    def test_shared_pad(self, image, mode):
        """A shared colorbar should be at a greater distance than a single one."""
        assert SHARED_PAD > DEFAULT_PAD
        fig, ax = kplt.subplots(ncols=2, cbar=mode, figsize=(8, 4))
        plots = [kplt.imview(image, ax=axs) for axs in ax]
        fig.canvas.draw()
        boxes = [plot.axes.get_window_extent() for plot in plots]
        cbox = ax[0].cax.get_window_extent()
        assert cbox.x0 - max(box.x1 for box in boxes) == pytest.approx(
            SHARED_PAD * cell(fig, ax[0])[0]
        )

    def test_explicit_pad(self, image):
        """A specified pad should be used whether or not the axes is shared."""
        for mode in (True, "shared"):
            fig, ax = kplt.subplots(ncols=2, cbar=mode, cbar_pad=0.01, figsize=(8, 4))
            plots = [kplt.imview(image, ax=axs) for axs in ax]
            fig.canvas.draw()
            boxes = [plot.axes.get_window_extent() for plot in plots]
            cbox = reserved_axes(ax[1]).get_window_extent()
            assert cbox.x0 - max(box.x1 for box in boxes) == pytest.approx(
                0.01 * cell(fig, ax[0])[0]
            )

    def test_slider_pad(self, volume):
        """A slider should be at the default distance below its plot."""
        fig, ax = kplt.subplots(ncols=2, slider=True, figsize=(8, 6))
        vview = kplt.volview(volume, ax=ax[0])
        fig.canvas.draw()
        box = vview.axes.get_window_extent()
        sbox = vview.slider_axes.get_window_extent()
        assert box.y0 - sbox.y1 == pytest.approx(DEFAULT_PAD * cell(fig, ax[0])[1])
