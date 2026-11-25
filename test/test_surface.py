# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._surface."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import komplot as kplt
from komplot import FigureEventManager, SurfacePlot, figure_event_manager


def assert_limits_cover(lim, lo: float, hi: float, margin: float = 0.1):
    """Assert that axis limits cover a data range without much excess.

    A 3d axes extends its axis limits slightly beyond the data range, so
    axis limits cannot be compared directly with the data range.
    """
    span = hi - lo
    assert lim[0] <= lo <= hi <= lim[1]
    assert lim[0] >= lo - margin * span
    assert lim[1] <= hi + margin * span


class TestSurface:
    """Tests for function surface."""

    def test_return_type(self, surface_data):
        """The returned object should be a fully initialized SurfacePlot."""
        splot = kplt.surface(surface_data)
        assert isinstance(splot, SurfacePlot)
        assert splot.figure is splot.axes.get_figure()
        assert isinstance(splot.poly3dc, Poly3DCollection)

    def test_axes_projection(self, surface_data):
        """The plot axes should be 3d."""
        assert kplt.surface(surface_data).axes.name == "3d"

    def test_default_axis_values(self, surface_data):
        """Axis ranges should default to the array indices."""
        splot = kplt.surface(surface_data)
        assert_limits_cover(splot.axes.get_xlim(), 0.0, surface_data.shape[1] - 1.0)
        assert_limits_cover(splot.axes.get_ylim(), 0.0, surface_data.shape[0] - 1.0)

    def test_specified_axis_values(self, surface_data):
        """Axis ranges should be as specified."""
        x = np.linspace(2.0, 3.0, surface_data.shape[1])
        y = np.linspace(5.0, 6.0, surface_data.shape[0])
        splot = kplt.surface(surface_data, x=x, y=y)
        assert_limits_cover(splot.axes.get_xlim(), 2.0, 3.0)
        assert_limits_cover(splot.axes.get_ylim(), 5.0, 6.0)

    def test_no_contours(self, surface_data):
        """No contours should be plotted by default."""
        assert kplt.surface(surface_data).qcntset is None

    def test_contours(self, surface_data):
        """Contours should be plotted when levels are specified."""
        splot = kplt.surface(surface_data, levels=[-0.5, 0.0, 0.5])
        assert splot.qcntset is not None
        np.testing.assert_allclose(splot.qcntset.levels, [-0.5, 0.0, 0.5])

    def test_contours_offset_below_surface(self, surface_data):
        """Contours should be plotted below the surface."""
        splot = kplt.surface(surface_data, levels=3)
        zrange = surface_data.max() - surface_data.min()
        offset = np.around(surface_data.min() - 0.2 * zrange, 3)
        assert splot.axes.get_zlim()[0] == pytest.approx(offset)

    def test_view_angles(self, surface_data):
        """View angles should be as specified."""
        splot = kplt.surface(surface_data, elev=25.0, azim=35.0, roll=5.0)
        assert splot.axes.elev == 25.0
        assert splot.axes.azim == 35.0
        assert splot.axes.roll == 5.0

    def test_default_colormap(self, surface_data):
        """The default colormap should be YlOrRd."""
        assert kplt.surface(surface_data).poly3dc.get_cmap().name == "YlOrRd"

    def test_specified_colormap(self, surface_data):
        """A specified colormap should be used."""
        splot = kplt.surface(surface_data, cmap="viridis")
        assert splot.poly3dc.get_cmap().name == "viridis"

    def test_alpha(self, surface_data):
        """Transparency should be as specified."""
        assert kplt.surface(surface_data, alpha=0.5).poly3dc.get_alpha() == 0.5

    def test_labels_and_title(self, surface_data):
        """Axis labels and title should be set when specified."""
        splot = kplt.surface(
            surface_data, xlabel="X", ylabel="Y", zlabel="Z", title="T"
        )
        assert splot.axes.get_xlabel() == "X"
        assert splot.axes.get_ylabel() == "Y"
        assert splot.axes.get_zlabel() == "Z"
        assert splot.axes.get_title() == "T"

    def test_data_format(self, surface_data):
        """Cursor coordinate display formats should be set."""
        splot = kplt.surface(surface_data)
        assert splot.axes.fmt_xdata(1.234) == " 1.23"
        assert splot.axes.fmt_ydata(1.234) == " 1.23"
        assert splot.axes.fmt_zdata(-1.235) == "-1.24"

    def test_figure_attributes(self, surface_data):
        """Figure size and number should be as specified."""
        splot = kplt.surface(surface_data, figsize=(5, 4), fignum=6)
        assert splot.figure.number == 6
        assert tuple(splot.figure.get_size_inches()) == (5.0, 4.0)

    def test_specified_2d_axes(self, surface_data):
        """A specified 2d axes should be replaced by a 3d axes."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        splot = kplt.surface(surface_data, ax=ax[1])
        assert splot.figure is fig
        assert splot.axes is not ax[1]
        assert splot.axes.name == "3d"
        assert plt.get_fignums() == [fig.number]

    def test_specified_3d_axes(self, surface_data):
        """A specified 3d axes should be used."""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        splot = kplt.surface(surface_data, ax=ax)
        assert splot.axes is ax
        assert splot.figure is fig

    def test_figure_event_manager(self, surface_data):
        """A figure event manager should be attached."""
        splot = kplt.surface(surface_data)
        assert isinstance(figure_event_manager(splot.figure), FigureEventManager)

    def test_no_axes_event_manager(self, surface_data):
        """No axes event manager should be attached."""
        splot = kplt.surface(surface_data)
        fem = figure_event_manager(splot.figure)
        assert fem.get_axevman_for_axes(splot.axes) is None

    def test_shared_figure_event_manager(self, surface_data):
        """A single figure event manager should serve multiple axes."""
        fig = plt.figure()
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        splot0 = kplt.surface(surface_data, ax=ax0)
        splot1 = kplt.surface(surface_data, ax=ax1)
        assert figure_event_manager(splot0.figure) is figure_event_manager(
            splot1.figure
        )
