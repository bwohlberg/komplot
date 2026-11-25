# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._plot."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import komplot as kplt
from komplot import LinePlot, ZoomMixin, _plot, figure_event_manager


@pytest.fixture
def xy():
    """Coordinates of points on a line."""
    x = np.linspace(1.0, 10.0, 16)
    return x, 2.0 * x + 1.0


class TestPlot:
    """Tests for function plot."""

    def test_return_type(self, xy):
        """The returned object should be a fully initialized LinePlot."""
        lplot = kplt.plot(*xy)
        assert isinstance(lplot, LinePlot)
        assert lplot.figure is lplot.axes.get_figure()
        assert len(lplot.line2d) == 1
        assert lplot.line2d[0] in lplot.axes.lines

    def test_data(self, xy):
        """The plotted data should be the specified data."""
        x, y = xy
        lplot = kplt.plot(x, y)
        np.testing.assert_allclose(lplot.line2d[0].get_xdata(), x)
        np.testing.assert_allclose(lplot.line2d[0].get_ydata(), y)

    def test_y_only(self):
        """A single data argument should be plotted against its index."""
        y = np.array([3.0, 1.0, 4.0, 1.0])
        lplot = kplt.plot(y)
        np.testing.assert_allclose(lplot.line2d[0].get_xdata(), np.arange(y.size))
        np.testing.assert_allclose(lplot.line2d[0].get_ydata(), y)

    def test_multiple_lines(self, xy):
        """Multiple lines should be supported, as by matplotlib plot."""
        x, y = xy
        lplot = kplt.plot(x, y, x, 2 * y)
        assert len(lplot.line2d) == 2

    def test_format_string(self, xy):
        """A matplotlib format string should be supported."""
        lplot = kplt.plot(*xy, "r:o")
        line = lplot.line2d[0]
        assert line.get_color() == "r"
        assert line.get_linestyle() == ":"
        assert line.get_marker() == "o"

    def test_line_property_defaults(self, xy):
        """Line width and marker size should have non-matplotlib defaults."""
        lplot = kplt.plot(*xy)
        assert lplot.line2d[0].get_linewidth() == 1.5
        assert lplot.line2d[0].get_markersize() == 6.0

    @pytest.mark.parametrize("lw_name", ["lw", "linewidth"])
    def test_line_width_override(self, xy, lw_name):
        """A specified line width should override the default."""
        lplot = kplt.plot(*xy, **{lw_name: 3.0})
        assert lplot.line2d[0].get_linewidth() == 3.0

    @pytest.mark.parametrize("ms_name", ["ms", "markersize"])
    def test_marker_size_override(self, xy, ms_name):
        """A specified marker size should override the default."""
        lplot = kplt.plot(*xy, **{ms_name: 2.0})
        assert lplot.line2d[0].get_markersize() == 2.0

    def test_labels_and_title(self, xy):
        """Axis labels and title should be set when specified."""
        lplot = kplt.plot(*xy, xlabel="X", ylabel="Y", title="T")
        assert lplot.axes.get_xlabel() == "X"
        assert lplot.axes.get_ylabel() == "Y"
        assert lplot.axes.get_title() == "T"

    def test_no_labels_or_title(self, xy):
        """Axis labels and title should be empty when not specified."""
        lplot = kplt.plot(*xy)
        assert lplot.axes.get_xlabel() == ""
        assert lplot.axes.get_ylabel() == ""
        assert lplot.axes.get_title() == ""

    def test_legend(self, xy):
        """A legend should be displayed when specified."""
        x, y = xy
        lplot = kplt.plot(x, y, x, 2 * y, legend=["a", "b"], legend_loc="lower right")
        legend = lplot.axes.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["a", "b"]

    def test_no_legend(self, xy):
        """No legend should be displayed when not specified."""
        assert kplt.plot(*xy, legend_loc="best").axes.get_legend() is None

    @pytest.mark.parametrize(
        "kwargs,xscale,yscale",
        [
            ({}, "linear", "linear"),
            ({"xlog": True}, "log", "linear"),
            ({"ylog": True}, "linear", "log"),
            ({"xlog": True, "ylog": True}, "log", "log"),
        ],
    )
    def test_log_scale(self, xy, kwargs, xscale, yscale):
        """Axis scales should be set as specified."""
        lplot = kplt.plot(*xy, **kwargs)
        assert lplot.axes.get_xscale() == xscale
        assert lplot.axes.get_yscale() == yscale

    @pytest.mark.parametrize(
        "kwargs,xscale,yscale",
        [
            ({"xlog": {}}, "log", "linear"),
            ({"ylog": {}}, "linear", "log"),
            ({"xlog": {"base": 2}}, "log", "linear"),
            ({"ylog": {"base": 2}}, "linear", "log"),
            ({"xlog": {}, "ylog": {}}, "log", "log"),
            ({"xlog": True, "ylog": {"base": 2}}, "log", "log"),
        ],
    )
    def test_log_scale_dict(self, xy, kwargs, xscale, yscale):
        """A dict log scale specification should imply a log scale."""
        lplot = kplt.plot(*xy, **kwargs)
        assert lplot.axes.get_xscale() == xscale
        assert lplot.axes.get_yscale() == yscale

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_log_scale_base(self, xy, axis):
        """A log scale base should be passed to the axis scale."""
        lplot = kplt.plot(*xy, **{axis + "log": {"base": 2}})
        axis_obj = getattr(lplot.axes, axis + "axis")
        assert axis_obj.get_transform().base == 2

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_log_scale_subs(self, xy, axis):
        """Log scale minor tick subdivisions should be passed to the scale."""
        lplot = kplt.plot(*xy, **{axis + "log": {"subs": [2, 3]}})
        axis_obj = getattr(lplot.axes, axis + "axis")
        np.testing.assert_allclose(axis_obj.get_minor_locator()._subs, [2, 3])

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_log_scale_nonpositive(self, xy, axis):
        """Nonpositive value handling should be passed to the axis scale."""
        x, y = xy
        lplot = kplt.plot(x - 5.0, y, **{axis + "log": {"nonpositive": "mask"}})
        assert getattr(lplot.axes, "get_" + axis + "scale")() == "log"

    def test_log_scale_distinct_axis_properties(self, xy):
        """Each log scale axis should have its own properties."""
        lplot = kplt.plot(*xy, xlog={"base": 10}, ylog={"base": 2})
        assert lplot.axes.xaxis.get_transform().base == 10
        assert lplot.axes.yaxis.get_transform().base == 2

    def test_log_scale_kwargs_not_passed_to_plot(self, xy):
        """Line properties should still be passed to Axes.plot."""
        lplot = kplt.plot(*xy, xlog={"base": 2}, color="g")
        assert lplot.axes.xaxis.get_transform().base == 2
        assert lplot.line2d[0].get_color() == "g"

    def test_log_scale_invalid_property(self, xy):
        """An invalid log scale property should raise an exception."""
        with pytest.raises(TypeError):
            kplt.plot(*xy, xlog={"basex": 2})

    def test_data_cursor(self, xy, monkeypatch):
        """A data cursor should be attached when mplcursors is available."""
        cursor_args: list = []
        monkeypatch.setattr(
            _plot, "mplcrs", SimpleNamespace(cursor=cursor_args.append), raising=False
        )
        monkeypatch.setattr(_plot, "HAVE_MPLCRS", True)
        lplot = kplt.plot(*xy)
        assert cursor_args == [lplot.line2d]

    def test_data_format(self, xy):
        """Cursor coordinate display formats should be set."""
        lplot = kplt.plot(*xy)
        assert lplot.axes.fmt_xdata(1.234) == " 1.23"
        assert lplot.axes.fmt_ydata(-1.235) == "-1.24"

    def test_figure_attributes(self, xy):
        """Figure size and number should be as specified."""
        lplot = kplt.plot(*xy, figsize=(5, 4), fignum=3)
        assert lplot.figure.number == 3
        assert tuple(lplot.figure.get_size_inches()) == (5.0, 4.0)

    def test_specified_axes(self, xy):
        """Plotting into a specified axes should not create a figure."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        lplot = kplt.plot(*xy, ax=ax[1])
        assert lplot.axes is ax[1]
        assert lplot.figure is fig
        assert plt.get_fignums() == [fig.number]

    def test_event_manager(self, xy):
        """The plot should be registered with the figure event manager."""
        lplot = kplt.plot(*xy)
        fem = figure_event_manager(lplot.figure)
        assert fem.fig is lplot.figure
        assert lplot.event_manager is fem
        assert fem.plot_for_axes(lplot.axes) is lplot

    def test_event_manager_shared_figure(self, xy):
        """A single figure event manager should serve multiple plots."""
        _, ax = plt.subplots(nrows=1, ncols=2)
        lplot0 = kplt.plot(*xy, ax=ax[0])
        lplot1 = kplt.plot(*xy, ax=ax[1])
        fem = figure_event_manager(lplot0.figure)
        assert fem is figure_event_manager(lplot1.figure)
        assert fem.plots == [lplot0, lplot1]

    def test_zoom_support(self, xy):
        """A line plot should support zoom."""
        lplot = kplt.plot(*xy)
        assert isinstance(lplot, ZoomMixin)
        assert lplot.xlim_ref == lplot.axes.get_xlim()

    def test_repeated_plot_in_axes(self, xy):
        """Repeated plots in the same axes should both be drawn."""
        x, y = xy
        lplot0 = kplt.plot(x, y)
        lplot1 = kplt.plot(x, 2 * y, ax=lplot0.axes)
        assert len(lplot1.axes.lines) == 2
        fem = figure_event_manager(lplot0.figure)
        assert fem.plots == [lplot0, lplot1]
        assert fem.plot_for_axes(lplot0.axes) is lplot0
