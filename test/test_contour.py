# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._contour."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

import komplot as kplt
from komplot import ContourPlot, figure_event_manager


class TestContour:
    """Tests for function contour."""

    def test_return_type(self, surface_data):
        """The returned object should be a fully initialized ContourPlot."""
        cplot = kplt.contour(surface_data)
        assert isinstance(cplot, ContourPlot)
        assert cplot.figure is cplot.axes.get_figure()
        assert cplot.qcntset is not None
        assert isinstance(cplot.cbar_axes, Axes)
        assert isinstance(cplot.divider, AxesDivider)

    def test_default_axis_values(self, surface_data):
        """Axis ranges should default to the array indices."""
        cplot = kplt.contour(surface_data)
        assert cplot.axes.get_xlim() == pytest.approx(
            (0.0, surface_data.shape[1] - 1.0)
        )
        assert cplot.axes.get_ylim() == pytest.approx(
            (0.0, surface_data.shape[0] - 1.0)
        )

    def test_specified_axis_values(self, surface_data):
        """Axis ranges should be as specified."""
        x = np.linspace(2.0, 3.0, surface_data.shape[1])
        y = np.linspace(5.0, 6.0, surface_data.shape[0])
        cplot = kplt.contour(surface_data, x=x, y=y)
        assert cplot.axes.get_xlim() == pytest.approx((2.0, 3.0))
        assert cplot.axes.get_ylim() == pytest.approx((5.0, 6.0))

    def test_specified_levels(self, surface_data):
        """A sequence of levels should be plotted as specified."""
        levels = [-0.5, 0.0, 0.5]
        cplot = kplt.contour(surface_data, levels=levels)
        np.testing.assert_allclose(cplot.qcntset.levels, levels)

    def test_number_of_levels(self, surface_data):
        """An int number of levels should select the levels automatically."""
        cplot = kplt.contour(surface_data, levels=3)
        assert len(cplot.qcntset.levels) > 1

    def test_contour_labels(self, surface_data):
        """Contour labels should be displayed by default."""
        cplot = kplt.contour(surface_data, levels=[-0.5, 0.0, 0.5])
        assert len(cplot.qcntset.labelTexts) > 0
        assert cplot.qcntset.labelTexts[0].get_fontsize() == 10

    def test_contour_label_format(self, surface_data):
        """A contour label format should be applied when specified."""
        cplot = kplt.contour(
            surface_data, levels=[-0.5, 0.0, 0.5], clabel_format="%.3f"
        )
        assert {t.get_text() for t in cplot.qcntset.labelTexts} == {
            "-0.500",
            "0.000",
            "0.500",
        }

    def test_contour_label_fontsize(self, surface_data):
        """A contour label font size should be applied when specified."""
        cplot = kplt.contour(surface_data, levels=[0.0], clabel_fontsize=14)
        assert cplot.qcntset.labelTexts[0].get_fontsize() == 14

    @pytest.mark.parametrize("clabel_fontsize", [0, None])
    def test_no_contour_labels(self, surface_data, clabel_fontsize):
        """Contour labels should be suppressed by a zero or None font size."""
        cplot = kplt.contour(
            surface_data, levels=[0.0], clabel_fontsize=clabel_fontsize
        )
        assert len(cplot.qcntset.labelTexts) == 0

    def test_colormesh(self, surface_data):
        """The colormesh should represent the plotted array."""
        cplot = kplt.contour(surface_data)
        np.testing.assert_allclose(
            cplot.axesimage.get_array().reshape(surface_data.shape), surface_data
        )

    def test_default_colormap(self, surface_data):
        """The default colormap should be YlOrRd."""
        assert kplt.contour(surface_data).axesimage.get_cmap().name == "YlOrRd"

    def test_specified_colormap(self, surface_data):
        """A specified colormap should be used."""
        cplot = kplt.contour(surface_data, cmap="viridis")
        assert cplot.axesimage.get_cmap().name == "viridis"

    def test_colormap_limits(self, surface_data):
        """Colormap limits should default to the data range."""
        cplot = kplt.contour(surface_data)
        assert cplot.vmin_ref == pytest.approx(surface_data.min())
        assert cplot.vmax_ref == pytest.approx(surface_data.max())

    def test_specified_colormap_limits(self, surface_data):
        """Colormap limits should be as specified."""
        cplot = kplt.contour(surface_data, vmin=-0.5, vmax=0.5)
        assert cplot.vmin_ref == -0.5
        assert cplot.vmax_ref == 0.5

    def test_alpha(self, surface_data):
        """Transparency should be as specified."""
        assert kplt.contour(surface_data, alpha=0.5).axesimage.get_alpha() == 0.5

    @pytest.mark.parametrize(
        "kwargs,xscale,yscale",
        [
            ({}, "linear", "linear"),
            ({"xlog": True}, "log", "linear"),
            ({"ylog": True}, "linear", "log"),
            ({"xlog": True, "ylog": True}, "log", "log"),
        ],
    )
    def test_log_scale(self, surface_data, kwargs, xscale, yscale):
        """Axis scales should be set as specified."""
        x = np.linspace(1.0, 2.0, surface_data.shape[1])
        y = np.linspace(1.0, 2.0, surface_data.shape[0])
        cplot = kplt.contour(surface_data, x=x, y=y, **kwargs)
        assert cplot.axes.get_xscale() == xscale
        assert cplot.axes.get_yscale() == yscale

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
    def test_log_scale_dict(self, surface_data, kwargs, xscale, yscale):
        """A dict log scale specification should imply a log scale."""
        x = np.linspace(1.0, 2.0, surface_data.shape[1])
        y = np.linspace(1.0, 2.0, surface_data.shape[0])
        cplot = kplt.contour(surface_data, x=x, y=y, **kwargs)
        assert cplot.axes.get_xscale() == xscale
        assert cplot.axes.get_yscale() == yscale

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_log_scale_base(self, surface_data, axis):
        """A log scale base should be passed to the axis scale."""
        x = np.linspace(1.0, 2.0, surface_data.shape[1])
        y = np.linspace(1.0, 2.0, surface_data.shape[0])
        cplot = kplt.contour(surface_data, x=x, y=y, **{axis + "log": {"base": 2}})
        axis_obj = getattr(cplot.axes, axis + "axis")
        assert axis_obj.get_transform().base == 2

    def test_log_scale_distinct_axis_properties(self, surface_data):
        """Each log scale axis should have its own properties."""
        x = np.linspace(1.0, 2.0, surface_data.shape[1])
        y = np.linspace(1.0, 2.0, surface_data.shape[0])
        cplot = kplt.contour(
            surface_data, x=x, y=y, xlog={"base": 10}, ylog={"base": 2}
        )
        assert cplot.axes.xaxis.get_transform().base == 10
        assert cplot.axes.yaxis.get_transform().base == 2

    def test_log_scale_invalid_property(self, surface_data):
        """An invalid log scale property should raise an exception."""
        with pytest.raises(TypeError):
            kplt.contour(surface_data, xlog={"basex": 2})

    @pytest.mark.parametrize(
        "kwargs,xfmt,yfmt",
        [
            ({}, " 1.23", " 1.23"),
            ({"xlog": True, "ylog": True}, " 1.23e+00", " 1.23e+00"),
            ({"xlog": {}, "ylog": {"base": 2}}, " 1.23e+00", " 1.23e+00"),
            ({"xlog": {"base": 2}}, " 1.23e+00", " 1.23"),
        ],
    )
    def test_data_format(self, surface_data, kwargs, xfmt, yfmt):
        """Cursor coordinate display formats should follow the axis scales."""
        x = np.linspace(1.0, 2.0, surface_data.shape[1])
        y = np.linspace(1.0, 2.0, surface_data.shape[0])
        cplot = kplt.contour(surface_data, x=x, y=y, **kwargs)
        assert cplot.axes.fmt_xdata(1.234) == xfmt
        assert cplot.axes.fmt_ydata(1.234) == yfmt

    def test_labels_and_title(self, surface_data):
        """Axis labels and title should be set when specified."""
        cplot = kplt.contour(
            surface_data, xlabel="X", ylabel="Y", title="T", xylabel_fontsize=14
        )
        assert cplot.axes.get_xlabel() == "X"
        assert cplot.axes.get_ylabel() == "Y"
        assert cplot.axes.get_title() == "T"
        assert cplot.axes.xaxis.get_label().get_fontsize() == 14

    def test_figure_attributes(self, surface_data):
        """Figure size and number should be as specified."""
        cplot = kplt.contour(surface_data, figsize=(5, 4), fignum=5)
        assert cplot.figure.number == 5
        assert tuple(cplot.figure.get_size_inches()) == (5.0, 4.0)

    def test_specified_axes(self, surface_data):
        """Plotting into a specified axes should not create a figure."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        cplot = kplt.contour(surface_data, ax=ax[0])
        assert cplot.axes is ax[0]
        assert cplot.figure is fig
        assert plt.get_fignums() == [fig.number]

    def test_event_manager(self, surface_data):
        """The plot should be registered with the figure event manager."""
        cplot = kplt.contour(surface_data)
        fem = figure_event_manager(cplot.figure)
        assert cplot.event_manager is fem
        assert fem.plot_for_axes(cplot.axes) is cplot

    def test_event_manager_shared_figure(self, surface_data):
        """A single figure event manager should serve multiple plots."""
        _, ax = plt.subplots(nrows=1, ncols=2)
        cplot0 = kplt.contour(surface_data, ax=ax[0])
        cplot1 = kplt.contour(surface_data, ax=ax[1])
        fem = figure_event_manager(cplot0.figure)
        assert fem is figure_event_manager(cplot1.figure)
        assert fem.plots == [cplot0, cplot1]

    def test_feature_support(self, surface_data):
        """The plot should support colormap range adjustment and zoom."""
        cplot = kplt.contour(surface_data)
        assert isinstance(cplot, (kplt.ColorbarMixin, kplt.ZoomMixin))
        assert cplot.vmin_ref == cplot.axesimage.norm.vmin
        assert cplot.xlim_ref == cplot.axes.get_xlim()
