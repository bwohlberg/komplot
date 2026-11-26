# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._deprecated.

Delete this file when the API deprecated in version 0.0.5 is removed,
which is step 3 of the list in the docstring of the module that it
tests, that list being the full set of changes required.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import komplot as kplt
from komplot._deprecated import DEPRECATED_MEMBERS, DEPRECATED_NAMES, install


class TestDeprecatedNames:
    """Tests for the deprecated public names of the package."""

    @pytest.mark.parametrize("name", list(DEPRECATED_NAMES))
    def test_warns_and_resolves(self, name):
        """A deprecated name should warn and resolve to its replacement."""
        with pytest.warns(DeprecationWarning, match=f"komplot.{name} is deprecated"):
            attribute = getattr(kplt, name)
        assert attribute is DEPRECATED_NAMES[name]

    @pytest.mark.parametrize("name", list(DEPRECATED_NAMES))
    def test_names_replacement(self, name):
        """The replacement for a deprecated name should be a current name."""
        replacement = DEPRECATED_NAMES[name]
        assert getattr(kplt, replacement.__name__) is replacement
        assert replacement.__name__ in kplt.__all__

    def test_warning_reports_the_call_site(self):
        """A deprecation warning should be reported at the use of the name."""
        with pytest.warns(DeprecationWarning) as record:
            kplt.GenericPlot
        assert Path(record[0].filename) == Path(__file__)

    def test_import_from_package(self):
        """A deprecated name should be importable from the package."""
        with pytest.warns(DeprecationWarning):
            from komplot import ZoomablePlot
        assert ZoomablePlot is kplt.ZoomMixin

    def test_isinstance(self, image):
        """A plot should be an instance of the deprecated classes it replaces."""
        iview = kplt.imview(image, show_cbar=True)
        for name in ("GenericPlot", "ZoomablePlot", "ColorbarPlot"):
            with pytest.warns(DeprecationWarning):
                assert isinstance(iview, getattr(kplt, name))

    def test_unknown_name(self):
        """A name that is neither current nor deprecated should raise."""
        with pytest.raises(AttributeError, match="no attribute 'NoSuchName'"):
            kplt.NoSuchName

    def test_not_advertised(self):
        """The deprecated names should not be part of the documented API."""
        assert not set(DEPRECATED_NAMES) & set(kplt.__all__)


class TestDeprecatedSharing:
    """Tests for the deprecated axes-based sharing methods."""

    def test_set_cmap_share(self, image):
        """Colormap sharing by axes should share the colormap range."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        iview0 = kplt.imview(image, show_cbar=True, ax=ax[0])
        iview1 = kplt.imview(2.0 * image, show_cbar=True, ax=ax[1])
        manager = kplt.figure_event_manager(fig)
        with pytest.warns(DeprecationWarning, match="set_cmap_share is deprecated"):
            group = manager.set_cmap_share(ax)
        assert isinstance(group, kplt.ShareGroup)
        limits = iview1.cmap_limits
        iview0.shift_cmap_vmax(-0.1)
        assert iview1.cmap_limits != limits

    def test_set_slice_share(self, volume):
        """Slice sharing by axes should share the slice index."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        vview0 = kplt.volview(volume, ax=ax[0])
        vview1 = kplt.volview(volume, ax=ax[1])
        manager = kplt.figure_event_manager(fig)
        with pytest.warns(DeprecationWarning, match="set_slice_share is deprecated"):
            group = manager.set_slice_share([ax[0], ax[1]])
        assert isinstance(group, kplt.ShareGroup)
        vview0.set_volume_slice(3)
        assert vview1.slice_index == 3

    def test_axes_array(self, image):
        """Sharing should accept an array of axes as well as a list."""
        fig, ax = plt.subplots(nrows=2, ncols=2)
        views = [kplt.imview(image, show_cbar=True, ax=axs) for axs in ax.ravel()]
        manager = kplt.figure_event_manager(fig)
        with pytest.warns(DeprecationWarning):
            group = manager.set_cmap_share(ax)
        assert set(group.members()) == set(views)

    def test_sharing_requires_plots(self):
        """Sharing axes without plots should raise an informative error."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        manager = kplt.FigureEventManager(fig)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="2 of the 2 specified axes"):
                manager.set_cmap_share(ax)


class TestDeprecatedMembers:
    """Tests for the deprecated members of the retained classes."""

    def test_get_axevman_for_axes(self, image):
        """Retrieval by axes should return the plot that owns the axes."""
        iview = kplt.imview(image, show_cbar=True)
        manager = kplt.figure_event_manager(iview.figure)
        with pytest.warns(
            DeprecationWarning, match="get_axevman_for_axes is deprecated"
        ):
            assert manager.get_axevman_for_axes(iview.axes) is iview

    def test_get_axevman_for_axes_without_plot(self):
        """Retrieval for an axes without a plot should return ``None``."""
        fig, ax = plt.subplots()
        manager = kplt.FigureEventManager(fig)
        with pytest.warns(DeprecationWarning):
            assert manager.get_axevman_for_axes(ax) is None

    def test_plot_attribute(self, image):
        """The plot attribute of a plot should be the plot itself."""
        iview = kplt.imview(image, show_cbar=True)
        with pytest.warns(DeprecationWarning, match="plot attribute"):
            assert iview.plot is iview

    def test_old_chained_idiom(self, volume):
        """The documented chain of the removed API should still work."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        vview = kplt.volview(volume, ax=ax[0])
        kplt.volview(volume, ax=ax[1])
        manager = kplt.figure_event_manager(fig)
        with pytest.warns(DeprecationWarning):
            manager.get_axevman_for_axes(ax[0]).plot.set_volume_slice(2)
        assert vview.slice_index == 2

    def test_fig_event_man(self, image):
        """The fig_event_man attribute should be the figure event manager."""
        iview = kplt.imview(image, show_cbar=True)
        with pytest.warns(DeprecationWarning, match="fig_event_man attribute"):
            assert iview.fig_event_man is kplt.figure_event_manager(iview.figure)

    def test_axevman_from_ax(self, image):
        """The axes to manager mapping should map axes to their plots."""
        fig, ax = plt.subplots(nrows=1, ncols=2)
        views = [kplt.imview(image, show_cbar=True, ax=axs) for axs in ax]
        manager = kplt.figure_event_manager(fig)
        with pytest.warns(DeprecationWarning, match="axevman_from_ax"):
            mapping = manager.axevman_from_ax
        assert mapping == {axs: view for axs, view in zip(ax, views)}

    def test_register_axevman_for_axes(self, image):
        """Registration by axes should register the plot.

        The registration is of a plot that is already registered, since
        a plot registers itself on construction, so that the effect of
        the deprecated method is visible as a repeated registration.
        """
        iview = kplt.imview(image, show_cbar=True)
        manager = kplt.figure_event_manager(iview.figure)
        count = len(manager.plots)
        with pytest.warns(
            DeprecationWarning, match="register_axevman_for_axes is deprecated"
        ):
            manager.register_axevman_for_axes(iview.axes, iview)
        assert len(manager.plots) == count + 1


class TestRemovability:
    """Tests that the compatibility layer is isolated from the package."""

    def test_only_the_package_init_refers_to_the_layer(self):
        """No module of the package but its __init__ should use the layer."""
        package = Path(kplt.__file__).parent
        referring = sorted(
            path.name
            for path in package.glob("*.py")
            if path.name not in ("__init__.py", "_deprecated.py")
            and "_deprecated" in path.read_text()
        )
        assert referring == []

    def test_layer_is_used_only_at_the_end_of_the_package_init(self):
        """Use of the layer should be confined to a block that may be deleted.

        Comments are not use, the deletable block being referred to by a
        comment at the head of the file, where an import required only by
        that block is to be deleted with it.
        """
        source = Path(kplt.__file__).read_text()
        marker = "# Compatibility support for the API deprecated"
        assert source.count(marker) == 1
        code = [
            line
            for line in source[: source.index(marker)].split("\n")
            if not line.lstrip().startswith("#")
        ]
        assert [line for line in code if "_deprecated" in line] == []

    def test_package_does_not_use_the_layer(self, image, volume):
        """Current use of the package should emit no deprecation warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            fig, ax = plt.subplots(nrows=1, ncols=2)
            iview = kplt.imview(image, show_cbar=True, ax=ax[0])
            vview = kplt.volview(volume, show_cbar=True, ax=ax[1])
            kplt.share_cmap([iview, vview])
            vview.slice_index = 2
            iview.shift_cmap_vmax(-0.1)
            assert kplt.plot_for_axes(ax[0]) is iview
            fig.canvas.draw()

    def test_install_is_idempotent(self):
        """Repeated installation should be accepted, as after a reload."""
        install()
        for cls, name, _ in DEPRECATED_MEMBERS:
            assert hasattr(cls, name)

    def test_install_rejects_a_reintroduced_name(self):
        """Installation should fail if a deprecated name is in current use."""
        cls, name, member = DEPRECATED_MEMBERS[0]
        setattr(cls, name, lambda self, axes: None)
        try:
            with pytest.raises(RuntimeError, match="is in current use"):
                install()
        finally:
            setattr(cls, name, member)
