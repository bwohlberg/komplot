# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for the komplot package namespace."""

import re
from importlib.metadata import PackageNotFoundError

import matplotlib.pyplot as plt
import numpy as np
import pytest

import komplot as kplt

PLOT_FUNCTIONS = ["plot", "contour", "surface", "imview", "volview"]


@pytest.fixture
def plot_args():
    """Arguments for each of the main plotting functions."""
    rng = np.random.default_rng(2718)
    array2d = rng.uniform(size=(8, 6))
    return {
        "plot": (np.arange(8),),
        "contour": (array2d,),
        "surface": (array2d,),
        "imview": (array2d,),
        "volview": (rng.uniform(size=(4, 8, 6)),),
    }


class TestNamespace:
    """Tests for the package namespace."""

    @pytest.mark.parametrize("name", kplt.__all__)
    def test_public_name_exists(self, name):
        """Every name in __all__ should be an attribute of the package."""
        assert hasattr(kplt, name)

    @pytest.mark.parametrize("name", kplt.__all__)
    def test_public_name_module(self, name):
        """Every name in __all__ should appear to originate in the package."""
        assert getattr(kplt, name).__module__ == "komplot"

    def test_no_duplicate_public_names(self):
        """Names in __all__ should be unique."""
        assert len(kplt.__all__) == len(set(kplt.__all__))

    @pytest.mark.parametrize(
        "name",
        [
            "cm",
            "colors",
            "gridspec",
            "rcParams",
            "ticker",
            "figure",
            "gca",
            "gcf",
            "savefig",
            "subplot",
            "subplots",
        ],
    )
    def test_matplotlib_reexport(self, name):
        """Selected matplotlib names should be accessible via the package."""
        assert hasattr(kplt, name)

    def test_subplots(self):
        """The re-exported subplots function should create a figure and axes."""
        fig, ax = kplt.subplots(nrows=1, ncols=2)
        assert fig is ax[0].get_figure()
        assert plt.get_fignums() == [fig.number]


class TestVersion:
    """Tests for the package version identifier."""

    def test_version_type(self):
        """The package version should be a non-empty string."""
        assert isinstance(kplt.__version__, str)
        assert kplt.__version__ != ""

    def test_package_version(self):
        """The package version should extend the public version."""
        assert kplt._package_version().startswith(kplt._public_version)

    def test_installed_version(self):
        """The installed version should be used when it is available."""
        assert kplt._installed_version() == kplt.__version__

    def test_installed_version_fallback(self, monkeypatch):
        """The package version should be used when not installed."""

        def raise_not_found(name):
            raise PackageNotFoundError(name)

        monkeypatch.setattr(kplt, "version", raise_not_found)
        assert kplt._installed_version() == kplt._package_version()

    def test_public_version_format(self):
        """The public version should be a valid version identifier."""
        assert re.match(r"^[0-9]+\.[0-9]+\.[0-9]+", kplt._public_version)


class TestDiscardReturnFunctions:
    """Tests for the automatically constructed no-return-value functions."""

    @pytest.mark.parametrize("name", PLOT_FUNCTIONS)
    def test_exists(self, name):
        """A no-return-value variant of each plot function should exist."""
        assert callable(getattr(kplt, name + "_"))

    @pytest.mark.parametrize("name", PLOT_FUNCTIONS)
    def test_names(self, name):
        """Function names should correspond to the attribute names."""
        func = getattr(kplt, name + "_")
        assert func.__name__ == name + "_"
        assert func.__qualname__ == name + "_"

    @pytest.mark.parametrize("name", PLOT_FUNCTIONS)
    def test_discards_return_value(self, name, plot_args):
        """The return value should be discarded."""
        assert getattr(kplt, name + "_")(*plot_args[name]) is None

    @pytest.mark.parametrize("name", PLOT_FUNCTIONS)
    def test_plot_created(self, name, plot_args):
        """A plot should nevertheless be created."""
        getattr(kplt, name + "_")(*plot_args[name])
        assert len(plt.get_fignums()) == 1

    @pytest.mark.parametrize("name", PLOT_FUNCTIONS)
    def test_keyword_arguments(self, name, plot_args):
        """Keyword arguments should be passed to the wrapped function."""
        getattr(kplt, name + "_")(*plot_args[name], title="T", fignum=9)
        assert plt.figure(9).axes[0].get_title() == "T"

    @pytest.mark.parametrize("name", PLOT_FUNCTIONS)
    def test_annotations(self, name):
        """Annotations should be inherited without the return annotation."""
        func = getattr(kplt, name + "_")
        wrapped = getattr(kplt, name)
        assert "return" not in func.__annotations__
        assert set(func.__annotations__) == set(wrapped.__annotations__) - {"return"}

    @pytest.mark.parametrize("name", PLOT_FUNCTIONS)
    def test_docstring(self, name):
        """The docstring should be derived from that of the wrapped function."""
        func = getattr(kplt, name + "_")
        wrapped_summary = getattr(kplt, name).__doc__.split("\n")[0]
        assert func.__doc__.split("\n")[0] == wrapped_summary
        assert "discards the return value" in func.__doc__
