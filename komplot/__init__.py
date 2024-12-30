# -*- coding: utf-8 -*-
# Copyright (C) 2024 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Convenient plotting functions with interactive features.

Alternative high-level interface to selected :mod:`matplotlib`
plotting functions.
"""

import sys
from importlib.metadata import PackageNotFoundError, version

from matplotlib import cm, rcParams
from matplotlib.pyplot import figure, gca, gcf, savefig, subplot, subplots

# isort: off

from ._contour import contour, ContourPlot
from ._imview import imview, ImageView, ImageViewEventManager
from ._volview import volview, VolumeView, VolumeViewEventManager
from ._plot import plot, LinePlot
from ._surface import surface, SurfacePlot
from ._event import (
    figure_event_manager,
    FigureEventManager,
    AxesEventManager,
    ZoomEventManager,
    ColorbarEventManager,
)
from ._state import GenericPlot, ZoomablePlot, ColorbarPlot
from ._misc import close
from ._ipython import (
    config_notebook_plotting,
    set_ipython_plot_backend,
    set_notebook_plot_backend,
)
from ._version import local_version_label


_public_version = "0.0.1.dev2"


def _package_version():
    return _public_version + local_version_label(_public_version)


def _installed_version():
    try:
        ver = version("komplot")
    except PackageNotFoundError:
        ver = _package_version()
    return ver


__version__ = _installed_version()


__all__ = [
    "contour",
    "imview",
    "plot",
    "surface",
    "close",
    "ContourPlot",
    "ImageView",
    "LinePlot",
    "SurfacePlot",
    "GenericPlot",
    "ZoomablePlot",
    "ColorbarPlot",
    "figure_event_manager",
    "FigureEventManager",
    "AxesEventManager",
    "ZoomEventManager",
    "ColorbarEventManager",
    "ImageViewEventManager",
    "set_ipython_plot_backend",
    "set_notebook_plot_backend",
    "config_notebook_plotting",
]


# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
