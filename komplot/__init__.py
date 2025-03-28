# -*- coding: utf-8 -*-
# Copyright (C) 2024-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Convenient plotting functions with interactive features.

Alternative high-level interface to selected :mod:`matplotlib`
plotting functions.
"""

import functools
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


_public_version = "0.0.1rc4"


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
    "volview",
    "close",
    "ContourPlot",
    "ImageView",
    "LinePlot",
    "SurfacePlot",
    "VolumeView",
    "GenericPlot",
    "ZoomablePlot",
    "ColorbarPlot",
    "figure_event_manager",
    "FigureEventManager",
    "AxesEventManager",
    "ZoomEventManager",
    "ColorbarEventManager",
    "ImageViewEventManager",
    "VolumeViewEventManager",
    "set_ipython_plot_backend",
    "set_notebook_plot_backend",
    "config_notebook_plotting",
]


# Imported items in __all__ appear to originate in top-level module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__


# Construct no-return-value versions of main plotting functions
def _discard_return(func, name):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    attr = "__annotate__" if hasattr(func, "__annotate__") else "__annotations__"
    setattr(wrapper, attr, getattr(func, attr).copy())
    del getattr(wrapper, attr)["return"]
    if hasattr(func, "__type_params__"):
        wrapper.__type_params__ = func.__type_params__
    docs = func.__doc__.split("\n")
    wrapper.__doc__ = (
        docs[0]
        + "\n"
        + f"""
    This version of :func:`{func.__name__}` discards the return value, for use in
    Jupyter notebooks where the return value is not needed, and which would clutter
    the following output cell.
    """
    )
    return wrapper


for func in (plot, contour, surface, imview, volview):
    name = func.__name__ + "_"
    setattr(sys.modules[__name__], name, _discard_return(func, name))
del func
