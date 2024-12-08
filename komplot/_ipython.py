# -*- coding: utf-8 -*-
# Copyright (C) 2024 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Support for ipython and Jupyter notebook plots."""

import importlib.util
import os
import sys
from functools import wraps
from typing import Callable, Optional

HAVE_IPYMPL = bool(importlib.util.find_spec("ipympl"))


def _in_ipython() -> bool:
    """Determine whether code is running in an ipython shell.

    Returns:
        ``True`` if running in an ipython shell, ``False`` otherwise.
    """
    try:
        # See https://stackoverflow.com/questions/15411967
        shell = get_ipython().__class__.__name__  # type: ignore
        return bool(shell == "TerminalInteractiveShell")
    except NameError:
        return False


def _in_notebook() -> bool:
    """Determine whether code is running in a Jupyter Notebook shell.

    Returns:
        ``True`` if running in a notebook shell, ``False`` otherwise.
    """
    try:
        # See https://stackoverflow.com/questions/15411967
        shell = get_ipython().__class__.__name__  # type: ignore
        return bool(shell == "ZMQInteractiveShell")
    except NameError:
        return False


def set_ipython_plot_backend(backend: str = "qt"):
    """Set matplotlib backend within an ipython shell.

    Set matplotlib backend within an ipython shell. This function has the
    same effect as the line magic :code:`%matplotlib [backend]` but is
    called as a function and includes a check to determine whether the
    code is running in an ipython shell, so that it can safely be used
    within a normal python script since it has no effect when not running
    in an ipython shell.

    Args:
        backend: Name of backend to be passed to the :code:`%matplotlib`
           line magic command.
    """
    if _in_ipython():
        # See https://stackoverflow.com/questions/35595766
        get_ipython().run_line_magic("matplotlib", backend)  # type: ignore


def set_notebook_plot_backend(backend: Optional[str] = None):
    """Set matplotlib backend within a Jupyter Notebook shell.

    Set matplotlib backend within a Jupyter Notebook shell. This function
    has the same effect as the line magic :code:`%matplotlib [backend]`
    but is called as a function and includes a check to determine whether
    the code is running in a notebook shell, so that it can safely be
    used within a normal python script since it has no effect when not
    running in a notebook shell.

    Args:
        backend: Name of backend to be passed to the :code:`%matplotlib`
           line magic command. Defaults to "ipympl" if
           `ipympl <https://matplotlib.org/ipympl/>`__ is installed,
           otherwise defaults to "inline".
    """
    if backend is None:
        backend = "ipympl" if HAVE_IPYMPL else "inline"
    if _in_notebook():
        # See https://stackoverflow.com/questions/35595766
        get_ipython().run_line_magic("matplotlib", backend)  # type: ignore


def discard_func_return(func: Callable) -> Callable:
    """Return value discarding wrapper."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        discard = kwargs.pop("discard_return", True)
        ret = func(*args, **kwargs)
        if discard:
            return None
        else:
            return ret

    wrapper._discard_return = True  # type: ignore  # pylint: disable=W0212
    return wrapper


def config_notebook_plotting(
    backend: Optional[str] = None, discard_return: bool = True
):
    """Configure plotting functions for inline plotting.

    Configure plotting functions for inline plotting within a Jupyter
    notebook shell. This function has no effect when not within a
    notebook shell, and may therefore be used within a normal Python
    script. If environment variable ``MATPLOTLIB_IPYNB_BACKEND`` is set,
    the matplotlib backend is explicitly set to the specified value.

    Args:
        backend: Name of backend to be passed to
            :func:`set_notebook_plot_backend`.
        discard_return: Flag indicating whether to discard the return
            value of functions :func:`.plot`, :func:`.surf`,
            :func:`.contour`, and :func:`.imview` to avoid undesired
            output in a notebook when the output is not assigned to a
            variable. If ``True``, this choice can be overridden for
            individual function calls by setting parameter
            :code:`keep_return` to ``True``. (This parameter is added
            by the function wrapper installed by
            :func:`config_notebook_plotting`.)

    """
    # Check whether running within a notebook shell and have
    # not already monkey patched the plot function
    module = sys.modules[__name__.split(".")[0]]
    if _in_notebook() and not hasattr(module.plot, "_discard_return"):
        # Set backend if specified by environment variable
        if "MATPLOTLIB_IPYNB_BACKEND" in os.environ:
            if os.environ["MATPLOTLIB_IPYNB_BACKEND"] != "":
                set_notebook_plot_backend(os.environ["MATPLOTLIB_IPYNB_BACKEND"])
        else:
            set_notebook_plot_backend(backend)

        # Replace plot etc. functions with a wrapper function that discards
        # their return values (within a notebook with inline plotting, plots
        # are duplicated if the return value from the original function is
        # not assigned to a variable)
        if discard_return:
            for func in (module.plot, module.surf, module.contour, module.imview):
                setattr(module, func.__name__, discard_func_return(func))

        # Disable figure show method (results in a warning if used within
        # a notebook with inline plotting)
        import matplotlib.figure

        def show_disable(self):
            pass

        matplotlib.figure.Figure.show = show_disable
