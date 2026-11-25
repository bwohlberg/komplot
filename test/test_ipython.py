# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._ipython."""

import builtins

import matplotlib.figure
import pytest

import komplot as kplt
from komplot import _ipython


class FakeShell:
    """Stand-in for an ipython shell that records line magic calls."""

    def __init__(self):
        self.line_magics: list = []

    def run_line_magic(self, name: str, arg: str):
        """Record a line magic call."""
        self.line_magics.append((name, arg))


def make_shell_class(name: str):
    """Construct a shell class with a specified class name."""
    return type(name, (FakeShell,), {})


@pytest.fixture
def no_shell(monkeypatch):
    """Ensure that there is no get_ipython function in builtins."""
    monkeypatch.delattr(builtins, "get_ipython", raising=False)


@pytest.fixture
def shell(monkeypatch, request):
    """Install a fake shell of a class specified by indirect parametrization.

    Returns: The installed fake shell.
    """
    shell = make_shell_class(request.param)()
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell, raising=False)
    return shell


TERMINAL_SHELL = "TerminalInteractiveShell"
NOTEBOOK_SHELLS = ["ZMQInteractiveShell", "Shell"]


class TestShellDetection:
    """Tests for functions _in_ipython and _in_notebook."""

    def test_no_shell(self, no_shell):
        """Neither shell should be detected outside of ipython."""
        assert not _ipython._in_ipython()
        assert not _ipython._in_notebook()

    @pytest.mark.parametrize("shell", [TERMINAL_SHELL], indirect=True)
    def test_terminal_shell(self, shell):
        """An ipython terminal shell should be detected."""
        assert _ipython._in_ipython()
        assert not _ipython._in_notebook()

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_notebook_shell(self, shell):
        """A notebook shell should be detected."""
        assert _ipython._in_notebook()
        assert not _ipython._in_ipython()

    @pytest.mark.parametrize("shell", ["SomeOtherShell"], indirect=True)
    def test_other_shell(self, shell):
        """An unknown shell should not be detected as either shell type."""
        assert not _ipython._in_ipython()
        assert not _ipython._in_notebook()


class TestSetIpythonPlotBackend:
    """Tests for function set_ipython_plot_backend."""

    def test_no_shell(self, no_shell):
        """The backend should not be set outside of ipython."""
        kplt.set_ipython_plot_backend("agg")

    @pytest.mark.parametrize("shell", [TERMINAL_SHELL], indirect=True)
    def test_terminal_shell(self, shell):
        """The backend should be set within an ipython terminal shell."""
        kplt.set_ipython_plot_backend("agg")
        assert shell.line_magics == [("matplotlib", "agg")]

    @pytest.mark.parametrize("shell", [TERMINAL_SHELL], indirect=True)
    def test_default_backend(self, shell):
        """The default backend should be "qt"."""
        kplt.set_ipython_plot_backend()
        assert shell.line_magics == [("matplotlib", "qt")]

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_notebook_shell(self, shell):
        """The backend should not be set within a notebook shell."""
        kplt.set_ipython_plot_backend("agg")
        assert shell.line_magics == []


class TestSetNotebookPlotBackend:
    """Tests for function set_notebook_plot_backend."""

    def test_no_shell(self, no_shell):
        """The backend should not be set outside of a notebook."""
        kplt.set_notebook_plot_backend("agg")

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_notebook_shell(self, shell):
        """The backend should be set within a notebook shell."""
        kplt.set_notebook_plot_backend("agg")
        assert shell.line_magics == [("matplotlib", "agg")]

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_default_backend(self, shell):
        """The default backend should depend on availability of ipympl."""
        kplt.set_notebook_plot_backend()
        expected = "ipympl" if _ipython.HAVE_IPYMPL else "inline"
        assert shell.line_magics == [("matplotlib", expected)]

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_default_backend_without_ipympl(self, shell, monkeypatch):
        """The default backend should be "inline" without ipympl."""
        monkeypatch.setattr(_ipython, "HAVE_IPYMPL", False)
        kplt.set_notebook_plot_backend()
        assert shell.line_magics == [("matplotlib", "inline")]

    @pytest.mark.parametrize("shell", [TERMINAL_SHELL], indirect=True)
    def test_terminal_shell(self, shell):
        """The backend should not be set within an ipython terminal shell."""
        kplt.set_notebook_plot_backend("agg")
        assert shell.line_magics == []


class TestConfigNotebookPlotting:
    """Tests for function config_notebook_plotting."""

    @pytest.fixture(autouse=True)
    def preserve_figure_show(self, monkeypatch):
        """Ensure that a patched Figure.show method is restored."""
        monkeypatch.setattr(
            matplotlib.figure.Figure, "show", matplotlib.figure.Figure.show
        )

    def test_no_shell(self, no_shell):
        """Nothing should be changed outside of a notebook."""
        show = matplotlib.figure.Figure.show
        kplt.config_notebook_plotting()
        assert matplotlib.figure.Figure.show is show

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_notebook_shell(self, shell, monkeypatch):
        """The backend should be set and Figure.show disabled."""
        monkeypatch.delenv("MATPLOTLIB_IPYNB_BACKEND", raising=False)
        show = matplotlib.figure.Figure.show
        kplt.config_notebook_plotting("agg")
        assert shell.line_magics == [("matplotlib", "agg")]
        assert matplotlib.figure.Figure.show is not show

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_show_disabled(self, shell, monkeypatch):
        """The disabled Figure.show method should do nothing."""
        monkeypatch.delenv("MATPLOTLIB_IPYNB_BACKEND", raising=False)
        kplt.config_notebook_plotting("agg")
        fig = matplotlib.figure.Figure()
        assert fig.show() is None

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_backend_environment_variable(self, shell, monkeypatch):
        """The backend should be set from the environment variable."""
        monkeypatch.setenv("MATPLOTLIB_IPYNB_BACKEND", "agg")
        kplt.config_notebook_plotting("inline")
        assert shell.line_magics == [("matplotlib", "agg")]

    @pytest.mark.parametrize("shell", NOTEBOOK_SHELLS, indirect=True)
    def test_empty_backend_environment_variable(self, shell, monkeypatch):
        """An empty environment variable should leave the backend unchanged."""
        monkeypatch.setenv("MATPLOTLIB_IPYNB_BACKEND", "")
        kplt.config_notebook_plotting("inline")
        assert shell.line_magics == []

    @pytest.mark.parametrize("shell", [TERMINAL_SHELL], indirect=True)
    def test_terminal_shell(self, shell):
        """Nothing should be changed within an ipython terminal shell."""
        show = matplotlib.figure.Figure.show
        kplt.config_notebook_plotting("agg")
        assert shell.line_magics == []
        assert matplotlib.figure.Figure.show is show
