# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Shared fixtures and event simulation helpers for the test suite.

The non-interactive "Agg" backend is selected before :mod:`matplotlib.pyplot`
is imported so that no attempt is made to open a display when running the
tests.
"""

import contextlib

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # isort: skip
import numpy as np  # isort: skip
import pytest  # isort: skip
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent  # isort: skip

# -- Event simulation helpers ------------------------------------------------
#
# Events are constructed directly and dispatched via the (private) Event
# ._process method, as is done within the matplotlib test suite itself, since
# the "Agg" backend has no mechanism for generating GUI events. All such use
# is confined to this module so that it can readily be adapted if required.


def press_key(fig, key: str):
    """Simulate a key press event on a figure."""
    KeyEvent("key_press_event", fig.canvas, key)._process()


def release_key(fig, key: str):
    """Simulate a key release event on a figure."""
    KeyEvent("key_release_event", fig.canvas, key)._process()


def leave_figure(fig):
    """Simulate a figure leave event."""
    Event("figure_leave_event", fig.canvas)._process()


def move_mouse(ax, xdata: float, ydata: float):
    """Simulate mouse motion to a data coordinate within an axes."""
    fig = ax.get_figure()
    fig.canvas.draw()
    x, y = ax.transData.transform((xdata, ydata))
    MouseEvent("motion_notify_event", fig.canvas, x, y)._process()


def scroll_at_data(ax, xdata: float, ydata: float, button: str = "up"):
    """Simulate a mouse scroll at a data coordinate within an axes.

    Args:
        ax: Axes in which the scroll event occurs.
        xdata: Data `x` coordinate of the event.
        ydata: Data `y` coordinate of the event.
        button: Either "up" or "down".
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    x, y = ax.transData.transform((xdata, ydata))
    MouseEvent("scroll_event", fig.canvas, x, y, button=button)._process()


def scroll_in_axes(ax, xfrac: float = 0.5, yfrac: float = 0.5, button: str = "up"):
    """Simulate a mouse scroll at a relative position within an axes.

    Args:
        ax: Axes in which the scroll event occurs.
        xfrac: Horizontal position of the event as a fraction of the axes
           width.
        yfrac: Vertical position of the event as a fraction of the axes
           height.
        button: Either "up" or "down".
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    x0, y0, width, height = ax.get_window_extent().bounds
    MouseEvent(
        "scroll_event",
        fig.canvas,
        x0 + xfrac * width,
        y0 + yfrac * height,
        button=button,
    )._process()


def scroll_outside_axes(fig, button: str = "up"):
    """Simulate a mouse scroll outside of any axes of a figure."""
    fig.canvas.draw()
    MouseEvent("scroll_event", fig.canvas, 0, 0, button=button)._process()


class RecordingToolbar:
    """Minimal stand-in for a backend toolbar that records messages.

    The "Agg" backend canvas has no toolbar, so toolbar-dependent code paths
    are exercised by attaching an instance of this class to a canvas.
    """

    def __init__(self):
        self.messages: list = []

    def set_message(self, msg: str):
        """Record a toolbar message."""
        self.messages.append(msg)

    def _wait_cursor_for_draw_cm(self):
        """Context manager required by the canvas draw method."""
        return contextlib.nullcontext()


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def image() -> np.ndarray:
    """A monochrome test image with distinct pixel values."""
    return np.arange(48, dtype=np.float64).reshape(8, 6)


@pytest.fixture
def wide_image() -> np.ndarray:
    """A monochrome test image that is wider than it is tall."""
    return np.arange(48, dtype=np.float64).reshape(4, 12)


@pytest.fixture
def color_image() -> np.ndarray:
    """A three channel colour test image."""
    rng = np.random.default_rng(1234)
    return rng.uniform(size=(6, 4, 3))


@pytest.fixture
def volume() -> np.ndarray:
    """A monochrome test volume."""
    rng = np.random.default_rng(5678)
    return rng.uniform(size=(5, 8, 6))


@pytest.fixture
def color_volume() -> np.ndarray:
    """A three channel colour test volume."""
    rng = np.random.default_rng(9012)
    return rng.uniform(size=(4, 8, 6, 3))


@pytest.fixture
def surface_data() -> np.ndarray:
    """A smooth 2d array suitable for surface and contour plots."""
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 2.0, 8)
    return np.outer(np.sin(np.pi * y), np.cos(np.pi * x / 2.0))


@pytest.fixture
def recording_toolbar() -> RecordingToolbar:
    """A message-recording stand-in for a backend toolbar."""
    return RecordingToolbar()


@pytest.fixture
def toolbar_axes():
    """A figure and axes with a message-recording toolbar attached.

    Returns: A tuple consisting of the figure, the axes, and the toolbar.
    """
    fig, ax = plt.subplots()
    toolbar = RecordingToolbar()
    fig.canvas.toolbar = toolbar
    return fig, ax, toolbar
