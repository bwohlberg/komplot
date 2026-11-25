# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._misc."""

import matplotlib.pyplot as plt

import komplot as kplt


class TestClose:
    """Tests for function close."""

    def test_close_figure_object(self):
        """Closing a figure object should close only that figure."""
        fig0 = plt.figure()
        fig1 = plt.figure()
        kplt.close(fig0)
        assert plt.get_fignums() == [fig1.number]

    def test_close_figure_number(self):
        """Closing a figure number should close only that figure."""
        fig0 = plt.figure()
        fig1 = plt.figure()
        kplt.close(fig1.number)
        assert plt.get_fignums() == [fig0.number]

    def test_close_all(self):
        """Calling close without arguments should close all figures."""
        plt.figure()
        plt.figure()
        kplt.close()
        assert plt.get_fignums() == []

    def test_close_all_without_figures(self):
        """Calling close without any open figures should be harmless."""
        kplt.close()
        assert plt.get_fignums() == []
