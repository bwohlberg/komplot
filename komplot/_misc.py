# -*- coding: utf-8 -*-
# Copyright (C) 2024 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Miscellaneous functions."""


from typing import Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def close(fig: Optional[Union[Figure, int]] = None):
    """Close figure(s).

    Close figure(s). If a figure object reference or figure number is
    provided, close the specified figure, otherwise close all figures.

    Args:
        fig: Figure object or number of figure to close. If ``None``,
           close all figures.
    """

    if fig is None:
        plt.close("all")
    else:
        plt.close(fig)
