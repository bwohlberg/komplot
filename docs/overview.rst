Overview
--------

`KomPlot <https://github.com/bwohlberg/komplot>`__ provides a convenience layer around selected `matplotlib <https://matplotlib.org/>`__ plotting functions, making it possible to construct a useful plot in a single function call, which is particularly useful for use within an interactive `ipython <https://ipython.org/>`__ or `JupyterLab <https://jupyter.org/>`__ session. KomPlot also provides a number of interactive controls, including zooming by mouse wheel scroll, colormap shifts when viewing images, and shifting between displayed slices of a volume.


Plot Types
==========

KomPlot provides the following plotting functions:


:func:`~komplot.plot`
    Plotting in 2D of lines and points.

:func:`~komplot.contour`
    A contour plot representation of a 3D surface.

:func:`~komplot.surf`
    A surface plot representation of a 3D surface.

:func:`~komplot.imview`
    A viewer for 2D images or slices of 3D volumes.


Interactive features
====================

It also provides interactive navigation support in addition to the standard `matplotlib <https://matplotlib.org/>`__ `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__. Specifically, all plot types support:

**q**
   Close figure. (This is also a standard keyboard shortcut.)

**PageUp/PageDown**
   Increase or decrease figure size by a scaling factor.


All plot types except for :func:`~komplot.surf` also support:

**Mouse wheel scroll**
   Zoom in or out at current cursor location.

Plots with a visible colorbar also support:

**Mouse wheel scroll in bottom half of colorbar**
   Increase or decrease colormap `vmin`.

**Mouse wheel scroll in top half of colorbar**
   Increase or decrease colormap `vmax`.

The :func:`~komplot.imview` plot type also supports moving between slices of a 3D volume by use of the attached slider or by mouse wheel scroll while the shift key is depressed. Note that none of the keyboard shortcuts (including detection of the shift key while the mouse wheel is scrolled) are functional within Jupyter notebooks with the
`ipympl <https://matplotlib.org/ipympl/>`__ matplotlib backend.


Usage Examples
==============

A number of example scripts, and a Jupyter notebook, illustrating usage are available in the :code:`examples` directory.
