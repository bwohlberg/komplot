Overview
--------

`KomPlot <https://github.com/bwohlberg/komplot>`__ provides a convenience layer around selected `matplotlib <https://matplotlib.org/>`__ plotting functions, making it possible to construct a useful plot in a single function call, which is particularly useful for use within an interactive `ipython <https://ipython.org/>`__ or `JupyterLab <https://jupyter.org/>`__ session. KomPlot also provides a number of interactive controls, including zooming by mouse wheel scroll, colormap shifts when viewing images, and shifting between displayed slices of a volume.


Plot Types
==========

KomPlot supports the following types of plots.

|

Lines and points in 2D
^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :show-source-link: False

    import numpy as np
    import komplot as kplt
    x = np.linspace(-1, 1, 101)
    y1 = np.abs(x)
    y2 = np.abs(x) ** 1.5
    y3 = x**2
    kplt.plot(x, np.stack((y1, y2, y3)).T, xlabel="x", ylabel="y", title="Plot Example",
	      legend=("$|x|$", "$|x|^{(3/2)}$", "$x^2$"), legend_loc="upper center")


Plotting of lines and points in 2D is supported by the :func:`~komplot.plot` function. A `usage example <https://github.com/bwohlberg/komplot/blob/main/examples/plot.py>`__ is available.

|

Contour plot representation of a 3D surface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :context:
    :show-source-link: False

    import numpy as np
    import komplot as kplt
    x = np.linspace(0, 2, 50)[np.newaxis, :]
    y = np.linspace(-1, 1, 51)[:, np.newaxis]
    z = np.sin(y) * np.cos(2 * x * y)
    kplt.contour(z, x, y, xlabel="x", ylabel="y", title="Contour Plot Example", figsize=(6, 5))


Contour plot representations of a 3D surface are supported by the :func:`~komplot.contour` function. A `usage example <https://github.com/bwohlberg/komplot/blob/main/examples/surfcont.py>`__ is available.

|

Surface plot representation of a 3D surface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :context: close-figs
    :show-source-link: False

    kplt.surface(z, x, y, elev=25, azim=-25, xlabel="x", ylabel="y", zlabel="z",
		 title="Surface Plot Example", levels=5, figsize=(7, 6))


Surface plot representations of a 3D surface are supported by the :func:`~komplot.surface` function. A `usage example <https://github.com/bwohlberg/komplot/blob/main/examples/surfcont.py>`__ is available.

|

Viewer for 2D images
^^^^^^^^^^^^^^^^^^^^

.. plot::
    :show-source-link: False

    from imageio.v3 import imread
    import komplot as kplt
    imc = imread("imageio:immunohistochemistry.png")
    img = imc[..., 0]
    fig, ax = kplt.subplots(nrows=1, ncols=2, cbar=[False, True], sharex=True, sharey=True,
			    figsize=(16, 8), gridspec_kw={"wspace": 0.1})
    fig.suptitle("Figure Title", fontsize=14)
    kplt.imview(imc, title="Colour Image", ax=ax[0])
    kplt.imview(img, cmap=kplt.cm.coolwarm, title="Monochrome Image", ax=ax[1])
    fig.show()


Function :func:`~komplot.imview` provides a viewer for 2D images. A `usage example <https://github.com/bwohlberg/komplot/blob/main/examples/imview.py>`__ is available.

|

Viewer for slices of 3D volumes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::
    :show-source-link: False

    from imageio.v3 import imread
    import komplot as kplt
    vol = imread("imageio:stent.npz")
    iv = kplt.volview(vol.transpose((1, 2, 0)), slice_axis=0, vmin_quantile=1e-2, cmap="viridis",
		      show_cbar=True)
    iv.slice_index = 110


Function :func:`~komplot.volview` provides a viewer for slices of 3D volumes. A `usage example <https://github.com/bwohlberg/komplot/blob/main/examples/volview.py>`__ is available.



Interactive features
====================

It also provides interactive adjustment/navigation support in addition to the standard `matplotlib <https://matplotlib.org/>`__ `interactive features <https://matplotlib.org/stable/users/explain/figure/interactive.html#interactive-navigation>`__:


+------------------------+--------------------------------+----------------------------+
| Action/Key             | Effect                         |  Valid                     |
+========================+================================+============================+
| **q**                  | Close figure. (This is also a  | All plot types             |
|			 | standard keyboard shortcut.)   |                            |
+------------------------+--------------------------------+----------------------------+
| **PageUp/PageDown**    | Increase or decrease figure    | All plot types             |
|			 | size by a scaling factor.      |                            |
+------------------------+--------------------------------+----------------------------+
| **Mouse wheel scroll** | Zoom in or out at current      | All plot types except      |
| in main figure         | cursor location.               | :func:`~komplot.surface`   |
+------------------------+--------------------------------+----------------------------+
| **Mouse wheel scroll** | Increase or decrease colormap  | All plots with a           |
| in bottom half of      | :code:`vmin`, within the range | visible colorbar           |
| colorbar               | of the displayed data.         |                            |
+------------------------+--------------------------------+----------------------------+
| **Mouse wheel scroll** | Increase or decrease colormap  | All plots with a           |
| in top half of         | :code:`vmax`, within the range | visible colorbar           |
| colorbar               | of the displayed data.         |                            |
+------------------------+--------------------------------+----------------------------+
| **Mouse wheel scroll** | Increase or decrease slice     | A :func:`~komplot.volview` |
| in main figure with    | index.                         | plot of a 3D volume.       |
| **Shift** depressed    |                                |                            |
+------------------------+--------------------------------+----------------------------+
| Slice **slider bar**   | Increase or decrease slice     | A :func:`~komplot.volview` |
|			 | index.                         | plot of a 3D volume.       |
+------------------------+--------------------------------+----------------------------+


Note that none of the keyboard shortcuts (including detection of the shift key while the mouse wheel is scrolled) are functional within Jupyter notebooks with the `ipympl <https://matplotlib.org/ipympl/>`__ matplotlib backend.


Figure Layout
=============

A colorbar is usually attached to a plot by taking the space for it from the axes of the plot itself, with the result that a plot with a colorbar is smaller than one without, that the plots of a figure are misaligned, and that part of the space of each plot is occupied by a colorbar rather than by the plot itself. Function :func:`~komplot.subplots`, which is otherwise a drop-in replacement for :func:`matplotlib.pyplot.subplots`, to which it delegates when no space is requested, avoids this by reserving the space within the layout of the figure itself, as additional gridspec tracks, so that the axes of the plots are of equal size whether or not they have a colorbar, and remain so when the figure is resized

.. code-block:: python

    fig, ax = kplt.subplots(nrows=1, ncols=2, cbar=[False, True])
    kplt.imview(imc, ax=ax[0])
    kplt.imview(img, ax=ax[1])

The :code:`cbar` parameter specifies the plots for which space is to be reserved: ``True`` for every plot of the grid, or a sequence of bools, in row-major order, for individual plots. Each axes of the grid refers to the axes reserved for it by its :code:`cax` attribute, and a plot displayed in an axes for which space has been reserved displays a colorbar by default, i.e. it is not necessary to specify :code:`show_cbar=True`. Space for the slice selection slider of a :func:`~komplot.volview` plot, which has the same effect on the alignment of the plots of a figure, is reserved in the same way by the :code:`slider` parameter, each axes referring to the axes reserved for it by its :code:`sax` attribute. The side on which space is reserved, its size, and the space between it and the plot, are specified by parameters :code:`cbar_location`, :code:`cbar_size`, and :code:`cbar_pad`, and by the corresponding :code:`slider` parameters. Since the sizes are specified as fractions of the size of a plot, the layout is unchanged by a resize of the figure.

A single colorbar may also be shared by multiple plots, space for one shared by the whole grid of plots being reserved by :code:`cbar="shared"`, and for one shared by each row or column of plots by :code:`cbar="row"` or :code:`cbar="col"` respectively

.. code-block:: python

    fig, ax = kplt.subplots(nrows=1, ncols=3, cbar="shared")
    ivs = [kplt.imview(image, norm=norm, ax=axs) for image, axs in zip(images, ax)]
    kplt.share_cmap(ivs)

A shared colorbar is drawn by the first plot displayed in the axes sharing it, and represents the colormap range of that plot, a warning being emitted if a plot that defers to it has a different range. It is placed at a greater distance from the plots than a colorbar representing a single plot, so that the distance indicates which plots a colorbar represents; both distances may be specified by :code:`cbar_pad`. A reserved axes is sized and positioned to match the extent of the plots that it serves, which is smaller than their gridspec cells when they have a fixed aspect ratio, as an image does: a vertical colorbar extends from the top of the topmost of the plots that it serves to the bottom of the bottommost, whatever the aspect of each of them, and a colorbar or slider representing a single plot matches that plot. Placing the plots in a colormap share group (see `Sharing plot properties`_) ensures that an interactive adjustment of the range of any of them is applied to all.

Since the spacing of the tracks of the layout is specified by the gridspec itself, which matplotlib declines to adjust, a tight layout is not compatible with reserved space; a constrained layout may be used instead.


Plot Objects
============

Each of the plotting functions returns a plot object that provides access to the plot components and to the state of its interactive features. For example, the volume slice displayed by a :func:`~komplot.volview` plot is determined by the :attr:`~komplot.SliceMixin.slice_index` property of the returned object

.. code-block:: python

    iv = kplt.volview(vol)
    iv.slice_index = 100

and the colormap range of a plot with a colorbar may be adjusted via the
:meth:`~komplot.ColorbarMixin.shift_cmap_vmin` and :meth:`~komplot.ColorbarMixin.shift_cmap_vmax` methods. Each shift, whether by mouse wheel scroll in the colorbar or by a call to one of those methods, is a fraction :attr:`~komplot.ColorbarMixin.cmap_delta` of the *current* colormap range, so that the number of shifts required to traverse a range is independent of its size, and is bounded by :attr:`~komplot.ColorbarMixin.vmin_bound` and :attr:`~komplot.ColorbarMixin.vmax_bound`, which default to the range of the displayed data. An initial colormap range narrower than the data, such as that specified by the :code:`vmin_quantile` parameter of :func:`~komplot.imview`, may therefore be widened to include all of the data, and either bound may be changed by assignment if some other limit is wanted. Narrowing of the range is limited to :attr:`~komplot.ColorbarMixin.cmap_span_min`, since a colormap range of zero is not reliably representable. The plot object corresponding to an axes may be retrieved via :func:`~komplot.plot_for_axes`, and all the plot objects of a figure via :func:`~komplot.plots`.


Sharing plot properties
^^^^^^^^^^^^^^^^^^^^^^^

The colormap range and the volume slice index may be shared between any number of plots, in the same figure or in different figures, so that a change applied to one of them, whether interactively or programmatically, is applied to all of them

.. code-block:: python

    fig, ax = kplt.subplots(nrows=1, ncols=2)
    iv0 = kplt.volview(vol, slice_axis=1, show_cbar=True, ax=ax[0])
    iv1 = kplt.volview(vol, slice_axis=2, show_cbar=True, ax=ax[1])
    kplt.share_slice([iv0, iv1])
    kplt.share_cmap([iv0, iv1])

Each call to :func:`~komplot.share_slice` or :func:`~komplot.share_cmap` defines a complete share group, which is returned by the call: a plot that belonged to another group for the same property leaves it, and a group left with fewer than two members is dissolved. Plots may be removed from their groups via :func:`~komplot.unshare_slice` and :func:`~komplot.unshare_cmap`, which accept either plots or a share group. Since the colormap range is shared as a relative shift, plots in a colormap share group are not required to have the same colormap range, but all plots in a slice share group are required to have the same number of slices.


Usage Examples
==============

A number of example scripts, and a Jupyter notebook, illustrating usage are available in the `examples <https://github.com/bwohlberg/komplot/blob/main/examples>`__ directory.
