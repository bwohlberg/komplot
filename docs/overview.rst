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
    fig, ax = kplt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18, 8))
    fig.suptitle("Figure Title", fontsize=14)
    kplt.imview(imc, title="Colour Image", show_cbar=None, ax=ax[0])
    kplt.imview(img, cmap=kplt.cm.coolwarm, title="Monochrome Image", show_cbar=True, ax=ax[1])
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
    iv.set_volume_slice(110)


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
| in bottom half of      | :code:`vmin`.                  | visible colorbar           |
| colorbar               |                                |                            |
+------------------------+--------------------------------+----------------------------+
| **Mouse wheel scroll** | Increase or decrease colormap  | All plots with a           |
| in top half of         | :code:`vmax`.                  | visible colorbar           |
| colorbar               |                                |                            |
+------------------------+--------------------------------+----------------------------+
| **Mouse wheel scroll** | Increase or decrease slice     | A :func:`~komplot.volview` |
| in main figure with    | index.                         | plot of a 3D volume.       |
| **Shift** depressed    |                                |                            |
+------------------------+--------------------------------+----------------------------+
| Slice **slider bar**   | Increase or decrease slice     | A :func:`~komplot.volview` |
|			 | index.                         | plot of a 3D volume.       |
+------------------------+--------------------------------+----------------------------+


Note that none of the keyboard shortcuts (including detection of the shift key while the mouse wheel is scrolled) are functional within Jupyter notebooks with the `ipympl <https://matplotlib.org/ipympl/>`__ matplotlib backend.


Usage Examples
==============

A number of example scripts, and a Jupyter notebook, illustrating usage are available in the `examples <https://github.com/bwohlberg/komplot/blob/main/examples>`__ directory.
