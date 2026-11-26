from imageio.v3 import imread

import komplot as kplt

imc = imread("imageio:immunohistochemistry.png")
img = imc[..., 0]


"""
We can view both images as subplots within the same figure, but a colorbar attached to the second image takes the space for itself from the axes of that image, with the undesirable result that the two images are displayed at different sizes.
"""

fig, ax = kplt.subplots(nrows=1, ncols=2, figsize=(16, 8), gridspec_kw={"wspace": 0.1})
fig.suptitle("Figure Title", fontsize=14)
kplt.imview(imc, title="Colour Image", ax=ax[0])
kplt.imview(
    img,
    cmap=kplt.cm.coolwarm,
    title="Monochrome Image",
    show_cbar=True,
    ax=ax[1],
)
fig.show()


"""
The solution is to reserve the space for the colorbar within the layout of the figure itself, via the `cbar` parameter of :func:`komplot.subplots`, which specifies the plots for which space is to be reserved. The images are then of equal size, and remain so when the figure is resized, since none of the space of either of them is taken by the colorbar. A colorbar is displayed by default in an axes for which space has been reserved. The space between the images is specified as usual by `wspace`, as a fraction of the width of an image, the default being that of matplotlib itself. We can also share x and y axes so that a zoom in one image is replicated in the other (this is, of course, only possible in the interactive version of this demonstration script).
"""

fig, ax = kplt.subplots(
    nrows=1,
    ncols=2,
    cbar=[False, True],
    sharex=True,
    sharey=True,
    figsize=(16, 8),
    gridspec_kw={"wspace": 0.1},
)
fig.suptitle("Figure Title", fontsize=14)
kplt.imview(imc, title="Colour Image", ax=ax[0])
kplt.imview(img, cmap=kplt.cm.coolwarm, title="Monochrome Image", ax=ax[1])
fig.show()


"""
When a set of images is displayed with the same colormap range, a single colorbar can represent all of them. Space for one that is shared by every plot in the figure is reserved by `cbar="shared"`, the colorbar being drawn by the first plot that is displayed. It is placed further from the images than a colorbar representing a single image, so that the distance indicates which images it represents. Since the colormap range of the plot that draws it is the one it represents, the plots are placed in a colormap share group so that an interactive adjustment of the range of any of them is applied to all.
"""

norm = kplt.colors.Normalize(vmin=0, vmax=255)
fig, ax = kplt.subplots(
    nrows=1, ncols=3, cbar="shared", figsize=(18, 6), gridspec_kw={"wspace": 0.1}
)
fig.suptitle("Colour Image Channels", fontsize=14)
ivs = [
    kplt.imview(imc[..., n], norm=norm, title=f"Channel {n}", ax=ax[n])
    for n in range(3)
]
kplt.share_cmap(ivs)
fig.show()


"""
Space is reserved in the same way for a grid of plots. A colorbar shared by every plot of a 2x2 grid of images, placed at the right of the grid, spans all of them, from the top of the images in the first row to the bottom of those in the second.
"""

chan = (imc[..., 0], imc[..., 1], imc[..., 2], imc.mean(axis=-1))
titles = ("Channel 0", "Channel 1", "Channel 2", "Channel Mean")

fig, ax = kplt.subplots(
    nrows=2,
    ncols=2,
    cbar="shared",
    sharex=True,
    sharey=True,
    figsize=(9, 9),
    gridspec_kw={"wspace": 0.1, "hspace": 0.1},
)
fig.suptitle("Shared Colorbar at the Right", fontsize=14)
ivs = [
    kplt.imview(data, norm=norm, title=title, ax=axs)
    for data, title, axs in zip(chan, titles, ax.ravel())
]
kplt.share_cmap(ivs)
fig.show()


"""
The side on which space is reserved is specified by `cbar_location`. A colorbar below the grid spans the full width of the images instead, from the left of those in the first column to the right of those in the second.
"""

fig, ax = kplt.subplots(
    nrows=2,
    ncols=2,
    cbar="shared",
    cbar_location="bottom",
    sharex=True,
    sharey=True,
    figsize=(9, 9),
    gridspec_kw={"wspace": 0.1, "hspace": 0.1},
)
fig.suptitle("Shared Colorbar at the Bottom", fontsize=14)
ivs = [
    kplt.imview(data, norm=norm, title=title, ax=axs)
    for data, title, axs in zip(chan, titles, ax.ravel())
]
kplt.share_cmap(ivs)
fig.show()


# Wait for enter on keyboard
input()
