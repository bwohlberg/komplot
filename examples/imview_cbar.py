from imageio.v3 import imread

import komplot as kplt

imc = imread("imageio:immunohistochemistry.png")
img = imc[..., 0]


"""
We can view both images as subplots within the same figure, but a colorbar attached to the second image takes the space for itself from the axes of that image, with the undesirable result that the two images are displayed at different sizes.
"""

fig, ax = kplt.subplots(nrows=1, ncols=2, figsize=(18, 8))
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
The solution is to reserve the space for the colorbar within the layout of the figure itself, via the `cbar` parameter of :func:`komplot.subplots`, which specifies the plots for which space is to be reserved. The images are then of equal size, and remain so when the figure is resized, since none of the space of either of them is taken by the colorbar. A colorbar is displayed by default in an axes for which space has been reserved. We can also share x and y axes so that a zoom in one image is replicated in the other (this is, of course, only possible in the interactive version of this demonstration script).
"""

fig, ax = kplt.subplots(
    nrows=1,
    ncols=2,
    cbar=[False, True],
    sharex=True,
    sharey=True,
    figsize=(18, 8),
)
fig.suptitle("Figure Title", fontsize=14)
kplt.imview(imc, title="Colour Image", ax=ax[0])
kplt.imview(img, cmap=kplt.cm.coolwarm, title="Monochrome Image", ax=ax[1])
fig.show()


"""
When a set of images is displayed with the same colormap range, a single colorbar can represent all of them. Space for one that is shared by every plot in the figure is reserved by `cbar="shared"`, the colorbar being drawn by the first plot that is displayed. Since the colormap range of that plot is the one represented by the colorbar, the plots are placed in a colormap share group so that an interactive adjustment of the range of any of them is applied to all.
"""

norm = kplt.colors.Normalize(vmin=0, vmax=255)
fig, ax = kplt.subplots(nrows=1, ncols=3, cbar="shared", figsize=(18, 6))
fig.suptitle("Colour Image Channels", fontsize=14)
ivs = [
    kplt.imview(imc[..., n], norm=norm, title=f"Channel {n}", ax=ax[n])
    for n in range(3)
]
kplt.share_cmap(ivs)
fig.show()


# Wait for enter on keyboard
input()
