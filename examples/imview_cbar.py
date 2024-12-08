from imageio.v3 import imread

import komplot as kplt

imc = imread("imageio:immunohistochemistry.png")
img = imc[..., 0]


"""
We can view both images as subplots within the same figure, but the colour bar on the second image changes its aspect ratio, which has the undesirable result of the two images being displayed with different sizes.
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
One solution is to adjust the ratios of the widths of the two subplots. We can also share x and y axes so that a zoom in one image is replicated in the other (this is, of course, only possible in the interactive version of this demonstration script).
"""

fig, ax = kplt.subplots(
    nrows=1,
    ncols=2,
    sharex=True,
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1.07]},
    figsize=(19.5, 8),
)
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
An alternative solution is to add an invisible colorbar to the first image so that they have the same size. This can be achieved by setting `show_cbar=None` instead of `cbar=True`.
"""


fig, ax = kplt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18, 8))
fig.suptitle("Figure Title", fontsize=14)
kplt.imview(imc, title="Colour Image", show_cbar=None, ax=ax[0])
kplt.imview(
    img,
    cmap=kplt.cm.coolwarm,
    title="Monochrome Image",
    show_cbar=True,
    ax=ax[1],
)
fig.show()


# Wait for enter on keyboard
input()
