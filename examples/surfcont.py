import numpy as np

import komplot as kplt

"""
Define `x` and `y` arrays and a 2-d surface on `x`, `y`.
"""
x = np.linspace(0, 2, 50)[np.newaxis, :]
y = np.linspace(-1, 1, 51)[:, np.newaxis]
z = np.sin(y) * np.cos(2 * x * y)


"""
Plot a surface plot of the surface, including contour lines at the bottom of the `z` axis.
"""
kplt.surface(
    z,
    x,
    y,
    elev=25,
    azim=-25,
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="Surface Plot Example",
    levels=5,
    figsize=(7, 6),
)


"""
Plot a contour plot of the same surface.
"""
kplt.contour(
    z, x, y, xlabel="x", ylabel="y", title="Contour Plot Example", figsize=(6, 5)
)


"""
We can also plot within subplots of the same figure.
"""
fig, ax = kplt.subplots(nrows=1, ncols=2, figsize=(12.1, 5))
fig.suptitle("Figure Title", fontsize=14)
kplt.surface(
    z,
    x,
    y,
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="Surface Plot Example",
    ax=ax[0],
)
kplt.contour(z, x, y, xlabel="x", ylabel="y", title="Contour Plot Example", ax=ax[1])
fig.show()


input()
