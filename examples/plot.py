import numpy as np

import komplot as kplt

"""
Define an `x` array and three 1-d functions of it.
"""
x = np.linspace(-1, 1, 101)
y1 = np.abs(x)
y2 = np.abs(x) ** 1.5
y3 = x**2


"""
Plot the three functions on the same axes.
"""
kplt.plot(
    x,
    np.stack((y1, y2, y3)).T,
    xlabel="x",
    ylabel="y",
    title="Plot Example",
    legend=("$|x|$", "$|x|^{(3/2)}$", "$x^2$"),
    legend_loc="upper center",
)


"""
We can also create a plot and then add to it. In this case we need to create the figure object separately and pass it as argument to the :func:`komplot.plot` function so that it doesn't automatically call `fig.show` after the first plot call.
"""
fig, ax = kplt.subplots()
kplt.plot(
    x,
    np.stack((y1, y2, y3)).T,
    xlabel="x",
    ylabel="y",
    title="Plot Example",
    legend=("$|x|$", "$|x|^{(3/2)}$", "$x^2$"),
    legend_loc="upper center",
    ax=ax,
)
kplt.plot(x[::5], y1[::5], lw=0, ms=8.0, marker="o", ax=ax)
fig.show()


input()
