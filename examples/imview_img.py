from imageio.v3 import imread

from komplot import figure_event_manager, imview, subplots

"""
Load examples images.
"""
img = imread("imageio:camera.png")
imc = imread("imageio:immunohistochemistry.png")


"""
Plot an image and return an image state object.
"""
iv = imview(imc)


"""
Plot two images as subplots of the same figure.
"""
fig, ax = subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18, 8))
imview(img, cmap="Greys", show_cbar=True, ax=ax[0])
imview(img, cmap="Blues", show_cbar=True, ax=ax[1])
fem = figure_event_manager(fig, error=True)
fem.set_cmap_share(ax)
fig.show()


input()
