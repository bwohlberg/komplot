from imageio.v3 import imread

from komplot import imview, share_cmap, subplots

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
fig, ax = subplots(
    nrows=1, ncols=2, cbar=True, sharex=True, sharey=True, figsize=(18, 8)
)
iv0 = imview(img, cmap="Greys", ax=ax[0])
iv1 = imview(img, cmap="Blues", ax=ax[1])
share_cmap([iv0, iv1])
fig.show()


input()
