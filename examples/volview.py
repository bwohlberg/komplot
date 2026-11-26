from imageio.v3 import imread

from komplot import share_cmap, share_slice, subplots, volview

"""
Load examples volume.
"""
vol = imread("imageio:stent.npz")


"""
Display slice of the volume.
"""
iv = volview(vol, slice_axis=0, cmap="Blues")
iv.slice_index = 128


"""
Display slice of the transposed volume with colorbar.
"""
iv = volview(vol.transpose((1, 2, 0)), slice_axis=0, cmap="Blues", show_cbar=True)
iv.slice_index = 110


"""
Plot slices of two different axes as subplots of the same figure. This is only feasible because the  volume has the same dimensions on these two axes: plot axes should not be slice-shared across volume axes of different dimensions.
"""
fig, ax = subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 8))
iv0 = volview(
    vol, slice_axis=2, title="Axis 0-1 slices", cmap="Blues", show_cbar=True, ax=ax[0]
)
iv1 = volview(
    vol, slice_axis=1, title="Axis 0-2 slices", cmap="Blues", show_cbar=True, ax=ax[1]
)
share_slice([iv0, iv1])
share_cmap([iv0, iv1])
iv0.slice_index = 100
fig.show()


input()
