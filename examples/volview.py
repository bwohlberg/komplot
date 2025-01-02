from imageio.v3 import imread

from komplot import figure_event_manager, subplots, volview

"""
Load examples volume.
"""
vol = imread("imageio:stent.npz")


"""
Display slice of the volume.
"""
iv = volview(vol, slice_axis=0, cmap="Blues")
iv.set_volume_slice(128)


"""
Display slice of the transposed volume with colorbar.
"""
iv = volview(vol.transpose((1, 2, 0)), slice_axis=0, cmap="Blues", show_cbar=True)
iv.set_volume_slice(110)


"""
Plot slices of two different axes as subplots of the same figure. This is only feasible because the  volume has the same dimensions on these two axes: plot axes should not be slice-shared across volume axes of different dimensions.
"""
fig, ax = subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 8))
volview(
    vol, slice_axis=2, title="Axis 0-1 slices", cmap="Blues", show_cbar=True, ax=ax[0]
)
volview(
    vol, slice_axis=1, title="Axis 0-2 slices", cmap="Blues", show_cbar=True, ax=ax[1]
)
fem = figure_event_manager(fig, error=True)
fem.set_slice_share(ax)
fem.set_cmap_share(ax)
fem.get_axevman_for_axes(ax[0]).plot.set_volume_slice(100)
fig.show()


input()
