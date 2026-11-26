=====================
KomPlot Release Notes
=====================


Version 0.0.5   (xxxx-xx-xx)
----------------------------

• Add test suite.
• Add function subplots, a drop-in replacement for matplotlib.pyplot.subplots
  that can reserve space for colorbars and volume slice sliders within the
  layout of the figure, via parameters cbar and slider, instead of taking it
  from the axes of the plots themselves. The plots of a figure are thereby of
  equal size whether or not they have a colorbar, and remain so when the
  figure is resized, and no space is wasted alongside plots that do not have
  one. A single colorbar may also be shared by all of the plots of a figure,
  or of each of its rows or columns. A colorbar or slider in reserved space is
  sized and positioned to match the extent of the plots that it represents,
  which is smaller than their gridspec cells when they have a fixed aspect
  ratio, as an image does. A colorbar shared by a set of plots is placed at a
  greater distance from them than one representing a single plot, so that the
  distance indicates which plots it represents.
• A colorbar is now displayed by default by a plot in an axes for which space
  has been reserved, i.e. the show_cbar parameter of imview and volview now
  defaults to None, meaning that a colorbar is displayed if space has been
  reserved for one.
• The invisible colorbar created by passing show_cbar=None to imview or
  volview, which was the previous mechanism for maintaining the alignment of
  the plots of a figure, has been removed in favour of reserved space, which
  is more reliable and does not reduce the size of the plots. The
  cbar_visible attribute of a plot with a colorbar has been removed.
• Major restructuring of the class hierarchy: each plot type is now a single
  class, combining plot state, construction, and event handling, composed
  from feature mixin classes (ZoomMixin, ColorbarMixin, ImageMixin, and
  SliceMixin). The plot state classes GenericPlot, ZoomablePlot, and
  ColorbarPlot, and the event manager classes AxesEventManager,
  ZoomEventManager, ColorbarEventManager, ImageViewEventManager, and
  VolumeViewEventManager have been deprecated, FigureEventManager being
  retained.
• Add functions share_cmap, share_slice, unshare_cmap, and unshare_slice,
  supporting sharing of colormap range and volume slice index between any
  number of plots, in the same or in different figures. They replace methods
  set_cmap_share and set_slice_share of FigureEventManager, which have been
  deprecated.
• Add functions plots and plot_for_axes, replacing method
  get_axevman_for_axes of FigureEventManager, which has been deprecated.
• The volume slice displayed by a volume view is now specified via the
  slice_index property.
• The API deprecated by the restructuring remains usable until version 0.0.7.
  Each deprecated name emits a DeprecationWarning and is translated into the
  corresponding part of the current API: a deprecated class name resolves to
  the class that replaces it, and a deprecated method is applied via the
  function or method that replaces it. The deprecated sharing methods require
  the plots that are to share the property to have been created, since they
  are translated into calls that act on plots rather than on axes. The event
  handler methods of the removed event manager classes, and attributes
  cmap_share_axes and slice_share_axes of FigureEventManager, are not
  supported.
• Interactive adjustment of the colormap range is now bounded by the range of
  the displayed data rather than by the initial colormap range, so that an
  initial range narrower than the data, as specified by e.g. the vmin_quantile
  parameter, may be widened to include all of the data. The bounds are
  available as the vmin_bound and vmax_bound attributes of a plot with a
  colorbar, and may be changed by assignment.
• Each interactive shift of a colormap limit is now a fraction cmap_delta of
  the current colormap range rather than of the initial range, so that the
  number of shifts required to traverse a range does not depend on its size.
  Narrowing of the range is limited to the cmap_span_min attribute, which
  defaults to a small fraction of the initial colormap range.
• Fix bug in handling of log scale properties of plot function. Log scale
  properties of the plot and contour functions are now specified by passing
  a dict of keyword arguments as parameter xlog or ylog, allowing distinct
  properties for each axis.
• Fix bug in handling of cmap_delta parameter of image and volume views.
• Fix bugs in interactive colormap range adjustment: the adjustment failed
  for a plot with colormap limits determined by matplotlib autoscaling of an
  array of integer type, and a mouse scroll event in an invisible colorbar
  raised an exception instead of being ignored.
• Increase minimum supported Python version to 3.11.

..
   The entries above are the record of the deprecation of the API replaced in
   this version, and remain when that API is removed, which requires an entry
   of its own. The full set of changes required to remove it is the list in the
   docstring of komplot/_deprecated.py.


Version 0.0.4   (2026-10-29)
----------------------------

• Fix bug in plot function return value discard mechanism.


Version 0.0.3   (2026-05-31)
----------------------------

• Fix bug in event handling.


Version 0.0.2   (2026-05-28)
----------------------------

• Bug and typing fixes.
• Add some `matplotlib` imports accessible from top-level package.


Version 0.0.1   (2025-07-04)
----------------------------

• Initial release.
