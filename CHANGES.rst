=====================
KomPlot Release Notes
=====================


Version 0.0.5   (xxxx-xx-xx)
----------------------------

• Add test suite.
• Fix bug in handling of log scale properties of plot function. Log scale
  properties of the plot and contour functions are now specified by passing
  a dict of keyword arguments as parameter xlog or ylog, allowing distinct
  properties for each axis.
• Fix bug in handling of cmap_delta parameter of ImageViewEventManager and
  VolumeViewEventManager.
• Increase minimum supported Python version to 3.11.


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
