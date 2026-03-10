# Phase 2 — Spatial Constraint Modeling

All Phase 2 processing was performed manually in QGIS 3.x.

## What Was Done
- Slope constraint: excluded pixels > 10° (calculated on UTM DEM)
- WDPA buffer: 1km exclusion around all protected areas
- Water buffer: 500m exclusion around water bodies
- Urban mask: excluded dense built-up areas

## Key Fix
Initial rasterization used 0 as NoData causing Raster Calculator failure.
Fixed by setting NoData = -9999 before boolean combination.

## Result
69.2% of Odisha confirmed buildable across all 314 blocks.
Output: constraint_map_267.tif (not included — exceeds 100MB)

## Files
- data/Odisha.geojson — State boundary used for clipping