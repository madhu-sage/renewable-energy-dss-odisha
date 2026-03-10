# Phase 1 — Data Acquisition & Preprocessing

All Phase 1 processing was performed manually in QGIS 3.x.
No automation scripts were used at this stage.

## What Was Done in QGIS
- All 14 datasets downloaded and loaded into QGIS
- Each layer clipped to Odisha state boundary
- All layers reprojected to EPSG:32645 (UTM Zone 45N)
- Reference grid established at 267m × 267m resolution
- All rasters aligned to reference shape (2013 × 2422 pixels)
- Critical fix: DEM slope recalculated after UTM reprojection
  (original WGS84 slope gave impossible 87.94° mean — fixed to 6.51°)

## Data Files Included
- data/infrastructure/ — Roads, rail, substations, transmission lines (OSM)
- data/districts.geojson — 30 Odisha districts (GADM Level 2)

## Data Not Included (Too Large for GitHub)
Raw rasters (.tif) are not uploaded due to file size limits.
See main README for complete dataset table and sources.