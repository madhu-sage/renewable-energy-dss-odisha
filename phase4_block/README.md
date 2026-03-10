# Phase 4 (Block Level) — Block-Level Feature Extraction

Upgraded version of Phase 4 using 314 administrative blocks 
instead of 30 districts. This is the dataset used for ML training.

## Why Block Level?
30 district samples were insufficient for statistically valid 
5-fold cross-validation. Block level gives 10× more samples.

## Contents
- data/blocks.geojson — 314 Odisha block boundaries
- data/top_solar/wind/biomass_zones.geojson — High potential zones
- phase4_block.rar — Complete phase outputs including classified rasters

## Output
block_features.csv — 314 blocks × 7 features (zero missing values)
This file is the direct input to Phase 5 ML pipeline.