"""Quantum Land-Cover Change Detector.

Multi-sensor fusion with pseudo-quantum classification for
land-cover change detection from 1990 to present at 30 m
aggregate resolution.

Submodules
----------
aoi             Area-of-interest builder (WGS-84 + UTM projection).
acquisition     Multi-epoch STAC data fetcher (Landsat 5/7/8/9, Sentinel-2,
                Sentinel-1 SAR, NAIP, 3DEP DEM).
preprocessing   Cloud masking, spatial alignment, temporal compositing.
feature_engineering  Spectral, SAR, terrain, and texture feature extraction.
quantum_classifier   4-qubit quantum feature encoder + ensemble classifier.
change_detection     Year-to-year transition analysis and change mapping.
accuracy        Confusion matrix, Kappa, NLCD cross-validation.
terrain         DEM / HAND / slope / curvature processing.
sar_processor   Sentinel-1 SAR feature extraction.
viz             Map outputs, charts, Sankey diagrams.
flowchart       Pipeline architecture diagram generator (HTML/PNG/PDF).
constants       Land-cover class definitions, colours, thresholds.
cli             Click command-line interface.
"""

__version__ = "1.0.0"
