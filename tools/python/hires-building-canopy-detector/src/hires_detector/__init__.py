"""
hires_detector
==============
High-resolution building footprint + tree canopy detection using
Capella Space X-band SAR (0.5 m) and NAIP optical imagery (0.6 m).

Key capabilities
----------------
* **Building footprints** — Morphological Building Index + local contrast
  from SAR, fused with optical NDVI masking, regularised with MRR.
* **Tree canopy mapping** — NDVI-based segmentation with individual
  tree crown delineation via marker-controlled watershed.
* **Tree species grouping** — Unsupervised spectral clustering on
  per-crown NAIP (R, G, B, NIR) features, labelled as conifer /
  deciduous broadleaf / mixed groups.
"""

__version__ = "1.0.0"

from .aoi import AOIBuilder, AOIResult
from .fetcher import HiResImageryFetcher, HiResImageryData
from .analysis import HiResAnalyser, HiResResult
from .export import HiResOutputWriter

__all__ = [
    "AOIBuilder",
    "AOIResult",
    "HiResImageryFetcher",
    "HiResImageryData",
    "HiResAnalyser",
    "HiResResult",
    "HiResOutputWriter",
]
