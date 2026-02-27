"""
sub_canopy_detector
===================
Python port of the Sub-Canopy Structure Detector GEE script.

Pulls Sentinel-1 RTC and Sentinel-2 L2A imagery from Microsoft
Planetary Computer and runs the same weighted-fusion pipeline
entirely in-process using xarray, rioxarray, scipy, and geopandas.

Submodules
----------
aoi        -- Define AOI from shapefile, FGDB, bounding box, or geocode
fetcher    -- Query and stack imagery from Planetary Computer
analysis   -- Core detection algorithm (SAR + optical fusion)
export     -- Save GeoTIFF, GeoJSON/Shapefile, CSV, and PNG outputs
viz        -- Interactive folium map and matplotlib summary charts
"""

from .aoi import AOIBuilder
from .fetcher import ImageryFetcher
from .analysis import SubCanopyAnalyser
from .export import OutputWriter
from .viz import ResultVisualiser

__version__ = "1.0.0"
__all__ = [
    "AOIBuilder",
    "ImageryFetcher",
    "SubCanopyAnalyser",
    "OutputWriter",
    "ResultVisualiser",
]
