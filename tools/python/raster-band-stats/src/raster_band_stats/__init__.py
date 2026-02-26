"""
Raster Band Stats Reporter
===========================
A GeoScriptHub tool for computing per-band statistics of any rasterio-supported
raster file.

Public API::

    from src.raster_band_stats import BandStatsReporter, BandStatsConfig, BandStats
"""

from src.raster_band_stats.stats import BandStats, BandStatsConfig, BandStatsReporter

__all__ = ["BandStatsReporter", "BandStatsConfig", "BandStats"]
__version__ = "1.0.0"
