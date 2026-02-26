"""
Spectral Index Calculator
==========================
Compute NDVI, NDWI, SAVI, EVI from satellite band GeoTIFFs.
"""

from spectral_index_calculator.calculator import (
    ALL_STRATEGIES,
    BandFileMap,
    EVIStrategy,
    IndexResult,
    IndexStrategy,
    NDVIStrategy,
    NDWIStrategy,
    SAVIStrategy,
    SpectralIndexCalculator,
)

__all__ = [
    "SpectralIndexCalculator",
    "BandFileMap",
    "IndexResult",
    "IndexStrategy",
    "NDVIStrategy",
    "NDWIStrategy",
    "SAVIStrategy",
    "EVIStrategy",
    "ALL_STRATEGIES",
]
