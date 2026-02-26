"""
Batch Geocoder
===============
A GeoScriptHub tool for converting CSV addresses to GeoJSON points via
configurable geocoding backends.

Public API::

    from src.batch_geocoder import BatchGeocoder, NominatimBackend, GoogleBackend
"""

from src.batch_geocoder.geocoder import (
    BatchGeocoder,
    GeocodeResult,
    GeocoderBackend,
    GoogleBackend,
    NominatimBackend,
)

__all__ = [
    "BatchGeocoder",
    "GeocodeResult",
    "GeocoderBackend",
    "NominatimBackend",
    "GoogleBackend",
]
__version__ = "1.0.0"
