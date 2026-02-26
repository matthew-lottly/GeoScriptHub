"""
GeoScriptHub — Shared Python Package
=====================================
Re-exports the shared base class, exception hierarchy, and validator
utilities so individual tools can import from a single location::

    from shared.python import GeoTool, Validators
    from shared.python.exceptions import CRSError
"""

from shared.python.base_tool import GeoTool
from shared.python.exceptions import (
    BandIndexError,
    ColumnNotFoundError,
    CRSError,
    GeocodingError,
    GeocodingRateLimitError,
    GeoScriptHubError,
    InputValidationError,
    OutputWriteError,
    RasterError,
    SpectralIndexError,
)
from shared.python.validators import Validators

__all__ = [
    "GeoTool",
    "Validators",
    "GeoScriptHubError",
    "InputValidationError",
    "ColumnNotFoundError",
    "CRSError",
    "RasterError",
    "BandIndexError",
    "GeocodingError",
    "GeocodingRateLimitError",
    "SpectralIndexError",
    "OutputWriteError",
]
