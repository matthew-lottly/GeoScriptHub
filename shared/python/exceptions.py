"""
GeoScriptHub — Custom Exception Hierarchy
==========================================
All GeoScriptHub tools raise exceptions from this module so callers can
catch them at the right level of granularity.

Hierarchy::

    GeoScriptHubError                    ← catch-all base
    ├── InputValidationError             ← bad files, missing columns, etc.
    │   ├── FileNotFoundError            ← re-exported Python built-in alias
    │   └── ColumnNotFoundError          ← CSV/table column missing
    ├── CRSError                         ← invalid / unknown CRS string
    ├── RasterError                      ← rasterio / numpy raster issues
    │   └── BandIndexError               ← requested band does not exist
    ├── GeocodingError                   ← geocoder API / parse failures
    │   └── GeocodingRateLimitError      ← API rate limit exceeded
    ├── SpectralIndexError               ← unsupported index or bad bands
    └── OutputWriteError                 ← cannot write to output path

Usage::

    from shared.python.exceptions import CRSError

    raise CRSError(f"Unrecognised CRS string: {crs!r}")
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class GeoScriptHubError(Exception):
    """Base exception for all GeoScriptHub tools.

    Catch this to handle any tool-specific error without caring about
    the exact subtype.

    Args:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class InputValidationError(GeoScriptHubError):
    """Raised when a tool's inputs fail pre-processing validation.

    This is the parent class for more specific input problems.
    """


class ColumnNotFoundError(InputValidationError):
    """Raised when an expected column is absent from a tabular dataset.

    Args:
        column: The name of the missing column.
        available: List of column names that ARE present, used to
                   generate a helpful error message.

    Example::

        raise ColumnNotFoundError("latitude", df.columns.tolist())
    """

    def __init__(self, column: str, available: list[str]) -> None:
        available_str = ", ".join(f"'{c}'" for c in available)
        super().__init__(
            f"Column '{column}' not found. Available columns: {available_str}"
        )
        self.column: str = column
        self.available: list[str] = available


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------


class CRSError(GeoScriptHubError):
    """Raised when a coordinate reference system string cannot be parsed
    or matched to a known CRS.

    Args:
        crs_string: The raw CRS string that caused the error
                    (e.g. ``"EPSG:99999"``).

    Example::

        raise CRSError("EPSG:99999")
    """

    def __init__(self, crs_string: str) -> None:
        super().__init__(
            f"Invalid or unrecognised CRS: '{crs_string}'. "
            "Use an EPSG code (e.g. 'EPSG:4326') or a valid WKT/PROJ string."
        )
        self.crs_string: str = crs_string


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------


class RasterError(GeoScriptHubError):
    """Raised for general raster processing failures (rasterio / numpy).

    Subclass this for more specific raster errors.
    """


class BandIndexError(RasterError):
    """Raised when a requested raster band index does not exist.

    Args:
        band_index: The 1-based band number that was requested.
        total_bands: Total number of bands in the raster file.

    Example::

        raise BandIndexError(band_index=5, total_bands=4)
    """

    def __init__(self, band_index: int, total_bands: int) -> None:
        super().__init__(
            f"Band {band_index} does not exist. "
            f"This raster has {total_bands} band(s) (1-indexed)."
        )
        self.band_index: int = band_index
        self.total_bands: int = total_bands


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------


class GeocodingError(GeoScriptHubError):
    """Raised when a geocoding operation fails for any reason.

    Subclass this for provider-specific errors.
    """


class GeocodingRateLimitError(GeocodingError):
    """Raised when the geocoding provider returns a rate-limit response.

    Args:
        provider: Name of the geocoding service (e.g. ``"Nominatim"``).
        retry_after: Suggested seconds to wait before retrying, if
                     provided by the API.  ``None`` if unknown.

    Example::

        raise GeocodingRateLimitError("Nominatim", retry_after=60)
    """

    def __init__(self, provider: str, retry_after: int | None = None) -> None:
        hint = f" Retry after {retry_after}s." if retry_after else ""
        super().__init__(f"Rate limit exceeded for provider '{provider}'.{hint}")
        self.provider: str = provider
        self.retry_after: int | None = retry_after


# ---------------------------------------------------------------------------
# Spectral index
# ---------------------------------------------------------------------------


class SpectralIndexError(GeoScriptHubError):
    """Raised when a spectral index cannot be calculated.

    Common causes: unsupported index name, missing required band file,
    or mismatched band raster dimensions.

    Args:
        index_name: The name of the index that failed
                    (e.g. ``"NDVI"``, ``"EVI"``).
        reason: Short explanation of why calculation failed.

    Example::

        raise SpectralIndexError("NDVI", "NIR band file not provided")
    """

    def __init__(self, index_name: str, reason: str) -> None:
        super().__init__(f"Cannot calculate {index_name}: {reason}")
        self.index_name: str = index_name
        self.reason: str = reason


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class OutputWriteError(GeoScriptHubError):
    """Raised when the tool cannot write its output to disk.

    Args:
        output_path: String representation of the path that failed.
        reason: Underlying OS or library error message.

    Example::

        raise OutputWriteError("/read-only/dir/out.geojson", "Permission denied")
    """

    def __init__(self, output_path: str, reason: str) -> None:
        super().__init__(
            f"Failed to write output to '{output_path}': {reason}"
        )
        self.output_path: str = output_path
        self.reason: str = reason
