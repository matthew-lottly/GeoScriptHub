"""
Batch Geocoder — Core Module
=============================
Converts a CSV of addresses to a GeoJSON FeatureCollection by geocoding
each address through a pluggable backend (Nominatim by default).

Architecture:
    ``GeocoderBackend`` is an abstract strategy — swap backends without
    changing the ``BatchGeocoder`` orchestrator.  Each backend returns a
    ``GeocodeResult`` for every address attempted.

Classes:
    GeocodeResult       Immutable result for one address geocoding attempt.
    GeocoderBackend     Abstract base for geocoding providers.
    NominatimBackend    Free OSM-powered geocoder (no API key required).
    GoogleBackend       Google Maps Geocoding API (requires API key).
    BatchGeocoder       Primary tool class (inherits GeoTool).

Usage::

    from pathlib import Path
    from src.batch_geocoder.geocoder import BatchGeocoder, NominatimBackend

    tool = BatchGeocoder(
        input_path=Path("data/addresses.csv"),
        output_path=Path("output/addresses.geojson"),
        address_col="full_address",
        backend=NominatimBackend(user_agent="my-project/1.0"),
    )
    tool.run()
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from shared.python.base_tool import GeoTool
from shared.python.exceptions import (
    GeocodingError,
    GeocodingRateLimitError,
    InputValidationError,
    OutputWriteError,
)
from shared.python.validators import Validators

logger = logging.getLogger("geoscripthub.batch_geocoder")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeocodeResult:
    """Immutable result for a single address geocoding attempt.

    Attributes:
        address: The original address string that was geocoded.
        longitude: Longitude in WGS84, or ``None`` if geocoding failed.
        latitude: Latitude in WGS84, or ``None`` if geocoding failed.
        confidence: Provider-specific confidence score (0.0–1.0),
                    or ``None`` if not provided.
        display_name: Full formatted address returned by the geocoder,
                      or ``None`` on failure.
        success: ``True`` if the geocoder returned a valid coordinate pair.
        error: Error message if ``success`` is ``False``.
    """

    address: str
    longitude: float | None
    latitude: float | None
    confidence: float | None
    display_name: str | None
    success: bool
    error: str | None = None

    def to_geojson_feature(self, extra_props: dict[str, Any] | None = None) -> dict[str, Any]:
        """Convert this result to a GeoJSON Feature dict.

        Args:
            extra_props: Additional properties to include in the Feature
                         ``properties`` object (e.g. other CSV columns).

        Returns:
            A dict conforming to the GeoJSON Feature spec, or ``None``
            geometry if geocoding failed.
        """
        props: dict[str, Any] = {
            "address": self.address,
            "display_name": self.display_name,
            "confidence": self.confidence,
            "geocode_success": self.success,
        }
        if extra_props:
            props.update(extra_props)

        geometry = (
            {"type": "Point", "coordinates": [self.longitude, self.latitude]}
            if self.success
            else None
        )
        return {"type": "Feature", "geometry": geometry, "properties": props}


# ---------------------------------------------------------------------------
# Backend strategies
# ---------------------------------------------------------------------------


class GeocoderBackend(ABC):
    """Abstract strategy for a geocoding provider.

    Subclass this and implement :meth:`geocode_one` to add a new provider.
    The ``BatchGeocoder`` will call it for each address row.
    """

    @abstractmethod
    def geocode_one(self, address: str) -> GeocodeResult:
        """Geocode a single address string.

        Args:
            address: The full address to geocode.

        Returns:
            A :class:`GeocodeResult` — success flag indicates whether a
            coordinate was returned.

        Raises:
            GeocodingRateLimitError: If the provider returns HTTP 429.
            GeocodingError: For any other provider error.
        """


class NominatimBackend(GeocoderBackend):
    """Geocoder backend powered by OpenStreetMap's Nominatim API.

    **Free to use** — no API key required.  Must comply with the
    Nominatim Usage Policy: include a descriptive ``user_agent`` and
    do not exceed 1 request/second (use ``rate_limit_seconds >= 1.0``).

    Args:
        user_agent: Identifies your application to Nominatim.  Use a
                    descriptive name (e.g. ``"my-company-geocoder/1.0"``).
        rate_limit_seconds: Seconds to sleep between requests.  Must be
                            ``>= 1.0`` to comply with Nominatim usage policy.
        timeout: HTTP request timeout in seconds.

    Reference:
        https://nominatim.org/release-docs/develop/api/Search/
    """

    _BASE_URL = "https://nominatim.openstreetmap.org/search"

    def __init__(
        self,
        user_agent: str = "geoscripthub-geocoder/1.0",
        rate_limit_seconds: float = 1.1,
        timeout: int = 10,
    ) -> None:
        self.user_agent = user_agent
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = self.user_agent

    def geocode_one(self, address: str) -> GeocodeResult:
        """Geocode *address* via Nominatim.

        Args:
            address: The address string to geocode.

        Returns:
            :class:`GeocodeResult` with WGS84 coordinates if found.

        Raises:
            GeocodingRateLimitError: On HTTP 429.
            GeocodingError: On any other HTTP error or parse failure.
        """
        # Respect Nominatim rate-limit policy
        time.sleep(self.rate_limit_seconds)

        params = {"q": address, "format": "json", "limit": 1}
        try:
            response = self._session.get(
                self._BASE_URL, params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            return GeocodeResult(
                address=address,
                longitude=None,
                latitude=None,
                confidence=None,
                display_name=None,
                success=False,
                error=str(exc),
            )

        if response.status_code == 429:
            raise GeocodingRateLimitError("Nominatim", retry_after=60)

        if not response.ok:
            raise GeocodingError(f"Nominatim returned HTTP {response.status_code}")

        data = response.json()
        if not data:
            return GeocodeResult(
                address=address,
                longitude=None,
                latitude=None,
                confidence=None,
                display_name=None,
                success=False,
                error="No results returned.",
            )

        hit = data[0]
        return GeocodeResult(
            address=address,
            longitude=float(hit["lon"]),
            latitude=float(hit["lat"]),
            confidence=float(hit.get("importance", 0.0)),
            display_name=hit.get("display_name"),
            success=True,
        )


class GoogleBackend(GeocoderBackend):
    """Geocoder backend powered by the Google Maps Geocoding API.

    Requires a valid Google Maps API key with the Geocoding API enabled.

    Args:
        api_key: Your Google Maps Geocoding API key.  Never commit this
                 value to version control — use an environment variable
                 or ``.env`` file instead.
        rate_limit_seconds: Seconds to sleep between requests.
        timeout: HTTP request timeout in seconds.

    Reference:
        https://developers.google.com/maps/documentation/geocoding
    """

    _BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(
        self,
        api_key: str,
        rate_limit_seconds: float = 0.05,
        timeout: int = 10,
    ) -> None:
        self.api_key = api_key
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self._session = requests.Session()

    def geocode_one(self, address: str) -> GeocodeResult:
        """Geocode *address* via the Google Maps Geocoding API.

        Args:
            address: The address string to geocode.

        Returns:
            :class:`GeocodeResult` with WGS84 coordinates if found.

        Raises:
            GeocodingRateLimitError: On OVER_QUERY_LIMIT status.
            GeocodingError: On REQUEST_DENIED or other API errors.
        """
        time.sleep(self.rate_limit_seconds)

        params = {"address": address, "key": self.api_key}
        try:
            response = self._session.get(
                self._BASE_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return GeocodeResult(
                address=address,
                longitude=None,
                latitude=None,
                confidence=None,
                display_name=None,
                success=False,
                error=str(exc),
            )

        data = response.json()
        status = data.get("status", "UNKNOWN")

        if status == "OVER_QUERY_LIMIT":
            raise GeocodingRateLimitError("Google")
        if status == "REQUEST_DENIED":
            raise GeocodingError("Google geocoding request denied — check your API key.")
        if status != "OK" or not data.get("results"):
            return GeocodeResult(
                address=address,
                longitude=None,
                latitude=None,
                confidence=None,
                display_name=None,
                success=False,
                error=f"API status: {status}",
            )

        location = data["results"][0]["geometry"]["location"]
        return GeocodeResult(
            address=address,
            longitude=float(location["lng"]),
            latitude=float(location["lat"]),
            confidence=None,  # Google does not return a direct confidence score
            display_name=data["results"][0].get("formatted_address"),
            success=True,
        )


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class BatchGeocoder(GeoTool):
    """Geocode every address in a CSV file and write a GeoJSON output.

    Iterates over each row, calls the configured :class:`GeocoderBackend`
    for the address column, and assembles results into a GeoJSON
    FeatureCollection.  Failed rows are included with ``null`` geometry and
    a ``geocode_success: false`` property so no data is silently lost.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path for the output GeoJSON file.
        address_col: Name of the CSV column containing address strings.
        backend: A :class:`GeocoderBackend` instance.  Defaults to
                 :class:`NominatimBackend`.
        extra_cols: Additional CSV columns to carry through as GeoJSON
                    feature properties.
        verbose: Enable DEBUG-level logging.

    Example::

        from src.batch_geocoder.geocoder import BatchGeocoder, NominatimBackend

        BatchGeocoder(
            input_path=Path("data/stores.csv"),
            output_path=Path("output/stores.geojson"),
            address_col="address",
            backend=NominatimBackend(user_agent="my-app/1.0"),
            extra_cols=["name", "city"],
        ).run()
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        address_col: str = "address",
        backend: GeocoderBackend | None = None,
        extra_cols: list[str] | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__(input_path, output_path, verbose=verbose)
        self.address_col = address_col
        self.backend: GeocoderBackend = backend or NominatimBackend()
        self.extra_cols: list[str] = extra_cols or []

        self._results: list[GeocodeResult] = []

    # ------------------------------------------------------------------
    # GeoTool abstract method implementations
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate the CSV and configuration before geocoding begins.

        Raises:
            InputValidationError: If file is missing, not a CSV, or
                the address column does not exist.
            OutputWriteError: If the output directory cannot be created.
        """
        Validators.assert_file_exists(self.input_path)
        Validators.assert_supported_extension(self.input_path, [".csv"])
        Validators.assert_output_dir_writable(self.output_path)

        df_peek = pd.read_csv(self.input_path, nrows=0)
        required = [self.address_col] + self.extra_cols
        Validators.assert_columns_exist(df_peek, required)
        logger.debug("Inputs validated successfully.")

    def process(self) -> None:
        """Geocode every address row and write the GeoJSON output.

        Iterates over each row, calls the backend, and collects results.
        All rows are included in the output regardless of geocoding success.

        Raises:
            GeocodingRateLimitError: If the provider rate-limits the run.
            OutputWriteError: If writing the output file fails.
        """
        df = pd.read_csv(self.input_path)
        total = len(df)
        logger.info("Starting geocoding of %d addresses via %s...", total, self.backend.__class__.__name__)

        results: list[GeocodeResult] = []
        for i, row in df.iterrows():
            address = str(row[self.address_col])
            logger.debug("[%d/%d] Geocoding: %s", i + 1, total, address)  # type: ignore[operator]
            result = self.backend.geocode_one(address)
            results.append(result)

            if not result.success:
                logger.warning("  ✗ Failed: %s — %s", address, result.error)
            else:
                logger.debug("  ✓ %s → (%.5f, %.5f)", address, result.longitude, result.latitude)

        self._results = results
        self._write_geojson(df, results)

        success_count = sum(1 for r in results if r.success)
        logger.info(
            "Geocoding complete: %d/%d succeeded, %d failed.",
            success_count, total, total - success_count,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_geojson(self, df: pd.DataFrame, results: list[GeocodeResult]) -> None:
        """Write results as a GeoJSON FeatureCollection.

        Args:
            df: The original DataFrame (provides ``extra_cols`` values).
            results: Parallel list of :class:`GeocodeResult` objects.

        Raises:
            OutputWriteError: If the file cannot be written.
        """
        features = []
        for result, (_, row) in zip(results, df.iterrows()):
            extra = {col: row[col] for col in self.extra_cols if col in row.index}
            features.append(result.to_geojson_feature(extra_props=extra))

        geojson: dict[str, Any] = {
            "type": "FeatureCollection",
            "features": features,
        }

        try:
            with open(self.output_path, "w", encoding="utf-8") as fh:
                json.dump(geojson, fh, indent=2, default=str)
        except OSError as exc:
            raise OutputWriteError(str(self.output_path), str(exc)) from exc

    @property
    def results(self) -> list[GeocodeResult]:
        """All :class:`GeocodeResult` objects from the last run, or ``[]``."""
        return self._results
