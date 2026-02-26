"""
Tests — Batch Geocoder
=======================
Unit tests for :class:`~src.batch_geocoder.geocoder.BatchGeocoder`,
:class:`~src.batch_geocoder.geocoder.NominatimBackend`, and
:class:`~src.batch_geocoder.geocoder.GoogleBackend`.

All HTTP calls are mocked via the ``responses`` library — no real
network requests are made during testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import responses as rsps_lib

from src.batch_geocoder.geocoder import (
    BatchGeocoder,
    GeocodeResult,
    GoogleBackend,
    NominatimBackend,
)
from shared.python.exceptions import ColumnNotFoundError, InputValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def address_csv(tmp_path: Path) -> Path:
    """Write a small CSV with address, name, and city columns."""
    path = tmp_path / "addresses.csv"
    df = pd.DataFrame(
        {
            "address": ["1600 Pennsylvania Ave NW, Washington DC", "Times Square, New York City"],
            "name": ["White House", "Times Square"],
            "city": ["Washington DC", "New York City"],
        }
    )
    df.to_csv(path, index=False)
    return path


def _nominatim_hit(lon: float, lat: float, display: str) -> dict:
    """Build a mock Nominatim JSON response for one result."""
    return [{"lon": str(lon), "lat": str(lat), "importance": 0.85, "display_name": display}]


def _google_hit(lon: float, lat: float, display: str) -> dict:
    """Build a mock Google Geocoding JSON response."""
    return {
        "status": "OK",
        "results": [
            {
                "formatted_address": display,
                "geometry": {"location": {"lat": lat, "lng": lon}},
            }
        ],
    }


# ---------------------------------------------------------------------------
# GeocodeResult tests
# ---------------------------------------------------------------------------


class TestGeocodeResult:
    def test_successful_result_to_feature(self) -> None:
        r = GeocodeResult(
            address="123 Main St", longitude=-96.8, latitude=32.7,
            confidence=0.9, display_name="123 Main St, Dallas, TX",
            success=True,
        )
        feature = r.to_geojson_feature()
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        assert feature["geometry"]["coordinates"] == [-96.8, 32.7]
        assert feature["properties"]["geocode_success"] is True

    def test_failed_result_has_null_geometry(self) -> None:
        r = GeocodeResult(
            address="bad address", longitude=None, latitude=None,
            confidence=None, display_name=None,
            success=False, error="No results",
        )
        feature = r.to_geojson_feature()
        assert feature["geometry"] is None
        assert feature["properties"]["geocode_success"] is False

    def test_extra_props_included(self) -> None:
        r = GeocodeResult(
            address="123 Main St", longitude=-96.8, latitude=32.7,
            confidence=0.9, display_name="Dallas",
            success=True,
        )
        feature = r.to_geojson_feature(extra_props={"name": "Store A", "zip": "75001"})
        assert feature["properties"]["name"] == "Store A"
        assert feature["properties"]["zip"] == "75001"


# ---------------------------------------------------------------------------
# NominatimBackend tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestNominatimBackend:
    @rsps_lib.activate
    def test_successful_geocode(self) -> None:
        rsps_lib.add(
            rsps_lib.GET,
            "https://nominatim.openstreetmap.org/search",
            json=_nominatim_hit(-77.036, 38.897, "White House, Washington DC"),
            status=200,
        )
        backend = NominatimBackend(user_agent="test/1.0", rate_limit_seconds=0)
        result = backend.geocode_one("1600 Pennsylvania Ave NW, Washington DC")
        assert result.success is True
        assert abs(result.longitude + 77.036) < 0.001  # type: ignore[operator]
        assert abs(result.latitude - 38.897) < 0.001   # type: ignore[operator]

    @rsps_lib.activate
    def test_no_results_returns_failed(self) -> None:
        rsps_lib.add(
            rsps_lib.GET,
            "https://nominatim.openstreetmap.org/search",
            json=[],
            status=200,
        )
        backend = NominatimBackend(user_agent="test/1.0", rate_limit_seconds=0)
        result = backend.geocode_one("zzz-nonexistent-place-xyz")
        assert result.success is False

    @rsps_lib.activate
    def test_rate_limit_raises(self) -> None:
        from shared.python.exceptions import GeocodingRateLimitError

        rsps_lib.add(
            rsps_lib.GET, "https://nominatim.openstreetmap.org/search", status=429
        )
        backend = NominatimBackend(user_agent="test/1.0", rate_limit_seconds=0)
        with pytest.raises(GeocodingRateLimitError):
            backend.geocode_one("any address")


# ---------------------------------------------------------------------------
# BatchGeocoder integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestBatchGeocoder:
    @rsps_lib.activate
    def test_output_geojson_created(self, tmp_path: Path, address_csv: Path) -> None:
        rsps_lib.add(
            rsps_lib.GET, "https://nominatim.openstreetmap.org/search",
            json=_nominatim_hit(-77.036, 38.897, "White House"), status=200,
        )
        rsps_lib.add(
            rsps_lib.GET, "https://nominatim.openstreetmap.org/search",
            json=_nominatim_hit(-73.985, 40.758, "Times Square"), status=200,
        )
        output = tmp_path / "out.geojson"
        backend = NominatimBackend(user_agent="test/1.0", rate_limit_seconds=0)
        BatchGeocoder(address_csv, output, address_col="address", backend=backend).run()
        assert output.exists()

    @rsps_lib.activate
    def test_feature_count_matches_input(self, tmp_path: Path, address_csv: Path) -> None:
        for _ in range(2):
            rsps_lib.add(
                rsps_lib.GET, "https://nominatim.openstreetmap.org/search",
                json=_nominatim_hit(-77.036, 38.897, "Result"), status=200,
            )
        output = tmp_path / "out.geojson"
        backend = NominatimBackend(user_agent="test/1.0", rate_limit_seconds=0)
        tool = BatchGeocoder(address_csv, output, address_col="address", backend=backend)
        tool.run()
        data = json.loads(output.read_text())
        assert len(data["features"]) == 2

    def test_missing_column_raises(self, tmp_path: Path, address_csv: Path) -> None:
        output = tmp_path / "out.geojson"
        backend = NominatimBackend(user_agent="test/1.0", rate_limit_seconds=0)
        tool = BatchGeocoder(
            address_csv, output,
            address_col="nonexistent_col",  # ← bad column
            backend=backend,
        )
        with pytest.raises(ColumnNotFoundError):
            tool.run()

    def test_missing_input_file_raises(self, tmp_path: Path) -> None:
        output = tmp_path / "out.geojson"
        backend = NominatimBackend(user_agent="test/1.0", rate_limit_seconds=0)
        tool = BatchGeocoder(tmp_path / "no_file.csv", output, backend=backend)
        with pytest.raises(InputValidationError):
            tool.run()
