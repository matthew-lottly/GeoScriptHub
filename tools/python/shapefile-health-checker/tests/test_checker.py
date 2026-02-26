"""
Tests — Shapefile Health Checker
==================================
Unit tests for :class:`~src.shapefile_health_checker.checker.ShapefileHealthChecker`
and each individual :class:`~src.shapefile_health_checker.checker.CheckStrategy`.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from src.shapefile_health_checker.checker import (
    CheckStatus,
    CRSPresenceCheck,
    DEFAULT_CHECKS,
    DuplicateFeaturesCheck,
    ExtentSanityCheck,
    NullGeometryCheck,
    SelfIntersectionCheck,
    ShapefileHealthChecker,
)
from shared.python.exceptions import InputValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_geojson(tmp_path: Path) -> Path:
    """Write a valid GeoJSON with three clean point features."""
    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"]},
        geometry=[Point(1, 1), Point(2, 2), Point(3, 3)],
        crs="EPSG:4326",
    )
    path = tmp_path / "clean.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture()
def clean_gdf() -> gpd.GeoDataFrame:
    """Return a clean in-memory GeoDataFrame for unit-testing individual checks."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[Point(10, 20), Point(30, 40), Point(50, 60)],
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Individual check unit tests
# ---------------------------------------------------------------------------


class TestNullGeometryCheck:
    def test_passes_with_no_nulls(self, clean_gdf: gpd.GeoDataFrame) -> None:
        result = NullGeometryCheck().run(clean_gdf)
        assert result.status == CheckStatus.PASSED

    def test_fails_with_null_geometry(self, clean_gdf: gpd.GeoDataFrame) -> None:
        clean_gdf.at[0, "geometry"] = None
        result = NullGeometryCheck().run(clean_gdf)
        assert result.status == CheckStatus.FAILED
        assert 0 in result.affected_rows


class TestSelfIntersectionCheck:
    def test_passes_with_valid_polygon(self, clean_gdf: gpd.GeoDataFrame) -> None:
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        result = SelfIntersectionCheck().run(gdf)
        assert result.status == CheckStatus.PASSED

    def test_fails_with_bowtie_polygon(self) -> None:
        # A bowtie (figure-8) polygon is self-intersecting and invalid
        bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[bowtie], crs="EPSG:4326")
        result = SelfIntersectionCheck().run(gdf)
        assert result.status == CheckStatus.FAILED


class TestDuplicateFeaturesCheck:
    def test_passes_with_unique_geometries(self, clean_gdf: gpd.GeoDataFrame) -> None:
        result = DuplicateFeaturesCheck().run(clean_gdf)
        assert result.status == CheckStatus.PASSED

    def test_warns_with_duplicates(self, clean_gdf: gpd.GeoDataFrame) -> None:
        # Add a duplicate of row 0
        dup = clean_gdf.iloc[[0]].copy()
        gdf = gpd.GeoDataFrame(
            gpd.pd.concat([clean_gdf, dup], ignore_index=True),
            crs="EPSG:4326",
        )
        result = DuplicateFeaturesCheck().run(gdf)
        assert result.status == CheckStatus.WARNING


class TestCRSPresenceCheck:
    def test_passes_with_crs(self, clean_gdf: gpd.GeoDataFrame) -> None:
        result = CRSPresenceCheck().run(clean_gdf)
        assert result.status == CheckStatus.PASSED

    def test_fails_without_crs(self, clean_gdf: gpd.GeoDataFrame) -> None:
        gdf = clean_gdf.set_crs(None, allow_override=True)
        result = CRSPresenceCheck().run(gdf)
        assert result.status == CheckStatus.FAILED


class TestExtentSanityCheck:
    def test_passes_within_bounds(self, clean_gdf: gpd.GeoDataFrame) -> None:
        result = ExtentSanityCheck().run(clean_gdf)
        assert result.status == CheckStatus.PASSED

    def test_fails_outside_bounds(self) -> None:
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(200, 95)],  # outside world bounds
            crs="EPSG:4326",
        )
        result = ExtentSanityCheck().run(gdf)
        assert result.status == CheckStatus.FAILED


# ---------------------------------------------------------------------------
# Integration tests — ShapefileHealthChecker.run()
# ---------------------------------------------------------------------------


class TestShapefileHealthChecker:
    def test_report_created(self, tmp_path: Path, clean_geojson: Path) -> None:
        output = tmp_path / "report.md"
        ShapefileHealthChecker(clean_geojson, output).run()
        assert output.exists()

    def test_report_has_all_checks(self, tmp_path: Path, clean_geojson: Path) -> None:
        output = tmp_path / "report.md"
        tool = ShapefileHealthChecker(clean_geojson, output)
        tool.run()
        assert tool.report is not None
        assert len(tool.report.results) == len(DEFAULT_CHECKS)

    def test_clean_file_passes_all(self, tmp_path: Path, clean_geojson: Path) -> None:
        output = tmp_path / "report.md"
        tool = ShapefileHealthChecker(clean_geojson, output)
        tool.run()
        assert tool.report is not None
        assert tool.report.overall_status == CheckStatus.PASSED

    def test_html_report_created(self, tmp_path: Path, clean_geojson: Path) -> None:
        output = tmp_path / "report.html"
        ShapefileHealthChecker(clean_geojson, output, report_format="html").run()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content
        assert "FeatureCollection" not in content  # raw GeoJSON should not leak

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InputValidationError):
            ShapefileHealthChecker(
                tmp_path / "no_file.shp", tmp_path / "report.md"
            ).run()
