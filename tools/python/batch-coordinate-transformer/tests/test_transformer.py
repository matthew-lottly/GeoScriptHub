"""
Tests — Batch Coordinate Transformer
======================================
Unit tests for :class:`~src.batch_coord_transformer.transformer.CoordinateTransformer`.

Test strategy:
- Build minimal in-memory CSV inputs using ``tmp_path`` fixtures.
- Assert transformed coordinate values are within an acceptable tolerance.
- Assert that validation errors are raised for bad inputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.batch_coord_transformer.transformer import CoordinateTransformer, TransformerConfig
from shared.python.exceptions import CRSError, ColumnNotFoundError, InputValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def utm_csv(tmp_path: Path) -> Path:
    """Write a small CSV with UTM Zone 14N coordinates and return the path.

    Coordinates are real-world points near Dallas, TX (EPSG:32614).
    """
    csv_path = tmp_path / "points_utm.csv"
    df = pd.DataFrame(
        {
            "easting": [730_000.0, 731_000.0, 732_000.0],
            "northing": [3_634_000.0, 3_635_000.0, 3_636_000.0],
            "name": ["A", "B", "C"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def utm_config() -> TransformerConfig:
    """Return a TransformerConfig for UTM Zone 14N → WGS84."""
    return TransformerConfig(
        from_crs="EPSG:32614",
        to_crs="EPSG:4326",
        lon_col="easting",
        lat_col="northing",
        output_format="csv",
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestCoordinateTransformerHappyPath:
    """Tests for successful transformation scenarios."""

    def test_csv_output_created(
        self, tmp_path: Path, utm_csv: Path, utm_config: TransformerConfig
    ) -> None:
        """Output CSV file should be created at the specified path."""
        output = tmp_path / "output.csv"
        CoordinateTransformer(utm_csv, output, utm_config).run()
        assert output.exists(), "Output CSV was not created."

    def test_csv_row_count_preserved(
        self, tmp_path: Path, utm_csv: Path, utm_config: TransformerConfig
    ) -> None:
        """Output should contain the same number of rows as the input."""
        output = tmp_path / "output.csv"
        CoordinateTransformer(utm_csv, output, utm_config).run()
        result_df = pd.read_csv(output)
        assert len(result_df) == 3

    def test_wgs84_longitude_range(
        self, tmp_path: Path, utm_csv: Path, utm_config: TransformerConfig
    ) -> None:
        """Transformed longitudes should be in WGS84 range [-180, 180]."""
        output = tmp_path / "output.csv"
        CoordinateTransformer(utm_csv, output, utm_config).run()
        df = pd.read_csv(output)
        assert df["easting"].between(-180, 180).all()

    def test_wgs84_latitude_range(
        self, tmp_path: Path, utm_csv: Path, utm_config: TransformerConfig
    ) -> None:
        """Transformed latitudes should be in WGS84 range [-90, 90]."""
        output = tmp_path / "output.csv"
        CoordinateTransformer(utm_csv, output, utm_config).run()
        df = pd.read_csv(output)
        assert df["northing"].between(-90, 90).all()

    def test_geojson_output(
        self, tmp_path: Path, utm_csv: Path, utm_config: TransformerConfig
    ) -> None:
        """GeoJSON output should contain a valid FeatureCollection."""
        import json

        cfg = TransformerConfig(
            from_crs="EPSG:32614",
            to_crs="EPSG:4326",
            lon_col="easting",
            lat_col="northing",
            output_format="geojson",
        )
        output = tmp_path / "output.geojson"
        CoordinateTransformer(utm_csv, output, cfg).run()

        with open(output) as fh:
            geo = json.load(fh)

        assert geo["type"] == "FeatureCollection"
        assert len(geo["features"]) == 3

    def test_result_object_populated(
        self, tmp_path: Path, utm_csv: Path, utm_config: TransformerConfig
    ) -> None:
        """``tool.result`` should be a populated TransformResult after run()."""
        output = tmp_path / "output.csv"
        tool = CoordinateTransformer(utm_csv, output, utm_config)
        assert tool.result is None  # not yet run
        tool.run()
        assert tool.result is not None
        assert tool.result.rows_processed == 3
        assert tool.result.rows_skipped == 0

    def test_null_rows_skipped(self, tmp_path: Path) -> None:
        """Rows with null coordinates should be skipped, not crash the tool."""
        csv_path = tmp_path / "with_nulls.csv"
        df = pd.DataFrame({"longitude": [1.0, None, 3.0], "latitude": [10.0, 20.0, None]})
        df.to_csv(csv_path, index=False)

        cfg = TransformerConfig(from_crs="EPSG:4326", to_crs="EPSG:3857")
        output = tmp_path / "output.csv"
        tool = CoordinateTransformer(csv_path, output, cfg)
        tool.run()

        assert tool.result is not None
        assert tool.result.rows_skipped == 2
        assert tool.result.rows_processed == 1


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestCoordinateTransformerValidation:
    """Tests for input validation error handling."""

    def test_missing_input_file_raises(self, tmp_path: Path) -> None:
        """A non-existent input path should raise InputValidationError."""
        cfg = TransformerConfig(from_crs="EPSG:4326", to_crs="EPSG:3857")
        tool = CoordinateTransformer(
            tmp_path / "does_not_exist.csv",
            tmp_path / "out.csv",
            cfg,
        )
        with pytest.raises(InputValidationError):
            tool.run()

    def test_invalid_from_crs_raises(self, tmp_path: Path, utm_csv: Path) -> None:
        """An invalid from_crs string should raise CRSError."""
        cfg = TransformerConfig(
            from_crs="EPSG:99999",  # ← intentionally invalid
            to_crs="EPSG:4326",
        )
        tool = CoordinateTransformer(utm_csv, tmp_path / "out.csv", cfg)
        with pytest.raises(CRSError):
            tool.run()

    def test_missing_coordinate_column_raises(
        self, tmp_path: Path, utm_csv: Path
    ) -> None:
        """Specifying a non-existent column name should raise ColumnNotFoundError."""
        cfg = TransformerConfig(
            from_crs="EPSG:32614",
            to_crs="EPSG:4326",
            lon_col="x_coord",  # ← column does not exist in utm_csv
            lat_col="northing",
        )
        tool = CoordinateTransformer(utm_csv, tmp_path / "out.csv", cfg)
        with pytest.raises(ColumnNotFoundError):
            tool.run()
