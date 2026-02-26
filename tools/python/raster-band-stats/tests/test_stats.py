"""
Tests — Raster Band Stats Reporter
=====================================
Unit tests for :class:`~src.raster_band_stats.stats.BandStatsReporter`.

Uses ``rasterio.MemoryFile`` to create in-memory test rasters rather than
requiring real GeoTIFF files on disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.raster_band_stats.stats import BandStats, BandStatsConfig, BandStatsReporter
from shared.python.exceptions import BandIndexError, InputValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_geotiff(
    tmp_path: Path,
    filename: str = "test.tif",
    bands: int = 3,
    width: int = 10,
    height: int = 10,
    nodata: float | None = 0.0,
) -> Path:
    """Create a small synthetic GeoTIFF and return its path."""
    path = tmp_path / filename
    transform = from_bounds(0, 0, 1, 1, width, height)
    data = np.arange(1, bands * width * height + 1, dtype=np.float32).reshape(bands, height, width)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# Unit tests — BandStats
# ---------------------------------------------------------------------------


class TestBandStats:
    def test_str_representation(self) -> None:
        s = BandStats(
            band_index=1, min=0.0, max=255.0, mean=127.5, std_dev=73.6,
            valid_pixels=100, nodata_pixels=0, nodata_value=None,
        )
        assert "Band 1" in str(s)
        assert "127.5000" in str(s)


# ---------------------------------------------------------------------------
# Integration tests — BandStatsReporter
# ---------------------------------------------------------------------------


class TestBandStatsReporter:
    def test_json_output_created(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path)
        output = tmp_path / "stats.json"
        BandStatsReporter(tif, output).run()
        assert output.exists()

    def test_json_has_correct_band_count(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path, bands=4)
        output = tmp_path / "stats.json"
        BandStatsReporter(tif, output).run()
        data = json.loads(output.read_text())
        assert len(data["bands"]) == 4

    def test_csv_output_created(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path)
        output = tmp_path / "stats.csv"
        cfg = BandStatsConfig(output_format="csv")
        BandStatsReporter(tif, output, cfg).run()
        assert output.exists()
        lines = output.read_text().splitlines()
        assert lines[0].startswith("band_index")  # header row
        assert len(lines) == 4  # 1 header + 3 bands

    def test_specific_bands_processed(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path, bands=4)
        output = tmp_path / "stats.json"
        cfg = BandStatsConfig(bands=[1, 4])
        tool = BandStatsReporter(tif, output, cfg)
        tool.run()
        assert len(tool.band_stats) == 2
        assert tool.band_stats[0].band_index == 1
        assert tool.band_stats[1].band_index == 4

    def test_min_lt_mean_lt_max(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path, bands=1, nodata=None)
        output = tmp_path / "stats.json"
        tool = BandStatsReporter(tif, output)
        tool.run()
        s = tool.band_stats[0]
        assert s.min <= s.mean <= s.max

    def test_std_dev_is_positive(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path, bands=1, nodata=None)
        output = tmp_path / "stats.json"
        tool = BandStatsReporter(tif, output)
        tool.run()
        assert tool.band_stats[0].std_dev >= 0

    def test_invalid_band_index_raises(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path, bands=2)
        output = tmp_path / "stats.json"
        cfg = BandStatsConfig(bands=[5])  # ← bands 1-2 only exist
        tool = BandStatsReporter(tif, output, cfg)
        with pytest.raises(BandIndexError):
            tool.run()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        tool = BandStatsReporter(tmp_path / "no_file.tif", tmp_path / "out.json")
        with pytest.raises(InputValidationError):
            tool.run()

    def test_band_stats_property_empty_before_run(self, tmp_path: Path) -> None:
        tif = _create_geotiff(tmp_path)
        tool = BandStatsReporter(tif, tmp_path / "stats.json")
        assert tool.band_stats == []
