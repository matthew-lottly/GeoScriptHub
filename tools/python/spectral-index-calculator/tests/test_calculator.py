"""
Tests for Spectral Index Calculator
=====================================
All raster I/O uses temporary GeoTIFFs created with rasterio so no real
satellite imagery is required.

Test classes:
    TestIndexStrategies      Formula correctness for each strategy.
    TestSpectralIndexCalculatorHappyPath   End-to-end output file assertions.
    TestSpectralIndexCalculatorValidation  Error conditions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from spectral_index_calculator.calculator import (
    ALL_STRATEGIES,
    BandFileMap,
    EVIStrategy,
    NDVIStrategy,
    NDWIStrategy,
    SAVIStrategy,
    SpectralIndexCalculator,
)
from shared.python.exceptions import SpectralIndexError, InputValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_band(tmp_path: Path, name: str, values: npt.ArrayLike, dtype: str = "float32") -> Path:
    """Write a 4×4 single-band GeoTIFF with the given constant or array values.

    Args:
        tmp_path: Pytest temporary directory.
        name: File base name (without extension).
        values: Value or 2-D array to fill the raster with.
        dtype: numpy dtype string for the raster.

    Returns:
        Path to the created GeoTIFF.
    """
    arr = np.full((4, 4), values, dtype=dtype) if np.isscalar(values) else np.array(values, dtype=dtype)
    fpath = tmp_path / f"{name}.tif"
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "crs": CRS.from_epsg(4326),
        "transform": from_bounds(0, 0, 1, 1, arr.shape[1], arr.shape[0]),
    }
    with rasterio.open(fpath, "w", **profile) as dst:
        dst.write(arr, 1)
    return fpath


# ---------------------------------------------------------------------------
# Strategy formula tests
# ---------------------------------------------------------------------------

class TestIndexStrategies:
    """Unit tests for each IndexStrategy formula using known scalar inputs."""

    def _as_dict(self, **kwargs: float) -> dict[str, npt.NDArray[np.float32]]:
        """Build dummy band dict from scalar values, shape (1, 1)."""
        return {k: np.array([[v]], dtype=np.float32) for k, v in kwargs.items()}

    def test_ndvi_typical_vegetation(self) -> None:
        """NDVI = (0.5 - 0.1) / (0.5 + 0.1) ≈ 0.6667."""
        bands = self._as_dict(nir=0.5, red=0.1)
        result = NDVIStrategy().compute(bands)
        assert pytest.approx(result[0, 0], rel=1e-4) == (0.5 - 0.1) / (0.5 + 0.1)

    def test_ndvi_negative_water(self) -> None:
        """NDVI = (0.1 - 0.5) / (0.1 + 0.5) ≈ -0.6667 for water."""
        bands = self._as_dict(nir=0.1, red=0.5)
        result = NDVIStrategy().compute(bands)
        assert pytest.approx(result[0, 0], rel=1e-4) == (0.1 - 0.5) / (0.1 + 0.5)

    def test_ndvi_zero_denominator_returns_nodata(self) -> None:
        """NDVI with NIR == Red == 0 should return -9999."""
        bands = self._as_dict(nir=0.0, red=0.0)
        result = NDVIStrategy().compute(bands)
        assert result[0, 0] == pytest.approx(-9999.0)

    def test_ndwi_positive_water(self) -> None:
        """NDWI = (0.4 - 0.1) / (0.4 + 0.1) = 0.6 for open water."""
        bands = self._as_dict(green=0.4, nir=0.1)
        result = NDWIStrategy().compute(bands)
        assert pytest.approx(result[0, 0], rel=1e-4) == (0.4 - 0.1) / (0.4 + 0.1)

    def test_savi_default_l(self) -> None:
        """SAVI with default L=0.5: ((0.5-0.2)/(0.5+0.2+0.5)) * 1.5."""
        nir, red, L = 0.5, 0.2, 0.5
        expected = ((nir - red) / (nir + red + L)) * (1 + L)
        bands = self._as_dict(nir=nir, red=red)
        result = SAVIStrategy(soil_factor=L).compute(bands)
        assert pytest.approx(result[0, 0], rel=1e-4) == expected

    def test_savi_custom_l(self) -> None:
        """SAVI with custom L=0.25 should differ from default."""
        nir, red, L = 0.5, 0.2, 0.25
        expected = ((nir - red) / (nir + red + L)) * (1 + L)
        bands = self._as_dict(nir=nir, red=red)
        result = SAVIStrategy(soil_factor=L).compute(bands)
        assert pytest.approx(result[0, 0], rel=1e-4) == expected

    def test_evi_typical(self) -> None:
        """EVI = 2.5 * (0.5 - 0.1) / (0.5 + 6*0.1 - 7.5*0.02 + 1)."""
        nir, red, blue = 0.5, 0.1, 0.02
        expected = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        bands = self._as_dict(nir=nir, red=red, blue=blue)
        result = EVIStrategy().compute(bands)
        assert pytest.approx(result[0, 0], rel=1e-4) == expected


# ---------------------------------------------------------------------------
# Happy-path end-to-end tests
# ---------------------------------------------------------------------------

class TestSpectralIndexCalculatorHappyPath:
    """Integration tests that run the full tool pipeline."""

    def test_ndvi_output_file_created(self, tmp_path: Path) -> None:
        """Running NDVI should produce NDVI.tif in output_dir."""
        red = _make_band(tmp_path, "red", 0.2)
        nir = _make_band(tmp_path, "nir", 0.6)
        out_dir = tmp_path / "out"

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir),  # type: ignore[misc]
            output_dir=out_dir,
            strategies=[NDVIStrategy()],
        )
        tool.run()

        assert (out_dir / "NDVI.tif").exists()

    def test_ndvi_pixel_values_correct(self, tmp_path: Path) -> None:
        """NDVI pixel values in output raster should match expected formula."""
        red_val, nir_val = 0.2, 0.6
        red = _make_band(tmp_path, "red", red_val)
        nir = _make_band(tmp_path, "nir", nir_val)
        out_dir = tmp_path / "out"

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir),  # type: ignore[misc]
            output_dir=out_dir,
            strategies=[NDVIStrategy()],
        )
        tool.run()

        expected = (nir_val - red_val) / (nir_val + red_val)
        with rasterio.open(out_dir / "NDVI.tif") as src:
            arr = src.read(1)
        assert pytest.approx(float(arr.mean()), rel=1e-4) == expected

    def test_multiple_indices_create_multiple_files(self, tmp_path: Path) -> None:
        """Running NDVI + NDWI should write two output files."""
        red = _make_band(tmp_path, "red", 0.2)
        nir = _make_band(tmp_path, "nir", 0.6)
        green = _make_band(tmp_path, "green", 0.3)
        out_dir = tmp_path / "out"

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir, green=green),  # type: ignore[misc]
            output_dir=out_dir,
            strategies=[NDVIStrategy(), NDWIStrategy()],
        )
        tool.run()

        assert (out_dir / "NDVI.tif").exists()
        assert (out_dir / "NDWI.tif").exists()

    def test_results_list_populated(self, tmp_path: Path) -> None:
        """tool.results should hold one IndexResult per strategy."""
        red = _make_band(tmp_path, "red", 0.2)
        nir = _make_band(tmp_path, "nir", 0.6)
        out_dir = tmp_path / "out"

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir),  # type: ignore[misc]
            output_dir=out_dir,
            strategies=[NDVIStrategy()],
        )
        tool.run()

        assert len(tool.results) == 1
        result = tool.results[0]
        assert result.index_name == "NDVI"
        assert -1.0 <= result.min_value <= 1.0
        assert -1.0 <= result.max_value <= 1.0

    def test_output_dir_created_automatically(self, tmp_path: Path) -> None:
        """output_dir should be created if it does not yet exist."""
        red = _make_band(tmp_path, "red", 0.2)
        nir = _make_band(tmp_path, "nir", 0.6)
        out_dir = tmp_path / "does" / "not" / "exist"

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir),  # type: ignore[misc]
            output_dir=out_dir,
            strategies=[NDVIStrategy()],
        )
        tool.run()

        assert out_dir.is_dir()

    def test_evi_requires_blue_band(self, tmp_path: Path) -> None:
        """EVI run with all required bands should succeed."""
        red = _make_band(tmp_path, "red", 0.1)
        nir = _make_band(tmp_path, "nir", 0.5)
        blue = _make_band(tmp_path, "blue", 0.02)
        out_dir = tmp_path / "out"

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir, blue=blue),  # type: ignore[misc]
            output_dir=out_dir,
            strategies=[EVIStrategy()],
        )
        tool.run()

        assert (out_dir / "EVI.tif").exists()

    def test_savi_soil_factor_custom(self, tmp_path: Path) -> None:
        """SAVI output should differ when soil_factor=0.25 vs default 0.5."""
        red = _make_band(tmp_path, "red", 0.2)
        nir = _make_band(tmp_path, "nir", 0.6)

        out_default = tmp_path / "savi_default"
        out_custom = tmp_path / "savi_custom"

        SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir),  # type: ignore[misc]
            output_dir=out_default,
            strategies=[SAVIStrategy(soil_factor=0.5)],
        ).run()

        SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir),  # type: ignore[misc]
            output_dir=out_custom,
            strategies=[SAVIStrategy(soil_factor=0.25)],
        ).run()

        with rasterio.open(out_default / "SAVI.tif") as src:
            val_default = float(src.read(1).mean())
        with rasterio.open(out_custom / "SAVI.tif") as src:
            val_custom = float(src.read(1).mean())

        assert val_default != pytest.approx(val_custom, rel=1e-4)


# ---------------------------------------------------------------------------
# Validation / error-path tests
# ---------------------------------------------------------------------------

class TestSpectralIndexCalculatorValidation:
    """Tests for validation errors raised before processing begins."""

    def test_missing_required_band_raises(self, tmp_path: Path) -> None:
        """EVI without blue band should raise SpectralIndexError."""
        red = _make_band(tmp_path, "red", 0.1)
        nir = _make_band(tmp_path, "nir", 0.5)
        # blue is intentionally omitted

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red, nir=nir),  # type: ignore[misc]
            output_dir=tmp_path / "out",
            strategies=[EVIStrategy()],
        )
        with pytest.raises(SpectralIndexError, match="blue"):
            tool.run()

    def test_nonexistent_band_file_raises(self, tmp_path: Path) -> None:
        """Pointing to a non-existent file should raise InputValidationError."""
        not_a_file = tmp_path / "ghost.tif"

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(nir=not_a_file, red=not_a_file),  # type: ignore[misc]
            output_dir=tmp_path / "out",
            strategies=[NDVIStrategy()],
        )
        with pytest.raises(InputValidationError):
            tool.run()

    def test_mismatched_raster_shapes_raise(self, tmp_path: Path) -> None:
        """Bands of different pixel dimensions should raise InputValidationError."""
        # 4×4 red band
        red_arr = np.full((4, 4), 0.2, dtype="float32")
        red_path = tmp_path / "red.tif"
        profile_small = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": 4, "width": 4,
            "crs": CRS.from_epsg(4326),
            "transform": from_bounds(0, 0, 1, 1, 4, 4),
        }
        with rasterio.open(red_path, "w", **profile_small) as dst:
            dst.write(red_arr, 1)

        # 8×8 NIR band (different shape)
        nir_arr = np.full((8, 8), 0.6, dtype="float32")
        nir_path = tmp_path / "nir.tif"
        profile_large = {**profile_small, "height": 8, "width": 8,
                         "transform": from_bounds(0, 0, 1, 1, 8, 8)}
        with rasterio.open(nir_path, "w", **profile_large) as dst:
            dst.write(nir_arr, 1)

        tool = SpectralIndexCalculator(
            band_files=BandFileMap(red=red_path, nir=nir_path),  # type: ignore[misc]
            output_dir=tmp_path / "out",
            strategies=[NDVIStrategy()],
        )
        with pytest.raises(InputValidationError, match="mismatch"):
            tool.run()
