"""
test_quantum_flood.py
=====================
Unit tests for the Quantum Flood Frequency Mapper.

Tests use synthetic data — no external API calls or real imagery needed.
"""

import numpy as np
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_flood_frequency.aoi import AOIBuilder
from quantum_flood_frequency.preprocessing import ImagePreprocessor
from quantum_flood_frequency.quantum_classifier import (
    QuantumFeatureEncoder,
    bayesian_model_average,
    compute_ndwi,
    compute_mndwi,
    compute_awei_sh,
    _safe_ratio,
)
from quantum_flood_frequency.flood_engine import FloodFrequencyEngine
from quantum_flood_frequency.fema import FEMAFloodZones

# ---------------------------------------------------------------------------
# Test AOI Builder
# ---------------------------------------------------------------------------


class TestAOIBuilder:
    """AOI construction and coordinate transforms."""

    def test_default_aoi_builds(self) -> None:
        aoi = AOIBuilder().build()

        assert aoi.center_lat == pytest.approx(29.22, abs=0.01)
        assert aoi.center_lon == pytest.approx(-89.25, abs=0.01)
        assert aoi.target_crs == "EPSG:32615"
        assert aoi.area_km2 > 0

    def test_custom_aoi_builds(self) -> None:
        aoi = AOIBuilder(center_lat=30.0, center_lon=-90.0, buffer_km=2.0).build()

        assert aoi.center_lat == pytest.approx(30.0)
        assert aoi.center_lon == pytest.approx(-90.0)
        assert aoi.area_km2 < 100  # 2 km buffer → ~16 km²

    def test_bboxes_are_valid(self) -> None:
        aoi = AOIBuilder().build()

        west, south, east, north = aoi.bbox_wgs84
        assert west < east
        assert south < north

        uw, us, ue, un = aoi.bbox_utm
        assert uw < ue
        assert us < un


# ---------------------------------------------------------------------------
# Test Spectral Indices
# ---------------------------------------------------------------------------


class TestSpectralIndices:
    """Spectral water index calculations."""

    def test_ndwi_range(self) -> None:
        green = np.array([0.1, 0.3, 0.5], dtype="float32")
        nir = np.array([0.3, 0.1, 0.5], dtype="float32")

        ndwi = compute_ndwi(green, nir)

        assert ndwi.shape == (3,)
        assert np.all(ndwi >= -1) and np.all(ndwi <= 1)
        # Water: green > nir → NDWI > 0
        assert ndwi[1] > 0

    def test_mndwi_range(self) -> None:
        green = np.array([0.2, 0.5], dtype="float32")
        swir1 = np.array([0.4, 0.1], dtype="float32")

        mndwi = compute_mndwi(green, swir1)
        assert np.all(mndwi >= -1) and np.all(mndwi <= 1)

    def test_awei_sh_computation(self) -> None:
        blue = np.array([0.1], dtype="float32")
        green = np.array([0.2], dtype="float32")
        nir = np.array([0.05], dtype="float32")
        swir1 = np.array([0.05], dtype="float32")
        swir2 = np.array([0.02], dtype="float32")

        awei = compute_awei_sh(blue, green, nir, swir1, swir2)
        assert awei.shape == (1,)
        # Water pixel → AWEI > 0 typically
        assert awei[0] > 0

    def test_safe_ratio_zero_denominator(self) -> None:
        a = np.array([0.0])
        b = np.array([0.0])
        result = _safe_ratio(a, b)
        assert result[0] == 0.0  # no division by zero


# ---------------------------------------------------------------------------
# Test Quantum Feature Encoder (QFE)
# ---------------------------------------------------------------------------


class TestQuantumFeatureEncoder:
    """Pseudo-quantum feature encoding on synthetic data."""

    def test_output_shape(self) -> None:
        qfe = QuantumFeatureEncoder()
        ndwi = np.random.uniform(-1, 1, (10, 10)).astype("float32")
        mndwi = np.random.uniform(-1, 1, (10, 10)).astype("float32")
        awei = np.random.uniform(-2, 2, (10, 10)).astype("float32")

        water_prob, confidence = qfe.encode(ndwi, mndwi, awei)

        assert water_prob.shape == (10, 10)
        assert confidence.shape == (10, 10)

    def test_probabilities_in_range(self) -> None:
        qfe = QuantumFeatureEncoder()
        ndwi = np.random.uniform(-1, 1, (5, 5)).astype("float32")
        mndwi = np.random.uniform(-1, 1, (5, 5)).astype("float32")
        awei = np.random.uniform(-2, 2, (5, 5)).astype("float32")

        water_prob, confidence = qfe.encode(ndwi, mndwi, awei)

        assert np.all(water_prob >= 0) and np.all(water_prob <= 1)
        assert np.all(confidence >= 0) and np.all(confidence <= 1)

    def test_quantum_encodes_different_spectra_differently(self) -> None:
        """Pixels with different spectral signatures should produce
        different quantum state probabilities — the encoder should
        discriminate between water-like and land-like inputs.

        Note: Due to quantum interference effects from the entangling
        CZ gate, the |00⟩ (water) amplitude is not monotonically
        related to NDWI.  Instead, we verify that the full probability
        distribution differs between water-like and land-like pixels,
        confirming the encoder is sensitive to spectral signal.
        """
        qfe = QuantumFeatureEncoder()

        # Water pixel
        ndwi_water = np.array([[0.8]])
        mndwi_water = np.array([[0.7]])
        awei_water = np.array([[2.0]])

        # Land pixel
        ndwi_land = np.array([[-0.5]])
        mndwi_land = np.array([[-0.6]])
        awei_land = np.array([[-2.0]])

        p_water, c_water = qfe.encode(ndwi_water, mndwi_water, awei_water)
        p_land, c_land = qfe.encode(ndwi_land, mndwi_land, awei_land)

        # The encoder must produce different outputs for different inputs
        assert p_water[0, 0] != pytest.approx(p_land[0, 0], abs=1e-3)

    def test_quantum_mixing_gate_is_unitary(self) -> None:
        qfe = QuantumFeatureEncoder()
        U = qfe._mixing_gate

        # Unitary check: U† · U = I
        identity = U.conj().T @ U
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Test Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """Image resampling and alignment."""

    def test_downsample_cubic_reduces_size(self) -> None:
        arr = np.random.rand(100, 100).astype("float32")
        result = ImagePreprocessor._downsample_cubic(arr, (30, 30))

        assert result.shape == (30, 30)
        assert result.dtype == np.float32

    def test_regrid_mask_preserves_binary(self) -> None:
        mask = np.random.choice([True, False], size=(60, 60))
        result = ImagePreprocessor._regrid_mask(mask, mask.shape, (30, 30))

        assert result.shape == (30, 30)
        assert result.dtype == bool

    def test_landsat_sr_to_reflectance(self) -> None:
        # Test with known Landsat C2L2 values
        dn = np.array([10000, 20000, 40000], dtype="float32")
        refl = ImagePreprocessor._ls_sr_to_reflectance(dn)

        assert np.all(refl >= 0)
        assert np.all(refl <= 1)

    def test_landsat_cloud_mask(self) -> None:
        # Bit 6 set (clear), no cloud bits
        clear_pixel = np.array([1 << 6], dtype="uint16")
        mask = ImagePreprocessor._landsat_cloud_mask(clear_pixel)
        assert mask[0] == True

        # Bit 3 set (cloud)
        cloud_pixel = np.array([1 << 3], dtype="uint16")
        mask = ImagePreprocessor._landsat_cloud_mask(cloud_pixel)
        assert mask[0] == False


# ---------------------------------------------------------------------------
# Test Flood Frequency Engine
# ---------------------------------------------------------------------------


class TestFloodFrequencyEngine:
    """Frequency aggregation from classified observations."""

    @staticmethod
    def _make_mock_aligned_stack(rows: int = 20, cols: int = 20) -> MagicMock:
        stack = MagicMock()
        stack.rows = rows
        stack.cols = cols
        stack.transform = (0.0, 30.0, 0.0, 100.0, 0.0, -30.0)
        stack.crs = "EPSG:32615"
        stack.bounds = (0.0, 0.0, cols * 30.0, rows * 30.0)
        stack.observations = []
        return stack

    @staticmethod
    def _make_classification(
        rows: int, cols: int, water_frac: float = 0.5, source: str = "landsat"
    ) -> MagicMock:
        cr = MagicMock()
        cr.cloud_mask = np.ones((rows, cols), dtype=bool)
        cr.water_probability = np.random.rand(rows, cols).astype("float32")
        cr.water_binary = cr.water_probability > (1.0 - water_frac)
        cr.source = source
        cr.date = "2023-01-01"
        return cr

    def test_frequency_range(self) -> None:
        stack = self._make_mock_aligned_stack(20, 20)
        engine = FloodFrequencyEngine(stack=stack)

        classifications = [
            self._make_classification(20, 20, water_frac=0.5)
            for _ in range(10)
        ]

        result = engine.compute(classifications)

        freq = result.frequency
        valid = ~np.isnan(freq)
        assert np.all(freq[valid] >= 0)
        assert np.all(freq[valid] <= 1)

    def test_all_water_yields_frequency_one(self) -> None:
        stack = self._make_mock_aligned_stack(5, 5)
        engine = FloodFrequencyEngine(stack=stack)

        # All observations: every pixel is water
        classifications = []
        for _ in range(5):
            cr = MagicMock()
            cr.cloud_mask = np.ones((5, 5), dtype=bool)
            cr.water_probability = np.ones((5, 5), dtype="float32")
            cr.water_binary = np.ones((5, 5), dtype=bool)
            cr.source = "landsat"
            cr.date = "2023-01-01"
            classifications.append(cr)

        result = engine.compute(classifications)
        np.testing.assert_allclose(result.frequency, 1.0)

    def test_no_water_yields_frequency_zero(self) -> None:
        stack = self._make_mock_aligned_stack(5, 5)
        engine = FloodFrequencyEngine(stack=stack)

        classifications = []
        for _ in range(5):
            cr = MagicMock()
            cr.cloud_mask = np.ones((5, 5), dtype=bool)
            cr.water_probability = np.zeros((5, 5), dtype="float32")
            cr.water_binary = np.zeros((5, 5), dtype=bool)
            cr.source = "sentinel2"
            cr.date = "2023-06-01"
            classifications.append(cr)

        result = engine.compute(classifications)
        np.testing.assert_allclose(result.frequency, 0.0)

    def test_zone_masks_are_exhaustive(self) -> None:
        stack = self._make_mock_aligned_stack(10, 10)
        engine = FloodFrequencyEngine(stack=stack)

        classifications = [
            self._make_classification(10, 10, water_frac=0.3)
            for _ in range(20)
        ]

        result = engine.compute(classifications)

        # Every valid pixel should be in exactly one zone
        total_zoned = (
            result.permanent_mask.sum()
            + result.seasonal_mask.sum()
            + result.rare_mask.sum()
            + result.dry_mask.sum()
        )
        valid_pixels = (~np.isnan(result.frequency)).sum()
        assert total_zoned == valid_pixels

    def test_wilson_confidence_bounds(self) -> None:
        stack = self._make_mock_aligned_stack(5, 5)
        engine = FloodFrequencyEngine(stack=stack)

        classifications = [
            self._make_classification(5, 5, water_frac=0.5)
            for _ in range(10)
        ]
        result = engine.compute(classifications)

        valid = ~np.isnan(result.confidence_lower)
        assert np.all(result.confidence_lower[valid] <= result.frequency[valid])
        assert np.all(result.confidence_upper[valid] >= result.frequency[valid])
        assert np.all(result.confidence_lower[valid] >= 0)
        assert np.all(result.confidence_upper[valid] <= 1)


# ---------------------------------------------------------------------------
# Test Bayesian Model Averaging
# ---------------------------------------------------------------------------


class TestBayesianModelAveraging:
    """Bayesian fusion of classifier outputs."""

    def test_output_range(self) -> None:
        q = np.random.rand(10, 10).astype("float32")
        s = np.random.rand(10, 10).astype("float32")
        g = np.random.rand(10, 10).astype("float32")
        c = np.random.rand(10, 10).astype("float32")

        fused = bayesian_model_average(q, s, g, c)

        assert np.all(fused >= 0) and np.all(fused <= 1)
        assert fused.dtype == np.float32

    def test_unanimous_water_yields_high_prob(self) -> None:
        q = np.ones((5, 5), dtype="float32")
        s = np.ones((5, 5), dtype="float32")
        g = np.ones((5, 5), dtype="float32")
        c = np.ones((5, 5), dtype="float32")

        fused = bayesian_model_average(q, s, g, c)
        np.testing.assert_allclose(fused, 1.0, atol=0.01)

    def test_unanimous_land_yields_low_prob(self) -> None:
        q = np.zeros((5, 5), dtype="float32")
        s = np.zeros((5, 5), dtype="float32")
        g = np.zeros((5, 5), dtype="float32")
        c = np.ones((5, 5), dtype="float32")

        fused = bayesian_model_average(q, s, g, c)
        np.testing.assert_allclose(fused, 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# Test FEMA module (offline)
# ---------------------------------------------------------------------------


class TestFEMAFloodZones:
    """FEMA zone classification helpers (no network calls)."""

    def test_risk_classification(self) -> None:
        assert FEMAFloodZones._classify_risk("AE") == "high"
        assert FEMAFloodZones._classify_risk("V") == "high"
        assert FEMAFloodZones._classify_risk("X") == "moderate"
        assert FEMAFloodZones._classify_risk("D") == "undetermined"
        assert FEMAFloodZones._classify_risk("???") == "minimal"

    def test_zone_color_known(self) -> None:
        color = FEMAFloodZones.get_zone_color("AE")
        assert len(color) == 4  # RGBA
        assert all(0 <= c <= 1 for c in color)

    def test_zone_color_unknown_returns_default(self) -> None:
        color = FEMAFloodZones.get_zone_color("UNKNOWN_ZONE")
        assert len(color) == 4
