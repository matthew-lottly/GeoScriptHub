"""
Tests for the analysis module's pure-numpy functions.

These tests run entirely offline -- no Planetary Computer API calls
are made.  Raster I/O and aoi tests are kept lightweight so they
can run in CI without optional geo dependencies.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sub_canopy_detector.analysis import SubCanopyAnalyser, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Helpers: create minimal synthetic imagery stubs
# ---------------------------------------------------------------------------

def _make_vv_linear(shape=(50, 60), n_time=20, seed=42) -> np.ndarray:
    """Return (time, y, x) synthetic VV linear-power array."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.01, 0.15, size=shape)
    noise = rng.normal(0, 0.005, size=(n_time, *shape))
    return np.clip(base[np.newaxis, :, :] + noise, 1e-6, None).astype(np.float32)


def _make_vh_linear(shape=(50, 60), n_time=20, seed=7) -> np.ndarray:
    np.random.seed(seed)
    return np.clip(np.random.uniform(0.002, 0.05, (n_time, *shape)), 1e-6, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

class TestStability:
    """Stability = 1 - CoV, clipped to [stability_floor, 1]."""

    def _call(self, vv_mean, vv_std):
        params = {**DEFAULT_PARAMS}
        # Use SubCanopyAnalyser._stability as a static method via instance
        # We instantiate with aoi/imagery=None to avoid side effects; we just
        # need the helper method.
        class _Stub:
            params = DEFAULT_PARAMS
        inst = _Stub()
        inst.params = params
        return SubCanopyAnalyser._stability(inst, vv_mean, vv_std)  # type: ignore[attr-defined]

    def test_uniform_surface_is_high(self):
        # Zero variance -> stability == 1.0 (before floor)
        vv_mean = np.ones((10, 10)) * 0.1
        vv_std  = np.zeros((10, 10))
        stab = self._call(vv_mean, vv_std)
        assert np.allclose(stab, 1.0), "Perfectly stable surface should score 1.0"

    def test_floor_applied(self):
        # Very high CoV (noisy) should floor at stability_floor
        vv_mean = np.ones((10, 10)) * 0.01
        vv_std  = np.ones((10, 10)) * 100.0    # huge std -> CoV >> 1
        stab = self._call(vv_mean, vv_std)
        floor = DEFAULT_PARAMS["stability_floor"]
        assert np.all(stab >= floor), "Stability must never go below stability_floor"

    def test_output_range(self):
        vv_data = _make_vv_linear()
        vv_mean = np.nanmean(vv_data, axis=0)
        vv_std  = np.nanstd(vv_data, axis=0)
        stab = self._call(vv_mean, vv_std)
        assert stab.min() >= 0.0
        assert stab.max() <= 1.0


# ---------------------------------------------------------------------------
# Polarimetric ratio
# ---------------------------------------------------------------------------

class TestPolRatio:
    def _call(self, vv_mean, vh_mean):
        class _Stub:
            params = dict(DEFAULT_PARAMS)
        inst = _Stub()
        return SubCanopyAnalyser._pol_ratio(inst, vv_mean, vh_mean)  # type: ignore

    def test_clamped_to_unit_interval(self):
        vv = np.ones((8, 8)) * 0.1
        vh = np.random.uniform(0, 0.5, (8, 8)).astype(np.float32)
        result = self._call(vv, vh)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_division_by_zero_safe(self):
        vv = np.zeros((5, 5))    # all-zero VV should not raise
        vh = np.ones((5, 5)) * 0.01
        result = self._call(vv, vh)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Anomaly
# ---------------------------------------------------------------------------

class TestAnomaly:
    def _call(self, image):
        return SubCanopyAnalyser._anomaly(
            None,  # type: ignore[arg-type]  # bypass self
            image=image,
            kernel_radius=DEFAULT_PARAMS["anomaly_kernel_radius"],
            sigma_scale=DEFAULT_PARAMS["anomaly_sigma"],
        )

    def test_output_range(self):
        arr = np.random.randn(40, 40).astype(np.float32)
        result = self._call(arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_nan_safe(self):
        arr = np.random.randn(30, 30).astype(np.float32)
        arr[5:10, 5:10] = np.nan
        result = self._call(arr)
        # Should not propagate NaN beyond the seeded patch
        assert np.sum(np.isnan(result)) == 0


# ---------------------------------------------------------------------------
# Forest mask
# ---------------------------------------------------------------------------

class TestForestMask:
    def _call(self, ndvi, ndwi):
        class _Stub:
            params = dict(DEFAULT_PARAMS)
        inst = _Stub()
        return SubCanopyAnalyser._forest_mask(inst, ndvi, ndwi)  # type: ignore

    def test_high_ndvi_not_wet_is_forest(self):
        ndvi = np.full((5, 5), 0.8)
        ndwi = np.full((5, 5), -0.1)
        mask = self._call(ndvi, ndwi)
        assert np.all(mask)

    def test_low_ndvi_excluded(self):
        ndvi = np.full((5, 5), 0.2)
        ndwi = np.full((5, 5), -0.1)
        mask = self._call(ndvi, ndwi)
        assert not np.any(mask)

    def test_high_ndwi_excluded(self):
        ndvi = np.full((5, 5), 0.9)
        ndwi = np.full((5, 5), 0.5)   # wet / water
        mask = self._call(ndvi, ndwi)
        assert not np.any(mask)


# ---------------------------------------------------------------------------
# Morphological opening
# ---------------------------------------------------------------------------

class TestMorphologicalOpen:
    def test_removes_single_isolated_pixel(self):
        prob = np.zeros((20, 20), dtype=np.float32)
        # One bright isolated pixel -- should be removed by opening
        prob[10, 10] = 0.9
        result = SubCanopyAnalyser._morphological_open(prob, threshold=0.45)
        assert np.isnan(result[10, 10]) or result[10, 10] == 0.0

    def test_preserves_large_blob(self):
        prob = np.zeros((30, 30), dtype=np.float32)
        # 5x5 patch should survive erosion + dilation
        prob[10:15, 10:15] = 0.8
        result = SubCanopyAnalyser._morphological_open(prob, threshold=0.45)
        centre = result[12, 12]
        assert not np.isnan(centre) and centre > 0.0


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_bad_weights_raise(self):
        import types, unittest.mock as mock

        class _FakeImagery:
            s1_count = 10
            s1 = None
            s2 = None
            dem = None

        class _FakeAOI:
            utm_crs = type("CRS", (), {"to_epsg": lambda s: 32616, "to_wkt": lambda s: ""})()
            bbox_utm = (0, 0, 10000, 10000)

        with pytest.raises(ValueError, match="weights must sum"):
            SubCanopyAnalyser(
                aoi=_FakeAOI(),  # type: ignore[arg-type]
                imagery=_FakeImagery(),  # type: ignore[arg-type]
                w_stability=0.99,    # bad sum
                w_polratio=0.01,
                w_texture=0.01,
                w_anomaly=0.01,
                w_optical=0.01,
            )
