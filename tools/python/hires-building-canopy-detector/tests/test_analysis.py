"""Basic smoke tests for HiResAnalyser."""

import numpy as np
import pytest

# Import will fail until package is installed — that's OK for a structure placeholder.
from hires_detector.analysis import HiResAnalyser


class TestLeeFilter:
    def test_smooths_noise(self):
        rng = np.random.default_rng(42)
        noisy = rng.normal(loc=-10.0, scale=3.0, size=(64, 64)).astype(np.float32)
        filtered = HiResAnalyser._lee_filter(noisy, window=7)
        # Variance should decrease after filtering
        assert filtered.std() < noisy.std()

    def test_preserves_shape(self):
        img = np.zeros((100, 100), dtype=np.float32)
        out = HiResAnalyser._lee_filter(img, window=5)
        assert out.shape == img.shape


class TestLinearSE:
    @pytest.mark.parametrize("angle", [0, 45, 90, 135])
    def test_shape(self, angle):
        se = HiResAnalyser._linear_se(7, angle)
        assert se.shape[0] == se.shape[1]  # square
        assert se.any()

    def test_horizontal(self):
        se = HiResAnalyser._linear_se(5, 0)
        # Middle row should be all True
        mid = se.shape[0] // 2
        assert se[mid, :].all()


class TestMBI:
    def test_output_range(self):
        rng = np.random.default_rng(0)
        img = rng.normal(-10, 2, (64, 64)).astype(np.float32)
        analyser = _make_dummy_analyser(img.shape)
        mbi = analyser._morphological_building_index(img, [3, 7], [0, 90])
        assert mbi.min() >= 0.0
        assert mbi.max() <= 1.0


class TestNDVI:
    def test_pure_vegetation(self):
        naip = np.zeros((10, 10, 4), dtype=np.float32)
        naip[:, :, 0] = 0.05   # low red
        naip[:, :, 3] = 0.50   # high NIR
        ndvi = HiResAnalyser._compute_ndvi(naip)
        assert ndvi.mean() > 0.7


class TestCanopyMask:
    def test_excludes_buildings(self):
        ndvi = np.full((20, 20), 0.6, dtype=np.float32)
        bldg = np.zeros((20, 20), dtype=bool)
        bldg[5:10, 5:10] = True
        mask = HiResAnalyser._canopy_mask(ndvi, bldg, threshold=0.3)
        assert not mask[7, 7]      # building pixel excluded
        assert mask[0, 0]          # non-building veg included


# ---------------------------------------------------------------------------
# Helper to create a minimal analyser for method testing
# ---------------------------------------------------------------------------

def _make_dummy_analyser(shape=(64, 64)):
    """Create a HiResAnalyser with synthetic imagery data."""
    from hires_detector.fetcher import HiResImageryData
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    H, W = shape
    transform = from_bounds(0, 0, W, H, W, H)
    data = HiResImageryData(
        sar=np.random.default_rng(0).normal(-10, 2, (H, W)).astype(np.float32),
        sar_source="test",
        sar_resolution_m=1.0,
        naip=np.random.default_rng(1).random((H, W, 4)).astype(np.float32),
        naip_source="test",
        naip_resolution_m=1.0,
        dem=np.zeros((H, W), dtype=np.float32),
        transform=transform,
        crs=CRS.from_epsg(32614),
        height=H,
        width=W,
    )
    return HiResAnalyser(data)
