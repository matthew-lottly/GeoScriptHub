"""conftest.py — Shared fixtures for deep-fusion-landcover test suite."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def small_class_maps() -> list[np.ndarray]:
    """Four toy 32×32 class maps (values 1–12, nodata=0)."""
    rng = np.random.default_rng(0)
    maps = []
    for _ in range(4):
        m = rng.integers(1, 13, size=(32, 32), dtype=np.int8)
        maps.append(m)
    return maps


@pytest.fixture(scope="session")
def small_feature_array() -> np.ndarray:
    """Random (200, 50) float32 feature matrix."""
    rng = np.random.default_rng(42)
    return rng.random((200, 50), dtype=np.float32)


@pytest.fixture(scope="session")
def small_labels() -> np.ndarray:
    """200 random class labels in [0, 11]."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 12, size=200)


@pytest.fixture(scope="session")
def austin_aoi_result():
    """Small (2 km²) Austin sub-AOI for fast tests."""
    from deep_fusion_landcover.aoi import AOIBuilder
    return AOIBuilder(
        center_lat=30.2672,
        center_lon=-97.7431,
        buffer_km=1.0,
    ).build()
