"""DEM processing — slope, HAND, curvature, terrain features.

v1.0 — Quantum Land-Cover Change Detector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter, zoom

logger = logging.getLogger("geoscripthub.landcover_change.terrain")


@dataclass
class TerrainFeatures:
    """Terrain-derived feature layers at target resolution."""

    elevation: np.ndarray          # metres above sea level
    slope: np.ndarray              # degrees 0–90
    aspect: np.ndarray             # degrees 0–360
    hand: np.ndarray               # Height Above Nearest Drainage (m)
    curvature: np.ndarray          # profile curvature
    tpi: np.ndarray                # Topographic Position Index
    roughness: np.ndarray          # terrain roughness
    valid_mask: np.ndarray         # True where elevation is valid
    resolution: float


class TerrainProcessor:
    """Compute terrain features from DEM arrays."""

    def __init__(
        self,
        target_shape: tuple[int, int],
        target_resolution: float = 30.0,
    ) -> None:
        self.target_shape = target_shape
        self.target_resolution = target_resolution

    def process(
        self,
        dem: np.ndarray,
        dem_resolution: float = 30.0,
    ) -> TerrainFeatures:
        """Compute all terrain features from a DEM array."""
        # Regrid to target shape if needed
        if dem.shape != self.target_shape:
            zf = (
                self.target_shape[0] / dem.shape[0],
                self.target_shape[1] / dem.shape[1],
            )
            dem = np.asarray(
                zoom(dem.astype("float32"), zf, order=1, mode="nearest"),
            )

        valid_mask = np.isfinite(dem) & (dem > -500) & (dem < 9000)
        dem_clean = np.where(valid_mask, dem, 0.0).astype("float32")

        slope = self._compute_slope(dem_clean, dem_resolution)
        aspect = self._compute_aspect(dem_clean, dem_resolution)
        hand = self._compute_hand(dem_clean)
        curvature = self._compute_curvature(dem_clean, dem_resolution)
        tpi = self._compute_tpi(dem_clean)
        roughness = self._compute_roughness(dem_clean)

        return TerrainFeatures(
            elevation=dem_clean,
            slope=slope,
            aspect=aspect,
            hand=hand,
            curvature=curvature,
            tpi=tpi,
            roughness=roughness,
            valid_mask=valid_mask,
            resolution=self.target_resolution,
        )

    @staticmethod
    def _compute_slope(dem: np.ndarray, res: float) -> np.ndarray:
        """Horn (1981) 3×3 slope in degrees."""
        dy, dx = np.gradient(dem, res)
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    @staticmethod
    def _compute_aspect(dem: np.ndarray, res: float) -> np.ndarray:
        """Aspect in degrees (0=N, 90=E, 180=S, 270=W)."""
        dy, dx = np.gradient(dem, res)
        aspect = np.degrees(np.arctan2(-dx, dy))
        return np.where(aspect < 0, aspect + 360, aspect)

    @staticmethod
    def _compute_hand(dem: np.ndarray) -> np.ndarray:
        """Height Above Nearest Drainage — multi-scale minimum filter."""
        hand = dem.copy()
        for size in [15, 31, 63]:
            if min(dem.shape) > size:
                local_min = minimum_filter(dem, size=size, mode="nearest")
                hand = np.minimum(hand, dem - local_min)
        return np.maximum(hand, 0.0)

    @staticmethod
    def _compute_curvature(dem: np.ndarray, res: float) -> np.ndarray:
        """Profile curvature (second derivative along gradient direction)."""
        dy, dx = np.gradient(dem, res)
        dyy, _ = np.gradient(dy, res)
        _, dxx = np.gradient(dx, res)
        return -(dxx + dyy)

    @staticmethod
    def _compute_tpi(dem: np.ndarray, window: int = 11) -> np.ndarray:
        """Topographic Position Index — elevation minus local mean."""
        local_mean = uniform_filter(dem, size=window, mode="nearest")
        return dem - local_mean

    @staticmethod
    def _compute_roughness(dem: np.ndarray, window: int = 5) -> np.ndarray:
        """Terrain roughness — local standard deviation of elevation."""
        mean = uniform_filter(dem, size=window, mode="nearest")
        mean_sq = uniform_filter(dem**2, size=window, mode="nearest")
        variance = np.maximum(mean_sq - mean**2, 0.0)
        return np.sqrt(variance)
