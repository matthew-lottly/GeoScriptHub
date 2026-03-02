"""Sentinel-1 SAR processing — backscatter, water/urban masks.

v1.0 — Quantum Land-Cover Change Detector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter, zoom

logger = logging.getLogger("geoscripthub.landcover_change.sar_processor")


@dataclass
class SARFeatures:
    """SAR-derived feature layers at target resolution."""

    vv_db: np.ndarray               # VV backscatter (dB)
    vh_db: np.ndarray               # VH backscatter (dB)
    vh_vv_ratio: np.ndarray         # VH/VV cross-pol ratio (dB)
    sar_water_index: np.ndarray     # (VH − VV) / (VH + VV)
    water_mask_sar: np.ndarray      # boolean — SAR-detected water
    building_mask_sar: np.ndarray   # boolean — SAR-detected buildings
    forest_mask_sar: np.ndarray     # boolean — SAR-detected forest
    n_observations: int
    valid_mask: np.ndarray


class SARProcessor:
    """Accumulate Sentinel-1 SAR observations and compute composite features."""

    def __init__(self, target_shape: tuple[int, int]) -> None:
        self.target_shape = target_shape
        self._vv_sum = np.zeros(target_shape, dtype="float64")
        self._vh_sum = np.zeros(target_shape, dtype="float64")
        self._count = np.zeros(target_shape, dtype="int32")
        self._n_obs = 0

    def add_observation(self, vv: np.ndarray, vh: np.ndarray) -> None:
        """Add one SAR scene (linear power, not dB)."""
        # Lee speckle filter
        vv_filt = self._lee_filter(vv)
        vh_filt = self._lee_filter(vh)

        # Regrid to target if needed
        if vv_filt.shape != self.target_shape:
            zf = (
                self.target_shape[0] / vv_filt.shape[0],
                self.target_shape[1] / vv_filt.shape[1],
            )
            vv_filt = np.asarray(zoom(vv_filt, zf, order=1, mode="nearest"))
            vh_filt = np.asarray(zoom(vh_filt, zf, order=1, mode="nearest"))

        valid = (vv_filt > 1e-10) & (vh_filt > 1e-10)
        self._vv_sum += np.where(valid, vv_filt, 0.0)
        self._vh_sum += np.where(valid, vh_filt, 0.0)
        self._count += valid.astype("int32")
        self._n_obs += 1

    def compute_features(self) -> SARFeatures:
        """Build temporal-mean composite and derive all SAR features."""
        safe_count = np.maximum(self._count, 1)
        vv_mean = self._vv_sum / safe_count
        vh_mean = self._vh_sum / safe_count

        vv_mean = np.maximum(vv_mean, 1e-10)
        vh_mean = np.maximum(vh_mean, 1e-10)

        vv_db = 10 * np.log10(vv_mean)
        vh_db = 10 * np.log10(vh_mean)
        vh_vv_ratio = vh_db - vv_db

        # SAR Water Index
        denom = vh_mean + vv_mean
        swi = np.where(denom > 0, (vh_mean - vv_mean) / denom, 0.0)

        # Masks
        water_mask = vv_db < -15.0
        building_mask = vv_db > -5.0
        forest_mask = vh_vv_ratio > -6.0

        valid_mask = self._count > 0

        return SARFeatures(
            vv_db=vv_db.astype("float32"),
            vh_db=vh_db.astype("float32"),
            vh_vv_ratio=vh_vv_ratio.astype("float32"),
            sar_water_index=swi.astype("float32"),
            water_mask_sar=water_mask,
            building_mask_sar=building_mask,
            forest_mask_sar=forest_mask,
            n_observations=self._n_obs,
            valid_mask=valid_mask,
        )

    @staticmethod
    def _lee_filter(
        img: np.ndarray, size: int = 7, n_looks: float = 4.4,
    ) -> np.ndarray:
        """Adaptive Lee speckle filter."""
        img = np.maximum(img, 1e-10).astype("float64")
        mean = uniform_filter(img, size=size, mode="nearest")
        sq_mean = uniform_filter(img**2, size=size, mode="nearest")
        variance = np.maximum(sq_mean - mean**2, 0.0)

        noise_var = mean**2 / n_looks
        weight = np.where(
            variance > noise_var,
            1.0 - noise_var / np.maximum(variance, 1e-10),
            0.0,
        )
        weight = np.clip(weight, 0.0, 1.0)
        return (mean + weight * (img - mean)).astype("float32")
