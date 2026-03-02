"""Spectral, SAR, terrain, and texture feature extraction.

v1.0 — Quantum Land-Cover Change Detector

Builds a ~30-band feature vector per pixel per year from the
annual composite, SAR, and terrain layers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter

from .preprocessing import AnnualComposite, PreprocessedStack
from .sar_processor import SARFeatures
from .terrain import TerrainFeatures

logger = logging.getLogger("geoscripthub.landcover_change.feature_engineering")


@dataclass
class FeatureStack:
    """Per-pixel feature vector for one year."""

    year: int
    features: np.ndarray       # shape (rows, cols, n_features)
    feature_names: list[str]
    valid_mask: np.ndarray


def compute_spectral_indices(comp: AnnualComposite) -> dict[str, np.ndarray]:
    """Compute all spectral indices from an annual composite."""
    eps = 1e-8
    b, g, r, nir = comp.blue, comp.green, comp.red, comp.nir
    swir1, swir2 = comp.swir1, comp.swir2

    has_swir = not np.all(np.isnan(swir1))

    indices: dict[str, np.ndarray] = {}

    # Vegetation
    indices["ndvi"] = _safe_ratio(nir - r, nir + r, eps)
    indices["evi"] = np.clip(
        2.5 * (nir - r) / (nir + 6.0 * r - 7.5 * b + 1.0 + eps), -1, 1,
    ).astype("float32")
    indices["savi"] = np.clip(
        1.5 * (nir - r) / (nir + r + 0.5 + eps), -1, 2,
    ).astype("float32")

    # Water
    indices["ndwi"] = _safe_ratio(g - nir, g + nir, eps)
    if has_swir:
        indices["mndwi"] = _safe_ratio(g - swir1, g + swir1, eps)
        indices["awei"] = np.clip(
            4.0 * (g - swir1) - (0.25 * nir + 2.75 * swir2), -5, 5,
        ).astype("float32")
    else:
        indices["mndwi"] = np.zeros_like(g)
        indices["awei"] = np.zeros_like(g)

    # Built-up
    if has_swir:
        indices["ndbi"] = _safe_ratio(swir1 - nir, swir1 + nir, eps)
        indices["ui"] = _safe_ratio(swir2 - nir, swir2 + nir, eps)
    else:
        indices["ndbi"] = np.zeros_like(g)
        indices["ui"] = np.zeros_like(g)
    indices["bci"] = _safe_ratio(b + r - nir, b + r + nir, eps)

    # Soil / barren
    if has_swir:
        indices["bsi"] = _safe_ratio(
            swir1 + r - nir - b, swir1 + r + nir + b, eps,
        )
        indices["lswi"] = _safe_ratio(nir - swir1, nir + swir1, eps)
    else:
        indices["bsi"] = np.zeros_like(g)
        indices["lswi"] = np.zeros_like(g)

    return indices


def compute_texture_features(
    band: np.ndarray, window: int = 5,
) -> dict[str, np.ndarray]:
    """Compute simple texture statistics from a single band.

    Uses fast uniform-filter approximations of GLCM-like statistics.
    """
    band = band.astype("float32")
    mean = uniform_filter(band, size=window, mode="nearest")
    sq_mean = uniform_filter(band**2, size=window, mode="nearest")
    variance = np.maximum(sq_mean - mean**2, 0.0)
    std = np.sqrt(variance)

    # Entropy approximation (log of local std)
    entropy = np.log1p(std)

    # Contrast: difference between band and local mean
    contrast = np.abs(band - mean)

    # Homogeneity approximation
    homogeneity = 1.0 / (1.0 + contrast)

    return {
        "texture_mean": mean,
        "texture_std": std,
        "texture_entropy": entropy.astype("float32"),
        "texture_contrast": contrast.astype("float32"),
        "texture_homogeneity": homogeneity.astype("float32"),
    }


def build_feature_stack(
    composite: AnnualComposite,
    sar: Optional[SARFeatures] = None,
    terrain: Optional[TerrainFeatures] = None,
) -> FeatureStack:
    """Build a complete feature vector for one year.

    Feature layout (~30 bands):
      [0-5]   Raw bands: B, G, R, NIR, SWIR1, SWIR2
      [6-17]  Spectral indices (12): NDVI, EVI, SAVI, NDWI, MNDWI, AWEI,
              NDBI, UI, BCI, BSI, LSWI
      [18-22] Texture (5): mean, std, entropy, contrast, homogeneity
      [23-26] SAR (4): VV_dB, VH_dB, VH/VV, SWI  (zeros if unavailable)
      [27-30] Terrain (4): elevation, slope, HAND, TPI (zeros if unavail)
    """
    target_shape = composite.blue.shape
    layers: list[np.ndarray] = []
    names: list[str] = []

    # Raw bands
    for band_name, band_arr in [
        ("blue", composite.blue), ("green", composite.green),
        ("red", composite.red), ("nir", composite.nir),
        ("swir1", composite.swir1), ("swir2", composite.swir2),
    ]:
        arr = np.nan_to_num(band_arr, nan=0.0).astype("float32")
        layers.append(arr)
        names.append(band_name)

    # Spectral indices
    indices = compute_spectral_indices(composite)
    for idx_name in [
        "ndvi", "evi", "savi", "ndwi", "mndwi", "awei",
        "ndbi", "ui", "bci", "bsi", "lswi",
    ]:
        layers.append(np.nan_to_num(indices[idx_name], nan=0.0))
        names.append(idx_name)

    # Texture (from NIR band)
    tex = compute_texture_features(np.nan_to_num(composite.nir, nan=0.0))
    for tex_name in [
        "texture_mean", "texture_std", "texture_entropy",
        "texture_contrast", "texture_homogeneity",
    ]:
        layers.append(tex[tex_name])
        names.append(tex_name)

    # SAR features
    if sar is not None and sar.valid_mask.any():
        for sar_name, sar_arr in [
            ("vv_db", sar.vv_db), ("vh_db", sar.vh_db),
            ("vh_vv_ratio", sar.vh_vv_ratio),
            ("sar_water_index", sar.sar_water_index),
        ]:
            arr = sar_arr
            if arr.shape != target_shape:
                from scipy.ndimage import zoom as scipy_zoom
                zf = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
                arr = np.asarray(scipy_zoom(arr, zf, order=1, mode="nearest"))
            layers.append(np.nan_to_num(arr, nan=0.0).astype("float32"))
            names.append(sar_name)
    else:
        for sar_name in ["vv_db", "vh_db", "vh_vv_ratio", "sar_water_index"]:
            layers.append(np.zeros(target_shape, dtype="float32"))
            names.append(sar_name)

    # Terrain features
    if terrain is not None and terrain.valid_mask.any():
        for t_name, t_arr in [
            ("elevation", terrain.elevation), ("slope", terrain.slope),
            ("hand", terrain.hand), ("tpi", terrain.tpi),
        ]:
            arr = t_arr
            if arr.shape != target_shape:
                from scipy.ndimage import zoom as scipy_zoom
                zf = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
                arr = np.asarray(scipy_zoom(arr, zf, order=1, mode="nearest"))
            layers.append(np.nan_to_num(arr, nan=0.0).astype("float32"))
            names.append(t_name)
    else:
        for t_name in ["elevation", "slope", "hand", "tpi"]:
            layers.append(np.zeros(target_shape, dtype="float32"))
            names.append(t_name)

    # Stack: (rows, cols, n_features)
    feature_cube = np.stack(layers, axis=-1)

    return FeatureStack(
        year=composite.year,
        features=feature_cube,
        feature_names=names,
        valid_mask=composite.valid_mask,
    )


# ── Utilities ─────────────────────────────────────────────────────

def _safe_ratio(
    numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-8,
) -> np.ndarray:
    """Safe normalised difference ratio, clipped to [-1, 1]."""
    return np.clip(
        numerator / (denominator + eps), -1.0, 1.0,
    ).astype("float32")
