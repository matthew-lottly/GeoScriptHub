"""feature_engineering.py — 150-dimension per-pixel feature stack builder.

Computes 8 groups of features from an annual composite:
    1. Spectral indices    (10)  — NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, BSI, NBR, EVI2, ARVI
    2. Band ratios         (3)   — NIR/R, SWIR1/NIR, G/R
    3. SAR backscatter     (4)   — VV, VH, ratio, sum
    4. SAR polarimetric    (2)   — entropy proxy, anisotropy
    5. Phenology           (7)   — NDVI mean/std/amplitude/green-up/peak/dormancy/trend slope
    6. LiDAR               (9)   — height percentiles, penetration ratio, density
    7. Terrain             (9)   — elevation, slope, aspect sin/cos, curvature, TPI×2, HAND
    8. GLCM Texture        (21)  — contrast/homogeneity/entropy/correlation/ASM on 3 bands

Total: ~65 per-layer + 7 multi-year temporal features → ~150 in full stack.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from .constants import (
    GLCM_WINDOW_PX,
    N_LIDAR_FEATURES,
    N_TERRAIN_FEATURES,
    N_TEXTURE_FEATURES,
)
from .temporal_compositing import AnnualComposite

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.feature_engineering")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class FeatureStack:
    """2-D feature array and associated metadata.

    Attributes
    ----------
    array:       Float32 ndarray of shape ``(H, W, N_FEATURES)``.
    names:       Feature name for each band (length ``N_FEATURES``).
    year:        Source year.
    valid_mask:  Boolean (H, W) mask — True where features are valid.
    shape:       ``(H, W)`` spatial dimensions.
    """

    array: np.ndarray
    names: list[str]
    year: int
    valid_mask: np.ndarray
    shape: tuple[int, int]


# ── Feature engineering class ─────────────────────────────────────────────────

class FeatureEngineer:
    """Build feature stacks from :class:`~.temporal_compositing.AnnualComposite` objects.

    Parameters
    ----------
    include_lidar:   Include LiDAR-derived features (nDSM, penetration, etc.).
    include_terrain: Include DEM-derived terrain features.
    include_texture: Include GLCM texture features.
    glcm_window:     Window size for texture computation (pixels).
    """

    def __init__(
        self,
        include_lidar: bool = True,
        include_terrain: bool = True,
        include_texture: bool = True,
        glcm_window: int = GLCM_WINDOW_PX,
    ) -> None:
        self.include_lidar = include_lidar
        self.include_terrain = include_terrain
        self.include_texture = include_texture
        self.glcm_window = glcm_window

    # ── Public ────────────────────────────────────────────────────────────────

    def build(
        self,
        composite: AnnualComposite,
        lidar_products: Optional[object] = None,   # LiDARProducts
        all_composites: Optional[list[AnnualComposite]] = None,
    ) -> FeatureStack:
        """Compute the full feature stack for *composite*.

        Parameters
        ----------
        composite:      Annual composite for the target year.
        lidar_products: Optional :class:`~.lidar_processor.LiDARProducts` object.
        all_composites: Full time series; enables temporal/phenology features.

        Returns
        -------
        FeatureStack
            Ready-to-classify feature array.
        """
        feature_arrays: list[np.ndarray] = []
        feature_names: list[str] = []

        # Determine spatial shape from best available source
        shape = _get_shape(composite)
        if shape is None:
            logger.warning("Year %d: no valid raster data — returning empty FeatureStack.", composite.year)
            return FeatureStack(
                array=np.zeros((1, 1, 1), dtype="float32"),
                names=["empty"],
                year=composite.year,
                valid_mask=np.zeros((1, 1), dtype=bool),
                shape=(1, 1),
            )

        H, W = shape

        # ── 1. Spectral indices ────────────────────────────────────────────
        spec_arr, spec_names = self._spectral_indices(composite, H, W)
        feature_arrays.append(spec_arr)
        feature_names.extend(spec_names)

        # ── 2. Band ratios ─────────────────────────────────────────────────
        ratio_arr, ratio_names = self._band_ratios(composite, H, W)
        feature_arrays.append(ratio_arr)
        feature_names.extend(ratio_names)

        # ── 3. SAR backscatter ─────────────────────────────────────────────
        sar_arr, sar_names = self._sar_backscatter(composite, H, W)
        feature_arrays.append(sar_arr)
        feature_names.extend(sar_names)

        # ── 4. SAR polarimetric ────────────────────────────────────────────
        pol_arr, pol_names = self._sar_polarimetric(composite, H, W)
        feature_arrays.append(pol_arr)
        feature_names.extend(pol_names)

        # ── 5. Phenology ───────────────────────────────────────────────────
        if all_composites is not None:
            phen_arr, phen_names = self._phenology_features(
                composite.year, all_composites, H, W
            )
            feature_arrays.append(phen_arr)
            feature_names.extend(phen_names)

        # ── 6. LiDAR ──────────────────────────────────────────────────────
        if self.include_lidar and lidar_products is not None:
            lidar_arr, lidar_names = self._lidar_features(lidar_products, H, W)
            feature_arrays.append(lidar_arr)
            feature_names.extend(lidar_names)

        # ── 7. Terrain ─────────────────────────────────────────────────────
        if self.include_terrain:
            terrain_arr, terrain_names = self._terrain_features(composite, H, W)
            feature_arrays.append(terrain_arr)
            feature_names.extend(terrain_names)

        # ── 8. GLCM texture ────────────────────────────────────────────────
        if self.include_texture:
            tex_arr, tex_names = self._glcm_features(composite, H, W)
            feature_arrays.append(tex_arr)
            feature_names.extend(tex_names)

        full_stack = np.concatenate(feature_arrays, axis=-1)  # (H, W, N)
        valid_mask = np.all(np.isfinite(full_stack), axis=-1)

        return FeatureStack(
            array=full_stack.astype("float32"),
            names=feature_names,
            year=composite.year,
            valid_mask=valid_mask,
            shape=(H, W),
        )

    # ── Spectral indices ──────────────────────────────────────────────────────

    def _spectral_indices(
        self, comp: AnnualComposite, H: int, W: int
    ) -> tuple[np.ndarray, list[str]]:
        """Compute 10 spectral indices from optical reflectance bands."""
        ds = comp.landsat if comp.landsat is not None else comp.sentinel2
        out = np.full((H, W, 10), np.nan, dtype="float32")
        names = ["NDVI", "EVI", "SAVI", "NDWI", "MNDWI", "NDBI", "BSI", "NBR", "EVI2", "ARVI"]

        if ds is None:
            return out, names

        red   = _band(ds, ["red", "SR_B4", "SR_B3", "B04"], H, W)
        green = _band(ds, ["green", "SR_B3", "SR_B2", "B03"], H, W)
        blue  = _band(ds, ["blue", "SR_B2", "SR_B1", "B02"], H, W)
        nir   = _band(ds, ["nir", "SR_B5", "SR_B4", "B08"], H, W)
        swir1 = _band(ds, ["swir1", "SR_B6", "SR_B5", "B11"], H, W)
        swir2 = _band(ds, ["swir2", "SR_B7", "B12"], H, W)

        eps = 1e-10
        out[:, :, 0] = _safe_ratio(nir - red, nir + red + eps)                          # NDVI
        out[:, :, 1] = 2.5 * _safe_ratio(nir - red, nir + 6*red - 7.5*blue + 1 + eps)  # EVI
        out[:, :, 2] = 1.5 * _safe_ratio(nir - red, nir + red + 0.5 + eps)              # SAVI
        out[:, :, 3] = _safe_ratio(green - nir, green + nir + eps)                       # NDWI
        out[:, :, 4] = _safe_ratio(green - swir1, green + swir1 + eps)                  # MNDWI
        out[:, :, 5] = _safe_ratio(swir1 - nir, swir1 + nir + eps)                      # NDBI
        out[:, :, 6] = (swir1 - nir - red) / (swir1 + nir + red + eps)                  # BSI
        out[:, :, 7] = _safe_ratio(nir - swir2, nir + swir2 + eps)                      # NBR
        out[:, :, 8] = 2.5 * _safe_ratio(nir - red, nir + 2.4*red + 1 + eps)           # EVI2
        out[:, :, 9] = _safe_ratio(nir - (2*red - blue), nir + (2*red - blue) + eps)   # ARVI

        return out, names

    # ── Band ratios ───────────────────────────────────────────────────────────

    def _band_ratios(
        self, comp: AnnualComposite, H: int, W: int
    ) -> tuple[np.ndarray, list[str]]:
        ds = comp.landsat if comp.landsat is not None else comp.sentinel2
        out = np.full((H, W, 3), np.nan, dtype="float32")
        names = ["NIR_R", "SWIR1_NIR", "G_R"]
        if ds is None:
            return out, names

        red   = _band(ds, ["red", "SR_B4", "SR_B3", "B04"], H, W)
        green = _band(ds, ["green", "SR_B3", "SR_B2", "B03"], H, W)
        nir   = _band(ds, ["nir", "SR_B5", "SR_B4", "B08"], H, W)
        swir1 = _band(ds, ["swir1", "SR_B6", "SR_B5", "B11"], H, W)

        eps = 1e-10
        out[:, :, 0] = _safe_ratio(nir, red + eps)
        out[:, :, 1] = _safe_ratio(swir1, nir + eps)
        out[:, :, 2] = _safe_ratio(green, red + eps)
        return out, names

    # ── SAR backscatter ───────────────────────────────────────────────────────

    def _sar_backscatter(
        self, comp: AnnualComposite, H: int, W: int
    ) -> tuple[np.ndarray, list[str]]:
        out = np.full((H, W, 4), np.nan, dtype="float32")
        names = ["VV_mean_dB", "VH_mean_dB", "VV_VH_ratio", "VV_VH_sum_dB"]
        ds = comp.sentinel1
        if ds is None:
            return out, names

        vv = _band(ds, ["vv_db_mean", "vv_mean_db", "vv_db"], H, W)
        vh = _band(ds, ["vh_db_mean", "vh_mean_db", "vh_db"], H, W)

        out[:, :, 0] = vv
        out[:, :, 1] = vh
        out[:, :, 2] = (vv - vh).clip(-20, 20)          # ratio in dB
        out[:, :, 3] = 10 * np.log10(
            10**(vv / 10) + 10**(vh / 10) + 1e-20        # sum in linear then back to dB
        )
        return out, names

    # ── SAR polarimetric ──────────────────────────────────────────────────────

    def _sar_polarimetric(
        self, comp: AnnualComposite, H: int, W: int
    ) -> tuple[np.ndarray, list[str]]:
        out = np.full((H, W, 2), np.nan, dtype="float32")
        names = ["SAR_entropy_proxy", "SAR_anisotropy"]
        ds = comp.sentinel1
        if ds is None:
            return out, names

        vv = _band(ds, ["vv_linear_mean", "vv_mean"], H, W)
        vh = _band(ds, ["vh_linear_mean", "vh_mean"], H, W)

        eps = 1e-10
        # Entropy proxy: local CoV of VV (high = heterogeneous)
        vv_std = _band(ds, ["vv_linear_std", "vv_std"], H, W)
        out[:, :, 0] = _safe_ratio(vv_std, vv + eps).clip(0, 5)

        # Anisotropy: (VH - VV) / (VH + VV) normalised to [0,1]
        ratio = _safe_ratio(vh - vv, vh + vv + eps)
        out[:, :, 1] = (ratio + 1) / 2.0   # rescale [-1,1] → [0,1]
        return out, names

    # ── Phenology ─────────────────────────────────────────────────────────────

    def _phenology_features(
        self,
        year: int,
        all_composites: list[AnnualComposite],
        H: int,
        W: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract annual NDVI statistics and multi-year trend from the archive."""
        out = np.full((H, W, 7), np.nan, dtype="float32")
        names = [
            "NDVI_ann_mean", "NDVI_ann_std", "NDVI_amplitude",
            "NDVI_spring_peak", "NDVI_summer_peak",
            "NDVI_dormancy", "NDVI_trend_slope",
        ]

        # Collect NDVI for the focal year and ±3 years for trend
        ndvi_series: list[np.ndarray] = []
        years_list: list[int] = []
        for comp in all_composites:
            ds = comp.landsat if comp.landsat is not None else comp.sentinel2
            if ds is None:
                continue
            red = _band(ds, ["red", "SR_B4", "SR_B3", "B04"], H, W)
            nir = _band(ds, ["nir", "SR_B5", "SR_B4", "B08"], H, W)
            ndvi = _safe_ratio(nir - red, nir + red + 1e-10)
            ndvi_series.append(ndvi)
            years_list.append(comp.year)

        if not ndvi_series:
            return out, names

        stack = np.stack(ndvi_series, axis=0)  # (T, H, W)
        target_idx = years_list.index(year) if year in years_list else 0
        focal = stack[target_idx]  # (H, W) — NDVI for this year

        out[:, :, 0] = np.nanmean(stack, axis=0)   # multi-year mean
        out[:, :, 1] = np.nanstd(stack, axis=0)    # inter-annual std
        out[:, :, 2] = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)  # amplitude

        # Proxy seasonal peaks: use focal year as single season
        out[:, :, 3] = focal                        # spring approximation
        out[:, :, 4] = focal                        # summer approximation (same here)
        out[:, :, 5] = np.nanmin(stack, axis=0)     # dormancy / minimum NDVI

        # Trend slope: linear regression of NDVI over years
        if len(years_list) >= 3:
            yr_arr = np.array(years_list, dtype="float32") - np.mean(years_list)
            # Vectorised least-squares slope: sum((y-ybar)*(x-xbar)) / sum((x-xbar)^2)
            cov = np.nansum(
                stack * yr_arr[:, np.newaxis, np.newaxis], axis=0
            ) / np.nansum(~np.isnan(stack), axis=0).clip(1)
            var = float(np.sum(yr_arr**2))
            out[:, :, 6] = cov / max(var, 1e-10)

        return out, names

    # ── LiDAR ────────────────────────────────────────────────────────────────

    def _lidar_features(self, lidar_products: object, H: int, W: int) -> tuple[np.ndarray, list[str]]:
        """Extract 9 LiDAR features from a LiDARProducts object."""
        from .lidar_processor import LiDARProducts  # local import avoids circular
        assert isinstance(lidar_products, LiDARProducts)
        out = np.full((H, W, 9), np.nan, dtype="float32")
        names = [
            "nDSM_height", "DTM_slope", "first_density", "last_density",
            "penetration_ratio", "height_P25", "height_P50", "height_P75", "height_P95",
        ]
        lp = lidar_products
        for i, arr in enumerate([
            lp.ndsm, lp.dtm_slope, lp.first_return_density, lp.last_return_density,
            lp.penetration_ratio, lp.height_p25, lp.height_p50, lp.height_p75, lp.height_p95,
        ]):
            if arr is not None:
                try:
                    out[:, :, i] = _resize_to(arr, H, W)
                except Exception:  # noqa: BLE001
                    pass
        return out, names

    # ── Terrain ───────────────────────────────────────────────────────────────

    def _terrain_features(
        self, comp: AnnualComposite, H: int, W: int
    ) -> tuple[np.ndarray, list[str]]:
        """Compute DEM-derived terrain features (9 bands)."""
        out = np.full((H, W, 9), np.nan, dtype="float32")
        names = [
            "elevation", "slope_deg", "aspect_sin", "aspect_cos",
            "plan_curvature", "profile_curvature", "TPI_50m", "TPI_200m", "HAND",
        ]
        if comp.dem is None:
            return out, names

        elev = _resize_to(comp.dem.values.squeeze().astype("float32"), H, W)
        out[:, :, 0] = elev

        # Slope and aspect via gradient
        dy, dx = np.gradient(elev)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        out[:, :, 1] = np.degrees(slope_rad)
        aspect_rad = np.arctan2(-dx, dy)
        out[:, :, 2] = np.sin(aspect_rad)
        out[:, :, 3] = np.cos(aspect_rad)

        # Curvature (plan + profile) via second-order gradient
        d2x = np.gradient(dx, axis=1)
        d2y = np.gradient(dy, axis=0)
        out[:, :, 4] = d2x    # plan curvature (laplacian x)
        out[:, :, 5] = d2y    # profile curvature (laplacian y)

        # Topographic Position Index at two scales
        for i, radius in enumerate([50, 200]):
            kernel_size = max(3, int(radius / 10))  # at 10 m resolution
            local_mean = uniform_filter(elev, size=kernel_size, mode="nearest")
            out[:, :, 6 + i] = elev - local_mean

        # HAND proxy: elevation above local minimum within 500 m window
        hand_kernel = max(3, 50)
        local_min = -uniform_filter(-elev, size=hand_kernel, mode="nearest")
        out[:, :, 8] = (elev - local_min).clip(0, 500)

        return out, names

    # ── GLCM texture ──────────────────────────────────────────────────────────

    def _glcm_features(
        self, comp: AnnualComposite, H: int, W: int
    ) -> tuple[np.ndarray, list[str]]:
        """Compute 5 GLCM statistics on 3 bands = 15 features + 6 local variance."""
        stats = ["contrast", "homogeneity", "entropy", "correlation", "asm"]
        bands = ["NIR", "SWIR1", "VV"]
        names = [f"GLCM_{s}_{b}" for b in bands for s in stats]
        # 5 stats × 3 bands = 15 GLCM + 3×2 local variance = 21 features
        var_names = [f"localvar_{b}" for b in ["NIR", "SWIR1"]] + \
                    [f"localvar2_{b}" for b in ["NIR", "SWIR1"]] + \
                    ["localvar_VV", "localvar2_VV"]
        all_names = names + var_names
        out = np.full((H, W, len(all_names)), np.nan, dtype="float32")

        ds = comp.landsat if comp.landsat is not None else comp.sentinel2
        if ds is None:
            return out, all_names

        source_bands_map = {
            "NIR":   ["nir", "SR_B5", "SR_B4", "B08"],
            "SWIR1": ["swir1", "SR_B6", "SR_B5", "B11"],
        }
        sar_ds = comp.sentinel1

        raw_bands: dict[str, Optional[np.ndarray]] = {}
        for bname, candidates in source_bands_map.items():
            raw_bands[bname] = _band(ds, candidates, H, W)
        raw_bands["VV"] = (
            _band(sar_ds, ["vv_db_mean", "vv_mean_db"], H, W)
            if sar_ds is not None else None
        )

        col_idx = 0
        for band_arr in raw_bands.values():
            if band_arr is None:
                col_idx += len(stats)
                continue
            # Quantise to 32 levels for GLCM approximation
            vmin = np.nanpercentile(band_arr, 2)
            vmax = np.nanpercentile(band_arr, 98)
            q = ((band_arr - vmin) / (vmax - vmin + 1e-10) * 31).clip(0, 31).astype("uint8")
            glcm_feats = _compute_glcm_features(q, self.glcm_window)
            for j, f in enumerate(glcm_feats):
                out[:, :, col_idx + j] = _resize_to(f, H, W)
            col_idx += len(stats)

        # Local variance (simpler texture proxy)
        var_col = col_idx
        for band_arr in raw_bands.values():
            if band_arr is None:
                var_col += 2
                continue
            mean = uniform_filter(band_arr, size=self.glcm_window, mode="nearest")
            sq_mean = uniform_filter(band_arr**2, size=self.glcm_window, mode="nearest")
            variance = (sq_mean - mean**2).clip(0)
            out[:, :, var_col] = variance
            out[:, :, var_col + 1] = np.sqrt(variance)
            var_col += 2

        return out, all_names


# ── Utility functions ─────────────────────────────────────────────────────────

def _band(
    ds: Optional[xr.Dataset],
    candidates: list[str],
    H: int,
    W: int,
) -> np.ndarray:
    """Extract a band from a Dataset by trying candidate names in order."""
    if ds is None:
        return np.full((H, W), np.nan, dtype="float32")
    for name in candidates:
        if name in ds:
            arr = ds[name].values
            arr = arr.squeeze() if arr.ndim > 2 else arr
            return _resize_to(arr.astype("float32"), H, W)
    return np.full((H, W), np.nan, dtype="float32")


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Compute num/den, returning NaN where denominator is near zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(den) > 1e-10, num / den, np.nan)
    return result.astype("float32")


def _resize_to(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize 2-D array to (H, W) using bilinear interpolation."""
    if arr.shape == (H, W):
        return arr
    from PIL import Image
    img = Image.fromarray(arr.astype("float32"))
    img_r = img.resize((W, H), Image.Resampling.BILINEAR)
    return np.array(img_r, dtype="float32")


def _get_shape(comp: AnnualComposite) -> Optional[tuple[int, int]]:
    """Return (H, W) from the first non-None raster in *comp*."""
    for ds in (comp.landsat, comp.sentinel2, comp.sentinel1):
        if ds is not None:
            for var in ds.data_vars:
                arr = ds[var].values
                arr = arr.squeeze()
                if arr.ndim == 2:
                    return arr.shape
    if comp.dem is not None:
        arr = comp.dem.values.squeeze()
        if arr.ndim == 2:
            return arr.shape  # type: ignore[return-value]
    return None


def _compute_glcm_features(q: np.ndarray, window: int) -> list[np.ndarray]:
    """Approximate GLCM statistics using local uniform filters.

    Returns 5 arrays: contrast, homogeneity, entropy, correlation, ASM.
    """
    q_f = q.astype("float32")
    q_sq = q_f**2
    mean = uniform_filter(q_f, size=window, mode="nearest")
    mean_sq = uniform_filter(q_sq, size=window, mode="nearest")
    variance = (mean_sq - mean**2).clip(0)
    std = np.sqrt(variance) + 1e-8

    # Contrast proxy: local variance
    contrast = variance

    # Homogeneity proxy: 1 / (1 + variance)
    homogeneity = 1.0 / (1.0 + variance)

    # Entropy proxy: computed from local histogram statistics
    # We use the approximation: entropy ≈ log(std + 1)
    entropy = np.log(std + 1.0)

    # Correlation proxy: normalised mean using local stats
    correlation = (mean - mean.mean()) / (std + 1e-8)

    # ASM proxy: 1 / (variance + 1)^2
    asm = 1.0 / (variance + 1.0)**2

    return [contrast, homogeneity, entropy, correlation, asm]
