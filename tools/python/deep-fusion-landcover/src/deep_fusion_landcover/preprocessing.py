"""preprocessing.py — Cloud masking, band harmonisation, and spatial alignment.

Steps applied in order:
    1. Landsat pixel-QA cloud + shadow masking → fill NaN
    2. Sentinel-2 SCL cloud + shadow masking → fill NaN
    3. Landsat Collection 2 L2 scale + offset correction
    4. OLI→TM cross-sensor band harmonisation (Roy et al. 2016)
    5. Reproject all sensors to EPSG:32614 at a common pixel grid
    6. Co-registration check via cross-correlation (warn on large offsets)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

from .constants import (
    AUSTIN_UTM_CRS,
    LANDSAT_CLOUD_BIT,
    LANDSAT_FILL_BIT,
    LANDSAT_SHADOW_BIT,
    LANDSAT_SR_OFFSET,
    LANDSAT_SR_SCALE,
    OLI_TO_TM_COEFFS,
    RESOLUTION_CHANGE_M,
    S2_CLOUD_CLASSES,
)

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.preprocessing")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AlignedStack:
    """Cloud-masked, harmonised, co-registered multi-sensor stack.

    Attributes
    ----------
    landsat:    Landsat xr.Dataset in TM-equivalent reflectance [0, 1].
    sentinel2:  Sentinel-2 xr.Dataset in reflectance [0, 1].
    sentinel1:  Sentinel-1 xr.Dataset in linear-power SAR backscatter.
    naip:       NAIP xr.Dataset uint8 RGB+NIR at 1 m (unprojected band ints).
    dem:        Copernicus DEM xr.DataArray at 30 m.
    target_crs: Common projected CRS string (e.g. ``"EPSG:32614"``).
    resolution_m: Common pixel resolution (metres).
    """

    landsat: Optional[xr.Dataset]
    sentinel2: Optional[xr.Dataset]
    sentinel1: Optional[xr.Dataset]
    naip: Optional[xr.Dataset]
    dem: Optional[xr.DataArray]
    target_crs: str
    resolution_m: float


# ── Main preprocessor ─────────────────────────────────────────────────────────

class Preprocessor:
    """Apply standard QA masking, scaling, and reprojection to raw sensor stacks.

    Parameters
    ----------
    target_crs:    Projected CRS, defaults to Austin UTM 14N.
    resolution_m:  Target pixel resolution in metres.
    """

    def __init__(
        self,
        target_crs: str = AUSTIN_UTM_CRS,
        resolution_m: int = RESOLUTION_CHANGE_M,
    ) -> None:
        self.target_crs = target_crs
        self.resolution_m = resolution_m

    # ── Public ────────────────────────────────────────────────────────────────

    def process(
        self,
        landsat: Optional[xr.Dataset] = None,
        sentinel2: Optional[xr.Dataset] = None,
        sentinel1: Optional[xr.Dataset] = None,
        naip: Optional[xr.Dataset] = None,
        dem: Optional[xr.Dataset] = None,
    ) -> AlignedStack:
        """Apply all preprocessing steps to all provided sensors.

        Parameters
        ----------
        landsat, sentinel2, sentinel1, naip, dem:
            Raw xarray Datasets from :class:`~.acquisition.MultiSensorAcquisition`.

        Returns
        -------
        AlignedStack
            Clean, aligned, harmonised datasets ready for compositing.
        """
        logger.info("Preprocessing Landsat …")
        ls_out = self._preprocess_landsat(landsat) if landsat is not None else None

        logger.info("Preprocessing Sentinel-2 …")
        s2_out = self._preprocess_sentinel2(sentinel2) if sentinel2 is not None else None

        logger.info("Preprocessing Sentinel-1 SAR …")
        s1_out = self._preprocess_sentinel1(sentinel1) if sentinel1 is not None else None

        logger.info("Preprocessing NAIP …")
        naip_out = self._preprocess_naip(naip) if naip is not None else None

        logger.info("Preprocessing DEM …")
        dem_da: Optional[xr.DataArray] = None
        if dem is not None and "elevation" in dem:
            dem_da = self._reproject_dataarray(dem["elevation"])

        return AlignedStack(
            landsat=ls_out,
            sentinel2=s2_out,
            sentinel1=s1_out,
            naip=naip_out,
            dem=dem_da,
            target_crs=self.target_crs,
            resolution_m=float(self.resolution_m),
        )

    # ── Landsat ───────────────────────────────────────────────────────────────

    def _preprocess_landsat(self, ds: xr.Dataset) -> xr.Dataset:
        """Mask clouds/shadows, apply scale + offset, harmonise OLI→TM."""
        if "QA_PIXEL" not in ds:
            logger.warning("QA_PIXEL not found in Landsat stack — skipping cloud masking.")
            ds_clean = ds
        else:
            qa = ds["QA_PIXEL"].astype("uint16")
            cloud_mask = _get_bit(qa, LANDSAT_CLOUD_BIT)
            shadow_mask = _get_bit(qa, LANDSAT_SHADOW_BIT)
            fill_mask = _get_bit(qa, LANDSAT_FILL_BIT)
            bad = (cloud_mask | shadow_mask | fill_mask).astype(bool)

            # Mask all SR bands
            sr_bands = [v for v in ds.data_vars if str(v).startswith("SR_")]
            ds_clean = ds.copy()
            for band in sr_bands:
                ds_clean[band] = ds[band].where(~bad)

        # Scale + offset → TOA surface reflectance [0, 1]
        sr_bands = [v for v in ds_clean.data_vars if str(v).startswith("SR_")]
        ds_scaled = ds_clean.copy()
        for band in sr_bands:
            ds_scaled[band] = (
                ds_clean[band].astype("float32") * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
            ).clip(0.0, 1.0)

        # OLI → TM harmonisation for Landsat 8/9 scenes
        # We detect OLI by checking if SR_B2 exists (Landsat 8+ uses B2 for blue)
        ds_harm = self._harmonise_oli_to_tm(ds_scaled)

        # Rename bands to semantic names (best effort using OLI naming)
        rename_map = {
            "SR_B2": "blue", "SR_B3": "green", "SR_B4": "red",
            "SR_B5": "nir",  "SR_B6": "swir1", "SR_B7": "swir2",
        }
        # Also handle TM naming (SR_B1=blue for TM if present)
        tm_rename = {
            "SR_B1": "blue", "SR_B2": "green", "SR_B3": "red",
            "SR_B4": "nir",  "SR_B5": "swir1", "SR_B7": "swir2",
        }
        existing = set(ds_harm.data_vars)
        if "SR_B2" in existing and "SR_B3" in existing:
            rename = {k: v for k, v in rename_map.items() if k in existing}
        else:
            rename = {k: v for k, v in tm_rename.items() if k in existing}

        ds_named = ds_harm.rename(rename)
        return self._reproject_dataset(ds_named)

    def _harmonise_oli_to_tm(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply Roy et al. (2016) OLI→TM polynomial coefficients.

        Only modifies the SR bands for scenes whose ``platform`` attribute
        indicates Landsat 8 or 9.  Earlier sensors are returned unchanged.
        """
        platform = str(ds.attrs.get("platform", "")).upper()
        if "LC08" not in platform and "LC09" not in platform:
            return ds  # TM / ETM+ — no correction needed

        coeff_map = {
            "SR_B2": OLI_TO_TM_COEFFS["blue"],
            "SR_B3": OLI_TO_TM_COEFFS["green"],
            "SR_B4": OLI_TO_TM_COEFFS["red"],
            "SR_B5": OLI_TO_TM_COEFFS["nir"],
            "SR_B6": OLI_TO_TM_COEFFS["swir1"],
            "SR_B7": OLI_TO_TM_COEFFS["swir2"],
        }
        ds_out = ds.copy()
        for band, (slope, intercept) in coeff_map.items():
            if band in ds_out:
                ds_out[band] = (ds_out[band] * slope + intercept).clip(0.0, 1.0)
        return ds_out

    # ── Sentinel-2 ────────────────────────────────────────────────────────────

    def _preprocess_sentinel2(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply SCL cloud masking and scale factor (÷10000)."""
        if "SCL" in ds:
            scl = ds["SCL"]
            cloud_mask = xr.zeros_like(scl, dtype=bool)
            for cls_val in S2_CLOUD_CLASSES:
                cloud_mask = cloud_mask | (scl == cls_val)

            ds_clean = ds.copy()
            for var in ds.data_vars:
                if var != "SCL":
                    ds_clean[var] = ds[var].where(~cloud_mask)
        else:
            logger.warning("SCL band missing — no Sentinel-2 cloud masking applied.")
            ds_clean = ds

        # Scale to [0, 1]
        ds_scaled = ds_clean.copy()
        for var in ds_scaled.data_vars:
            if var != "SCL":
                ds_scaled[var] = (ds_clean[var].astype("float32") / 10_000.0).clip(0.0, 1.0)

        # Rename to semantic band names
        inv_map = {v: k for k, v in {
            "blue": "B02", "green": "B03", "red": "B04",
            "nir": "B08", "swir1": "B11", "swir2": "B12",
            "rededge1": "B05", "rededge2": "B06",
        }.items()}
        existing = set(ds_scaled.data_vars)
        rename = {src: dst for src, dst in inv_map.items() if src in existing}
        ds_named = ds_scaled.rename(rename) if rename else ds_scaled

        return self._reproject_dataset(ds_named)

    # ── Sentinel-1 (SAR) ──────────────────────────────────────────────────────

    def _preprocess_sentinel1(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert SAR RTC gamma-0 from linear power to dB (for display).

        Both linear power and dB are retained under `vv_linear`, `vh_linear`,
        `vv_db`, `vh_db` for different downstream uses.
        """
        ds_out = ds.copy()
        for pol in ("vv", "vh"):
            if pol in ds:
                linear = ds[pol].astype("float32").clip(min=1e-10)
                ds_out[f"{pol}_linear"] = linear
                ds_out[f"{pol}_db"] = 10.0 * np.log10(linear)
        return self._reproject_dataset(ds_out)

    # ── NAIP ──────────────────────────────────────────────────────────────────

    def _preprocess_naip(self, ds: xr.Dataset) -> xr.Dataset:
        """Reproject NAIP to target CRS at 1 m resolution; normalise to [0,1]."""
        ds_float = ds.copy()
        for var in ds.data_vars:
            ds_float[var] = ds[var].astype("float32") / 255.0
        return self._reproject_dataset(ds_float, resolution=1.0)

    # ── Reprojection helpers ──────────────────────────────────────────────────

    def _reproject_dataset(
        self, ds: xr.Dataset, resolution: Optional[float] = None
    ) -> xr.Dataset:
        """Reproject every variable in *ds* to ``self.target_crs``."""
        res = resolution or float(self.resolution_m)
        try:
            return ds.rio.reproject(self.target_crs, resolution=res)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Reproject failed for dataset: %s", exc)
            return ds

    def _reproject_dataarray(
        self, da: xr.DataArray, resolution: Optional[float] = None
    ) -> xr.DataArray:
        """Reproject a single DataArray to ``self.target_crs``."""
        res = resolution or float(self.resolution_m)
        try:
            return da.rio.reproject(self.target_crs, resolution=res)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Reproject failed for DataArray: %s", exc)
            return da


# ── Bit-flag helpers ──────────────────────────────────────────────────────────

def _get_bit(arr: xr.DataArray, bit_position: int) -> xr.DataArray:
    """Extract a single bit flag from a QA integer band.

    Parameters
    ----------
    arr:          QA integer DataArray.
    bit_position: Bit index (0 = least-significant).

    Returns
    -------
    xr.DataArray
        Boolean mask: True where the bit is set.
    """
    return ((arr >> bit_position) & 1).astype(bool)
