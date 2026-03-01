"""
preprocessing.py
================
Multi-sensor image preprocessing, alignment, and harmonisation.

**v2.0 — Super-Resolution Upsampled Pipeline**

This module ensures that imagery from Landsat (30 m), Sentinel-2 (20 m),
and NAIP (~1 m) are processed to a common **10 m** ground-sample
distance (GSD) — sampling **UP** from coarser sensors rather than
downsampling everything to 30 m:

1. **Landsat 30 m → 10 m** — Super-resolved via spectral-guided
   Laplacian pyramid fusion (using Sentinel-2 or NAIP as spatial
   guide) or bicubic upsampling as fallback.

2. **Sentinel-2 SWIR 20 m → 10 m** — Bicubic upsampling of SWIR
   bands to match the 10 m optical bands.  The 10 m bands (B02–B04,
   B08) are used natively without resampling.

3. **NAIP ~1 m → 10 m** — Area-weighted downsampling with anti-
   aliased Gaussian pre-filtering.  This is the *only* sensor we
   downsample, and 10 m preserves far more spatial detail than the
   previous 30 m target.

4. **Co-registered** to the same UTM grid origin and pixel alignment.

5. **Cloud-masked** — Landsat QA_PIXEL, Sentinel-2 SCL, NAIP assumed
   cloud-free.

6. **Spectrally normalised** — surface-reflectance values harmonised
   to [0, 1] scale.

The output is a single ``AlignedStack`` containing per-observation
normalised arrays all snapped to identical **10 m** grids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

from .aoi import AOIResult
from .super_resolution import (
    SuperResolutionEngine,
    SRMethod,
    _upsample_bicubic,
    _downsample_area,
)

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.preprocessing")

# Target resolution — now 10 m (3× finer than v1.0's 30 m)
TARGET_RESOLUTION = 10   # metres — Sentinel-2 native optical
NODATA = np.nan


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AlignedStack:
    """Pixel-aligned, cloud-masked, normalised multi-sensor observations.

    Every array in ``observations`` has identical shape ``(rows, cols)``
    and identical geotransform at **10 m** resolution.

    Each observation dictionary contains:

    * ``"green"``  — Green band reflectance [0, 1]
    * ``"nir"``    — Near-infrared reflectance [0, 1]
    * ``"swir1"``  — Short-wave infrared 1 reflectance [0, 1] (if avail)
    * ``"swir2"``  — Short-wave infrared 2 reflectance [0, 1] (if avail)
    * ``"red"``    — Red band reflectance [0, 1]
    * ``"blue"``   — Blue band reflectance [0, 1]
    * ``"source"`` — Sensor name (``"landsat"``, ``"sentinel2"``, ``"naip"``)
    * ``"date"``   — Acquisition date string (``"YYYY-MM-DD"``)
    * ``"cloud_mask"`` — Boolean mask (True = clear, False = cloudy/shadow)
    * ``"sr_method"`` — Super-resolution method used (str)

    Attributes:
        observations: List of observation dictionaries.
        rows: Number of pixel rows.
        cols: Number of pixel columns.
        resolution: Target resolution in metres.
        transform: Affine geotransform (rasterio-style 6-tuple).
        crs: CRS EPSG string.
        bounds: (west, south, east, north) in projected coordinates.
        total_scenes: Total number of valid observations.
    """

    observations: list[dict]
    rows: int
    cols: int
    resolution: int = TARGET_RESOLUTION
    transform: tuple = ()
    crs: str = ""
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    total_scenes: int = 0

    def __post_init__(self) -> None:
        self.total_scenes = len(self.observations)

    def __repr__(self) -> str:
        return (
            f"<AlignedStack  {self.total_scenes} observations  "
            f"{self.cols}×{self.rows} px  {self.resolution}m>"
        )


# ---------------------------------------------------------------------------
# Preprocessing engine
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Resample, align, cloud-mask, and normalise multi-sensor imagery.

    **v2.0**: Samples UP to 10 m using super-resolution for Landsat,
    preserving 3× more spatial detail than the previous 30 m pipeline.

    Parameters
    ----------
    aoi:
        Resolved AOIResult for spatial reference.
    target_resolution:
        Target pixel size in metres (default 10 m).
    sr_method:
        Super-resolution method for upsampling coarse bands.
    """

    def __init__(
        self,
        aoi: AOIResult,
        target_resolution: int = TARGET_RESOLUTION,
        sr_method: SRMethod = SRMethod.SPECTRAL_GUIDED,
    ) -> None:
        self.aoi = aoi
        self.target_resolution = target_resolution
        self.sr_method = sr_method

        # Derive target grid dimensions from UTM bounding box
        west, south, east, north = aoi.bbox_utm
        self.cols = int(np.ceil((east - west) / target_resolution))
        self.rows = int(np.ceil((north - south) / target_resolution))
        self.transform = (
            west,                      # x origin (top-left)
            target_resolution,         # pixel width
            0.0,                       # row rotation
            north,                     # y origin (top-left)
            0.0,                       # column rotation
            -target_resolution,        # pixel height (negative = north-up)
        )
        self.bounds = (west, south, east, north)

        # Initialise super-resolution engine
        self._sr_engine = SuperResolutionEngine(
            target_resolution=target_resolution,
            method=sr_method,
        )

        logger.info(
            "Target grid: %d × %d px @ %d m (SR method: %s), CRS %s",
            self.cols, self.rows, target_resolution,
            sr_method.name, aoi.target_crs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        landsat: xr.DataArray,
        sentinel2: xr.DataArray,
        naip: xr.DataArray,
    ) -> AlignedStack:
        """Process all sensors and return pixel-aligned observations.

        v2.0: Super-resolves Landsat (30→10 m) and Sentinel-2 SWIR
        (20→10 m) instead of downsampling everything to 30 m.

        Args:
            landsat: Raw Landsat DataArray from acquisition.
            sentinel2: Raw Sentinel-2 DataArray from acquisition.
            naip: Raw NAIP DataArray from acquisition.

        Returns:
            AlignedStack with all observations harmonised to the
            10 m target grid.
        """
        observations: list[dict] = []

        # Process Sentinel-2 first (used as guide for Landsat SR)
        s2_guide_obs: Optional[dict] = None
        if sentinel2.sizes.get("time", 0) > 0:
            s2_obs = self._process_sentinel2(sentinel2)
            observations.extend(s2_obs)
            logger.info("Processed %d Sentinel-2 observations (10 m)", len(s2_obs))
            if s2_obs:
                s2_guide_obs = s2_obs[0]  # use first S2 as spatial guide

        # Process NAIP (potential guide for both Landsat and S2)
        naip_guide_obs: Optional[dict] = None
        if naip.sizes.get("time", 0) > 0:
            naip_obs = self._process_naip(naip)
            observations.extend(naip_obs)
            logger.info("Processed %d NAIP observations (1→10 m downsample)", len(naip_obs))
            if naip_obs:
                naip_guide_obs = naip_obs[0]

        # Process Landsat with super-resolution (30→10 m)
        if landsat.sizes.get("time", 0) > 0:
            # Choose best available guide: NAIP > Sentinel-2
            guide = naip_guide_obs or s2_guide_obs
            ls_obs = self._process_landsat(landsat, guide_obs=guide)
            observations.extend(ls_obs)
            logger.info("Processed %d Landsat observations (30→10 m SR)", len(ls_obs))

        logger.info(
            "Alignment complete — %d total observations on %d×%d grid @ %d m",
            len(observations), self.cols, self.rows, self.target_resolution,
        )

        return AlignedStack(
            observations=observations,
            rows=self.rows,
            cols=self.cols,
            resolution=self.target_resolution,
            transform=self.transform,
            crs=self.aoi.target_crs,
            bounds=self.bounds,
        )

    # ------------------------------------------------------------------
    # Landsat processing — 30 m → 10 m super-resolution
    # ------------------------------------------------------------------

    def _process_landsat(
        self,
        data: xr.DataArray,
        guide_obs: Optional[dict] = None,
    ) -> list[dict]:
        """Extract Landsat observations with SR upsampling to 10 m.

        Cloud-masks using QA_PIXEL, then super-resolves each band
        from 30 m to 10 m using the guide observation (if available)
        for spectral-guided SR.

        Landsat C2L2 surface reflectance scale:
            DN × 0.0000275 − 0.2 → [0, 1] reflectance.
        """
        obs_list: list[dict] = []
        n_time = data.sizes["time"]
        target_shape = (self.rows, self.cols)

        for t in range(n_time):
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()

                band_names = list(
                    scene.coords.get("band", scene.coords.get("band_name", [])).values
                )
                band_lookup = {str(b): i for i, b in enumerate(band_names)}

                blue_raw = self._ls_sr_to_reflectance(scene[band_lookup.get("blue", 0)].values)
                green_raw = self._ls_sr_to_reflectance(scene[band_lookup.get("green", 1)].values)
                red_raw = self._ls_sr_to_reflectance(scene[band_lookup.get("red", 2)].values)
                nir_raw = self._ls_sr_to_reflectance(scene[band_lookup.get("nir08", 3)].values)
                swir1_raw = self._ls_sr_to_reflectance(scene[band_lookup.get("swir16", 4)].values)
                swir2_raw = self._ls_sr_to_reflectance(scene[band_lookup.get("swir22", 5)].values)
                qa = scene[band_lookup.get("qa_pixel", 6)].values.astype(np.uint16)

                cloud_mask = self._landsat_cloud_mask(qa)

                # Build a temp obs dict at native 30 m
                native_obs = {
                    "blue": blue_raw, "green": green_raw, "red": red_raw,
                    "nir": nir_raw, "swir1": swir1_raw, "swir2": swir2_raw,
                    "cloud_mask": cloud_mask,
                    "source": "landsat", "date": date_str,
                }

                # Super-resolve 30 m → 10 m
                sr_obs = self._sr_engine.upscale_observation(
                    native_obs,
                    source_resolution=30,
                    target_shape=target_shape,
                    guide_obs=guide_obs,
                )
                sr_obs["sr_method"] = self.sr_method.name

                obs_list.append(sr_obs)

            except Exception as exc:
                logger.warning("Skipping Landsat scene %s: %s", date_str, exc)
                continue

        return obs_list

    # ------------------------------------------------------------------
    # Sentinel-2 processing — native 10 m optical, 20→10 m SWIR
    # ------------------------------------------------------------------

    def _process_sentinel2(self, data: xr.DataArray) -> list[dict]:
        """Extract Sentinel-2 observations at 10 m resolution.

        B02, B03, B04, B08 are native 10 m — used without resampling.
        B11, B12 (SWIR) are 20 m — bicubic upsampled to 10 m.
        SCL is 20 m — nearest-neighbour upsampled for cloud mask.

        Surface reflectance scale: 0–10000 → /10000 for [0, 1].
        """
        obs_list: list[dict] = []
        n_time = data.sizes["time"]
        target_shape = (self.rows, self.cols)

        for t in range(n_time):
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()

                band_names = list(
                    scene.coords.get("band", scene.coords.get("band_name", [])).values
                )
                band_lookup = {str(b): i for i, b in enumerate(band_names)}

                # 10 m bands (native resolution for optical)
                blue = scene[band_lookup.get("B02", 0)].values.astype("float32") / 10000.0
                green = scene[band_lookup.get("B03", 1)].values.astype("float32") / 10000.0
                red = scene[band_lookup.get("B04", 2)].values.astype("float32") / 10000.0
                nir = scene[band_lookup.get("B08", 3)].values.astype("float32") / 10000.0

                # 20 m SWIR bands → upsample to 10 m
                swir1 = scene[band_lookup.get("B11", 4)].values.astype("float32") / 10000.0
                swir2 = scene[band_lookup.get("B12", 5)].values.astype("float32") / 10000.0
                scl = scene[band_lookup.get("SCL", 6)].values.astype(np.uint8)

                # SCL cloud mask
                cloud_mask = np.isin(scl, [4, 5, 6, 11])

                # Snap all bands to target grid
                blue = _upsample_bicubic(blue, target_shape) if blue.shape != target_shape else blue
                green = _upsample_bicubic(green, target_shape) if green.shape != target_shape else green
                red = _upsample_bicubic(red, target_shape) if red.shape != target_shape else red
                nir = _upsample_bicubic(nir, target_shape) if nir.shape != target_shape else nir

                # SWIR bands: 20 m → 10 m upsampling
                swir1 = _upsample_bicubic(swir1, target_shape)
                swir2 = _upsample_bicubic(swir2, target_shape)

                # Cloud mask: nearest-neighbour upsampling
                cloud_mask = self._regrid_mask(cloud_mask, cloud_mask.shape, target_shape)

                # Clip reflectance to valid range
                blue = np.clip(blue, 0.0, 1.0)
                green = np.clip(green, 0.0, 1.0)
                red = np.clip(red, 0.0, 1.0)
                nir = np.clip(nir, 0.0, 1.0)
                swir1 = np.clip(swir1, 0.0, 1.0)
                swir2 = np.clip(swir2, 0.0, 1.0)

                obs_list.append({
                    "blue": blue, "green": green, "red": red,
                    "nir": nir, "swir1": swir1, "swir2": swir2,
                    "cloud_mask": cloud_mask,
                    "source": "sentinel2",
                    "date": date_str,
                    "sr_method": "native_10m",
                })

            except Exception as exc:
                logger.warning("Skipping Sentinel-2 scene %s: %s", date_str, exc)
                continue

        return obs_list

    # ------------------------------------------------------------------
    # NAIP processing — 1 m → 10 m area-weighted downsample
    # ------------------------------------------------------------------

    def _process_naip(self, data: xr.DataArray) -> list[dict]:
        """Extract NAIP observations and downsample 1 m → 10 m.

        Still a downsample, but at 10 m (not 30 m), NAIP preserves
        far more spatial detail — building footprints, narrow streams,
        individual tree canopies.

        NAIP is 4-band (R, G, B, NIR) with uint8 [0, 255].
        SWIR bands are NaN (not available from aerial imagery).
        """
        obs_list: list[dict] = []
        n_time = data.sizes["time"]
        target_shape = (self.rows, self.cols)

        for t in range(n_time):
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()
                vals = scene.values

                if vals.ndim == 3 and vals.shape[0] >= 4:
                    red_raw = vals[0].astype("float32") / 255.0
                    green_raw = vals[1].astype("float32") / 255.0
                    blue_raw = vals[2].astype("float32") / 255.0
                    nir_raw = vals[3].astype("float32") / 255.0
                elif vals.ndim == 2:
                    logger.warning("NAIP scene %s has only 1 band, skipping", date_str)
                    continue
                else:
                    logger.warning("NAIP scene %s unexpected shape %s", date_str, vals.shape)
                    continue

                # Area-weighted downsample 1 m → 10 m (anti-aliased)
                red = _downsample_area(red_raw, target_shape)
                green = _downsample_area(green_raw, target_shape)
                blue = _downsample_area(blue_raw, target_shape)
                nir = _downsample_area(nir_raw, target_shape)

                # NAIP lacks SWIR
                swir_placeholder = np.full(target_shape, np.nan, dtype="float32")

                # No cloud mask — all clear
                cloud_mask = np.ones(target_shape, dtype=bool)

                obs_list.append({
                    "blue": blue, "green": green, "red": red,
                    "nir": nir, "swir1": swir_placeholder, "swir2": swir_placeholder,
                    "cloud_mask": cloud_mask,
                    "source": "naip",
                    "date": date_str,
                    "sr_method": "area_downsample_10m",
                })

            except Exception as exc:
                logger.warning("Skipping NAIP scene %s: %s", date_str, exc)
                continue

        return obs_list

    # ------------------------------------------------------------------
    # Resampling utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ls_sr_to_reflectance(arr: np.ndarray) -> np.ndarray:
        """Convert Landsat C2L2 SR digital numbers to [0, 1] reflectance.

        Official formula: reflectance = DN × 0.0000275 − 0.2
        Clamped to [0, 1].
        """
        refl = arr.astype("float32") * 0.0000275 - 0.2
        return np.clip(refl, 0.0, 1.0)

    @staticmethod
    def _landsat_cloud_mask(qa: np.ndarray) -> np.ndarray:
        """Derive a boolean clear-sky mask from Landsat QA_PIXEL band.

        Clear pixels have bit 6 set (clear) and bits 1, 3, 4 unset
        (dilated cloud, cloud, cloud shadow).

        Returns:
            Boolean array — True = clear pixel, False = cloudy/shadow.
        """
        dilated_cloud = (qa >> 1) & 1
        cloud = (qa >> 3) & 1
        cloud_shadow = (qa >> 4) & 1
        clear = (qa >> 6) & 1
        return (clear == 1) & (dilated_cloud == 0) & (cloud == 0) & (cloud_shadow == 0)

    @staticmethod
    def _downsample_cubic(
        arr: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Downsample a 2-D array using cubic (order=3) interpolation.

        Kept for backwards compatibility — new code uses
        ``_downsample_area`` for anti-aliased downsampling.
        """
        if arr.shape == target_shape:
            return arr

        zoom_factors = (
            target_shape[0] / arr.shape[0],
            target_shape[1] / arr.shape[1],
        )
        return np.asarray(zoom(arr.astype("float32"), zoom_factors, order=3, mode="nearest"))

    @staticmethod
    def _regrid(
        arr: np.ndarray,
        src_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Regrid an array to the target shape (bilinear for same-res snap)."""
        if src_shape == target_shape:
            return arr
        zoom_factors = (target_shape[0] / src_shape[0], target_shape[1] / src_shape[1])
        return np.asarray(zoom(arr.astype("float32"), zoom_factors, order=1, mode="nearest"))

    @staticmethod
    def _regrid_mask(
        mask: np.ndarray,
        src_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Regrid a boolean mask via nearest-neighbour (no interpolation).

        A pixel is clear only if the majority of contributing source
        pixels were clear — we use order=0 (nearest) to preserve
        binary semantics.
        """
        if src_shape == target_shape:
            return mask
        zoom_factors = (target_shape[0] / src_shape[0], target_shape[1] / src_shape[1])
        return np.asarray(zoom(mask.astype("float32"), zoom_factors, order=0, mode="nearest")) > 0.5
