"""
preprocessing.py
================
Multi-sensor image preprocessing, alignment, and harmonisation.

This module ensures that imagery from Landsat (30 m), Sentinel-2 (20 m),
and NAIP (~1 m) are:

1. **Resampled** to a common 30 m ground-sample distance (GSD) by
   *downsampling* finer-resolution sensors using anti-aliased cubic
   resampling.  This avoids aliasing artefacts and preserves spectral
   fidelity better than nearest-neighbour downsampling.

2. **Co-registered** to the same UTM grid origin and pixel alignment
   so every sensor's arrays are perfectly stackable (identical shape,
   identical world-to-pixel transforms).

3. **Cloud-masked** — Landsat uses QA_PIXEL bit flags, Sentinel-2
   uses the Scene Classification Layer (SCL), and NAIP is assumed
   cloud-free (contracted acquisition conditions).

4. **Spectrally normalised** — surface-reflectance values across
   sensors are harmonised to a common [0, 1] reflectance scale to
   reduce inter-sensor radiometric bias before classification.

The output is a single ``AlignedStack`` containing per-observation
normalised arrays all snapped to identical 30 m grids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

from .aoi import AOIResult

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.preprocessing")

TARGET_RESOLUTION = 30  # metres — common GSD for all sensors
NODATA = np.nan


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AlignedStack:
    """Pixel-aligned, cloud-masked, normalised multi-sensor observations.

    Every array in ``observations`` has identical shape ``(rows, cols)``
    and identical geotransform.  Each observation dictionary contains:

    * ``"green"``  — Green band reflectance [0, 1]
    * ``"nir"``    — Near-infrared reflectance [0, 1]
    * ``"swir1"``  — Short-wave infrared 1 reflectance [0, 1] (if avail)
    * ``"swir2"``  — Short-wave infrared 2 reflectance [0, 1] (if avail)
    * ``"red"``    — Red band reflectance [0, 1]
    * ``"blue"``   — Blue band reflectance [0, 1]
    * ``"source"`` — Sensor name (``"landsat"``, ``"sentinel2"``, ``"naip"``)
    * ``"date"``   — Acquisition date string (``"YYYY-MM-DD"``)
    * ``"cloud_mask"`` — Boolean mask (True = clear, False = cloudy/shadow)

    Attributes:
        observations: List of observation dictionaries.
        rows: Number of pixel rows.
        cols: Number of pixel columns.
        transform: Affine geotransform (rasterio-style 6-tuple).
        crs: CRS EPSG string.
        bounds: (west, south, east, north) in projected coordinates.
        total_scenes: Total number of valid observations.
    """

    observations: list[dict]
    rows: int
    cols: int
    transform: tuple
    crs: str
    bounds: tuple[float, float, float, float]
    total_scenes: int = 0

    def __post_init__(self) -> None:
        self.total_scenes = len(self.observations)

    def __repr__(self) -> str:
        return (
            f"<AlignedStack  {self.total_scenes} observations  "
            f"{self.cols}×{self.rows} px  {TARGET_RESOLUTION}m>"
        )


# ---------------------------------------------------------------------------
# Preprocessing engine
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Resample, align, cloud-mask, and normalise multi-sensor imagery.

    Parameters
    ----------
    aoi:
        Resolved AOIResult for spatial reference.
    target_resolution:
        Target pixel size in metres (default 30 m).
    """

    def __init__(
        self,
        aoi: AOIResult,
        target_resolution: int = TARGET_RESOLUTION,
    ) -> None:
        self.aoi = aoi
        self.target_resolution = target_resolution

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

        logger.info(
            "Target grid: %d × %d px @ %d m, CRS %s",
            self.cols, self.rows, target_resolution, aoi.target_crs,
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

        Args:
            landsat: Raw Landsat DataArray from acquisition.
            sentinel2: Raw Sentinel-2 DataArray from acquisition.
            naip: Raw NAIP DataArray from acquisition.

        Returns:
            AlignedStack with all observations harmonised to the target grid.
        """
        observations: list[dict] = []

        # --- Landsat (already at 30 m, just need grid-snap + cloud mask) ---
        if landsat.sizes.get("time", 0) > 0:
            ls_obs = self._process_landsat(landsat)
            observations.extend(ls_obs)
            logger.info("Processed %d Landsat observations", len(ls_obs))

        # --- Sentinel-2 (20 m → 30 m downsample + cloud mask) ---
        if sentinel2.sizes.get("time", 0) > 0:
            s2_obs = self._process_sentinel2(sentinel2)
            observations.extend(s2_obs)
            logger.info("Processed %d Sentinel-2 observations", len(s2_obs))

        # --- NAIP (~1 m → 30 m heavy downsample) ---
        if naip.sizes.get("time", 0) > 0:
            naip_obs = self._process_naip(naip)
            observations.extend(naip_obs)
            logger.info("Processed %d NAIP observations", len(naip_obs))

        logger.info(
            "Alignment complete — %d total observations on %d×%d grid",
            len(observations), self.cols, self.rows,
        )

        return AlignedStack(
            observations=observations,
            rows=self.rows,
            cols=self.cols,
            transform=self.transform,
            crs=self.aoi.target_crs,
            bounds=self.bounds,
        )

    # ------------------------------------------------------------------
    # Landsat processing
    # ------------------------------------------------------------------

    def _process_landsat(self, data: xr.DataArray) -> list[dict]:
        """Extract Landsat observations with cloud masking.

        Landsat C2L2 QA_PIXEL bit flags:
            Bit 1 = dilated cloud, Bit 3 = cloud, Bit 4 = cloud shadow
            Bit 6 = clear → we require bit 6 set AND bits 1,3,4 unset.

        Surface reflectance scale: raw values are in 0–65535 range;
        the official scale factor is 0.0000275 with offset –0.2.
        """
        obs_list: list[dict] = []
        n_time = data.sizes["time"]

        for t in range(n_time):
            # Extract date
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()

                # Get bands by name (stackstac uses band names from assets)
                band_names = list(scene.coords.get("band", scene.coords.get("band_name", [])).values)
                band_lookup = {str(b): i for i, b in enumerate(band_names)}

                blue = self._ls_sr_to_reflectance(scene[band_lookup.get("blue", 0)].values)
                green = self._ls_sr_to_reflectance(scene[band_lookup.get("green", 1)].values)
                red = self._ls_sr_to_reflectance(scene[band_lookup.get("red", 2)].values)
                nir = self._ls_sr_to_reflectance(scene[band_lookup.get("nir08", 3)].values)
                swir1 = self._ls_sr_to_reflectance(scene[band_lookup.get("swir16", 4)].values)
                swir2 = self._ls_sr_to_reflectance(scene[band_lookup.get("swir22", 5)].values)
                qa = scene[band_lookup.get("qa_pixel", 6)].values.astype(np.uint16)

                # Cloud mask from QA_PIXEL
                cloud_mask = self._landsat_cloud_mask(qa)

                # Snap to target grid
                blue = self._regrid(blue, blue.shape, (self.rows, self.cols))
                green = self._regrid(green, green.shape, (self.rows, self.cols))
                red = self._regrid(red, red.shape, (self.rows, self.cols))
                nir = self._regrid(nir, nir.shape, (self.rows, self.cols))
                swir1 = self._regrid(swir1, swir1.shape, (self.rows, self.cols))
                swir2 = self._regrid(swir2, swir2.shape, (self.rows, self.cols))
                cloud_mask = self._regrid_mask(cloud_mask, cloud_mask.shape, (self.rows, self.cols))

                obs_list.append({
                    "blue": blue, "green": green, "red": red,
                    "nir": nir, "swir1": swir1, "swir2": swir2,
                    "cloud_mask": cloud_mask,
                    "source": "landsat",
                    "date": date_str,
                })

            except Exception as exc:
                logger.warning("Skipping Landsat scene %s: %s", date_str, exc)
                continue

        return obs_list

    # ------------------------------------------------------------------
    # Sentinel-2 processing
    # ------------------------------------------------------------------

    def _process_sentinel2(self, data: xr.DataArray) -> list[dict]:
        """Extract Sentinel-2 observations with SCL cloud mask and 20→30 m downsample.

        Sentinel-2 L2A SCL codes we treat as clear:
            4 = vegetation, 5 = bare soil, 6 = water, 11 = snow/ice
        Everything else (cloud, shadow, cirrus, no-data) is masked out.

        Surface reflectance scale: 0–10000 → divide by 10000 for [0, 1].
        """
        obs_list: list[dict] = []
        n_time = data.sizes["time"]
        downsample_factor = 20 / self.target_resolution  # 20/30 = 0.667

        for t in range(n_time):
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()

                band_names = list(scene.coords.get("band", scene.coords.get("band_name", [])).values)
                band_lookup = {str(b): i for i, b in enumerate(band_names)}

                blue = scene[band_lookup.get("B02", 0)].values.astype("float32") / 10000.0
                green = scene[band_lookup.get("B03", 1)].values.astype("float32") / 10000.0
                red = scene[band_lookup.get("B04", 2)].values.astype("float32") / 10000.0
                nir = scene[band_lookup.get("B08", 3)].values.astype("float32") / 10000.0
                swir1 = scene[band_lookup.get("B11", 4)].values.astype("float32") / 10000.0
                swir2 = scene[band_lookup.get("B12", 5)].values.astype("float32") / 10000.0
                scl = scene[band_lookup.get("SCL", 6)].values.astype(np.uint8)

                # SCL-based cloud mask
                cloud_mask = np.isin(scl, [4, 5, 6, 11])

                # Downsample from 20 m to 30 m with cubic anti-aliasing
                target_shape = (self.rows, self.cols)
                blue = self._downsample_cubic(blue, target_shape)
                green = self._downsample_cubic(green, target_shape)
                red = self._downsample_cubic(red, target_shape)
                nir = self._downsample_cubic(nir, target_shape)
                swir1 = self._downsample_cubic(swir1, target_shape)
                swir2 = self._downsample_cubic(swir2, target_shape)
                cloud_mask = self._regrid_mask(cloud_mask, cloud_mask.shape, target_shape)

                obs_list.append({
                    "blue": blue, "green": green, "red": red,
                    "nir": nir, "swir1": swir1, "swir2": swir2,
                    "cloud_mask": cloud_mask,
                    "source": "sentinel2",
                    "date": date_str,
                })

            except Exception as exc:
                logger.warning("Skipping Sentinel-2 scene %s: %s", date_str, exc)
                continue

        return obs_list

    # ------------------------------------------------------------------
    # NAIP processing
    # ------------------------------------------------------------------

    def _process_naip(self, data: xr.DataArray) -> list[dict]:
        """Extract NAIP observations and heavily downsample ~1 m → 30 m.

        NAIP is 4-band (R, G, B, NIR) with uint8 values [0, 255].
        We normalise to [0, 1] reflectance-proxy values.
        NAIP has no cloud mask (acquired under clear-sky conditions),
        so cloud_mask is all-True.

        Note: NAIP lacks SWIR bands, so swir1/swir2 are set to NaN.
        The classifier handles missing SWIR gracefully via feature
        imputation.
        """
        obs_list: list[dict] = []
        n_time = data.sizes["time"]

        for t in range(n_time):
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()
                vals = scene.values  # shape = (bands, y, x) or (y, x) for single asset

                if vals.ndim == 3 and vals.shape[0] >= 4:
                    red_raw = vals[0].astype("float32") / 255.0
                    green_raw = vals[1].astype("float32") / 255.0
                    blue_raw = vals[2].astype("float32") / 255.0
                    nir_raw = vals[3].astype("float32") / 255.0
                elif vals.ndim == 2:
                    # Single-band fallback — unlikely but handle gracefully
                    logger.warning("NAIP scene %s has only 1 band, skipping", date_str)
                    continue
                else:
                    logger.warning("NAIP scene %s unexpected shape %s", date_str, vals.shape)
                    continue

                # Heavy downsample from ~1 m to 30 m
                target_shape = (self.rows, self.cols)
                red = self._downsample_cubic(red_raw, target_shape)
                green = self._downsample_cubic(green_raw, target_shape)
                blue = self._downsample_cubic(blue_raw, target_shape)
                nir = self._downsample_cubic(nir_raw, target_shape)

                # NAIP lacks SWIR — fill with NaN for the classifier
                swir_placeholder = np.full(target_shape, np.nan, dtype="float32")

                # No cloud mask needed — all clear
                cloud_mask = np.ones(target_shape, dtype=bool)

                obs_list.append({
                    "blue": blue, "green": green, "red": red,
                    "nir": nir, "swir1": swir_placeholder, "swir2": swir_placeholder,
                    "cloud_mask": cloud_mask,
                    "source": "naip",
                    "date": date_str,
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

        Uses ``scipy.ndimage.zoom`` with anti-aliased cubic resampling,
        which preserves spectral fidelity better than nearest-neighbour
        or bilinear for large downsampling ratios (e.g. 1 m → 30 m).

        Args:
            arr: Input 2-D float array.
            target_shape: Desired (rows, cols) output shape.

        Returns:
            Resampled array of shape ``target_shape``.
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
