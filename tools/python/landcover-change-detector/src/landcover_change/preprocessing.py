"""Cloud masking, spatial alignment, and temporal compositing.

v1.0 — Quantum Land-Cover Change Detector

All sensors are aggregated to 30 m (Landsat native) using
area-weighted downsampling. Annual median and greenest-pixel
composites are built per year.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

from .aoi import AOIResult
from .constants import TARGET_RESOLUTION, NODATA, MIN_CLEAR_FRACTION
from .sar_processor import SARProcessor, SARFeatures
from .terrain import TerrainProcessor, TerrainFeatures

logger = logging.getLogger("geoscripthub.landcover_change.preprocessing")


@dataclass
class AnnualComposite:
    """One year of co-registered, cloud-free spectral data."""

    year: int
    blue: np.ndarray       # [0, 1] reflectance
    green: np.ndarray
    red: np.ndarray
    nir: np.ndarray
    swir1: np.ndarray      # NaN for NAIP-only years
    swir2: np.ndarray
    valid_mask: np.ndarray  # bool — True where data exists
    n_observations: int
    sources: list[str]     # e.g. ["landsat", "sentinel2"]


@dataclass
class PreprocessedStack:
    """Full multi-year preprocessed dataset."""

    composites: list[AnnualComposite]
    rows: int
    cols: int
    resolution: int
    transform: tuple[float, ...]
    crs: str
    bounds: tuple[float, float, float, float]
    sar_features: Optional[SARFeatures] = None
    terrain_features: Optional[TerrainFeatures] = None


class ImagePreprocessor:
    """Align, cloud-mask, and composite multi-sensor imagery.

    All sensors aggregated to 30 m. Annual composites built from
    cloud-free observations using per-band pixel-wise median.
    """

    def __init__(
        self,
        aoi: AOIResult,
        target_resolution: int = TARGET_RESOLUTION,
    ) -> None:
        self.aoi = aoi
        self.target_resolution = target_resolution

        west, south, east, north = aoi.bbox_utm
        self.cols = int(np.ceil((east - west) / target_resolution))
        self.rows = int(np.ceil((north - south) / target_resolution))
        self.transform = (
            west, target_resolution, 0.0,
            north, 0.0, -target_resolution,
        )
        self.bounds = (west, south, east, north)

        logger.info(
            "Target grid: %d × %d px @ %d m, CRS %s",
            self.cols, self.rows, target_resolution, aoi.target_crs,
        )

    # ── Public API ────────────────────────────────────────────────

    def align(
        self,
        landsat: xr.DataArray,
        sentinel2: xr.DataArray,
        naip: xr.DataArray,
        sentinel1: Optional[xr.DataArray] = None,
        dem: Optional[xr.DataArray] = None,
        naip_items: Optional[list] = None,
    ) -> PreprocessedStack:
        """Process all sensors, build annual composites.

        Returns a PreprocessedStack with one AnnualComposite per year
        that has sufficient clear observations.
        """
        target_shape = (self.rows, self.cols)

        # Collect per-pixel per-year observations
        obs_by_year: dict[int, list[dict]] = {}

        # Process Landsat
        if landsat.sizes.get("time", 0) > 0:
            ls_obs = self._process_landsat(landsat)
            for obs in ls_obs:
                yr = int(obs["date"][:4])
                obs_by_year.setdefault(yr, []).append(obs)
            logger.info("Processed %d Landsat observations", len(ls_obs))

        # Process Sentinel-2
        if sentinel2.sizes.get("time", 0) > 0:
            s2_obs = self._process_sentinel2(sentinel2)
            for obs in s2_obs:
                yr = int(obs["date"][:4])
                obs_by_year.setdefault(yr, []).append(obs)
            logger.info("Processed %d Sentinel-2 observations", len(s2_obs))

        # Process NAIP
        if naip_items and len(naip_items) > 0:
            naip_obs = self._process_naip_rasterio(naip_items)
            for obs in naip_obs:
                yr = int(obs["date"][:4])
                obs_by_year.setdefault(yr, []).append(obs)
            logger.info("Processed %d NAIP observations", len(naip_obs))

        # Build annual composites
        composites: list[AnnualComposite] = []
        for year in sorted(obs_by_year.keys()):
            obs_list = obs_by_year[year]
            composite = self._build_annual_composite(year, obs_list, target_shape)
            if composite is not None:
                composites.append(composite)

        logger.info(
            "Built %d annual composites (%d–%d)",
            len(composites),
            composites[0].year if composites else 0,
            composites[-1].year if composites else 0,
        )

        # Process SAR
        sar_features: Optional[SARFeatures] = None
        if sentinel1 is not None and sentinel1.sizes.get("time", 0) > 0:
            sar_features = self._process_sentinel1(sentinel1)
            logger.info("Processed SAR composite (%d scenes)", sar_features.n_observations)

        # Process DEM / Terrain
        terrain_features: Optional[TerrainFeatures] = None
        if dem is not None:
            terrain_features = self._process_dem(dem)
            logger.info("Processed terrain features")

        return PreprocessedStack(
            composites=composites,
            rows=self.rows,
            cols=self.cols,
            resolution=self.target_resolution,
            transform=self.transform,
            crs=self.aoi.target_crs,
            bounds=self.bounds,
            sar_features=sar_features,
            terrain_features=terrain_features,
        )

    # ── Landsat processing ────────────────────────────────────────

    def _process_landsat(self, data: xr.DataArray) -> list[dict]:
        """Extract Landsat observations at native 30 m."""
        obs_list: list[dict] = []
        target_shape = (self.rows, self.cols)

        for t in range(data.sizes["time"]):
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()
                band_names = list(
                    scene.coords.get("band", scene.coords.get("band_name", [])).values
                )
                band_lookup = {str(b): i for i, b in enumerate(band_names)}

                blue = self._ls_reflectance(scene[band_lookup.get("blue", 0)].values)
                green = self._ls_reflectance(scene[band_lookup.get("green", 1)].values)
                red = self._ls_reflectance(scene[band_lookup.get("red", 2)].values)
                nir = self._ls_reflectance(scene[band_lookup.get("nir08", 3)].values)
                swir1 = self._ls_reflectance(scene[band_lookup.get("swir16", 4)].values)
                swir2 = self._ls_reflectance(scene[band_lookup.get("swir22", 5)].values)
                qa = scene[band_lookup.get("qa_pixel", 6)].values.astype(np.uint16)

                cloud_mask = self._landsat_cloud_mask(qa)

                # Check clear fraction
                clear_frac = np.mean(cloud_mask)
                if clear_frac < MIN_CLEAR_FRACTION:
                    continue

                obs = {
                    "blue": self._regrid(blue, target_shape),
                    "green": self._regrid(green, target_shape),
                    "red": self._regrid(red, target_shape),
                    "nir": self._regrid(nir, target_shape),
                    "swir1": self._regrid(swir1, target_shape),
                    "swir2": self._regrid(swir2, target_shape),
                    "cloud_mask": self._regrid_mask(cloud_mask, target_shape),
                    "source": "landsat",
                    "date": date_str,
                }
                obs_list.append(obs)

            except Exception as exc:
                logger.warning("Skipping Landsat %s: %s", date_str, exc)

        return obs_list

    # ── Sentinel-2 processing ─────────────────────────────────────

    def _process_sentinel2(self, data: xr.DataArray) -> list[dict]:
        """Extract S2 observations, downsample 10 m → 30 m."""
        obs_list: list[dict] = []
        target_shape = (self.rows, self.cols)

        for t in range(data.sizes["time"]):
            time_val = data.coords["time"].values[t]
            date_str = str(np.datetime_as_string(time_val, unit="D"))

            try:
                scene = data.isel(time=t).compute()
                band_names = list(
                    scene.coords.get("band", scene.coords.get("band_name", [])).values
                )
                band_lookup = {str(b): i for i, b in enumerate(band_names)}

                blue = scene[band_lookup.get("B02", 0)].values.astype("float32") / 10000.0
                green = scene[band_lookup.get("B03", 1)].values.astype("float32") / 10000.0
                red = scene[band_lookup.get("B04", 2)].values.astype("float32") / 10000.0
                nir = scene[band_lookup.get("B08", 3)].values.astype("float32") / 10000.0
                swir1 = scene[band_lookup.get("B11", 4)].values.astype("float32") / 10000.0
                swir2 = scene[band_lookup.get("B12", 5)].values.astype("float32") / 10000.0
                scl = scene[band_lookup.get("SCL", 6)].values.astype(np.uint8)

                cloud_mask = np.isin(scl, [4, 5, 6, 7])  # veg, soil, water, unclass

                clear_frac = np.mean(cloud_mask)
                if clear_frac < MIN_CLEAR_FRACTION:
                    continue

                obs = {
                    "blue": self._downsample(blue, target_shape),
                    "green": self._downsample(green, target_shape),
                    "red": self._downsample(red, target_shape),
                    "nir": self._downsample(nir, target_shape),
                    "swir1": self._downsample(swir1, target_shape),
                    "swir2": self._downsample(swir2, target_shape),
                    "cloud_mask": self._regrid_mask(cloud_mask, target_shape),
                    "source": "sentinel2",
                    "date": date_str,
                }
                obs_list.append(obs)

            except Exception as exc:
                logger.warning("Skipping S2 %s: %s", date_str, exc)

        return obs_list

    # ── NAIP processing ───────────────────────────────────────────

    def _process_naip_rasterio(self, naip_items: list) -> list[dict]:
        """Read NAIP 4-band GeoTIFF via rasterio."""
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds

        obs_list: list[dict] = []
        target_shape = (self.rows, self.cols)
        west, south, east, north = self.bounds

        for item in naip_items:
            try:
                date_str = str(item.datetime.date()) if item.datetime else "unknown"

                href = None
                for key in ("image", "rendered_preview"):
                    if key in item.assets:
                        href = item.assets[key].href
                        break
                if not href:
                    continue

                with rasterio.open(href) as src:
                    target_transform = from_bounds(
                        west, south, east, north, self.cols, self.rows,
                    )
                    dst = np.zeros((4, self.rows, self.cols), dtype="float32")
                    for bi in range(min(src.count, 4)):
                        reproject(
                            source=src.read(bi + 1).astype("float32"),
                            destination=dst[bi],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=self.aoi.target_crs,
                            resampling=Resampling.average,
                        )

                red = np.clip(dst[0] / 255.0, 0, 1)
                green = np.clip(dst[1] / 255.0, 0, 1)
                blue = np.clip(dst[2] / 255.0, 0, 1)
                nir = np.clip(dst[3] / 255.0, 0, 1)

                if red.max() < 0.01:
                    continue

                obs_list.append({
                    "blue": blue, "green": green, "red": red, "nir": nir,
                    "swir1": np.full(target_shape, np.nan, dtype="float32"),
                    "swir2": np.full(target_shape, np.nan, dtype="float32"),
                    "cloud_mask": np.ones(target_shape, dtype=bool),
                    "source": "naip",
                    "date": date_str,
                })

            except Exception as exc:
                logger.warning("Skipping NAIP: %s", exc)

        return obs_list

    # ── Sentinel-1 SAR ────────────────────────────────────────────

    def _process_sentinel1(self, data: xr.DataArray) -> SARFeatures:
        """Build SAR composite from all S1 scenes."""
        sar_proc = SARProcessor(target_shape=(self.rows, self.cols))

        for t in range(data.sizes.get("time", 0)):
            try:
                scene = data.isel(time=t).compute()
                band_names = list(
                    scene.coords.get("band", scene.coords.get("band_name", [])).values
                )
                band_lookup = {str(b): i for i, b in enumerate(band_names)}

                vv = np.maximum(scene[band_lookup.get("vv", 0)].values.astype("float32"), 1e-10)
                vh = np.maximum(scene[band_lookup.get("vh", 1)].values.astype("float32"), 1e-10)
                sar_proc.add_observation(vv, vh)
            except Exception as exc:
                logger.warning("Skipping SAR scene %d: %s", t, exc)

        return sar_proc.compute_features()

    # ── DEM / Terrain ─────────────────────────────────────────────

    def _process_dem(self, data: xr.DataArray) -> Optional[TerrainFeatures]:
        """Process DEM into terrain features."""
        terrain_proc = TerrainProcessor(
            target_shape=(self.rows, self.cols),
            target_resolution=float(self.target_resolution),
        )
        try:
            if "time" in data.dims and data.sizes["time"] > 0:
                arr = data.isel(time=0).compute().values
            else:
                arr = data.compute().values

            while arr.ndim > 2:
                arr = arr[0]

            return terrain_proc.process(arr.astype("float32"), dem_resolution=30.0)
        except Exception as exc:
            logger.warning("DEM processing failed: %s", exc)
            return None

    # ── Annual composite builder ──────────────────────────────────

    def _build_annual_composite(
        self,
        year: int,
        obs_list: list[dict],
        target_shape: tuple[int, int],
    ) -> Optional[AnnualComposite]:
        """Build a pixel-wise median composite from clear observations."""
        bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

        # Stack all clear observations
        stacks: dict[str, list[np.ndarray]] = {b: [] for b in bands}
        valid_masks: list[np.ndarray] = []
        sources: set[str] = set()

        for obs in obs_list:
            mask = obs["cloud_mask"]
            sources.add(obs["source"])
            for b in bands:
                arr = obs[b].copy()
                arr[~mask] = np.nan
                stacks[b].append(arr)
            valid_masks.append(mask)

        if not stacks["blue"]:
            return None

        # Pixel-wise median
        result: dict[str, np.ndarray] = {}
        for b in bands:
            stack = np.stack(stacks[b], axis=0)
            with np.errstate(all="ignore"):
                result[b] = np.nanmedian(stack, axis=0).astype("float32")

        # Valid mask = at least one clear observation
        combined_valid = np.zeros(target_shape, dtype=bool)
        for m in valid_masks:
            combined_valid |= m

        n_obs = len(obs_list)
        if np.sum(combined_valid) < 0.1 * target_shape[0] * target_shape[1]:
            logger.warning("Year %d: <10%% valid pixels, skipping", year)
            return None

        return AnnualComposite(
            year=year,
            blue=result["blue"],
            green=result["green"],
            red=result["red"],
            nir=result["nir"],
            swir1=result["swir1"],
            swir2=result["swir2"],
            valid_mask=combined_valid,
            n_observations=n_obs,
            sources=sorted(sources),
        )

    # ── Utility methods ───────────────────────────────────────────

    @staticmethod
    def _ls_reflectance(arr: np.ndarray) -> np.ndarray:
        """Landsat C2L2 SR DN → [0,1] reflectance."""
        return np.clip(arr.astype("float32") * 0.0000275 - 0.2, 0.0, 1.0)

    @staticmethod
    def _landsat_cloud_mask(qa: np.ndarray) -> np.ndarray:
        """Boolean clear-sky mask from QA_PIXEL."""
        dilated = (qa >> 1) & 1
        cloud = (qa >> 3) & 1
        shadow = (qa >> 4) & 1
        clear = (qa >> 6) & 1
        return (clear == 1) & (dilated == 0) & (cloud == 0) & (shadow == 0)

    def _regrid(self, arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Regrid to target shape (bilinear)."""
        if arr.shape == target_shape:
            return arr
        zf = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
        return np.asarray(zoom(arr.astype("float32"), zf, order=1, mode="nearest"))

    def _downsample(self, arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Area-weighted downsample to target shape."""
        if arr.shape == target_shape:
            return arr
        if arr.shape[0] > target_shape[0]:
            # True downsample — block average
            rh = arr.shape[0] // target_shape[0]
            rw = arr.shape[1] // target_shape[1]
            trimmed = arr[:rh * target_shape[0], :rw * target_shape[1]]
            reshaped = trimmed.reshape(target_shape[0], rh, target_shape[1], rw)
            return np.nanmean(reshaped, axis=(1, 3)).astype("float32")
        return self._regrid(arr, target_shape)

    @staticmethod
    def _regrid_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Nearest-neighbour regrid for boolean masks."""
        if mask.shape == target_shape:
            return mask
        zf = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
        return np.asarray(zoom(mask.astype("float32"), zf, order=0, mode="nearest")) > 0.5
