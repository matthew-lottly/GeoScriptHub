"""
mosaic.py
=========
Temporal mosaicking engine for multi-sensor imagery compositing.

Creates pixel-wise temporal composites from multi-temporal image
stacks, producing cleaner, cloud-free, and noise-reduced mosaics:

1. **Optical median composite** — per-pixel temporal median across
   all cloud-free observations.  Median is robust to outliers
   (clouds, shadows, atmospheric haze) and produces the "typical"
   spectral signature for each pixel.

2. **SAR mean composite** — per-pixel temporal mean across all SAR
   acquisitions.  Multi-temporal averaging is the gold-standard
   speckle reduction technique for SAR imagery (σ ∝ 1/√N).

3. **Best-pixel composite** — selects the observation with maximum
   NDVI (greenest pixel) for each pixel location, optimal for
   cloud-free vegetation analysis.

5× mosaicking
--------------
When fed 5× more imagery (wider temporal range, relaxed cloud filter),
the composites become substantially cleaner:
* Clouds: 5× more chances for cloud-free pixels → near-complete cover
* SAR speckle: 5× more observations → √5 ≈ 2.2× noise reduction
* Seasonal sampling: captures full hydrological cycle

References
----------
* White et al., "Pixel-Based Image Compositing for Large-Area Dense
  Time Series Applications and Science", Can. J. Remote Sensing 40(3),
  2014.
* Quegan & Yu, "Filtering of Multichannel SAR Images", IEEE TGRS
  39(11), 2001.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.mosaic")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MosaicResult:
    """Result of temporal mosaicking.

    Attributes:
        data: Dict of band_name → 2-D composited array.
        method: Compositing method used.
        n_input_observations: Number of input observations.
        n_valid_pixels: Number of pixels with at least one valid obs.
        coverage_fraction: Fraction of pixels with valid data.
    """
    data: dict[str, np.ndarray]
    method: str
    n_input_observations: int
    n_valid_pixels: int = 0
    coverage_fraction: float = 0.0

    def __repr__(self) -> str:
        bands = list(self.data.keys())
        shape = self.data[bands[0]].shape if bands else (0, 0)
        return (
            f"<MosaicResult  {shape[1]}×{shape[0]} px  "
            f"method={self.method}  n_obs={self.n_input_observations}  "
            f"coverage={self.coverage_fraction:.1%}>"
        )


# ---------------------------------------------------------------------------
# Temporal Mosaicker
# ---------------------------------------------------------------------------

class TemporalMosaicker:
    """Create pixel-wise temporal composites from observation stacks.

    Supports different compositing strategies optimised for different
    sensor types and analysis goals.

    The mosaicker works with observation dicts (same format as
    ``preprocessing.AlignedStack.observations``).
    """

    # ------------------------------------------------------------------
    # Median composite (optical)
    # ------------------------------------------------------------------

    @staticmethod
    def median_composite(
        observations: list[dict],
        band_names: Optional[list[str]] = None,
    ) -> MosaicResult:
        """Create a per-pixel median composite from optical observations.

        For each band, stacks all valid (non-NaN, cloud-free) values at
        each pixel and takes the median.  This produces a spectrally
        representative, noise-robust composite.

        Args:
            observations: List of observation dicts with band arrays.
            band_names: Band keys to composite (default: all spectral).

        Returns:
            MosaicResult with median-composited bands.
        """
        if not observations:
            return MosaicResult(data={}, method="median", n_input_observations=0)

        if band_names is None:
            band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]

        n_obs = len(observations)
        shape = observations[0].get("green", observations[0].get("nir", np.empty((1, 1)))).shape

        logger.info(
            "Creating median composite from %d observations (%d×%d px) …",
            n_obs, shape[1], shape[0],
        )

        result_bands: dict[str, np.ndarray] = {}

        for band_name in band_names:
            # Stack all observations for this band
            stack_list: list[np.ndarray] = []

            for obs in observations:
                if band_name not in obs:
                    continue

                arr = obs[band_name].copy()
                cloud_mask = obs.get("cloud_mask", np.ones(shape, dtype=bool))

                # Mask out cloudy and invalid pixels
                arr = np.where(cloud_mask & np.isfinite(arr), arr, np.nan)
                stack_list.append(arr)

            if not stack_list:
                result_bands[band_name] = np.full(shape, np.nan, dtype="float32")
                continue

            stack = np.stack(stack_list, axis=0)

            with np.errstate(all="ignore"):
                median = np.nanmedian(stack, axis=0).astype("float32")

            result_bands[band_name] = median

        # Count valid pixels (at least one valid observation)
        ref_band = result_bands.get("green", list(result_bands.values())[0] if result_bands else np.empty(shape))
        n_valid = int(np.sum(np.isfinite(ref_band)))
        coverage = n_valid / max(ref_band.size, 1)

        logger.info("Median composite: %d valid pixels (%.1f%% coverage)", n_valid, coverage * 100)

        return MosaicResult(
            data=result_bands,
            method="median",
            n_input_observations=n_obs,
            n_valid_pixels=n_valid,
            coverage_fraction=coverage,
        )

    # ------------------------------------------------------------------
    # Mean composite (SAR)
    # ------------------------------------------------------------------

    @staticmethod
    def mean_composite(
        observations: list[dict],
        band_names: Optional[list[str]] = None,
    ) -> MosaicResult:
        """Create a per-pixel mean composite (optimal for SAR speckle reduction).

        Multi-temporal mean reduces speckle noise by √N where N is the
        number of observations.  With 50+ SAR scenes, this produces
        very smooth, low-noise backscatter maps.

        Args:
            observations: List of observation dicts with band arrays.
            band_names: Band keys to composite.

        Returns:
            MosaicResult with mean-composited bands.
        """
        if not observations:
            return MosaicResult(data={}, method="mean", n_input_observations=0)

        if band_names is None:
            band_names = ["vv", "vh"]

        n_obs = len(observations)
        shape = observations[0][band_names[0]].shape if band_names[0] in observations[0] else (1, 1)

        logger.info("Creating mean composite from %d observations …", n_obs)

        result_bands: dict[str, np.ndarray] = {}

        for band_name in band_names:
            stack_list = [obs[band_name] for obs in observations if band_name in obs]
            if not stack_list:
                result_bands[band_name] = np.full(shape, np.nan, dtype="float32")
                continue

            stack = np.stack(stack_list, axis=0)
            mean = np.nanmean(stack, axis=0).astype("float32")
            result_bands[band_name] = mean

        ref = list(result_bands.values())[0] if result_bands else np.empty(shape)
        n_valid = int(np.sum(np.isfinite(ref)))

        return MosaicResult(
            data=result_bands,
            method="mean",
            n_input_observations=n_obs,
            n_valid_pixels=n_valid,
            coverage_fraction=n_valid / max(ref.size, 1),
        )

    # ------------------------------------------------------------------
    # Best-pixel composite (maximum NDVI)
    # ------------------------------------------------------------------

    @staticmethod
    def best_pixel_composite(
        observations: list[dict],
        band_names: Optional[list[str]] = None,
    ) -> MosaicResult:
        """Create a best-pixel composite selecting maximum NDVI per pixel.

        For each pixel, selects the observation with the highest NDVI
        (greenest/clearest).  This produces a cloud-free composite that
        represents peak vegetation conditions.

        Args:
            observations: List of observation dicts.
            band_names: Band keys to include.

        Returns:
            MosaicResult with best-pixel composite.
        """
        if not observations:
            return MosaicResult(data={}, method="best_pixel", n_input_observations=0)

        if band_names is None:
            band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]

        n_obs = len(observations)
        shape = observations[0].get("green", observations[0].get("nir", np.empty((1, 1)))).shape

        logger.info("Creating best-pixel (max NDVI) composite from %d observations …", n_obs)

        # Compute NDVI for each observation to find best pixel
        ndvi_stack = []
        for obs in observations:
            if "nir" in obs and "red" in obs:
                nir = obs["nir"]
                red = obs["red"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndvi = np.where(
                        (nir + red) > 0,
                        (nir - red) / (nir + red),
                        -1.0,
                    )
                cloud_mask = obs.get("cloud_mask", np.ones(shape, dtype=bool))
                ndvi = np.where(cloud_mask, ndvi, -999.0)
            else:
                ndvi = np.full(shape, -999.0)
            ndvi_stack.append(ndvi)

        ndvi_cube = np.stack(ndvi_stack, axis=0)  # (n_obs, rows, cols)
        best_idx = np.argmax(ndvi_cube, axis=0)  # (rows, cols)

        # Select best pixel for each band
        result_bands: dict[str, np.ndarray] = {}
        for band_name in band_names:
            out = np.full(shape, np.nan, dtype="float32")
            for i, obs in enumerate(observations):
                if band_name in obs:
                    mask = best_idx == i
                    out[mask] = obs[band_name][mask]
            result_bands[band_name] = out

        ref = result_bands.get("green", list(result_bands.values())[0] if result_bands else np.empty(shape))
        n_valid = int(np.sum(np.isfinite(ref)))

        return MosaicResult(
            data=result_bands,
            method="best_pixel",
            n_input_observations=n_obs,
            n_valid_pixels=n_valid,
            coverage_fraction=n_valid / max(ref.size, 1),
        )

    # ------------------------------------------------------------------
    # Percentile composite
    # ------------------------------------------------------------------

    @staticmethod
    def percentile_composite(
        observations: list[dict],
        percentile: float = 10.0,
        band_names: Optional[list[str]] = None,
    ) -> MosaicResult:
        """Create a per-pixel percentile composite.

        Low-percentile (e.g., 10th) composites emphasise dark features
        (water) whereas high-percentile (e.g., 90th) composites
        emphasise bright features (clouds, bare soil).

        Useful for creating a "darkest pixel" composite that highlights
        water bodies across the temporal stack.

        Args:
            observations: List of observation dicts.
            percentile: Target percentile (0–100).
            band_names: Band keys to composite.

        Returns:
            MosaicResult with percentile-composited bands.
        """
        if not observations:
            return MosaicResult(
                data={}, method=f"p{percentile:.0f}", n_input_observations=0
            )

        if band_names is None:
            band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]

        n_obs = len(observations)
        shape = observations[0].get("green", observations[0].get("nir", np.empty((1, 1)))).shape

        result_bands: dict[str, np.ndarray] = {}

        for band_name in band_names:
            stack_list: list[np.ndarray] = []
            for obs in observations:
                if band_name not in obs:
                    continue
                arr = obs[band_name].copy()
                cloud = obs.get("cloud_mask", np.ones(shape, dtype=bool))
                arr = np.where(cloud & np.isfinite(arr), arr, np.nan)
                stack_list.append(arr)

            if not stack_list:
                result_bands[band_name] = np.full(shape, np.nan, dtype="float32")
                continue

            stack = np.stack(stack_list, axis=0)
            with np.errstate(all="ignore"):
                result = np.nanpercentile(stack, percentile, axis=0).astype("float32")
            result_bands[band_name] = result

        ref = list(result_bands.values())[0] if result_bands else np.empty(shape)
        n_valid = int(np.sum(np.isfinite(ref)))

        return MosaicResult(
            data=result_bands,
            method=f"p{percentile:.0f}",
            n_input_observations=n_obs,
            n_valid_pixels=n_valid,
            coverage_fraction=n_valid / max(ref.size, 1),
        )
