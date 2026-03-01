"""
flood_engine.py
===============
Temporal Flood Frequency Aggregation Engine.

Takes the per-observation water classifications from the
``QuantumHybridClassifier`` and computes per-pixel **inundation
frequency** — the fraction of valid (cloud-free) observations in
which each pixel was classified as water.

In addition to the raw frequency surface, the engine computes:

* **Permanent water mask** — pixels inundated ≥90 % of the time
  (river channel, permanent lakes).
* **Seasonal flood zone** — pixels inundated 25–90 % of the time
  (seasonally flooded wetlands, floodplain).
* **Rare flood zone** — pixels inundated 5–25 % of the time
  (episodic flooding).
* **Statistical confidence** — Wilson score interval bounds for
  binomial proportion, giving a per-pixel uncertainty estimate on
  the frequency value.

The output frequency raster is a float32 GeoTIFF with values in
[0, 1] representing the fraction of times each pixel was wet.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds

from .quantum_classifier import ClassificationResult
from .preprocessing import AlignedStack

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.flood_engine")

# ---------------------------------------------------------------------------
# Zone thresholds (fraction of observations)
# ---------------------------------------------------------------------------

PERMANENT_THRESHOLD = 0.90   # ≥90 % → permanent water
SEASONAL_LOW = 0.25          # 25–90 % → seasonal flood zone
RARE_LOW = 0.05              # 5–25 % → rare flood zone
# < 5 % → dry land / noise


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FrequencyResult:
    """Per-pixel flood inundation frequency and derived products.

    Attributes:
        frequency: 2-D float32 array [0, 1] — fraction inundated.
        observation_count: 2-D int array — valid (cloud-free) observation count.
        water_count: 2-D int array — times classified as water.
        permanent_mask: Boolean — True where frequency ≥ 90 %.
        seasonal_mask: Boolean — True where 25 % ≤ freq < 90 %.
        rare_mask: Boolean — True where 5 % ≤ freq < 25 %.
        dry_mask: Boolean — True where freq < 5 %.
        confidence_lower: Wilson score lower bound.
        confidence_upper: Wilson score upper bound.
        rows: Grid rows.
        cols: Grid cols.
        transform: Rasterio-compatible affine transform.
        crs: CRS EPSG string.
        bounds: (west, south, east, north).
        sensor_counts: Dict of scene counts per sensor.
    """

    frequency: np.ndarray
    observation_count: np.ndarray
    water_count: np.ndarray
    permanent_mask: np.ndarray
    seasonal_mask: np.ndarray
    rare_mask: np.ndarray
    dry_mask: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    rows: int
    cols: int
    transform: Affine
    crs: str
    bounds: tuple[float, float, float, float]
    sensor_counts: dict

    def __repr__(self) -> str:
        return (
            f"<FrequencyResult  {self.cols}×{self.rows} px  "
            f"obs={int(self.observation_count.max())}  "
            f"permanent={self.permanent_mask.sum()} px  "
            f"seasonal={self.seasonal_mask.sum()} px>"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FloodFrequencyEngine:
    """Aggregate per-observation water classifications into frequency maps.

    Parameters
    ----------
    stack:
        The preprocessed AlignedStack (for grid metadata).
    confidence_level:
        Confidence level for the Wilson score interval (0.95 → 95 %).
    """

    def __init__(
        self,
        stack: AlignedStack,
        confidence_level: float = 0.95,
    ) -> None:
        self.rows = stack.rows
        self.cols = stack.cols
        self.transform_tuple = stack.transform
        self.crs = stack.crs
        self.bounds = stack.bounds
        self.confidence_level = confidence_level

        # Build a proper rasterio Affine from the 6-tuple
        self.transform = from_bounds(
            *stack.bounds, stack.cols, stack.rows,
        )

    def compute(
        self,
        classifications: Sequence[ClassificationResult],
    ) -> FrequencyResult:
        """Compute flood inundation frequency from classified observations.

        Args:
            classifications: List of ClassificationResult from the
                QuantumHybridClassifier.

        Returns:
            FrequencyResult with frequency surface, zone masks, and
            confidence intervals.
        """
        shape = (self.rows, self.cols)

        # Accumulators
        water_count = np.zeros(shape, dtype="int32")
        obs_count = np.zeros(shape, dtype="int32")
        sensor_counts: dict[str, int] = {}

        for cr in classifications:
            # Only count pixels that are cloud-free
            valid = cr.cloud_mask & ~np.isnan(cr.water_probability)

            # Hard threshold for counting
            is_water = cr.water_binary & valid

            water_count += is_water.astype("int32")
            obs_count += valid.astype("int32")

            sensor_counts[cr.source] = sensor_counts.get(cr.source, 0) + 1

        # Frequency = water_count / obs_count (avoid div-by-zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = np.where(
                obs_count > 0,
                water_count.astype("float32") / obs_count.astype("float32"),
                np.nan,
            )

        # Zone masks
        permanent = frequency >= PERMANENT_THRESHOLD
        seasonal = (frequency >= SEASONAL_LOW) & (frequency < PERMANENT_THRESHOLD)
        rare = (frequency >= RARE_LOW) & (frequency < SEASONAL_LOW)
        dry = frequency < RARE_LOW

        # Wilson score confidence interval for binomial proportion
        lower, upper = self._wilson_interval(water_count, obs_count)

        logger.info(
            "Flood frequency computed — %d total observations across %s",
            sum(sensor_counts.values()),
            sensor_counts,
        )
        logger.info(
            "  Permanent: %d px | Seasonal: %d px | Rare: %d px | Dry: %d px",
            permanent.sum(), seasonal.sum(), rare.sum(), dry.sum(),
        )

        return FrequencyResult(
            frequency=frequency,
            observation_count=obs_count,
            water_count=water_count,
            permanent_mask=permanent,
            seasonal_mask=seasonal,
            rare_mask=rare,
            dry_mask=dry,
            confidence_lower=lower,
            confidence_upper=upper,
            rows=self.rows,
            cols=self.cols,
            transform=self.transform,
            crs=self.crs,
            bounds=self.bounds,
            sensor_counts=sensor_counts,
        )

    def _wilson_interval(
        self,
        successes: np.ndarray,
        trials: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Wilson score confidence interval for a binomial proportion.

        The Wilson interval is preferred over the Wald (normal) interval
        for small sample sizes because it does not collapse to zero
        width when p̂ is near 0 or 1.

        Formula:
            p̂ = k/n
            z = z_{α/2}   (e.g. 1.96 for 95 %)
            centre = (p̂ + z²/(2n)) / (1 + z²/n)
            margin = z/(1 + z²/n) · √(p̂(1−p̂)/n + z²/(4n²))

            lower = centre − margin
            upper = centre + margin

        Args:
            successes: Array of water-detection counts.
            trials: Array of valid-observation counts.

        Returns:
            (lower_bound, upper_bound) arrays, float32.
        """
        from scipy.stats import norm as normal_dist

        alpha = 1.0 - self.confidence_level
        z = normal_dist.ppf(1.0 - alpha / 2.0)

        n = trials.astype("float64")
        k = successes.astype("float64")

        with np.errstate(divide="ignore", invalid="ignore"):
            p_hat = np.where(n > 0, k / n, 0.0)

            denominator = 1.0 + z ** 2 / np.where(n > 0, n, 1.0)
            centre = (p_hat + z ** 2 / (2.0 * np.where(n > 0, n, 1.0))) / denominator

            margin = (z / denominator) * np.sqrt(
                p_hat * (1.0 - p_hat) / np.where(n > 0, n, 1.0)
                + z ** 2 / (4.0 * np.where(n > 0, n, 1.0) ** 2)
            )

        lower = np.clip(centre - margin, 0.0, 1.0).astype("float32")
        upper = np.clip(centre + margin, 0.0, 1.0).astype("float32")

        # NaN where no observations
        lower = np.where(n > 0, lower, np.nan)
        upper = np.where(n > 0, upper, np.nan)

        return lower, upper

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_frequency_raster(
        self,
        result: FrequencyResult,
        output_path: Path,
    ) -> Path:
        """Write the flood frequency surface as a GeoTIFF.

        Args:
            result: Computed FrequencyResult.
            output_path: Path for the output GeoTIFF.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": result.cols,
            "height": result.rows,
            "count": 1,
            "crs": result.crs,
            "transform": result.transform,
            "compress": "deflate",
            "nodata": np.nan,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(result.frequency.astype("float32"), 1)
            dst.set_band_description(1, "Flood inundation frequency [0-1]")
            dst.update_tags(
                total_observations=int(result.observation_count.max()),
                sensors=str(result.sensor_counts),
                crs=result.crs,
                tool="quantum-flood-frequency v2.0.0",
                method="Pseudo-Quantum Hybrid AI Classification (QIEC v2.0 — 3-qubit, 10 m SR)",
            )

        logger.info("Frequency raster saved → %s", output_path)
        return output_path

    def save_zone_raster(
        self,
        result: FrequencyResult,
        output_path: Path,
    ) -> Path:
        """Write a categorical flood zone raster as GeoTIFF.

        Zone codes:
            0 = dry land (freq < 5 %)
            1 = rare flood (5–25 %)
            2 = seasonal flood (25–90 %)
            3 = permanent water (≥ 90 %)
            255 = no data

        Args:
            result: Computed FrequencyResult.
            output_path: Path for the output GeoTIFF.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        zones = np.full((result.rows, result.cols), 255, dtype="uint8")
        zones[result.dry_mask] = 0
        zones[result.rare_mask] = 1
        zones[result.seasonal_mask] = 2
        zones[result.permanent_mask] = 3

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": result.cols,
            "height": result.rows,
            "count": 1,
            "crs": result.crs,
            "transform": result.transform,
            "compress": "deflate",
            "nodata": 255,
            "tiled": True,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(zones, 1)
            dst.set_band_description(1, "Flood zone: 0=dry, 1=rare, 2=seasonal, 3=permanent")

        logger.info("Zone raster saved → %s", output_path)
        return output_path

    def save_confidence_raster(
        self,
        result: FrequencyResult,
        output_path: Path,
    ) -> Path:
        """Write a 2-band GeoTIFF with Wilson confidence bounds.

        Band 1 = lower bound, Band 2 = upper bound.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": result.cols,
            "height": result.rows,
            "count": 2,
            "crs": result.crs,
            "transform": result.transform,
            "compress": "deflate",
            "nodata": np.nan,
            "tiled": True,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(result.confidence_lower.astype("float32"), 1)
            dst.write(result.confidence_upper.astype("float32"), 2)
            dst.set_band_description(1, "Wilson 95% CI lower bound")
            dst.set_band_description(2, "Wilson 95% CI upper bound")

        logger.info("Confidence raster saved → %s", output_path)
        return output_path
