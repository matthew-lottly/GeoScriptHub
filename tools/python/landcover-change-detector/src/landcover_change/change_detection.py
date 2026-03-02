"""Change detection engine — transition matrices, decade maps, trends.

v1.0 — Quantum Land-Cover Change Detector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import NUM_CLASSES, CLASS_NAMES, CLASS_LABELS, EPOCHS
from .quantum_classifier import ClassificationResult, apply_transition_constraints

logger = logging.getLogger("geoscripthub.landcover_change.change_detection")


@dataclass
class TransitionMatrix:
    """Land-cover transition counts between two periods."""

    from_year: int
    to_year: int
    matrix: np.ndarray        # (NUM_CLASSES, NUM_CLASSES) — pixel counts
    area_ha_matrix: np.ndarray  # (NUM_CLASSES, NUM_CLASSES) — hectares
    pixel_size_m: float


@dataclass
class ChangeMap:
    """Change map between two time periods."""

    from_year: int
    to_year: int
    from_class: np.ndarray     # class map at from_year
    to_class: np.ndarray       # class map at to_year
    change_mask: np.ndarray    # bool — True where class changed
    change_code: np.ndarray    # from*NUM_CLASSES + to (encoded transition)
    shape: tuple[int, int]


@dataclass
class DecadeSummary:
    """Change summary for one decade period."""

    label: str                  # e.g. "1990s → 2000s"
    from_year: int
    to_year: int
    transition: TransitionMatrix
    change_map: ChangeMap
    gain_ha: dict[str, float]   # class_name → hectares gained
    loss_ha: dict[str, float]   # class_name → hectares lost
    net_ha: dict[str, float]    # class_name → net change (+ = gain)


@dataclass
class TrendResult:
    """Per-pixel temporal trend classification."""

    dominant_class: np.ndarray   # most frequent class over all years
    trend_label: np.ndarray      # categorical: urbanising, greening, stable, etc.
    class_frequency: np.ndarray  # (rows, cols, NUM_CLASSES) — fraction of years
    change_count: np.ndarray     # how many year-to-year changes per pixel


@dataclass
class ChangeDetectionResult:
    """Full change detection output."""

    yearly_maps: list[ClassificationResult]
    years: list[int]
    then_and_now: ChangeMap
    decade_summaries: list[DecadeSummary]
    transition_matrices: list[TransitionMatrix]
    trend: TrendResult
    shape: tuple[int, int]
    resolution: float


class ChangeDetectionEngine:
    """Compute land-cover change from a stack of annual classifications."""

    def __init__(self, resolution: float = 30.0) -> None:
        self.resolution = resolution
        self.pixel_area_ha = (resolution ** 2) / 10000.0  # m² → ha

    def compute(
        self,
        classifications: list[ClassificationResult],
    ) -> ChangeDetectionResult:
        """Run full change detection analysis.

        Parameters
        ----------
        classifications:
            List of ClassificationResult, one per year, sorted by year.

        Returns
        -------
        ChangeDetectionResult with all change products.
        """
        years = [c.year for c in classifications]
        class_maps = [c.class_map for c in classifications]

        # Apply physics constraints
        class_maps = apply_transition_constraints(class_maps, years)
        # Update the classification results with constrained maps
        for i, cm in enumerate(class_maps):
            classifications[i].class_map = cm

        shape = class_maps[0].shape

        # Then-and-now
        then_and_now = self._compute_change_map(
            class_maps[0], class_maps[-1], years[0], years[-1],
        )
        logger.info(
            "Then-and-now: %d → %d, %.1f%% changed",
            years[0], years[-1],
            100.0 * np.mean(then_and_now.change_mask),
        )

        # Decade summaries
        decade_summaries = self._compute_decade_summaries(
            class_maps, years,
        )

        # All consecutive transition matrices
        transitions: list[TransitionMatrix] = []
        for i in range(len(years) - 1):
            tm = self._compute_transition_matrix(
                class_maps[i], class_maps[i + 1], years[i], years[i + 1],
            )
            transitions.append(tm)

        # Temporal trend
        trend = self._compute_trend(class_maps, years)

        return ChangeDetectionResult(
            yearly_maps=classifications,
            years=years,
            then_and_now=then_and_now,
            decade_summaries=decade_summaries,
            transition_matrices=transitions,
            trend=trend,
            shape=shape,
            resolution=self.resolution,
        )

    def _compute_change_map(
        self,
        from_map: np.ndarray,
        to_map: np.ndarray,
        from_year: int,
        to_year: int,
    ) -> ChangeMap:
        """Build a change map between two classification maps."""
        change_mask = from_map != to_map
        change_code = from_map * NUM_CLASSES + to_map
        return ChangeMap(
            from_year=from_year,
            to_year=to_year,
            from_class=from_map,
            to_class=to_map,
            change_mask=change_mask,
            change_code=np.where(change_mask, change_code, -1).astype("int32"),
            shape=from_map.shape,
        )

    def _compute_transition_matrix(
        self,
        from_map: np.ndarray,
        to_map: np.ndarray,
        from_year: int,
        to_year: int,
    ) -> TransitionMatrix:
        """Compute the N×N transition count matrix."""
        matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype="int64")
        for fr in range(NUM_CLASSES):
            for to in range(NUM_CLASSES):
                matrix[fr, to] = np.sum((from_map == fr) & (to_map == to))

        area_matrix = matrix.astype("float64") * self.pixel_area_ha

        return TransitionMatrix(
            from_year=from_year,
            to_year=to_year,
            matrix=matrix,
            area_ha_matrix=area_matrix,
            pixel_size_m=self.resolution,
        )

    def _compute_decade_summaries(
        self,
        class_maps: list[np.ndarray],
        years: list[int],
    ) -> list[DecadeSummary]:
        """Compute change summaries per decade boundary."""
        summaries: list[DecadeSummary] = []

        # Define decade boundaries
        boundaries = [
            ("1990s → 2000s", 1990, 1999, 2000, 2009),
            ("2000s → 2010s", 2000, 2009, 2010, 2019),
            ("2010s → 2020s", 2010, 2019, 2020, 2030),
        ]

        for label, d1_start, d1_end, d2_start, d2_end in boundaries:
            # Find the closest year to the end of each decade
            d1_years = [i for i, y in enumerate(years) if d1_start <= y <= d1_end]
            d2_years = [i for i, y in enumerate(years) if d2_start <= y <= d2_end]

            if not d1_years or not d2_years:
                continue

            # Use latest year in decade 1, earliest in decade 2
            from_idx = d1_years[-1]
            to_idx = d2_years[0]

            from_map = class_maps[from_idx]
            to_map = class_maps[to_idx]
            from_year = years[from_idx]
            to_year = years[to_idx]

            transition = self._compute_transition_matrix(
                from_map, to_map, from_year, to_year,
            )
            change_map = self._compute_change_map(
                from_map, to_map, from_year, to_year,
            )

            # Compute gain/loss per class
            gain_ha: dict[str, float] = {}
            loss_ha: dict[str, float] = {}
            net_ha: dict[str, float] = {}

            for cls in range(NUM_CLASSES):
                name = CLASS_NAMES[cls]
                gained = transition.area_ha_matrix[:, cls].sum() - transition.area_ha_matrix[cls, cls]
                lost = transition.area_ha_matrix[cls, :].sum() - transition.area_ha_matrix[cls, cls]
                gain_ha[name] = float(gained)
                loss_ha[name] = float(lost)
                net_ha[name] = float(gained - lost)

            summaries.append(DecadeSummary(
                label=label,
                from_year=from_year,
                to_year=to_year,
                transition=transition,
                change_map=change_map,
                gain_ha=gain_ha,
                loss_ha=loss_ha,
                net_ha=net_ha,
            ))

            logger.info(
                "%s (%d→%d): %.1f%% area changed",
                label, from_year, to_year,
                100.0 * np.mean(change_map.change_mask),
            )

        return summaries

    def _compute_trend(
        self,
        class_maps: list[np.ndarray],
        years: list[int],
    ) -> TrendResult:
        """Compute per-pixel temporal trend classification."""
        shape = class_maps[0].shape
        n_years = len(years)

        # Class frequency
        class_freq = np.zeros((*shape, NUM_CLASSES), dtype="float32")
        for cm in class_maps:
            for cls in range(NUM_CLASSES):
                class_freq[:, :, cls] += (cm == cls).astype("float32")
        class_freq /= max(n_years, 1)

        # Dominant class (most frequent)
        dominant = np.argmax(class_freq, axis=-1).astype("int32")

        # Count year-to-year changes
        change_count = np.zeros(shape, dtype="int32")
        for i in range(1, n_years):
            change_count += (class_maps[i] != class_maps[i - 1]).astype("int32")

        # Trend labelling
        # 0=stable, 1=urbanising, 2=greening, 3=degrading, 4=variable
        trend_label = np.zeros(shape, dtype="int32")  # default stable

        # Urbanising: increasing developed_low or developed_high
        early_dev = np.zeros(shape, dtype="float32")
        late_dev = np.zeros(shape, dtype="float32")
        half = n_years // 2
        for cm in class_maps[:half]:
            early_dev += ((cm == 5) | (cm == 6)).astype("float32")
        for cm in class_maps[half:]:
            late_dev += ((cm == 5) | (cm == 6)).astype("float32")
        if half > 0:
            early_dev /= max(half, 1)
            late_dev /= max(n_years - half, 1)
        trend_label[late_dev > early_dev + 0.2] = 1  # urbanising

        # Greening: increasing forest/shrub
        early_veg = np.zeros(shape, dtype="float32")
        late_veg = np.zeros(shape, dtype="float32")
        for cm in class_maps[:half]:
            early_veg += ((cm == 2) | (cm == 3)).astype("float32")
        for cm in class_maps[half:]:
            late_veg += ((cm == 2) | (cm == 3)).astype("float32")
        if half > 0:
            early_veg /= max(half, 1)
            late_veg /= max(n_years - half, 1)
        trend_label[(late_veg > early_veg + 0.2) & (trend_label == 0)] = 2

        # Degrading: decreasing vegetation
        trend_label[(early_veg > late_veg + 0.2) & (trend_label == 0)] = 3

        # Variable: many changes
        trend_label[(change_count > n_years * 0.4) & (trend_label == 0)] = 4

        return TrendResult(
            dominant_class=dominant,
            trend_label=trend_label,
            class_frequency=class_freq,
            change_count=change_count,
        )

    # ── GeoTIFF Export ────────────────────────────────────────────

    def save_rasters(
        self,
        result: ChangeDetectionResult,
        output_dir: Path,
        transform: tuple,
        crs: str,
    ) -> list[Path]:
        """Save all raster outputs as GeoTIFFs."""
        import rasterio
        from rasterio.transform import Affine

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        affine = Affine(
            transform[1], transform[2], transform[0],
            transform[4], transform[5], transform[3],
        )
        profile = {
            "driver": "GTiff",
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "crs": crs,
            "transform": affine,
            "width": result.shape[1],
            "height": result.shape[0],
        }

        # Annual class maps
        for cr in result.yearly_maps:
            path = output_dir / f"landcover_{cr.year}.tif"
            with rasterio.open(path, "w", dtype="uint8", count=1, **profile) as dst:
                dst.write(cr.class_map.astype("uint8"), 1)
                dst.set_band_description(1, f"Land cover {cr.year}")
            saved.append(path)

        # Then-and-now
        path = output_dir / f"change_{result.years[0]}_{result.years[-1]}.tif"
        with rasterio.open(path, "w", dtype="uint8", count=2, **profile) as dst:
            dst.write(result.then_and_now.from_class.astype("uint8"), 1)
            dst.write(result.then_and_now.to_class.astype("uint8"), 2)
            dst.set_band_description(1, f"Class {result.years[0]}")
            dst.set_band_description(2, f"Class {result.years[-1]}")
        saved.append(path)

        # Decade change maps
        for ds in result.decade_summaries:
            path = output_dir / f"change_{ds.from_year}_{ds.to_year}.tif"
            with rasterio.open(path, "w", dtype="uint8", count=2, **profile) as dst:
                dst.write(ds.change_map.from_class.astype("uint8"), 1)
                dst.write(ds.change_map.to_class.astype("uint8"), 2)
            saved.append(path)

        # Trend map
        path = output_dir / "trend.tif"
        with rasterio.open(path, "w", dtype="uint8", count=1, **profile) as dst:
            dst.write(result.trend.trend_label.astype("uint8"), 1)
        saved.append(path)

        # Confidence
        if result.yearly_maps:
            latest = result.yearly_maps[-1]
            path = output_dir / "confidence.tif"
            with rasterio.open(path, "w", dtype="float32", count=1, **profile) as dst:
                dst.write(latest.confidence, 1)
            saved.append(path)

        logger.info("Saved %d raster files to %s", len(saved), output_dir)
        return saved
