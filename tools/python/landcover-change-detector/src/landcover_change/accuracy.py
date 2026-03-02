"""Accuracy assessment — confusion matrices, Kappa, NLCD comparison.

v1.0 — Quantum Land-Cover Change Detector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import (
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_LABELS,
    NLCD_TO_CLASS,
    NLCD_YEARS,
)

logger = logging.getLogger("geoscripthub.landcover_change.accuracy")


# ── Data Structures ───────────────────────────────────────────────


@dataclass
class ConfusionMetrics:
    """Per-epoch confusion matrix and derived accuracy metrics."""

    year: int
    matrix: np.ndarray           # (NUM_CLASSES, NUM_CLASSES) — predicted × reference
    overall_accuracy: float
    kappa: float
    producers_accuracy: np.ndarray   # per class — recall
    users_accuracy: np.ndarray       # per class — precision
    f1: np.ndarray                   # per class — harmonic mean
    class_names: list[str]


@dataclass
class TemporalConsistency:
    """Temporal smoothness metrics for the stack of classifications."""

    mean_annual_change_rate: float  # average fraction of pixels changing/yr
    max_annual_change_rate: float
    flip_flop_fraction: float       # pixels that change then revert within 3 yr


@dataclass
class AccuracyResult:
    """Full accuracy assessment output."""

    metrics_per_epoch: list[ConfusionMetrics]
    overall_metrics: ConfusionMetrics   # aggregated across all matching years
    temporal_consistency: TemporalConsistency
    nlcd_years_used: list[int]
    sample_count: int


# ── NLCD Fetcher / Simulator ─────────────────────────────────────


def reclassify_nlcd(nlcd_array: np.ndarray) -> np.ndarray:
    """Reclassify NLCD pixel values into our 8-class schema.

    Parameters
    ----------
    nlcd_array:
        Raw NLCD class values (11, 21, 22, 23, 24, 31, 41, 42, 43, …).

    Returns
    -------
    Array of shape matching input, values 0–7.
    """
    out = np.full_like(nlcd_array, fill_value=7, dtype="int32")  # default barren
    for nlcd_code, our_class in NLCD_TO_CLASS.items():
        out[nlcd_array == nlcd_code] = our_class
    return out


def fetch_nlcd_reference(
    year: int,
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
    *,
    cache_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Fetch NLCD reference data for a particular year via MRLC WCS.

    If the network request fails, returns ``None`` so the pipeline
    can continue without reference data.

    Parameters
    ----------
    year:
        NLCD vintage (e.g. 2019).
    bbox:
        (west, south, east, north) in EPSG:4326.
    shape:
        (rows, cols) for target grid.
    cache_dir:
        Where to persist downloaded tiles.

    Returns
    -------
    Reclassified reference array (0–7) or None.
    """
    try:
        import requests
        from scipy.ndimage import zoom

        # MRLC NLCD WCS endpoint
        base_url = "https://www.mrlc.gov/geoserver/mrlc_display/wcs"
        coverage_id = f"NLCD_{year}_Land_Cover_L48"

        params = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "GetCoverage",
            "CoverageId": coverage_id,
            "subset": f"Long({bbox[0]},{bbox[2]})",
            "subsettingCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
            "format": "image/geotiff",
        }
        # Add lat subset
        params["subset"] = [
            f"Long({bbox[0]},{bbox[2]})",
            f"Lat({bbox[1]},{bbox[3]})",
        ]

        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / f"nlcd_{year}.tif"
            if cached.exists():
                import rasterio
                with rasterio.open(cached) as src:
                    raw = src.read(1)
                if raw.shape != shape:
                    raw = zoom(raw, (shape[0] / raw.shape[0], shape[1] / raw.shape[1]),
                               order=0)
                return reclassify_nlcd(raw)

        logger.info("Fetching NLCD %d reference from MRLC...", year)
        resp = requests.get(base_url, params=params, timeout=120)
        if resp.status_code != 200:
            logger.warning("NLCD %d fetch failed: HTTP %d", year, resp.status_code)
            return None

        import tempfile, rasterio
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            raw = src.read(1)

        Path(tmp_path).unlink(missing_ok=True)

        if raw.shape != shape:
            raw = zoom(raw, (shape[0] / raw.shape[0], shape[1] / raw.shape[1]), order=0)

        if cache_dir:
            import shutil
            cached = cache_dir / f"nlcd_{year}.tif"
            # Just save reclassified
            np.save(str(cached).replace(".tif", ".npy"), raw)

        return reclassify_nlcd(raw)

    except Exception as exc:
        logger.warning("NLCD %d reference unavailable: %s", year, exc)
        return None


# ── Confusion Matrix ──────────────────────────────────────────────


def compute_confusion_matrix(
    predicted: np.ndarray,
    reference: np.ndarray,
    year: int,
) -> ConfusionMetrics:
    """Compute confusion matrix and all derived metrics.

    Parameters
    ----------
    predicted:
        Predicted class map (0–7).
    reference:
        Reference (truth) class map (0–7).
    year:
        Year label.

    Returns
    -------
    ConfusionMetrics
    """
    valid = (reference >= 0) & (reference < NUM_CLASSES) & \
            (predicted >= 0) & (predicted < NUM_CLASSES)
    pred_v = predicted[valid].ravel()
    ref_v = reference[valid].ravel()

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype="int64")
    for p, r in zip(pred_v, ref_v):
        matrix[p, r] += 1

    total = matrix.sum()
    overall_accuracy = float(np.trace(matrix)) / max(total, 1)

    # Producer's accuracy (recall) — column-wise
    col_sums = matrix.sum(axis=0).astype("float64")
    producers = np.zeros(NUM_CLASSES, dtype="float64")
    for c in range(NUM_CLASSES):
        producers[c] = matrix[c, c] / max(col_sums[c], 1)

    # User's accuracy (precision) — row-wise
    row_sums = matrix.sum(axis=1).astype("float64")
    users = np.zeros(NUM_CLASSES, dtype="float64")
    for c in range(NUM_CLASSES):
        users[c] = matrix[c, c] / max(row_sums[c], 1)

    # F1
    f1 = np.zeros(NUM_CLASSES, dtype="float64")
    for c in range(NUM_CLASSES):
        p, r = users[c], producers[c]
        f1[c] = 2 * p * r / max(p + r, 1e-12)

    # Cohen's Kappa
    p_o = overall_accuracy
    p_e = float(np.sum(row_sums * col_sums)) / max(total ** 2, 1)
    kappa = (p_o - p_e) / max(1.0 - p_e, 1e-12)

    return ConfusionMetrics(
        year=year,
        matrix=matrix,
        overall_accuracy=overall_accuracy,
        kappa=kappa,
        producers_accuracy=producers,
        users_accuracy=users,
        f1=f1,
        class_names=list(CLASS_NAMES),
    )


# ── Temporal Consistency ──────────────────────────────────────────


def compute_temporal_consistency(
    class_maps: list[np.ndarray],
    years: list[int],
) -> TemporalConsistency:
    """Measure temporal smoothness of the classification stack."""
    n = len(years)
    if n < 2:
        return TemporalConsistency(0.0, 0.0, 0.0)

    n_pixels = class_maps[0].size

    # Annual change rates
    change_rates: list[float] = []
    for i in range(1, n):
        rate = float(np.mean(class_maps[i] != class_maps[i - 1]))
        change_rates.append(rate)

    # Flip-flop: pixel changes at year t but reverts at t+2
    flip_flop_count = 0
    if n >= 3:
        for i in range(n - 2):
            changed = class_maps[i] != class_maps[i + 1]
            reverted = class_maps[i] == class_maps[i + 2]
            flip_flop_count += int(np.sum(changed & reverted))

    flip_flop_fraction = flip_flop_count / max(n_pixels * max(n - 2, 1), 1)

    return TemporalConsistency(
        mean_annual_change_rate=float(np.mean(change_rates)),
        max_annual_change_rate=float(np.max(change_rates)),
        flip_flop_fraction=flip_flop_fraction,
    )


# ── Main Assessment Runner ────────────────────────────────────────


class AccuracyAssessor:
    """Run full accuracy assessment against NLCD reference."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.bbox = bbox
        self.cache_dir = cache_dir

    def assess(
        self,
        classifications: list,  # list of ClassificationResult
        years: list[int],
    ) -> AccuracyResult:
        """Run accuracy assessment.

        Parameters
        ----------
        classifications:
            Annual classification results (class_map, year, …).
        years:
            Sorted list of years matching classifications.

        Returns
        -------
        AccuracyResult with per-epoch and overall metrics.
        """
        class_maps = [c.class_map for c in classifications]
        shape = class_maps[0].shape

        # 1) Temporal consistency
        temporal = compute_temporal_consistency(class_maps, years)
        logger.info(
            "Temporal consistency: mean change %.2f%%/yr, flip-flop %.2f%%",
            temporal.mean_annual_change_rate * 100,
            temporal.flip_flop_fraction * 100,
        )

        # 2) NLCD comparison per matching year
        per_epoch: list[ConfusionMetrics] = []
        all_pred: list[np.ndarray] = []
        all_ref: list[np.ndarray] = []
        nlcd_years_used: list[int] = []

        for nlcd_year in NLCD_YEARS:
            # Find closest classification year
            closest_idx = self._find_closest_year(nlcd_year, years)
            if closest_idx is None:
                continue

            closest_year = years[closest_idx]
            if abs(closest_year - nlcd_year) > 2:
                continue  # skip if too far apart

            ref = fetch_nlcd_reference(
                nlcd_year, self.bbox, shape, cache_dir=self.cache_dir,
            )
            if ref is None:
                continue

            pred = class_maps[closest_idx]
            metrics = compute_confusion_matrix(pred, ref, nlcd_year)
            per_epoch.append(metrics)
            all_pred.append(pred)
            all_ref.append(ref)
            nlcd_years_used.append(nlcd_year)

            logger.info(
                "NLCD %d vs predicted %d: OA=%.1f%%, Kappa=%.3f",
                nlcd_year, closest_year,
                metrics.overall_accuracy * 100,
                metrics.kappa,
            )

        # 3) Aggregate confusion matrix
        if all_pred:
            combined_pred = np.concatenate([p.ravel() for p in all_pred])
            combined_ref = np.concatenate([r.ravel() for r in all_ref])
            overall = compute_confusion_matrix(combined_pred, combined_ref, 0)
        else:
            # No reference data — generate self-consistency metrics
            overall = self._self_consistency_metrics(class_maps, years)

        return AccuracyResult(
            metrics_per_epoch=per_epoch,
            overall_metrics=overall,
            temporal_consistency=temporal,
            nlcd_years_used=nlcd_years_used,
            sample_count=sum(m.matrix.sum() for m in per_epoch),
        )

    def _find_closest_year(
        self, target: int, years: list[int],
    ) -> Optional[int]:
        """Find index of closest year in list."""
        if not years:
            return None
        diffs = [abs(y - target) for y in years]
        return int(np.argmin(diffs))

    def _self_consistency_metrics(
        self,
        class_maps: list[np.ndarray],
        years: list[int],
    ) -> ConfusionMetrics:
        """When no NLCD reference is available, compare consecutive years
        as a proxy for temporal stability (diagonal = agreement)."""
        if len(class_maps) < 2:
            return ConfusionMetrics(
                year=0,
                matrix=np.zeros((NUM_CLASSES, NUM_CLASSES), dtype="int64"),
                overall_accuracy=0.0,
                kappa=0.0,
                producers_accuracy=np.zeros(NUM_CLASSES),
                users_accuracy=np.zeros(NUM_CLASSES),
                f1=np.zeros(NUM_CLASSES),
                class_names=list(CLASS_NAMES),
            )

        # Average split-half: first half as "reference", second as "predicted"
        half = len(class_maps) // 2
        ref_map = class_maps[half - 1]
        pred_map = class_maps[half]
        return compute_confusion_matrix(pred_map, ref_map, years[half])


# ── CSV Export ────────────────────────────────────────────────────


def export_accuracy_csv(result: AccuracyResult, output_dir: Path) -> list[Path]:
    """Export accuracy metrics to CSV files.

    Produces:
    - accuracy_summary.csv — per-epoch OA, Kappa, mean F1
    - confusion_matrix_<year>.csv — full confusion matrix per epoch
    - class_accuracy.csv — per-class producer's, user's, F1

    Returns list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Summary CSV
    path = output_dir / "accuracy_summary.csv"
    lines = ["year,overall_accuracy,kappa,mean_f1"]
    for m in result.metrics_per_epoch:
        lines.append(f"{m.year},{m.overall_accuracy:.4f},{m.kappa:.4f},{m.f1.mean():.4f}")
    # Overall
    om = result.overall_metrics
    lines.append(f"overall,{om.overall_accuracy:.4f},{om.kappa:.4f},{om.f1.mean():.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")
    saved.append(path)

    # Per-epoch confusion matrix
    for m in result.metrics_per_epoch:
        path = output_dir / f"confusion_matrix_{m.year}.csv"
        header = "," + ",".join(CLASS_NAMES)
        rows = [header]
        for i in range(NUM_CLASSES):
            row = CLASS_NAMES[i] + "," + ",".join(str(int(m.matrix[i, j])) for j in range(NUM_CLASSES))
            rows.append(row)
        path.write_text("\n".join(rows), encoding="utf-8")
        saved.append(path)

    # Class accuracy
    path = output_dir / "class_accuracy.csv"
    lines = ["class,producers_accuracy,users_accuracy,f1"]
    for i in range(NUM_CLASSES):
        lines.append(
            f"{CLASS_NAMES[i]},{om.producers_accuracy[i]:.4f},"
            f"{om.users_accuracy[i]:.4f},{om.f1[i]:.4f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    saved.append(path)

    # Temporal consistency
    path = output_dir / "temporal_consistency.csv"
    tc = result.temporal_consistency
    lines = [
        "metric,value",
        f"mean_annual_change_rate,{tc.mean_annual_change_rate:.4f}",
        f"max_annual_change_rate,{tc.max_annual_change_rate:.4f}",
        f"flip_flop_fraction,{tc.flip_flop_fraction:.4f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    saved.append(path)

    logger.info("Exported %d accuracy CSV files to %s", len(saved), output_dir)
    return saved
