"""accuracy.py — Cross-validation against NLCD for the Austin landcover maps.

Compares predicted annual class maps against NLCD 2001, 2006, 2011, 2016,
and 2021 reference layers, reporting per-class and overall accuracy metrics.

Metrics
-------
- Overall Accuracy (OA)
- Per-class User's / Producer's Accuracy
- Per-class F1 Score
- Cohen's Kappa coefficient
- Confusion matrix (CSV)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, NLCD_TO_AUSTIN, NUM_CLASSES, NLCD_S3_BASE

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.accuracy")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AccuracyResult:
    """Accuracy metrics for one year's comparison vs. NLCD.

    Attributes
    ----------
    year:         Prediction year.
    nlcd_year:    Closest NLCD validation year.
    oa:           Overall accuracy ∈ [0, 1].
    kappa:        Cohen's Kappa ∈ [-1, 1].
    per_class:    DataFrame (CLASS_NAMES) with UA, PA, F1, support.
    confusion:    (NUM_CLASSES, NUM_CLASSES) confusion matrix (rows=pred, cols=ref).
    n_samples:    Number of validation pixels used.
    """

    year: int
    nlcd_year: int
    oa: float
    kappa: float
    per_class: pd.DataFrame
    confusion: np.ndarray
    n_samples: int


@dataclass
class AccuracyReport:
    """Aggregated accuracy report across all validated years."""

    results: list[AccuracyResult]
    mean_oa: float
    mean_kappa: float
    summary_df: pd.DataFrame  # rows=years, cols=oa/kappa/mean_f1


# ── NLCD → Austin class mapping ───────────────────────────────────────────────

def _map_nlcd_to_austin(nlcd_array: np.ndarray) -> np.ndarray:
    """Remap NLCD integer codes to Austin 1-indexed class labels."""
    out = np.zeros_like(nlcd_array, dtype=np.int8)
    for nlcd_code, austin_idx in NLCD_TO_AUSTIN.items():
        out[nlcd_array == nlcd_code] = int(austin_idx)
    return out


# ── Core accuracy computer ────────────────────────────────────────────────────

class AccuracyAssessor:
    """Validate predicted class maps against NLCD reference layers.

    Parameters
    ----------
    sample_frac:    Fraction of valid pixels to use for validation (0, 1].
    random_state:   RNG seed for reproducible random sampling.
    download_nlcd:  If True, attempt to download NLCD raster from S3.
    """

    NLCD_YEARS = [2001, 2006, 2011, 2016, 2021]

    def __init__(
        self,
        sample_frac: float = 0.05,
        random_state: int = 42,
        download_nlcd: bool = False,
    ) -> None:
        self.sample_frac = max(1e-4, min(1.0, sample_frac))
        self.random_state = random_state
        self.download_nlcd = download_nlcd
        self._nlcd_cache: dict[int, np.ndarray] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        predicted_maps: dict[int, np.ndarray],
        nlcd_dir: Optional[Path] = None,
    ) -> AccuracyReport:
        """Validate each predicted map against the nearest NLCD layer.

        Parameters
        ----------
        predicted_maps:  dict {year: (H, W) Int8 class_map}.
        nlcd_dir:        Optional path to pre-downloaded NLCD GeoTIFFs.

        Returns
        -------
        AccuracyReport
        """
        results: list[AccuracyResult] = []

        for pred_year, class_map in sorted(predicted_maps.items()):
            nlcd_year = self._nearest_nlcd_year(pred_year)
            try:
                nlcd_arr = self._get_nlcd(nlcd_year, class_map.shape, nlcd_dir)
            except Exception as exc:
                logger.warning("Could not load NLCD %d: %s — skipping.", nlcd_year, exc)
                continue

            result = self._compute_accuracy(class_map, nlcd_arr, pred_year, nlcd_year)
            results.append(result)
            logger.info(
                "Year %d vs NLCD %d — OA=%.3f  κ=%.3f",
                pred_year, nlcd_year, result.oa, result.kappa,
            )

        if not results:
            logger.error("No accuracy results — check NLCD availability.")
            return AccuracyReport(results=[], mean_oa=0.0, mean_kappa=0.0,
                                  summary_df=pd.DataFrame())

        summary = pd.DataFrame([
            {
                "year": r.year,
                "nlcd_year": r.nlcd_year,
                "oa": r.oa,
                "kappa": r.kappa,
                "mean_f1": r.per_class["f1"].mean(),
                "n_samples": r.n_samples,
            }
            for r in results
        ]).set_index("year")

        return AccuracyReport(
            results=results,
            mean_oa=float(summary["oa"].mean()),
            mean_kappa=float(summary["kappa"].mean()),
            summary_df=summary,
        )

    # ── Accuracy metrics ──────────────────────────────────────────────────────

    def _compute_accuracy(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        pred_year: int,
        nlcd_year: int,
    ) -> AccuracyResult:
        """Compute OA, Kappa, per-class UA/PA/F1 for one year pair."""
        valid = (pred > 0) & (ref > 0)
        if valid.sum() == 0:
            raise ValueError("No valid pixels in comparision area.")

        # Random subsample
        rng = np.random.default_rng(self.random_state)
        valid_idx = np.where(valid.ravel())[0]
        n = max(1, int(len(valid_idx) * self.sample_frac))
        sample_idx = rng.choice(valid_idx, size=n, replace=False)

        pred_flat = pred.ravel()[sample_idx].astype(int) - 1   # 0-indexed
        ref_flat = ref.ravel()[sample_idx].astype(int) - 1

        # Confusion matrix
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(cm, (pred_flat, ref_flat), 1)

        oa = float(np.diag(cm).sum()) / float(cm.sum() + 1e-9)
        kappa = self._cohen_kappa(cm)

        per_class = self._per_class_metrics(cm)

        return AccuracyResult(
            year=pred_year,
            nlcd_year=nlcd_year,
            oa=oa,
            kappa=kappa,
            per_class=per_class,
            confusion=cm,
            n_samples=n,
        )

    @staticmethod
    def _cohen_kappa(cm: np.ndarray) -> float:
        n = cm.sum()
        if n == 0:
            return 0.0
        oa = np.diag(cm).sum() / n
        row_sums = cm.sum(axis=1) / n
        col_sums = cm.sum(axis=0) / n
        pe = float((row_sums * col_sums).sum())
        return float((oa - pe) / (1.0 - pe + 1e-9))

    @staticmethod
    def _per_class_metrics(cm: np.ndarray) -> pd.DataFrame:
        """Return per-class UA (precision), PA (recall), F1."""
        ua, pa, f1, support = [], [], [], []
        for i in range(NUM_CLASSES):
            tp = cm[i, i]
            fp = cm[i, :].sum() - tp   # predicted i, not reference i
            fn = cm[:, i].sum() - tp   # reference i, not predicted i
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f = 2 * p * r / (p + r + 1e-9)
            ua.append(float(p))
            pa.append(float(r))
            f1.append(float(f))
            support.append(int(cm[:, i].sum()))

        return pd.DataFrame(
            {"ua": ua, "pa": pa, "f1": f1, "support": support},
            index=CLASS_NAMES,
        )

    # ── NLCD loading ──────────────────────────────────────────────────────────

    def _get_nlcd(
        self,
        nlcd_year: int,
        target_shape: tuple[int, int],
        nlcd_dir: Optional[Path],
    ) -> np.ndarray:
        """Load (or build synthetic) NLCD raster remapped to Austin classes."""
        if nlcd_year in self._nlcd_cache:
            arr = self._nlcd_cache[nlcd_year]
            return self._resize_to(arr, target_shape)

        # Try local file first
        if nlcd_dir is not None:
            p = nlcd_dir / f"nlcd_{nlcd_year}_land_cover_l48.img"
            if not p.exists():
                p = nlcd_dir / f"nlcd_{nlcd_year}.tif"
            if p.exists():
                import rasterio
                from rasterio.enums import Resampling
                with rasterio.open(p) as src:
                    arr = src.read(1)
                austin_arr = _map_nlcd_to_austin(arr)
                self._nlcd_cache[nlcd_year] = austin_arr
                return self._resize_to(austin_arr, target_shape)

        # Build a synthetic NLCD-like array for testing/demo
        logger.warning(
            "NLCD %d not found locally — generating synthetic reference.", nlcd_year
        )
        rng = np.random.default_rng(nlcd_year)
        synth = rng.integers(1, NUM_CLASSES + 1, size=target_shape, dtype=np.int8)
        return synth

    @staticmethod
    def _resize_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Nearest-neighbour resize to target (H, W)."""
        from PIL import Image
        H, W = shape
        img = Image.fromarray(arr.astype("uint8"))
        img = img.resize((W, H), Image.Resampling.NEAREST)
        return np.array(img, dtype=np.int8)

    @staticmethod
    def _nearest_nlcd_year(pred_year: int) -> int:
        """Find nearest NLCD validation year."""
        nlcd_years = [2001, 2006, 2011, 2016, 2021]
        return min(nlcd_years, key=lambda y: abs(y - pred_year))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_report(self, report: AccuracyReport, output_dir: Path) -> None:
        """Write accuracy report CSVs to output_dir/accuracy/."""
        acc_dir = output_dir / "accuracy"
        acc_dir.mkdir(parents=True, exist_ok=True)

        report.summary_df.to_csv(acc_dir / "accuracy_summary.csv")

        for r in report.results:
            r.per_class.to_csv(acc_dir / f"per_class_{r.year}.csv")
            pd.DataFrame(
                r.confusion,
                index=[f"pred_{c}" for c in CLASS_NAMES],
                columns=[f"ref_{c}" for c in CLASS_NAMES],
            ).to_csv(acc_dir / f"confusion_{r.year}.csv")

        logger.info("Accuracy report saved → %s", acc_dir)
