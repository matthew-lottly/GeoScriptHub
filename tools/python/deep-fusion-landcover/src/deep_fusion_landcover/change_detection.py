"""change_detection.py — Annual landcover change analysis for 35-year time series.

Analysis layers
---------------
1.  Year-pair transition matrices  — 12×12 class transition counts per epoch.
2.  Trajectory classification      — persistent / gain / loss / fluctuating per pixel.
3.  Change intensity analysis      — Aldwaik & Pontius (2012) intensity framework.
4.  Hotspot detection              — pixels with ≥3 class changes 1990–2025.
5.  Urban expansion logistic curve — city4of-Austin growth model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.change_detection")

# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class TransitionMatrix:
    """Single-epoch transition matrix.

    Attributes
    ----------
    from_year, to_year: bracket calendar years.
    matrix:  (NUM_CLASSES, NUM_CLASSES) pixel counts.
             matrix[i, j] = pixels that changed from class i → class j.
    """

    from_year: int
    to_year: int
    matrix: np.ndarray  # (NUM_CLASSES, NUM_CLASSES)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.matrix,
            index=[f"from_{c}" for c in CLASS_NAMES],
            columns=[f"to_{c}" for c in CLASS_NAMES],
        )

    def net_change(self) -> pd.Series:
        """Gain − Loss per class (pixels)."""
        gains = self.matrix.sum(axis=0) - np.diag(self.matrix)
        losses = self.matrix.sum(axis=1) - np.diag(self.matrix)
        return pd.Series(gains - losses, index=CLASS_NAMES)


@dataclass
class ChangeResult:
    """Full change-detection analysis for a 35-year time series.

    Attributes
    ----------
    transitions:    list of TransitionMatrix (one per consecutive year pair).
    trajectory_map: Int8 (H, W) — 0 persistent, 1 gain, 2 loss, 3 fluctuating.
    hotspot_map:    Bool (H, W) — True where ≥3 class changes occurred.
    change_freq_map:Int8 (H, W) — number of class changes per pixel.
    urban_logistic: dict with keys {year, predicted_impervious_fraction}.
    annual_fractions: DataFrame — rows=years, columns=CLASS_NAMES (area fraction).
    intensity_df:   DataFrame — Aldwaik-Pontius annual change intensities.
    """

    transitions: list[TransitionMatrix]
    trajectory_map: np.ndarray
    hotspot_map: np.ndarray
    change_freq_map: np.ndarray
    urban_logistic: dict
    annual_fractions: pd.DataFrame
    intensity_df: pd.DataFrame


# ── Detector ──────────────────────────────────────────────────────────────────

class ChangeDetector:
    """Analyze 35-year annual class_map time series.

    Parameters
    ----------
    min_map_area_ha:  Minimum mapping unit in hectares; smaller changes discarded.
    """

    def __init__(self, min_map_area_ha: float = 0.5) -> None:
        self.min_map_area_ha = min_map_area_ha

    def analyze(
        self,
        class_maps: list[np.ndarray],
        years: list[int],
        pixel_area_ha: float = 1.0,
    ) -> ChangeResult:
        """Build the full change-detection suite.

        Parameters
        ----------
        class_maps:    List of (H, W) Int8 maps ordered by year. Values 1..12.
        years:         Corresponding calendar years (same length as class_maps).
        pixel_area_ha: Area per pixel in hectares.

        Returns
        -------
        ChangeResult
        """
        if len(class_maps) != len(years):
            raise ValueError("class_maps and years must have the same length.")

        logger.info("Running change detection over %d annual maps …", len(years))

        # ── 1. Transition matrices ────────────────────────────────────────────
        transitions: list[TransitionMatrix] = []
        for i in range(len(class_maps) - 1):
            tm = self._build_transition_matrix(
                class_maps[i], class_maps[i + 1], years[i], years[i + 1]
            )
            transitions.append(tm)

        # ── 2. Trajectory & hotspot maps ─────────────────────────────────────
        traj, hotspot, change_freq = self._trajectory_analysis(class_maps)

        # ── 3. Annual area fractions ──────────────────────────────────────────
        annual_fractions = self._annual_fractions(class_maps, years)

        # ── 4. Intensity analysis ─────────────────────────────────────────────
        intensity_df = self._intensity_analysis(transitions, pixel_area_ha)

        # ── 5. Urban logistic fit ─────────────────────────────────────────────
        urban_logistic = self._fit_urban_logistic(annual_fractions, years)

        return ChangeResult(
            transitions=transitions,
            trajectory_map=traj,
            hotspot_map=hotspot,
            change_freq_map=change_freq,
            urban_logistic=urban_logistic,
            annual_fractions=annual_fractions,
            intensity_df=intensity_df,
        )

    # ── Internal methods ──────────────────────────────────────────────────────

    @staticmethod
    def _build_transition_matrix(
        map_from: np.ndarray, map_to: np.ndarray, y0: int, y1: int
    ) -> TransitionMatrix:
        """Build a count matrix from one year-pair of class maps."""
        H, W = map_from.shape
        valid = (map_from > 0) & (map_to > 0)
        f = map_from[valid].astype(int) - 1  # 0-indexed
        t = map_to[valid].astype(int) - 1
        matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(matrix, (f, t), 1)
        return TransitionMatrix(from_year=y0, to_year=y1, matrix=matrix)

    @staticmethod
    def _trajectory_analysis(
        maps: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify per-pixel trajectory over the full time series.

        Returns trajectory_map (0=persistent, 1=gain[last≠first],
        2=loss[last≠first], 3=fluctuating), hotspot_map (bool, ≥3 changes),
        change_freq_map (int8, n changes).
        """
        stack = np.stack(maps, axis=0).astype(np.int8)  # (T, H, W)
        T, H, W = stack.shape

        # Number of class-changes along time axis
        change_events = (np.diff(stack, axis=0) != 0).sum(axis=0).astype(np.int8)

        hotspot = change_events >= 3
        first = stack[0]
        last = stack[-1]

        traj = np.zeros((H, W), dtype=np.int8)  # persistent by default
        impervious_classes = {8, 9}  # High-dev / Impervious (1-indexed ints)

        changed = first != last
        # Gain = ended in impervious or development class; loss = started there
        gained_developed = changed & np.isin(last, list(impervious_classes))
        lost_developed = changed & np.isin(first, list(impervious_classes))

        traj[changed & ~gained_developed & ~lost_developed] = 3  # fluctuating
        traj[gained_developed] = 1  # gain
        traj[lost_developed] = 2  # loss

        return traj, hotspot, change_events

    @staticmethod
    def _annual_fractions(
        maps: list[np.ndarray], years: list[int]
    ) -> pd.DataFrame:
        """Compute annual class-area fractions (proportion of valid pixels)."""
        rows = []
        for cm, yr in zip(maps, years):
            valid = cm > 0
            total = valid.sum()
            if total == 0:
                rows.append({c: 0.0 for c in CLASS_NAMES})
                continue
            row: dict[str, float] = {}
            for k, c in enumerate(CLASS_NAMES, start=1):
                row[c] = float((cm[valid] == k).sum()) / total
            rows.append(row)

        return pd.DataFrame(rows, index=years, dtype="float32")

    @staticmethod
    def _intensity_analysis(
        transitions: list[TransitionMatrix], pixel_area_ha: float
    ) -> pd.DataFrame:
        """Aldwaik & Pontius (2012) annual-level intensity statistics.

        Returns DataFrame with columns: from_year, to_year, total_change_ha,
        change_rate_pct, gaining_intensity, losing_intensity.
        """
        records = []
        for tm in transitions:
            m = tm.matrix.astype("float64")
            total_pixels = m.sum()
            diag_pixels = np.diag(m).sum()
            changed_pixels = total_pixels - diag_pixels
            n_years = tm.to_year - tm.from_year or 1

            col_totals = m.sum(axis=0)
            row_totals = m.sum(axis=1)

            # Annual intensity (% of study area changed per year)
            annual_intensity = (
                100.0 * changed_pixels / (total_pixels * n_years)
                if total_pixels > 0 else 0.0
            )
            records.append(
                {
                    "from_year": tm.from_year,
                    "to_year": tm.to_year,
                    "total_change_ha": changed_pixels * pixel_area_ha,
                    "change_rate_pct": annual_intensity,
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def _fit_urban_logistic(
        annual_fractions: pd.DataFrame, years: list[int]
    ) -> dict:
        """Fit logistic growth curve to impervious surface fraction.

        Returns dict with keys ``years``, ``observed``, ``fitted``,
        ``params`` (L, k, x0), ``r2``.
        """
        from scipy.optimize import curve_fit
        from scipy.stats import pearsonr

        if "Impervious Surface" not in annual_fractions.columns:
            return {}

        obs = annual_fractions["Impervious Surface"].values.astype("float64")
        t = np.array(years, dtype="float64")

        def logistic(t: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
            return L / (1.0 + np.exp(-k * (t - x0)))

        try:
            p0 = [float(obs.max()) * 1.1, 0.05, float(np.median(t))]  # type: ignore[arg-type]
            popt, _ = curve_fit(
                logistic, t, obs, p0=p0, maxfev=10_000, bounds=([0, 0, 1900], [1, 1, 2100])
            )
            fitted = logistic(t, *popt)
            r2 = float(float(pearsonr(obs, fitted)[0]) ** 2)  # type: ignore[operator]
            return {
                "years": years,
                "observed": obs.tolist(),
                "fitted": fitted.tolist(),
                "params": {"L": popt[0], "k": popt[1], "x0": popt[2]},
                "r2": r2,
            }
        except Exception as exc:
            logger.debug("Logistic fit failed: %s", exc)
            return {"years": years, "observed": obs.tolist()}

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def save_transitions(self, result: ChangeResult, output_dir: Path) -> None:
        """Write each transition matrix to ``output_dir/transitions/`` as CSV."""
        tdir = output_dir / "transitions"
        tdir.mkdir(parents=True, exist_ok=True)
        for tm in result.transitions:
            fname = f"{tm.from_year}_{tm.to_year}_transition.csv"
            tm.to_dataframe().to_csv(tdir / fname)

        logger.info("Saved %d transition matrices → %s", len(result.transitions), tdir)

    def save_summary(self, result: ChangeResult, output_dir: Path) -> None:
        """Write annual_fractions.csv, intensity.csv, urban_logistic.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        result.annual_fractions.to_csv(output_dir / "annual_class_fractions.csv")
        result.intensity_df.to_csv(output_dir / "annual_intensity.csv", index=False)

        import json
        with open(output_dir / "urban_logistic.json", "w") as f:
            json.dump(result.urban_logistic, f, indent=2)

        logger.info("Change detection summary saved → %s", output_dir)
