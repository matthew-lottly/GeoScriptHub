"""
sar_processor.py
================
Sentinel-1 C-band SAR (Synthetic Aperture Radar) processing for
flood detection.

SAR is the single most discriminative data source for separating
water from buildings:

* **Water** — specular reflection → very low backscatter (< −15 dB VV)
* **Buildings** — corner reflectors → very high backscatter (> −5 dB VV)
* **Vegetation** — volume scattering → moderate backscatter

This module processes Sentinel-1 GRD (Ground Range Detected) imagery
in IW (Interferometric Wide) mode from Microsoft Planetary Computer:

1. Convert DN to calibrated σ⁰ backscatter (dB)
2. Apply Lee speckle filter (adaptive, edge-preserving)
3. Compute SAR Water Index (SWI) and VH/VV ratio
4. Create temporal composite (multi-temporal mean for speckle reduction)

References
----------
* Lee, J.-S., "Speckle Analysis and Smoothing of Synthetic Aperture
  Radar Images", CGIP 17(1), 1981.
* Twele et al., "Sentinel-1-based flood mapping", Remote Sensing of
  Environment 206, 2018.
* Bioresita et al., "A Method for Automatic and Rapid Mapping of Water
  Surfaces from Sentinel-1 Imagery", Remote Sensing 10(2), 2018.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter, zoom

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.sar_processor")

# ---------------------------------------------------------------------------
# SAR constants
# ---------------------------------------------------------------------------

# Sentinel-1 GRD backscatter thresholds (σ⁰ in dB)
SAR_WATER_VV_THRESHOLD = -15.0    # VV < −15 dB → likely water
SAR_BUILDING_VV_THRESHOLD = -5.0  # VV > −5 dB → likely building
SAR_NOISE_FLOOR_DB = -30.0        # Below noise floor → invalid

# Lee filter parameters
LEE_WINDOW_SIZE = 7               # 7×7 adaptive window
LEE_NLOOKS = 4.4                  # Effective number of looks for S-1 GRD IW

# Temporal composite minimum observations
MIN_SAR_OBSERVATIONS = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SARFeatures:
    """Processed SAR feature layers for flood classification.

    All arrays are 2-D float32 at the target analysis resolution.

    Attributes:
        vv_db: Temporal mean VV backscatter (dB).
        vh_db: Temporal mean VH backscatter (dB).
        vh_vv_ratio: VH/VV ratio (linear scale) — high for vegetation.
        sar_water_index: SWI = (1 − VV_linear) — high for water.
        water_mask_sar: Boolean — SAR-only water detection (VV < threshold).
        building_mask_sar: Boolean — SAR-only building detection (VV > threshold).
        n_observations: Number of SAR scenes used in composite.
        valid_mask: Boolean — True where enough observations exist.
    """

    vv_db: np.ndarray
    vh_db: np.ndarray
    vh_vv_ratio: np.ndarray
    sar_water_index: np.ndarray
    water_mask_sar: np.ndarray
    building_mask_sar: np.ndarray
    n_observations: int = 0
    valid_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    def __repr__(self) -> str:
        return (
            f"<SARFeatures  {self.vv_db.shape[1]}×{self.vv_db.shape[0]} px  "
            f"n_obs={self.n_observations}  "
            f"water_px={self.water_mask_sar.sum()}  "
            f"building_px={self.building_mask_sar.sum()}>"
        )


# ---------------------------------------------------------------------------
# Lee speckle filter
# ---------------------------------------------------------------------------

def lee_filter(
    img: np.ndarray,
    window_size: int = LEE_WINDOW_SIZE,
    n_looks: float = LEE_NLOOKS,
) -> np.ndarray:
    """Apply the Lee adaptive speckle filter.

    The Lee filter preserves edges while reducing multiplicative
    speckle noise.  It adapts the filtering strength based on
    local coefficient of variation (CV):

    * Low CV (homogeneous region) → heavy smoothing
    * High CV (edge/feature) → minimal smoothing

    Formula:
        Î = Ī + W × (I − Ī)
        W = 1 − (Cv_noise² / Cv_local²)

    where Cv_noise = 1/√n_looks and Cv_local = σ_local/μ_local.

    Args:
        img: Input SAR intensity image (linear scale, NOT dB).
        window_size: Filter window size (must be odd).
        n_looks: Effective number of looks.

    Returns:
        Filtered image (same shape, linear scale).
    """
    if window_size % 2 == 0:
        window_size += 1

    img = img.astype("float64")
    img = np.maximum(img, 1e-10)  # avoid division by zero

    # Local statistics
    local_mean = uniform_filter(img, size=window_size, mode="reflect")
    local_sq_mean = uniform_filter(img ** 2, size=window_size, mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    # Coefficient of variation
    cv_noise = 1.0 / np.sqrt(n_looks)
    cv_noise_sq = cv_noise ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        cv_local_sq = local_var / np.maximum(local_mean ** 2, 1e-10)

    # Adaptive weight
    weight = np.where(
        cv_local_sq > cv_noise_sq,
        1.0 - cv_noise_sq / np.maximum(cv_local_sq, cv_noise_sq + 1e-10),
        0.0,
    )
    weight = np.clip(weight, 0.0, 1.0)

    # Filtered image
    filtered = local_mean + weight * (img - local_mean)
    return np.maximum(filtered, 1e-10).astype("float32")


# ---------------------------------------------------------------------------
# SAR utility functions
# ---------------------------------------------------------------------------

def dn_to_sigma0_db(dn: np.ndarray) -> np.ndarray:
    """Convert Sentinel-1 GRD digital numbers to σ⁰ backscatter in dB.

    The Planetary Computer provides radiometrically terrain-corrected
    (RTC) Sentinel-1 GRD data with γ⁰ in linear power scale.

    σ⁰ (dB) = 10 × log₁₀(DN)

    Args:
        dn: Digital numbers (linear power scale, float).

    Returns:
        Backscatter in dB (float32).
    """
    dn = np.maximum(np.abs(dn).astype("float64"), 1e-10)
    db = 10.0 * np.log10(dn)
    return np.clip(db, SAR_NOISE_FLOOR_DB, 10.0).astype("float32")


def compute_sar_water_index(vv_linear: np.ndarray) -> np.ndarray:
    """SAR Water Index: SWI = 1 − clamp(VV_linear, 0, 0.1) / 0.1.

    Water has very low VV backscatter (near 0 in linear), so SWI → 1.
    Land/buildings have higher backscatter, so SWI → 0.

    Args:
        vv_linear: VV backscatter in linear power scale.

    Returns:
        SWI in [0, 1] — high = water.
    """
    vv_clamped = np.clip(vv_linear, 0.0, 0.1)
    return (1.0 - vv_clamped / 0.1).astype("float32")


def compute_vh_vv_ratio(
    vh_linear: np.ndarray,
    vv_linear: np.ndarray,
) -> np.ndarray:
    """Compute VH/VV cross-polarisation ratio (linear scale).

    * Low ratio (~0.05–0.1) → smooth surface (water, bare soil)
    * High ratio (~0.2–0.5) → volume scattering (vegetation)
    * Moderate ratio → built-up areas

    Args:
        vh_linear: VH backscatter (linear).
        vv_linear: VV backscatter (linear).

    Returns:
        VH/VV ratio (float32).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            vv_linear > 1e-10,
            vh_linear / vv_linear,
            0.0,
        )
    return np.clip(ratio, 0.0, 2.0).astype("float32")


# ---------------------------------------------------------------------------
# SAR Processor
# ---------------------------------------------------------------------------

class SARProcessor:
    """Process Sentinel-1 GRD SAR imagery for flood detection.

    Creates a temporal composite from multiple SAR acquisitions and
    derives water/building discrimination features.

    Parameters
    ----------
    target_shape:
        Output (rows, cols) at analysis resolution.
    window_size:
        Lee filter window size (default 7).
    """

    def __init__(
        self,
        target_shape: tuple[int, int],
        window_size: int = LEE_WINDOW_SIZE,
    ) -> None:
        self.target_shape = target_shape
        self.window_size = window_size
        self._vv_accumulator: list[np.ndarray] = []
        self._vh_accumulator: list[np.ndarray] = []

    def add_observation(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
    ) -> None:
        """Add a single SAR observation to the temporal accumulator.

        Applies speckle filtering and resamples to target shape before
        accumulating.

        Args:
            vv: VV polarisation band (linear power scale).
            vh: VH polarisation band (linear power scale).
        """
        # Resample to target shape if needed
        if vv.shape != self.target_shape:
            vv = self._resample(vv, self.target_shape)
            vh = self._resample(vh, self.target_shape)

        # Apply Lee speckle filter (in linear domain)
        vv_filtered = lee_filter(vv, self.window_size)
        vh_filtered = lee_filter(vh, self.window_size)

        self._vv_accumulator.append(vv_filtered)
        self._vh_accumulator.append(vh_filtered)

        logger.debug(
            "SAR observation %d added (mean VV=%.2f dB)",
            len(self._vv_accumulator),
            float(np.nanmean(dn_to_sigma0_db(vv_filtered))),
        )

    def compute_features(self) -> SARFeatures:
        """Compute SAR feature layers from accumulated observations.

        Creates a temporal mean composite (multi-temporal averaging
        is the most effective speckle reduction technique) and derives
        water/building discrimination indices.

        Returns:
            SARFeatures with all derived layers.

        Raises:
            ValueError: If fewer than MIN_SAR_OBSERVATIONS have been added.
        """
        n_obs = len(self._vv_accumulator)

        if n_obs == 0:
            logger.warning("No SAR observations — returning empty features")
            return self._empty_features()

        logger.info(
            "Computing SAR features from %d observations …", n_obs
        )

        # Temporal mean composite (linear domain)
        vv_stack = np.stack(self._vv_accumulator, axis=0)
        vh_stack = np.stack(self._vh_accumulator, axis=0)

        vv_mean = np.nanmean(vv_stack, axis=0).astype("float32")
        vh_mean = np.nanmean(vh_stack, axis=0).astype("float32")

        # Convert to dB
        vv_db = dn_to_sigma0_db(vv_mean)
        vh_db = dn_to_sigma0_db(vh_mean)

        # Derived indices
        vh_vv_ratio = compute_vh_vv_ratio(vh_mean, vv_mean)
        sar_water_idx = compute_sar_water_index(vv_mean)

        # Binary masks
        water_mask = vv_db < SAR_WATER_VV_THRESHOLD
        building_mask = vv_db > SAR_BUILDING_VV_THRESHOLD

        # Valid mask (enough observations)
        valid = np.ones(self.target_shape, dtype=bool)
        if n_obs < MIN_SAR_OBSERVATIONS:
            logger.warning(
                "Only %d SAR observations (min %d) — features may be noisy",
                n_obs, MIN_SAR_OBSERVATIONS,
            )

        logger.info(
            "SAR features: water=%d px, building=%d px, mean VV=%.1f dB",
            water_mask.sum(), building_mask.sum(), float(np.nanmean(vv_db)),
        )

        return SARFeatures(
            vv_db=vv_db,
            vh_db=vh_db,
            vh_vv_ratio=vh_vv_ratio,
            sar_water_index=sar_water_idx,
            water_mask_sar=water_mask,
            building_mask_sar=building_mask,
            n_observations=n_obs,
            valid_mask=valid,
        )

    def _empty_features(self) -> SARFeatures:
        """Return placeholder empty SAR features."""
        shape = self.target_shape
        return SARFeatures(
            vv_db=np.full(shape, np.nan, dtype="float32"),
            vh_db=np.full(shape, np.nan, dtype="float32"),
            vh_vv_ratio=np.full(shape, np.nan, dtype="float32"),
            sar_water_index=np.full(shape, np.nan, dtype="float32"),
            water_mask_sar=np.zeros(shape, dtype=bool),
            building_mask_sar=np.zeros(shape, dtype=bool),
            n_observations=0,
            valid_mask=np.zeros(shape, dtype=bool),
        )

    @staticmethod
    def _resample(
        arr: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Resample SAR array to target shape via bilinear interpolation."""
        if arr.shape == target_shape:
            return arr.copy()
        factors = (
            target_shape[0] / arr.shape[0],
            target_shape[1] / arr.shape[1],
        )
        return np.asarray(
            zoom(arr.astype("float32"), factors, order=1, mode="reflect"),
            dtype="float32",
        )
