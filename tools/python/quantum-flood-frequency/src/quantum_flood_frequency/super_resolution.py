"""
super_resolution.py
===================
Multi-sensor super-resolution engine for flood frequency mapping.

Instead of **downsampling** all sensors to the coarsest resolution
(30 m Landsat), this module **upsamples** coarser imagery to the
target analysis resolution (10 m — Sentinel-2 native) using a
hierarchy of increasingly sophisticated techniques:

1. **Bicubic baseline** — ``scipy.ndimage.zoom`` with order=3.
   Fast, deterministic, always available.

2. **Spectral-guided super-resolution (SGSR)** — Uses higher-
   resolution imagery from another sensor as a *guide* for
   sharpening.  Analogous to pan-sharpening but cross-sensor:
   NAIP (1 m) guides Sentinel-2 detail, Sentinel-2 (10 m)
   guides Landsat upsampling.  Implemented via Laplacian pyramid
   injection.

3. **Learned single-image super-resolution (SISR)** — ONNX Runtime
   inference with a lightweight CNN (ESPCN / sub-pixel convolution
   network).  The model is pre-trained on remote-sensing super-
   resolution datasets.  Falls back to SGSR/bicubic if ONNX
   Runtime is unavailable or the model file is missing.

Resolution hierarchy (sample UP):
    Landsat  30 m  →  10 m  (3× upscale)
    Sentinel-2  20 m  →  10 m  (2× upscale for SWIR; 10 m bands native)
    NAIP  1 m  →  10 m  (aggregate / downsample — preserves detail)

References
----------
* Shi et al., "Real-Time Single Image and Video Super-Resolution Using
  an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016.
* Lanaras et al., "Super-resolution of Sentinel-2 images", Remote
  Sensing of Environment 239, 2020.
* Galar et al., "Super-Resolution for Vital Remote Sensing Tasks",
  IEEE GRSL 19, 2022.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import zoom, gaussian_filter

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.super_resolution")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target analysis resolution (metres)
TARGET_RESOLUTION_M = 10

# Super-resolution upscale factors per sensor
UPSCALE_FACTORS = {
    "landsat": 3,      # 30 m → 10 m
    "sentinel2": 2,    # 20 m → 10 m (SWIR bands; 10 m bands are native)
    "naip": 1,         # 1 m → 10 m (downsample, not SR)
}

# Laplacian pyramid parameters
PYRAMID_LEVELS = 3
GAUSSIAN_SIGMA = 1.0


class SRMethod(Enum):
    """Super-resolution method selection."""
    BICUBIC = auto()
    SPECTRAL_GUIDED = auto()
    LEARNED_SISR = auto()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SRResult:
    """Super-resolved imagery band.

    Attributes:
        data: Upscaled 2-D float32 array at target resolution.
        source_resolution: Original resolution in metres.
        target_resolution: Output resolution in metres.
        upscale_factor: Effective upscale ratio.
        method: SR method used.
        quality_score: Optional quality metric (SSIM-like, 0–1).
    """
    data: np.ndarray
    source_resolution: int
    target_resolution: int
    upscale_factor: int
    method: SRMethod
    quality_score: float = 0.0


# ---------------------------------------------------------------------------
# Laplacian pyramid utilities
# ---------------------------------------------------------------------------

def _build_gaussian_pyramid(
    img: np.ndarray, levels: int, sigma: float = GAUSSIAN_SIGMA
) -> list[np.ndarray]:
    """Build a Gaussian pyramid by successive blur + downsample.

    Args:
        img: Input 2-D array.
        levels: Number of pyramid levels (including original).
        sigma: Gaussian blur sigma before each downsample.

    Returns:
        List of arrays from finest (original) to coarsest.
    """
    pyramid = [img.astype("float32")]
    current = img.astype("float32")
    for _ in range(levels - 1):
        blurred = gaussian_filter(current, sigma=sigma)
        # Downsample by 2×
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    return pyramid


def _build_laplacian_pyramid(
    img: np.ndarray, levels: int, sigma: float = GAUSSIAN_SIGMA
) -> list[np.ndarray]:
    """Build a Laplacian (detail) pyramid.

    Each level captures the high-frequency detail lost when moving
    to the next coarser Gaussian level.

    Args:
        img: Input 2-D array.
        levels: Number of pyramid levels.
        sigma: Gaussian blur sigma.

    Returns:
        List of Laplacian (detail) arrays + residual (coarsest Gaussian).
    """
    gauss = _build_gaussian_pyramid(img, levels, sigma)
    laplacian: list[np.ndarray] = []

    for i in range(len(gauss) - 1):
        # Upsample coarser level to match finer level
        upsampled = _upsample_bicubic(gauss[i + 1], gauss[i].shape)
        detail = gauss[i] - upsampled
        laplacian.append(detail)

    # Residual (lowest frequency)
    laplacian.append(gauss[-1])
    return laplacian


def _upsample_bicubic(
    arr: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Upsample a 2-D array to target_shape using bicubic interpolation.

    Args:
        arr: Input 2-D float array.
        target_shape: Desired (rows, cols) output shape.

    Returns:
        Bicubic-upsampled array.
    """
    if arr.shape == target_shape:
        return arr.copy()

    zoom_factors = (
        target_shape[0] / arr.shape[0],
        target_shape[1] / arr.shape[1],
    )
    return np.asarray(
        zoom(arr.astype("float32"), zoom_factors, order=3, mode="reflect"),
        dtype="float32",
    )


def _downsample_area(
    arr: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Downsample via area-weighted averaging (anti-aliased).

    This is preferred over simple decimation for large downscale
    ratios (e.g. NAIP 1 m → 10 m) as it preserves the spatial mean
    and avoids aliasing artefacts.

    Args:
        arr: Input 2-D float array.
        target_shape: Desired (rows, cols) output shape.

    Returns:
        Area-averaged downsampled array.
    """
    if arr.shape == target_shape:
        return arr.copy()

    src_h, src_w = arr.shape
    tgt_h, tgt_w = target_shape

    # Pre-blur to avoid aliasing
    sigma = max(src_h / tgt_h, src_w / tgt_w) / 2.0
    blurred = gaussian_filter(arr.astype("float32"), sigma=sigma)

    zoom_factors = (tgt_h / src_h, tgt_w / src_w)
    return np.asarray(
        zoom(blurred, zoom_factors, order=1, mode="nearest"),
        dtype="float32",
    )


# ---------------------------------------------------------------------------
# Super-resolution engine
# ---------------------------------------------------------------------------

class SuperResolutionEngine:
    """Multi-method super-resolution engine for satellite imagery.

    Upscales coarser-resolution bands (Landsat 30 m, Sentinel-2 20 m)
    to the target analysis resolution (10 m) using a cascade of
    techniques:

    1. **Bicubic** — default baseline, always available.
    2. **Spectral-guided** — uses a high-res guide image from another
       sensor to inject spatial detail via Laplacian pyramid fusion.
    3. **Learned SISR** — ONNX Runtime CNN inference (optional).

    Parameters
    ----------
    target_resolution:
        Target pixel GSD in metres.
    method:
        Preferred SR method.  Falls back to simpler methods if the
        preferred one is unavailable.
    onnx_model_path:
        Path to an ONNX super-resolution model (for LEARNED_SISR).
    """

    def __init__(
        self,
        target_resolution: int = TARGET_RESOLUTION_M,
        method: SRMethod = SRMethod.SPECTRAL_GUIDED,
        onnx_model_path: Optional[Path] = None,
    ) -> None:
        self.target_resolution = target_resolution
        self.preferred_method = method
        self.onnx_model_path = onnx_model_path

        # Try to load ONNX Runtime
        self._ort_session: object = None  # type: ignore[assignment]
        if method == SRMethod.LEARNED_SISR and onnx_model_path:
            self._ort_session = self._load_onnx_model(onnx_model_path)

        logger.info(
            "SR engine initialised — target %d m, method %s",
            target_resolution, method.name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upscale_band(
        self,
        band: np.ndarray,
        source_resolution: int,
        target_shape: tuple[int, int],
        guide_band: Optional[np.ndarray] = None,
    ) -> SRResult:
        """Super-resolve a single spectral band.

        Args:
            band: Input 2-D float32 array at source resolution.
            source_resolution: Source pixel GSD in metres.
            target_shape: Desired output (rows, cols) at target resolution.
            guide_band: Optional higher-resolution guide image for SGSR.

        Returns:
            SRResult with upscaled band and metadata.
        """
        upscale_factor = source_resolution // self.target_resolution

        if upscale_factor <= 1:
            # Already at or finer than target — just regrid
            result_data = _upsample_bicubic(band, target_shape)
            return SRResult(
                data=result_data,
                source_resolution=source_resolution,
                target_resolution=self.target_resolution,
                upscale_factor=1,
                method=SRMethod.BICUBIC,
                quality_score=1.0,
            )

        # Try methods in preference order
        method_used = SRMethod.BICUBIC
        result_data: np.ndarray

        if (
            self.preferred_method == SRMethod.LEARNED_SISR
            and self._ort_session is not None
        ):
            result_data = self._learned_sr(band, target_shape, upscale_factor)
            method_used = SRMethod.LEARNED_SISR

        elif (
            self.preferred_method in (SRMethod.SPECTRAL_GUIDED, SRMethod.LEARNED_SISR)
            and guide_band is not None
        ):
            result_data = self._spectral_guided_sr(
                band, guide_band, target_shape, upscale_factor
            )
            method_used = SRMethod.SPECTRAL_GUIDED

        else:
            result_data = self._bicubic_sr(band, target_shape)
            method_used = SRMethod.BICUBIC

        # Compute quality score (structural similarity proxy)
        quality = self._estimate_quality(band, result_data, target_shape)

        return SRResult(
            data=result_data,
            source_resolution=source_resolution,
            target_resolution=self.target_resolution,
            upscale_factor=upscale_factor,
            method=method_used,
            quality_score=quality,
        )

    def downsample_band(
        self,
        band: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Downsample a fine-resolution band (NAIP) using area averaging.

        Args:
            band: Input 2-D float array at fine resolution.
            target_shape: Target (rows, cols) at analysis resolution.

        Returns:
            Downsampled float32 array.
        """
        return _downsample_area(band, target_shape)

    def upscale_observation(
        self,
        obs: dict,
        source_resolution: int,
        target_shape: tuple[int, int],
        guide_obs: Optional[dict] = None,
    ) -> dict:
        """Super-resolve all bands in an observation dict.

        Args:
            obs: Observation dict with band arrays + metadata.
            source_resolution: Source pixel GSD.
            target_shape: Target output shape.
            guide_obs: Optional higher-res observation for guided SR.

        Returns:
            New observation dict with all bands at target resolution.
        """
        band_keys = ["blue", "green", "red", "nir", "swir1", "swir2"]
        result_obs = {}

        for key in band_keys:
            if key not in obs:
                continue

            band = obs[key]
            if np.all(np.isnan(band)):
                # NaN placeholder (e.g. NAIP SWIR) → just resize the NaN array
                result_obs[key] = np.full(target_shape, np.nan, dtype="float32")
                continue

            guide = None
            if guide_obs is not None and key in guide_obs:
                guide = guide_obs[key]

            sr_result = self.upscale_band(
                band, source_resolution, target_shape, guide_band=guide
            )
            result_obs[key] = sr_result.data

        # Handle cloud mask (nearest-neighbour, no interpolation)
        if "cloud_mask" in obs:
            mask = obs["cloud_mask"]
            if mask.shape != target_shape:
                zoom_factors = (
                    target_shape[0] / mask.shape[0],
                    target_shape[1] / mask.shape[1],
                )
                result_obs["cloud_mask"] = (
                    np.asarray(
                        zoom(mask.astype("float32"), zoom_factors, order=0, mode="nearest"),
                        dtype="float32",
                    ) > 0.5
                )
            else:
                result_obs["cloud_mask"] = mask.copy()

        # Copy metadata
        result_obs["source"] = obs.get("source", "unknown")
        result_obs["date"] = obs.get("date", "")

        return result_obs

    # ------------------------------------------------------------------
    # SR methods
    # ------------------------------------------------------------------

    @staticmethod
    def _bicubic_sr(
        band: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:
        """Simple bicubic upsampling (baseline).

        Args:
            band: Input 2-D array.
            target_shape: Output shape.

        Returns:
            Bicubic-interpolated array.
        """
        result = _upsample_bicubic(band, target_shape)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _spectral_guided_sr(
        band: np.ndarray,
        guide: np.ndarray,
        target_shape: tuple[int, int],
        upscale_factor: int,
    ) -> np.ndarray:
        """Spectral-guided super-resolution via Laplacian pyramid injection.

        The idea: the *guide* image (higher resolution) provides spatial
        detail (edges, texture) that the coarse band lacks.  We:

        1. Build a Laplacian pyramid of the guide at target resolution.
        2. Bicubic-upsample the coarse band to target resolution.
        3. Replace the coarsest level of the guide pyramid with the
           upsampled band's low-frequency content.
        4. Reconstruct → the result has the guide's spatial detail
           modulated by the coarse band's spectral content.

        This is analogous to pan-sharpening (Brovey, IHS, wavelet)
        but operates cross-sensor and cross-band.

        Args:
            band: Coarse-resolution input band.
            guide: High-resolution guide band (already at target res).
            target_shape: Desired output shape.
            upscale_factor: Upscale ratio.

        Returns:
            Spectral-guided super-resolved array.
        """
        n_levels = min(PYRAMID_LEVELS, upscale_factor + 1)

        # Ensure guide is at target shape
        if guide.shape != target_shape:
            guide = _upsample_bicubic(guide, target_shape)

        # Build Laplacian pyramid of the guide
        guide_lap = _build_laplacian_pyramid(guide, n_levels)

        # Bicubic-upsample the coarse band
        upsampled = _upsample_bicubic(band, target_shape)

        # Build Gaussian pyramid of the upsampled band
        up_gauss = _build_gaussian_pyramid(upsampled, n_levels)

        # Inject: replace the residual (coarsest) with the upsampled
        # band's coarsest frequency, keep guide's detail layers.
        # Weight the detail injection by spectral correlation
        guide_gauss = _build_gaussian_pyramid(guide, n_levels)
        correlation = _local_correlation(
            up_gauss[-1], guide_gauss[-1]
        )

        # Reconstruct from Laplacian pyramid with modified residual
        reconstructed = up_gauss[-1]  # use source's low frequency

        for i in range(len(guide_lap) - 2, -1, -1):
            # Upsample to match this level's shape
            reconstructed = _upsample_bicubic(
                reconstructed, guide_lap[i].shape
            )
            # Inject guide detail, weighted by correlation
            detail_weight = np.clip(correlation, 0.0, 1.0)
            if isinstance(detail_weight, np.ndarray) and detail_weight.shape != guide_lap[i].shape:
                detail_weight = _upsample_bicubic(
                    detail_weight.reshape(guide_gauss[-1].shape),
                    guide_lap[i].shape,
                )
            reconstructed = reconstructed + detail_weight * guide_lap[i]

        return np.clip(reconstructed, 0.0, 1.0).astype("float32")

    def _learned_sr(
        self,
        band: np.ndarray,
        target_shape: tuple[int, int],
        upscale_factor: int,
    ) -> np.ndarray:
        """Learned SISR via ONNX Runtime.

        Loads a pre-trained ESPCN-style model and runs inference.
        Falls back to spectral-guided or bicubic if inference fails.

        Args:
            band: Input coarse-resolution band.
            target_shape: Output (rows, cols).
            upscale_factor: Upscale ratio.

        Returns:
            Super-resolved array.
        """
        if self._ort_session is None:
            logger.warning("ONNX session unavailable — falling back to bicubic")
            return self._bicubic_sr(band, target_shape)

        try:
            import onnxruntime as ort  # type: ignore[import-unresolved]

            # Prepare input: NCHW format, float32
            input_tensor = band.astype("float32")[np.newaxis, np.newaxis, :, :]
            input_name = self._ort_session.get_inputs()[0].name  # type: ignore[union-attr]
            output_name = self._ort_session.get_outputs()[0].name  # type: ignore[union-attr]

            result = self._ort_session.run(  # type: ignore[union-attr]
                [output_name],
                {input_name: input_tensor},
            )[0]

            # Squeeze batch/channel dims
            result = result.squeeze()

            # Resize to exact target shape if needed
            if result.shape != target_shape:
                result = _upsample_bicubic(result, target_shape)

            return np.clip(result, 0.0, 1.0).astype("float32")

        except Exception as exc:
            logger.warning("ONNX inference failed: %s — falling back to bicubic", exc)
            return self._bicubic_sr(band, target_shape)

    # ------------------------------------------------------------------
    # Quality estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_quality(
        original: np.ndarray,
        upscaled: np.ndarray,
        target_shape: tuple[int, int],
    ) -> float:
        """Estimate SR quality via downscale–compare consistency check.

        Downscales the SR result back to original resolution and measures
        RMSE against the original.  Lower RMSE → higher quality score.

        Returns:
            Quality score in [0, 1] — 1.0 is perfect consistency.
        """
        # Downscale SR result back to original resolution
        original_shape = original.shape
        downscaled = _downsample_area(upscaled, original_shape)

        # RMSE between original and round-tripped
        with np.errstate(invalid="ignore"):
            valid = ~np.isnan(original) & ~np.isnan(downscaled)
            if valid.sum() == 0:
                return 0.0

            rmse = np.sqrt(np.mean((original[valid] - downscaled[valid]) ** 2))

        # Map RMSE to [0, 1] quality score (sigmoid decay)
        quality = 1.0 / (1.0 + 10.0 * rmse)
        return float(np.clip(quality, 0.0, 1.0))

    # ------------------------------------------------------------------
    # ONNX model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_onnx_model(model_path: Path) -> object | None:
        """Load an ONNX Runtime inference session.

        Args:
            model_path: Path to .onnx model file.

        Returns:
            ONNX InferenceSession or None if unavailable.
        """
        try:
            import onnxruntime as ort  # type: ignore[import-unresolved]

            if not model_path.exists():
                logger.warning("ONNX model not found: %s", model_path)
                return None

            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNX SR model loaded: %s", model_path)
            return session

        except ImportError:
            logger.info("onnxruntime not installed — learned SR unavailable")
            return None
        except Exception as exc:
            logger.warning("Failed to load ONNX model: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Helper: local correlation between two images
# ---------------------------------------------------------------------------

def _local_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    window: int = 5,
) -> float | np.ndarray:
    """Compute Pearson correlation between two images.

    Used to weight detail injection in spectral-guided SR — high
    correlation means the guide's spatial details are spectrally
    consistent with the source band.

    Args:
        img1: First image array.
        img2: Second image array.
        window: Not used for global correlation (reserved for future
            block-wise implementation).

    Returns:
        Scalar correlation coefficient in [-1, 1].
    """
    if img1.shape != img2.shape:
        img2 = _upsample_bicubic(img2, img1.shape)

    flat1 = img1.ravel().astype("float64")
    flat2 = img2.ravel().astype("float64")

    valid = np.isfinite(flat1) & np.isfinite(flat2)
    if valid.sum() < 10:
        return 0.5

    corr = np.corrcoef(flat1[valid], flat2[valid])[0, 1]
    return float(np.clip(corr, 0.0, 1.0)) if np.isfinite(corr) else 0.5
