"""sam_segmenter.py — Object-based segmentation using Meta SAM2.

Applies SAM2 (Segment Anything Model 2) to NAIP RGBNIR tiles to produce
semantically meaningful image segments.  Each segment becomes an "object"
for the OBIA (Object-Based Image Analysis) classification branch.

SAM2 produces class-agnostic mask proposals; the classification label is
assigned from the ensemble classifier's prediction aggregated over the
segment's pixels (voted or mean-probability).

When SAM2 is not installed, falls back to a SLIC superpixel segmentation
(scikit-image) at reduced quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import xarray as xr

from .constants import AUSTIN_UTM_CRS, CNN_CHIP_SIZE, CNN_OVERLAP_PX

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.sam_segmenter")

SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-not-found]
    SAM2_AVAILABLE = True
except ImportError:
    logger.debug("SAM2 not installed — will fall back to SLIC superpixel segmentation.")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SegmentFeatures:
    """Per-segment features and metadata extracted from NAIP imagery.

    Attributes
    ----------
    segment_array:  Int32 (H, W) label array; 0 = background.
    n_segments:     Number of segments found.
    gdf:            GeoDataFrame with columns: segment_id, centroid,
                    area_m2, perimeter_m, compactness + mean band values.
    crs:            CRS of ``gdf``.
    shape:          (H, W).
    method_used:    ``"sam2"`` or ``"slic_fallback"``.
    """

    segment_array: np.ndarray
    n_segments: int
    gdf: gpd.GeoDataFrame
    crs: str
    shape: tuple[int, int]
    method_used: str


# ── Main segmenter ────────────────────────────────────────────────────────────

class SAMSegmenter:
    """Tile NAIP imagery and run SAM2 (or SLIC fallback) to produce segments.

    Parameters
    ----------
    checkpoint_path: Path to SAM2 model checkpoint (e.g. ``sam2_hiera_large.pt``).
    model_cfg:       SAM2 model config name (e.g. ``"sam2_hiera_large.yaml"``).
    tile_size:       Tile dimension in pixels.
    overlap:         Overlap between adjacent tiles in pixels.
    device:          PyTorch device string (``"cpu"`` or ``"cuda"``).
    slic_n_segments: Number of SLIC segments (used only in fallback mode).
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        model_cfg: str = "sam2_hiera_large.yaml",
        tile_size: int = CNN_CHIP_SIZE,
        overlap: int = CNN_OVERLAP_PX,
        device: str = "cpu",
        slic_n_segments: int = 5000,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.slic_n_segments = slic_n_segments
        self._predictor: Optional[object] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def segment(self, naip_ds: xr.Dataset) -> SegmentFeatures:
        """Segment the NAIP mosaic into objects.

        Parameters
        ----------
        naip_ds:  NAIP xr.Dataset with ``red``, ``green``, ``blue`` (and
                  optionally ``nir``) bands at 1 m resolution.

        Returns
        -------
        SegmentFeatures
            Per-segment label array and attribute GeoDataFrame.
        """
        rgb = self._extract_rgb(naip_ds)
        H, W = rgb.shape[:2]

        if SAM2_AVAILABLE and self.checkpoint_path is not None and self.checkpoint_path.exists():
            logger.info("Running SAM2 segmentation (%d × %d px) …", H, W)
            labels, method = self._run_sam2(rgb)
        else:
            logger.info("Running SLIC fallback segmentation (%d × %d px) …", H, W)
            labels, method = self._run_slic(rgb)

        n_seg = int(labels.max())
        gdf = self._build_gdf(labels, naip_ds, H, W)

        return SegmentFeatures(
            segment_array=labels,
            n_segments=n_seg,
            gdf=gdf,
            crs=AUSTIN_UTM_CRS,
            shape=(H, W),
            method_used=method,
        )

    # ── SAM2 ──────────────────────────────────────────────────────────────────

    def _run_sam2(self, rgb: np.ndarray) -> tuple[np.ndarray, str]:
        """Run SAM2 automatic mask generation on tiled RGB image.

        Tiles the large mosaic into ``tile_size × tile_size`` chips,
        runs SAM2 ``generate()`` on each, stitches labels back.
        """
        H, W = rgb.shape[:2]
        labels = np.zeros((H, W), dtype="int32")
        label_offset = 0
        step = self.tile_size - self.overlap

        if self._predictor is None:
            try:
                sam2_model = build_sam2(
                    self.model_cfg, str(self.checkpoint_path), device=self.device
                )
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore[import-not-found]
                self._predictor = SAM2AutomaticMaskGenerator(
                    sam2_model,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    min_mask_region_area=100,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SAM2 init failed: %s — falling back to SLIC.", exc)
                return self._run_slic(rgb)

        for y in range(0, H, step):
            for x in range(0, W, step):
                tile = rgb[y: y + self.tile_size, x: x + self.tile_size]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                try:
                    masks = self._predictor.generate(tile)  # type: ignore[union-attr]
                    for m in masks:
                        seg_mask = m["segmentation"]
                        label_offset += 1
                        labels[
                            y: y + seg_mask.shape[0],
                            x: x + seg_mask.shape[1],
                        ] = np.where(seg_mask, label_offset, labels[
                            y: y + seg_mask.shape[0],
                            x: x + seg_mask.shape[1],
                        ])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("SAM2 tile failed at (%d,%d): %s", y, x, exc)

        return labels, "sam2"

    # ── SLIC fallback ─────────────────────────────────────────────────────────

    def _run_slic(self, rgb: np.ndarray) -> tuple[np.ndarray, str]:
        """SLIC superpixel segmentation as SAM2 fallback."""
        from skimage.segmentation import slic
        from skimage.util import img_as_float

        img_float = img_as_float(rgb.clip(0, 255).astype("uint8"))
        segments = slic(
            img_float,
            n_segments=self.slic_n_segments,
            compactness=10,
            sigma=1,
            start_label=1,
        )
        return segments.astype("int32"), "slic_fallback"

    # ── GeoDataFrame builder ──────────────────────────────────────────────────

    def _build_gdf(
        self,
        labels: np.ndarray,
        naip_ds: xr.Dataset,
        H: int,
        W: int,
    ) -> gpd.GeoDataFrame:
        """Compute per-segment attributes and vectorise to GeoDataFrame."""
        from rasterio.features import shapes as rio_shapes
        from rasterio.transform import from_bounds
        from shapely.geometry import shape

        try:
            transform = naip_ds.rio.transform()
            crs_str = str(naip_ds.rio.crs)
        except Exception:  # noqa: BLE001
            minx, miny, maxx, maxy = 550000, 3340000, 670000, 3420000
            transform = from_bounds(minx, miny, maxx, maxy, W, H)
            crs_str = AUSTIN_UTM_CRS

        geoms_vals = list(
            rio_shapes(labels.astype("int32"), mask=(labels > 0).astype("uint8"), transform=transform)
        )

        records = []
        for geom_dict, val in geoms_vals:
            if int(val) == 0:
                continue
            poly = shape(geom_dict)
            area = poly.area
            perim = poly.length
            compactness = (4 * np.pi * area) / (perim**2 + 1e-10)
            records.append({
                "geometry": poly,
                "segment_id": int(val),
                "area_m2": area,
                "perimeter_m": perim,
                "compactness": compactness,
            })

        if not records:
            return gpd.GeoDataFrame(
                columns=["geometry", "segment_id", "area_m2", "perimeter_m", "compactness"],
                crs=crs_str,
            )

        gdf = gpd.GeoDataFrame(records, crs=crs_str)

        # Attach mean band values
        for band_name in ("red", "green", "blue", "nir"):
            if band_name in naip_ds:
                arr = naip_ds[band_name].values.squeeze().astype("float32")
                from scipy.ndimage import labeled_comprehension
                means = labeled_comprehension(
                    arr, labels, index=gdf["segment_id"].values, func=np.nanmean,
                    out_dtype="float32", default=np.nan,
                )
                gdf[f"mean_{band_name}"] = means

        return gdf

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_rgb(self, naip_ds: xr.Dataset) -> np.ndarray:
        """Stack NAIP RGB bands into uint8 (H, W, 3) array."""
        channels = []
        for b in ("red", "green", "blue"):
            if b in naip_ds:
                arr = naip_ds[b].values.squeeze()
                # If float [0,1] rescale to uint8
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype("uint8")
                channels.append(arr.astype("uint8"))

        if len(channels) == 3:
            return np.stack(channels, axis=-1)
        H_fallback, W_fallback = 1024, 1024
        return np.zeros((H_fallback, W_fallback, 3), dtype="uint8")
