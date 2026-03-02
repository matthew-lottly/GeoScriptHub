"""mosaic.py — Dask-parallel tiled mosaic engine for metro-scale AOI processing.

Splits the Austin 110 × 70 km bounding box into overlapping tiles, runs the
full classification pipeline on each tile in parallel using a Dask
LocalCluster, then feather-blends overlapping margins and assembles a single
seamless COG output per year.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .aoi import TileSpec
from .constants import AUSTIN_BBOX_WGS84, TILE_SIZE_PX, TILE_OVERLAP_PX

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.mosaic")

# ── Blend weight kernel ───────────────────────────────────────────────────────

def _cosine_blend_kernel(size: int, overlap: int) -> np.ndarray:
    """Create a 2-D cosine taper blend weight kernel (size × size)."""
    ramp = np.ones(size, dtype="float32")
    if overlap > 0:
        cos_ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, overlap))
        ramp[:overlap] = cos_ramp
        ramp[-overlap:] = cos_ramp[::-1]
    kernel_2d = np.outer(ramp, ramp)
    return kernel_2d.astype("float32")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MosaicResult:
    """Blended metro-scale output.

    Attributes
    ----------
    class_map:    Int8 (H_total, W_total) mosaic. 0 = nodata.
    class_probs:  Float32 (H_total, W_total, NUM_CLASSES) weighted average.
    confidence:   Float32 (H_total, W_total).
    n_tiles:      Total tiles processed.
    failed_tiles: List of TileSpec that raised exceptions.
    year:         Calendar year.
    """

    class_map: np.ndarray
    class_probs: np.ndarray
    confidence: np.ndarray
    n_tiles: int
    failed_tiles: list[TileSpec]
    year: int


# ── Mosaic engine ─────────────────────────────────────────────────────────────

class TiledMosaicEngine:
    """Coordinate tile decomposition, parallel scheduling, and blending.

    Parameters
    ----------
    tile_size_px:   Tile side length in pixels (default from constants).
    tile_overlap_px: Overlap between adjacent tiles in pixels.
    n_workers:      Number of Dask workers. -1 → one per CPU core.
    threads_per_worker: Threads per worker.
    memory_limit:   Per-worker memory limit string, e.g. "4GB".
    """

    def __init__(
        self,
        tile_size_px: int = TILE_SIZE_PX,
        tile_overlap_px: int = TILE_OVERLAP_PX,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "4GB",
    ) -> None:
        self.tile_size_px = tile_size_px
        self.tile_overlap_px = tile_overlap_px
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        tiles: list[TileSpec],
        process_tile_fn,
        year: int,
        total_shape: tuple[int, int],
        n_classes: int = 12,
    ) -> MosaicResult:
        """Process all tiles and blend into a single mosaic.

        Parameters
        ----------
        tiles:           List of TileSpec from aoi.tile_aoi().
        process_tile_fn: Callable(TileSpec, year) → ClassificationResult.
        year:            Calendar year.
        total_shape:     (H, W) pixel dimensions of the full AOI.
        n_classes:       Number of landcover classes.

        Returns
        -------
        MosaicResult
        """
        H, W = total_shape
        blend_sum = np.zeros((H, W, n_classes), dtype="float64")
        weight_sum = np.zeros((H, W), dtype="float64")
        class_map_acc = np.zeros((H, W), dtype="float32")

        failed: list[TileSpec] = []
        n_ok = 0

        kernel = _cosine_blend_kernel(self.tile_size_px, self.tile_overlap_px)

        try:
            results = self._schedule_parallel(tiles, process_tile_fn, year)
        except Exception as exc:
            logger.error("Dask scheduling failed: %s — falling back to serial.", exc)
            results = self._schedule_serial(tiles, process_tile_fn, year)

        for tile, result in zip(tiles, results):
            if result is None:
                failed.append(tile)
                continue

            row0 = tile.row * self.tile_size_px
            col0 = tile.col * self.tile_size_px
            th = min(self.tile_size_px, H - row0)
            tw = min(self.tile_size_px, W - col0)
            if th <= 0 or tw <= 0:
                continue

            probs = result.class_probs[:th, :tw, :]  # (th, tw, C)
            w = kernel[:th, :tw, np.newaxis]

            blend_sum[row0:row0+th, col0:col0+tw, :] += probs * w
            weight_sum[row0:row0+th, col0:col0+tw] += w[:, :, 0]
            n_ok += 1

        # Normalize
        valid_w = weight_sum > 0
        final_probs = np.zeros((H, W, n_classes), dtype="float32")
        final_probs[valid_w] = (
            blend_sum[valid_w] / weight_sum[valid_w, np.newaxis]
        ).astype("float32")

        class_map = (final_probs.argmax(axis=-1) + 1).astype("int8")
        class_map[~valid_w] = 0
        confidence = final_probs.max(axis=-1).astype("float32")

        logger.info("Mosaic year=%d: %d/%d tiles OK, %d failed.",
                    year, n_ok, len(tiles), len(failed))

        return MosaicResult(
            class_map=class_map,
            class_probs=final_probs,
            confidence=confidence,
            n_tiles=n_ok,
            failed_tiles=failed,
            year=year,
        )

    # ── Scheduler wrappers ────────────────────────────────────────────────────

    def _schedule_parallel(
        self,
        tiles: list[TileSpec],
        process_tile_fn,
        year: int,
    ) -> list:
        """Run tile processing in parallel with a Dask LocalCluster."""
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError:
            logger.debug("dask.distributed not available — using serial schedule.")
            return self._schedule_serial(tiles, process_tile_fn, year)

        cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
        )
        client = Client(cluster)
        futures = [client.submit(process_tile_fn, tile, year) for tile in tiles]
        results = []
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as exc:
                logger.warning("Tile failed: %s", exc)
                results.append(None)

        client.close()
        cluster.close()
        return results

    @staticmethod
    def _schedule_serial(
        tiles: list[TileSpec],
        process_tile_fn,
        year: int,
    ) -> list:
        """Fallback: process tiles sequentially."""
        results = []
        for tile in tiles:
            try:
                results.append(process_tile_fn(tile, year))
            except Exception as exc:
                logger.warning("Tile %s failed: %s", tile, exc)
                results.append(None)
        return results

    # ── COG writer ────────────────────────────────────────────────────────────

    @staticmethod
    def write_cog(
        class_map: np.ndarray,
        transform,
        crs_epsg: int,
        output_path: Path,
    ) -> None:
        """Write a Cloud-Optimised GeoTIFF from an (H, W) int8 array."""
        import rasterio
        from rasterio.transform import Affine

        output_path.parent.mkdir(parents=True, exist_ok=True)
        H, W = class_map.shape

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype="int8",
            crs=f"EPSG:{crs_epsg}",
            transform=transform,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            interleave="band",
        ) as dst:
            from rasterio.enums import Resampling
            dst.write(class_map[np.newaxis, ...])
            dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")

        logger.info("COG written → %s", output_path)
