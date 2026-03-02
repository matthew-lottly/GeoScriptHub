"""
tiled_pipeline.py
=================
Tiled / batched processing for large-raster flood frequency analysis.

At NAIP resolution (1 m), a typical study area (6 km × 6 km) produces
~36 million pixels per observation.  Processing hundreds of temporal
observations at this resolution requires careful memory management.

This module provides:

1. **TileGrid** — divides the full AOI into overlapping tiles with
   configurable size and overlap (border blending zone).

2. **TiledAccumulator** — processes observations one-at-a-time,
   accumulating water/observation counts per tile, then stitches
   the tiles into the full-extent frequency surface.

3. **BatchScheduler** — groups observations into memory-efficient
   batches, ensuring peak RAM usage stays within a configurable
   soft limit.

Architecture
------------
::

    Full AOI (6000 × 6000 px @ 1 m)
    ┌──────┬──────┬──────┐
    │ T1   │ T2   │ T3   │     Each tile: 512×512 + 32px overlap
    ├──────┼──────┼──────┤     → 576×576 px per tile
    │ T4   │ T5   │ T6   │     12×12 = 144 tiles in this example
    ├──────┼──────┼──────┤
    │ T7   │ T8   │ T9   │     Per-tile processing:
    └──────┴──────┴──────┘       1. Extract tile extent
                                  2. Classify → water binary mask
                                  3. Accumulate water_count, obs_count
                                  4. Repeat for next observation
                                  5. Stitch tiles → full frequency

Memory budget (example):
  - 1 tile, 6 bands, float32: 576² × 6 × 4 = 7.9 MB
  - Running counters (full AOI): 6000² × 2 × 2 = 144 MB
  - Total for 1 obs: ~152 MB — well within laptop RAM.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.tiled_pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TILE_SIZE = 512      # pixels per tile edge (at analysis resolution)
DEFAULT_OVERLAP = 32         # overlap border in pixels
MAX_MEMORY_MB = 4096         # soft memory cap (MB)


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    """A single tile within the processing grid.

    Coordinates are in *pixel* space (row, col) of the full-extent raster.
    The ``padded_*`` attributes include the overlap border.

    Attributes:
        tile_id: Sequential tile identifier.
        row: Top-left pixel row (inner, no overlap).
        col: Top-left pixel column (inner, no overlap).
        height: Inner tile height in pixels.
        width: Inner tile width in pixels.
        padded_row: Top-left row including overlap.
        padded_col: Top-left column including overlap.
        padded_height: Total height including overlap.
        padded_width: Total width including overlap.
    """
    tile_id: int
    row: int
    col: int
    height: int
    width: int
    padded_row: int
    padded_col: int
    padded_height: int
    padded_width: int


class TileGrid:
    """Divide a raster extent into overlapping tiles.

    Parameters
    ----------
    rows, cols:
        Full raster dimensions in pixels.
    tile_size:
        Inner tile size (pixels).
    overlap:
        Overlap border (pixels) for seamless stitching.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.tile_size = tile_size
        self.overlap = overlap

        self.n_tiles_y = math.ceil(rows / tile_size)
        self.n_tiles_x = math.ceil(cols / tile_size)
        self.n_tiles = self.n_tiles_y * self.n_tiles_x

        self.tiles = self._build_tiles()

        logger.info(
            "Tile grid: %d × %d = %d tiles (tile=%d px, overlap=%d px) "
            "for %d × %d full extent",
            self.n_tiles_x, self.n_tiles_y, self.n_tiles,
            tile_size, overlap, cols, rows,
        )

    def _build_tiles(self) -> list[Tile]:
        tiles = []
        tid = 0

        for ty in range(self.n_tiles_y):
            for tx in range(self.n_tiles_x):
                # Inner tile bounds
                r0 = ty * self.tile_size
                c0 = tx * self.tile_size
                r1 = min(r0 + self.tile_size, self.rows)
                c1 = min(c0 + self.tile_size, self.cols)
                h = r1 - r0
                w = c1 - c0

                # Padded bounds (with overlap, clamped to raster extent)
                pr0 = max(r0 - self.overlap, 0)
                pc0 = max(c0 - self.overlap, 0)
                pr1 = min(r1 + self.overlap, self.rows)
                pc1 = min(c1 + self.overlap, self.cols)

                tiles.append(Tile(
                    tile_id=tid,
                    row=r0, col=c0, height=h, width=w,
                    padded_row=pr0, padded_col=pc0,
                    padded_height=pr1 - pr0, padded_width=pc1 - pc0,
                ))
                tid += 1

        return tiles

    def extract_tile(self, data: np.ndarray, tile: Tile) -> np.ndarray:
        """Extract a padded tile from a full-extent array.

        Args:
            data: Full-extent 2-D array.
            tile: Tile specification.

        Returns:
            2-D array of shape (padded_height, padded_width).
        """
        return data[
            tile.padded_row: tile.padded_row + tile.padded_height,
            tile.padded_col: tile.padded_col + tile.padded_width,
        ].copy()

    def insert_tile(
        self,
        target: np.ndarray,
        tile_data: np.ndarray,
        tile: Tile,
    ) -> None:
        """Insert the inner (non-overlap) portion of a processed tile
        back into the full-extent array.

        Args:
            target: Full-extent output array (modified in-place).
            tile_data: Processed padded tile array.
            tile: Tile specification.
        """
        # Compute the offset of the inner region within the padded tile
        inner_r0 = tile.row - tile.padded_row
        inner_c0 = tile.col - tile.padded_col

        target[
            tile.row: tile.row + tile.height,
            tile.col: tile.col + tile.width,
        ] = tile_data[
            inner_r0: inner_r0 + tile.height,
            inner_c0: inner_c0 + tile.width,
        ]

    def __repr__(self) -> str:
        return (
            f"<TileGrid  {self.n_tiles_x}×{self.n_tiles_y} = {self.n_tiles} tiles  "
            f"tile={self.tile_size}px  overlap={self.overlap}px>"
        )


# ---------------------------------------------------------------------------
# Tiled accumulator for frequency computation
# ---------------------------------------------------------------------------

class TiledAccumulator:
    """Accumulate water classifications across observations using tiled processing.

    Maintains running ``water_count`` and ``obs_count`` arrays at
    full extent. Each observation is processed tile-by-tile to limit
    peak memory usage.

    Parameters
    ----------
    rows, cols:
        Full raster dimensions.
    tile_size:
        Inner tile size in pixels.
    overlap:
        Tile overlap in pixels.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.grid = TileGrid(rows, cols, tile_size, overlap)

        # Running accumulators (full extent — uint16 supports up to 65535 obs)
        self.water_count = np.zeros((rows, cols), dtype="uint16")
        self.obs_count = np.zeros((rows, cols), dtype="uint16")
        self.n_observations = 0

    def accumulate(
        self,
        water_binary: np.ndarray,
        cloud_mask: np.ndarray,
    ) -> None:
        """Add one classified observation to the accumulators.

        Only cloud-free pixels (cloud_mask=True) are counted.

        Args:
            water_binary: Full-extent boolean water mask.
            cloud_mask: Full-extent boolean clear-sky mask.
        """
        valid = cloud_mask & np.isfinite(water_binary.astype("float32"))
        is_water = water_binary & valid

        self.water_count += is_water.astype("uint16")
        self.obs_count += valid.astype("uint16")
        self.n_observations += 1

    def accumulate_tiled(
        self,
        classify_fn: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
        observation_bands: dict[str, np.ndarray],
    ) -> None:
        """Classify and accumulate one observation using tiled processing.

        Args:
            classify_fn: Function (bands_tile) → (water_binary, cloud_mask).
            observation_bands: Dict of band_name → full-extent array.
        """
        for tile in self.grid.tiles:
            # Extract tile for each band
            tile_bands = {}
            for name, arr in observation_bands.items():
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    tile_bands[name] = self.grid.extract_tile(arr, tile)
                else:
                    tile_bands[name] = arr  # scalar / metadata

            # Classify this tile
            water_tile, cloud_tile = classify_fn(tile_bands)

            # Accumulate inner region only
            inner_r0 = tile.row - tile.padded_row
            inner_c0 = tile.col - tile.padded_col

            water_inner = water_tile[
                inner_r0: inner_r0 + tile.height,
                inner_c0: inner_c0 + tile.width,
            ]
            cloud_inner = cloud_tile[
                inner_r0: inner_r0 + tile.height,
                inner_c0: inner_c0 + tile.width,
            ]

            valid = cloud_inner & np.isfinite(water_inner.astype("float32"))
            is_water = water_inner & valid

            self.water_count[
                tile.row: tile.row + tile.height,
                tile.col: tile.col + tile.width,
            ] += is_water.astype("uint16")

            self.obs_count[
                tile.row: tile.row + tile.height,
                tile.col: tile.col + tile.width,
            ] += valid.astype("uint16")

        self.n_observations += 1

    def compute_frequency(self) -> np.ndarray:
        """Compute flood inundation frequency from accumulated counts.

        Returns:
            2-D float32 array [0, 1] — fraction inundated.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = np.where(
                self.obs_count > 0,
                self.water_count.astype("float32") / self.obs_count.astype("float32"),
                np.nan,
            )
        return frequency

    def __repr__(self) -> str:
        return (
            f"<TiledAccumulator  {self.cols}×{self.rows} px  "
            f"{self.n_observations} observations  "
            f"grid={self.grid}>"
        )


# ---------------------------------------------------------------------------
# Batch scheduler
# ---------------------------------------------------------------------------

class BatchScheduler:
    """Group observations into memory-efficient batches.

    Estimates memory required per observation and creates batches
    that fit within the configured memory limit.

    Parameters
    ----------
    rows, cols:
        Full raster dimensions.
    n_bands:
        Number of spectral bands per observation.
    max_memory_mb:
        Soft memory cap in megabytes.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        n_bands: int = 6,
        max_memory_mb: float = MAX_MEMORY_MB,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.n_bands = n_bands
        self.max_memory_mb = max_memory_mb

        # Memory per observation (bytes): bands × float32 + masks × bool
        self.bytes_per_obs = (
            rows * cols * n_bands * 4    # spectral bands (float32)
            + rows * cols * 1            # cloud mask (bool)
            + rows * cols * 1            # water mask (bool)
        )
        self.mb_per_obs = self.bytes_per_obs / (1024 * 1024)

        # How many observations fit in memory simultaneously
        self.batch_size = max(1, int(max_memory_mb / self.mb_per_obs))

        logger.info(
            "Batch scheduler: %.1f MB/obs → batch size %d (limit %.0f MB)",
            self.mb_per_obs, self.batch_size, max_memory_mb,
        )

    def create_batches(self, n_observations: int) -> list[tuple[int, int]]:
        """Create batch index ranges.

        Args:
            n_observations: Total number of observations.

        Returns:
            List of (start_idx, end_idx) tuples.
        """
        batches = []
        for start in range(0, n_observations, self.batch_size):
            end = min(start + self.batch_size, n_observations)
            batches.append((start, end))

        logger.info(
            "Scheduled %d observations into %d batches (size %d)",
            n_observations, len(batches), self.batch_size,
        )
        return batches
