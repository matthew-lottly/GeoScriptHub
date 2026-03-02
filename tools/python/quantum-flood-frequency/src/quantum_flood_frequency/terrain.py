"""
terrain.py
==========
DEM-based terrain analysis for flood extent constraint.

Uses the Copernicus GLO-30 DEM (30 m, global) to derive terrain
features that physically constrain where flooding can occur:

1. **Elevation** — raw surface height (m above geoid).
2. **Slope** — terrain gradient in degrees.  Steep slopes shed water
   rapidly and are unlikely to be inundated.
3. **HAND** — Height Above Nearest Drainage.  This is the vertical
   distance from each pixel to the nearest stream/drainage path.
   HAND is the most powerful single predictor of flood inundation:
   * HAND < 5 m → high flood probability
   * HAND 5–15 m → moderate flood probability
   * HAND > 15 m → very low flood probability (buildings on hills)

The HAND computation uses a simplified approach (distance to local
elevation minimum within a search radius) that approximates the
full hydrological HAND without requiring a complete flow-direction
analysis.

References
----------
* Nobre et al., "Height Above the Nearest Drainage — a hydrologically
  relevant new terrain model", J. Hydrology 404, 2011.
* Jafarzadegan & Merwade, "A DEM-based approach for large-scale
  floodplain mapping", J. Hydrology 550, 2017.
* Copernicus DEM documentation: https://spacedata.copernicus.eu/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import (
    minimum_filter,
    uniform_filter,
    zoom,
    label,
    gaussian_filter,
)

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.terrain")

# ---------------------------------------------------------------------------
# Terrain thresholds
# ---------------------------------------------------------------------------

# HAND thresholds (metres above nearest drainage)
HAND_HIGH_RISK = 5.0       # < 5 m → high flood risk
HAND_MODERATE_RISK = 15.0  # 5–15 m → moderate risk
HAND_LOW_RISK = 30.0       # > 30 m → negligible flood risk

# Slope threshold (degrees) — steeper than this rarely floods
SLOPE_FLOOD_MAX = 5.0      # degrees — flat to gentle slopes only

# HAND search radius (pixels) — controls drainage search extent
HAND_SEARCH_RADIUS = 100   # pixels (~100 m at 1 m, ~1 km at 10 m)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TerrainFeatures:
    """DEM-derived terrain feature layers for flood constraint.

    All arrays are 2-D float32 at the target analysis resolution.

    Attributes:
        elevation: Surface elevation (m above geoid).
        slope: Terrain slope (degrees, 0–90).
        hand: Height Above Nearest Drainage (m).
        flood_susceptibility: Combined flood susceptibility [0, 1].
        terrain_mask: Boolean — True where terrain allows flooding.
        resolution: DEM source resolution (metres).
        valid_mask: Boolean — True where DEM data exists.
    """

    elevation: np.ndarray
    slope: np.ndarray
    hand: np.ndarray
    flood_susceptibility: np.ndarray
    terrain_mask: np.ndarray
    resolution: float = 30.0
    valid_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    def __repr__(self) -> str:
        return (
            f"<TerrainFeatures  {self.elevation.shape[1]}×{self.elevation.shape[0]} px  "
            f"elev=[{np.nanmin(self.elevation):.0f}–{np.nanmax(self.elevation):.0f}] m  "
            f"hand=[{np.nanmin(self.hand):.1f}–{np.nanmax(self.hand):.1f}] m  "
            f"floodable={self.terrain_mask.sum()} px>"
        )


# ---------------------------------------------------------------------------
# Terrain processing functions
# ---------------------------------------------------------------------------

def compute_slope(
    elevation: np.ndarray,
    cell_size: float = 1.0,
) -> np.ndarray:
    """Compute terrain slope from DEM using finite differences.

    Uses a 3×3 Horn (1981) algorithm — the same method used by
    GDAL ``gdaldem slope`` and ArcGIS Spatial Analyst.

    slope (degrees) = arctan(√(dz/dx² + dz/dy²)) × 180/π

    Args:
        elevation: 2-D elevation array (metres).
        cell_size: Pixel size in metres (for correct gradient scaling).

    Returns:
        Slope in degrees (0–90), float32.
    """
    elev = elevation.astype("float64")
    # Pad to handle edges
    padded = np.pad(elev, 1, mode="reflect")

    # Horn's method — 3×3 weighted finite differences
    # dz/dx (east–west gradient)
    dzdx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * cell_size)

    # dz/dy (north–south gradient)
    dzdy = (
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8.0 * cell_size)

    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad)

    return slope_deg.astype("float32")


def compute_hand(
    elevation: np.ndarray,
    search_radius: int = HAND_SEARCH_RADIUS,
) -> np.ndarray:
    """Compute simplified Height Above Nearest Drainage (HAND).

    Uses a moving-window minimum filter to approximate the elevation
    of the nearest drainage path, then computes the vertical height
    above that drainage elevation.

    This is a simplified approximation that does not require full
    hydrological flow-direction analysis.  For flat urban areas
    (Houston), this is typically within 1–2 m of the true HAND.

    HAND = elevation − local_minimum_elevation

    Args:
        elevation: 2-D elevation array (metres).
        search_radius: Search window half-size in pixels.

    Returns:
        HAND in metres (float32, ≥ 0).
    """
    elev = elevation.astype("float64")

    # Apply multi-scale minimum filter for robust drainage detection
    # Use progressively larger windows to find drainage at different scales
    min_elev = elev.copy()
    for scale in [search_radius // 4, search_radius // 2, search_radius]:
        if scale < 3:
            scale = 3
        window = 2 * scale + 1
        local_min = minimum_filter(elev, size=window, mode="reflect")
        min_elev = np.minimum(min_elev, local_min)

    # HAND = elevation above nearest drainage
    hand = np.maximum(elev - min_elev, 0.0)

    return hand.astype("float32")


def compute_flood_susceptibility(
    hand: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """Compute combined flood susceptibility index from HAND and slope.

    susceptibility = sigmoid(-HAND/5) × sigmoid(-slope/3)

    This produces a smooth [0, 1] index where:
    * 1.0 = flat, low-lying → extremely flood-susceptible
    * 0.0 = steep, elevated → not flood-susceptible

    Args:
        hand: HAND array (metres).
        slope: Slope array (degrees).

    Returns:
        Flood susceptibility [0, 1], float32.
    """
    # Sigmoid decay: drops to 0.5 at hand=5m, ~0.01 at hand=23m
    hand_factor = 1.0 / (1.0 + np.exp(hand / 5.0 - 1.0))

    # Sigmoid decay: drops to 0.5 at slope=3°, ~0.01 at slope=17°
    slope_factor = 1.0 / (1.0 + np.exp(slope / 3.0 - 1.0))

    susceptibility = (hand_factor * slope_factor).astype("float32")
    return np.clip(susceptibility, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Terrain Processor
# ---------------------------------------------------------------------------

class TerrainProcessor:
    """Process DEM data into terrain constraint features for flood mapping.

    Computes elevation, slope, HAND, and a combined flood susceptibility
    surface from the Copernicus GLO-30 DEM.

    Parameters
    ----------
    target_shape:
        Output (rows, cols) at analysis resolution.
    target_resolution:
        Target pixel size in metres (for slope computation).
    hand_search_radius:
        HAND local-minimum search radius in pixels.
    """

    def __init__(
        self,
        target_shape: tuple[int, int],
        target_resolution: float = 1.0,
        hand_search_radius: int = HAND_SEARCH_RADIUS,
    ) -> None:
        self.target_shape = target_shape
        self.target_resolution = target_resolution
        self.hand_search_radius = hand_search_radius

    def process(
        self,
        dem: np.ndarray,
        dem_resolution: float = 30.0,
    ) -> TerrainFeatures:
        """Process raw DEM into terrain features at target resolution.

        Args:
            dem: Raw DEM array (metres above geoid).
            dem_resolution: Source DEM pixel size in metres.

        Returns:
            TerrainFeatures with all derived layers.
        """
        logger.info(
            "Processing DEM (%d × %d @ %.0f m) → target %d × %d @ %.0f m",
            dem.shape[1], dem.shape[0], dem_resolution,
            self.target_shape[1], self.target_shape[0], self.target_resolution,
        )

        # Clean DEM
        dem = dem.astype("float32")
        dem = np.where(np.isfinite(dem), dem, np.nanmean(dem))

        # Compute terrain features at native DEM resolution
        slope_native = compute_slope(dem, cell_size=dem_resolution)
        hand_native = compute_hand(dem, search_radius=self.hand_search_radius)

        # Resample all layers to target shape
        elevation = self._resample(dem, self.target_shape)
        slope = self._resample(slope_native, self.target_shape)
        hand = self._resample(hand_native, self.target_shape)

        # Compute flood susceptibility
        flood_susc = compute_flood_susceptibility(hand, slope)

        # Terrain mask: allow flooding where HAND < threshold AND slope < threshold
        terrain_mask = (hand < HAND_LOW_RISK) & (slope < SLOPE_FLOOD_MAX * 3)

        valid_mask = np.isfinite(elevation)

        logger.info(
            "Terrain features: elev=[%.0f–%.0f]m, HAND=[%.1f–%.1f]m, "
            "floodable=%d px (%.1f%%)",
            np.nanmin(elevation), np.nanmax(elevation),
            np.nanmin(hand), np.nanmax(hand),
            terrain_mask.sum(),
            100.0 * terrain_mask.sum() / terrain_mask.size,
        )

        return TerrainFeatures(
            elevation=elevation,
            slope=slope,
            hand=hand,
            flood_susceptibility=flood_susc,
            terrain_mask=terrain_mask,
            resolution=dem_resolution,
            valid_mask=valid_mask,
        )

    def create_empty(self) -> TerrainFeatures:
        """Create placeholder terrain features (no DEM available)."""
        shape = self.target_shape
        return TerrainFeatures(
            elevation=np.full(shape, np.nan, dtype="float32"),
            slope=np.zeros(shape, dtype="float32"),
            hand=np.zeros(shape, dtype="float32"),
            flood_susceptibility=np.ones(shape, dtype="float32"),
            terrain_mask=np.ones(shape, dtype=bool),
            resolution=0.0,
            valid_mask=np.zeros(shape, dtype=bool),
        )

    @staticmethod
    def _resample(
        arr: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Bilinear resample to target shape."""
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
