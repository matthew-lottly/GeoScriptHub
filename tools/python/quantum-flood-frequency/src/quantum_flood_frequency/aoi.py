"""
aoi.py
======
Define the Area of Interest for the Houston, TX flood study.

The default AOI is centred on Houston, Texas (29.76°N, 95.37°W),
a metropolitan area with significant flood exposure along Buffalo
Bayou, Brays Bayou, and surrounding watersheds.

All coordinates are returned in both WGS-84 (EPSG:4326) for STAC
queries and in the appropriate UTM zone (EPSG:15N) for raster
analysis at metre-level precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import box, Polygon
from pyproj import Transformer

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.aoi")

# ---------------------------------------------------------------------------
# Houston, TX — flood-prone metropolitan area
# Downtown Houston ≈ 29.76°N, 95.37°W
# Covers Buffalo Bayou, Brays Bayou, and surrounding watersheds
# ---------------------------------------------------------------------------
_DEFAULT_CENTER_LAT = 29.76
_DEFAULT_CENTER_LON = -95.37
_DEFAULT_BUFFER_KM = 5.0  # ~5 km in each direction from centre

# Target CRS for all raster processing
TARGET_CRS = "EPSG:32615"  # UTM Zone 15N (covers Houston, TX)
WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class AOIResult:
    """Resolved Area of Interest for the flood frequency study.

    Attributes:
        bbox_wgs84: Bounding box in WGS-84 as (west, south, east, north).
        bbox_utm: Bounding box in UTM as (west, south, east, north) in metres.
        geometry_wgs84: Shapely polygon of the AOI in WGS-84.
        geometry_utm: Shapely polygon in UTM projection.
        target_crs: EPSG string for the target projected CRS.
        center_lat: Centre latitude (WGS-84).
        center_lon: Centre longitude (WGS-84).
        area_km2: Approximate area in square kilometres.
    """

    bbox_wgs84: tuple[float, float, float, float]
    bbox_utm: tuple[float, float, float, float]
    geometry_wgs84: Polygon
    geometry_utm: Polygon
    target_crs: str
    center_lat: float
    center_lon: float
    area_km2: float
    description: str = ""


class AOIBuilder:
    """Construct the study-area bounding box for the Houston flood study.

    The default configuration targets central Houston, TX.
    Users may override the centre coordinates and buffer distance.

    Parameters
    ----------
    center_lat:
        Latitude of AOI centre in decimal degrees (WGS-84).
    center_lon:
        Longitude of AOI centre in decimal degrees (WGS-84).
    buffer_km:
        Half-width of the bounding box in kilometres.
    target_crs:
        Target projected CRS (EPSG string).  Defaults to UTM 15N.
    """

    def __init__(
        self,
        center_lat: float = _DEFAULT_CENTER_LAT,
        center_lon: float = _DEFAULT_CENTER_LON,
        buffer_km: float = _DEFAULT_BUFFER_KM,
        target_crs: str = TARGET_CRS,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.buffer_km = buffer_km
        self.target_crs = target_crs

    def build(self) -> AOIResult:
        """Compute the bounding box and return a frozen AOIResult.

        Returns:
            AOIResult with WGS-84 and UTM bounding boxes / geometries.
        """
        logger.info(
            "Building AOI centred at (%.4f, %.4f), buffer %.1f km",
            self.center_lat,
            self.center_lon,
            self.buffer_km,
        )

        # Approximate degree offsets from km (at this latitude)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.radians(self.center_lat))

        dlat = self.buffer_km / km_per_deg_lat
        dlon = self.buffer_km / km_per_deg_lon

        west = self.center_lon - dlon
        east = self.center_lon + dlon
        south = self.center_lat - dlat
        north = self.center_lat + dlat

        bbox_wgs84 = (west, south, east, north)
        geom_wgs84 = box(*bbox_wgs84)

        # Transform to UTM
        to_utm = Transformer.from_crs(WGS84, self.target_crs, always_xy=True)
        from_utm = Transformer.from_crs(self.target_crs, WGS84, always_xy=True)

        utm_coords = [to_utm.transform(x, y) for x, y in geom_wgs84.exterior.coords]
        geom_utm = Polygon(utm_coords)
        b = geom_utm.bounds
        bbox_utm = (b[0], b[1], b[2], b[3])

        area_km2 = geom_utm.area / 1e6

        result = AOIResult(
            bbox_wgs84=bbox_wgs84,
            bbox_utm=bbox_utm,
            geometry_wgs84=geom_wgs84,
            geometry_utm=geom_utm,
            target_crs=self.target_crs,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            area_km2=area_km2,
            description=(
                f"Houston, TX flood study area, "
                f"centred at ({self.center_lat:.4f}°N, {abs(self.center_lon):.4f}°W), "
                f"≈{area_km2:.1f} km²"
            ),
        )

        logger.info("AOI resolved: %s", result.description)
        logger.debug("WGS-84 bbox: %s", bbox_wgs84)
        logger.debug("UTM bbox: %s", bbox_utm)

        return result
