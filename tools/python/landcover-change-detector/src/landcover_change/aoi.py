"""Area-of-interest builder.

Constructs a WGS-84 bounding box, projects to UTM, and provides
the spatial reference for all downstream processing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from shapely.geometry import box as shapely_box

from .constants import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON, DEFAULT_BUFFER_KM, DEFAULT_CRS

logger = logging.getLogger("geoscripthub.landcover_change.aoi")


@dataclass(frozen=True)
class AOIResult:
    """Immutable container for the resolved area of interest."""

    bbox_wgs84: tuple[float, float, float, float]  # (west, south, east, north)
    bbox_utm: tuple[float, float, float, float]
    geometry_wgs84: object  # shapely Polygon
    geometry_utm: object
    target_crs: str
    center_lat: float
    center_lon: float
    area_km2: float
    description: str


class AOIBuilder:
    """Build an AOIResult from centre coordinates and a buffer radius."""

    def __init__(
        self,
        center_lat: float = DEFAULT_CENTER_LAT,
        center_lon: float = DEFAULT_CENTER_LON,
        buffer_km: float = DEFAULT_BUFFER_KM,
        target_crs: str = DEFAULT_CRS,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.buffer_km = buffer_km
        self.target_crs = target_crs

    def build(self) -> AOIResult:
        """Compute WGS-84 and UTM bounding boxes."""
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.radians(self.center_lat))

        d_lat = self.buffer_km / km_per_deg_lat
        d_lon = self.buffer_km / km_per_deg_lon

        west = self.center_lon - d_lon
        east = self.center_lon + d_lon
        south = self.center_lat - d_lat
        north = self.center_lat + d_lat

        bbox_wgs84 = (west, south, east, north)
        geom_wgs84 = shapely_box(*bbox_wgs84)

        transformer = Transformer.from_crs(
            "EPSG:4326", self.target_crs, always_xy=True,
        )
        x_min, y_min = transformer.transform(west, south)
        x_max, y_max = transformer.transform(east, north)
        bbox_utm = (x_min, y_min, x_max, y_max)
        geom_utm = shapely_box(*bbox_utm)

        area_km2 = (x_max - x_min) * (y_max - y_min) / 1e6

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
                f"AOI: {self.center_lat:.4f}°N, {abs(self.center_lon):.4f}°W "
                f"± {self.buffer_km:.1f} km ({area_km2:.1f} km²)"
            ),
        )
        logger.info(result.description)
        return result
