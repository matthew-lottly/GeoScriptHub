"""
aoi.py
======
Area of Interest (AOI) construction and coordinate utilities.

Builds a UTM-projected bounding box from (lon, lat) + buffer and
provides helpers consumed by the fetcher and analysis modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping


@dataclass
class AOIResult:
    """Resolved AOI geometry in both WGS-84 and UTM."""

    label: str
    bbox_wgs84: Tuple[float, float, float, float]  # (W, S, E, N)
    bbox_utm: Tuple[float, float, float, float]
    utm_crs: CRS
    geojson: dict                                    # GeoJSON geometry

    @property
    def width_m(self) -> float:
        return self.bbox_utm[2] - self.bbox_utm[0]

    @property
    def height_m(self) -> float:
        return self.bbox_utm[3] - self.bbox_utm[1]


class AOIBuilder:
    """Factory for creating AOI definitions."""

    @staticmethod
    def from_point(lon: float, lat: float, buffer_km: float = 1.0) -> AOIResult:
        """Create a square AOI centred on a point.

        Parameters
        ----------
        lon, lat : WGS-84 longitude / latitude.
        buffer_km : Half-side length in kilometres.
        """
        # Determine UTM zone
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        utm_crs = CRS.from_proj4(
            f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"
        )

        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        cx, cy = to_utm.transform(lon, lat)
        buf = buffer_km * 1000.0

        x_min, y_min = cx - buf, cy - buf
        x_max, y_max = cx + buf, cy + buf

        w, s = to_wgs.transform(x_min, y_min)
        e, n = to_wgs.transform(x_max, y_max)

        geojson = mapping(box(w, s, e, n))

        label = f"Point({lon:.4f},{lat:.4f})+{buffer_km}km"
        return AOIResult(
            label=label,
            bbox_wgs84=(w, s, e, n),
            bbox_utm=(x_min, y_min, x_max, y_max),
            utm_crs=utm_crs,
            geojson=geojson,
        )

    @staticmethod
    def from_bbox(west: float, south: float, east: float, north: float) -> AOIResult:
        """Create an AOI from a WGS-84 bounding box."""
        lon_c = (west + east) / 2.0
        lat_c = (south + north) / 2.0
        utm_zone = int((lon_c + 180) / 6) + 1
        hemisphere = "north" if lat_c >= 0 else "south"
        utm_crs = CRS.from_proj4(
            f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"
        )
        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

        x0, y0 = to_utm.transform(west, south)
        x1, y1 = to_utm.transform(east, north)

        return AOIResult(
            label=f"BBOX({west:.4f},{south:.4f},{east:.4f},{north:.4f})",
            bbox_wgs84=(west, south, east, north),
            bbox_utm=(x0, y0, x1, y1),
            utm_crs=utm_crs,
            geojson=mapping(box(west, south, east, north)),
        )

    @staticmethod
    def summarise(aoi: AOIResult) -> None:
        """Print a human-readable AOI summary."""
        w, s, e, n = aoi.bbox_wgs84
        print("=== AOI Summary ======================================")
        print(f"  Label       : {aoi.label}")
        print(f"  WGS84 bbox  : W={w:.4f}  S={s:.4f}  E={e:.4f}  N={n:.4f}")
        print(f"  UTM CRS     : {aoi.utm_crs.to_epsg()}")
        print(f"  Extent      : {aoi.width_m:.0f} x {aoi.height_m:.0f} m")
        print(f"  Area         : {aoi.width_m * aoi.height_m / 1e6:.2f} km²")
        print("======================================================")
