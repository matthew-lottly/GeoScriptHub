"""aoi.py — Area-of-interest builder for Austin, TX.

Constructs bounding-box AOIs in WGS-84 and UTM Zone 14N (EPSG:32614),
produces STAC-compatible dicts, and generates tile grids for the
dask-parallel mosaic engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box, mapping
from shapely.ops import transform

from .constants import (
    AUSTIN_CENTER_LAT,
    AUSTIN_CENTER_LON,
    AUSTIN_BUFFER_KM,
    AUSTIN_UTM_CRS,
    AUSTIN_WGS84_CRS,
    RESOLUTION_CHANGE_M,
    TILE_SIZE_PX,
    TILE_OVERLAP_PX,
)


@dataclass
class AOIResult:
    """Fully described study-area AOI.

    Attributes
    ----------
    center_lat:     Centre latitude in WGS-84 decimal degrees.
    center_lon:     Centre longitude in WGS-84 decimal degrees.
    buffer_km:      Half-width of bounding box in km.
    wgs84_poly:     Shapely Polygon in EPSG:4326.
    utm_poly:       Shapely Polygon in target_crs (UTM).
    bbox_wgs84:     (minx, miny, maxx, maxy) in WGS-84.
    bbox_utm:       (minx, miny, maxx, maxy) in UTM metres.
    target_crs:     EPSG string for projected CRS.
    wgs84_gdf:      GeoDataFrame (EPSG:4326, single row).
    utm_gdf:        GeoDataFrame (UTM, single row).
    stac_bbox:      List [minx, miny, maxx, maxy] for STAC queries.
    """

    center_lat: float
    center_lon: float
    buffer_km: float
    wgs84_poly: BaseGeometry
    utm_poly: BaseGeometry
    bbox_wgs84: tuple[float, float, float, float]
    bbox_utm: tuple[float, float, float, float]
    target_crs: str
    wgs84_gdf: gpd.GeoDataFrame
    utm_gdf: gpd.GeoDataFrame
    stac_bbox: list[float]


@dataclass
class TileSpec:
    """A single processing tile within the AOI.

    Attributes
    ----------
    col:        Column index (0-based).
    row:        Row index (0-based).
    bbox_utm:   (minx, miny, maxx, maxy) in target_crs metres.
    bbox_wgs84: (minx, miny, maxx, maxy) in WGS-84 degrees.
    width_px:   Tile width in pixels at the target resolution.
    height_px:  Tile height in pixels.
    """

    col: int
    row: int
    bbox_utm: tuple[float, float, float, float]
    bbox_wgs84: tuple[float, float, float, float]
    width_px: int
    height_px: int


class AOIBuilder:
    """Build an :class:`AOIResult` from a centre point and buffer.

    Parameters
    ----------
    center_lat:
        Latitude of AOI centre in WGS-84 decimal degrees.
    center_lon:
        Longitude of AOI centre in WGS-84 decimal degrees.
    buffer_km:
        Half-width of square bounding box in kilometres.
    target_crs:
        EPSG string for the projected CRS used throughout the pipeline.
        Defaults to UTM Zone 14N (EPSG:32614) for Austin, TX.

    Examples
    --------
    >>> aoi = AOIBuilder(30.2672, -97.7431, 55.0).build()
    >>> print(aoi.target_crs)
    EPSG:32614
    """

    def __init__(
        self,
        center_lat: float = AUSTIN_CENTER_LAT,
        center_lon: float = AUSTIN_CENTER_LON,
        buffer_km: float = AUSTIN_BUFFER_KM,
        target_crs: str = AUSTIN_UTM_CRS,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.buffer_km = buffer_km
        self.target_crs = target_crs

    # ── Public interface ───────────────────────────────────────────────────────

    def build(self) -> AOIResult:
        """Compute the full AOIResult for the configured parameters.

        Returns
        -------
        AOIResult
            All AOI representations needed by downstream pipeline modules.
        """
        # Convert buffer from km to degrees (approximate — 1° lat ≈ 111.32 km)
        buf_lat_deg = self.buffer_km / 111.32
        buf_lon_deg = self.buffer_km / (111.32 * math.cos(math.radians(self.center_lat)))

        minx_wgs = self.center_lon - buf_lon_deg
        maxx_wgs = self.center_lon + buf_lon_deg
        miny_wgs = self.center_lat - buf_lat_deg
        maxy_wgs = self.center_lat + buf_lat_deg

        wgs84_poly = box(minx_wgs, miny_wgs, maxx_wgs, maxy_wgs)

        # Project to UTM
        wgs84_to_utm = pyproj.Transformer.from_crs(
            AUSTIN_WGS84_CRS, self.target_crs, always_xy=True
        ).transform
        utm_poly = transform(wgs84_to_utm, wgs84_poly)
        utm_bounds = utm_poly.bounds  # (minx, miny, maxx, maxy)

        # Also project corners to get the tight UTM bbox
        minx_utm, miny_utm, maxx_utm, maxy_utm = utm_bounds

        wgs84_gdf = gpd.GeoDataFrame(
            {"geometry": [wgs84_poly]}, crs=AUSTIN_WGS84_CRS
        )
        utm_gdf = gpd.GeoDataFrame(
            {"geometry": [utm_poly]}, crs=self.target_crs
        )

        return AOIResult(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            buffer_km=self.buffer_km,
            wgs84_poly=wgs84_poly,
            utm_poly=utm_poly,
            bbox_wgs84=(minx_wgs, miny_wgs, maxx_wgs, maxy_wgs),
            bbox_utm=(minx_utm, miny_utm, maxx_utm, maxy_utm),
            target_crs=self.target_crs,
            wgs84_gdf=wgs84_gdf,
            utm_gdf=utm_gdf,
            stac_bbox=[minx_wgs, miny_wgs, maxx_wgs, maxy_wgs],
        )

    def tile_aoi(
        self,
        aoi: Optional[AOIResult] = None,
        tile_size_px: int = TILE_SIZE_PX,
        overlap_px: int = TILE_OVERLAP_PX,
        resolution_m: int = RESOLUTION_CHANGE_M,
    ) -> list[TileSpec]:
        """Decompose the AOI into a grid of overlapping tiles.

        Parameters
        ----------
        aoi:
            Pre-built AOIResult; if None, ``build()`` is called first.
        tile_size_px:
            Target tile dimension in pixels (square tiles).
        overlap_px:
            Edge overlap in pixels between neighbouring tiles.
        resolution_m:
            Ground-sample distance in metres.

        Returns
        -------
        list[TileSpec]
            Ordered list of tile specs (row-major).
        """
        if aoi is None:
            aoi = self.build()

        utm_to_wgs = pyproj.Transformer.from_crs(
            self.target_crs, AUSTIN_WGS84_CRS, always_xy=True
        ).transform

        minx, miny, maxx, maxy = aoi.bbox_utm
        step_m = tile_size_px * resolution_m         # metres per tile step
        overlap_m = overlap_px * resolution_m        # metres of overlap

        xs = _frange(minx, maxx, step_m - overlap_m)
        ys = _frange(miny, maxy, step_m - overlap_m)

        tiles: list[TileSpec] = []
        for r, ty in enumerate(ys):
            for c, tx in enumerate(xs):
                tx1 = min(tx + step_m, maxx)
                ty1 = min(ty + step_m, maxy)

                utm_tile_box = box(tx, ty, tx1, ty1)
                wgs_tile_box = transform(utm_to_wgs, utm_tile_box)
                wx0, wy0, wx1, wy1 = wgs_tile_box.bounds

                w_px = max(1, round((tx1 - tx) / resolution_m))
                h_px = max(1, round((ty1 - ty) / resolution_m))

                tiles.append(
                    TileSpec(
                        col=c,
                        row=r,
                        bbox_utm=(tx, ty, tx1, ty1),
                        bbox_wgs84=(wx0, wy0, wx1, wy1),
                        width_px=w_px,
                        height_px=h_px,
                    )
                )

        return tiles


# ── Pre-built singleton for Austin metro ──────────────────────────────────────

def AustinMetroAOI() -> AOIResult:
    """Return the pre-configured Austin metro AOI.

    Convenience factory — equivalent to
    ``AOIBuilder(AUSTIN_CENTER_LAT, AUSTIN_CENTER_LON, AUSTIN_BUFFER_KM).build()``.
    """
    return AOIBuilder().build()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _frange(start: float, stop: float, step: float) -> list[float]:
    """Generate a float range from *start* to *stop* with *step*."""
    values: list[float] = []
    v = start
    while v < stop:
        values.append(v)
        v += step
    return values
