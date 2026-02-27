"""
aoi.py
======
Define the analysis area from multiple input types:
  - Shapefile or GeoPackage
  - Esri FileGDB layer
  - Bounding box [min_lon, min_lat, max_lon, max_lat]
  - Polygon as a list of (lon, lat) coordinate pairs
  - Place name via Nominatim geocoding + km buffer

All methods return a resolved ``AOIResult`` with attributes that the
fetcher and analysis modules expect.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AOIResult:
    """Holds the resolved analysis area in multiple representations."""

    # Dissolved geometry in WGS84 (EPSG:4326)
    gdf_wgs84: gpd.GeoDataFrame

    # Tight bounding box in WGS84 for STAC queries
    bbox_wgs84: Tuple[float, float, float, float]   # (min_lon, min_lat, max_lon, max_lat)

    # Best-fit projected CRS (UTM zone derived from AOI centroid)
    utm_crs: CRS

    # Dissolved geometry reprojected to UTM
    gdf_utm: gpd.GeoDataFrame

    # Bounding box in UTM (minx, miny, maxx, maxy) -- for raster clips
    bbox_utm: Tuple[float, float, float, float]

    # Human-readable description
    label: str

    def __repr__(self) -> str:  # noqa: D105
        w = self.bbox_wgs84[2] - self.bbox_wgs84[0]
        h = self.bbox_wgs84[3] - self.bbox_wgs84[1]
        return (
            f"<AOIResult '{self.label}' "
            f"bbox=({self.bbox_wgs84[0]:.4f},{self.bbox_wgs84[1]:.4f},"
            f"{self.bbox_wgs84[2]:.4f},{self.bbox_wgs84[3]:.4f}) "
            f"~{w:.3f}x{h:.3f} deg>"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    """Return the EPSG UTM CRS that covers *lon*, *lat*."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = "N" if lat >= 0 else "S"
    base = 32600 if hemisphere == "N" else 32700
    return CRS.from_epsg(base + zone)


def _build_result(gdf: gpd.GeoDataFrame, label: str) -> AOIResult:
    """Dissolve, reproject, and package a GeoDataFrame into an AOIResult."""
    # Ensure WGS84
    if gdf.crs is None:
        warnings.warn(
            "Input geometry has no CRS -- assuming WGS84 (EPSG:4326).",
            stacklevel=3,
        )
        gdf = gdf.set_crs("EPSG:4326")
    gdf_wgs84 = gdf.to_crs("EPSG:4326")

    # Dissolve to a single geometry
    dissolved = gpd.GeoDataFrame(
        geometry=[unary_union(gdf_wgs84.geometry)], crs="EPSG:4326"
    )
    bbox = dissolved.total_bounds  # (minx, miny, maxx, maxy)
    bbox_wgs84 = (bbox[0], bbox[1], bbox[2], bbox[3])

    # Derive UTM CRS from centroid
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    utm = _utm_crs_from_lonlat(cx, cy)
    gdf_utm = dissolved.to_crs(utm)
    _b = gdf_utm.total_bounds
    bbox_utm = (float(_b[0]), float(_b[1]), float(_b[2]), float(_b[3]))

    return AOIResult(
        gdf_wgs84=dissolved,
        bbox_wgs84=bbox_wgs84,
        utm_crs=utm,
        gdf_utm=gdf_utm,
        bbox_utm=bbox_utm,
        label=label,
    )


# ---------------------------------------------------------------------------
# Public builder class
# ---------------------------------------------------------------------------

class AOIBuilder:
    """Fluent builder that resolves an analysis AOI from various sources."""

    # ------------------------------------------------------------------
    # From vector files
    # ------------------------------------------------------------------

    @staticmethod
    def from_shapefile(path: str, layer: str | None = None) -> AOIResult:
        """Read the first (or named) layer from a shapefile or GeoPackage.

        Parameters
        ----------
        path:
            Path to the ``.shp`` or ``.gpkg`` file.
        layer:
            Layer name -- relevant for GeoPackage with multiple layers.

        Returns
        -------
        AOIResult
        """
        if layer:
            gdf = gpd.read_file(path, layer=layer)
        else:
            gdf = gpd.read_file(path)
        name = layer or path.split("/")[-1].split("\\")[-1]
        return _build_result(gdf, label=name)

    @staticmethod
    def from_fgdb(path: str, layer: str) -> AOIResult:
        """Read a layer from an Esri FileGDB.

        Requires pyogrio (bundled with the package; uses the OpenFileGDB
        GDAL driver which is statically compiled into the pyogrio wheel).

        Parameters
        ----------
        path:
            Path to the ``.gdb`` directory.
        layer:
            Feature class name inside the GDB.
        """
        gdf = gpd.read_file(path, layer=layer, engine="pyogrio")
        return _build_result(gdf, label=f"{layer} ({path.split('/')[-1]})")

    # ------------------------------------------------------------------
    # From coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def from_bbox(
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> AOIResult:
        """Define the AOI from a WGS84 bounding box.

        Parameters
        ----------
        min_lon, min_lat, max_lon, max_lat:
            Bounding box corners in decimal degrees.
        """
        geom = box(min_lon, min_lat, max_lon, max_lat)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        label = f"bbox({min_lon:.3f},{min_lat:.3f},{max_lon:.3f},{max_lat:.3f})"
        return _build_result(gdf, label=label)

    @staticmethod
    def from_polygon(coordinates: Sequence[Tuple[float, float]]) -> AOIResult:
        """Define the AOI from an explicit polygon.

        Parameters
        ----------
        coordinates:
            List of ``(lon, lat)`` pairs in WGS84.  The polygon is
            automatically closed if the first and last points differ.
        """
        coords = list(coordinates)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        geom = Polygon([(lon, lat) for lon, lat in coords])
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        return _build_result(gdf, label="User-defined polygon")

    @staticmethod
    def from_point(lon: float, lat: float, buffer_km: float = 5.0) -> AOIResult:
        """Create a circular AOI around a single point.

        Parameters
        ----------
        lon, lat:
            Centre point in WGS84 decimal degrees.
        buffer_km:
            Buffer radius in kilometres.
        """
        # Buffer in a local UTM CRS for metric accuracy, then back to WGS84
        utm = _utm_crs_from_lonlat(lon, lat)
        pt_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        pt_utm = pt_gdf.to_crs(utm)
        buffered_utm = pt_utm.buffer(buffer_km * 1000.0)
        gdf = gpd.GeoDataFrame(geometry=buffered_utm, crs=utm).to_crs("EPSG:4326")
        return _build_result(gdf, label=f"Point({lon:.4f},{lat:.4f})+{buffer_km}km")

    # ------------------------------------------------------------------
    # From place name (geocoding)
    # ------------------------------------------------------------------

    @staticmethod
    def from_place_name(
        name: str,
        buffer_km: float = 5.0,
        user_agent: str = "sub-canopy-detector/1.0",
    ) -> AOIResult:
        """Geocode a place name with OpenStreetMap Nominatim and buffer it.

        Parameters
        ----------
        name:
            Human-readable place name, e.g. ``"Peten, Guatemala"``.
        buffer_km:
            Buffer radius in kilometres around the geocoded centroid.
        user_agent:
            Nominatim requires a unique user-agent string (any value works).

        Returns
        -------
        AOIResult

        Raises
        ------
        ValueError
            If Nominatim cannot resolve *name*.
        """
        from geopy.geocoders import Nominatim

        geocoder = Nominatim(user_agent=user_agent)
        location = geocoder.geocode(name)  # type: ignore[union-attr]
        if location is None:
            raise ValueError(
                f"Could not geocode '{name}'.  "
                "Try a more specific name or use from_bbox() / from_polygon() instead."
            )

        lat: float = location.latitude  # type: ignore[union-attr]
        lon: float = location.longitude  # type: ignore[union-attr]
        print(f"Geocoded '{name}' -> ({lat:.4f}, {lon:.4f})")
        result = AOIBuilder.from_point(
            lon=lon,
            lat=lat,
            buffer_km=buffer_km,
        )
        # Override the label with the human-friendly name
        return AOIResult(
            gdf_wgs84=result.gdf_wgs84,
            bbox_wgs84=result.bbox_wgs84,
            utm_crs=result.utm_crs,
            gdf_utm=result.gdf_utm,
            bbox_utm=result.bbox_utm,
            label=f"{name} (+{buffer_km} km buffer)",
        )

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(aoi: AOIResult) -> None:
        """Print a brief summary of the resolved AOI to the console."""
        b = aoi.bbox_wgs84
        butm = aoi.bbox_utm
        w_km = (bqdm := aoi.gdf_utm.total_bounds)[2] - bqdm[0]
        h_km = bqdm[3] - bqdm[1]
        print("=== AOI Summary ======================================")
        print(f"  Label       : {aoi.label}")
        print(f"  WGS84 bbox  : W={b[0]:.4f}  S={b[1]:.4f}  E={b[2]:.4f}  N={b[3]:.4f}")
        print(f"  UTM CRS     : {aoi.utm_crs.to_epsg()}")
        print(f"  UTM extent  : {w_km/1000:.1f} km x {h_km/1000:.1f} km")
        print(f"  Area        : {aoi.gdf_utm.area.sum() / 1e6:.2f} km2")
        print("======================================================")
