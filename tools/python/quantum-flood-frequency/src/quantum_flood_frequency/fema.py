"""
fema.py
=======
FEMA National Flood Hazard Layer (NFHL) integration.

Downloads FEMA flood zone boundaries from the FEMA Map Service REST API
and reprojects them to match the study-area grid for side-by-side
comparison with the computed flood frequency surface.

FEMA Flood Zone Codes (primary zones):
    A, AE, AH, AO, AR   — High-risk Special Flood Hazard Areas (1% annual)
    V, VE                — High-risk coastal (wave action ≥ 3 ft)
    X (shaded)           — Moderate risk (0.2% annual / 500-year)
    X (unshaded)         — Minimal risk
    D                    — Undetermined risk

Data source:
    FEMA National Flood Hazard Layer (NFHL) — ArcGIS REST MapServer
    https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer

This module queries the NFHL via the ArcGIS REST API ``/query`` endpoint,
returning GeoJSON features clipped to the AOI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import box, shape
from pyproj import Transformer

from .aoi import AOIResult, WGS84

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.fema")

# FEMA NFHL MapServer — Flood Hazard Zones layer
NFHL_FLOOD_ZONES_URL = (
    "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"
)

# Zone classification mapping
HIGH_RISK_ZONES = {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE"}
MODERATE_RISK_ZONES = {"X"}  # shaded variant (0.2% annual chance)
MINIMAL_RISK_ZONES = {"X"}   # unshaded variant
UNDETERMINED_ZONES = {"D"}

# Colour scheme for FEMA zones (RGBA for matplotlib)
FEMA_COLORS = {
    "A":  (0.0, 0.0, 0.8, 0.5),    # blue — high risk riverine
    "AE": (0.0, 0.2, 0.9, 0.5),
    "AH": (0.0, 0.3, 0.7, 0.5),
    "AO": (0.1, 0.1, 0.7, 0.5),
    "AR": (0.2, 0.0, 0.8, 0.5),
    "V":  (0.6, 0.0, 0.6, 0.5),    # purple — coastal high risk
    "VE": (0.7, 0.0, 0.5, 0.5),
    "X":  (1.0, 0.6, 0.0, 0.3),    # orange — moderate risk
    "D":  (0.5, 0.5, 0.5, 0.3),    # grey — undetermined
}
FEMA_DEFAULT_COLOR = (0.3, 0.3, 0.3, 0.3)


class FEMAFloodZones:
    """Download and process FEMA flood zone polygons for the study area.

    Parameters
    ----------
    aoi:
        Resolved AOI from ``AOIBuilder``.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(self, aoi: AOIResult, timeout: int = 60) -> None:
        self.aoi = aoi
        self.timeout = timeout
        self._gdf: Optional[gpd.GeoDataFrame] = None

    @property
    def zones(self) -> Optional[gpd.GeoDataFrame]:
        """Return the fetched FEMA flood zones GeoDataFrame, if available."""
        return self._gdf

    def fetch(self) -> gpd.GeoDataFrame:
        """Query the FEMA NFHL REST API for flood zones in the AOI.

        Returns:
            GeoDataFrame with FEMA flood zone polygons clipped to the AOI,
            reprojected to the study-area CRS.

        Raises:
            RuntimeError: If the FEMA API request fails.
        """
        logger.info("Fetching FEMA flood zones from NFHL MapServer …")

        west, south, east, north = self.aoi.bbox_wgs84
        bbox_str = f"{west},{south},{east},{north}"

        params = {
            "where": "1=1",
            "geometry": bbox_str,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }

        try:
            resp = requests.get(NFHL_FLOOD_ZONES_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("FEMA API request failed: %s — FEMA overlay will be unavailable", exc)
            self._gdf = gpd.GeoDataFrame()
            return self._gdf

        if "features" not in data or len(data["features"]) == 0:
            logger.warning("No FEMA flood zones found in AOI — overlay will be empty")
            self._gdf = gpd.GeoDataFrame()
            return self._gdf

        gdf = gpd.GeoDataFrame.from_features(data["features"], crs=WGS84)

        # Clip to AOI polygon
        aoi_poly = gpd.GeoDataFrame(
            geometry=[self.aoi.geometry_wgs84], crs=WGS84
        )
        gdf = gpd.clip(gdf, aoi_poly)

        # Reproject to study-area CRS
        gdf = gdf.to_crs(self.aoi.target_crs)

        # Add risk classification column
        if "FLD_ZONE" in gdf.columns:
            gdf["risk_level"] = gdf["FLD_ZONE"].map(self._classify_risk)
        else:
            gdf["risk_level"] = "unknown"

        self._gdf = gdf
        logger.info("FEMA: Retrieved %d flood zone polygons", len(gdf))

        return gdf

    def save_geojson(self, output_path: Path) -> Path:
        """Save FEMA flood zones as GeoJSON.

        Args:
            output_path: Output file path.

        Returns:
            Path to the saved file.
        """
        if self._gdf is None or self._gdf.empty:
            logger.warning("No FEMA data to save")
            return output_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in WGS-84 for GeoJSON compatibility
        gdf_wgs84 = self._gdf.to_crs(WGS84)
        gdf_wgs84.to_file(output_path, driver="GeoJSON")

        logger.info("FEMA zones saved → %s", output_path)
        return output_path

    @staticmethod
    def _classify_risk(zone: str) -> str:
        """Map a FEMA zone code to a risk classification string."""
        zone = str(zone).strip().upper()
        if zone in HIGH_RISK_ZONES:
            return "high"
        if zone == "X":
            return "moderate"
        if zone == "D":
            return "undetermined"
        return "minimal"

    @staticmethod
    def get_zone_color(zone: str) -> tuple[float, float, float, float]:
        """Return RGBA colour for a FEMA zone code."""
        return FEMA_COLORS.get(str(zone).strip().upper(), FEMA_DEFAULT_COLOR)
