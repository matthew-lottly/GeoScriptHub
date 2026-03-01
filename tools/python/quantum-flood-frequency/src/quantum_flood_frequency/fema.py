"""
fema.py
=======
FEMA National Flood Hazard Layer (NFHL) integration.

Downloads FEMA flood zone boundaries from the FEMA Map Service REST API
and reprojects them to match the study-area grid for side-by-side
comparison with the computed flood frequency surface.

Flood Zone Categories
---------------------
We classify every polygon into one of three regulatory categories
that the user cares about:

    **Floodway**
        Regulatory floodway — the channel plus adjacent areas that
        must be kept free of encroachment.  Identified by
        ``ZONE_SUBTY`` containing ``'FLOODWAY'``.

    **100-year (1 % annual chance / SFHA)**
        Special Flood Hazard Area.  Zone codes A, AE, AH, AO, AR,
        A99, V, VE — *excluding* the floodway subset.

    **500-year (0.2 % annual chance)**
        Moderate-risk area.  Zone X with ``ZONE_SUBTY`` containing
        ``'0.2 PCT ANNUAL CHANCE'``.

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

# FEMA NFHL MapServer — Flood Hazard Zones layer (S_Fld_Haz_Ar)
NFHL_FLOOD_ZONES_URL = (
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
)

# Maximum features per request (ESRI REST limit)
_MAX_RECORD_COUNT = 1000

# Zone classification mapping
HIGH_RISK_ZONES = {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE"}
MODERATE_RISK_ZONES = {"X"}  # shaded variant (0.2% annual chance)
MINIMAL_RISK_ZONES = {"X"}   # unshaded variant
UNDETERMINED_ZONES = {"D"}

# Flood category constants
CATEGORY_FLOODWAY = "floodway"
CATEGORY_100_YEAR = "100-year"
CATEGORY_500_YEAR = "500-year"
CATEGORY_OTHER    = "other"

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

# Category-level colours (for FEMA comparison maps)
FEMA_CATEGORY_COLORS = {
    CATEGORY_FLOODWAY: (0.8, 0.0, 0.0, 0.55),   # red
    CATEGORY_100_YEAR: (0.0, 0.2, 0.9, 0.45),    # blue
    CATEGORY_500_YEAR: (1.0, 0.6, 0.0, 0.35),    # orange
    CATEGORY_OTHER:    (0.5, 0.5, 0.5, 0.25),    # grey
}


class FEMAFloodZones:
    """Download and process FEMA flood zone polygons for the study area.

    Classifies every downloaded polygon into one of four categories:

    * ``floodway``  — regulatory floodway
    * ``100-year``  — 1 % annual-chance SFHA (excluding floodway)
    * ``500-year``  — 0.2 % annual-chance (shaded Zone X)
    * ``other``     — minimal risk, undetermined, etc.

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

    # --- Accessors for the three user-requested categories ---

    @property
    def floodway(self) -> gpd.GeoDataFrame:
        """Regulatory floodway polygons."""
        return self._filter_category(CATEGORY_FLOODWAY)

    @property
    def zones_100yr(self) -> gpd.GeoDataFrame:
        """100-year (1 % annual chance / SFHA) polygons, *excluding* floodway."""
        return self._filter_category(CATEGORY_100_YEAR)

    @property
    def zones_500yr(self) -> gpd.GeoDataFrame:
        """500-year (0.2 % annual chance) polygons."""
        return self._filter_category(CATEGORY_500_YEAR)

    def _filter_category(self, category: str) -> gpd.GeoDataFrame:
        if self._gdf is None or self._gdf.empty or "flood_category" not in self._gdf.columns:
            return gpd.GeoDataFrame()
        return self._gdf[self._gdf["flood_category"] == category].copy()

    # ------------------------------------------------------------------
    # Fetch — paginated to handle AOIs that exceed the 1 000-record limit
    # ------------------------------------------------------------------

    def fetch(self) -> gpd.GeoDataFrame:
        """Query the FEMA NFHL REST API for flood zones in the AOI.

        Uses offset-based pagination so that large AOIs retrieve all
        features even when the server imposes a 1 000–2 000 record cap.

        Returns:
            GeoDataFrame with FEMA flood zone polygons clipped to the AOI,
            reprojected to the study-area CRS, with ``flood_category``
            and ``risk_level`` columns added.

        Raises:
            RuntimeError: If the FEMA API request fails critically.
        """
        logger.info("Fetching FEMA flood zones from NFHL MapServer …")

        west, south, east, north = self.aoi.bbox_wgs84
        bbox_str = f"{west},{south},{east},{north}"

        all_features: list[dict] = []
        offset = 0

        while True:
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
                "resultOffset": str(offset),
                "resultRecordCount": str(_MAX_RECORD_COUNT),
            }

            try:
                resp = requests.get(
                    NFHL_FLOOD_ZONES_URL, params=params, timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logger.warning(
                    "FEMA API request failed: %s — FEMA overlay will be unavailable", exc,
                )
                self._gdf = gpd.GeoDataFrame()
                return self._gdf

            features = data.get("features", [])
            if not features:
                break

            all_features.extend(features)
            logger.debug(
                "FEMA: page at offset %d returned %d features", offset, len(features),
            )

            # If fewer than max returned, we've got everything
            if len(features) < _MAX_RECORD_COUNT:
                break

            offset += len(features)

        if not all_features:
            logger.warning("No FEMA flood zones found in AOI — overlay will be empty")
            self._gdf = gpd.GeoDataFrame()
            return self._gdf

        gdf = gpd.GeoDataFrame.from_features(all_features, crs=WGS84)

        # Clip to AOI polygon
        aoi_poly = gpd.GeoDataFrame(
            geometry=[self.aoi.geometry_wgs84], crs=WGS84,
        )
        gdf = gpd.clip(gdf, aoi_poly)

        # Reproject to study-area CRS
        gdf = gdf.to_crs(self.aoi.target_crs)

        # Add risk classification + flood_category columns
        if "FLD_ZONE" in gdf.columns:
            gdf["risk_level"] = gdf["FLD_ZONE"].map(self._classify_risk)
            gdf["flood_category"] = gdf.apply(self._categorise_flood, axis=1)
        else:
            gdf["risk_level"] = "unknown"
            gdf["flood_category"] = CATEGORY_OTHER

        self._gdf = gdf

        # Log per-category summary
        cats = gdf["flood_category"].value_counts()
        logger.info(
            "FEMA: Retrieved %d flood zone polygons  "
            "[floodway=%d, 100-yr=%d, 500-yr=%d, other=%d]",
            len(gdf),
            cats.get(CATEGORY_FLOODWAY, 0),
            cats.get(CATEGORY_100_YEAR, 0),
            cats.get(CATEGORY_500_YEAR, 0),
            cats.get(CATEGORY_OTHER, 0),
        )

        return gdf

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_geojson(self, output_path: Path) -> Path:
        """Save *all* FEMA flood zones as GeoJSON.

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

    def save_shapefiles(self, output_dir: Path) -> dict[str, Path]:
        """Save floodway, 100-year, and 500-year as individual Shapefiles.

        Creates an ``fema/`` subdirectory inside *output_dir* with:

        * ``fema_floodway.shp``
        * ``fema_100yr.shp``
        * ``fema_500yr.shp``

        Returns:
            Dict mapping category name → shapefile path.
        """
        if self._gdf is None or self._gdf.empty:
            logger.warning("No FEMA data to save as shapefiles")
            return {}

        fema_dir = Path(output_dir) / "fema"
        fema_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[str, Path] = {}

        category_files = {
            CATEGORY_FLOODWAY: "fema_floodway",
            CATEGORY_100_YEAR: "fema_100yr",
            CATEGORY_500_YEAR: "fema_500yr",
        }

        for category, filename in category_files.items():
            subset = self._filter_category(category)
            shp_path = fema_dir / f"{filename}.shp"

            if subset.empty:
                logger.info("FEMA %s: 0 polygons — shapefile skipped", category)
                continue

            # Shapefiles use the study-area CRS (projected)
            subset.to_file(shp_path, driver="ESRI Shapefile")
            saved[category] = shp_path
            logger.info(
                "FEMA %s: %d polygons → %s", category, len(subset), shp_path,
            )

        return saved

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

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
    def _categorise_flood(row: "gpd.pd.Series") -> str:  # type: ignore[name-defined]
        """Assign a regulatory flood category to a single feature row.

        Decision tree:
        1. If ZONE_SUBTY mentions "FLOODWAY" → floodway
        2. If FLD_ZONE is in the SFHA set (A, AE …) → 100-year
        3. If ZONE_SUBTY mentions "0.2 PCT ANNUAL CHANCE" → 500-year
        4. Otherwise → other
        """
        zone = str(row.get("FLD_ZONE", "")).strip().upper()
        subtype = str(row.get("ZONE_SUBTY", "")).strip().upper()

        if "FLOODWAY" in subtype:
            return CATEGORY_FLOODWAY
        if zone in HIGH_RISK_ZONES:
            return CATEGORY_100_YEAR
        if "0.2 PCT ANNUAL CHANCE" in subtype:
            return CATEGORY_500_YEAR
        return CATEGORY_OTHER

    @staticmethod
    def get_zone_color(zone: str) -> tuple[float, float, float, float]:
        """Return RGBA colour for a FEMA zone code."""
        return FEMA_COLORS.get(str(zone).strip().upper(), FEMA_DEFAULT_COLOR)

    @staticmethod
    def get_category_color(category: str) -> tuple[float, float, float, float]:
        """Return RGBA colour for a flood category."""
        return FEMA_CATEGORY_COLORS.get(category, FEMA_DEFAULT_COLOR)
