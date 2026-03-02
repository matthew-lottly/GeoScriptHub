"""Multi-epoch STAC data acquisition.

Fetches Landsat 5/7/8/9, Sentinel-2, Sentinel-1 SAR, NAIP, and
Copernicus DEM from Microsoft Planetary Computer for the given
AOI and temporal range.

v1.0 — Quantum Land-Cover Change Detector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr

from .aoi import AOIResult

logger = logging.getLogger("geoscripthub.landcover_change.acquisition")


@dataclass
class SensorStack:
    """Container for all fetched sensor data."""

    landsat: xr.DataArray = field(default_factory=lambda: xr.DataArray())
    sentinel2: xr.DataArray = field(default_factory=lambda: xr.DataArray())
    sentinel1: xr.DataArray = field(default_factory=lambda: xr.DataArray())
    naip: xr.DataArray = field(default_factory=lambda: xr.DataArray())
    dem: xr.DataArray = field(default_factory=lambda: xr.DataArray())
    naip_items: list = field(default_factory=list)
    landsat_count: int = 0
    sentinel2_count: int = 0
    sentinel1_count: int = 0
    naip_count: int = 0


class MultiEpochAcquisition:
    """Fetch multi-sensor imagery from Planetary Computer STAC.

    Unlike the flood-frequency tool which queries a single date range,
    this fetches across the full 1990–present timeline, intelligently
    handling sensor availability per era.

    Parameters
    ----------
    aoi:
        Resolved AOI for spatial querying.
    start_date:
        Earliest date to query (e.g. '1990-01-01').
    end_date:
        Latest date to query (e.g. '2026-12-31').
    max_cloud:
        Maximum cloud cover percentage for optical scenes.
    """

    def __init__(
        self,
        aoi: AOIResult,
        start_date: str = "1990-01-01",
        end_date: str = "2026-12-31",
        max_cloud: int = 15,
    ) -> None:
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        self.max_cloud = max_cloud

    def fetch_all(
        self,
        *,
        use_sar: bool = True,
        use_terrain: bool = True,
    ) -> SensorStack:
        """Query all sensor collections and return stacked data.

        Returns
        -------
        SensorStack with imagery for all available sensors.
        """
        import pystac_client
        import planetary_computer
        import stackstac

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        bbox = list(self.aoi.bbox_wgs84)
        utm_bounds = list(self.aoi.bbox_utm)
        crs = self.aoi.target_crs

        stack = SensorStack()

        # ── Landsat C2L2 (all missions — TM, ETM+, OLI) ─────────
        logger.info("Querying Landsat C2L2 (cloud ≤%d%%) …", self.max_cloud)
        try:
            ls_search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime=f"{self.start_date}/{self.end_date}",
                query={"eo:cloud_cover": {"lt": self.max_cloud}},
            )
            ls_items = list(ls_search.items())
            stack.landsat_count = len(ls_items)
            logger.info("→ Found %d Landsat scenes", stack.landsat_count)

            if ls_items:
                signed = [planetary_computer.sign(i) for i in ls_items]
                stack.landsat = stackstac.stack(
                    signed,
                    assets=["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"],
                    epsg=int(crs.split(":")[1]),
                    bounds=utm_bounds,
                    chunksize=1024,
                )
        except Exception as exc:
            logger.warning("Landsat query failed: %s", exc)

        # ── Sentinel-2 L2A (2015+) ──────────────────────────────
        s2_start = max(self.start_date, "2015-07-01")
        if s2_start < self.end_date:
            logger.info("Querying Sentinel-2 L2A (cloud ≤%d%%) …", self.max_cloud)
            try:
                s2_search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=f"{s2_start}/{self.end_date}",
                    query={"eo:cloud_cover": {"lt": self.max_cloud}},
                )
                s2_items = list(s2_search.items())
                stack.sentinel2_count = len(s2_items)
                logger.info("→ Found %d Sentinel-2 scenes", stack.sentinel2_count)

                if s2_items:
                    signed = [planetary_computer.sign(i) for i in s2_items]
                    stack.sentinel2 = stackstac.stack(
                        signed,
                        assets=["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
                        epsg=int(crs.split(":")[1]),
                        bounds=utm_bounds,
                        chunksize=1024,
                    )
            except Exception as exc:
                logger.warning("Sentinel-2 query failed: %s", exc)

        # ── Sentinel-1 SAR (2014+) ──────────────────────────────
        if use_sar:
            sar_start = max(self.start_date, "2014-10-01")
            if sar_start < self.end_date:
                logger.info("Querying Sentinel-1 GRD SAR …")
                try:
                    s1_search = catalog.search(
                        collections=["sentinel-1-grd"],
                        bbox=bbox,
                        datetime=f"{sar_start}/{self.end_date}",
                    )
                    s1_items = list(s1_search.items())
                    stack.sentinel1_count = len(s1_items)
                    logger.info("→ Found %d SAR scenes", stack.sentinel1_count)

                    if s1_items:
                        signed = [planetary_computer.sign(i) for i in s1_items]
                        stack.sentinel1 = stackstac.stack(
                            signed,
                            assets=["vv", "vh"],
                            epsg=int(crs.split(":")[1]),
                            bounds=utm_bounds,
                            chunksize=1024,
                        )
                except Exception as exc:
                    logger.warning("Sentinel-1 query failed: %s", exc)

        # ── NAIP (~2003+) ────────────────────────────────────────
        naip_start = max(self.start_date, "2003-01-01")
        if naip_start < self.end_date:
            logger.info("Querying NAIP aerial imagery …")
            try:
                naip_search = catalog.search(
                    collections=["naip"],
                    bbox=bbox,
                    datetime=f"{naip_start}/{self.end_date}",
                )
                naip_items = list(naip_search.items())
                stack.naip_count = len(naip_items)
                logger.info("→ Found %d NAIP scenes", stack.naip_count)

                if naip_items:
                    stack.naip_items = [
                        planetary_computer.sign(i) for i in naip_items
                    ]
                    # Placeholder DataArray — actual read via rasterio
                    stack.naip = xr.DataArray(
                        np.empty((0,)),
                        attrs={"_placeholder": True},
                    )
            except Exception as exc:
                logger.warning("NAIP query failed: %s", exc)

        # ── Copernicus DEM 30 m (static) ─────────────────────────
        if use_terrain:
            logger.info("Querying Copernicus DEM GLO-30 …")
            try:
                dem_search = catalog.search(
                    collections=["cop-dem-glo-30"],
                    bbox=bbox,
                )
                dem_items = list(dem_search.items())
                logger.info("→ Found %d DEM tiles", len(dem_items))

                if dem_items:
                    signed = [planetary_computer.sign(i) for i in dem_items]
                    stack.dem = stackstac.stack(
                        signed,
                        assets=["data"],
                        epsg=int(crs.split(":")[1]),
                        bounds=utm_bounds,
                        chunksize=1024,
                    )
            except Exception as exc:
                logger.warning("DEM query failed: %s", exc)

        logger.info(
            "Acquisition complete — Landsat: %d, S2: %d, S1: %d, NAIP: %d",
            stack.landsat_count, stack.sentinel2_count,
            stack.sentinel1_count, stack.naip_count,
        )
        return stack
