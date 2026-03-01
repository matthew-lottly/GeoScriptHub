"""
acquisition.py
==============
Multi-sensor imagery acquisition from Microsoft Planetary Computer.

Queries three STAC collections and returns lazy dask-backed xarray
DataArrays for each:

* **Landsat 8 & 9 Collection 2 Level-2** (``landsat-c2-l2``)
  — 30 m optical / thermal, surface reflectance.
* **Sentinel-2 Level-2A** (``sentinel-2-l2a``)
  — 10–20 m optical, surface reflectance.
* **NAIP** (``naip``)
  — ~0.6–1 m aerial imagery (RGBIR), typically summer acquisitions.

All imagery is fetched in the AOI bounding box with cloud-cover
filtering (≤15 % for optical sensors).  No full-scene downloads —
stackstac streams only the required spatial window via COG byte-range
reads.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pystac_client
    import planetary_computer
    import stackstac
    import xarray as xr
    import rioxarray  # noqa: F401 — activates .rio accessor
except ImportError as exc:
    raise ImportError(
        f"Missing dependency: {exc}. "
        "Install with:  pip install pystac-client planetary-computer stackstac rioxarray"
    ) from exc

from .aoi import AOIResult

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.acquisition")

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ---------------------------------------------------------------------------
# Sensor band configurations
# ---------------------------------------------------------------------------

# Landsat C2L2 bands relevant for water detection (SR = Surface Reflectance)
LANDSAT_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
LANDSAT_RESOLUTION = 30  # metres

# Sentinel-2 L2A bands relevant for water detection
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
S2_RESOLUTION = 20  # metres — we use 20 m to capture SWIR bands natively

# NAIP — 4-band (R, G, B, NIR) at ~1 m
NAIP_BANDS = ["image"]  # NAIP stores all 4 bands in a single asset
NAIP_RESOLUTION = 1  # metre


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SensorStack:
    """Aggregated multi-sensor imagery stacks.

    Attributes:
        landsat: Landsat 8/9 C2L2 xarray DataArray — dims (time, band, y, x).
        sentinel2: Sentinel-2 L2A xarray DataArray — dims (time, band, y, x).
        naip: NAIP xarray DataArray — dims (time, band, y, x).
        landsat_count: Number of Landsat scenes retrieved.
        sentinel2_count: Number of Sentinel-2 scenes retrieved.
        naip_count: Number of NAIP tiles retrieved.
        aoi: The AOI used for acquisition.
    """

    landsat: xr.DataArray
    sentinel2: xr.DataArray
    naip: xr.DataArray
    landsat_count: int
    sentinel2_count: int
    naip_count: int
    aoi: AOIResult

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<SensorStack  Landsat={self.landsat_count}  "
            f"Sentinel-2={self.sentinel2_count}  NAIP={self.naip_count}>"
        )


# ---------------------------------------------------------------------------
# Acquisition engine
# ---------------------------------------------------------------------------

class MultiSensorAcquisition:
    """Acquires Landsat, Sentinel-2, and NAIP imagery from Planetary Computer.

    Parameters
    ----------
    aoi:
        Resolved AOI from ``AOIBuilder``.
    start_date:
        ISO-8601 start date for temporal query, e.g. ``"2015-01-01"``.
    end_date:
        ISO-8601 end date, e.g. ``"2025-12-31"``.
    max_cloud_cover:
        Maximum scene-level cloud cover percentage (0–100).
    chunk_size:
        Dask chunk size (pixels) for x/y dims.

    Notes
    -----
    We fetch as many cloud-free scenes as Planetary Computer holds
    to maximise temporal depth for frequency analysis.
    Landsat archive starts 2013 (L8), Sentinel-2 starts 2017, NAIP
    varies by state (Louisiana has ~biennial coverage since 2004).
    """

    def __init__(
        self,
        aoi: AOIResult,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
        max_cloud_cover: int = 15,
        chunk_size: int = 1024,
    ) -> None:
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        self.max_cloud_cover = max_cloud_cover
        self.chunk_size = chunk_size
        self._datetime_range = f"{start_date}/{end_date}"
        self._bbox = (
            float(aoi.bbox_wgs84[0]),
            float(aoi.bbox_wgs84[1]),
            float(aoi.bbox_wgs84[2]),
            float(aoi.bbox_wgs84[3]),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> SensorStack:
        """Fetch all three sensor stacks and return a ``SensorStack``.

        Returns:
            SensorStack containing lazy dask-backed DataArrays for
            Landsat, Sentinel-2, and NAIP.
        """
        logger.info("Opening Planetary Computer STAC catalogue …")
        catalog = pystac_client.Client.open(
            PLANETARY_COMPUTER_URL,
            modifier=planetary_computer.sign_inplace,
        )

        landsat, n_ls = self._fetch_landsat(catalog)
        sentinel2, n_s2 = self._fetch_sentinel2(catalog)
        naip, n_naip = self._fetch_naip(catalog)

        logger.info(
            "Acquisition complete — Landsat: %d, Sentinel-2: %d, NAIP: %d scenes",
            n_ls, n_s2, n_naip,
        )

        return SensorStack(
            landsat=landsat,
            sentinel2=sentinel2,
            naip=naip,
            landsat_count=n_ls,
            sentinel2_count=n_s2,
            naip_count=n_naip,
            aoi=self.aoi,
        )

    # ------------------------------------------------------------------
    # Private — per-sensor fetchers
    # ------------------------------------------------------------------

    def _fetch_landsat(
        self, catalog: pystac_client.Client
    ) -> tuple[xr.DataArray, int]:
        """Query Landsat Collection 2 Level-2 SR scenes."""
        logger.info("Querying Landsat C2L2 (cloud ≤%d%%) …", self.max_cloud_cover)

        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=self._bbox,
            datetime=self._datetime_range,
            query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
        )
        items = list(search.items())
        n_items = len(items)
        logger.info("  → Found %d Landsat scenes", n_items)

        if n_items == 0:
            warnings.warn("No Landsat scenes found — returning empty DataArray")
            return self._empty_array(), 0

        # Sign STAC items for direct COG access
        signed = [planetary_computer.sign(item) for item in items]

        stack = stackstac.stack(
            signed,
            assets=LANDSAT_BANDS,
            resolution=LANDSAT_RESOLUTION,
            epsg=int(self.aoi.target_crs.split(":")[1]),
            bounds=self.aoi.bbox_utm,
            chunksize=self.chunk_size,
            dtype=np.dtype("float64"),
            fill_value=np.nan,
        )

        return stack, n_items

    def _fetch_sentinel2(
        self, catalog: pystac_client.Client
    ) -> tuple[xr.DataArray, int]:
        """Query Sentinel-2 L2A scenes."""
        logger.info("Querying Sentinel-2 L2A (cloud ≤%d%%) …", self.max_cloud_cover)

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=self._bbox,
            datetime=self._datetime_range,
            query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
        )
        items = list(search.items())
        n_items = len(items)
        logger.info("  → Found %d Sentinel-2 scenes", n_items)

        if n_items == 0:
            warnings.warn("No Sentinel-2 scenes found — returning empty DataArray")
            return self._empty_array(), 0

        signed = [planetary_computer.sign(item) for item in items]

        stack = stackstac.stack(
            signed,
            assets=S2_BANDS,
            resolution=S2_RESOLUTION,
            epsg=int(self.aoi.target_crs.split(":")[1]),
            bounds=self.aoi.bbox_utm,
            chunksize=self.chunk_size,
            dtype=np.dtype("float64"),
            fill_value=np.nan,
        )

        return stack, n_items

    def _fetch_naip(
        self, catalog: pystac_client.Client
    ) -> tuple[xr.DataArray, int]:
        """Query NAIP aerial imagery tiles.

        NAIP has no ``eo:cloud_cover`` field — it is acquired under
        clear-sky conditions by contract.  We fetch all available tiles
        within the time window.
        """
        logger.info("Querying NAIP imagery …")

        search = catalog.search(
            collections=["naip"],
            bbox=self._bbox,
            datetime=self._datetime_range,
        )
        items = list(search.items())
        n_items = len(items)
        logger.info("  → Found %d NAIP tiles", n_items)

        if n_items == 0:
            warnings.warn("No NAIP scenes found — returning empty DataArray")
            return self._empty_array(), 0

        signed = [planetary_computer.sign(item) for item in items]

        stack = stackstac.stack(
            signed,
            assets=NAIP_BANDS,
            resolution=NAIP_RESOLUTION,
            epsg=int(self.aoi.target_crs.split(":")[1]),
            bounds=self.aoi.bbox_utm,
            chunksize=self.chunk_size,
            dtype=np.dtype("float64"),
            fill_value=np.nan,
        )

        return stack, n_items

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_array() -> xr.DataArray:
        """Return a minimal empty DataArray as placeholder."""
        return xr.DataArray(
            data=np.empty((0, 0, 0, 0), dtype="float32"),
            dims=("time", "band", "y", "x"),
        )
