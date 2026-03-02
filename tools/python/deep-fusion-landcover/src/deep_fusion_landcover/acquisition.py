"""acquisition.py — Multi-sensor STAC data acquisition.

Fetches Landsat 4/5/7/8/9, Sentinel-1/2, NAIP, Copernicus DEM, and
3DEP elevation tiles from Microsoft Planetary Computer and USGS.

All imagery is returned as lazy ``xarray.Dataset`` objects backed by
dask arrays — no large data is downloaded until ``.compute()`` is
called explicitly, giving the rest of the pipeline full control over
memory usage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
from rasterio.enums import Resampling as RioResampling

from .aoi import AOIResult
from .constants import (
    AUSTIN_UTM_CRS,
    LANDSAT_BANDS,
    PC_COLLECTIONS,
    PC_STAC_URL,
    S2_BANDS,
    S2_CLOUD_CLASSES,
)

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.acquisition")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SensorStack:
    """Raw, unprocessed imagery stacked for one sensor + time period.

    Attributes
    ----------
    sensor:         Sensor identifier string (e.g. ``"landsat"``, ``"sentinel2"``).
    collection:     PC STAC collection ID.
    dataset:        Lazy xarray Dataset with band dimension + time coordinate.
    item_count:     Number of STAC items found.
    years_covered:  Sorted list of years covered by the acquired scenes.
    bbox_wgs84:     Query bounding box (minx, miny, maxx, maxy).
    """

    sensor: str
    collection: str
    dataset: xr.Dataset
    item_count: int
    years_covered: list[int]
    bbox_wgs84: tuple[float, float, float, float]


@dataclass
class MultiSensorStack:
    """All sensor stacks for a given acquisition run.

    Attributes
    ----------
    landsat:    Landsat C2 L2 stack (all generations, 30 m).
    sentinel2:  Sentinel-2 L2A stack (10/20 m).
    sentinel1:  Sentinel-1 RTC SAR stack (10 m).
    naip:       NAIP RGBNIR stack (0.6 m).
    dem:        Copernicus 30 m DEM (static).
    """

    landsat: Optional[SensorStack] = None
    sentinel2: Optional[SensorStack] = None
    sentinel1: Optional[SensorStack] = None
    naip: Optional[SensorStack] = None
    dem: Optional[SensorStack] = None


# ── Main acquisition class ────────────────────────────────────────────────────

class MultiSensorAcquisition:
    """Fetch all sensors needed by the deep-fusion pipeline.

    Parameters
    ----------
    aoi:            Study-area AOI produced by :class:`~.aoi.AOIBuilder`.
    start_year:     First year to acquire (inclusive).
    end_year:       Last year to acquire (inclusive).
    max_cloud:      Maximum scene-level cloud cover percentage (0–100).
    resolution_m:   Target pixel resolution for resampling (metres).
    use_sar:        Whether to fetch Sentinel-1 SAR.
    use_naip:       Whether to fetch NAIP imagery.
    use_dem:        Whether to fetch the Copernicus DEM.

    Examples
    --------
    >>> aoi = AOIBuilder().build()
    >>> acq = MultiSensorAcquisition(aoi, start_year=2020, end_year=2022)
    >>> stack = acq.fetch_all()
    """

    def __init__(
        self,
        aoi: AOIResult,
        start_year: int = 1990,
        end_year: int = 2025,
        max_cloud: float = 15.0,
        resolution_m: int = 10,
        use_sar: bool = True,
        use_naip: bool = True,
        use_dem: bool = True,
    ) -> None:
        self.aoi = aoi
        self.start_year = start_year
        self.end_year = end_year
        self.max_cloud = max_cloud
        self.resolution_m = resolution_m
        self.use_sar = use_sar
        self.use_naip = use_naip
        self.use_dem = use_dem

        self._catalog: Optional[pystac_client.Client] = None

    # ── Catalog connection ─────────────────────────────────────────────────────

    def _get_catalog(self) -> pystac_client.Client:
        """Return a signed Planetary Computer STAC client (cached)."""
        if self._catalog is None:
            self._catalog = pystac_client.Client.open(
                PC_STAC_URL,
                modifier=pc.sign_inplace,
            )
        return self._catalog

    # ── Public interface ───────────────────────────────────────────────────────

    def fetch_all(self) -> MultiSensorStack:
        """Fetch all enabled sensors and return a :class:`MultiSensorStack`.

        Returns
        -------
        MultiSensorStack
            Lazy xarray stacks, ready for preprocessing.
        """
        logger.info("Fetching Landsat archive %d–%d …", self.start_year, self.end_year)
        landsat = self.fetch_landsat()

        sentinel2 = None
        if self.end_year >= 2015:
            s2_start = max(self.start_year, 2015)
            logger.info("Fetching Sentinel-2 %d–%d …", s2_start, self.end_year)
            sentinel2 = self.fetch_sentinel2(start_year=s2_start)

        sentinel1 = None
        if self.use_sar and self.end_year >= 2014:
            s1_start = max(self.start_year, 2014)
            logger.info("Fetching Sentinel-1 SAR %d–%d …", s1_start, self.end_year)
            sentinel1 = self.fetch_sentinel1(start_year=s1_start)

        naip = None
        if self.use_naip and self.end_year >= 2003:
            naip_start = max(self.start_year, 2003)
            logger.info("Fetching NAIP %d–%d …", naip_start, self.end_year)
            naip = self.fetch_naip(start_year=naip_start)

        dem = None
        if self.use_dem:
            logger.info("Fetching Copernicus DEM …")
            dem = self.fetch_dem()

        return MultiSensorStack(
            landsat=landsat,
            sentinel2=sentinel2,
            sentinel1=sentinel1,
            naip=naip,
            dem=dem,
        )

    # ── Per-sensor fetch methods ───────────────────────────────────────────────

    def fetch_landsat(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> SensorStack:
        """Fetch Landsat Collection 2 Level-2 surface reflectance scenes.

        Covers Landsat 4/5 TM (1984–2013), Landsat 7 ETM+ (1999–2022),
        Landsat 8 OLI (2013–2021), and Landsat 9 OLI-2 (2021–present).

        Parameters
        ----------
        start_year:  Override for the instance ``start_year``.
        end_year:    Override for the instance ``end_year``.

        Returns
        -------
        SensorStack
            Lazy stacked dataset with bands named by spectral role
            (blue, green, red, nir, swir1, swir2, tir, QA_PIXEL).
        """
        sy = start_year or self.start_year
        ey = end_year or self.end_year
        date_range = f"{sy}-01-01/{ey}-12-31"

        catalog = self._get_catalog()
        search = catalog.search(
            collections=[PC_COLLECTIONS["landsat"]],
            bbox=self.aoi.stac_bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": self.max_cloud}},
        )
        items = list(search.items())
        logger.debug("Landsat: %d scenes found for %s–%s", len(items), sy, ey)

        if not items:
            logger.warning("No Landsat scenes found for the specified parameters.")
            # Return empty dataset
            empty_ds: xr.Dataset = xr.Dataset()
            return SensorStack(
                sensor="landsat",
                collection=PC_COLLECTIONS["landsat"],
                dataset=empty_ds,
                item_count=0,
                years_covered=[],
                bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            )

        # Determine common bands across all Landsat generations
        common_assets = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6",
                         "SR_B7", "QA_PIXEL"]
        # For Landsat 8/9, blue = B2; for 4/5/7, blue = B1 — handled in preprocessing

        stack_da = stackstac.stack(
            items,
            assets=common_assets,
            bounds=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            resolution=30,
            dtype=np.dtype("float32"),  # type: ignore[arg-type]
            fill_value=np.nan,
            resampling=RioResampling.nearest,
        )

        ds = stack_da.to_dataset(dim="band")
        years = sorted({int(str(t)[:4]) for t in stack_da.time.values})

        return SensorStack(
            sensor="landsat",
            collection=PC_COLLECTIONS["landsat"],
            dataset=ds,
            item_count=len(items),
            years_covered=years,
            bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
        )

    def fetch_sentinel2(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> SensorStack:
        """Fetch Sentinel-2 L2A scenes from Planetary Computer.

        Returns 10 spectral bands + SCL at the configured resolution.

        Parameters
        ----------
        start_year:  Override for first acquisition year (min: 2015).
        end_year:    Override for last acquisition year.

        Returns
        -------
        SensorStack
            Lazy xarray Dataset with S2 band names.
        """
        sy = max(start_year or self.start_year, 2015)
        ey = end_year or self.end_year
        date_range = f"{sy}-01-01/{ey}-12-31"

        catalog = self._get_catalog()
        search = catalog.search(
            collections=[PC_COLLECTIONS["sentinel2"]],
            bbox=self.aoi.stac_bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": self.max_cloud}},
        )
        items = list(search.items())
        logger.debug("Sentinel-2: %d scenes found for %s–%s", len(items), sy, ey)

        if not items:
            empty_ds = xr.Dataset()
            return SensorStack(
                sensor="sentinel2",
                collection=PC_COLLECTIONS["sentinel2"],
                dataset=empty_ds,
                item_count=0,
                years_covered=[],
                bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            )

        s2_assets = [
            S2_BANDS["blue"], S2_BANDS["green"], S2_BANDS["red"],
            S2_BANDS["nir"], S2_BANDS["swir1"], S2_BANDS["swir2"],
            S2_BANDS["rededge1"], S2_BANDS["rededge2"],
            S2_BANDS["scl"],
        ]

        stack_da = stackstac.stack(
            items,
            assets=s2_assets,
            bounds=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            resolution=self.resolution_m,
            dtype=np.dtype("float32"),  # type: ignore[arg-type]
            fill_value=np.nan,
            resampling=RioResampling.bilinear,
        )

        ds = stack_da.to_dataset(dim="band")
        years = sorted({int(str(t)[:4]) for t in stack_da.time.values})

        return SensorStack(
            sensor="sentinel2",
            collection=PC_COLLECTIONS["sentinel2"],
            dataset=ds,
            item_count=len(items),
            years_covered=years,
            bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
        )

    def fetch_sentinel1(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> SensorStack:
        """Fetch Sentinel-1 RTC (Radiometric Terrain Corrected) SAR scenes.

        Returns gamma-0 VV and VH backscatter in linear power units.

        Parameters
        ----------
        start_year:  Override (min: 2014).
        end_year:    Override.

        Returns
        -------
        SensorStack
            Dataset with ``vv`` and ``vh`` variables.
        """
        sy = max(start_year or self.start_year, 2014)
        ey = end_year or self.end_year
        date_range = f"{sy}-01-01/{ey}-12-31"

        catalog = self._get_catalog()
        search = catalog.search(
            collections=[PC_COLLECTIONS["sentinel1"]],
            bbox=self.aoi.stac_bbox,
            datetime=date_range,
            query={"platform": {"in": ["sentinel-1a", "sentinel-1b"]}},
        )
        items = list(search.items())
        logger.debug("Sentinel-1: %d scenes found for %s–%s", len(items), sy, ey)

        if not items:
            empty_ds = xr.Dataset()
            return SensorStack(
                sensor="sentinel1",
                collection=PC_COLLECTIONS["sentinel1"],
                dataset=empty_ds,
                item_count=0,
                years_covered=[],
                bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            )

        stack_da = stackstac.stack(
            items,
            assets=["vv", "vh"],
            bounds=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            resolution=self.resolution_m,
            dtype=np.dtype("float32"),  # type: ignore[arg-type]
            fill_value=np.nan,
            resampling=RioResampling.bilinear,
        )

        ds = stack_da.to_dataset(dim="band")
        years = sorted({int(str(t)[:4]) for t in stack_da.time.values})

        return SensorStack(
            sensor="sentinel1",
            collection=PC_COLLECTIONS["sentinel1"],
            dataset=ds,
            item_count=len(items),
            years_covered=years,
            bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
        )

    def fetch_naip(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> SensorStack:
        """Fetch NAIP (National Agriculture Imagery Program) RGBNIR tiles.

        Tiles are at 0.6 m or 1 m native resolution; returned at native
        resolution for the 1 m present-day classification layer.

        Parameters
        ----------
        start_year:  Override (min: 2003).
        end_year:    Override.

        Returns
        -------
        SensorStack
            Dataset with ``red``, ``green``, ``blue``, ``nir`` bands.
        """
        sy = max(start_year or self.start_year, 2003)
        ey = end_year or self.end_year
        date_range = f"{sy}-01-01/{ey}-12-31"

        catalog = self._get_catalog()
        search = catalog.search(
            collections=[PC_COLLECTIONS["naip"]],
            bbox=self.aoi.stac_bbox,
            datetime=date_range,
        )
        items = list(search.items())
        logger.debug("NAIP: %d tiles found for %s–%s", len(items), sy, ey)

        if not items:
            empty_ds = xr.Dataset()
            return SensorStack(
                sensor="naip",
                collection=PC_COLLECTIONS["naip"],
                dataset=empty_ds,
                item_count=0,
                years_covered=[],
                bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            )

        stack_da = stackstac.stack(
            items,
            assets=["red", "green", "blue", "nir"],
            bounds=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            resolution=1.0,   # NAIP native ≈ 1 m
            dtype=np.dtype("uint8"),  # type: ignore[arg-type]
            fill_value=0,
            resampling=RioResampling.bilinear,
        )

        ds = stack_da.to_dataset(dim="band")
        years = sorted({int(str(t)[:4]) for t in stack_da.time.values})

        return SensorStack(
            sensor="naip",
            collection=PC_COLLECTIONS["naip"],
            dataset=ds,
            item_count=len(items),
            years_covered=years,
            bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
        )

    def fetch_dem(self) -> SensorStack:
        """Fetch the Copernicus GLO-30 Digital Elevation Model (static).

        The DEM is returned as a single-time Dataset; it contains only
        the ``data`` variable (elevation in metres above EGM2008 geoid).

        Returns
        -------
        SensorStack
            Dataset with ``elevation`` variable.
        """
        catalog = self._get_catalog()
        search = catalog.search(
            collections=[PC_COLLECTIONS["cop_dem"]],
            bbox=self.aoi.stac_bbox,
        )
        items = list(search.items())
        logger.debug("CopDEM: %d tiles found", len(items))

        if not items:
            empty_ds = xr.Dataset()
            return SensorStack(
                sensor="cop_dem",
                collection=PC_COLLECTIONS["cop_dem"],
                dataset=empty_ds,
                item_count=0,
                years_covered=[],
                bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            )

        stack_da = stackstac.stack(
            items,
            assets=["data"],
            bounds=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
            resolution=30,
            dtype=np.dtype("float32"),  # type: ignore[arg-type]
            fill_value=np.nan,
            resampling=RioResampling.bilinear,
        )

        # Median-of-time (should be single date for a static DEM)
        dem_da = stack_da.median(dim="time")
        ds = dem_da.to_dataset(name="elevation")

        return SensorStack(
            sensor="cop_dem",
            collection=PC_COLLECTIONS["cop_dem"],
            dataset=ds,
            item_count=len(items),
            years_covered=[],
            bbox_wgs84=tuple(self.aoi.stac_bbox),  # type: ignore[arg-type]
        )
