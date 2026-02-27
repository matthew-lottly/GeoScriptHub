"""
fetcher.py
==========
Query and lazy-stream imagery from Microsoft Planetary Computer via the
STAC API.  No full-scene downloads -- stackstac reads only the spatial
window covered by the AOI.

Collections used
----------------
sentinel-1-rtc     -- Sentinel-1 GRD, RTC-processed to gamma0 linear power
sentinel-2-l2a     -- Sentinel-2 Level-2A surface reflectance (0-10000 scale)
cop-dem-glo-30     -- Copernicus GLO-30 Digital Elevation Model (30 m)

All imagery is returned as dask-backed xarray DataArrays clipped to the
AOI bounding box and reprojected to the AOI's UTM CRS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import warnings

try:
    import pystac_client
    import planetary_computer
    import stackstac
    import xarray as xr
    import rioxarray  # noqa: F401 -- activates the .rio accessor
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}.  "
        "Install with: pip install pystac-client planetary-computer stackstac rioxarray"
    ) from e

from .aoi import AOIResult

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ImageryData:
    """Holds all fetched imagery and DEM as lazy xarray DataArrays."""

    # Sentinel-1 RTC -- dims (time, band, y, x), bands = ["vv", "vh"]
    # Values: linear power (gamma0), float32
    s1: xr.DataArray

    # Sentinel-2 L2A -- dims (time, band, y, x), bands = ["B04","B03","B02","B08","B11","B12","SCL"]
    # Values: integer 0-10000 surface reflectance (SCL is class codes)
    s2: xr.DataArray

    # Copernicus GLO-30 DEM -- dims (band, y, x), band = ["data"]
    # Values: elevation in metres
    dem: xr.DataArray

    # Number of S1 scenes found
    s1_count: int

    # Number of S2 scenes found (before cloud filtering)
    s2_count: int

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<ImageryData  S1={self.s1_count} scenes  S2={self.s2_count} scenes  "
            f"grid={self.s1.sizes.get('x')}x{self.s1.sizes.get('y')} px>"
        )


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

class ImageryFetcher:
    """Streams Sentinel-1, Sentinel-2, and DEM data from Planetary Computer.

    Parameters
    ----------
    aoi:
        Resolved AOI produced by ``AOIBuilder``.
    start_date:
        ISO 8601 start date, e.g. ``"2022-01-01"``.
    end_date:
        ISO 8601 end date, e.g. ``"2024-12-31"``.
    max_cloud_cover:
        Maximum allowed Sentinel-2 scene cloud percentage (0-100).
    orbit_direction:
        Sentinel-1 orbit direction: ``"ascending"`` or ``"descending"``.
    chunk_size:
        Dask chunk size in pixels for x and y dimensions.  Larger chunks
        are faster for bulk operations; smaller chunks reduce peak memory.
    """

    def __init__(
        self,
        aoi: AOIResult,
        start_date: str,
        end_date: str,
        max_cloud_cover: int = 15,
        orbit_direction: str = "ascending",
        chunk_size: int = 1024,
    ) -> None:
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        self.max_cloud_cover = max_cloud_cover
        self.orbit_direction = orbit_direction.lower()
        self.chunk_size = chunk_size
        self._datetime_range = f"{start_date}/{end_date}"
        self._bbox: tuple[float, float, float, float] = (
            float(aoi.bbox_wgs84[0]),
            float(aoi.bbox_wgs84[1]),
            float(aoi.bbox_wgs84[2]),
            float(aoi.bbox_wgs84[3]),
        )

        # Open catalog once; sign_inplace adds SAS tokens to asset hrefs
        self._catalog = pystac_client.Client.open(
            PLANETARY_COMPUTER_URL,
            modifier=planetary_computer.sign_inplace,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fetch_all(self, verbose: bool = True) -> ImageryData:
        """Fetch S1, S2, and DEM in sequence and return an ImageryData object.

        The DataArrays are lazy (dask-backed) -- computation happens only
        when ``.compute()`` or ``.values`` is called.
        """
        s1_da, s1_count = self._fetch_s1(verbose=verbose)
        s2_da, s2_count = self._fetch_s2(verbose=verbose)
        dem_da = self._fetch_dem(verbose=verbose)

        # Harmonize: snap all sensors to S1's master 10 m pixel grid
        if verbose:
            print("Harmonizing sensor grids ...")
        s1_da, s2_da, dem_da = self._harmonize_grids(
            s1_da, s2_da, dem_da, verbose=verbose,
        )

        return ImageryData(
            s1=s1_da,
            s2=s2_da,
            dem=dem_da,
            s1_count=s1_count,
            s2_count=s2_count,
        )

    # ------------------------------------------------------------------
    # Sentinel-1 RTC
    # ------------------------------------------------------------------

    def _fetch_s1(self, verbose: bool = True) -> tuple[xr.DataArray, int]:
        """Search and stack Sentinel-1 RTC VV/VH for the AOI + date range."""
        if verbose:
            print(f"Searching Sentinel-1 RTC ({self.start_date} -- {self.end_date}) ...")

        search = self._catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=self._bbox,
            datetime=self._datetime_range,
            query={"sat:orbit_state": {"eq": self.orbit_direction}},
        )
        items = list(search.items())

        if verbose:
            print(f"  Found {len(items)} S1 scenes.")

        if len(items) == 0:
            raise RuntimeError(
                "No Sentinel-1 RTC scenes found.  "
                "Try a wider date range or change orbit_direction."
            )
        if len(items) < 15:
            warnings.warn(
                f"Only {len(items)} S1 scenes found.  "
                "Stability estimates are more reliable with >=15 scenes.",
                stacklevel=2,
            )

        stack = stackstac.stack(
            items,
            assets=["vv", "vh"],
            bounds_latlon=self._bbox,
            epsg=self.aoi.utm_crs.to_epsg(),
            resolution=10,
            dtype="float32",  # type: ignore[arg-type]
            fill_value=np.float32("nan"),  # type: ignore[arg-type]
            rescale=False,   # S1 RTC values are already in linear power units
            chunksize={"x": self.chunk_size, "y": self.chunk_size},  # type: ignore[arg-type]
        )

        if verbose:
            t, _, h, w = stack.sizes["time"], stack.sizes.get("band"), stack.sizes["y"], stack.sizes["x"]
            print(f"  S1 stack: {t} scenes x {h}x{w} px at 10 m.")

        return stack, len(items)

    # ------------------------------------------------------------------
    # Sentinel-2 L2A
    # ------------------------------------------------------------------

    def _fetch_s2(self, verbose: bool = True) -> tuple[xr.DataArray, int]:
        """Search and stack Sentinel-2 L2A for the AOI + date range."""
        if verbose:
            print(f"Searching Sentinel-2 L2A ({self.start_date} -- {self.end_date}) ...")

        search = self._catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=self._bbox,
            datetime=self._datetime_range,
            query={"eo:cloud_cover": {"lte": self.max_cloud_cover}},
        )
        items = list(search.items())

        if verbose:
            print(f"  Found {len(items)} S2 scenes with <={self.max_cloud_cover}% cloud cover.")

        if len(items) == 0:
            raise RuntimeError(
                "No Sentinel-2 L2A scenes found.  "
                "Try increasing max_cloud_cover or widening the date range."
            )

        # Request only the bands needed for the analysis + SCL for cloud masking
        bands = ["B04", "B03", "B02", "B08", "B11", "B12", "SCL"]

        stack = stackstac.stack(
            items,
            assets=bands,
            bounds_latlon=self._bbox,
            epsg=self.aoi.utm_crs.to_epsg(),
            resolution=10,
            dtype="float32",  # type: ignore[arg-type]
            fill_value=np.float32("nan"),  # type: ignore[arg-type]
            rescale=False,   # S2 L2A values are raw UInt16 (0-10000 DN); analysis.py divides by 10000
            chunksize={"x": self.chunk_size, "y": self.chunk_size},  # type: ignore[arg-type]
        )

        stack = self._drop_duplicate_times(stack)

        if verbose:
            t, _, h, w = stack.sizes["time"], stack.sizes.get("band"), stack.sizes["y"], stack.sizes["x"]
            print(f"  S2 stack: {t} scenes x {h}x{w} px at 10 m.")

        return stack, len(items)

    # ------------------------------------------------------------------
    # Copernicus DEM GLO-30
    # ------------------------------------------------------------------

    def _fetch_dem(self, verbose: bool = True) -> xr.DataArray:
        """Fetch and mosaic the COP-DEM-GLO-30 tiles covering the AOI."""
        if verbose:
            print("Fetching Copernicus GLO-30 DEM ...")

        search = self._catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=self._bbox,
        )
        items = list(search.items())

        if len(items) == 0:
            raise RuntimeError("No COP-DEM tiles found for the AOI.")

        if verbose:
            print(f"  Found {len(items)} DEM tile(s).")

        stack = stackstac.stack(
            items,
            assets=["data"],
            bounds_latlon=self._bbox,
            epsg=self.aoi.utm_crs.to_epsg(),
            resolution=30,                  # DEM native resolution
            dtype="float32",  # type: ignore[arg-type]
            fill_value=np.float32("nan"),  # type: ignore[arg-type]
            rescale=False,   # DEM values are raw elevation in metres
            chunksize={"x": self.chunk_size, "y": self.chunk_size},  # type: ignore[arg-type]
        )

        # Mosaic across tiles (take the first valid value)
        dem = stack.isel(band=0).median(dim="time", skipna=True)
        dem = dem.expand_dims("band").assign_coords(band=["elevation"])
        dem.attrs.update({"units": "metres", "long_name": "Elevation (m)"})

        if verbose:
            h, w = dem.sizes["y"], dem.sizes["x"]
            print(f"  DEM mosaic: {h}x{w} px at 30 m.")

        return dem

    # ------------------------------------------------------------------
    # Grid harmonization
    # ------------------------------------------------------------------

    @staticmethod
    def _harmonize_grids(
        s1: xr.DataArray,
        s2: xr.DataArray,
        dem: xr.DataArray,
        verbose: bool = True,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Snap all sensor rasters to a common pixel grid.

        Multi-sensor fusion requires pixel-perfect alignment.  Different
        sensors have different native CRS origins and resolutions, so
        independent stackstac calls can produce grids with sub-pixel
        offsets even when given the same target parameters.

        Remote-sensing best-practice strategy:
          1. Designate S1 (10 m) as the **master grid** -- its y/x
             coordinates become the authoritative pixel locations.
          2. Reindex S2 (10 m) to the S1 grid via nearest-neighbour
             (preserves discrete SCL class values and avoids blending
             reflectance across pixel boundaries).
          3. Bilinearly interpolate the DEM (30 m) to the 10 m master
             grid -- bilinear is appropriate for smooth continuous
             elevation surfaces.
          4. Trim 2 pixels from all edges to remove reprojection-boundary
             artefacts where partial-pixel coverage degrades accuracy.
        """
        ref_y = s1["y"].values
        ref_x = s1["x"].values

        # --- S2 alignment check ----------------------------------------
        s2_y = s2["y"].values
        s2_x = s2["x"].values

        if len(ref_y) == len(s2_y) and len(ref_x) == len(s2_x):
            dy = float(np.abs(ref_y - s2_y).max())
            dx = float(np.abs(ref_x - s2_x).max())
        else:
            dy = dx = float("inf")

        if verbose:
            print(f"  S1-S2 grid offset: dy={dy:.2f} m, dx={dx:.2f} m")

        if dy > 0.5 or dx > 0.5 or len(ref_y) != len(s2_y):
            s2 = s2.interp(y=ref_y, x=ref_x, method="nearest")
            if verbose:
                print("  --> Resampled S2 to S1 master grid (nearest-neighbour).")
        else:
            # Coordinates close enough -- force exact match
            s2 = s2.assign_coords(y=ref_y, x=ref_x)

        # --- DEM upsampling (30 m -> 10 m) -----------------------------
        dem_h, dem_w = dem.sizes["y"], dem.sizes["x"]
        ref_h, ref_w = len(ref_y), len(ref_x)
        dem = dem.interp(y=ref_y, x=ref_x, method="linear")
        if verbose:
            print(
                f"  Resampled DEM {dem_h}x{dem_w} -> {ref_h}x{ref_w} px "
                f"(bilinear, 30 m -> 10 m)."
            )

        # --- Edge trimming (remove reprojection boundary artefacts) ----
        trim = 2
        if ref_h > 2 * trim + 10 and ref_w > 2 * trim + 10:
            y_sl = slice(trim, ref_h - trim)
            x_sl = slice(trim, ref_w - trim)
            s1 = s1.isel(y=y_sl, x=x_sl)
            s2 = s2.isel(y=y_sl, x=x_sl)
            dem = dem.isel(y=y_sl, x=x_sl)
            if verbose:
                print(
                    f"  Trimmed {trim}-px edges -> "
                    f"{s1.sizes['y']}x{s1.sizes['x']} px."
                )

        return s1, s2, dem

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_duplicate_times(da: xr.DataArray) -> xr.DataArray:
        """Keep only the first acquisition when multiple granules share a timestamp."""
        _, idx = np.unique(da.time.values, return_index=True)
        return da.isel(time=idx)
