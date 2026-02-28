"""
fetcher.py
==========
Data acquisition layer for high-resolution SAR + optical imagery.

Data sources
------------
* **Capella Space Open Data** — X-band SAR (0.3-1 m), single-pol HH/VV,
  accessed via the static STAC catalog on S3.
* **NAIP** (National Agriculture Imagery Program) — 4-band (R, G, B, NIR)
  at 0.6 m, fetched from Microsoft Planetary Computer.
* **Sentinel-1 RTC** — C-band SAR fallback at 10 m if no Capella data
  is available for the AOI.
* **Copernicus GLO-30 DEM** — 30 m elevation for slope masking.

Resolution harmonisation resamples all layers to a common 1 m pixel grid
aligned to the SAR image.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from scipy.ndimage import zoom
import pystac
import pystac_client
import planetary_computer

from .aoi import AOIResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HiResImageryData:
    """All fetched imagery arrays aligned to a common 1 m pixel grid.

    Every array is (H, W) or (H, W, bands) in the UTM CRS of the AOI.
    """

    # SAR amplitude (linear power, float32) — (H, W)
    sar: np.ndarray
    sar_source: str           # "capella" | "sentinel-1"
    sar_resolution_m: float   # native resolution before resampling

    # Optical (float32 0-1) — (H, W, 4) for R, G, B, NIR
    naip: np.ndarray
    naip_source: str          # "naip" | "sentinel-2"
    naip_resolution_m: float

    # DEM (float32 metres) — (H, W)
    dem: np.ndarray

    # Common grid metadata
    transform: Affine
    crs: CRS
    height: int
    width: int

    def __repr__(self) -> str:
        return (
            f"<HiResImageryData  SAR={self.sar_source}@{self.sar_resolution_m}m  "
            f"Optical={self.naip_source}@{self.naip_resolution_m}m  "
            f"grid={self.height}x{self.width} px>"
        )


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

CAPELLA_CATALOG_URL = (
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json"
)
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


class HiResImageryFetcher:
    """Fetch and harmonise high-resolution SAR + optical data.

    Parameters
    ----------
    aoi : Resolved AOI.
    target_resolution : Common pixel size in metres (default 1.0).
    naip_year_range : (start, end) years for NAIP search.
    s1_fallback : If *True* (default), fall back to Sentinel-1 10 m if
        no Capella data intersects the AOI.
    """

    def __init__(
        self,
        aoi: AOIResult,
        target_resolution: float = 1.0,
        naip_year_range: Tuple[int, int] = (2019, 2023),
        s1_fallback: bool = True,
    ) -> None:
        self.aoi = aoi
        self.res = target_resolution
        self.naip_years = naip_year_range
        self.s1_fallback = s1_fallback

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fetch_all(self, verbose: bool = True) -> HiResImageryData:
        """Fetch SAR + optical + DEM and harmonise to a common grid."""

        # -- Common grid definition ------------------------------------
        x0, y0, x1, y1 = self.aoi.bbox_utm
        width  = int(round((x1 - x0) / self.res))
        height = int(round((y1 - y0) / self.res))
        transform = transform_from_bounds(x0, y0, x1, y1, width, height)
        crs = self.aoi.utm_crs

        if verbose:
            print(f"Target grid: {height}x{width} px @ {self.res} m  ({crs.to_epsg()})")

        # -- SAR -------------------------------------------------------
        sar_arr, sar_src, sar_res = self._fetch_sar(
            transform, crs, height, width, verbose,
        )

        # -- Optical ---------------------------------------------------
        opt_arr, opt_src, opt_res = self._fetch_optical(
            transform, crs, height, width, verbose,
        )

        # -- DEM -------------------------------------------------------
        dem_arr = self._fetch_dem(transform, crs, height, width, verbose)

        return HiResImageryData(
            sar=sar_arr,
            sar_source=sar_src,
            sar_resolution_m=sar_res,
            naip=opt_arr,
            naip_source=opt_src,
            naip_resolution_m=opt_res,
            dem=dem_arr,
            transform=transform,
            crs=crs,
            height=height,
            width=width,
        )

    # ------------------------------------------------------------------
    # SAR acquisition
    # ------------------------------------------------------------------

    def _fetch_sar(
        self, transform, crs, height, width, verbose,
    ) -> Tuple[np.ndarray, str, float]:
        """Try Capella first, fall back to Sentinel-1 if needed."""
        capella = self._try_capella(transform, crs, height, width, verbose)
        if capella is not None:
            return capella

        if not self.s1_fallback:
            raise RuntimeError(
                "No Capella data found for the AOI and s1_fallback is disabled."
            )

        if verbose:
            print("  No Capella data found — falling back to Sentinel-1 RTC …")
        return self._fetch_s1_fallback(transform, crs, height, width, verbose)

    def _try_capella(self, transform, crs, height, width, verbose):
        """Search Capella Open Data via fast parallel HTTP catalog walk.

        The Capella static STAC catalog on S3 is organised as:
            root → by-capital → {city}/collection.json → items
        We fetch all collection spatial extents in parallel, then only
        drill into collections whose bbox intersects our AOI.
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from shapely.geometry import shape as shapely_shape, box as shapely_box

        if verbose:
            print("Searching Capella Open Data catalog …")

        w, s, e, n = self.aoi.bbox_wgs84
        aoi_box = shapely_box(w, s, e, n)
        _TIMEOUT = 8

        session = requests.Session()
        session.headers.update({"Accept": "application/json"})

        # -- Step 1: load capital index ---------------------------------
        capital_index_url = (
            "https://capella-open-data.s3.us-west-2.amazonaws.com/"
            "stac/capella-open-data-by-capital/catalog.json"
        )
        try:
            resp = session.get(capital_index_url, timeout=_TIMEOUT)
            resp.raise_for_status()
            capital_cat = resp.json()
        except Exception as exc:
            if verbose:
                print(f"  Capella catalog unavailable: {exc}")
            return None

        child_links = [
            lk for lk in capital_cat.get("links", [])
            if lk.get("rel") == "child"
        ]
        if verbose:
            print(f"  Scanning {len(child_links)} capital collections …")

        base = (
            "https://capella-open-data.s3.us-west-2.amazonaws.com/"
            "stac/capella-open-data-by-capital/"
        )

        # -- Step 2: parallel bbox check of all collections -------------
        def _fetch_collection(lk):
            """Return (col_json, col_url) or None."""
            col_href = lk["href"]
            col_url = base + col_href.lstrip("./")
            try:
                r = session.get(col_url, timeout=_TIMEOUT)
                r.raise_for_status()
                col = r.json()
                bbox_list = (
                    col.get("extent", {})
                    .get("spatial", {})
                    .get("bbox", [])
                )
                bbox = bbox_list[0] if bbox_list else []
                if bbox and len(bbox) >= 4:
                    if not aoi_box.intersects(shapely_box(*bbox[:4])):
                        return None     # fast spatial reject
                else:
                    return None         # no extent → skip
                return (col, col_url)
            except Exception:
                return None

        matching_collections = []
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {pool.submit(_fetch_collection, lk): lk for lk in child_links}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    matching_collections.append(result)

        if verbose:
            n_match = len(matching_collections)
            print(f"  {n_match} collection(s) to scan  (no Capella data near "
                  f"AOI)" if not matching_collections
                  else f"  {n_match} collection(s) to scan for items …")

        if not matching_collections:
            return None

        # -- Step 3: check items in matching collections ----------------
        best_item_json = None
        best_res = 999.0
        best_href: Optional[str] = None
        n_checked = 0

        for col_json, col_url in matching_collections:
            col_dir = col_url.rsplit("/", 1)[0] + "/"
            item_links = [
                il for il in col_json.get("links", [])
                if il.get("rel") == "item"
            ]
            for il in item_links:
                item_href = il["href"]
                item_url = (
                    item_href if item_href.startswith("http")
                    else col_dir + item_href.lstrip("./")
                )
                try:
                    item_resp = session.get(item_url, timeout=_TIMEOUT)
                    item_resp.raise_for_status()
                    item_json = item_resp.json()
                except Exception:
                    continue

                n_checked += 1
                geom = item_json.get("geometry")
                if geom is None:
                    continue
                if not aoi_box.intersects(shapely_shape(geom)):
                    continue

                props = item_json.get("properties", {})
                res = float(props.get(
                    "sar:resolution_range",
                    props.get("sar:resolution_azimuth", 1.0),
                ))
                if res < best_res:
                    assets = item_json.get("assets", {})
                    for ak in ("HH", "VV", "GEO", "GEC", "analytic"):
                        if ak in assets:
                            best_item_json = item_json
                            best_res = res
                            best_href = assets[ak].get("href")
                            break

        if verbose:
            print(f"  Checked {n_checked} Capella items.")

        if best_item_json is None or best_href is None:
            return None

        if verbose:
            print(
                f"  Found Capella scene: {best_item_json.get('id')}  "
                f"(~{best_res:.2f} m)"
            )

        try:
            arr = self._read_and_reproject(
                best_href, transform, crs, height, width,
            )
            return (arr, "capella", best_res)
        except Exception as exc:
            if verbose:
                print(f"  Failed to read Capella asset: {exc}")
            return None

    def _fetch_s1_fallback(self, transform, crs, height, width, verbose):
        """Fetch Sentinel-1 RTC from Planetary Computer as SAR fallback."""
        import stackstac

        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            intersects=self.aoi.geojson,
            datetime="2022-01-01/2024-12-31",
        )
        items = list(search.items())
        if verbose:
            print(f"  Found {len(items)} S1 RTC scenes.")

        if not items:
            raise RuntimeError("No Sentinel-1 data for the AOI.")

        # Stack and compute median VV amplitude
        # bounds must be in the target CRS (UTM metres)
        x0, y0, x1, y1 = self.aoi.bbox_utm
        stack = stackstac.stack(
            items[:20],           # limit to 20 most recent for speed
            assets=["vv"],
            bounds=(x0, y0, x1, y1),
            epsg=crs.to_epsg(),
            resolution=self.res,
        )
        median_vv = stack.median(dim="time").compute(
            scheduler="synchronous"
        ).values.squeeze()
        median_vv = np.nan_to_num(median_vv, nan=0.0).astype(np.float32)

        # Ensure correct shape
        if median_vv.shape != (height, width):
            median_vv = np.asarray(zoom(
                median_vv,
                (height / median_vv.shape[0], width / median_vv.shape[1]),
                order=1,
            )).astype(np.float32)

        if verbose:
            print(f"  S1 VV median: {median_vv.shape} @ {self.res} m")

        return (median_vv, "sentinel-1", 10.0)

    # ------------------------------------------------------------------
    # Optical acquisition
    # ------------------------------------------------------------------

    def _fetch_optical(self, transform, crs, height, width, verbose):
        """Try NAIP first, fall back to Sentinel-2 if needed."""
        naip = self._try_naip(transform, crs, height, width, verbose)
        if naip is not None:
            return naip

        if verbose:
            print("  No NAIP data found — falling back to Sentinel-2 …")
        return self._fetch_s2_fallback(transform, crs, height, width, verbose)

    def _try_naip(self, transform, crs, height, width, verbose):
        """Fetch NAIP 4-band (R, G, B, NIR) from Planetary Computer.

        Mosaics **all** overlapping tiles from the most recent year so
        that AOIs straddling tile boundaries do not have gaps.
        """
        if verbose:
            print("Searching NAIP imagery …")

        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace,
        )
        y0, y1 = self.naip_years
        search = catalog.search(
            collections=["naip"],
            intersects=self.aoi.geojson,
            datetime=f"{y0}-01-01/{y1}-12-31",
        )
        items = sorted(
            search.items(),
            key=lambda it: it.datetime or it.properties.get("datetime", ""),
            reverse=True,
        )

        if not items:
            return None

        # Group tiles by year — use the most-recent year that has data
        best_year = (items[0].datetime or items[0].properties.get("datetime", ""))
        if hasattr(best_year, "year"):
            best_year_str = str(best_year.year)
        else:
            best_year_str = str(best_year)[:4]

        year_items = [
            it for it in items
            if str(getattr(it.datetime, "year", str(it.properties.get("datetime", ""))[:4]))
            .startswith(best_year_str[:4])
        ]
        if not year_items:
            year_items = items[:1]

        if verbose:
            print(f"  Found {len(items)} NAIP tiles.  Mosaicking {len(year_items)} from {best_year_str}.")

        # Accumulate mosaic: later tiles fill in zeros left by earlier ones
        mosaic = np.zeros((height, width, 4), dtype=np.float32)

        for it in year_items:
            href = it.assets["image"].href
            try:
                tile = self._read_naip_and_reproject(
                    href, transform, crs, height, width,
                )
                # Fill only where mosaic is still zero (no data yet)
                empty_mask = mosaic.max(axis=2) < 1e-6
                mosaic[empty_mask] = tile[empty_mask]
            except Exception:
                continue

        if mosaic.max() < 1e-6:
            return None

        if verbose:
            print(f"  NAIP: {mosaic.shape[:2]} px, 4 bands @ ~0.6 m")
        return (mosaic, "naip", 0.6)

    def _fetch_s2_fallback(self, transform, crs, height, width, verbose):
        """Sentinel-2 fallback: median R/G/B/NIR at 10 m."""
        import stackstac

        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=self.aoi.geojson,
            datetime="2022-01-01/2024-12-31",
            query={"eo:cloud_cover": {"lt": 20}},
        )
        items = list(search.items())
        if verbose:
            print(f"  Found {len(items)} S2 scenes (<=20 % cloud).")

        if not items:
            raise RuntimeError("No optical imagery found for the AOI.")

        x0, y0, x1, y1 = self.aoi.bbox_utm
        bands = ["B04", "B03", "B02", "B08"]
        stack = stackstac.stack(
            items[:10], assets=bands,
            bounds=(x0, y0, x1, y1),
            epsg=crs.to_epsg(),
            resolution=self.res,
        )
        median = stack.median(dim="time").compute(
            scheduler="synchronous",
        ).values  # (bands, H, W)

        # Normalise 0-1
        rgbnir = np.stack([
            median[0], median[1], median[2], median[3],
        ], axis=-1).astype(np.float32) / 10000.0
        rgbnir = np.clip(rgbnir, 0.0, 1.0)

        if rgbnir.shape[:2] != (height, width):
            from scipy.ndimage import zoom as nd_zoom
            factors = (height / rgbnir.shape[0], width / rgbnir.shape[1], 1)
            rgbnir = np.asarray(nd_zoom(rgbnir, factors, order=1)).astype(np.float32)

        if verbose:
            print(f"  S2 median RGBNIR: {rgbnir.shape[:2]} @ {self.res} m")

        return (rgbnir, "sentinel-2", 10.0)

    # ------------------------------------------------------------------
    # DEM
    # ------------------------------------------------------------------

    def _fetch_dem(self, transform, crs, height, width, verbose):
        """Fetch Copernicus GLO-30 DEM from Planetary Computer."""
        import stackstac

        if verbose:
            print("Fetching Copernicus GLO-30 DEM …")

        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            intersects=self.aoi.geojson,
        )
        items = list(search.items())

        if not items:
            if verbose:
                print("  No DEM found — using flat terrain assumption.")
            return np.zeros((height, width), dtype=np.float32)

        if verbose:
            print(f"  Found {len(items)} DEM tile(s).")

        x0, y0, x1, y1 = self.aoi.bbox_utm
        stack = stackstac.stack(
            items, assets=["data"],
            bounds=(x0, y0, x1, y1),
            epsg=crs.to_epsg(),
            resolution=self.res,
        )
        dem = stack.median(dim="time").compute(
            scheduler="synchronous",
        ).values.squeeze().astype(np.float32)

        dem = np.nan_to_num(dem, nan=0.0)

        if dem.shape != (height, width):
            dem = np.asarray(zoom(
                dem, (height / dem.shape[0], width / dem.shape[1]), order=1,
            )).astype(np.float32)

        if verbose:
            print(f"  DEM: {dem.shape} @ {self.res} m")

        return dem

    # ------------------------------------------------------------------
    # Raster I/O helpers
    # ------------------------------------------------------------------

    def _read_and_reproject(self, href, transform, crs, height, width):
        """Read a single-band raster and reproject to the common grid."""
        dst = np.zeros((height, width), dtype=np.float32)

        with rasterio.open(href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
        return dst

    def _read_naip_and_reproject(self, href, transform, crs, height, width):
        """Read a 4-band NAIP tile and reproject to the common grid."""
        dst = np.zeros((4, height, width), dtype=np.float32)

        with rasterio.open(href) as src:
            n_bands = min(src.count, 4)
            for b in range(1, n_bands + 1):
                band_dst = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, b),
                    destination=band_dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                )
                dst[b - 1] = band_dst

        # (bands, H, W) -> (H, W, bands), normalise to 0-1
        rgbnir = np.moveaxis(dst, 0, -1)
        if rgbnir.max() > 2.0:          # NAIP stores 0--255
            rgbnir = rgbnir / 255.0
        return np.clip(rgbnir, 0.0, 1.0).astype(np.float32)
