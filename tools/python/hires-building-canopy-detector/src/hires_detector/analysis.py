"""
analysis.py
===========
Core analysis pipeline: SAR feature extraction, building footprint detection,
tree canopy mapping, and spectral species classification.

Algorithms
----------
* **Lee sigma speckle filter** — adaptive noise reduction for SAR amplitude.
* **Morphological Building Index (MBI)** — directional white top-hat
  profiles with linear structuring elements at 0°/45°/90°/135° and
  multiple scales, capturing building walls and rooftop edges.
* **Local contrast ratio** — pixel-to-neighbourhood ratio highlighting
  double-bounce building signatures in SAR.
* **Edge density** — Canny-based edge detection smoothed into a density
  map; high density indicates structured features.
* **Multi-criterion building fusion** — weighted combination of SAR +
  optical features with seven-component regularisation scoring.
* **Marker-controlled watershed** — individual tree crown delineation
  from Gaussian-smoothed NDVI with local-maxima seed points.
* **K-means spectral clustering** — per-crown NAIP (R/G/B/NIR) + SAR
  feature extraction followed by unsupervised grouping into
  deciduous / conifer / mixed species categories.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import shape, mapping
from rasterio.features import shapes, geometry_mask, rasterize
from rasterio.transform import Affine
from scipy.ndimage import (
    uniform_filter,
    gaussian_filter,
    label as ndi_label,
    binary_erosion,
    binary_dilation,
    grey_opening,
)
from scipy.cluster.vq import kmeans2
from skimage.feature import canny, peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops

from .fetcher import HiResImageryData


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HiResResult:
    """Complete outputs produced by the high-resolution analysis pipeline."""

    # ---- SAR features ---------------------------------------------------
    sar_db: np.ndarray                  # amplitude in dB
    sar_despeckled: np.ndarray          # Lee-filtered dB
    mbi: np.ndarray                     # Morphological Building Index [0-1]
    sar_contrast: np.ndarray            # local contrast ratio [0-1]
    sar_edges: np.ndarray               # edge density [0-1]
    shadow_mask: np.ndarray             # boolean shadow mask

    # ---- Optical features -----------------------------------------------
    ndvi: np.ndarray                    # Normalised Difference Vegetation Index
    brightness: np.ndarray              # mean(R, G, B)

    # ---- Building detection ---------------------------------------------
    building_score: np.ndarray          # continuous [0-1]
    building_mask: np.ndarray           # final binary mask
    building_footprints: gpd.GeoDataFrame  # regularised footprints

    # ---- Tree canopy ----------------------------------------------------
    canopy_mask: np.ndarray             # binary canopy mask
    crown_labels: np.ndarray            # integer crown IDs (0 = no crown)
    tree_crowns: gpd.GeoDataFrame       # vectorised crown polygons

    # ---- Species classification -----------------------------------------
    species_map: np.ndarray             # pixel-level species class (0 = none)
    species_crowns: gpd.GeoDataFrame    # crowns with species attributes
    species_legend: Dict[int, str]      # class ID → label string

    # ---- RGB composites -------------------------------------------------
    optical_rgb: np.ndarray             # (H, W, 3) float32 0-1
    sar_rgb: np.ndarray                 # (H, W, 3) pseudo-colour

    # ---- Metadata -------------------------------------------------------
    transform: Affine = field(repr=False)
    crs_wkt: str = field(repr=False)
    height: int = 0
    width: int = 0
    params: Dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable detection summary."""
        n_bldg = len(self.building_footprints)
        bldg_area = (
            self.building_footprints["area_m2"].sum() if n_bldg else 0
        )
        n_crowns = len(self.tree_crowns)
        canopy_frac = self.canopy_mask.sum() / max(self.canopy_mask.size, 1)
        n_species = len(self.species_legend)
        print("=== Hi-Res Analysis Summary ==============================")
        print(f"  Grid            : {self.height} × {self.width} px")
        print(f"  Buildings       : {n_bldg:,}")
        print(f"  Building area   : {bldg_area:,.0f} m²")
        print(f"  Tree crowns     : {n_crowns:,}")
        print(f"  Canopy cover    : {canopy_frac * 100:.1f}%")
        print(f"  Species groups  : {n_species}")
        for k, v in sorted(self.species_legend.items()):
            ct = int((self.species_map == k).sum())
            print(f"    {k}: {v}  ({ct:,} px)")
        print("==========================================================")


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict[str, Any] = {
    # SAR preprocessing
    "lee_window": 7,
    "mbi_scales": [3, 7, 15, 25],
    "mbi_angles": [0, 45, 90, 135],
    "contrast_window": 21,
    "shadow_k": 2.0,
    "edge_sigma": 1.5,

    # Building-fusion weights
    "w_mbi": 0.30,
    "w_contrast": 0.25,
    "w_edge": 0.15,
    "w_non_veg": 0.20,
    "w_shadow_prox": 0.10,

    # Building thresholds
    "building_threshold": 0.35,
    "min_building_area": 25.0,
    "max_building_area": 50_000.0,
    "building_score_threshold": 0.40,
    "morph_cleanup_iter": 1,

    # Canopy / crown
    "ndvi_threshold": 0.30,
    "crown_smooth_sigma": 3.0,
    "crown_min_distance": 5,
    "min_crown_area": 10.0,

    # Species
    "n_species_clusters": 5,
    "min_crown_pixels_for_species": 4,
}


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class HiResAnalyser:
    """Orchestrates the building + canopy + species pipeline.

    Parameters
    ----------
    imagery : HiResImageryData from the fetcher.
    **overrides : keyword overrides for any parameter in ``DEFAULT_PARAMS``.
    """

    def __init__(self, imagery: HiResImageryData, **overrides) -> None:
        self.img = imagery
        self.params: Dict[str, Any] = {**DEFAULT_PARAMS, **overrides}

        # Adapt parameters to SAR resolution
        if imagery.sar_resolution_m > 5.0:
            # Sentinel-1 at 10 m — widen kernels & raise thresholds for
            # the coarser, noisier SAR grid.
            self.params.setdefault("_hires", False)
            if "mbi_scales" not in overrides:
                self.params["mbi_scales"] = [2, 4, 8, 12]
            if "contrast_window" not in overrides:
                self.params["contrast_window"] = 11
            if "crown_min_distance" not in overrides:
                self.params["crown_min_distance"] = 3
            # S1 building scores are inflated — raise fusion threshold
            if "building_threshold" not in overrides:
                self.params["building_threshold"] = 0.50
            if "morph_cleanup_iter" not in overrides:
                self.params["morph_cleanup_iter"] = 2
            if "building_score_threshold" not in overrides:
                self.params["building_score_threshold"] = 0.45
        else:
            self.params["_hires"] = True

    # ==================================================================
    # Main entry point
    # ==================================================================

    def run(self, verbose: bool = True) -> HiResResult:
        """Execute the full seven-step analysis pipeline."""
        p = self.params
        sar_raw = self.img.sar
        naip = self.img.naip
        H, W = self.img.height, self.img.width
        transform = self.img.transform
        crs_wkt = str(self.img.crs)

        # ---- Step 1: SAR preprocessing --------------------------------
        if verbose:
            print("Step 1/7 — SAR preprocessing …")
        sar_db = self._to_db(sar_raw)
        sar_filt = self._lee_filter(sar_db, p["lee_window"])

        # ---- Step 2: SAR feature extraction ---------------------------
        if verbose:
            print("Step 2/7 — SAR feature extraction …")
        mbi = self._morphological_building_index(
            sar_filt, p["mbi_scales"], p["mbi_angles"],
        )
        contrast = self._local_contrast(sar_filt, p["contrast_window"])
        edges = self._edge_density(sar_filt, p["edge_sigma"])
        shadows = self._shadow_detection(sar_filt, p["shadow_k"])

        # ---- Step 3: Optical feature extraction -----------------------
        if verbose:
            print("Step 3/7 — Optical feature extraction …")
        ndvi = self._compute_ndvi(naip)
        brightness = self._brightness(naip)

        # ---- Step 4: Building detection -------------------------------
        if verbose:
            print("Step 4/7 — Building detection …")
        bldg_score = self._building_fusion(mbi, contrast, edges, ndvi, shadows)
        bldg_mask = self._morphological_cleanup(
            bldg_score > p["building_threshold"],
            p["morph_cleanup_iter"],
        )
        raw_fp = self._vectorize_footprints(
            bldg_mask, bldg_score, transform, crs_wkt, p["min_building_area"],
        )
        footprints = self._regularize_footprints(
            raw_fp, bldg_score, ndvi, transform,
        )
        bldg_mask_final = self._rasterize_footprints(footprints, H, W, transform)

        # ---- Step 5: Canopy detection ---------------------------------
        if verbose:
            print("Step 5/7 — Canopy detection …")
        canopy = self._canopy_mask(ndvi, bldg_mask_final, p["ndvi_threshold"])

        # ---- Step 6: Crown delineation --------------------------------
        if verbose:
            print("Step 6/7 — Crown delineation …")
        crown_labels = self._crown_delineation(
            ndvi, canopy, p["crown_smooth_sigma"], p["crown_min_distance"],
        )
        tree_gdf = self._vectorize_crowns(
            crown_labels, transform, crs_wkt, p["min_crown_area"],
        )

        # ---- Step 7: Species classification ---------------------------
        if verbose:
            print("Step 7/7 — Species classification …")
        sp_map, sp_gdf, sp_legend = self._classify_species(
            naip, sar_filt, crown_labels, tree_gdf,
            p["n_species_clusters"], p["min_crown_pixels_for_species"],
        )

        # ---- Composites -----------------------------------------------
        optical_rgb = np.clip(naip[:, :, :3], 0, 1).astype(np.float32)
        sar_rgb = self._sar_pseudocolor(sar_filt, mbi, shadows)

        result = HiResResult(
            sar_db=sar_db,
            sar_despeckled=sar_filt,
            mbi=mbi,
            sar_contrast=contrast,
            sar_edges=edges,
            shadow_mask=shadows,
            ndvi=ndvi,
            brightness=brightness,
            building_score=bldg_score,
            building_mask=bldg_mask_final,
            building_footprints=footprints,
            canopy_mask=canopy,
            crown_labels=crown_labels,
            tree_crowns=tree_gdf,
            species_map=sp_map,
            species_crowns=sp_gdf,
            species_legend=sp_legend,
            optical_rgb=optical_rgb,
            sar_rgb=sar_rgb,
            transform=transform,
            crs_wkt=crs_wkt,
            height=H,
            width=W,
            params=dict(p),
        )

        if verbose:
            result.summary()

        return result

    # ==================================================================
    # SAR preprocessing
    # ==================================================================

    @staticmethod
    def _to_db(sar: np.ndarray) -> np.ndarray:
        """Convert linear amplitude to decibel scale."""
        safe = np.clip(sar, 1e-10, None)
        return (10.0 * np.log10(safe)).astype(np.float32)

    @staticmethod
    def _lee_filter(sar_db: np.ndarray, window: int = 7) -> np.ndarray:
        """Lee sigma speckle filter.

        Computes a weighted combination of the local mean and the observed
        value.  The weight is the ratio of *local* variance to *overall*
        variance — homogeneous areas collapse to the local mean while
        strong scatterers are preserved.
        """
        img = sar_db.astype(np.float64)
        local_mean = uniform_filter(img, size=window)
        local_sq   = uniform_filter(img ** 2, size=window)
        local_var  = np.maximum(local_sq - local_mean ** 2, 0.0)
        overall_var = np.var(img[np.isfinite(img)]) if np.any(np.isfinite(img)) else 1.0
        weight = np.clip(local_var / (local_var + overall_var + 1e-12), 0, 1)
        return (local_mean + weight * (img - local_mean)).astype(np.float32)

    # ==================================================================
    # SAR feature extraction
    # ==================================================================

    @staticmethod
    def _linear_se(length: int, angle_deg: float) -> np.ndarray:
        """Create a linear (line-shaped) structuring element at *angle_deg*.

        The SE is a boolean 2-D array where only pixels along the line
        direction are ``True``.
        """
        half = max(length // 2, 1)
        size = 2 * half + 1
        se = np.zeros((size, size), dtype=bool)
        rad = np.radians(angle_deg)
        for k in range(-half, half + 1):
            r = int(round(half - k * np.sin(rad)))
            c = int(round(half + k * np.cos(rad)))
            se[np.clip(r, 0, size - 1), np.clip(c, 0, size - 1)] = True
        return se

    def _morphological_building_index(
        self,
        sar: np.ndarray,
        scales: List[int],
        angles: List[float],
    ) -> np.ndarray:
        """Morphological Building Index (MBI).

        For each orientation, a *white top-hat* (image − opening) is
        computed with a linear structuring element at several scales.
        Building walls and rooftop edges produce strong residuals because
        the narrow linear SE "fits inside" them, generating a large
        difference after opening.  The responses across all
        scale × angle combinations are averaged and percentile-normalised.
        """
        responses: List[np.ndarray] = []
        for angle in angles:
            for scale in scales:
                se = self._linear_se(scale, angle)
                wth = sar - grey_opening(sar, footprint=se)  # white top-hat
                responses.append(np.maximum(wth, 0.0))

        if not responses:
            return np.zeros_like(sar, dtype=np.float32)

        mbi = np.mean(responses, axis=0)
        valid = mbi[np.isfinite(mbi) & (mbi > 0)]
        if valid.size == 0:
            return np.zeros_like(sar, dtype=np.float32)
        lo, hi = np.percentile(valid, [2, 98])
        return np.clip((mbi - lo) / (hi - lo + 1e-10), 0, 1).astype(np.float32)

    @staticmethod
    def _local_contrast(sar: np.ndarray, window: int = 21) -> np.ndarray:
        """Local contrast ratio: pixel / local-mean.

        Buildings produce bright double-bounce returns against darker
        surroundings, yielding high contrast values.
        """
        local_mean = uniform_filter(sar.astype(np.float64), size=window)
        ratio = sar / (local_mean + 1e-10)
        valid = ratio[np.isfinite(ratio) & (ratio > 0)]
        if valid.size == 0:
            return np.zeros_like(sar, dtype=np.float32)
        lo, hi = np.percentile(valid, [2, 98])
        return np.clip((ratio - lo) / (hi - lo + 1e-10), 0, 1).astype(np.float32)

    @staticmethod
    def _edge_density(sar: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """Edge density map from Canny edges.

        High edge density reveals structured man-made features.
        The binary Canny output is Gaussian-smoothed into a continuous
        density surface (0-1).
        """
        s = sar.astype(np.float64)
        s_min, s_max = np.nanmin(s), np.nanmax(s)
        s_norm = (s - s_min) / (s_max - s_min + 1e-10)
        edge_binary = canny(s_norm, sigma=sigma).astype(np.float32)
        density = gaussian_filter(edge_binary, sigma=5.0)
        d_max = density.max()
        if d_max > 0:
            density /= d_max
        return density.astype(np.float32)

    @staticmethod
    def _shadow_detection(sar: np.ndarray, k: float = 2.0) -> np.ndarray:
        """Detect SAR shadow regions — pixels below mean − k × std.

        Shadows in SAR imagery appear directly *behind* buildings (relative
        to the sensor look direction) as dark regions adjacent to the
        bright double-bounce return.
        """
        finite = sar[np.isfinite(sar)]
        if finite.size == 0:
            return np.zeros_like(sar, dtype=bool)
        threshold = np.mean(finite) - k * np.std(finite)
        return (sar < threshold)

    # ==================================================================
    # Optical features
    # ==================================================================

    @staticmethod
    def _compute_ndvi(naip: np.ndarray) -> np.ndarray:
        """NDVI from NAIP (H, W, 4) where band order is R, G, B, NIR."""
        r   = naip[:, :, 0].astype(np.float64)
        nir = naip[:, :, 3].astype(np.float64)
        return ((nir - r) / (nir + r + 1e-10)).astype(np.float32)

    @staticmethod
    def _brightness(naip: np.ndarray) -> np.ndarray:
        """Mean of R, G, B channels."""
        return naip[:, :, :3].mean(axis=2).astype(np.float32)

    # ==================================================================
    # Building detection
    # ==================================================================

    def _building_fusion(
        self,
        mbi: np.ndarray,
        contrast: np.ndarray,
        edges: np.ndarray,
        ndvi: np.ndarray,
        shadows: np.ndarray,
    ) -> np.ndarray:
        """Fuse SAR and optical features into a continuous building score.

        .. math::
            S = w_{mbi}  \\cdot \\text{MBI}
              + w_{contrast} \\cdot C
              + w_{edge}  \\cdot E
              + w_{nv} \\cdot (1 - \\text{NDVI})
              + w_{sp} \\cdot \\text{shadow\\_proximity}
        """
        p = self.params
        non_veg = (1.0 - np.clip(ndvi, 0, 1)).astype(np.float32)

        # Shadow proximity — dilate shadow mask for spatial tolerance
        shadow_prox = binary_dilation(
            shadows, structure=np.ones((7, 7)), iterations=2,
        ).astype(np.float32)

        score = (
            p["w_mbi"]        * mbi
            + p["w_contrast"] * contrast
            + p["w_edge"]     * edges
            + p["w_non_veg"]  * non_veg
            + p["w_shadow_prox"] * shadow_prox
        )
        return np.clip(score, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------

    @staticmethod
    def _morphological_cleanup(
        mask: np.ndarray, iterations: int = 1,
    ) -> np.ndarray:
        """Binary opening to remove speckle-sized false positives.

        Set *iterations=0* to skip cleanup entirely.
        """
        if iterations <= 0:
            return mask.astype(bool)
        se = np.ones((3, 3), dtype=bool)
        eroded  = binary_erosion(mask, se, iterations=iterations)
        return binary_dilation(eroded, se, iterations=iterations)

    # ------------------------------------------------------------------

    @staticmethod
    def _vectorize_footprints(
        mask: np.ndarray,
        score: np.ndarray,
        transform: Affine,
        crs_wkt: str,
        min_area: float,
    ) -> gpd.GeoDataFrame:
        """Convert binary building mask → polygon GeoDataFrame.

        Each connected component becomes one record with ``area_m2``,
        ``score_mean``, and ``score_max`` attributes.
        """
        _COLS = ["geometry", "area_m2", "score_mean", "score_max"]
        labeled, n_feats = ndi_label(mask)  # type: ignore[misc]
        if n_feats == 0:
            return gpd.GeoDataFrame(
                columns=_COLS, geometry="geometry",
            ).set_crs(crs_wkt)

        pixel_area = abs(transform.a * transform.e)
        records: List[Dict] = []

        for region in regionprops(labeled, intensity_image=score):
            area_m2 = region.area * pixel_area
            if area_m2 < min_area:
                continue
            blob = (labeled == region.label).astype(np.uint8)
            polys = list(shapes(blob, mask=blob, transform=transform))
            if not polys:
                continue
            geom = shape(polys[0][0])
            pixels = score[labeled == region.label]
            records.append({
                "geometry": geom,
                "area_m2":  round(area_m2, 1),
                "score_mean": round(float(np.mean(pixels)), 4),
                "score_max":  round(float(np.max(pixels)), 4),
            })

        if not records:
            return gpd.GeoDataFrame(
                columns=_COLS, geometry="geometry",
            ).set_crs(crs_wkt)

        return gpd.GeoDataFrame(records, geometry="geometry").set_crs(crs_wkt)

    # ------------------------------------------------------------------

    def _regularize_footprints(
        self,
        footprints: gpd.GeoDataFrame,
        score: np.ndarray,
        ndvi: np.ndarray,
        transform: Affine,
    ) -> gpd.GeoDataFrame:
        """Score each footprint for building-likeness and filter.

        Seven-component scoring
        -----------------------
        1. **Rectangularity** — area / MRR area.
        2. **Compactness** — Polsby–Popper 4πA / P².
        3. **Solidity** — area / convex-hull area.
        4. **Edge sharpness** — gradient magnitude sampled on boundary.
        5. **Size appropriateness** — log-normal peaked at typical houses.
        6. **Detection probability** — mean score within polygon.
        7. **Vegetation penalty** — suppresses high-NDVI blobs.

        Polygons that pass the ``building_score_threshold`` are replaced
        by their minimum rotated rectangle when rectangularity > 0.6.
        """
        _COLS = [
            "geometry", "area_m2", "score_mean", "score_max",
            "compactness", "rectangularity", "solidity",
            "aspect_ratio", "edge_sharpness", "ndvi_mean",
            "building_score", "is_rectangular",
        ]
        if footprints.empty:
            return gpd.GeoDataFrame(
                columns=_COLS, geometry="geometry",
            ).set_crs(footprints.crs or "EPSG:4326")

        p = self.params

        # Pre-compute gradient for edge-sharpness
        score_c = np.nan_to_num(score, nan=0.0)
        gy, gx = np.gradient(score_c)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_max = grad_mag.max() + 1e-10

        ndvi_c = np.nan_to_num(ndvi, nan=0.0)

        records: List[Dict] = []
        for _, row in footprints.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            area = geom.area
            perim = geom.length

            # 1. Compactness (Polsby-Popper)
            compactness = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0

            # 2. Solidity
            solidity = area / max(geom.convex_hull.area, 1e-10)

            # 3. Rectangularity via MRR
            mrr = geom.minimum_rotated_rectangle
            rectangularity = area / max(mrr.area, 1e-10)

            # 4. Aspect ratio from MRR edges
            coords = list(mrr.exterior.coords)
            edges_len = sorted(
                np.hypot(
                    coords[i + 1][0] - coords[i][0],
                    coords[i + 1][1] - coords[i][1],
                )
                for i in range(len(coords) - 1)
            )
            aspect = edges_len[-1] / max(edges_len[0], 0.01) if edges_len else 1.0

            # 5. Edge sharpness — sample gradient along boundary
            boundary = geom.boundary
            n_pts = max(int(boundary.length / abs(transform.a)), 8)
            sharp_vals: List[float] = []
            for frac in np.linspace(0, 1, n_pts, endpoint=False):
                pt = boundary.interpolate(frac, normalized=True)
                ci = int((pt.x - transform.c) / transform.a)
                ri = int((pt.y - transform.f) / transform.e)
                if 0 <= ri < grad_mag.shape[0] and 0 <= ci < grad_mag.shape[1]:
                    sharp_vals.append(float(grad_mag[ri, ci]))
            edge_sharpness = float(np.mean(sharp_vals)) if sharp_vals else 0.0

            # 6. NDVI within polygon
            ndvi_vals = self._sample_polygon(ndvi_c, geom, transform)
            ndvi_mean = float(np.mean(ndvi_vals)) if ndvi_vals.size else 0.0

            # ---- Composite building score ----
            rect_sc  = np.clip(rectangularity, 0, 1)
            comp_sc  = np.clip(compactness, 0, 1)
            sol_sc   = np.clip(solidity, 0, 1)
            sharp_sc = min(edge_sharpness / grad_max, 1.0)
            log_a    = np.log10(max(area, 1.0))
            size_sc  = float(np.exp(-0.5 * ((log_a - 2.5) / 1.0) ** 2))
            prob_sc  = float(row.get("score_mean", 0.5))
            veg_pen  = max(0.0, ndvi_mean - 0.2) * 2.0

            bldg_sc = (
                0.20 * rect_sc
                + 0.15 * comp_sc
                + 0.15 * sol_sc
                + 0.10 * sharp_sc
                + 0.15 * size_sc
                + 0.15 * prob_sc
                - 0.10 * veg_pen
            )
            bldg_sc = float(np.clip(bldg_sc, 0, 1))

            if bldg_sc < p["building_score_threshold"]:
                continue
            if area > p["max_building_area"]:
                continue

            records.append({
                "geometry":       mrr if rectangularity > 0.6 else geom.convex_hull,
                "area_m2":        round(area, 1),
                "score_mean":     round(float(row.get("score_mean", 0)), 4),
                "score_max":      round(float(row.get("score_max", 0)), 4),
                "compactness":    round(compactness, 4),
                "rectangularity": round(rectangularity, 4),
                "solidity":       round(solidity, 4),
                "aspect_ratio":   round(aspect, 2),
                "edge_sharpness": round(edge_sharpness, 6),
                "ndvi_mean":      round(ndvi_mean, 4),
                "building_score": round(bldg_sc, 4),
                "is_rectangular": rectangularity > 0.6,
            })

        if not records:
            return gpd.GeoDataFrame(
                columns=_COLS, geometry="geometry",
            ).set_crs(footprints.crs or "EPSG:4326")

        return gpd.GeoDataFrame(records, geometry="geometry").set_crs(
            footprints.crs or "EPSG:4326"
        )

    # ------------------------------------------------------------------
    # Raster helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_polygon(
        raster: np.ndarray, geom, transform: Affine,
    ) -> np.ndarray:
        """Return raster values falling inside *geom*.

        Uses the polygon bounding-box to clip the raster window first,
        avoiding a full-size geometry mask for every polygon.
        """
        H, W = raster.shape[:2]
        minx, miny, maxx, maxy = geom.bounds
        # Convert geo coords → pixel indices
        col0 = int((minx - transform.c) / transform.a)
        col1 = int((maxx - transform.c) / transform.a) + 1
        row0 = int((maxy - transform.f) / transform.e)       # e is negative
        row1 = int((miny - transform.f) / transform.e) + 1
        col0, col1 = max(col0, 0), min(col1, W)
        row0, row1 = max(row0, 0), min(row1, H)
        if col0 >= col1 or row0 >= row1:
            return np.array([], dtype=raster.dtype)
        # Window transform & clipped raster
        win_transform = Affine(
            transform.a, transform.b, transform.c + col0 * transform.a,
            transform.d, transform.e, transform.f + row0 * transform.e,
        )
        win_raster = raster[row0:row1, col0:col1]
        mask = ~geometry_mask(
            [mapping(geom)],
            out_shape=win_raster.shape,
            transform=win_transform,
            all_touched=True,
        )
        return win_raster[mask]

    @staticmethod
    def _rasterize_footprints(
        gdf: gpd.GeoDataFrame, H: int, W: int, transform: Affine,
    ) -> np.ndarray:
        """Burn building footprints into a boolean raster."""
        if gdf.empty:
            return np.zeros((H, W), dtype=bool)
        geoms = [(g, 1) for g in gdf.geometry if g is not None]
        if not geoms:
            return np.zeros((H, W), dtype=bool)
        result = rasterize(
            geoms, out_shape=(H, W), transform=transform, dtype=np.uint8,
        )
        if result is None:
            return np.zeros((H, W), dtype=bool)
        return result.astype(bool)

    # ==================================================================
    # Canopy detection
    # ==================================================================

    @staticmethod
    def _canopy_mask(
        ndvi: np.ndarray,
        building_mask: np.ndarray,
        threshold: float = 0.3,
    ) -> np.ndarray:
        """Vegetation pixels (NDVI > threshold), excluding buildings."""
        return ((ndvi > threshold) & ~building_mask).astype(bool)

    @staticmethod
    def _crown_delineation(
        ndvi: np.ndarray,
        canopy: np.ndarray,
        smooth_sigma: float = 3.0,
        min_distance: int = 5,
    ) -> np.ndarray:
        """Marker-controlled watershed for individual tree crowns.

        1. Gaussian-smooth NDVI within canopy mask.
        2. Detect local maxima (tree tops) with minimum separation.
        3. Watershed on inverted smoothed NDVI from the seed maxima.
        """
        smoothed = gaussian_filter(ndvi.astype(np.float64), sigma=smooth_sigma)
        smoothed[~canopy] = 0.0

        # Seed markers at NDVI local maxima
        coords = peak_local_max(
            smoothed,
            min_distance=min_distance,
            threshold_abs=0.25,
            labels=canopy.astype(np.intp),
        )
        markers = np.zeros_like(ndvi, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        if markers.max() == 0:
            return np.zeros_like(ndvi, dtype=np.int32)

        return watershed(-smoothed, markers, mask=canopy).astype(np.int32)

    @staticmethod
    def _vectorize_crowns(
        crown_labels: np.ndarray,
        transform: Affine,
        crs_wkt: str,
        min_area: float = 10.0,
    ) -> gpd.GeoDataFrame:
        """Convert integer crown raster → polygon GeoDataFrame."""
        _COLS = ["geometry", "crown_id", "area_m2"]
        if crown_labels.max() == 0:
            return gpd.GeoDataFrame(
                columns=_COLS, geometry="geometry",
            ).set_crs(crs_wkt)

        pixel_area = abs(transform.a * transform.e)
        records: List[Dict] = []

        for region in regionprops(crown_labels):
            area_m2 = region.area * pixel_area
            if area_m2 < min_area:
                continue
            # Crop to bounding box for fast per-crown vectorisation
            r0, c0, r1, c1 = region.bbox
            crop = crown_labels[r0:r1, c0:c1]
            blob = (crop == region.label).astype(np.uint8)
            crop_tf = transform * Affine.translation(c0, r0)
            polys = list(shapes(blob, mask=blob, transform=crop_tf))
            if not polys:
                continue
            geom = shape(polys[0][0])
            records.append({
                "geometry": geom,
                "crown_id": int(region.label),
                "area_m2":  round(area_m2, 1),
            })

        if not records:
            return gpd.GeoDataFrame(
                columns=_COLS, geometry="geometry",
            ).set_crs(crs_wkt)

        return gpd.GeoDataFrame(records, geometry="geometry").set_crs(crs_wkt)

    # ==================================================================
    # Species classification
    # ==================================================================

    def _classify_species(
        self,
        naip: np.ndarray,
        sar: np.ndarray,
        crown_labels: np.ndarray,
        tree_gdf: gpd.GeoDataFrame,
        n_clusters: int = 5,
        min_pixels: int = 4,
    ) -> Tuple[np.ndarray, gpd.GeoDataFrame, Dict[int, str]]:
        """Unsupervised spectral clustering of tree crowns.

        Per-crown features extracted from NAIP (R/G/B/NIR) and SAR:

        ====  ==================================
        Feat  Description
        ====  ==================================
        0     NIR / Red ratio  (conifer ↔ deciduous)
        1     Green chromaticity  G / (R+G+B)
        2     Mean NDVI
        3     NIR standard deviation
        4     Mean NIR reflectance
        5     Mean SAR backscatter (dB)
        6     log₁₀(crown area in pixels)
        ====  ==================================

        K-means clusters are auto-labelled using spectral-signature
        heuristics (NIR/Red ratio, NDVI, green chromaticity).
        """
        H, W = crown_labels.shape
        species_map = np.zeros((H, W), dtype=np.int32)
        empty_legend: Dict[int, str] = {}

        if tree_gdf.empty or crown_labels.max() == 0:
            return species_map, tree_gdf.copy(), empty_legend

        # ---- per-crown feature extraction -----------------------------
        feature_rows: List[np.ndarray] = []
        crown_ids: List[int] = []

        for cid in np.unique(crown_labels):
            if cid == 0:
                continue
            mask = crown_labels == cid
            if mask.sum() < min_pixels:
                continue

            pix = naip[mask]      # (N, 4)
            pix_sar = sar[mask]   # (N,)

            r_m  = pix[:, 0].mean()
            g_m  = pix[:, 1].mean()
            b_m  = pix[:, 2].mean()
            nir_m = pix[:, 3].mean()

            feat = np.array([
                nir_m / (r_m + 1e-6),                              # NIR/Red
                g_m / (r_m + g_m + b_m + 1e-6),                    # green chrom
                (nir_m - r_m) / (nir_m + r_m + 1e-6),              # NDVI
                pix[:, 3].std(),                                    # NIR σ
                nir_m,                                              # NIR mean
                float(pix_sar.mean()),                              # SAR mean
                np.log10(max(float(mask.sum()), 1.0)),              # log area
            ], dtype=np.float64)

            feature_rows.append(feat)
            crown_ids.append(int(cid))

        # ---- cluster ---------------------------------------------------
        if len(feature_rows) < n_clusters:
            # Too few crowns — label everything as unclassified
            legend = {1: "Unclassified"}
            for cid in crown_ids:
                species_map[crown_labels == cid] = 1
            gdf = tree_gdf.copy()
            gdf["species_id"]   = 1
            gdf["species_name"] = "Unclassified"
            return species_map, gdf, legend

        X = np.array(feature_rows)
        stds = X.std(axis=0)
        stds[stds < 1e-10] = 1.0
        X_white = (X - X.mean(axis=0)) / stds

        k = min(n_clusters, len(X_white))
        try:
            _centroids, labels = kmeans2(X_white, k, minit="points", iter=30)
        except Exception:
            labels = np.zeros(len(X_white), dtype=int)

        # ---- auto-label clusters --------------------------------------
        legend: Dict[int, str] = {}
        for cid_label in range(k):
            members = X[labels == cid_label]
            if len(members) == 0:
                legend[cid_label + 1] = f"Group {cid_label + 1}"
                continue
            avg = members.mean(axis=0)
            nir_red  = avg[0]
            green_c  = avg[1]
            ndvi_val = avg[2]
            nir_val  = avg[4]

            if ndvi_val > 0.6 and nir_red > 2.0:
                name = "Deciduous Broadleaf"
            elif ndvi_val > 0.5 and nir_red < 1.6:
                name = "Conifer / Evergreen"
            elif nir_red > 2.5:
                name = "High-NIR Deciduous"
            elif green_c > 0.38:
                name = "Mixed / Broadleaf"
            elif nir_val > 0.4:
                name = "Dense Canopy"
            else:
                name = f"Species Group {cid_label + 1}"
            legend[cid_label + 1] = name

        # ---- map back to raster + GeoDataFrame ------------------------
        cid_to_sp: Dict[int, int] = {}
        for i, cid in enumerate(crown_ids):
            sp = int(labels[i]) + 1          # 1-based species ID
            cid_to_sp[cid] = sp
            species_map[crown_labels == cid] = sp

        gdf = tree_gdf.copy()
        gdf["species_id"]   = gdf["crown_id"].map(cid_to_sp).fillna(0).astype(int)
        gdf["species_name"] = gdf["species_id"].map(legend).fillna("Unclassified")

        return species_map, gdf, legend

    # ==================================================================
    # Visualisation helper
    # ==================================================================

    @staticmethod
    def _sar_pseudocolor(
        sar: np.ndarray, mbi: np.ndarray, shadows: np.ndarray,
    ) -> np.ndarray:
        """Pseudo-RGB from SAR features.

        * Red   = MBI (buildings)
        * Green = normalised SAR amplitude
        * Blue  = shadow mask (×0.6)
        """
        s = sar.astype(np.float64)
        lo, hi = np.nanpercentile(s, [2, 98])
        s_norm = np.clip((s - lo) / (hi - lo + 1e-10), 0, 1)

        rgb = np.stack(
            [
                np.clip(mbi, 0, 1),
                s_norm.astype(np.float32),
                shadows.astype(np.float32) * 0.6,
            ],
            axis=-1,
        )
        return rgb.astype(np.float32)
