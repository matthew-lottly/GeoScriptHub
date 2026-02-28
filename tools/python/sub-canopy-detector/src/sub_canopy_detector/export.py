"""
export.py
=========
Save analysis outputs to disk.

Supported formats
-----------------
GeoTIFF   -- probability, confidence, and all indicator rasters
GeoJSON   -- detected footprint polygons (with WGS84 reprojection)
Shapefile -- same as GeoJSON but as .shp
CSV       -- per-footprint attribute table
PNG       -- static summary map + individual indicator panels
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for headless execution
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from rasterio.crs import CRS

from .analysis import AnalysisResult
from .aoi import AOIResult


class OutputWriter:
    """Write all analysis outputs to a directory tree.

    Parameters
    ----------
    result:
        Completed ``AnalysisResult`` from ``SubCanopyAnalyser.run()``.
    aoi:
        Resolved AOI for reprojection of vector outputs.
    output_dir:
        Root directory for all saved files.  Created if it does not exist.
    study_name:
        Short prefix added to every output filename.
    """

    def __init__(
        self,
        result: AnalysisResult,
        aoi: AOIResult,
        output_dir: str = "./outputs",
        study_name: str = "sub_canopy",
    ) -> None:
        self.result = result
        self.aoi = aoi
        self.study_name = study_name
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Convenience: save everything
    # ------------------------------------------------------------------

    def save_all(self, fmt_vector: str = "geojson", verbose: bool = True) -> Dict[str, Path]:
        """Save all outputs; return a dict of {label: path}."""
        paths = {}
        paths.update(self.save_rasters(verbose=verbose))
        paths.update(self.save_vector(fmt=fmt_vector, verbose=verbose))
        paths.update(self.save_csv(verbose=verbose))
        paths.update(self.save_png_summary(verbose=verbose))
        paths.update(self.save_rgb_comparison(verbose=verbose))
        paths.update(self.save_building_stats(verbose=verbose))
        if verbose:
            print(f"\nAll outputs written to: {self.out_dir.resolve()}")
        return paths

    # ------------------------------------------------------------------
    # Rasters (GeoTIFF)
    # ------------------------------------------------------------------

    def save_rasters(self, verbose: bool = True) -> Dict[str, Path]:
        """Write all raster arrays as single-band GeoTIFFs.

        Returned dict keys: 'probability', 'confidence', 'stability',
        'pol_ratio', 'texture', 'sar_anomaly', 'optical', 'ndvi', 'ndbi',
        'cross_pol_entropy', 'coherence_proxy', 'seasonal_inv'.
        """
        r = self.result
        layers: Dict[str, tuple] = {
            "probability":       (r.probability,          "float32", -9999.0),
            "confidence":        (r.confidence,           "int8",    0),
            "stability":         (r.stability,            "float32", -9999.0),
            "pol_ratio":         (r.pol_ratio_norm,       "float32", -9999.0),
            "texture":           (r.texture,              "float32", -9999.0),
            "sar_anomaly":       (r.sar_anomaly,          "float32", -9999.0),
            "optical":           (r.optical_indicator,    "float32", -9999.0),
            "ndvi":              (r.ndvi,                 "float32", -9999.0),
            "ndbi":              (r.ndbi,                 "float32", -9999.0),
            "cross_pol_entropy": (r.cross_pol_entropy,    "float32", -9999.0),
            "coherence_proxy":   (r.coherence_proxy,      "float32", -9999.0),
            "seasonal_inv":      (r.seasonal_invariance,  "float32", -9999.0),
        }

        paths: Dict[str, Path] = {}
        for name, (arr, dtype, nodata) in layers.items():
            p = self._write_tiff(arr, name, dtype, nodata)
            paths[name] = p
            if verbose:
                print(f"  Saved raster : {p.name}")

        return paths

    def _write_tiff(
        self,
        arr: np.ndarray,
        name: str,
        dtype: str,
        nodata,
    ) -> Path:
        """Write a single 2-D array to a Cloud-Optimised GeoTIFF."""
        path = self.out_dir / f"{self.study_name}_{name}.tif"

        if dtype == "float32":
            out = np.where(np.isnan(arr), nodata, arr).astype(np.float32)
        elif dtype == "int8":
            out = arr.astype(np.int8)
        else:
            out = arr.astype(dtype)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=dtype,
            crs=CRS.from_wkt(self.result.crs_wkt),
            transform=self.result.transform,
            nodata=nodata,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        ) as dst:
            dst.write(out, 1)
            dst.update_tags(
                layer=name,
                study=self.study_name,
                generator="sub-canopy-detector v1.0",
            )
        return path

    # ------------------------------------------------------------------
    # Vector outputs
    # ------------------------------------------------------------------

    def save_vector(self, fmt: str = "geojson", verbose: bool = True) -> Dict[str, Path]:
        """Write raw footprints and regularised building polygons.

        Both layers are re-projected to WGS84 before saving.
        """
        paths: Dict[str, Path] = {}
        ext = "geojson" if fmt.lower() == "geojson" else "shp"
        driver = "GeoJSON" if ext == "geojson" else "ESRI Shapefile"

        # Raw footprints
        fp = self.result.footprints
        if not fp.empty:
            fp_wgs = fp.to_crs("EPSG:4326")
            path = self.out_dir / f"{self.study_name}_footprints.{ext}"
            fp_wgs.to_file(str(path), driver=driver)
            paths["footprints"] = path
            if verbose:
                print(f"  Saved vector : {path.name}  ({len(fp_wgs)} raw footprints)")

        # Regularised building footprints
        reg = self.result.regularized_footprints
        if not reg.empty:
            reg_wgs = reg.to_crs("EPSG:4326")
            path_b = self.out_dir / f"{self.study_name}_buildings.{ext}"
            reg_wgs.to_file(str(path_b), driver=driver)
            paths["buildings"] = path_b
            if verbose:
                print(f"  Saved vector : {path_b.name}  ({len(reg_wgs)} buildings)")
        elif verbose:
            print("  No building footprints to export.")

        return paths

    # ------------------------------------------------------------------
    # CSV table
    # ------------------------------------------------------------------

    def save_csv(self, verbose: bool = True) -> Dict[str, Path]:
        """Write footprint attribute tables as CSV (no geometry column)."""
        paths: Dict[str, Path] = {}

        # Raw footprints
        fp = self.result.footprints
        if not fp.empty:
            path = self.out_dir / f"{self.study_name}_footprints.csv"
            fp.drop(columns="geometry", errors="ignore").to_csv(str(path), index=False)
            paths["csv"] = path
            if verbose:
                print(f"  Saved CSV    : {path.name}")

        # Regularised building footprints
        reg = self.result.regularized_footprints
        if not reg.empty:
            path_b = self.out_dir / f"{self.study_name}_buildings.csv"
            reg.drop(columns="geometry", errors="ignore").to_csv(str(path_b), index=False)
            paths["buildings_csv"] = path_b
            if verbose:
                print(f"  Saved CSV    : {path_b.name}")

        return {"csv": path}

    # ------------------------------------------------------------------
    # PNG summary map
    # ------------------------------------------------------------------

    def _geo_to_pixel(self, geom):
        """Convert a shapely geometry's exterior coords to (col, row) arrays."""
        t = self.result.transform
        xs, ys = geom.exterior.xy
        cols = [(x - t.c) / t.a for x in xs]
        rows = [(y - t.f) / t.e for y in ys]
        return cols, rows

    def _overlay_polygons(
        self,
        ax,
        gdf: gpd.GeoDataFrame,
        edgecolor: str = "cyan",
        facecolor: str = "none",
        linewidth: float = 0.7,
        alpha: float = 0.85,
    ) -> None:
        """Draw polygon outlines from a GeoDataFrame onto a pixel-space axes."""
        if gdf.empty:
            return
        from matplotlib.patches import Polygon as MplPoly
        from matplotlib.collections import PatchCollection

        patches = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)  # type: ignore[attr-defined]
            for poly in polys:
                cols, rows = self._geo_to_pixel(poly)
                patches.append(MplPoly(list(zip(cols, rows)), closed=True))

        if patches:
            pc = PatchCollection(
                patches,
                edgecolors=edgecolor,
                facecolors=facecolor,
                linewidths=linewidth,
                alpha=alpha,
            )
            ax.add_collection(pc)

    def _overlay_filled_polygons(
        self,
        ax,
        gdf: gpd.GeoDataFrame,
        score_col: str = "building_score",
        cmap_name: str = "plasma",
        alpha: float = 0.80,
        linewidth: float = 0.5,
    ) -> None:
        """Draw filled polygons coloured by a numeric column."""
        if gdf.empty:
            return
        from matplotlib.patches import Polygon as MplPoly
        from matplotlib.collections import PatchCollection

        cmap = plt.get_cmap(cmap_name)
        vals = np.asarray(gdf[score_col].values, dtype=float)
        vmin, vmax = float(vals.min()), max(float(vals.max()), float(vals.min()) + 0.01)

        patches, colours = [], []
        for idx, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)  # type: ignore[attr-defined]
            for poly in polys:
                cols, rows = self._geo_to_pixel(poly)
                patches.append(MplPoly(list(zip(cols, rows)), closed=True))
                normed = (vals[idx] - vmin) / (vmax - vmin)
                colours.append(cmap(normed))

        if patches:
            pc = PatchCollection(
                patches,
                facecolors=colours,
                edgecolors="white",
                linewidths=linewidth,
                alpha=alpha,
            )
            ax.add_collection(pc)

    def save_png_summary(self, verbose: bool = True) -> Dict[str, Path]:
        """Save a 4x3 panel PNG with building-polygon overlays.

        Row 1: Probability + buildings | Confidence + buildings | Building Footprint Map
        Row 2: NDVI + outlines | SAR Anomaly + outlines | Texture + outlines
        Row 3: Cross-Pol Entropy | Coherence Proxy | Seasonal Invariance
        Row 4: Pol Ratio | Raw vs Regularised | Building Score (all with outlines)
        """
        r = self.result
        reg = r.regularized_footprints
        raw = r.footprints

        fig, axes = plt.subplots(4, 3, figsize=(18, 22))
        fig.suptitle(
            f"Sub-Canopy Structure Detector — {self.study_name}",
            fontsize=16, fontweight="bold", y=0.98,
        )

        # ---------- helper to show a raster panel -----------------------
        def _panel(ax, arr, title, cmap, vmin, vmax, overlay=True):
            im = ax.imshow(
                np.asarray(arr), cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if overlay:
                self._overlay_polygons(ax, reg, edgecolor="cyan", linewidth=0.8)

        # -------- Row 1: Core results -----------------------------------
        _panel(axes[0, 0], r.probability, "Detection Probability + Buildings",
               "YlOrRd", 0.0, 1.0, overlay=True)

        _panel(axes[0, 1], r.confidence, "Confidence Zones + Buildings",
               "RdYlGn", 0.0, 3.0, overlay=True)

        # Panel 3: Building footprint map on RGB
        ax_fp = axes[0, 2]
        ax_fp.imshow(r.rgb_composite, interpolation="nearest")
        self._overlay_filled_polygons(ax_fp, reg, score_col="building_score",
                                      cmap_name="plasma", alpha=0.55)
        self._overlay_polygons(ax_fp, reg, edgecolor="white", linewidth=0.8, alpha=0.95)
        n_bld = len(reg)
        ax_fp.set_title(f"Building Footprints ({n_bld}) on RGB", fontsize=10, fontweight="bold")
        ax_fp.axis("off")

        # -------- Row 2: Key indicators + outlines ----------------------
        _panel(axes[1, 0], r.ndvi, "NDVI + Buildings",
               "RdYlGn", -0.1, 1.0)

        _panel(axes[1, 1], r.sar_anomaly, "SAR Anomaly + Buildings",
               "PuRd", 0.0, 1.0)

        _panel(axes[1, 2], r.texture, "Texture + Buildings",
               "Blues", 0.0, 1.0)

        # -------- Row 3: Advanced SAR indicators + outlines -------------
        _panel(axes[2, 0], r.cross_pol_entropy, "Cross-Pol Entropy",
               "magma", 0.0, 1.0)

        _panel(axes[2, 1], r.coherence_proxy, "Coherence Proxy",
               "inferno", 0.0, 1.0)

        _panel(axes[2, 2], r.seasonal_invariance, "Seasonal Invariance",
               "cividis", 0.0, 1.0)

        # -------- Row 4: Pol Ratio + Raw vs Regularised + Score ---------
        _panel(axes[3, 0], r.pol_ratio_norm, "Pol Ratio",
               "viridis", 0.0, 1.0)

        # Panel 11: Raw blobs vs regularised buildings on RGB
        ax_cmp = axes[3, 1]
        ax_cmp.imshow(r.rgb_composite, interpolation="nearest")
        self._overlay_polygons(ax_cmp, raw, edgecolor="#ff6b6b",
                               linewidth=0.6, alpha=0.7)
        self._overlay_polygons(ax_cmp, reg, edgecolor="#00ffcc",
                               linewidth=0.8, alpha=0.9)
        ax_cmp.set_title("Raw (red) vs Regularised (cyan)", fontsize=10,
                         fontweight="bold")
        ax_cmp.axis("off")

        # Panel 12: Building score scatter-map on RGB
        ax_sc = axes[3, 2]
        ax_sc.imshow(r.rgb_composite, interpolation="nearest")
        if not reg.empty:
            t = r.transform
            for _, brow in reg.iterrows():
                geom = brow.geometry
                cent = geom.centroid
                cx = (cent.x - t.c) / t.a
                cy = (cent.y - t.f) / t.e
                sc = brow["building_score"]
                sz = 8 + 30 * sc
                ax_sc.scatter(
                    cx, cy, s=sz, c=sc, cmap="plasma",
                    vmin=0.3, vmax=1.0, edgecolors="white",
                    linewidths=0.3, zorder=3,
                )
            sm = plt.cm.ScalarMappable(
                cmap="plasma", norm=mcolors.Normalize(0.3, 1.0),
            )
            sm.set_array([])
            plt.colorbar(sm, ax=ax_sc, fraction=0.046, pad=0.04, label="Score")
        ax_sc.set_title("Building Score Map", fontsize=10, fontweight="bold")
        ax_sc.axis("off")

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        path = self.out_dir / f"{self.study_name}_summary.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        if verbose:
            print(f"  Saved PNG    : {path.name}")

        return {"summary_png": path}

    # ------------------------------------------------------------------
    # RGB imagery comparison
    # ------------------------------------------------------------------

    def save_rgb_comparison(self, verbose: bool = True) -> Dict[str, Path]:
        """Save side-by-side: pure RGB | RGB + building polygons.

        This is a standalone high-resolution figure (dpi=200) for visual
        inspection of polygon accuracy against actual satellite imagery.
        """
        r = self.result
        rgb = r.rgb_composite  # (H, W, 3) float32 0-1
        reg = r.regularized_footprints

        fig, (ax_rgb, ax_bld) = plt.subplots(1, 2, figsize=(22, 10))

        # Left: pure RGB true-colour
        ax_rgb.imshow(rgb, interpolation="nearest")
        ax_rgb.set_title("Sentinel-2 True Colour", fontsize=13, fontweight="bold")
        ax_rgb.axis("off")

        # Right: RGB + building polygons (solid red fill + red border)
        ax_bld.imshow(rgb, interpolation="nearest")
        self._overlay_polygons(
            ax_bld, reg,
            edgecolor="#cc0000",
            facecolor="#ff2222",
            linewidth=1.4,
            alpha=0.68,
        )
        n = len(reg)
        ax_bld.set_title(
            f"Building Footprints ({n}) on RGB",
            fontsize=13, fontweight="bold",
        )
        ax_bld.axis("off")

        fig.suptitle(
            f"RGB Imagery & Building Detection — {self.study_name}",
            fontsize=15, fontweight="bold", y=0.98,
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        path = self.out_dir / f"{self.study_name}_rgb_buildings.png"
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        plt.close(fig)

        if verbose:
            print(f"  Saved PNG    : {path.name}")

        return {"rgb_buildings": path}

    # ------------------------------------------------------------------
    # Building detection statistics dashboard
    # ------------------------------------------------------------------

    def save_building_stats(self, verbose: bool = True) -> Dict[str, Path]:
        """Save a 2×3 statistical dashboard for building detections."""
        r = self.result
        reg = r.regularized_footprints
        raw = r.footprints

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(
            f"Building Detection Statistics — {self.study_name}",
            fontsize=15, fontweight="bold", y=0.99,
        )

        # P1: Score histogram
        ax = axes[0, 0]
        if not reg.empty:
            scores = reg["building_score"].values
            ax.hist(scores, bins=20, color="#6366f1", edgecolor="white", alpha=0.85)
            ax.axvline(
                r.params.get("min_building_score", 0.35),
                color="red", linestyle="--", linewidth=1.5, label="Threshold",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("Building Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution", fontweight="bold")

        # P2: Area histogram (log-x)
        ax = axes[0, 1]
        if not reg.empty:
            areas = reg["area_m2"].values
            ax.hist(areas, bins=30, color="#f59e0b", edgecolor="white", alpha=0.85)
            ax.set_xscale("log")
        ax.set_xlabel("Area (m²)")
        ax.set_ylabel("Count")
        ax.set_title("Area Distribution", fontweight="bold")

        # P3: Compactness vs Rectangularity (color = score)
        ax = axes[0, 2]
        if not reg.empty:
            sc = ax.scatter(
                reg["rectangularity"], reg["compactness"],
                c=reg["building_score"], cmap="plasma", s=22,
                edgecolors="white", linewidths=0.3, alpha=0.85,
            )
            plt.colorbar(sc, ax=ax, label="Score")
        ax.set_xlabel("Rectangularity")
        ax.set_ylabel("Compactness")
        ax.set_title("Shape Quality", fontweight="bold")

        # P4: Score vs Area (color = solidity)
        ax = axes[1, 0]
        if not reg.empty and "solidity" in reg.columns:
            sc2 = ax.scatter(
                reg["area_m2"], reg["building_score"],
                c=reg["solidity"], cmap="viridis", s=22,
                edgecolors="white", linewidths=0.3, alpha=0.85,
            )
            sm = plt.cm.ScalarMappable(
                cmap="viridis", norm=mcolors.Normalize(0, 1),
            )
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Solidity")
            ax.set_xscale("log")
        ax.set_xlabel("Area (m²)")
        ax.set_ylabel("Building Score")
        ax.set_title("Score vs Area", fontweight="bold")

        # P5: Solidity distribution
        ax = axes[1, 1]
        if not reg.empty and "solidity" in reg.columns:
            ax.hist(
                reg["solidity"].values, bins=20, color="#10b981",
                edgecolor="white", alpha=0.85,
            )
        ax.set_xlabel("Solidity (area / convex hull)")
        ax.set_ylabel("Count")
        ax.set_title("Solidity Distribution", fontweight="bold")

        # P6: Filtering summary pie
        ax = axes[1, 2]
        n_raw = len(raw)
        n_bld = len(reg)
        n_rejected = n_raw - n_bld
        if n_raw > 0:
            labels = [f"Buildings\n({n_bld})", f"Rejected\n({n_rejected})"]
            sizes = [max(n_bld, 1), max(n_rejected, 1)]
            colours = ["#6366f1", "#ef4444"]
            ax.pie(
                sizes, labels=labels, colors=colours,
                autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 10},
            )
        ax.set_title("Filtering Summary", fontweight="bold")

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        path = self.out_dir / f"{self.study_name}_building_stats.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        if verbose:
            print(f"  Saved PNG    : {path.name}")

        return {"building_stats": path}