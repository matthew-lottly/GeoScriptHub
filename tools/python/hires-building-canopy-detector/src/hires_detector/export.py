"""
export.py
=========
Output generation: GeoTIFF rasters, GeoJSON vectors, multi-panel PNG
summaries, overlay composites, and a statistics dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio.transform import Affine

from .analysis import HiResResult


class HiResOutputWriter:
    """Persist analysis results as rasters, vectors, and figures.

    Parameters
    ----------
    result : The completed ``HiResResult`` from ``HiResAnalyser.run()``.
    output_dir : Directory into which all outputs are written.
    """

    def __init__(self, result: HiResResult, output_dir: str | Path) -> None:
        self.r = result
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def write_all(self, verbose: bool = True) -> None:
        """Write every output type."""
        if verbose:
            print(f"\nWriting outputs to {self.out} …")
        self.write_geotiffs(verbose)
        self.write_vectors(verbose)
        self.save_summary_png(verbose)
        self.save_building_overlay(verbose)
        self.save_canopy_overlay(verbose)
        self.save_species_map(verbose)
        self.save_stats_dashboard(verbose)

    # ==================================================================
    # GeoTIFFs
    # ==================================================================

    def write_geotiffs(self, verbose: bool = True) -> None:
        """Write analysis rasters as GeoTIFF files."""
        layers = [
            ("sar_db",          self.r.sar_db),
            ("sar_despeckled",  self.r.sar_despeckled),
            ("mbi",             self.r.mbi),
            ("sar_contrast",    self.r.sar_contrast),
            ("sar_edges",       self.r.sar_edges),
            ("ndvi",            self.r.ndvi),
            ("building_score",  self.r.building_score),
            ("canopy_mask",     self.r.canopy_mask.astype(np.float32)),
            ("species_map",     self.r.species_map.astype(np.float32)),
        ]
        for name, arr in layers:
            path = self.out / f"{name}.tif"
            self._write_tiff(path, arr)
            if verbose:
                print(f"  GeoTIFF : {path.name}")

    def _write_tiff(self, path: Path, arr: np.ndarray) -> None:
        H, W = arr.shape[:2]
        n_bands = arr.shape[2] if arr.ndim == 3 else 1
        profile = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": n_bands,
            "dtype": str(arr.dtype),
            "crs": self.r.crs_wkt,
            "transform": self.r.transform,
            "compress": "deflate",
        }
        with rasterio.open(str(path), "w", **profile) as dst:
            if n_bands == 1:
                dst.write(arr, 1)
            else:
                for b in range(n_bands):
                    dst.write(arr[:, :, b], b + 1)

    # ==================================================================
    # Vectors
    # ==================================================================

    def write_vectors(self, verbose: bool = True) -> None:
        """Write vector layers as GeoJSON."""
        for name, gdf in [
            ("building_footprints", self.r.building_footprints),
            ("tree_crowns",         self.r.tree_crowns),
            ("species_crowns",      self.r.species_crowns),
        ]:
            path = self.out / f"{name}.geojson"
            if not gdf.empty:
                gdf.to_file(str(path), driver="GeoJSON")
            if verbose:
                print(f"  Vector  : {path.name}  ({len(gdf)} features)")

    # ==================================================================
    # Summary panel PNG (3 × 4)
    # ==================================================================

    def save_summary_png(self, verbose: bool = True) -> None:
        """12-panel overview of every analysis layer."""
        fig, axes = plt.subplots(3, 4, figsize=(26, 18))
        fig.suptitle(
            "Hi-Res Building + Canopy Detection — Summary",
            fontsize=17, fontweight="bold",
        )

        panels = [
            (0, 0, self.r.optical_rgb,                    "Optical RGB",      None),
            (0, 1, self.r.sar_rgb,                        "SAR Pseudo-RGB",   None),
            (0, 2, self.r.sar_despeckled,                 "SAR Lee-filtered", "gray"),
            (0, 3, self.r.mbi,                            "MBI",              "hot"),
            (1, 0, self.r.sar_contrast,                   "SAR Contrast",     "inferno"),
            (1, 1, self.r.sar_edges,                      "Edge Density",     "magma"),
            (1, 2, self.r.building_score,                 "Building Score",   "YlOrRd"),
            (1, 3, self.r.building_mask.astype(float),    "Building Mask",    "Reds"),
            (2, 0, self.r.ndvi,                           "NDVI",             "RdYlGn"),
            (2, 1, self.r.canopy_mask.astype(float),      "Canopy Mask",      "Greens"),
            (2, 2, self.r.crown_labels.astype(float),     "Tree Crowns",      "nipy_spectral"),
            (2, 3, self.r.species_map.astype(float),      "Species Map",      "tab10"),
        ]

        for row, col, data, title, cmap in panels:
            ax = axes[row, col]
            if data.ndim == 3:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        path = self.out / "summary_panels.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  PNG     : {path.name}")

    # ==================================================================
    # Building overlay on optical + SAR
    # ==================================================================

    def save_building_overlay(self, verbose: bool = True) -> None:
        """Side-by-side: buildings on optical RGB and SAR pseudo-RGB."""
        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        for idx, (base, base_title, fill_colour) in enumerate([
            (self.r.optical_rgb, "Buildings on Optical", "red"),
            (self.r.sar_rgb,     "Buildings on SAR",     "cyan"),
        ]):
            ax = axes[idx]
            ax.imshow(base)
            for _, row in self.r.building_footprints.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                xs, ys = geom.exterior.xy
                pxs = [(x - self.r.transform.c) / self.r.transform.a for x in xs]
                pys = [(y - self.r.transform.f) / self.r.transform.e for y in ys]
                ax.fill(pxs, pys, alpha=0.35, fc=fill_colour, ec=fill_colour, lw=1.5)
            n = len(self.r.building_footprints)
            ax.set_title(f"{base_title}  ({n} buildings)", fontsize=13, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        path = self.out / "building_overlay.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  PNG     : {path.name}")

    # ==================================================================
    # Canopy overlay
    # ==================================================================

    def save_canopy_overlay(self, verbose: bool = True) -> None:
        """Left: canopy highlight on optical.  Right: crown outlines."""
        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        # Left — semitransparent green canopy overlay
        overlay = self.r.optical_rgb.copy()
        green = np.zeros_like(overlay)
        green[:, :, 1] = 1.0
        mask = self.r.canopy_mask
        overlay[mask] = 0.6 * overlay[mask] + 0.4 * green[mask]
        axes[0].imshow(overlay)
        pct = 100 * mask.sum() / max(mask.size, 1)
        axes[0].set_title(
            f"Canopy Cover: {pct:.1f}%", fontsize=13, fontweight="bold",
        )
        axes[0].axis("off")

        # Right — crown outlines on optical
        axes[1].imshow(self.r.optical_rgb)
        for _, row in self.r.tree_crowns.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            xs, ys = geom.exterior.xy
            pxs = [(x - self.r.transform.c) / self.r.transform.a for x in xs]
            pys = [(y - self.r.transform.f) / self.r.transform.e for y in ys]
            axes[1].plot(pxs, pys, color="lime", lw=0.8)
        n = len(self.r.tree_crowns)
        axes[1].set_title(
            f"Individual Tree Crowns: {n:,}", fontsize=13, fontweight="bold",
        )
        axes[1].axis("off")

        plt.tight_layout()
        path = self.out / "canopy_overlay.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  PNG     : {path.name}")

    # ==================================================================
    # Species classification map
    # ==================================================================

    def save_species_map(self, verbose: bool = True) -> None:
        """Species coloured overlay + crown outlines by species."""
        if not self.r.species_legend:
            return

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))
        n_sp = max(len(self.r.species_legend), 1)
        _cmap = plt.colormaps.get_cmap("Set2")
        colours = _cmap(np.linspace(0, 1, n_sp))

        # Left — raster overlay
        axes[0].imshow(self.r.optical_rgb)
        for idx, (sp_id, _sp_name) in enumerate(sorted(self.r.species_legend.items())):
            sp_mask = self.r.species_map == sp_id
            if not sp_mask.any():
                continue
            rgba = np.zeros((*self.r.optical_rgb.shape[:2], 4), dtype=np.float32)
            rgba[:, :, :3] = colours[idx % n_sp][:3]
            rgba[:, :, 3]  = sp_mask.astype(np.float32) * 0.50
            axes[0].imshow(rgba)
        patches = [
            Patch(facecolor=colours[i % n_sp][:3], label=f"{k}: {v}")
            for i, (k, v) in enumerate(sorted(self.r.species_legend.items()))
        ]
        axes[0].legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.85)
        axes[0].set_title("Species Classification (raster)", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        # Right — crown polygons coloured by species
        axes[1].imshow(self.r.optical_rgb)
        for _, row in self.r.species_crowns.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            sp_id = row.get("species_id", 0)
            clr = colours[(sp_id - 1) % n_sp][:3] if sp_id > 0 else "gray"
            xs, ys = geom.exterior.xy
            pxs = [(x - self.r.transform.c) / self.r.transform.a for x in xs]
            pys = [(y - self.r.transform.f) / self.r.transform.e for y in ys]
            axes[1].fill(pxs, pys, alpha=0.35, fc=clr, ec=clr, lw=0.8)
        axes[1].set_title("Species Crowns (vector)", fontsize=13, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        path = self.out / "species_map.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  PNG     : {path.name}")

    # ==================================================================
    # Statistics dashboard
    # ==================================================================

    def save_stats_dashboard(self, verbose: bool = True) -> None:
        """Six-panel statistics dashboard."""
        r = self.r
        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "Analysis Statistics Dashboard",
            fontsize=17, fontweight="bold",
        )

        # 1. Building area histogram
        ax = axes[0, 0]
        if not r.building_footprints.empty:
            areas = r.building_footprints["area_m2"]
            ax.hist(areas, bins=30, color="steelblue", edgecolor="white")
            ax.set_xlabel("Building Area (m²)")
            ax.set_ylabel("Count")
        ax.set_title(f"Building Size (n={len(r.building_footprints)})")

        # 2. Building score histogram
        ax = axes[0, 1]
        if not r.building_footprints.empty:
            ax.hist(
                r.building_footprints["building_score"],
                bins=20, color="coral", edgecolor="white",
            )
            ax.set_xlabel("Building Score")
            ax.set_ylabel("Count")
        ax.set_title("Building Score Distribution")

        # 3. NDVI histogram
        ax = axes[0, 2]
        ndvi_valid = r.ndvi[np.isfinite(r.ndvi)].ravel()
        ax.hist(ndvi_valid, bins=50, color="forestgreen", edgecolor="white", alpha=0.85)
        ax.axvline(
            r.params.get("ndvi_threshold", 0.3),
            color="red", ls="--", lw=1.5, label="Threshold",
        )
        ax.set_xlabel("NDVI")
        ax.set_ylabel("Count")
        ax.set_title("NDVI Distribution")
        ax.legend()

        # 4. Crown area distribution
        ax = axes[1, 0]
        if not r.tree_crowns.empty:
            ax.hist(
                r.tree_crowns["area_m2"],
                bins=30, color="olive", edgecolor="white",
            )
            ax.set_xlabel("Crown Area (m²)")
            ax.set_ylabel("Count")
        ax.set_title(f"Crown Size (n={len(r.tree_crowns)})")

        # 5. Species pie chart
        ax = axes[1, 1]
        if r.species_legend and not r.species_crowns.empty:
            sp_counts = {}
            for sp_id, sp_name in r.species_legend.items():
                cnt = int((r.species_crowns["species_id"] == sp_id).sum())
                if cnt > 0:
                    sp_counts[sp_name] = cnt
            if sp_counts:
                ax.pie(
                    sp_counts.values(),
                    labels=sp_counts.keys(),
                    autopct="%1.0f%%",
                    colors=plt.colormaps.get_cmap("Set2").colors,
                )
        ax.set_title("Species Group Distribution")

        # 6. Land-cover summary bar
        ax = axes[1, 2]
        total = max(r.building_mask.size, 1)
        bldg_pct   = 100 * r.building_mask.sum()  / total
        canopy_pct = 100 * r.canopy_mask.sum()     / total
        other_pct  = max(100 - bldg_pct - canopy_pct, 0)
        cats   = ["Buildings", "Canopy", "Other"]
        vals   = [bldg_pct, canopy_pct, other_pct]
        colors = ["#e74c3c", "#2ecc71", "#95a5a6"]
        ax.bar(cats, vals, color=colors, edgecolor="white")
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
        ax.set_ylabel("% of AOI")
        ax.set_title("Land Cover Summary")
        ax.set_ylim(0, max(vals) * 1.25 + 5)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        path = self.out / "stats_dashboard.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  PNG     : {path.name}")
