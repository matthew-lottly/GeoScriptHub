"""
viz.py
======
Cartographic visualisation for flood frequency results.

Produces publication-quality maps:

1. **Flood Frequency Map** — continuous colour ramp from yellow
   (rarely flooded) through orange → red → dark blue (permanently
   inundated), with the study-area boundary, scale bar, and north
   arrow.

2. **FEMA Comparison Map** — same frequency surface with FEMA NFHL
   flood zone polygons overlaid at 40 % transparency for direct
   regulatory-vs-observed comparison.

3. **Interactive Folium Map** — web-based Leaflet map with frequency
   tile layer, FEMA overlay toggle, and click-for-stats popups.

4. **Observation Histogram** — bar chart of observations per sensor
   and temporal coverage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import folium
    from folium.raster_layers import ImageOverlay  # type: ignore[import-untyped]
    from folium.plugins import FloatImage
    from branca.colormap import LinearColormap
except ImportError:
    folium = None  # type: ignore[assignment]
    ImageOverlay = None  # type: ignore[assignment, misc]

import geopandas as gpd

from .flood_engine import FrequencyResult
from .fema import FEMAFloodZones, FEMA_COLORS, FEMA_DEFAULT_COLOR
from .aoi import AOIResult

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.viz")


# ---------------------------------------------------------------------------
# Custom colour map — scientific flood frequency palette
# ---------------------------------------------------------------------------

def _flood_frequency_cmap() -> mcolors.LinearSegmentedColormap:
    """Build a perceptually-uniform flood frequency colour ramp.

    Palette progression:
        0.00 → tan/beige  (dry land, never flooded)
        0.05 → pale yellow (very rarely flooded)
        0.15 → gold        (rarely flooded)
        0.30 → orange      (occasionally flooded)
        0.50 → red-orange  (frequently flooded)
        0.70 → crimson     (very frequently flooded)
        0.90 → dark blue   (almost always water)
        1.00 → navy        (permanent water)
    """
    colours = [
        (0.00, "#F5DEB3"),  # wheat — dry
        (0.05, "#FFF176"),  # pale yellow
        (0.15, "#FFD54F"),  # gold
        (0.30, "#FF9800"),  # orange
        (0.50, "#F44336"),  # red
        (0.70, "#C62828"),  # dark red
        (0.90, "#1565C0"),  # blue
        (1.00, "#0D47A1"),  # navy
    ]
    positions = [c[0] for c in colours]
    hex_colors = [c[1] for c in colours]

    # Convert hex to RGB tuples
    rgb = [mcolors.to_rgb(h) for h in hex_colors]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "flood_frequency",
        list(zip(positions, rgb)),
        N=256,
    )
    cmap.set_bad(color="white", alpha=0.0)  # NaN = transparent
    return cmap


FLOOD_CMAP = _flood_frequency_cmap()


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------

class FloodMapper:
    """Generate cartographic outputs from flood frequency results.

    Parameters
    ----------
    result:
        Computed FrequencyResult.
    aoi:
        Study-area AOI.
    fema:
        Optional FEMAFloodZones (if available).
    output_dir:
        Directory for saving output images and HTML.
    """

    def __init__(
        self,
        result: FrequencyResult,
        aoi: AOIResult,
        fema: Optional[FEMAFloodZones] = None,
        output_dir: Path = Path("outputs/quantum_flood_frequency"),
    ) -> None:
        self.result = result
        self.aoi = aoi
        self.fema = fema
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Flood Frequency Map
    # ------------------------------------------------------------------

    def plot_frequency_map(self, dpi: int = 200) -> Path:
        """Render the flood frequency map as a PNG.

        Returns:
            Path to the saved PNG file.
        """
        logger.info("Rendering flood frequency map …")

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Plot frequency surface
        extent = (
            self.result.bounds[0],  # west
            self.result.bounds[2],  # east
            self.result.bounds[1],  # south
            self.result.bounds[3],  # north
        )

        freq_display = np.ma.masked_invalid(self.result.frequency)

        im = ax.imshow(
            freq_display,
            extent=extent,
            origin="upper",
            cmap=FLOOD_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        # Colour bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, label="Inundation Frequency")
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

        # Title and labels
        ax.set_title(
            "Pseudo-Quantum Hybrid AI Flood Frequency Map\n"
            f"Mississippi River — {self.aoi.description}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Easting (m)", fontsize=10)
        ax.set_ylabel("Northing (m)", fontsize=10)

        # Legend for zone categories
        legend_patches = [
            mpatches.Patch(facecolor="#0D47A1", label=f"Permanent (≥{90}%)"),
            mpatches.Patch(facecolor="#C62828", label=f"Seasonal ({25}–{90}%)"),
            mpatches.Patch(facecolor="#FF9800", label=f"Rare ({5}–{25}%)"),
            mpatches.Patch(facecolor="#F5DEB3", label=f"Dry (<{5}%)"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="lower left",
            fontsize=8,
            framealpha=0.9,
            title="Flood Zones",
        )

        # Metadata annotation
        n_obs = int(self.result.observation_count.max())
        sensors = self.result.sensor_counts
        ax.annotate(
            f"Observations: {n_obs}  |  Sensors: {sensors}\n"
            f"CRS: {self.result.crs}  |  Resolution: 30 m\n"
            f"Method: QIEC (Quantum-Inspired Ensemble Classification)",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        # North arrow
        ax.annotate(
            "N", xy=(0.97, 0.97), xycoords="axes fraction",
            fontsize=14, fontweight="bold", ha="center", va="top",
        )
        ax.annotate(
            "↑", xy=(0.97, 0.93), xycoords="axes fraction",
            fontsize=18, ha="center", va="top",
        )

        fig.tight_layout()
        out_path = self.output_dir / "flood_frequency_map.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info("Frequency map saved → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # 2. FEMA Comparison Map
    # ------------------------------------------------------------------

    def plot_fema_comparison(self, dpi: int = 200) -> Path:
        """Render the frequency map with FEMA flood zones overlaid.

        FEMA zones are drawn as semi-transparent hatched polygons on top
        of the frequency raster for easy visual comparison.

        Returns:
            Path to the saved PNG file.
        """
        logger.info("Rendering FEMA comparison map …")

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        extent = (
            self.result.bounds[0],
            self.result.bounds[2],
            self.result.bounds[1],
            self.result.bounds[3],
        )

        freq_display = np.ma.masked_invalid(self.result.frequency)

        im = ax.imshow(
            freq_display,
            extent=extent,
            origin="upper",
            cmap=FLOOD_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        # Overlay FEMA flood zones
        fema_gdf = self.fema.zones if self.fema else None
        fema_legend_patches: list[mpatches.Patch] = []

        if fema_gdf is not None and not fema_gdf.empty and "FLD_ZONE" in fema_gdf.columns:
            for zone_code in fema_gdf["FLD_ZONE"].unique():
                subset = fema_gdf[fema_gdf["FLD_ZONE"] == zone_code]
                color = FEMAFloodZones.get_zone_color(zone_code)

                subset.plot(
                    ax=ax,
                    facecolor=color[:3],
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=color[3],  # semi-transparent
                    hatch="///",
                )

                fema_legend_patches.append(
                    mpatches.Patch(
                        facecolor=color[:3],
                        alpha=color[3],
                        edgecolor="black",
                        linewidth=0.5,
                        label=f"FEMA Zone {zone_code}",
                        hatch="///",
                    )
                )
        else:
            ax.annotate(
                "FEMA flood zone data unavailable for this area",
                xy=(0.5, 0.5), xycoords="axes fraction",
                fontsize=12, ha="center", va="center",
                color="red", alpha=0.7,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            )

        # Colour bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, label="Inundation Frequency")
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

        # Title
        ax.set_title(
            "Flood Frequency vs. FEMA National Flood Hazard Layer\n"
            f"Mississippi River — {self.aoi.description}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Easting (m)", fontsize=10)
        ax.set_ylabel("Northing (m)", fontsize=10)

        # Combined legend
        freq_patches = [
            mpatches.Patch(facecolor="#0D47A1", label="Permanent water (≥90%)"),
            mpatches.Patch(facecolor="#FF9800", label="Seasonal/rare flood"),
            mpatches.Patch(facecolor="#F5DEB3", label="Dry land"),
        ]
        all_patches = freq_patches + fema_legend_patches
        ax.legend(
            handles=all_patches,
            loc="lower left",
            fontsize=7,
            framealpha=0.9,
            title="Layers",
            ncol=2,
        )

        # Metadata
        ax.annotate(
            "Frequency: QIEC hybrid model  |  Overlay: FEMA NFHL (hatched, 40% opacity)\n"
            f"CRS: {self.result.crs}  |  Resolution: 30 m",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        fig.tight_layout()
        out_path = self.output_dir / "fema_comparison_map.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info("FEMA comparison map saved → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # 3. Interactive Folium Map
    # ------------------------------------------------------------------

    def create_interactive_map(self) -> Path:
        """Build an interactive Leaflet map via folium.

        Returns:
            Path to the saved HTML file.
        """
        if folium is None:
            logger.warning("folium not installed — skipping interactive map")
            return self.output_dir / "interactive_map.html"

        logger.info("Building interactive folium map …")

        center = [self.aoi.center_lat, self.aoi.center_lon]
        m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

        # Add satellite tile layer
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            name="Satellite",
            attr="Esri",
        ).add_to(m)

        # Frequency raster as image overlay
        # Convert frequency to RGBA image
        freq = self.result.frequency.copy()
        freq_norm = np.nan_to_num(freq, nan=-1.0)

        rgba = FLOOD_CMAP(np.clip(freq_norm, 0, 1))
        rgba[freq_norm < 0] = [1, 1, 1, 0]  # transparent for NaN

        # Save temporary PNG for overlay
        from PIL import Image
        img_array = (rgba * 255).astype("uint8")
        img = Image.fromarray(img_array)
        temp_img_path = self.output_dir / "_freq_overlay.png"
        img.save(temp_img_path)

        # Bounds for overlay
        west, south, east, north = self.aoi.bbox_wgs84
        bounds = [[south, west], [north, east]]

        ImageOverlay(  # type: ignore[misc]
            image=str(temp_img_path),
            bounds=bounds,
            name="Flood Frequency",
            opacity=0.7,
            interactive=True,
        ).add_to(m)

        # FEMA overlay if available
        fema_gdf = self.fema.zones if self.fema else None
        if fema_gdf is not None and not fema_gdf.empty:
            fema_wgs84 = fema_gdf.to_crs("EPSG:4326")

            def style_fema(feature: dict) -> dict:
                zone = feature.get("properties", {}).get("FLD_ZONE", "")
                color_rgba = FEMAFloodZones.get_zone_color(zone)
                hex_color = mcolors.to_hex(color_rgba[:3])
                return {
                    "fillColor": hex_color,
                    "fillOpacity": 0.35,
                    "color": "#333",
                    "weight": 1,
                }

            folium.GeoJson(
                fema_wgs84.__geo_interface__,
                name="FEMA Flood Zones",
                style_function=style_fema,
                tooltip=folium.GeoJsonTooltip(
                    fields=["FLD_ZONE", "risk_level"],
                    aliases=["FEMA Zone", "Risk Level"],
                ),
            ).add_to(m)

        # Colour legend
        colormap = LinearColormap(
            colors=["#F5DEB3", "#FFF176", "#FFD54F", "#FF9800",
                     "#F44336", "#C62828", "#1565C0", "#0D47A1"],
            vmin=0, vmax=100,
            caption="Inundation Frequency (%)",
        )
        colormap.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        out_path = self.output_dir / "interactive_flood_map.html"
        m.save(str(out_path))

        logger.info("Interactive map saved → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # 4. Observation statistics chart
    # ------------------------------------------------------------------

    def plot_observation_stats(self, dpi: int = 150) -> Path:
        """Bar chart of observations per sensor + summary statistics.

        Returns:
            Path to the saved PNG.
        """
        logger.info("Rendering observation statistics chart …")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: observations per sensor
        sensors = self.result.sensor_counts
        names = list(sensors.keys())
        counts = list(sensors.values())
        colors_bar = ["#4CAF50", "#2196F3", "#FF9800"][:len(names)]

        axes[0].bar(names, counts, color=colors_bar, edgecolor="black", linewidth=0.5)
        axes[0].set_title("Observations per Sensor", fontweight="bold")
        axes[0].set_ylabel("Scene Count")
        for i, v in enumerate(counts):
            axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

        # Right: frequency histogram
        freq_valid = self.result.frequency[~np.isnan(self.result.frequency)]
        axes[1].hist(
            freq_valid.ravel(), bins=50, color="#1565C0", edgecolor="white",
            linewidth=0.3, alpha=0.85,
        )
        axes[1].axvline(0.05, color="orange", linestyle="--", linewidth=1, label="5% (rare)")
        axes[1].axvline(0.25, color="red", linestyle="--", linewidth=1, label="25% (seasonal)")
        axes[1].axvline(0.90, color="navy", linestyle="--", linewidth=1, label="90% (permanent)")
        axes[1].set_title("Flood Frequency Distribution", fontweight="bold")
        axes[1].set_xlabel("Inundation Frequency")
        axes[1].set_ylabel("Pixel Count")
        axes[1].legend(fontsize=8)

        fig.suptitle(
            "Quantum-Hybrid Flood Frequency Analysis — Observation Summary",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()

        out_path = self.output_dir / "observation_stats.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info("Statistics chart saved → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Convenience: generate all outputs
    # ------------------------------------------------------------------

    def generate_all(self) -> dict[str, Path]:
        """Generate all visualisation products.

        Returns:
            Dict mapping output name to file path.
        """
        outputs = {}
        outputs["frequency_map"] = self.plot_frequency_map()
        outputs["fema_comparison"] = self.plot_fema_comparison()
        outputs["interactive_map"] = self.create_interactive_map()
        outputs["observation_stats"] = self.plot_observation_stats()

        logger.info("All visualisations generated: %s", list(outputs.keys()))
        return outputs
