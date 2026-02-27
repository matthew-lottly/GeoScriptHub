"""
viz.py
======
Interactive and static visualisation helpers for Jupyter notebooks.

``ResultVisualiser`` renders:
  - An interactive folium map with raster overlays and clickable footprints
  - Matploltib SAR time-series chart (mean VV over AOI)
  - Matplotlib indicator breakdown panel (bar chart per zone)

All methods return displayable objects that Jupyter renders inline.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure as _Figure
from PIL import Image
import rasterio
from rasterio.crs import CRS
import geopandas as gpd
import folium
import folium.raster_layers
import branca.colormap as bc

from .analysis import AnalysisResult
from .aoi import AOIResult


def _arr_to_png_b64(
    arr: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    alpha: float = 0.7,
) -> str:
    """Convert a 2-D float array to a base64-encoded RGBA PNG string.

    Used by folium.ImageOverlay to display rasters on a Leaflet map.
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    # to_rgba with bytes=True returns an ndarray; wrap in np.array to
    # make the type explicit so the type checker accepts indexed assignment.
    rgba = np.array(mapper.to_rgba(arr, alpha=alpha, bytes=True), dtype=np.uint8)

    # Mask NaN pixels as fully transparent
    nan_mask = np.isnan(arr)
    rgba[nan_mask, 3] = 0

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ResultVisualiser:
    """Visualise analysis results inside a Jupyter notebook.

    Parameters
    ----------
    result:
        Completed ``AnalysisResult`` from ``SubCanopyAnalyser.run()``.
    aoi:
        Resolved AOI for coordinate reference.
    s1_stack:
        Optional raw S1 xarray DataArray for the time-series chart.
    """

    def __init__(
        self,
        result: AnalysisResult,
        aoi: AOIResult,
        s1_stack=None,
    ) -> None:
        self.result = result
        self.aoi = aoi
        self.s1 = s1_stack

    # ------------------------------------------------------------------
    # Interactive Folium map
    # ------------------------------------------------------------------

    def folium_map(
        self,
        show_probability: bool = True,
        show_confidence: bool = True,
        show_footprints: bool = True,
        tiles: str = "CartoDB positron",
    ) -> folium.Map:
        """Return an interactive Leaflet map with raster and vector overlays.

        The map is centred on the AOI and contains up to three toggleable
        layer groups, each with a layer-control checkbox.

        Parameters
        ----------
        show_probability:
            Add a semi-transparent probability heatmap.
        show_confidence:
            Add coloured confidence-zone overlay (red=high, orange=medium).
        show_footprints:
            Add footprint polygons as a GeoJSON vector layer.
        tiles:
            Basemap tile set name, passed directly to ``folium.Map``.
        """
        b = self.aoi.bbox_wgs84
        centre = [(b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0]
        m = folium.Map(location=centre, tiles=tiles, zoom_start=13)

        # Bounds for ImageOverlay: [[south, west], [north, east]]
        img_bounds = [[b[1], b[0]], [b[3], b[2]]]

        if show_probability:
            png_b64 = _arr_to_png_b64(
                self.result.probability, cmap="YlOrRd", vmin=0.0, vmax=1.0, alpha=0.65
            )
            url = f"data:image/png;base64,{png_b64}"
            fg = folium.FeatureGroup(name="Probability (0-1)", show=True)
            folium.raster_layers.ImageOverlay(
                image=url,
                bounds=img_bounds,
                opacity=0.7,
                name="Probability",
            ).add_to(fg)
            fg.add_to(m)

            # Add colour bar as a branca element
            colormap = bc.LinearColormap(
                ["#ffffcc", "#fd8d3c", "#bd0026"],
                vmin=0.0, vmax=1.0, caption="Sub-canopy probability"
            )
            colormap.add_to(m)

        if show_confidence:
            # Colour map: 1=yellow, 2=orange, 3=red
            conf_rgba = np.zeros((*self.result.confidence.shape, 4), dtype=np.uint8)
            conf_rgba[self.result.confidence == 2] = [253, 141, 60, 200]   # orange
            conf_rgba[self.result.confidence == 3] = [189,   0, 38, 220]   # dark red
            img = Image.fromarray(conf_rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            fg = folium.FeatureGroup(name="Confidence Zones", show=True)
            folium.raster_layers.ImageOverlay(
                image=url, bounds=img_bounds, opacity=0.75
            ).add_to(fg)
            fg.add_to(m)

        if show_footprints and not self.result.footprints.empty:
            fp_wgs84 = self.result.footprints.to_crs("EPSG:4326")
            fg = folium.FeatureGroup(name="Detected Footprints", show=True)
            folium.GeoJson(
                fp_wgs84.__geo_interface__,
                style_function=lambda f: {
                    "fillColor": self._prob_colour(f["properties"].get("prob_mean", 0)),
                    "color": "#1a1a2e",
                    "weight": 1,
                    "fillOpacity": 0.5,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["area_m2", "prob_mean", "prob_max"],
                    aliases=["Area (m2)", "Mean prob.", "Max prob."],
                ),
            ).add_to(fg)
            fg.add_to(m)

        # AOI boundary
        aoi_fg = folium.FeatureGroup(name="AOI Boundary", show=True)
        folium.GeoJson(
            self.aoi.gdf_wgs84.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "none", "color": "#3a86ff", "weight": 2
            },
        ).add_to(aoi_fg)
        aoi_fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m

    @staticmethod
    def _prob_colour(prob: float) -> str:
        """Map a probability value [0,1] to a hex colour."""
        if prob >= 0.65:
            return "#bd0026"
        if prob >= 0.45:
            return "#fd8d3c"
        return "#ffffcc"

    # ------------------------------------------------------------------
    # SAR time-series chart
    # ------------------------------------------------------------------

    def sar_timeseries_chart(
        self,
        high_only: bool = False,
        figsize=(12, 4),
    ) -> _Figure:
        """Plot mean VV backscatter (dB) over the AOI across the time series.

        Parameters
        ----------
        high_only:
            If True, restrict the spatial mean to HIGH-confidence pixels only.
            If no high-confidence pixels exist, falls back to the full AOI.
        figsize:
            Matplotlib figure size tuple.
        """
        if self.s1 is None:
            fig_empty, ax_empty = plt.subplots(figsize=figsize)
            ax_empty.text(0.5, 0.5, "S1 stack not available", ha="center",
                          transform=ax_empty.transAxes)
            return fig_empty  # type: ignore[return-value]

        vv_band = next(
            (b for b in self.s1.band.values if "vv" in str(b).lower()), None
        )
        if vv_band is None:
            raise ValueError("No VV band found in S1 stack.")

        vv_stack = self.s1.sel(band=vv_band).values.astype(np.float32)  # (time, y, x)
        times    = self.s1.time.values

        if high_only and np.any(self.result.confidence == 3):
            mask = (self.result.confidence == 3)[np.newaxis, :, :]
            label_full = "Mean VV -- full AOI"
            label_high = "Mean VV -- HIGH zones only"
            show_high = True
        else:
            mask = np.ones((1, *vv_stack.shape[1:]), dtype=bool)
            label_full = "Mean VV (dB)"
            show_high = False

        vv_db = 10.0 * np.log10(np.where(vv_stack > 0, vv_stack, np.nan))

        def _scene_mean(db_stack, spatial_mask):
            masked = np.where(spatial_mask, db_stack, np.nan)
            return np.nanmean(masked.reshape(db_stack.shape[0], -1), axis=1)

        full_means = _scene_mean(vv_db, np.ones_like(vv_stack[0], dtype=bool))

        import pandas as pd
        ts = pd.Series(full_means, index=pd.DatetimeIndex(times)).sort_index()
        ts_smooth = ts.rolling(window=5, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(ts.index, np.asarray(ts.values), color="#3b528b", s=12, zorder=3, label="Individual scenes")
        ax.plot(ts_smooth.index, np.asarray(ts_smooth.values), color="#3b528b", linewidth=1.5,
                label=label_full, zorder=4)

        if show_high:
            high_means = _scene_mean(vv_db, self.result.confidence == 3)
            ts_high = pd.Series(high_means, index=pd.DatetimeIndex(times)).sort_index()
            ts_high_smooth = ts_high.rolling(window=5, center=True, min_periods=1).mean()
            ax.plot(ts_high_smooth.index, np.asarray(ts_high_smooth.values),
                    color="#d73027", linewidth=1.5, label=label_high)

        ax.set_xlabel("Date")
        ax.set_ylabel("Mean VV Backscatter (dB)")
        ax.set_title("Sentinel-1 VV Temporal Profile")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Indicator summary panel
    # ------------------------------------------------------------------

    def indicator_panel(self, figsize=(14, 10)) -> _Figure:
        """6-panel matplotlib figure showing each normalised indicator."""
        r = self.result
        panels = [
            (r.stability,        "Stability",           "Blues",   0, 1),
            (r.pol_ratio_norm,   "Pol Ratio (VH/VV)",   "viridis", 0, 1),
            (r.texture,          "SAR Texture",         "Greys_r", 0, 1),
            (r.sar_anomaly,      "SAR Anomaly",         "PuRd",    0, 1),
            (r.optical_indicator,"Optical Indicator",   "YlOrBr",  0, 1),
            (r.probability,      "Fused Probability",   "YlOrRd",  0, 1),
        ]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Sub-Canopy Detection -- Indicator Breakdown", fontsize=13)

        for ax, (arr, title, cmap, vmin, vmax) in zip(axes.flat, panels):
            masked = np.ma.masked_invalid(arr)
            im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Probability histogram
    # ------------------------------------------------------------------

    def probability_histogram(self, figsize=(8, 4)) -> _Figure:
        """Histogram of per-pixel probability values within the forest mask."""
        prob = self.result.probability
        valid = prob[~np.isnan(prob)].ravel()

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(valid, bins=50, color="#3b528b", edgecolor="white", linewidth=0.3)
        ax.axvline(self.result.params["thresh_medium"], color="#fd8d3c",
                   linestyle="--", linewidth=1.5, label=f"MEDIUM ({self.result.params['thresh_medium']})")
        ax.axvline(self.result.params["thresh_high"], color="#bd0026",
                   linestyle="--", linewidth=1.5, label=f"HIGH ({self.result.params['thresh_high']})")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Pixel Count")
        ax.set_title("Distribution of Sub-Canopy Probability Scores")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig
