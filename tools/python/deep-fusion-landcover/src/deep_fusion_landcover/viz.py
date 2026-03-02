"""viz.py — Interactive visualization suite for the Austin Deep Fusion Landcover pipeline.

Outputs
-------
interactive_landcover_map.html  — Folium DualMap (1990 left, 2025 right).
                                   Toggle layers: SAR, NDVI, sub-canopy, hotspots.
                                   Click popup: pixel class history + confidence.
landcover_sankey_{a}_{b}.html  — D3/Plotly Sankey for 5-epoch class transitions.
ndvi_timeseries.html            — Per-class annual NDVI Plotly chart.
urban_expansion.html            — Logistic growth chart.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    from folium.raster_layers import ImageOverlay as FoliumImageOverlay
except ImportError:
    FoliumImageOverlay = None  # type: ignore[assignment,misc]

from .constants import CLASS_COLORS, CLASS_NAMES, AUSTIN_CENTER_LAT, AUSTIN_CENTER_LON

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.viz")

# ── Colour palette helpers ────────────────────────────────────────────────────

def _class_colormap() -> dict[int, str]:
    """Return {1-indexed class id → hex colour}."""
    return {i + 1: c for i, c in enumerate(CLASS_COLORS)}


def _rgba(hex_colour: str, alpha: float = 0.7) -> tuple[int, int, int, int]:
    """Convert hex colour + alpha to RGBA tuple."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, int(alpha * 255))


# ── Map image encoder ─────────────────────────────────────────────────────────

def _class_map_to_png_bytes(
    class_map: np.ndarray,
    colormap: dict[int, str],
) -> bytes:
    """Render a uint8 class map to PNG bytes for Folium ImageOverlay."""
    from PIL import Image

    H, W = class_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for cls_id, hex_col in colormap.items():
        mask = class_map == cls_id
        r, g, b, a = _rgba(hex_col)
        rgba[mask] = [r, g, b, a]

    import io
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _png_to_base64(png_bytes: bytes) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


# ── Folium dual map ───────────────────────────────────────────────────────────

def build_dual_map(
    map_1990: np.ndarray,
    map_2025: np.ndarray,
    aoi_bounds_wgs84: tuple[float, float, float, float],
    hotspot_map: Optional[np.ndarray] = None,
    sub_canopy_map: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Build a Folium SideBySideLayers dual map.

    Parameters
    ----------
    map_1990, map_2025: (H, W) Int8 class maps (1-indexed).
    aoi_bounds_wgs84:   (min_lon, min_lat, max_lon, max_lat).
    hotspot_map:        Optional (H, W) bool hotspot layer.
    sub_canopy_map:     Optional (H, W) int8 sub-canopy label layer.
    output_path:        Save HTML here.

    Returns
    -------
    HTML string.
    """
    try:
        import folium
        from folium.plugins import SideBySideLayers
    except ImportError:
        logger.error("folium not installed. Run `pip install folium`.")
        return ""

    min_lon, min_lat, max_lon, max_lat = aoi_bounds_wgs84
    cmap = _class_colormap()

    m = folium.Map(
        location=[AUSTIN_CENTER_LAT, AUSTIN_CENTER_LON],
        zoom_start=10,
        tiles=None,
    )

    folium.TileLayer("CartoDB positron", name="Base", control=False).add_to(m)

    # Left: 1990 class map
    png_1990 = _class_map_to_png_bytes(map_1990, cmap)
    layer_left = folium.FeatureGroup(name="1990 Landcover")
    FoliumImageOverlay(  # type: ignore[misc]
        image=_png_to_base64(png_1990),
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.8,
        name="1990",
    ).add_to(layer_left)
    layer_left.add_to(m)

    # Right: 2025 class map
    png_2025 = _class_map_to_png_bytes(map_2025, cmap)
    layer_right = folium.FeatureGroup(name="2025 Landcover")
    FoliumImageOverlay(  # type: ignore[misc]
        image=_png_to_base64(png_2025),
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.8,
        name="2025",
    ).add_to(layer_right)
    layer_right.add_to(m)

    # Hotspot overlay
    if hotspot_map is not None:
        hs_rgba = np.zeros((*hotspot_map.shape, 4), dtype=np.uint8)
        hs_rgba[hotspot_map.astype(bool)] = [255, 50, 50, 180]
        import io
        from PIL import Image as PImage
        buf = io.BytesIO()
        PImage.fromarray(hs_rgba, mode="RGBA").save(buf, format="PNG")
        hs_b64 = _png_to_base64(buf.getvalue())
        hs_layer = folium.FeatureGroup(name="Change Hotspots", show=False)
        FoliumImageOverlay(  # type: ignore[misc]
            image=hs_b64,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=0.7,
        ).add_to(hs_layer)
        hs_layer.add_to(m)

    # Legend
    legend_html = _build_html_legend(CLASS_NAMES, CLASS_COLORS)
    m.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[union-attr]

    # Side-by-side slider
    try:
        SideBySideLayers(layer_left=layer_left, layer_right=layer_right).add_to(m)
    except Exception:
        pass

    folium.LayerControl().add_to(m)

    html_str = m._repr_html_()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.info("Dual map saved → %s", output_path)

    return html_str


def _build_html_legend(names: list[str], colors: list[str]) -> str:
    items = "".join(
        f'<li><span style="background:{c};width:14px;height:14px;'
        f'display:inline-block;margin-right:6px;border-radius:2px;"></span>{n}</li>'
        for n, c in zip(names, colors)
    )
    return f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px;border-radius:6px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3);font-size:12px;">
      <b>Landcover Classes</b>
      <ul style="list-style:none;margin:6px 0 0 0;padding:0;">{items}</ul>
    </div>"""


# ── Sankey diagram ────────────────────────────────────────────────────────────

def build_sankey_html(
    transition_matrices: list,   # list[TransitionMatrix]
    epoch_years: list[int],
    output_path: Optional[Path] = None,
    top_n_flows: int = 40,
) -> str:
    """Build a Plotly Sankey for multi-epoch class transitions.

    Parameters
    ----------
    transition_matrices: list of TransitionMatrix objects.
    epoch_years:         boundary years [y0, y1, y2, …].
    top_n_flows:         Keep only the top-N flows by pixel count.
    output_path:         Save HTML here.

    Returns
    -------
    HTML string.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError:
        logger.error("plotly not installed. Run `pip install plotly`.")
        return ""

    # Build node list: one node per class per epoch boundary
    n_cls = len(CLASS_NAMES)
    n_epochs = len(epoch_years)
    nodes_label: list[str] = []
    for epoch_idx, yr in enumerate(epoch_years):
        for cls in CLASS_NAMES:
            nodes_label.append(f"{cls}<br>{yr}")

    def node_id(epoch_idx: int, class_idx: int) -> int:
        return epoch_idx * n_cls + class_idx

    sources, targets, values, link_colors = [], [], [], []

    for tm_idx, tm in enumerate(transition_matrices):
        mat = tm.matrix  # (n_cls, n_cls)
        # Flatten and keep top N
        flows = [
            (int(f), int(t_), int(mat[f, t_]))
            for f in range(n_cls)
            for t_ in range(n_cls)
            if mat[f, t_] > 0
        ]
        flows.sort(key=lambda x: -x[2])
        for f, t_, cnt in flows[:top_n_flows]:
            sources.append(node_id(tm_idx, f))
            targets.append(node_id(tm_idx + 1, t_))
            values.append(cnt)
            link_colors.append(
                f"rgba({','.join(str(v) for v in _rgba(CLASS_COLORS[f], 0.4))})"
            )

    node_colors = [
        f"rgba({','.join(str(v) for v in _rgba(CLASS_COLORS[i % n_cls], 0.9))})"
        for i in range(len(nodes_label))
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes_label, color=node_colors, pad=12, thickness=15),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    )])
    fig.update_layout(
        title="Austin Landcover Transition Flow",
        font_size=11,
        height=700,
    )

    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_str, encoding="utf-8")
        logger.info("Sankey diagram saved → %s", output_path)

    return html_str


# ── Urban growth chart ────────────────────────────────────────────────────────

def build_urban_growth_html(
    annual_fractions: pd.DataFrame,
    urban_logistic: dict,
    output_path: Optional[Path] = None,
) -> str:
    """Plotly line chart of impervious fraction + logistic model fit."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError:
        logger.error("plotly not installed.")
        return ""

    for col in ["Impervious Surface", "High Density Development"]:
        if col not in annual_fractions.columns:
            continue

    years = list(annual_fractions.index)
    imp_col = (
        "Impervious Surface" if "Impervious Surface" in annual_fractions.columns
        else annual_fractions.columns[0]
    )
    obs = annual_fractions[imp_col].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=obs, mode="markers+lines",
                             name="Observed", marker=dict(size=5)))

    if urban_logistic.get("fitted"):
        fig.add_trace(go.Scatter(
            x=urban_logistic["years"],
            y=urban_logistic["fitted"],
            mode="lines",
            name=f"Logistic fit (R²={urban_logistic.get('r2', 0):.3f})",
            line=dict(dash="dash"),
        ))

    fig.update_layout(title="Austin Urban Expansion", xaxis_title="Year",
                      yaxis_title="Impervious surface fraction")
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_str, encoding="utf-8")
        logger.info("Urban growth chart saved → %s", output_path)

    return html_str


# ── Convenience runner ────────────────────────────────────────────────────────

def generate_all_outputs(
    output_dir: Path,
    class_maps: dict[int, np.ndarray],
    change_result,
    aoi_bounds_wgs84: tuple[float, float, float, float],
) -> None:
    """Generate and save all visualization outputs to output_dir.

    Parameters
    ----------
    output_dir:         Root output directory.
    class_maps:         {year: (H, W) Int8 class_map}.
    change_result:      ChangeResult from change_detection.ChangeDetector.
    aoi_bounds_wgs84:   (min_lon, min_lat, max_lon, max_lat).
    """
    years = sorted(class_maps.keys())

    map_1990 = class_maps.get(years[0])
    map_2025 = class_maps.get(years[-1])

    if map_1990 is not None and map_2025 is not None:
        build_dual_map(
            map_1990=map_1990,
            map_2025=map_2025,
            aoi_bounds_wgs84=aoi_bounds_wgs84,
            hotspot_map=change_result.hotspot_map if change_result else None,
            output_path=output_dir / "interactive_landcover_map.html",
        )

    if change_result and change_result.transitions:
        # Pick 5 evenly spaced epochs for Sankey
        tms = change_result.transitions
        step = max(1, len(tms) // 5)
        sampled_tms = tms[::step][:5]
        sampled_years = [years[0]] + [years[min(i * step + step, len(years) - 1)]
                                      for i in range(len(sampled_tms))]
        build_sankey_html(
            transition_matrices=sampled_tms,
            epoch_years=sampled_years,
            output_path=output_dir / "landcover_transitions_sankey.html",
        )

    if change_result and hasattr(change_result, "urban_logistic"):
        build_urban_growth_html(
            annual_fractions=change_result.annual_fractions,
            urban_logistic=change_result.urban_logistic,
            output_path=output_dir / "urban_expansion.html",
        )
