"""Visualisation — land-cover maps, change maps, charts, Sankey diagrams.

v1.0 — Quantum Land-Cover Change Detector
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import (
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_LABELS,
    CLASS_COLOURS,
    CLASS_CMAP_RGBA,
    LANDCOVER_CLASSES,
)

logger = logging.getLogger("geoscripthub.landcover_change.viz")


# ── Colour helpers ────────────────────────────────────────────────


def _hex_colours() -> list[str]:
    """Return list of hex colour strings for each landcover class."""
    return list(CLASS_COLOURS)


def _class_colour_map(class_map: np.ndarray) -> np.ndarray:
    """Convert integer class map → RGBA uint8 image."""
    rgba = (CLASS_CMAP_RGBA * 255).astype("uint8")  # float 0-1 → uint8
    h, w = class_map.shape
    img = np.zeros((h, w, 4), dtype="uint8")
    for c in range(NUM_CLASSES):
        mask = class_map == c
        img[mask] = rgba[c]
    return img


# ── Classification Maps ──────────────────────────────────────────


def plot_classification_map(
    class_map: np.ndarray,
    year: int,
    output_path: Path,
    *,
    title: Optional[str] = None,
    dpi: int = 200,
) -> Path:
    """Plot a single land-cover classification map with legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    img = _class_colour_map(class_map)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img[:, :, :3])
    ax.set_title(title or f"Land Cover — {year}", fontsize=14, fontweight="bold")
    ax.set_axis_off()

    legend_patches = [
        Patch(facecolor=_hex_colours()[i], edgecolor="black", linewidth=0.5,
              label=CLASS_LABELS[i])
        for i in range(NUM_CLASSES)
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7,
              framealpha=0.9, ncol=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return output_path


def plot_decade_maps(
    class_maps: list[np.ndarray],
    years: list[int],
    output_dir: Path,
    *,
    dpi: int = 200,
) -> list[Path]:
    """Plot one panel per decade (pick representative year)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    decade_targets = [1990, 2000, 2010, 2020]
    panels: list[tuple[np.ndarray, int]] = []

    for d in decade_targets:
        closest_idx = int(np.argmin([abs(y - d) for y in years]))
        panels.append((class_maps[closest_idx], years[closest_idx]))

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    for ax, (cm, yr) in zip(axes, panels):
        img = _class_colour_map(cm)
        ax.imshow(img[:, :, :3])
        ax.set_title(f"{yr}", fontsize=12, fontweight="bold")
        ax.set_axis_off()

    legend_patches = [
        Patch(facecolor=_hex_colours()[i], edgecolor="black", linewidth=0.5,
              label=CLASS_LABELS[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=7,
               ncol=NUM_CLASSES, framealpha=0.9)
    fig.suptitle("Land Cover by Decade", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "decade_panel.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return [path]


# ── Change Maps ───────────────────────────────────────────────────


def plot_change_map(
    from_map: np.ndarray,
    to_map: np.ndarray,
    from_year: int,
    to_year: int,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Side-by-side then-and-now change map with highlighted changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Then
    axes[0].imshow(_class_colour_map(from_map)[:, :, :3])
    axes[0].set_title(f"{from_year}", fontsize=12, fontweight="bold")
    axes[0].set_axis_off()

    # Now
    axes[1].imshow(_class_colour_map(to_map)[:, :, :3])
    axes[1].set_title(f"{to_year}", fontsize=12, fontweight="bold")
    axes[1].set_axis_off()

    # Change overlay
    change = from_map != to_map
    overlay = _class_colour_map(to_map)[:, :, :3].copy()
    overlay[~change] = (overlay[~change] * 0.4).astype("uint8")  # dim unchanged
    # Highlight changes with bright border
    axes[2].imshow(overlay)
    axes[2].set_title("Change Pixels", fontsize=12, fontweight="bold")
    axes[2].set_axis_off()

    legend_patches = [
        Patch(facecolor=_hex_colours()[i], edgecolor="black", linewidth=0.5,
              label=CLASS_LABELS[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc="lower center", fontsize=7,
               ncol=NUM_CLASSES, framealpha=0.9)
    pct = 100.0 * np.mean(change)
    fig.suptitle(
        f"Land Cover Change: {from_year} → {to_year}  ({pct:.1f}% changed)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return output_path


# ── Area Time-Series ──────────────────────────────────────────────


def plot_area_timeseries(
    class_maps: list[np.ndarray],
    years: list[int],
    pixel_area_ha: float,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Stacked area chart of class areas over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    areas = np.zeros((len(years), NUM_CLASSES), dtype="float64")
    for t, cm in enumerate(class_maps):
        for c in range(NUM_CLASSES):
            areas[t, c] = np.sum(cm == c) * pixel_area_ha

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = _hex_colours()
    ax.stackplot(years, areas.T, labels=CLASS_LABELS, colors=colours, alpha=0.85)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Area (ha)", fontsize=11)
    ax.set_title("Land Cover Area Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    ax.set_xlim(years[0], years[-1])
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_class_percentage_timeseries(
    class_maps: list[np.ndarray],
    years: list[int],
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Line chart showing percentage of each class over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pcts = np.zeros((len(years), NUM_CLASSES), dtype="float64")
    for t, cm in enumerate(class_maps):
        total = cm.size
        for c in range(NUM_CLASSES):
            pcts[t, c] = 100.0 * np.sum(cm == c) / total

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = _hex_colours()
    for c in range(NUM_CLASSES):
        ax.plot(years, pcts[:, c], label=CLASS_LABELS[c], color=colours[c],
                linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Class %", fontsize=11)
    ax.set_title("Land Cover Class Percentage Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    ax.set_xlim(years[0], years[-1])
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Transition Sankey Diagram ─────────────────────────────────────


def plot_sankey_diagram(
    transition_matrix: np.ndarray,
    from_year: int,
    to_year: int,
    output_path: Path,
    *,
    min_flow_pct: float = 0.5,
    dpi: int = 200,
) -> Path:
    """Plot a Sankey flow diagram of land-cover transitions.

    Falls back to a heatmap if plotly is not available.
    """
    try:
        return _sankey_plotly(transition_matrix, from_year, to_year,
                             output_path, min_flow_pct=min_flow_pct)
    except ImportError:
        return _sankey_heatmap(transition_matrix, from_year, to_year,
                               output_path, dpi=dpi)


def _sankey_plotly(
    matrix: np.ndarray,
    from_year: int,
    to_year: int,
    output_path: Path,
    *,
    min_flow_pct: float = 0.5,
) -> Path:
    """Plotly-based Sankey diagram."""
    import plotly.graph_objects as go

    total = matrix.sum()
    colours = _hex_colours()

    # Build source → target links (skip tiny flows)
    sources, targets, values, link_colours = [], [], [], []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                continue  # skip stable pixels
            pct = 100.0 * matrix[i, j] / max(total, 1)
            if pct < min_flow_pct:
                continue
            sources.append(i)
            targets.append(NUM_CLASSES + j)
            values.append(int(matrix[i, j]))
            link_colours.append(colours[i] + "80")  # semi-transparent

    node_labels = [f"{n} ({from_year})" for n in CLASS_LABELS] + \
                  [f"{n} ({to_year})" for n in CLASS_LABELS]
    node_colours = colours + colours

    fig = go.Figure(go.Sankey(
        node={"label": node_labels, "color": node_colours, "pad": 20},
        link={"source": sources, "target": targets, "value": values,
              "color": link_colours},
    ))
    fig.update_layout(
        title_text=f"Land Cover Transition: {from_year} → {to_year}",
        font_size=10, width=1000, height=600,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
    return output_path


def _sankey_heatmap(
    matrix: np.ndarray,
    from_year: int,
    to_year: int,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Matplotlib heatmap fallback for transition matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = matrix.sum()
    pct_matrix = 100.0 * matrix / max(total, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pct_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_LABELS, fontsize=8)
    ax.set_xlabel(f"To ({to_year})", fontsize=11)
    ax.set_ylabel(f"From ({from_year})", fontsize=11)
    ax.set_title(
        f"Transition Matrix: {from_year} → {to_year} (% pixels)",
        fontsize=13, fontweight="bold",
    )

    # Annotate cells
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = pct_matrix[i, j]
            if val > 0.1:
                colour = "white" if val > 10 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=colour)

    fig.colorbar(im, ax=ax, label="% of total pixels", shrink=0.8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Confusion Matrix Heatmap ─────────────────────────────────────


def plot_confusion_matrix(
    matrix: np.ndarray,
    year: int,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot confusion matrix as annotated heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Normalise rows to percentages
    row_sums = matrix.sum(axis=1, keepdims=True).astype("float64")
    row_sums[row_sums == 0] = 1
    pct = 100.0 * matrix / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_LABELS, fontsize=8)
    ax.set_xlabel("Reference (NLCD)", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(f"Confusion Matrix — {year}", fontsize=13, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = pct[i, j]
            count = int(matrix[i, j])
            if count > 0:
                colour = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color=colour)

    fig.colorbar(im, ax=ax, label="Row %", shrink=0.8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Trend Map ─────────────────────────────────────────────────────


def plot_trend_map(
    trend_label: np.ndarray,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot trend classification (stable/urbanising/greening/etc.)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    trend_names = ["Stable", "Urbanising", "Greening", "Degrading", "Variable"]
    trend_colours = ["#808080", "#FF4444", "#22AA22", "#CC8800", "#8844FF"]

    h, w = trend_label.shape
    img = np.zeros((h, w, 3), dtype="uint8")
    for i, colour_hex in enumerate(trend_colours):
        r = int(colour_hex[1:3], 16)
        g = int(colour_hex[3:5], 16)
        b = int(colour_hex[5:7], 16)
        mask = trend_label == i
        img[mask] = [r, g, b]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title("Temporal Trend (1990–Present)", fontsize=14, fontweight="bold")
    ax.set_axis_off()

    patches = [
        Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=n)
        for n, c in zip(trend_names, trend_colours)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Statistics CSV ────────────────────────────────────────────────


def export_area_stats_csv(
    class_maps: list[np.ndarray],
    years: list[int],
    pixel_area_ha: float,
    output_path: Path,
) -> Path:
    """Export area statistics per class per year to CSV."""
    lines = ["year," + ",".join(f"{n}_ha" for n in CLASS_NAMES)]
    for t, (cm, yr) in enumerate(zip(class_maps, years)):
        row_vals = []
        for c in range(NUM_CLASSES):
            row_vals.append(f"{np.sum(cm == c) * pixel_area_ha:.2f}")
        lines.append(f"{yr}," + ",".join(row_vals))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_transition_csv(
    transition_matrix: np.ndarray,
    from_year: int,
    to_year: int,
    output_path: Path,
    pixel_area_ha: float,
) -> Path:
    """Export transition matrix (hectares) to CSV."""
    area = transition_matrix.astype("float64") * pixel_area_ha
    header = f"from\\to," + ",".join(CLASS_NAMES)
    lines = [header]
    for i in range(NUM_CLASSES):
        row = CLASS_NAMES[i] + "," + ",".join(f"{area[i,j]:.1f}" for j in range(NUM_CLASSES))
        lines.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── Interactive Folium Map ────────────────────────────────────────


def create_interactive_map(
    class_maps: list[np.ndarray],
    years: list[int],
    bbox: tuple[float, float, float, float],
    output_path: Path,
) -> Path:
    """Create an interactive Folium HTML map with layer toggles.

    Parameters
    ----------
    class_maps:
        List of classification arrays.
    years:
        Corresponding years.
    bbox:
        (west, south, east, north) in EPSG:4326.
    output_path:
        Where to save the HTML file.
    """
    try:
        import folium
        from folium.raster_layers import ImageOverlay
    except ImportError:
        logger.warning("folium not installed; skipping interactive map")
        return output_path

    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles="OpenStreetMap")

    # Add a few representative years as overlays
    sample_years = [years[0], years[len(years)//3], years[2*len(years)//3], years[-1]]

    for sy in sample_years:
        if sy not in years:
            continue
        idx = years.index(sy)
        cm = class_maps[idx]
        img = _class_colour_map(cm)

        # Convert to PNG data URL
        try:
            from PIL import Image
            import io, base64
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            ImageOverlay(
                image=data_url,
                bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
                name=f"Land Cover {sy}",
                opacity=0.7,
            ).add_to(m)
        except ImportError:
            logger.debug("PIL not available for interactive overlay; skipping year %d", sy)

    folium.LayerControl().add_to(m)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info("Interactive map saved to %s", output_path)
    return output_path


# ── Master Visualisation Runner ───────────────────────────────────


def generate_all_visualisations(
    change_result,  # ChangeDetectionResult
    accuracy_result,  # AccuracyResult
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    *,
    dpi: int = 200,
) -> list[Path]:
    """Generate all visualisation outputs.

    Returns list of all generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    years = change_result.years
    class_maps = [cr.class_map for cr in change_result.yearly_maps]
    pixel_area_ha = (change_result.resolution ** 2) / 10000.0

    # 1) Annual class maps (first, mid, last at minimum)
    sample_indices = [0, len(years) // 2, len(years) - 1]
    for idx in sample_indices:
        if idx < len(years):
            p = plot_classification_map(
                class_maps[idx], years[idx],
                output_dir / f"landcover_{years[idx]}.png", dpi=dpi,
            )
            saved.append(p)

    # 2) Decade panel
    saved.extend(plot_decade_maps(class_maps, years, output_dir, dpi=dpi))

    # 3) Then-and-now change map
    tn = change_result.then_and_now
    p = plot_change_map(
        tn.from_class, tn.to_class, tn.from_year, tn.to_year,
        output_dir / f"change_then_now_{tn.from_year}_{tn.to_year}.png", dpi=dpi,
    )
    saved.append(p)

    # 4) Decade change maps
    for ds in change_result.decade_summaries:
        cm = ds.change_map
        p = plot_change_map(
            cm.from_class, cm.to_class, cm.from_year, cm.to_year,
            output_dir / f"change_{cm.from_year}_{cm.to_year}.png", dpi=dpi,
        )
        saved.append(p)

    # 5) Transition Sankey for then-and-now
    from .change_detection import ChangeDetectionEngine
    engine = ChangeDetectionEngine(change_result.resolution)
    tm = engine._compute_transition_matrix(tn.from_class, tn.to_class, tn.from_year, tn.to_year)
    p = plot_sankey_diagram(
        tm.matrix, tn.from_year, tn.to_year,
        output_dir / f"sankey_{tn.from_year}_{tn.to_year}.png", dpi=dpi,
    )
    saved.append(p)

    # 6) Area time-series
    p = plot_area_timeseries(
        class_maps, years, pixel_area_ha,
        output_dir / "area_timeseries.png", dpi=dpi,
    )
    saved.append(p)

    p = plot_class_percentage_timeseries(
        class_maps, years,
        output_dir / "class_percentage.png", dpi=dpi,
    )
    saved.append(p)

    # 7) Trend map
    p = plot_trend_map(
        change_result.trend.trend_label,
        output_dir / "trend_map.png", dpi=dpi,
    )
    saved.append(p)

    # 8) Confusion matrices
    for m in accuracy_result.metrics_per_epoch:
        p = plot_confusion_matrix(
            m.matrix, m.year,
            output_dir / f"confusion_{m.year}.png", dpi=dpi,
        )
        saved.append(p)

    # 9) Area stats CSV
    p = export_area_stats_csv(class_maps, years, pixel_area_ha,
                              output_dir / "area_stats.csv")
    saved.append(p)

    # 10) Transition CSVs
    for ds in change_result.decade_summaries:
        p = export_transition_csv(
            ds.transition.matrix, ds.from_year, ds.to_year,
            output_dir / f"transition_{ds.from_year}_{ds.to_year}.csv",
            pixel_area_ha,
        )
        saved.append(p)

    # 11) Interactive Folium map
    p = create_interactive_map(class_maps, years, bbox, output_dir / "interactive_map.html")
    saved.append(p)

    logger.info("Generated %d visualisation files in %s", len(saved), output_dir)
    return saved
