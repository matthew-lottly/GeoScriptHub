"""
cli.py
======
Click command-line interface for the Quantum Flood Frequency Mapper.

Usage::

    quantum-flood-frequency run --output ./outputs/flood \\
        --start-date 2015-01-01 --end-date 2025-12-31 \\
        --max-cloud 15 --verbose

    quantum-flood-frequency run --output ./outputs/flood \\
        --center-lat 29.22 --center-lon -89.25 --buffer-km 5 \\
        --no-quantum-svm   # faster run without QK-SVM
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.cli")


@click.group()
@click.version_option(version="1.0.0", prog_name="quantum-flood-frequency")
def main() -> None:
    """Pseudo-Quantum Hybrid AI Flood Frequency Mapper.

    Fuses Landsat, Sentinel-2 and NAIP imagery with quantum-inspired
    classification to map flood inundation frequency on the Mississippi
    River.
    """


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("outputs/quantum_flood_frequency"),
    show_default=True,
    help="Directory for all output products (rasters, PNGs, HTML).",
)
@click.option(
    "--center-lat",
    type=float,
    default=29.22,
    show_default=True,
    help="AOI centre latitude (WGS-84).  Default ≈ 5 mi upstream of Head of Passes, LA.",
)
@click.option(
    "--center-lon",
    type=float,
    default=-89.25,
    show_default=True,
    help="AOI centre longitude (WGS-84).",
)
@click.option(
    "--buffer-km",
    type=float,
    default=5.0,
    show_default=True,
    help="Half-width of AOI bounding box in km.",
)
@click.option(
    "--start-date",
    type=str,
    default="2015-01-01",
    show_default=True,
    help="Start of temporal window (ISO 8601).",
)
@click.option(
    "--end-date",
    type=str,
    default="2025-12-31",
    show_default=True,
    help="End of temporal window (ISO 8601).",
)
@click.option(
    "--max-cloud",
    type=int,
    default=15,
    show_default=True,
    help="Maximum scene cloud cover %% (0–100).",
)
@click.option(
    "--quantum-svm/--no-quantum-svm",
    default=True,
    show_default=True,
    help="Enable/disable the Quantum Kernel SVM component (slower but more accurate).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def run(
    output: Path,
    center_lat: float,
    center_lon: float,
    buffer_km: float,
    start_date: str,
    end_date: str,
    max_cloud: int,
    quantum_svm: bool,
    verbose: bool,
) -> None:
    """Run the full flood frequency analysis pipeline.

    Steps:
      1. Build AOI around the Mississippi River
      2. Acquire Landsat, Sentinel-2, NAIP imagery from Planetary Computer
      3. Preprocess: resample → align → cloud-mask → normalise
      4. Classify water via Quantum-Inspired Ensemble (QIEC)
      5. Aggregate into flood frequency surface
      6. Fetch FEMA flood zones for comparison
      7. Generate maps, rasters, and interactive viewer
    """
    # --- Logging setup ---
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger = logging.getLogger("geoscripthub")
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    start = time.perf_counter()

    click.secho(
        "╔══════════════════════════════════════════════════════════════╗",
        fg="cyan", bold=True,
    )
    click.secho(
        "║   Pseudo-Quantum Hybrid AI Flood Frequency Mapper  v1.0   ║",
        fg="cyan", bold=True,
    )
    click.secho(
        "╚══════════════════════════════════════════════════════════════╝",
        fg="cyan", bold=True,
    )

    # --- Import heavy modules lazily ---
    from .aoi import AOIBuilder
    from .acquisition import MultiSensorAcquisition
    from .preprocessing import ImagePreprocessor
    from .quantum_classifier import QuantumHybridClassifier
    from .flood_engine import FloodFrequencyEngine
    from .fema import FEMAFloodZones
    from .viz import FloodMapper

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # --- Step 1: AOI ---
    click.secho("\n▶ Step 1/7: Building study area …", fg="yellow", bold=True)
    aoi = AOIBuilder(
        center_lat=center_lat,
        center_lon=center_lon,
        buffer_km=buffer_km,
    ).build()
    click.echo(f"  AOI: {aoi.description}")

    # --- Step 2: Imagery acquisition ---
    click.secho("\n▶ Step 2/7: Acquiring multi-sensor imagery …", fg="yellow", bold=True)
    acquisition = MultiSensorAcquisition(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud,
    )
    sensor_stack = acquisition.fetch_all()
    click.echo(f"  {sensor_stack}")

    # --- Step 3: Preprocessing ---
    click.secho("\n▶ Step 3/7: Preprocessing & alignment (30 m grid) …", fg="yellow", bold=True)
    preprocessor = ImagePreprocessor(aoi=aoi)
    aligned = preprocessor.align(
        landsat=sensor_stack.landsat,
        sentinel2=sensor_stack.sentinel2,
        naip=sensor_stack.naip,
    )
    click.echo(f"  {aligned}")

    # --- Step 4: Quantum-hybrid classification ---
    click.secho("\n▶ Step 4/7: Quantum-Inspired Ensemble Classification …", fg="yellow", bold=True)
    classifier = QuantumHybridClassifier(use_quantum_svm=quantum_svm)
    classifications = classifier.classify_stack(aligned)
    click.echo(f"  Classified {len(classifications)} observations")

    # --- Step 5: Flood frequency ---
    click.secho("\n▶ Step 5/7: Computing flood frequency surface …", fg="yellow", bold=True)
    engine = FloodFrequencyEngine(stack=aligned)
    freq_result = engine.compute(classifications)
    click.echo(f"  {freq_result}")

    # Save rasters
    engine.save_frequency_raster(freq_result, output / "flood_frequency.tif")
    engine.save_zone_raster(freq_result, output / "flood_zones.tif")
    engine.save_confidence_raster(freq_result, output / "confidence_bounds.tif")

    # --- Step 6: FEMA overlay ---
    click.secho("\n▶ Step 6/7: Fetching FEMA flood zone data …", fg="yellow", bold=True)
    fema = FEMAFloodZones(aoi=aoi)
    try:
        fema_gdf = fema.fetch()
        click.echo(f"  FEMA zones: {len(fema_gdf)} polygons")
        if not fema_gdf.empty:
            fema.save_geojson(output / "fema_flood_zones.geojson")
    except Exception as exc:
        click.secho(f"  FEMA data unavailable: {exc}", fg="red")

    # --- Step 7: Visualisation ---
    click.secho("\n▶ Step 7/7: Generating maps and visualisations …", fg="yellow", bold=True)
    mapper = FloodMapper(
        result=freq_result,
        aoi=aoi,
        fema=fema,
        output_dir=output,
    )
    viz_outputs = mapper.generate_all()
    for name, path in viz_outputs.items():
        click.echo(f"  {name}: {path}")

    elapsed = time.perf_counter() - start

    click.secho(
        f"\n✓ Pipeline complete in {elapsed:.1f}s → {output}",
        fg="green", bold=True,
    )
    click.echo(
        f"  Frequency GeoTIFF: flood_frequency.tif\n"
        f"  Zone GeoTIFF:      flood_zones.tif\n"
        f"  Confidence:        confidence_bounds.tif\n"
        f"  Frequency map:     flood_frequency_map.png\n"
        f"  FEMA comparison:   fema_comparison_map.png\n"
        f"  Interactive map:   interactive_flood_map.html\n"
        f"  Statistics:        observation_stats.png"
    )


if __name__ == "__main__":
    main()
