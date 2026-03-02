"""
cli.py
======
Click command-line interface for the Quantum Flood Frequency Mapper.

Usage::

    quantum-flood-frequency run --output ./outputs/flood \\
        --start-date 2015-01-01 --end-date 2025-12-31 \\
        --max-cloud 15 --verbose

    quantum-flood-frequency run --output ./outputs/flood \\
        --center-lat 29.76 --center-lon -95.37 --buffer-km 5 \\
        --no-quantum-svm   # faster run without QK-SVM
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger("geoscripthub.quantum_flood_frequency.cli")


@click.group()
@click.version_option(version="4.0.0", prog_name="quantum-flood-frequency")
def main() -> None:
    """Pseudo-Quantum Hybrid AI Flood Frequency Mapper v4.0.

    Fuses Landsat, Sentinel-2, NAIP, Sentinel-1 SAR, and Copernicus DEM
    imagery with quantum-inspired classification, SAR discrimination,
    and terrain constraints at 30 m aggregate resolution to map flood
    inundation frequency near San Jacinto River, TX.
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
    default=29.75645586726091,
    show_default=True,
    help="AOI centre latitude (WGS-84).  Default: San Jacinto River, TX.",
)
@click.option(
    "--center-lon",
    type=float,
    default=-95.08540365044576,
    show_default=True,
    help="AOI centre longitude (WGS-84).",
)
@click.option(
    "--buffer-km",
    type=float,
    default=3.0,
    show_default=True,
    help="Half-width of AOI bounding box in km.",
)
@click.option(
    "--start-date",
    type=str,
    default="2013-01-01",
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
    "--resolution",
    type=int,
    default=30,
    show_default=True,
    help="Target analysis resolution in metres (30=Landsat native, 10=S2, 1=NAIP).",
)
@click.option(
    "--use-sar/--no-sar",
    default=True,
    show_default=True,
    help="Use Sentinel-1 SAR for building/water discrimination.",
)
@click.option(
    "--use-terrain/--no-terrain",
    default=True,
    show_default=True,
    help="Use Copernicus DEM for terrain-based flood constraint.",
)
@click.option(
    "--sr-method",
    type=click.Choice(["bicubic", "spectral_guided", "learned"], case_sensitive=False),
    default="spectral_guided",
    show_default=True,
    help="Super-resolution method for upsampling coarse imagery.",
)
@click.option(
    "--meta-learner/--no-meta-learner",
    default=True,
    show_default=True,
    help="Use Ridge meta-learner for ensemble fusion (vs fixed Bayesian average).",
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
    resolution: int,
    use_sar: bool,
    use_terrain: bool,
    sr_method: str,
    meta_learner: bool,
    verbose: bool,
) -> None:
    """Run the full flood frequency analysis pipeline.

    v4.0 Steps:
      1. Build AOI around San Jacinto River, TX
      2. Acquire Landsat, Sentinel-2, NAIP, Sentinel-1 SAR, DEM imagery
      3. Aggregate & align to target resolution (default 30 m)
      4. Process SAR composite + terrain HAND features
      5. Classify water via 3-qubit QIEC v4.0 + SAR + terrain
      6. Aggregate into flood frequency surface
      7. Fetch USGS gauge data & FEMA flood zones for comparison
      8. Generate maps, rasters, and interactive viewer
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
        "║   Pseudo-Quantum Hybrid AI Flood Frequency Mapper  v4.0   ║",
        fg="cyan", bold=True,
    )
    click.secho(
        "║   SAR + DEM + Quantum-Inspired Ensemble Classification    ║",
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
    from .super_resolution import SRMethod

    # Map CLI string to SRMethod enum
    sr_method_map = {
        "bicubic": SRMethod.BICUBIC,
        "spectral_guided": SRMethod.SPECTRAL_GUIDED,
        "learned": SRMethod.LEARNED_SISR,
    }
    sr_enum = sr_method_map.get(sr_method, SRMethod.SPECTRAL_GUIDED)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Pipeline step counter
    n_steps = 9

    # Pipeline status tracker: green=success, yellow=no data, red=failed
    pipeline_status: dict[str, dict[str, str]] = {}

    def _track(step: str, status: str, detail: str = "") -> None:
        """Record pipeline step status."""
        pipeline_status[step] = {"status": status, "detail": detail}
        color = {"green": "green", "yellow": "yellow", "red": "red"}.get(status, "white")
        icon = {"green": "✓", "yellow": "~", "red": "✗"}.get(status, "?")
        if detail:
            click.secho(f"  {icon} {step}: {detail}", fg=color)

    # --- Step 1: AOI ---
    click.secho(f"\n▶ Step 1/{n_steps}: Building study area …", fg="yellow", bold=True)
    try:
        aoi = AOIBuilder(
            center_lat=center_lat,
            center_lon=center_lon,
            buffer_km=buffer_km,
        ).build()
        click.echo(f"  AOI: {aoi.description}")
        _track("AOI", "green", f"{aoi.description}")
    except Exception as exc:
        _track("AOI", "red", str(exc))
        raise

    # --- Step 2: Imagery acquisition ---
    click.secho(f"\n▶ Step 2/{n_steps}: Acquiring multi-sensor imagery …", fg="yellow", bold=True)
    try:
        acquisition = MultiSensorAcquisition(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud,
        )
        sensor_stack = acquisition.fetch_all()
        click.echo(f"  {sensor_stack}")
        _track("Landsat",    "green" if sensor_stack.landsat is not None else "yellow",
               f"{sensor_stack.landsat_count} scenes")
        _track("Sentinel-2", "green" if sensor_stack.sentinel2 is not None else "yellow",
               f"{sensor_stack.sentinel2_count} scenes")
        _track("NAIP",       "green" if (sensor_stack.naip_items and len(sensor_stack.naip_items) > 0) else "yellow",
               f"{len(sensor_stack.naip_items) if sensor_stack.naip_items else 0} items")
        if use_sar:
            _track("Sentinel-1 SAR",
                   "green" if sensor_stack.sentinel1 is not None else "yellow",
                   f"{sensor_stack.sentinel1_count} scenes")
        if use_terrain:
            _track("Copernicus DEM",
                   "green" if sensor_stack.dem is not None else "yellow",
                   "loaded" if sensor_stack.dem is not None else "not found")
    except Exception as exc:
        _track("Acquisition", "red", str(exc))
        raise

    # --- Step 3: Preprocessing (super-resolution + alignment) ---
    click.secho(
        f"\n▶ Step 3/{n_steps}: Super-resolve & align to {resolution} m grid …",
        fg="yellow", bold=True,
    )
    try:
        preprocessor = ImagePreprocessor(
            aoi=aoi,
            target_resolution=resolution,
            sr_method=sr_enum,
        )
        aligned = preprocessor.align(
            landsat=sensor_stack.landsat,
            sentinel2=sensor_stack.sentinel2,
            naip=sensor_stack.naip,
            sentinel1=sensor_stack.sentinel1 if use_sar else None,
            dem=sensor_stack.dem if use_terrain else None,
            naip_items=sensor_stack.naip_items if sensor_stack.naip_items else None,
        )
        click.echo(f"  {aligned}")
        _track("Preprocessing", "green",
               f"{aligned.total_scenes} scenes aligned to {resolution}m")
    except Exception as exc:
        _track("Preprocessing", "red", str(exc))
        raise

    # --- Step 4: SAR + terrain context ---
    click.secho(
        f"\n▶ Step 4/{n_steps}: Processing SAR & terrain context layers …",
        fg="yellow", bold=True,
    )
    sar_features = aligned.sar_features
    terrain_features = aligned.terrain_features
    if sar_features is not None:
        click.echo(
            f"  SAR: {sar_features.n_observations} observations, "
            f"water_px={int(sar_features.water_mask_sar.sum())}, "
            f"building_px={int(sar_features.building_mask_sar.sum())}"
        )
        _track("SAR Processing", "green",
               f"{sar_features.n_observations} obs, water={sar_features.water_mask_sar.sum()}")
    else:
        click.echo("  SAR: not available (classification will use spectral-only)")
        _track("SAR Processing", "yellow", "no SAR data available")
    if terrain_features is not None:
        click.echo(
            f"  Terrain: HAND [{np.nanmin(terrain_features.hand):.1f}–"
            f"{np.nanmax(terrain_features.hand):.1f}] m, "
            f"floodable_px={int(terrain_features.terrain_mask.sum())}"
        )
        _track("Terrain/DEM", "green",
               f"HAND range [{np.nanmin(terrain_features.hand):.1f}–"
               f"{np.nanmax(terrain_features.hand):.1f}] m")
    else:
        click.echo("  Terrain: not available (no HAND constraint)")
        _track("Terrain/DEM", "yellow", "no DEM data available")

    # --- Step 5: Quantum-hybrid classification (v4.0: seasonal + urban mask) ---
    click.secho(
        f"\n▶ Step 5/{n_steps}: QIEC v4.0 Classification (seasonal + urban mask) …",
        fg="yellow", bold=True,
    )
    try:
        classifier = QuantumHybridClassifier(
            use_quantum_svm=quantum_svm,
            use_meta_learner=meta_learner,
        )
        classifications = classifier.classify_stack(
            aligned,
            sar_features=sar_features,
            terrain_features=terrain_features,
        )
        click.echo(f"  Classified {len(classifications)} observations")
        _track("Classification", "green", f"{len(classifications)} observations classified")
    except Exception as exc:
        _track("Classification", "red", str(exc))
        raise

    # --- Step 6: Flood frequency ---
    click.secho(f"\n▶ Step 6/{n_steps}: Computing flood frequency surface …", fg="yellow", bold=True)
    try:
        engine = FloodFrequencyEngine(
            stack=aligned,
            terrain_features=terrain_features,
        )
        freq_result = engine.compute(classifications)
        click.echo(f"  {freq_result}")
        _track("Flood Frequency", "green",
               f"permanent={freq_result.permanent_mask.sum()} "
               f"seasonal={freq_result.seasonal_mask.sum()}")

        engine.save_frequency_raster(freq_result, output / "flood_frequency.tif")
        engine.save_zone_raster(freq_result, output / "flood_zones.tif")
        engine.save_confidence_raster(freq_result, output / "confidence_bounds.tif")
    except Exception as exc:
        _track("Flood Frequency", "red", str(exc))
        raise

    # --- Step 7: USGS gauge data for accuracy assessment ---
    click.secho(f"\n▶ Step 7/{n_steps}: Fetching USGS gauge data …", fg="yellow", bold=True)
    try:
        from .gauge_data import USGSGaugeData

        gauge = USGSGaugeData(
            center_lat=center_lat,
            center_lon=center_lon,
        )
        stations = gauge.discover_stations()
        if stations:
            # Collect observation dates from classifications
            obs_dates = [(cr.date, cr.source) for cr in classifications]
            gauge_validations = gauge.fetch_all_observation_dates(obs_dates)

            # Save gauge report
            gauge.save_gauge_report(gauge_validations, output / "gauge_validation.csv")

            n_with_data = sum(1 for v in gauge_validations if v.readings_dayof)
            n_flooding = sum(1 for v in gauge_validations if v.any_flooding)
            click.echo(
                f"  {len(stations)} stations, {n_with_data}/{len(gauge_validations)} "
                f"dates with gauge data, {n_flooding} flagged flooding"
            )
            _track("USGS Gauge Data", "green",
                   f"{len(stations)} stations, {n_with_data} dates matched")
        else:
            click.echo("  No USGS gauge stations found near study area")
            _track("USGS Gauge Data", "yellow", "no stations in search radius")
    except Exception as exc:
        click.secho(f"  USGS gauge data unavailable: {exc}", fg="red")
        _track("USGS Gauge Data", "yellow", f"error: {exc}")

    # --- Step 8: FEMA overlay ---
    click.secho(f"\n▶ Step 8/{n_steps}: Fetching FEMA flood zone data …", fg="yellow", bold=True)
    fema = FEMAFloodZones(aoi=aoi)
    try:
        fema_gdf = fema.fetch()
        click.echo(f"  FEMA zones: {len(fema_gdf)} polygons")
        if not fema_gdf.empty:
            click.echo(f"    Floodway:  {len(fema.floodway)} polygons")
            click.echo(f"    100-year:  {len(fema.zones_100yr)} polygons")
            click.echo(f"    500-year:  {len(fema.zones_500yr)} polygons")
            fema.save_geojson(output / "fema_flood_zones.geojson")
            shp_paths = fema.save_shapefiles(output)
            for cat, shp_path in shp_paths.items():
                click.echo(f"    Shapefile: {shp_path.name}  ({cat})")
            _track("FEMA Zones", "green", f"{len(fema_gdf)} polygons")
        else:
            _track("FEMA Zones", "yellow", "no polygons returned")
    except Exception as exc:
        click.secho(f"  FEMA data unavailable: {exc}", fg="red")
        _track("FEMA Zones", "yellow", f"error: {exc}")

    # --- Step 9: Visualisation ---
    click.secho(f"\n▶ Step 9/{n_steps}: Generating maps and visualisations …", fg="yellow", bold=True)
    try:
        mapper = FloodMapper(
            result=freq_result,
            aoi=aoi,
            fema=fema,
            output_dir=output,
        )
        viz_outputs = mapper.generate_all()
        for name, path in viz_outputs.items():
            click.echo(f"  {name}: {path}")
        _track("Visualisation", "green", f"{len(viz_outputs)} outputs generated")
    except Exception as exc:
        click.secho(f"  Visualisation failed: {exc}", fg="red")
        _track("Visualisation", "red", str(exc))

    # --- Pipeline status summary ---
    elapsed = time.perf_counter() - start

    click.secho(
        f"\n{'='*60}",
        fg="cyan", bold=True,
    )
    click.secho(
        "  PIPELINE STATUS SUMMARY",
        fg="cyan", bold=True,
    )
    click.secho(
        f"{'='*60}",
        fg="cyan", bold=True,
    )
    for step_name, info in pipeline_status.items():
        status = info["status"]
        detail = info["detail"]
        icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(status, "⚪")
        click.echo(f"  {icon} {step_name}: {detail}")

    # Save pipeline status as JSON for the HTML flowchart
    import json
    status_path = output / "pipeline_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump({
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "steps": pipeline_status,
        }, f, indent=2)

    # Generate HTML flowchart from status
    try:
        from .status_flowchart import generate_flowchart
        flowchart_path = generate_flowchart(status_path)
        click.echo(f"  Pipeline flowchart: {flowchart_path.name}")
    except Exception as exc:
        click.secho(f"  Flowchart generation failed: {exc}", fg="red")

    click.secho(
        f"\n✓ Pipeline complete in {elapsed:.1f}s → {output}",
        fg="green", bold=True,
    )
    click.echo(
        f"  Frequency GeoTIFF:  flood_frequency.tif\n"
        f"  Zone GeoTIFF:       flood_zones.tif\n"
        f"  Confidence:         confidence_bounds.tif\n"
        f"  Gauge validation:   gauge_validation.csv\n"
        f"  Pipeline status:    pipeline_status.json\n"
        f"  Interactive map:    interactive_flood_map.html\n"
        f"  Statistics:         observation_stats.png"
    )


if __name__ == "__main__":
    main()
