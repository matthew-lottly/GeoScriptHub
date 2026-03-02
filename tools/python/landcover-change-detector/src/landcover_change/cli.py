"""CLI — Click-based entry point for the land-cover change detector.

v1.0 — Quantum Land-Cover Change Detector
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

from . import __version__

logger = logging.getLogger("geoscripthub.landcover_change")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(name)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


@click.group()
@click.version_option(__version__, prog_name="landcover-change-detector")
def cli() -> None:
    """Quantum-Enhanced Land-Cover Change Detector.

    Produces annual land-cover maps (1990–present) using
    multi-sensor satellite data, pseudo-quantum classification,
    and ensemble machine learning.
    """


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="output_landcover",
              help="Output directory.", show_default=True)
@click.option("--center-lat", type=float, default=29.757,
              help="Centre latitude (WGS-84).", show_default=True)
@click.option("--center-lon", type=float, default=-95.085,
              help="Centre longitude (WGS-84).", show_default=True)
@click.option("--buffer-km", type=float, default=5.0,
              help="Radius of AOI in km.", show_default=True)
@click.option("--start-year", type=int, default=1990,
              help="First year of analysis.", show_default=True)
@click.option("--end-year", type=int, default=2026,
              help="Last year of analysis.", show_default=True)
@click.option("--max-cloud", type=float, default=15.0,
              help="Max cloud cover (%%) for optical scenes.", show_default=True)
@click.option("--resolution", type=float, default=30.0,
              help="Target resolution (metres).", show_default=True)
@click.option("--use-sar/--no-sar", default=True, show_default=True,
              help="Include Sentinel-1 SAR features.")
@click.option("--use-terrain/--no-terrain", default=True, show_default=True,
              help="Include DEM-derived terrain features.")
@click.option("--use-quantum/--no-quantum", default=True, show_default=True,
              help="Enable quantum-enhanced classification.")
@click.option("--skip-accuracy/--run-accuracy", default=False, show_default=True,
              help="Skip NLCD accuracy assessment (faster).")
@click.option("--flowchart-only", is_flag=True, default=False,
              help="Only generate the pipeline flowchart, then exit.")
@click.option("--dpi", type=int, default=200, show_default=True,
              help="DPI for output figures.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable debug-level logging.")
def run(
    output: str,
    center_lat: float,
    center_lon: float,
    buffer_km: float,
    start_year: int,
    end_year: int,
    max_cloud: float,
    resolution: float,
    use_sar: bool,
    use_terrain: bool,
    use_quantum: bool,
    skip_accuracy: bool,
    flowchart_only: bool,
    dpi: int,
    verbose: bool,
) -> None:
    """Run the full land-cover change detection pipeline."""
    _setup_logging(verbose)
    t0 = time.perf_counter()

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("QUANTUM LAND-COVER CHANGE DETECTOR  v%s", __version__)
    logger.info("=" * 72)
    logger.info("AOI centre: %.5f°N, %.5f°E  buffer=%s km", center_lat, center_lon, buffer_km)
    logger.info("Period: %d – %d  resolution: %dm  max cloud: %s%%",
                start_year, end_year, int(resolution), max_cloud)
    logger.info("SAR=%s  Terrain=%s  Quantum=%s", use_sar, use_terrain, use_quantum)
    logger.info("Output: %s", output_dir.resolve())

    # ── Step 0 — Flowchart ────────────────────────────────────────
    logger.info("[0/10] Generating pipeline flowchart...")
    from .flowchart import generate_flowchart
    generate_flowchart(output_dir, html=True, png=True, pdf=True)

    if flowchart_only:
        logger.info("Flowchart-only mode — exiting.")
        return

    # ── Step 1 — AOI ──────────────────────────────────────────────
    logger.info("[1/10] Building AOI...")
    from .aoi import AOIBuilder
    aoi = AOIBuilder(center_lat, center_lon, buffer_km).build()
    logger.info("  CRS: %s  bounds: %s", aoi.target_crs, aoi.bbox_utm)

    # ── Step 2 — Multi-Epoch Acquisition ──────────────────────────
    logger.info("[2/10] Acquiring satellite data (STAC)...")
    from .acquisition import MultiEpochAcquisition, SensorStack
    acq = MultiEpochAcquisition(
        aoi=aoi,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        max_cloud=int(max_cloud),
    )
    sensor_stack: SensorStack = acq.fetch_all(use_sar=use_sar, use_terrain=use_terrain)
    logger.info("  Landsat: %d  S2: %d  S1: %d  NAIP: %d",
                sensor_stack.landsat_count, sensor_stack.sentinel2_count,
                sensor_stack.sentinel1_count, sensor_stack.naip_count)

    # ── Step 3 — Preprocessing & Compositing ──────────────────────
    logger.info("[3/10] Preprocessing & annual compositing...")
    from .preprocessing import ImagePreprocessor
    preprocessor = ImagePreprocessor(aoi=aoi, target_resolution=int(resolution))
    preprocessed = preprocessor.align(
        landsat=sensor_stack.landsat,
        sentinel2=sensor_stack.sentinel2,
        naip=sensor_stack.naip,
        sentinel1=sensor_stack.sentinel1 if use_sar else None,
        dem=sensor_stack.dem if use_terrain else None,
        naip_items=sensor_stack.naip_items,
    )
    logger.info("  %d annual composites built (%d–%d)",
                len(preprocessed.composites),
                preprocessed.composites[0].year if preprocessed.composites else 0,
                preprocessed.composites[-1].year if preprocessed.composites else 0)

    # ── Step 4 — Terrain Features ─────────────────────────────────
    # Already computed inside preprocessing if DEM was provided
    terrain_features = preprocessed.terrain_features
    if terrain_features:
        logger.info("[4/10] Terrain features ready.")
    else:
        logger.info("[4/10] No terrain features available.")

    # ── Step 5 — SAR Features ─────────────────────────────────────
    sar_features = preprocessed.sar_features
    if sar_features:
        logger.info("[5/10] SAR features ready.")
    else:
        logger.info("[5/10] No SAR features available.")

    # ── Step 6 — Feature Engineering ──────────────────────────────
    logger.info("[6/10] Building feature stacks...")
    from .feature_engineering import build_feature_stack, FeatureStack
    feature_stacks: list[FeatureStack] = []
    for comp in preprocessed.composites:
        fs = build_feature_stack(
            composite=comp,
            sar=sar_features,
            terrain=terrain_features,
        )
        feature_stacks.append(fs)
    logger.info("  %d feature stacks × %d bands",
                len(feature_stacks),
                feature_stacks[0][1].shape[-1] if feature_stacks else 0)

    # ── Step 7 — Quantum-Enhanced Classification ──────────────────
    logger.info("[7/10] Running quantum-enhanced classification...")
    from .quantum_classifier import QuantumLandCoverClassifier
    classifier = QuantumLandCoverClassifier(use_quantum=use_quantum)
    classifications = classifier.classify_stack(feature_stacks)
    logger.info("  %d annual maps classified", len(classifications))

    # ── Step 8 — Change Detection ─────────────────────────────────
    logger.info("[8/10] Computing land-cover change...")
    from .change_detection import ChangeDetectionEngine
    engine = ChangeDetectionEngine(resolution=resolution)
    change_result = engine.compute(classifications)
    logger.info("  Then-and-now: %.1f%% changed",
                100.0 * change_result.then_and_now.change_mask.mean())
    for ds in change_result.decade_summaries:
        logger.info("  %s: %.1f%% changed", ds.label,
                     100.0 * ds.change_map.change_mask.mean())

    # ── Step 9 — Accuracy Assessment ──────────────────────────────
    if not skip_accuracy:
        logger.info("[9/10] Accuracy assessment (NLCD reference)...")
        from .accuracy import AccuracyAssessor, export_accuracy_csv
        assessor = AccuracyAssessor(
            bbox=aoi.bbox_wgs84,
            cache_dir=output_dir / "cache",
        )
        accuracy_result = assessor.assess(classifications, change_result.years)
        export_accuracy_csv(accuracy_result, output_dir)
        logger.info("  Overall accuracy: %.1f%%  Kappa: %.3f",
                     accuracy_result.overall_metrics.overall_accuracy * 100,
                     accuracy_result.overall_metrics.kappa)
    else:
        logger.info("[9/10] Skipping accuracy assessment.")
        # Build a minimal dummy result
        from .accuracy import AccuracyResult, ConfusionMetrics, TemporalConsistency
        import numpy as np
        accuracy_result = AccuracyResult(
            metrics_per_epoch=[],
            overall_metrics=ConfusionMetrics(
                year=0, matrix=np.zeros((8, 8), dtype="int64"),
                overall_accuracy=0, kappa=0,
                producers_accuracy=np.zeros(8), users_accuracy=np.zeros(8),
                f1=np.zeros(8), class_names=[],
            ),
            temporal_consistency=TemporalConsistency(0, 0, 0),
            nlcd_years_used=[], sample_count=0,
        )

    # ── Step 10 — Visualisation & Export ──────────────────────────
    logger.info("[10/10] Generating visualisations & exports...")
    from .viz import generate_all_visualisations
    bbox_wgs84 = aoi.bbox_wgs84
    vis_files = generate_all_visualisations(
        change_result, accuracy_result, bbox_wgs84, output_dir, dpi=dpi,
    )

    # Save GeoTIFF rasters
    transform_tuple = (
        aoi.bbox_utm[0],  # west
        resolution, 0.0,
        aoi.bbox_utm[3],  # north
        0.0, -resolution,
    )
    raster_files = engine.save_rasters(
        change_result, output_dir / "rasters",
        transform=transform_tuple,
        crs=aoi.target_crs,
    )

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────
    summary = {
        "version": __version__,
        "aoi": {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "buffer_km": buffer_km,
            "target_crs": aoi.target_crs,
        },
        "period": {"start": start_year, "end": end_year},
        "n_years_classified": len(change_result.years),
        "years": change_result.years,
        "change_pct_overall": round(
            100.0 * float(change_result.then_and_now.change_mask.mean()), 2,
        ),
        "decade_summaries": [
            {
                "label": ds.label,
                "from_year": ds.from_year,
                "to_year": ds.to_year,
                "change_pct": round(100.0 * float(ds.change_map.change_mask.mean()), 2),
                "net_ha": ds.net_ha,
            }
            for ds in change_result.decade_summaries
        ],
        "accuracy": {
            "overall_accuracy": round(accuracy_result.overall_metrics.overall_accuracy * 100, 2),
            "kappa": round(accuracy_result.overall_metrics.kappa, 4),
            "nlcd_years_used": accuracy_result.nlcd_years_used,
        },
        "temporal_consistency": {
            "mean_annual_change_rate": round(
                accuracy_result.temporal_consistency.mean_annual_change_rate * 100, 2,
            ),
            "flip_flop_pct": round(
                accuracy_result.temporal_consistency.flip_flop_fraction * 100, 2,
            ),
        },
        "output_files": {
            "visualisations": [str(p) for p in vis_files],
            "rasters": [str(p) for p in raster_files],
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE  (%s files in %.0fs)", len(vis_files) + len(raster_files), elapsed)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 72)

    # Print key metrics
    click.echo(f"\n{'='*60}")
    click.echo(f"  QUANTUM LAND-COVER CHANGE DETECTOR  v{__version__}")
    click.echo(f"{'='*60}")
    click.echo(f"  Years classified:  {len(change_result.years)}")
    click.echo(f"  Overall change:    {summary['change_pct_overall']:.1f}%")
    click.echo(f"  Accuracy (OA):     {summary['accuracy']['overall_accuracy']:.1f}%")
    click.echo(f"  Kappa:             {summary['accuracy']['kappa']:.4f}")
    click.echo(f"  Runtime:           {elapsed:.0f}s")
    click.echo(f"  Output:            {output_dir.resolve()}")
    click.echo(f"{'='*60}\n")


def _log_acquisition(sensor_stack) -> None:
    """Log summary of acquired scenes."""
    logger.info("  Landsat: %d, S2: %d, SAR: %d, NAIP: %d",
                sensor_stack.landsat_count, sensor_stack.sentinel2_count,
                sensor_stack.sentinel1_count, sensor_stack.naip_count)


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
