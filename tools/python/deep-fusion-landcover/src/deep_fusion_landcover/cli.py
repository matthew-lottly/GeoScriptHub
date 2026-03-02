"""cli.py — Click command-line interface for the Deep Fusion Landcover tool.

Usage
-----
  deep-fusion-landcover run      # acquire data + classify + change detection
  deep-fusion-landcover validate # cross-validate against NLCD
  deep-fusion-landcover visualize# build interactive maps + diagrams
  deep-fusion-landcover flowchart# generate pipeline architecture diagram
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.cli")


# ── CLI root ──────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
@click.version_option(package_name="deep-fusion-landcover")
def main(verbose: bool) -> None:
    """Deep Fusion Landcover — 35-year 12-class Austin TX landcover mapper.

    Combines Landsat/Sentinel/NAIP/3DEP data with a 5-branch stacked ensemble
    (Random Forest, Gradient Boosting, Quantum VQC, TorchGeo CNN, SAM2-OBIA)
    to produce annual landcover maps from 1990 to 2025.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )


# ── run ───────────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--output", "-o",
    default="outputs/austin_landcover",
    show_default=True,
    help="Root output directory.",
)
@click.option(
    "--center-lat", default=30.2672, show_default=True,
    help="AOI centre latitude (WGS-84).",
)
@click.option(
    "--center-lon", default=-97.7431, show_default=True,
    help="AOI centre longitude (WGS-84).",
)
@click.option(
    "--buffer-km", default=55.0, show_default=True,
    help="Radius buffer around centre point (km).",
)
@click.option(
    "--start-year", default=1990, show_default=True,
    help="First year to classify.",
)
@click.option(
    "--end-year", default=2025, show_default=True,
    help="Last year to classify.",
)
@click.option(
    "--resolution", default=10, show_default=True,
    help="Pixel resolution in metres (Sentinel-2 native = 10 m).",
)
@click.option(
    "--no-quantum", is_flag=True, default=False,
    help="Disable the quantum VQC branch.",
)
@click.option(
    "--no-cnn", is_flag=True, default=False,
    help="Disable the CNN encoder branch.",
)
@click.option(
    "--no-obia", is_flag=True, default=False,
    help="Disable SAM2 OBIA segmentation.",
)
@click.option(
    "--n-workers", default=4, show_default=True,
    help="Dask LocalCluster worker count for tile processing.",
)
@click.option(
    "--training-sample-n", default=50_000, show_default=True,
    help="Number of labeled pixels for supervised training.",
)
def run(
    output: str,
    center_lat: float,
    center_lon: float,
    buffer_km: float,
    start_year: int,
    end_year: int,
    resolution: int,
    no_quantum: bool,
    no_cnn: bool,
    no_obia: bool,
    n_workers: int,
    training_sample_n: int,
) -> None:
    """Acquire data, classify, detect change, and save all outputs."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(click.style("Deep Fusion Landcover — run", fg="cyan", bold=True))
    click.echo(f"  AOI centre : {center_lat:.4f}°N  {center_lon:.4f}°E")
    click.echo(f"  Buffer     : {buffer_km} km")
    click.echo(f"  Years      : {start_year}–{end_year}")
    click.echo(f"  Resolution : {resolution} m")
    click.echo(f"  Output     : {output_dir.resolve()}")
    click.echo("")

    # ── Build AOI ────────────────────────────────────────────────────────────
    from .aoi import AOIBuilder

    aoi_builder = AOIBuilder(
        center_lat=center_lat,
        center_lon=center_lon,
        buffer_km=buffer_km,
    )
    aoi_result = aoi_builder.build()
    click.echo(f"  AOI UTM extent : {aoi_result.bbox_utm}")

    tiles = aoi_builder.tile_aoi(aoi_result)
    click.echo(f"  Tiles          : {len(tiles)}")

    # ── Acquire ──────────────────────────────────────────────────────────────
    from .acquisition import MultiSensorAcquisition

    acq = MultiSensorAcquisition(aoi=aoi_result, start_year=start_year, end_year=end_year)
    click.echo("\nAcquiring data from Planetary Computer …")
    sensor_stack = acq.fetch_all()

    # ── Preprocess ───────────────────────────────────────────────────────────
    from .preprocessing import Preprocessor

    preprocessor = Preprocessor(resolution_m=resolution)
    aligned = preprocessor.process(
        landsat=sensor_stack.landsat.dataset if sensor_stack.landsat else None,
        sentinel2=sensor_stack.sentinel2.dataset if sensor_stack.sentinel2 else None,
        sentinel1=sensor_stack.sentinel1.dataset if sensor_stack.sentinel1 else None,
        naip=sensor_stack.naip.dataset if sensor_stack.naip else None,
        dem=sensor_stack.dem.dataset if sensor_stack.dem else None,
    )

    # ── Temporal compositing ─────────────────────────────────────────────────
    from .temporal_compositing import TemporalCompositor

    compositor = TemporalCompositor(start_year=start_year, end_year=end_year)
    composites = compositor.compose(
        landsat=aligned.landsat,
        sentinel2=aligned.sentinel2,
        sentinel1=aligned.sentinel1,
        dem=aligned.dem,
    )
    click.echo(f"  Annual composites built: {len(composites)}")

    # ── LiDAR ────────────────────────────────────────────────────────────────
    from .lidar_processor import LiDARProcessor

    lidar_proc = LiDARProcessor(aoi=aoi_result, cache_dir=output_dir / "cache" / "lidar")
    lidar_products = lidar_proc.process()

    # ── Feature engineering ──────────────────────────────────────────────────
    from .feature_engineering import FeatureEngineer

    feat_eng = FeatureEngineer()

    # ── Sub-canopy detection (most-recent composite) ──────────────────────────
    from .sub_canopy_detector import SubCanopyDetector

    subcanopy_det = SubCanopyDetector()

    # ── SAM2 OBIA (NAIP present-day) ─────────────────────────────────────────
    from .sam_segmenter import SAMSegmenter

    sam_seg = SAMSegmenter() if not no_obia else None

    # ── CNN encoder ──────────────────────────────────────────────────────────
    from .cnn_encoder import CNNEncoder

    cnn = CNNEncoder() if not no_cnn else None

    # ── Ensemble classifier ───────────────────────────────────────────────────
    from .ensemble import EnsembleClassifier

    ensemble = EnsembleClassifier(
        use_quantum=not no_quantum,
        use_cnn=not no_cnn,
        use_obia=not no_obia,
    )

    # ── Classify each year ────────────────────────────────────────────────────
    from .mosaic import TiledMosaicEngine

    mosaic_engine = TiledMosaicEngine(n_workers=n_workers)
    class_maps: dict[int, "np.ndarray"] = {}  # type: ignore[name-defined]

    years = list(range(start_year, end_year + 1))

    with click.progressbar(composites, label="Classifying annual composites") as bar:
        for composite in bar:
            try:
                feat_stack = feat_eng.build(composite, lidar_products=lidar_products)
                result = ensemble.predict(
                    feature_stack=feat_stack.array,
                    year=composite.year,
                )
                class_maps[composite.year] = result.class_map
            except Exception as exc:
                logger.warning("Year %d failed: %s", composite.year, exc)

    if not class_maps:
        click.secho("ERROR: No annual maps produced.", fg="red")
        sys.exit(1)

    # ── Save annual COGs ──────────────────────────────────────────────────────
    import numpy as np
    from rasterio.transform import from_bounds

    annual_dir = output_dir / "landcover_annual"
    annual_dir.mkdir(parents=True, exist_ok=True)

    for yr, cm in class_maps.items():
        H, W = cm.shape
        transform = from_bounds(*aoi_result.bbox_utm, W, H)
        mosaic_engine.write_cog(cm, transform, 32614, annual_dir / f"landcover_{yr}.tif")

    click.echo(f"\n  COG maps saved → {annual_dir}")

    # ── Change detection ──────────────────────────────────────────────────────
    from .change_detection import ChangeDetector

    detector = ChangeDetector()
    sorted_years = sorted(class_maps.keys())
    sorted_maps = [class_maps[y] for y in sorted_years]
    change_result = detector.analyze(sorted_maps, sorted_years)

    change_dir = output_dir / "change_detection"
    detector.save_transitions(change_result, change_dir)
    detector.save_summary(change_result, change_dir)

    # ── Accuracy assessment ───────────────────────────────────────────────────
    from .accuracy import AccuracyAssessor

    assessor = AccuracyAssessor(sample_frac=0.02)
    acc_report = assessor.validate(class_maps)
    assessor.save_report(acc_report, output_dir)

    # ── Visualizations ────────────────────────────────────────────────────────
    from .viz import generate_all_outputs

    generate_all_outputs(
        output_dir=output_dir,
        class_maps=class_maps,
        change_result=change_result,
        aoi_bounds_wgs84=aoi_result.bbox_wgs84,
    )

    # ── Pipeline flowchart ────────────────────────────────────────────────────
    from .flowchart import generate_flowchart

    generate_flowchart(output_dir)

    click.secho("\n✓ Pipeline complete.", fg="green", bold=True)
    click.echo(f"  Results → {output_dir.resolve()}")
    click.echo(f"  OA={acc_report.mean_oa:.3f}  κ={acc_report.mean_kappa:.3f}")


# ── validate ──────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--output", "-o",
    default="outputs/austin_landcover",
    show_default=True,
    help="Root output directory containing landcover_annual/ COGs.",
)
@click.option(
    "--nlcd-dir", default=None,
    help="Local directory containing NLCD IMG/TIF files.",
)
@click.option(
    "--sample-frac", default=0.05, show_default=True,
    help="Fraction of pixels to use for validation [0–1].",
)
def validate(output: str, nlcd_dir: str | None, sample_frac: float) -> None:
    """Cross-validate annual class maps against NLCD reference."""
    import rasterio
    import numpy as np

    output_dir = Path(output)
    annual_dir = output_dir / "landcover_annual"

    if not annual_dir.exists():
        click.secho(f"landcover_annual/ not found in {output_dir}", fg="red")
        sys.exit(1)

    # Load class maps from COGs
    class_maps: dict[int, np.ndarray] = {}
    for p in sorted(annual_dir.glob("landcover_*.tif")):
        year = int(p.stem.split("_")[-1])
        with rasterio.open(p) as src:
            class_maps[year] = src.read(1)

    if not class_maps:
        click.secho("No COG files found.", fg="red")
        sys.exit(1)

    click.echo(f"Validating {len(class_maps)} years …")

    from .accuracy import AccuracyAssessor

    assessor = AccuracyAssessor(
        sample_frac=sample_frac,
        download_nlcd=nlcd_dir is None,
    )
    nlcd_path = Path(nlcd_dir) if nlcd_dir else None
    report = assessor.validate(class_maps, nlcd_dir=nlcd_path)
    assessor.save_report(report, output_dir)

    click.echo(report.summary_df.to_string())
    click.secho(
        f"\nMean OA={report.mean_oa:.3f}  Mean κ={report.mean_kappa:.3f}",
        fg="green", bold=True,
    )


# ── visualize ─────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--output", "-o",
    default="outputs/austin_landcover",
    show_default=True,
    help="Root output directory containing class map COGs.",
)
@click.option(
    "--center-lat", default=30.2672, show_default=True,
)
@click.option(
    "--center-lon", default=-97.7431, show_default=True,
)
@click.option(
    "--buffer-km", default=55.0, show_default=True,
)
def visualize(
    output: str, center_lat: float, center_lon: float, buffer_km: float
) -> None:
    """Regenerate interactive maps, Sankey diagrams, and growth charts."""
    import rasterio
    import numpy as np

    output_dir = Path(output)
    annual_dir = output_dir / "landcover_annual"

    class_maps: dict[int, np.ndarray] = {}
    for p in sorted(annual_dir.glob("landcover_*.tif")):
        year = int(p.stem.split("_")[-1])
        with rasterio.open(p) as src:
            class_maps[year] = src.read(1)

    from .aoi import AOIBuilder
    from .change_detection import ChangeDetector
    from .viz import generate_all_outputs

    aoi_result = AOIBuilder(
        center_lat=center_lat,
        center_lon=center_lon,
        buffer_km=buffer_km,
    ).build()

    sorted_years = sorted(class_maps.keys())
    sorted_maps = [class_maps[y] for y in sorted_years]

    if len(sorted_maps) >= 2:
        change_result = ChangeDetector().analyze(sorted_maps, sorted_years)
    else:
        change_result = None

    generate_all_outputs(
        output_dir=output_dir,
        class_maps=class_maps,
        change_result=change_result,
        aoi_bounds_wgs84=aoi_result.bbox_wgs84,
    )

    click.secho("✓ Visualizations refreshed.", fg="green")
    click.echo(f"  {output_dir.resolve()}")


# ── flowchart ─────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--output", "-o",
    default="outputs/austin_landcover",
    show_default=True,
    help="Directory in which to write pipeline_architecture.html.",
)
def flowchart(output: str) -> None:
    """Generate the pipeline architecture HTML diagram."""
    from .flowchart import generate_flowchart

    p = generate_flowchart(Path(output))
    click.secho(f"✓ Flowchart saved → {p}", fg="green")


if __name__ == "__main__":
    main()
