"""
Raster Band Stats Reporter — CLI Entry Point
=============================================
Installed as the ``geo-raster-stats`` command via ``pyproject.toml``.

Usage:
    geo-raster-stats --input data/landsat.tif --output output/stats.json
    geo-raster-stats --input data/sentinel2.tif --output stats.csv --format csv --bands 1,2,3,4
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from src.raster_band_stats.stats import BandStatsConfig, BandStatsReporter
from shared.python.exceptions import GeoScriptHubError


@click.command(
    name="geo-raster-stats",
    help="Compute per-band statistics for a GeoTIFF and write the results to JSON or CSV.",
)
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Path to the input raster file (GeoTIFF, .img, etc.).\n"
        # PLACEHOLDER: replace with your raster file path
        "Example: --input data/landsat_scene.tif"
    ),
)
@click.option(
    "--output", "-o", "output_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Path for the output stats file.\n"
        # PLACEHOLDER: replace with your desired output path
        "Example: --output output/band_stats.json"
    ),
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv"], case_sensitive=False),
    default="json",
    show_default=True,
    help=(
        "Output file format.\n"
        # PLACEHOLDER: choose 'json' for machine-readable or 'csv' for spreadsheet use
    ),
)
@click.option(
    "--bands",
    default="",
    help=(
        "Comma-separated list of 1-based band indices to process.  "
        "Omit to process all bands.\n"
        # PLACEHOLDER: specify bands like --bands 1,4 to process only bands 1 and 4
        "Example: --bands 1,3,4   (processes bands 1, 3, and 4 only)"
    ),
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
def main(
    input_path: Path,
    output_path: Path,
    output_format: str,
    bands: str,
    verbose: bool,
) -> None:
    """CLI entry point — wires Click options into BandStatsReporter."""
    band_list = [int(b.strip()) for b in bands.split(",") if b.strip()] or None

    config = BandStatsConfig(
        output_format=output_format,  # type: ignore[arg-type]
        bands=band_list,
    )

    tool = BandStatsReporter(input_path, output_path, config, verbose=verbose)

    try:
        tool.run()
        click.echo(f"\nStats written to: {output_path}")
        for s in tool.band_stats:
            click.echo(f"  {s}")
    except GeoScriptHubError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
