"""
Batch Coordinate Transformer — CLI Entry Point
===============================================
Command-line interface built with Click.  Installed as the ``geo-transform``
command via ``pyproject.toml``.

Usage:
    geo-transform --input data/points.csv --output out/points_wgs84.csv \\
                  --from-crs EPSG:32614 --to-crs EPSG:4326

Run ``geo-transform --help`` for a full list of options.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from src.batch_coord_transformer.transformer import (
    CoordinateTransformer,
    TransformerConfig,
)
from shared.python.exceptions import GeoScriptHubError


@click.command(
    name="geo-transform",
    help=(
        "Reproject coordinate pairs in a CSV file from one CRS to another.\n\n"
        "Reads INPUT_FILE, transforms every row's coordinates from FROM_CRS "
        "to TO_CRS, and writes the result to OUTPUT_FILE."
    ),
)
# ---------------------------------------------------------------------------
# Required arguments
# ---------------------------------------------------------------------------
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the input CSV file.",
)
@click.option(
    "--output", "-o",
    "output_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path for the output file. Parent directories are created if absent.",
)
@click.option(
    "--from-crs",
    required=True,
    help="Source coordinate reference system (e.g. EPSG:32614).",
)
@click.option(
    "--to-crs",
    required=True,
    help="Target coordinate reference system (e.g. EPSG:4326).",
)
# ---------------------------------------------------------------------------
# Optional arguments
# ---------------------------------------------------------------------------
@click.option(
    "--lon-col",
    default="longitude",
    show_default=True,
    help="Column name containing X / longitude / easting values.",
)
@click.option(
    "--lat-col",
    default="latitude",
    show_default=True,
    help="Column name containing Y / latitude / northing values.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "geojson"], case_sensitive=False),
    default="csv",
    show_default=True,
    help="Output file format.  Choose 'csv' to keep tabular structure or "
         "'geojson' to produce a GeoJSON FeatureCollection.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug-level logging output.",
)
def main(
    input_path: Path,
    output_path: Path,
    from_crs: str,
    to_crs: str,
    lon_col: str,
    lat_col: str,
    output_format: str,
    verbose: bool,
) -> None:
    """CLI entry point — wires Click options into CoordinateTransformer."""
    config = TransformerConfig(
        from_crs=from_crs,
        to_crs=to_crs,
        lon_col=lon_col,
        lat_col=lat_col,
        output_format=output_format,  # type: ignore[arg-type]
    )

    tool = CoordinateTransformer(
        input_path=input_path,
        output_path=output_path,
        config=config,
        verbose=verbose,
    )

    try:
        tool.run()
    except GeoScriptHubError as exc:
        # User-facing errors: print a clean message, no stack trace
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
