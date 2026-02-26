"""
Shapefile Health Checker — CLI Entry Point
==========================================
Installed as the ``geo-check`` command via ``pyproject.toml``.

Usage:
    geo-check --input data/parcels.shp --output output/report.md
    geo-check --input data/roads.geojson --output report.html --format html
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from src.shapefile_health_checker.checker import ShapefileHealthChecker
from shared.python.exceptions import GeoScriptHubError


@click.command(
    name="geo-check",
    help=(
        "Validate a vector file (Shapefile, GeoJSON, GeoPackage) against "
        "a suite of six health checks and write a report."
    ),
)
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Path to the vector file to check.\n"
        # PLACEHOLDER: replace with your actual input file path
        "Example: --input data/parcels.shp"
    ),
)
@click.option(
    "--output", "-o",
    "output_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Path for the output report file.\n"
        # PLACEHOLDER: replace with your desired output path
        "Example: --output output/health_report.md"
    ),
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["markdown", "html"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help=(
        "Report output format.\n"
        # PLACEHOLDER: choose 'markdown' for a .md file or 'html' for a
        #              browser-viewable .html file
    ),
)
@click.option(
    "--skip-check",
    "skip_checks",
    multiple=True,
    help=(
        "Name of a check to skip (can be repeated).\n"
        "Available check names: null-geometry, self-intersection, "
        "duplicate-features, crs-presence, encoding, extent-sanity\n"
        # PLACEHOLDER: add --skip-check duplicate-features if your data
        #              is known to have intentional duplicate geometries
        "Example: --skip-check duplicate-features --skip-check encoding"
    ),
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug-level logging.",
)
def main(
    input_path: Path,
    output_path: Path,
    report_format: str,
    skip_checks: tuple[str, ...],
    verbose: bool,
) -> None:
    """CLI entry point — wires Click options into ShapefileHealthChecker."""
    from src.shapefile_health_checker.checker import DEFAULT_CHECKS  # noqa: PLC0415

    # Filter out any checks the user wants to skip
    skip_normalized = {s.lower().replace(" ", "-") for s in skip_checks}
    active_checks = [
        c for c in DEFAULT_CHECKS
        if c.name.lower().replace(" ", "-").replace("/", "-").split()[0] not in skip_normalized
    ]

    tool = ShapefileHealthChecker(
        input_path=input_path,
        output_path=output_path,
        report_format=report_format,  # type: ignore[arg-type]
        checks=active_checks,
        verbose=verbose,
    )

    try:
        tool.run()
        if tool.report:
            r = tool.report
            click.echo(
                f"\nResults: {r.passed_count} passed | "
                f"{r.warning_count} warnings | "
                f"{r.failed_count} failed"
            )
            click.echo(f"Report written to: {output_path}")
    except GeoScriptHubError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
