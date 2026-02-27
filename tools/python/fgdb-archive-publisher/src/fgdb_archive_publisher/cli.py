"""
FGDB Archive Publisher — CLI Entry Point
=========================================
Command-line interface built with Click.  Installed as the
``geo-archive`` command via ``pyproject.toml``.

Usage:
    geo-archive --config pipeline.json --output backups/

Run ``geo-archive --help`` for a full list of options.
"""

from __future__ import annotations

from pathlib import Path

import click  # type: ignore[import-unresolved]

from src.fgdb_archive_publisher.pipeline import ArchivePipeline
from shared.python.exceptions import GeoScriptHubError


@click.command(
    name="geo-archive",
    help=(
        "Archive ArcGIS portal data to a File Geodatabase, clean schemas, "
        "validate topology, and optionally republish.\n\n"
        "Reads a JSON configuration file that defines portal connection, "
        "layer URLs, schema changes, topology rules, and publish settings."
    ),
)
@click.option(
    "--config", "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the JSON pipeline configuration file.",
)
@click.option(
    "--output", "-o",
    "output_path",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the output FGDB will be created.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug-level logging output.",
)
def main(config_path: Path, output_path: Path, verbose: bool) -> None:
    """Run the full archive-clean-validate-publish pipeline.

    Args:
        config_path: Path to the JSON configuration file.
        output_path: Directory for the output File Geodatabase.
        verbose: When set, debug messages are printed.
    """
    try:
        pipeline = ArchivePipeline(
            input_path=config_path,
            output_path=output_path,
            verbose=verbose,
        )
        pipeline.run()
    except GeoScriptHubError as exc:
        click.secho(f"Error: {exc.message}", fg="red", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.secho(f"Unexpected error: {exc}", fg="red", err=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
