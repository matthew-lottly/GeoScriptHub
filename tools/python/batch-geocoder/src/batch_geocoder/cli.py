"""
Batch Geocoder — CLI Entry Point
=================================
Installed as the ``geo-geocode`` command via ``pyproject.toml``.

Usage:
    geo-geocode --input data/addresses.csv --output output/addresses.geojson \\
                --address-col full_address --backend nominatim \\
                --user-agent "my-app/1.0"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from src.batch_geocoder.geocoder import (
    BatchGeocoder,
    GoogleBackend,
    NominatimBackend,
)
from shared.python.exceptions import GeoScriptHubError


@click.command(
    name="geo-geocode",
    help="Convert a CSV of addresses to a GeoJSON FeatureCollection.",
)
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Path to the input CSV file.\n"
        # PLACEHOLDER: replace with your actual CSV path
        "Example: --input data/addresses.csv"
    ),
)
@click.option(
    "--output", "-o", "output_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Path for the output GeoJSON file.\n"
        # PLACEHOLDER: replace with your desired output path
        "Example: --output output/addresses.geojson"
    ),
)
@click.option(
    "--address-col",
    default="address",
    show_default=True,
    help=(
        "CSV column containing address strings.\n"
        # PLACEHOLDER: replace with the actual column name in your CSV,
        #              e.g. --address-col full_address
    ),
)
@click.option(
    "--backend",
    type=click.Choice(["nominatim", "google"], case_sensitive=False),
    default="nominatim",
    show_default=True,
    help=(
        "Geocoding provider to use.\n"
        # PLACEHOLDER: choose 'nominatim' (free, no key) or 'google' (requires API key)
    ),
)
@click.option(
    "--user-agent",
    default="geoscripthub-geocoder/1.0",
    show_default=True,
    help=(
        "User-agent string for Nominatim (ignored for Google).\n"
        # PLACEHOLDER: replace with a string that identifies your app,
        #              e.g. --user-agent "my-company-geocoder/1.0"
    ),
)
@click.option(
    "--google-api-key",
    default=None,
    envvar="GOOGLE_MAPS_API_KEY",
    help=(
        "Google Maps API key (required when --backend google).\n"
        # PLACEHOLDER: replace YOUR_GOOGLE_API_KEY or set the
        #              GOOGLE_MAPS_API_KEY environment variable instead
        "Can also be set via the GOOGLE_MAPS_API_KEY environment variable."
    ),
)
@click.option(
    "--rate-limit",
    default=1.1,
    show_default=True,
    type=float,
    help=(
        "Seconds to wait between geocoding requests.\n"
        # PLACEHOLDER: keep >= 1.0 for Nominatim; can be ~0.05 for Google
    ),
)
@click.option(
    "--extra-cols",
    default="",
    help=(
        "Comma-separated list of extra CSV columns to include in GeoJSON properties.\n"
        # PLACEHOLDER: replace with your column names,
        #              e.g. --extra-cols name,city,zip_code
        "Example: --extra-cols name,city,zip"
    ),
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
def main(
    input_path: Path,
    output_path: Path,
    address_col: str,
    backend: str,
    user_agent: str,
    google_api_key: str | None,
    rate_limit: float,
    extra_cols: str,
    verbose: bool,
) -> None:
    """CLI entry point — wires Click options into BatchGeocoder."""
    extra = [c.strip() for c in extra_cols.split(",") if c.strip()]

    # Build the backend
    geocoder_backend = None
    if backend == "nominatim":
        geocoder_backend = NominatimBackend(
            user_agent=user_agent, rate_limit_seconds=rate_limit
        )
    elif backend == "google":
        key = google_api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not key:
            click.echo(
                "Error: --google-api-key or GOOGLE_MAPS_API_KEY env var required "
                "when using --backend google",
                err=True,
            )
            sys.exit(1)
        geocoder_backend = GoogleBackend(api_key=key, rate_limit_seconds=rate_limit)

    tool = BatchGeocoder(
        input_path=input_path,
        output_path=output_path,
        address_col=address_col,
        backend=geocoder_backend,
        extra_cols=extra,
        verbose=verbose,
    )

    try:
        tool.run()
        success = sum(1 for r in tool.results if r.success)
        click.echo(f"\nGeoJSON written to: {output_path}")
        click.echo(f"Geocoded: {success}/{len(tool.results)} addresses successfully.")
    except GeoScriptHubError as exc:
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
