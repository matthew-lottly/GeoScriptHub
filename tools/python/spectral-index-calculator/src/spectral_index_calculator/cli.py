"""
Spectral Index Calculator — CLI Entry Point
=============================================
Exposes :class:`~spectral_index_calculator.calculator.SpectralIndexCalculator`
as the ``geo-spectral`` command.

Usage::

    geo-spectral \\
        --red  LC08_B4.TIF \\
        --nir  LC08_B5.TIF \\
        --green LC08_B3.TIF \\
        --blue LC08_B2.TIF \\
        --index NDVI,NDWI,SAVI,EVI \\
        --output-dir output/spectral

Run ``geo-spectral --help`` for the full option list.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from spectral_index_calculator.calculator import (
    ALL_STRATEGIES,
    BandFileMap,
    EVIStrategy,
    IndexStrategy,
    NDVIStrategy,
    NDWIStrategy,
    SAVIStrategy,
    SpectralIndexCalculator,
)

logger = logging.getLogger("geoscripthub.spectral_index_calculator.cli")

# Registry of available index names → strategy factory
_STRATEGY_REGISTRY: dict[str, IndexStrategy] = {
    "NDVI": NDVIStrategy(),
    "NDWI": NDWIStrategy(),
    "SAVI": SAVIStrategy(),
    "EVI": EVIStrategy(),
}


def _parse_index_list(raw: str) -> list[str]:
    """Parse a comma-separated index list into uppercase tokens.

    Args:
        raw: Comma-separated string, e.g. ``"NDVI,NDWI,EVI"``.

    Returns:
        List of uppercase index names.
    """
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@click.command("geo-spectral")
@click.option(
    "--red",
    "red_band",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    # PLACEHOLDER: Path to the red band GeoTIFF
    # Landsat 8/9: B4.TIF  |  Sentinel-2: B04.tif
    help="Red band file path (required for NDVI, SAVI, EVI).",
)
@click.option(
    "--nir",
    "nir_band",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    # PLACEHOLDER: Path to the near-infrared band GeoTIFF
    # Landsat 8/9: B5.TIF  |  Sentinel-2: B08.tif
    help="NIR band file path (required for NDVI, NDWI, SAVI, EVI).",
)
@click.option(
    "--green",
    "green_band",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    # PLACEHOLDER: Path to the green band GeoTIFF
    # Landsat 8/9: B3.TIF  |  Sentinel-2: B03.tif
    help="Green band file path (required for NDWI).",
)
@click.option(
    "--blue",
    "blue_band",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    # PLACEHOLDER: Path to the blue band GeoTIFF
    # Landsat 8/9: B2.TIF  |  Sentinel-2: B02.tif
    help="Blue band file path (required for EVI).",
)
@click.option(
    "--index",
    "index_list",
    default="NDVI",
    show_default=True,
    # PLACEHOLDER: Comma-separated list of indices to compute
    # Valid values: NDVI, NDWI, SAVI, EVI — or "ALL" to compute every available index
    help="Comma-separated indices to compute (NDVI, NDWI, SAVI, EVI) or ALL.",
)
@click.option(
    "--savi-l",
    "savi_soil_factor",
    default=0.5,
    show_default=True,
    type=float,
    # PLACEHOLDER: SAVI soil brightness correction factor L
    # 0.25 = dense vegetation, 0.5 = intermediate (default), 1.0 = very sparse
    help="SAVI soil brightness correction factor L (default 0.5).",
)
@click.option(
    "--output-dir",
    "output_dir",
    default="output",
    show_default=True,
    # PLACEHOLDER: Directory where output GeoTIFFs will be written, one per index
    help="Directory for output GeoTIFF files.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def cli(
    red_band: str | None,
    nir_band: str | None,
    green_band: str | None,
    blue_band: str | None,
    index_list: str,
    savi_soil_factor: float,
    output_dir: str,
    verbose: bool,
) -> None:
    """Compute spectral indices (NDVI, NDWI, SAVI, EVI) from satellite band GeoTIFFs.

    Each computed index is saved as a single-band float32 GeoTIFF in OUTPUT_DIR.

    \b
    Examples:
        # Compute NDVI only (Landsat 8)
        geo-spectral --red LC08_B4.TIF --nir LC08_B5.TIF --index NDVI

        # Compute NDVI + NDWI (Sentinel-2)
        geo-spectral --red B04.tif --nir B08.tif --green B03.tif --index NDVI,NDWI

        # Compute all indices with Landsat 8
        geo-spectral --red LC08_B4.TIF --nir LC08_B5.TIF \\
                     --green LC08_B3.TIF --blue LC08_B2.TIF \\
                     --index ALL --output-dir results/
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build BandFileMap from provided options
    band_files: BandFileMap = {}  # type: ignore[assignment]
    if red_band:
        band_files["red"] = Path(red_band)  # type: ignore[index]
    if nir_band:
        band_files["nir"] = Path(nir_band)  # type: ignore[index]
    if green_band:
        band_files["green"] = Path(green_band)  # type: ignore[index]
    if blue_band:
        band_files["blue"] = Path(blue_band)  # type: ignore[index]

    if not band_files:
        click.echo("Error: At least one band file must be provided. See --help.", err=True)
        sys.exit(1)

    # Resolve requested strategies
    raw_indices = _parse_index_list(index_list)
    if "ALL" in raw_indices:
        strategies = list(ALL_STRATEGIES)
    else:
        strategies = []
        invalid = []
        for idx_name in raw_indices:
            if idx_name in _STRATEGY_REGISTRY:
                strategies.append(_STRATEGY_REGISTRY[idx_name])
            else:
                invalid.append(idx_name)
        if invalid:
            click.echo(
                f"Error: Unknown index name(s): {', '.join(invalid)}. "
                f"Valid options: {', '.join(_STRATEGY_REGISTRY)}",
                err=True,
            )
            sys.exit(1)

    # Override SAVI soil factor if user supplied one
    strategies = [
        SAVIStrategy(soil_factor=savi_soil_factor)
        if isinstance(s, SAVIStrategy)
        else s
        for s in strategies
    ]

    # Run the calculator
    tool = SpectralIndexCalculator(
        band_files=band_files,
        output_dir=Path(output_dir),
        strategies=strategies,
        verbose=verbose,
    )
    try:
        tool.run()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"\n{len(tool.results)} index raster(s) written to: {output_dir}")
    for result in tool.results:
        click.echo(f"  {result}")



if __name__ == "__main__":
    cli()
