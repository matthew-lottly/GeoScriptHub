"""
cli.py
======
Command-line interface for the high-resolution building + canopy detector.

Usage
-----
::

    python -m hires_detector --lon -97.743 --lat 30.267 --buffer-km 0.75
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .aoi import AOIBuilder
from .fetcher import HiResImageryFetcher
from .analysis import HiResAnalyser
from .export import HiResOutputWriter


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="hires_detector",
        description=(
            "Hi-Res Building Footprint + Tree Canopy Detector — "
            "Capella X-band SAR + NAIP optical imagery"
        ),
    )

    # Location
    ap.add_argument("--lon",        type=float, required=True, help="Longitude (WGS-84)")
    ap.add_argument("--lat",        type=float, required=True, help="Latitude  (WGS-84)")
    ap.add_argument("--buffer-km",  type=float, default=1.0,   help="Half-side of AOI square (km)")

    # Output
    ap.add_argument("--output-dir", type=str,   default="output_hires", help="Output directory")

    # Resolution
    ap.add_argument("--resolution", type=float, default=1.0, help="Common pixel size (m)")

    # Thresholds
    ap.add_argument("--building-threshold", type=float, default=0.35)
    ap.add_argument("--ndvi-threshold",     type=float, default=0.30)
    ap.add_argument("--species-clusters",   type=int,   default=5)

    # Fallback
    ap.add_argument("--no-s1-fallback", action="store_true",
                    help="Disable Sentinel-1 SAR fallback if no Capella data found")

    args = ap.parse_args()

    # 1. AOI
    aoi = AOIBuilder.from_point(args.lon, args.lat, args.buffer_km)
    AOIBuilder.summarise(aoi)

    # 2. Fetch imagery
    fetcher = HiResImageryFetcher(
        aoi,
        target_resolution=args.resolution,
        s1_fallback=not args.no_s1_fallback,
    )
    imagery = fetcher.fetch_all(verbose=True)
    print(imagery)

    # 3. Analyse
    analyser = HiResAnalyser(
        imagery,
        building_threshold=args.building_threshold,
        ndvi_threshold=args.ndvi_threshold,
        n_species_clusters=args.species_clusters,
    )
    result = analyser.run(verbose=True)

    # 4. Export
    writer = HiResOutputWriter(result, args.output_dir)
    writer.write_all(verbose=True)

    print(f"\n✓ All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
