#!/usr/bin/env python
"""
demo_test.py
============
Quick demonstration of the high-resolution building + canopy detector.

Uses a residential neighbourhood in Austin, TX with a rich mix of
buildings and tree canopy — a good test for both detection pipelines
and the species classifier.

Run::

    python demo_test.py
"""

from pathlib import Path

from hires_detector import (
    AOIBuilder,
    HiResImageryFetcher,
    HiResAnalyser,
    HiResOutputWriter,
)


def main() -> None:
    # Austin, TX — mixed residential / trees near Zilker Park
    LON, LAT  = -97.7690, 30.2630
    BUFFER_KM = 0.75
    OUTPUT_DIR = Path("output_hires_austin")

    print("=" * 62)
    print("  Hi-Res Building + Canopy Detector — Demo (Austin, TX)")
    print("=" * 62)

    # ── 1. Define AOI ─────────────────────────────────────────────────
    aoi = AOIBuilder.from_point(LON, LAT, buffer_km=BUFFER_KM)
    AOIBuilder.summarise(aoi)

    # ── 2. Fetch imagery ──────────────────────────────────────────────
    fetcher = HiResImageryFetcher(
        aoi,
        target_resolution=1.0,
        naip_year_range=(2019, 2023),
        s1_fallback=True,          # use S1 if no Capella data available
    )
    imagery = fetcher.fetch_all(verbose=True)
    print(imagery)

    # ── 3. Run analysis ──────────────────────────────────────────────
    analyser = HiResAnalyser(
        imagery,
        building_threshold=0.35,
        ndvi_threshold=0.30,
        n_species_clusters=5,
        min_building_area=25.0,
    )
    result = analyser.run(verbose=True)

    # ── 4. Export outputs ─────────────────────────────────────────────
    writer = HiResOutputWriter(result, OUTPUT_DIR)
    writer.write_all(verbose=True)

    print(f"\n{'=' * 62}")
    print(f"  All outputs saved → {OUTPUT_DIR.resolve()}/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
