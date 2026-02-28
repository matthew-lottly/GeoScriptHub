#!/usr/bin/env python
"""
savannah_test.py
================
Test the high-resolution building + canopy detector on Savannah, GA.

Savannah is famous for its live-oak canopy draped over historic squares,
giving an excellent mix of distinct buildings and dense tree cover to
stress-test both detection pipelines and the species classifier.

Run::

    python savannah_test.py
"""

from pathlib import Path

from hires_detector import (
    AOIBuilder,
    HiResImageryFetcher,
    HiResAnalyser,
    HiResOutputWriter,
)


def main() -> None:
    # Savannah, GA — historic district around Forsyth Park
    LON, LAT  = -81.0998, 32.0678
    BUFFER_KM = 0.75
    OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "hires_savannah"

    print("=" * 62)
    print("  Hi-Res Building + Canopy Detector — Savannah, GA")
    print("=" * 62)

    # ── 1. Define AOI ─────────────────────────────────────────────────
    aoi = AOIBuilder.from_point(LON, LAT, buffer_km=BUFFER_KM)
    AOIBuilder.summarise(aoi)

    # ── 2. Fetch imagery ──────────────────────────────────────────────
    fetcher = HiResImageryFetcher(
        aoi,
        target_resolution=1.0,
        naip_year_range=(2019, 2023),
        s1_fallback=True,
    )
    imagery = fetcher.fetch_all(verbose=True)
    print(imagery)

    # ── 3. Run analysis ──────────────────────────────────────────────
    analyser = HiResAnalyser(
        imagery,
        # Let adaptive S1 thresholds handle building_threshold automatically
        ndvi_threshold=0.28,        # slightly lower — Savannah has dense shade
        n_species_clusters=5,
        min_building_area=25.0,
    )
    result = analyser.run(verbose=True)

    # ── 4. Export outputs ─────────────────────────────────────────────
    writer = HiResOutputWriter(result, OUTPUT_DIR)
    writer.write_all(verbose=True)

    print(f"\n{'=' * 62}")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
