#!/usr/bin/env python
"""Debug script to identify where step 4 crashes for Savannah, GA."""
import sys
import traceback
import faulthandler

faulthandler.enable()

sys.path.insert(0, "tools/python/hires-building-canopy-detector/src")

from pathlib import Path
from hires_detector import (
    AOIBuilder,
    HiResImageryFetcher,
    HiResAnalyser,
    HiResOutputWriter,
)

LOG = Path("outputs/savannah_debug.log")
LOG.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main() -> None:
    log("=== Savannah debug run ===")

    aoi = AOIBuilder.from_point(-81.0998, 32.0678, buffer_km=0.75)
    log("AOI built")

    fetcher = HiResImageryFetcher(
        aoi,
        target_resolution=1.0,
        naip_year_range=(2019, 2023),
        s1_fallback=True,
    )
    imagery = fetcher.fetch_all(verbose=True)
    log(f"Imagery: {imagery}")

    a = HiResAnalyser(
        imagery,
        building_threshold=0.35,
        ndvi_threshold=0.28,
        n_species_clusters=5,
        min_building_area=25.0,
    )
    p = a.params
    log("Analyser created")

    # Step 1
    log("1a: to_db ...")
    sar_db = a._to_db(imagery.sar)
    log(f"1a done: range [{sar_db.min():.2f}, {sar_db.max():.2f}]")

    log("1b: lee_filter ...")
    sar_filt = a._lee_filter(sar_db, p["lee_window"])
    log(f"1b done: range [{sar_filt.min():.2f}, {sar_filt.max():.2f}]")

    # Step 2
    log("2a: MBI ...")
    mbi = a._morphological_building_index(sar_filt, p["mbi_scales"], p["mbi_angles"])
    log(f"2a done: range [{mbi.min():.3f}, {mbi.max():.3f}]")

    log("2b: contrast ...")
    contrast = a._local_contrast(sar_filt, p["contrast_window"])
    log(f"2b done")

    log("2c: edges ...")
    edges = a._edge_density(sar_filt, p["edge_sigma"])
    log(f"2c done")

    log("2d: shadows ...")
    shadows = a._shadow_detection(sar_filt, p["shadow_k"])
    log(f"2d done")

    # Step 3
    log("3a: NDVI ...")
    ndvi = a._compute_ndvi(imagery.naip)
    log(f"3a done: range [{ndvi.min():.3f}, {ndvi.max():.3f}]")

    # Step 4
    log("4a: building_fusion ...")
    bldg_score = a._building_fusion(mbi, contrast, edges, ndvi, shadows)
    log(f"4a done: range [{bldg_score.min():.3f}, {bldg_score.max():.3f}]")

    log("4b: morph cleanup ...")
    bldg_mask = a._morphological_cleanup(
        bldg_score > p["building_threshold"],
        p["morph_cleanup_iter"],
    )
    log(f"4b done: {int(bldg_mask.sum())} building pix")

    log("4c: vectorize_footprints ...")
    raw_fp = a._vectorize_footprints(
        bldg_mask, bldg_score, imagery.transform, str(imagery.crs),
        p["min_building_area"],
    )
    log(f"4c done: {len(raw_fp)} raw footprints")

    log("4d: regularize_footprints ...")
    footprints = a._regularize_footprints(raw_fp, bldg_score, ndvi, imagery.transform)
    log(f"4d done: {len(footprints)} regularized footprints")

    log("4e: rasterize ...")
    H, W = imagery.height, imagery.width
    bldg_mask_final = a._rasterize_footprints(footprints, H, W, imagery.transform)
    log(f"4e done: {int(bldg_mask_final.sum())} final pix")

    # Steps 5-7 + export via run()
    log("Running full pipeline ...")
    result = a.run(verbose=True)

    writer = HiResOutputWriter(result, Path("outputs/hires_savannah"))
    writer.write_all(verbose=True)
    log("ALL DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        with open(LOG, "a") as f:
            traceback.print_exc(file=f)
        sys.exit(1)
