"""
coast_test.py
=============
End-to-end test for a coastal area near Westport / Grays Harbor, WA.

The target area is a challenging coastal environment featuring:
  - Small-town buildings (Westport, Ocosta) mixed with dense forest
  - Tidal mudflats and estuaries that produce SAR false positives
  - Salt-marsh vegetation that can confuse NDVI-based forest masking
  - Route 105 corridor with scattered residential structures

This test validates the improved building regularisation pipeline
with 7-component scoring (solidity + NDWI water penalty) against
a landscape where tidal features previously generated massive
false positives.

Run:
    python coast_test.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sub_canopy_detector.aoi      import AOIBuilder
from sub_canopy_detector.fetcher   import ImageryFetcher
from sub_canopy_detector.analysis  import SubCanopyAnalyser
from sub_canopy_detector.export    import OutputWriter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Define AOI -- coastal town near Grays Harbor, WA
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Sub-Canopy Detector -- Grays Harbor Coast test")
print("=" * 60)

AOI = AOIBuilder.from_point(
    lon=-124.05633071432928,
    lat=46.86414242464418,
    buffer_km=2.5,
)
AOIBuilder.summarise(AOI)

# ---------------------------------------------------------------------------
# 2. Fetch imagery (2022-2024, ascending orbit, <=20% cloud)
# ---------------------------------------------------------------------------
fetcher = ImageryFetcher(
    aoi             = AOI,
    start_date      = "2022-01-01",
    end_date        = "2024-12-31",
    max_cloud_cover = 20,
    orbit_direction = "ascending",
)

imagery = fetcher.fetch_all(verbose=True)
print(imagery)

# ---------------------------------------------------------------------------
# 3. Run analysis -- tuned for coastal environment
# ---------------------------------------------------------------------------
analyser = SubCanopyAnalyser(
    aoi                   = AOI,
    imagery               = imagery,
    # -- Spectral / mask thresholds --
    forest_ndvi_threshold = 0.25,   # Low: include open residential + mixed pixels
    water_ndwi_threshold  = 0.08,   # Aggressive water masking for tidal zone
    stability_floor       = 0.65,
    # -- Detection thresholds --
    thresh_high           = 0.58,
    thresh_medium         = 0.38,
    min_footprint_area    = 35,     # Small: catch ~3-4 px houses at 10 m
    morph_open_iterations = 0,      # Disable opening -- preserves sub-pixel detections;
                                    # geometric filters in regularisation clean noise
    # -- Building regularisation (tighter for coastal noise) --
    min_compactness       = 0.18,   # Kills fractal tidal shapes
    min_solidity          = 0.45,   # Kills branching channel outlines
    min_building_score    = 0.38,   # Stricter score gate
    max_footprint_area    = 12000,  # Smaller ceiling for coastal area
    max_aspect_ratio      = 8.0,
)

result = analyser.run(verbose=True)

# ---------------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------------
writer = OutputWriter(
    result      = result,
    aoi         = AOI,
    output_dir  = "./outputs/coast",
    study_name  = "coast_test",
)
saved = writer.save_all(fmt_vector="geojson", verbose=True)

# ---------------------------------------------------------------------------
# 5. SAR time-series + probability histogram
# ---------------------------------------------------------------------------
from sub_canopy_detector.viz import ResultVisualiser

vis = ResultVisualiser(result=result, aoi=AOI, s1_stack=imagery.s1)

fig_ts = vis.sar_timeseries_chart(high_only=True)
ts_path = "./outputs/coast/coast_test_timeseries.png"
fig_ts.savefig(ts_path, dpi=120, bbox_inches="tight")
plt.close(fig_ts)
print(f"  Saved PNG    : coast_test_timeseries.png")

fig_hist = vis.probability_histogram()
hist_path = "./outputs/coast/coast_test_histogram.png"
fig_hist.savefig(hist_path, dpi=120, bbox_inches="tight")
plt.close(fig_hist)
print(f"  Saved PNG    : coast_test_histogram.png")

print("\nTest complete.  All outputs in ./outputs/coast/")
