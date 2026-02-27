"""
houston_test.py
===============
End-to-end test run of the sub-canopy detector for a forested patch
near Houston, Texas.

The Sam Houston National Forest (Montgomery County, TX) contains mixed
pine-hardwood stands where encroaching residential development and camp
structures occasionally sit under partial canopy cover -- a real use-case
for sub-canopy detection.

Run:
    python houston_test.py
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
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Define AOI -- northern edge of Sam Houston National Forest
#    (~10 km south of Huntsville, TX)
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Sub-Canopy Detector -- Houston / Sam Houston NF test")
print("=" * 60)

# A ~5x5 km patch inside the national forest
AOI = AOIBuilder.from_bbox(
    min_lon = -95.62,
    min_lat =  30.48,
    max_lon = -95.57,
    max_lat =  30.53,
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
# 3. Run analysis
# ---------------------------------------------------------------------------
analyser = SubCanopyAnalyser(
    aoi                   = AOI,
    imagery               = imagery,
    forest_ndvi_threshold = 0.45,   # East Texas pine forest is less NDVI-dense than tropics
    water_ndwi_threshold  = 0.10,
    stability_floor       = 0.65,
    thresh_high           = 0.60,   # slightly looser for temperate forest
    thresh_medium         = 0.40,
    min_footprint_area    = 50,
)

result = analyser.run(verbose=True)

# ---------------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------------
writer = OutputWriter(
    result      = result,
    aoi         = AOI,
    output_dir  = "./outputs/houston",
    study_name  = "houston_test",
)
saved = writer.save_all(fmt_vector="geojson", verbose=True)

# ---------------------------------------------------------------------------
# 5. SAR time-series + probability histogram saved as PNG
# ---------------------------------------------------------------------------
from sub_canopy_detector.viz import ResultVisualiser

vis = ResultVisualiser(result=result, aoi=AOI, s1_stack=imagery.s1)

fig_ts = vis.sar_timeseries_chart(high_only=True)
ts_path = "./outputs/houston/houston_test_timeseries.png"
fig_ts.savefig(ts_path, dpi=120, bbox_inches="tight")
plt.close(fig_ts)
print(f"  Saved PNG    : houston_test_timeseries.png")

fig_hist = vis.probability_histogram()
hist_path = "./outputs/houston/houston_test_histogram.png"
fig_hist.savefig(hist_path, dpi=120, bbox_inches="tight")
plt.close(fig_hist)
print(f"  Saved PNG    : houston_test_histogram.png")

print("\nTest complete.  All outputs in ./outputs/houston/")
