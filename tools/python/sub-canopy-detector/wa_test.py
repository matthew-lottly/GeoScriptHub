"""
wa_test.py
==========
End-to-end test run of the sub-canopy detector for a temperate rainforest
patch in south-western Washington State.

The target area (Capitol State Forest / Chehalis Basin foothills) is
dominated by dense Douglas-Fir and Western Red Cedar stands that present
an excellent challenge for sub-canopy structure detection due to:
  - Thick, year-round canopy with minimal leaf-off signal
  - Frequent cloud cover requiring aggressive cloud masking
  - Managed timber lands where equipment sheds and access roads
    may exist under partial canopy

Run:
    python wa_test.py
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
# 1. Define AOI -- temperate conifer forest, SW Washington
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Sub-Canopy Detector -- Washington State test")
print("=" * 60)

AOI = AOIBuilder.from_point(
    lon=-123.72187893808666,
    lat=46.781013459422326,
    buffer_km=2.5,
)
AOIBuilder.summarise(AOI)

# ---------------------------------------------------------------------------
# 2. Fetch imagery (2022-2024, ascending orbit, <=25% cloud)
# ---------------------------------------------------------------------------
fetcher = ImageryFetcher(
    aoi             = AOI,
    start_date      = "2022-01-01",
    end_date        = "2024-12-31",
    max_cloud_cover = 25,       # PNW is cloudier than average
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
    forest_ndvi_threshold = 0.50,   # Dense PNW conifers have high NDVI
    water_ndwi_threshold  = 0.12,
    stability_floor       = 0.65,
    thresh_high           = 0.63,
    thresh_medium         = 0.42,
    min_footprint_area    = 60,
)

result = analyser.run(verbose=True)

# ---------------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------------
writer = OutputWriter(
    result      = result,
    aoi         = AOI,
    output_dir  = "./outputs/washington",
    study_name  = "wa_test",
)
saved = writer.save_all(fmt_vector="geojson", verbose=True)

# ---------------------------------------------------------------------------
# 5. SAR time-series + probability histogram saved as PNG
# ---------------------------------------------------------------------------
from sub_canopy_detector.viz import ResultVisualiser

vis = ResultVisualiser(result=result, aoi=AOI, s1_stack=imagery.s1)

fig_ts = vis.sar_timeseries_chart(high_only=True)
ts_path = "./outputs/washington/wa_test_timeseries.png"
fig_ts.savefig(ts_path, dpi=120, bbox_inches="tight")
plt.close(fig_ts)
print(f"  Saved PNG    : wa_test_timeseries.png")

fig_hist = vis.probability_histogram()
hist_path = "./outputs/washington/wa_test_histogram.png"
fig_hist.savefig(hist_path, dpi=120, bbox_inches="tight")
plt.close(fig_hist)
print(f"  Saved PNG    : wa_test_histogram.png")

print("\nTest complete.  All outputs in ./outputs/washington/")
