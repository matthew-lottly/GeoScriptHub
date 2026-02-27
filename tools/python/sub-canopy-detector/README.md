# Sub-Canopy Structure Detector (Python)

Detects hidden built structures beneath forest canopy using fused Sentinel-1 SAR
and Sentinel-2 optical imagery.  This is a Python/local port of the companion Google
Earth Engine script, running entirely from your machine via the
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC API.

No account or API key is required.

---

## What It Does

The tool fuses five independent remote sensing indicators into a single sub-canopy
probability score, then vectorises connected detections into building footprints:

| Indicator | Source | Why |
|-----------|--------|-----|
| SAR temporal stability | Sentinel-1 RTC (VV) | Built structures reflect consistently; vegetation fluctuates |
| Polarimetric ratio (VH/VV) | Sentinel-1 RTC | Double-bounce signature of walls and roofs |
| SAR texture (GLCM homogeneity proxy) | Sentinel-1 RTC | Flat surfaces produce uniform backscatter patterns |
| SAR anomaly (Gaussian residual z-score) | Sentinel-1 RTC | Localised brightness above the canopy average |
| Optical indicator (NDBI anomaly) | Sentinel-2 L2A | Elevated SWIR response in cleared sub-canopy zones |

---

## Requirements

- Python 3.10+
- A working internet connection (imagery is streamed -- not pre-downloaded)

---

## Installation

```bash
cd tools/python/sub-canopy-detector
pip install -e .
```

Or install a development environment with test tools:

```bash
pip install -e ".[dev]"
```

---

## Usage

### Interactive Notebook (recommended)

Open `notebook.ipynb` in VS Code with the Jupyter extension.  All user-facing
settings are in a single configuration cell near the top.  Run cells top-to-bottom
and results appear inline.

### Setting the Analysis Area

The notebook supports four AOI input methods -- uncomment the one you want:

```python
# Option 1 -- geocode a place name
aoi = AOIBuilder.from_place_name("Peten, Guatemala", buffer_km=5.0)

# Option 2 -- bounding box
aoi = AOIBuilder.from_bbox(min_lon=-90.2, min_lat=17.1, max_lon=-89.95, max_lat=17.3)

# Option 3 -- explicit polygon
aoi = AOIBuilder.from_polygon([(-90.2, 17.1), (-89.95, 17.1), ...])

# Option 4 -- shapefile or GeoPackage
aoi = AOIBuilder.from_shapefile("path/to/my_aoi.shp")

# Option 5 -- Esri FileGDB layer
aoi = AOIBuilder.from_fgdb("path/to/my.gdb", layer="StudyArea")
```

### Fetching Imagery

```python
fetcher = ImageryFetcher(
    aoi=aoi,
    start_date="2022-01-01",
    end_date="2024-12-31",
    max_cloud_cover=15,
    orbit_direction="ascending",
)
imagery = fetcher.fetch_all()
```

Imagery is streamed only for the AOI window -- no full Sentinel scenes are written
to disk.

### Running the Analysis

Pass parameters to the constructor, then call `run()`:

```python
# Default parameters
analyser = SubCanopyAnalyser(aoi=aoi, imagery=imagery)
result = analyser.run()

# Custom parameters -- override any of the defaults
analyser = SubCanopyAnalyser(
    aoi=aoi,
    imagery=imagery,
    forest_ndvi_threshold=0.60,
    thresh_high=0.70,
    min_footprint_area=100,
)
result = analyser.run()
```

### Visualising Results

```python
vis = ResultVisualiser(result=result, aoi=aoi, s1_stack=imagery.s1)

vis.folium_map()            # interactive Leaflet map (Jupyter inline)
vis.indicator_panel()       # 6-panel matplotlib figure
vis.sar_timeseries_chart()  # VV backscatter through time
vis.probability_histogram() # distribution of probability scores
```

### Saving Outputs

```python
writer = OutputWriter(result=result, aoi=aoi, output_dir="./outputs", study_name="my_study")
writer.save_all()
```

Saved files:

| File | Format | Contents |
|------|--------|----------|
| `my_study_probability.tif` | GeoTIFF | Sub-canopy probability (0.0-1.0) |
| `my_study_confidence.tif` | GeoTIFF | Confidence zones (0=bg, 1=low, 2=medium, 3=high) |
| `my_study_stability.tif` | GeoTIFF | Temporal stability indicator |
| `my_study_sar_anomaly.tif` | GeoTIFF | SAR Gaussian anomaly |
| `my_study_ndvi.tif` | GeoTIFF | Sentinel-2 NDVI |
| `my_study_footprints.geojson` | GeoJSON | Vectorised detections (WGS84) |
| `my_study_footprints.csv` | CSV | Footprint attribute table |
| `my_study_summary.png` | PNG | 6-panel indicator overview |

---

## Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `forest_ndvi_threshold` | 0.55 | Minimum NDVI to include a pixel in the analysis |
| `water_ndwi_threshold` | 0.15 | Maximum NDWI (excludes waterlogged / river pixels) |
| `stability_floor` | 0.70 | Minimum stability score clamped to prevent false zeros |
| `texture_kernel_radius` | 3 px | Half-size of the local texture window |
| `anomaly_kernel_radius` | 15 px | Gaussian sigma for the spatial smoothing baseline |
| `anomaly_sigma` | 1.5 | Z-score scale factor for anomaly normalisation |
| `slope_threshold` | 15 deg | Pixels on steeper slopes are excluded |
| `pol_ratio_min` | 0.02 | VH/VV ratio mapped to 0 (lower bound) |
| `pol_ratio_max` | 0.30 | VH/VV ratio mapped to 1 (upper bound) |
| `w_stability` | 0.30 | Weight of stability in fusion score |
| `w_polratio` | 0.20 | Weight of polarimetric ratio |
| `w_texture` | 0.20 | Weight of texture |
| `w_anomaly` | 0.20 | Weight of SAR anomaly |
| `w_optical` | 0.10 | Weight of optical indicator |
| `thresh_high` | 0.65 | Probability above which a pixel is HIGH confidence |
| `thresh_medium` | 0.45 | Probability above which a pixel is MEDIUM confidence |
| `min_footprint_area` | 80 m2 | Minimum polygon area kept as a detection |

---

## Running Tests

```bash
cd tools/python/sub-canopy-detector
pytest tests/ -v
```

Tests run offline and exercise the core numpy analysis functions with synthetic data.

---

## Data Sources

| Dataset | Collection | Provider |
|---------|-----------|----------|
| Sentinel-1 GRD RTC | `sentinel-1-rtc` | ESA / Microsoft Planetary Computer |
| Sentinel-2 L2A | `sentinel-2-l2a` | ESA / Microsoft Planetary Computer |
| Copernicus DEM GLO-30 | `cop-dem-glo-30` | Copernicus / Microsoft Planetary Computer |

---

## Companion GEE Script

The Google Earth Engine implementation of this analysis is at
`tools/gee/sub-canopy-structure-detector/script.js`.  The Python port
mirrors the same 13-step pipeline using local Python packages instead of
the GEE API, making it runnable without a GEE account and inspectable
at every intermediate step.

---

## Limitations

- Requires a stable internet connection; very large AOIs (>100 km2) may hit
  Planetary Computer query limits.
- Sentinel-1 RTC scenes are not available before approximately 2017.
- Results are most reliable in lowland tropical forest with dense, continuous canopy.
- The analysis detects indicators consistent with built structures; ground-truth
  validation is always needed before operational use.
