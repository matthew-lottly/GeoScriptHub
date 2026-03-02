# deep-fusion-landcover

> **Annual 12-class landcover mapper for Austin, TX metro (1990–2025)**  
> Multi-sensor fusion · 5-branch stacked ensemble · Sub-canopy structure detection

---

## Overview

`deep-fusion-landcover` is a Python tool that fuses six free data sources into annual
12-class landcover maps covering the Austin, Texas metro area from 1990 to 2025.
It combines classical machine learning, deep learning, pseudo-quantum classification,
and object-based image analysis in a single ensemble pipeline.

### What it produces

| Output | Format | Description |
|---|---|---|
| `landcover_YYYY.tif` × 35 | COG GeoTIFF (EPSG:32614, int8) | Annual 10 m class maps |
| `interactive_landcover_map.html` | Folium HTML | 1990 ↔ 2025 dual-panel slider map |
| `landcover_transitions_sankey.html` | Plotly HTML | Multi-epoch class flow diagram |
| `urban_expansion.html` | Plotly HTML | Logistic growth curve |
| `transitions/YYYY_YYYY_transition.csv` × 34 | CSV | 12×12 year-pair transition matrices |
| `annual_class_fractions.csv` | CSV | Per-class area fractions by year |
| `annual_intensity.csv` | CSV | Aldwaik-Pontius change intensities |
| `accuracy/accuracy_summary.csv` | CSV | OA, κ, F1 vs. NLCD validation |
| `accuracy/confusion_YYYY.csv` | CSV | Confusion matrix per validation year |
| `sub_canopy/building_footprints.geojson` | GeoJSON | Buildings detected under canopy |
| `models/` | pkl / npz | Trained branch models + meta-learner |
| `pipeline_architecture.html` | HTML | Interactive pipeline flowchart |

### 12-class Austin scheme

| ID | Class | Colour |
|---|---|---|
| 1 | Open Water | `#4472C4` |
| 2 | Emergent Wetland | `#70AD47` |
| 3 | Woody Wetland | `#375623` |
| 4 | Upland Juniper/Oak Forest | `#1E6B00` |
| 5 | Cedar Brush/Shrub | `#6B8E23` |
| 6 | Grassland/Pasture | `#D4E157` |
| 7 | Agricultural Land | `#FFA500` |
| 8 | Low-Density Development | `#FFC000` |
| 9 | High-Density Development | `#FF0000` |
| 10 | Impervious Surface | `#808080` |
| 11 | Barren/Quarry | `#C0C0C0` |
| 12 | Other/Transitional | `#BFBFBF` |

---

## Architecture

```
Data Sources → Preprocessing → Feature Engineering → 5-Branch Ensemble → Outputs
     ↓                ↓                ↓
Landsat 4-9       Cloud mask      Spectral indices    Branch 1: RandomForest
Sentinel-1 SAR    Harmonise       SAR backscatter     Branch 2: GradientBoosting
Sentinel-2 L2A    Medoid          Phenology           Branch 3: 8-qubit VQC ★
NAIP 1 m          STARFM          LiDAR products      Branch 4: TorchGeo CNN ★
3DEP LiDAR        SLC-off fill    GLCM texture        Branch 5: SAM2 OBIA ★
                                                      ─────────────────────────
                                                      LightGBM Meta-Stacker
                                                      ↓
                                                Sub-canopy detector
                                                      ↓
                                               Annual COG maps + HTML
```

★ See [Design notes](#design-notes) below.

---

## Installation

```bash
# Clone GeoScriptHub
git clone https://github.com/your-org/GeoScriptHub.git
cd GeoScriptHub/tools/python/deep-fusion-landcover

# Create environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install (core)
pip install -e ".[dev]"

# Optional: TorchGeo CNN branch
pip install -e ".[dev,sam]"
```

---

## Usage

### Quick run (defaults to Austin 55 km buffer, 1990–2025)

```bash
deep-fusion-landcover run
```

### Custom AOI and date range

```bash
deep-fusion-landcover run \
  --output outputs/austin_landcover \
  --center-lat 30.2672 \
  --center-lon -97.7431 \
  --buffer-km 30 \
  --start-year 2000 \
  --end-year 2025
```

### Disable optional branches (faster)

```bash
deep-fusion-landcover run --no-quantum --no-cnn
```

### Cross-validate against your own NLCD files

```bash
deep-fusion-landcover validate \
  --output outputs/austin_landcover \
  --nlcd-dir /path/to/nlcd/
```

### Regenerate visualizations

```bash
deep-fusion-landcover visualize --output outputs/austin_landcover
```

### Show pipeline flowchart

```bash
deep-fusion-landcover flowchart --output outputs/austin_landcover
```

### Python API

```python
from deep_fusion_landcover.aoi import AustinMetroAOI
from deep_fusion_landcover.ensemble import EnsembleClassifier

aoi = AustinMetroAOI()
ens = EnsembleClassifier()
# … fit and predict …
```

---

## Data Sources

| Sensor | Bands | Years | Provider |
|---|---|---|---|
| Landsat 4/5 TM | 6 reflectance | 1990–1999 | USGS via Planetary Computer |
| Landsat 7 ETM+ | 6 reflectance | 1999–2012 | USGS via Planetary Computer |
| Landsat 8/9 OLI | 6 harmonised | 2013–2025 | USGS via Planetary Computer |
| Sentinel-2 L2A | 10 bands | 2017–2025 | ESA via Planetary Computer |
| Sentinel-1 RTC | VV, VH | 2014–2025 | ESA via Planetary Computer |
| NAIP | RGB + NIR | 2003–2023 | USDA via Planetary Computer |
| Copernicus DEM | Elevation | Static | ESA via Planetary Computer |
| USGS 3DEP | DSM + DTM | ~2017+ | USGS 3DEP WCS |

---

## Feature Engineering (150 dimensions)

| Group | Features |
|---|---|
| Spectral indices | NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, BSI, NBR, EVI2, ARVI |
| Band ratios | Per-band pairwise ratios |
| SAR backscatter | VV, VH, VV/VH, VV+VH, VV−VH |
| Phenology | 7 NDVI seasonal statistics (median, amplitude, greenup DOY …) |
| LiDAR | nDSM, DTM slope, penetration ratio, height percentiles, canopy density |
| Terrain | Slope, aspect, curvature, hillshade, HAND proxy |
| GLCM texture | Contrast, correlation, energy, homogeneity (4 directions × 5 bands) |

---

## Design Notes

### Quantum VQC
An 8-qubit pseudo-quantum variational circuit is simulated in pure NumPy.
Angle encoding maps the first 8 PCA-reduced feature dimensions to qubit rotation
angles. Three variational layers with Ry gates and CX entanglement are trained via
parameter-shift gradients. A downstream linear classifier maps qubit Z-expectation
values to 12 class probabilities.  
*No quantum computer is required; this is a deliberate classical simulation for
methodological exploration.*

### TorchGeo CNN
A ResNet-50 backbone pretrained on BigEarthNet (via `torchgeo`) is fine-tuned on
NLCD-labeled pixels. The model is exported to ONNX opset 17 for CPU inference via
`onnxruntime`. A PCA fallback is used when PyTorch is unavailable.

### SAM2 OBIA
The Segment Anything Model 2 (SAM2) runs in automatic mask generation mode on
256×256 NAIP chips. Per-segment mean feature values are used as object-level
classification inputs. A SLIC superpixel fallback is used when SAM2 is not
installed.

### Cross-sensor harmonisation
Landsat OLI/OLI-2 bands are harmonised to TM/ETM+ using the Roy et al. (2016)
piecewise linear coefficients to ensure temporal consistency across the
1990–2025 time series.

---

## Development

```bash
# Install dev extras
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type check
pyright src/

# Lint
ruff check src/
```

---

## Repository layout

```
deep-fusion-landcover/
├── pyproject.toml
├── pyrightconfig.json
├── README.md
├── pipeline_architecture.html        ← generated by `flowchart` command
├── src/deep_fusion_landcover/
│   ├── __init__.py
│   ├── constants.py
│   ├── aoi.py
│   ├── acquisition.py
│   ├── preprocessing.py
│   ├── temporal_compositing.py
│   ├── feature_engineering.py
│   ├── lidar_processor.py
│   ├── sub_canopy_detector.py
│   ├── sam_segmenter.py
│   ├── cnn_encoder.py
│   ├── quantum_classifier.py
│   ├── ensemble.py
│   ├── change_detection.py
│   ├── mosaic.py
│   ├── accuracy.py
│   ├── viz.py
│   ├── flowchart.py
│   ├── cli.py
│   └── __main__.py
└── tests/
    ├── conftest.py
    ├── test_aoi.py
    ├── test_quantum_classifier.py
    ├── test_ensemble.py
    └── test_change_detection.py
```

---

## References

- Roy, D.P. et al. (2016). Characterization of Landsat-7 to Landsat-8 reflective wavelength and BRDF-normalized reflectance and implications for Landsat long time series. *Remote Sensing of Environment*, 185, 57–70.
- Aldwaik, S.Z. & Pontius, R.G. (2012). Intensity analysis to unify measurements of size and stationarity of land changes by interval, category, and transition. *Landscape and Urban Planning*, 106(1), 103–114.
- Gao, F. et al. (2006). On the blending of the Landsat and MODIS surface reflectance: Predicting daily Landsat surface reflectance. *IEEE TGRS*, 44(8), 2207–2218.
- Stewart, A.J. et al. (2022). TorchGeo: Deep Learning With Geospatial Data. *ACM SIGKDD*, 2353–2362.
- Kirillov, A. et al. (2023). Segment Anything. *arXiv:2304.02643*.
