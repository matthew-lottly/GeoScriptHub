# Geospatial Data Access & Analysis Methods Research Report

## Table of Contents
1. [Capella Space Open Data](#1-capella-space-open-data)
2. [NAIP on Microsoft Planetary Computer](#2-naip-on-microsoft-planetary-computer)
3. [Building Detection from High-Resolution SAR](#3-building-detection-from-high-resolution-sar)
4. [Tree Canopy Delineation from High-Resolution Optical](#4-tree-canopy-delineation-from-high-resolution-optical)
5. [Tree Species Classification from RGBNIR Imagery](#5-tree-species-classification-from-rgbnir-imagery)

---

## 1. Capella Space Open Data

### STAC Catalog URLs
- **Static STAC Catalog**: `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json`
- **STAC Browser**: `https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json`
- **Interactive Map (Felt)**: `https://felt.com/map/Capella-Space-Open-Data-bB24xsH3SuiUlpMdDbVRaA`
- **AWS S3 Bucket**: `s3://capella-open-data/data/` (region: us-west-2, no auth required)

### STAC Specification
- Version: **1.0.0**
- Type: **Static STAC Catalog** (NOT a STAC API)
- License: **CC BY 4.0**

### Catalog Structure
The catalog is organized into 6 sub-catalogs:
1. **by-capital** — ~200+ world capital city collections
2. **by-product-type** — 7 product type collections
3. **by-use-case** — 16 thematic use case categories
4. **by-date** — Temporal organization
5. **ieee-data-contest-2026** — Special collection for 2026 IEEE GRSS Data Analysis Contest
6. **other** — Miscellaneous collections

### Data Formats (Product Types)
| Product Type | Description | Format |
|---|---|---|
| **SLC** | Single Look Complex | HDF5 (complex-valued) |
| **GEO** | Geocoded (terrain-corrected) | Cloud-Optimized GeoTIFF |
| **GEC** | Geocoded Ellipsoid Corrected | Cloud-Optimized GeoTIFF |
| **SICD** | Sensor Independent Complex Data | NGA standard format |
| **SIDD** | Sensor Independent Derived Data | NGA standard format |
| **CPHD** | Compensated Phase History Data | NGA standard format |
| **CSI** | Complex Sub-aperture Image | HDF5 |

### Reading Capella Data
- **SLC data**: Use `capellaspace/tools` SLC reader — https://github.com/capellaspace/tools/tree/master/capella-slc-reader
- **GEO/GEC (COG)**: Standard rasterio/rioxarray reading
- **Console SDK**: https://capella-console-client.readthedocs.io/en/main/

### Sensor Characteristics
- **Band**: X-band SAR (~9.65 GHz)
- **Resolution**: Sub-meter (0.35m–0.5m Spotlight), up to ~1m for other modes
- **Imaging Modes**: Spotlight, Sliding Spotlight, Stripmap
- **Polarizations**: Primarily VV, some HH
- **~1000+ open collects** available

### US Locations
The "by-capital" catalog covers **world capitals**, so US mainland cities are not directly listed there. However:
- **US Territories**: Charlotte Amalie (USVI), Pago Pago (American Samoa), Hagåtña (Guam)
- **Use cases covering US areas**: Disaster Response, Urban Development, Public Infrastructure categories likely include US cities
- **IEEE Data Contest 2026**: May contain US collections
- Use the interactive Felt map to browse all available US coverage: https://felt.com/map/Capella-Space-Open-Data-bB24xsH3SuiUlpMdDbVRaA

### Can pystac/pystac_client Search Capella Open Data?

**pystac (YES)** — Capella is a static STAC catalog. Use `pystac.Catalog.from_file()`:
```python
import pystac

catalog = pystac.Catalog.from_file(
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json"
)

# Walk the catalog tree
for root, catalogs, items in catalog.walk():
    for item in items:
        print(item.id, item.datetime, item.geometry)
        for key, asset in item.assets.items():
            print(f"  {key}: {asset.href}")
```

**pystac_client (NO)** — `pystac_client` requires a STAC API endpoint (with `/search`). Capella Open Data is a **static catalog** (plain JSON files on S3), not a STAC API. You cannot use `pystac_client.Client.open()` or `.search()` with it.

### Availability on Cloud Platforms

| Platform | Available? | Notes |
|---|---|---|
| **Microsoft Planetary Computer** | **NO** | Not among the ~135 collections. Has Sentinel-1 (C-band SAR) but not Capella (X-band). |
| **AWS Earth Search** | **NO** | Only 9 collections (Sentinel-1/2, NAIP, Landsat, Copernicus DEM). No Capella. |
| **AWS S3 (direct)** | **YES** | `s3://capella-open-data/data/` — no sign required, us-west-2 region |

### AWS CLI Access
```bash
aws s3 ls --no-sign-request s3://capella-open-data/data/
```

---

## 2. NAIP on Microsoft Planetary Computer

### Collection Details
| Property | Value |
|---|---|
| **Collection ID** | `naip` |
| **STAC API** | `https://planetarycomputer.microsoft.com/api/stac/v1` |
| **Collection URL** | `https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip` |
| **Bands** | Red, Green, Blue, NIR (4 bands) |
| **GSD** | 0.3m, 0.6m, 1.0m (varies by year/state) |
| **Format** | Cloud-Optimized GeoTIFF (COG) |
| **Temporal Range** | 2010–2023 |
| **Spatial Coverage** | CONUS + Hawaii + Puerto Rico + USVI |
| **Asset Key** | `image` (4-band RGBNIR COG tile) |
| **License** | Public Domain |

### NAIP Also on AWS Earth Search
- **Earth Search Collection**: `naip`
- **Earth Search STAC API**: `https://earth-search.aws.element84.com/v1`
- **GSD**: 0.6m, 1.0m
- **Temporal**: 2010–2022

### Search Pattern (Planetary Computer)
```python
import pystac_client
import planetary_computer

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["naip"],
    bbox=[-122.27, 47.54, -121.96, 47.74],  # Example: Seattle area
    datetime="2020-01-01/2020-12-31",
)

items = search.item_collection()
print(f"Found {len(items)} NAIP tiles")

# Access a tile
item = items[0]
signed_href = item.assets["image"].href  # Already signed via modifier
```

### Loading NAIP Data
```python
import rioxarray

# Open a single tile
ds = rioxarray.open_rasterio(item.assets["image"].href)
# ds has shape (4, height, width) — bands: R, G, B, NIR

# Or with xarray + stackstac for mosaicking
import stackstac
stack = stackstac.stack(items, epsg=32610, resolution=0.6)
```

### Search Pattern (AWS Earth Search)
```python
import pystac_client

catalog = pystac_client.Client.open(
    "https://earth-search.aws.element84.com/v1"
)

search = catalog.search(
    collections=["naip"],
    bbox=[-77.12, 38.79, -76.91, 38.99],  # Example: Washington DC
    datetime="2018-01-01/2018-12-31",
)

items = search.item_collection()
# No signing needed — NAIP on AWS is publicly accessible
```

---

## 3. Building Detection from High-Resolution SAR

### Overview
Building detection from SAR imagery (particularly high-resolution X-band at 0.5–1m) leverages backscatter intensity, double-bounce scattering, shadow geometry, and increasingly deep learning approaches.

### Key SAR Scattering Mechanisms for Buildings
1. **Double-bounce reflection**: Wall–ground dihedral corner reflector creates strong returns
2. **Layover**: Building rooftop appears shifted toward sensor
3. **Shadow**: Building casts radar shadow on far side
4. The **layover–building–shadow** triplet is the fundamental radiometric signature

### State-of-the-Art Methods

#### A. Traditional / Statistical Methods
- **CFAR (Constant False Alarm Rate) Detectors**: Adaptive thresholding on backscatter intensity. Variants include CA-CFAR, OS-CFAR, and VI-CFAR. Effective for isolated bright targets but struggles in dense urban areas.
- **Template Matching on Double-Bounce Lines**: Detect linear features in SAR amplitude corresponding to wall–ground intersections.
- **Shadow-Based Geometric Reconstruction**: Extract building height from shadow length given incidence angle. Key reference: Brunner et al. (2010), "Building Height Retrieval from VHR SAR Imagery Based on an Iterative Simulation and Matching Technique," *IEEE TGRS*.

#### B. Deep Learning — Semantic Segmentation
- **U-Net on SAR Amplitude**: Encoder–decoder architecture trained on SAR intensity patches. Most widely adopted baseline.
  - Reference: Shahzad et al. (2019), "Buildings Detection in VHR SAR Images Using Fully Convolution Neural Networks," *IEEE TGRS*.
- **ResU-Net / Attention U-Net**: Residual connections and attention gates improve building boundary delineation in SAR speckle.
- **DeepLabv3+ on SAR**: Atrous spatial pyramid pooling captures multi-scale building patterns.
  - Reference: Henry et al. (2021), "Road Segmentation in SAR Satellite Images with Deep Fully Convolutional Neural Networks," *IEEE GRSL*.

#### C. Deep Learning — Instance/Object Detection
- **Faster R-CNN / YOLOv5 on SAR**: Object detection architectures adapted for SAR imagery to detect individual building footprints.
  - Reference: SpaceNet 6 Challenge (2020) — Multi-sensor all-weather mapping of building footprints using SAR and optical. Dataset: 120k+ building labels on Capella-equivalent resolution SAR over Rotterdam.
  - URL: https://spacenet.ai/sn6-challenge/
- **Mask R-CNN for SAR Building Instance Segmentation**: Instance-level building masks from amplitude SAR.

#### D. SAR-Specific Approaches
- **InSAR Coherence for Built-Up Area Mapping**: Interferometric coherence remains high in built-up areas across temporal baselines.
  - Reference: Esch et al. (2010), "Delineation of Urban Footprints from TerraSAR-X Data by Analyzing Speckle Characteristics and Intensity Information," *IEEE TGRS*.
- **Polarimetric Decomposition** (where dual/quad-pol available): Freeman-Durden, Yamaguchi 4-component decomposition to isolate double-bounce component.
- **Multi-temporal SAR Stacking**: Averaging multiple SAR acquisitions suppresses speckle and enhances persistent scatterers (buildings).

#### E. Key Datasets & Benchmarks
- **SpaceNet 6**: SAR + optical building footprint dataset (Rotterdam), 0.5m Capella-like resolution. https://spacenet.ai/sn6-challenge/
- **SpaceNet 8**: Flood detection + building/road mapping from SAR. https://spacenet.ai/sn8-challenge/
- **IEEE GRSS Data Fusion Contest 2023**: SAR-based urban mapping.
- **WHU Building Dataset (SAR)**: Large-scale SAR building segmentation dataset.
- **Capella IEEE Data Contest 2026**: Upcoming contest using Capella Open Data.

#### F. Practical Considerations for X-band 0.5–1m SAR
- At sub-meter resolution, individual buildings are clearly resolvable
- Speckle filtering (Lee, Frost, or refined Lee) is essential preprocessing
- Geocoded (GEO) products are preferred over SLC for building footprint extraction
- Incidence angle affects shadow length and double-bounce strength — steeper angles = less shadow
- Urban areas with uniform building orientation produce cardinal effect (aligned with azimuth direction = stronger double-bounce)

### Recommended Pipeline
```
SAR GEO/GEC (COG) → Speckle Filter → U-Net/DeepLabv3+ Segmentation → Post-processing (morphological ops, vectorization) → Building Footprints (GeoJSON)
```

---

## 4. Tree Canopy Delineation from High-Resolution Optical (0.6m NAIP)

### Overview
Individual tree crown delineation (ITCD) from high-resolution optical imagery is a well-studied problem. At 0.6m NAIP resolution, individual tree crowns (typically 3–15m diameter) are composed of many pixels, enabling segmentation.

### State-of-the-Art Methods

#### A. Classical Image Processing
- **Local Maxima Detection + Region Growing**: Identify treetops as local brightness maxima in NIR band or NDVI, then grow regions outward. Simple but effective for isolated trees.
  - Reference: Wulder et al. (2000), "Local Maximum Filtering for the Extraction of Tree Locations and Basal Area from High Spatial Resolution Imagery," *Remote Sensing of Environment*.
- **Watershed Segmentation**: Treat inverted canopy height or NDVI as topographic surface; watershed lines delineate crown boundaries.
  - Reference: Chen et al. (2006), "Isolating Individual Trees in a Savanna Woodland Using Small Footprint Lidar Data," *PE&RS*.
- **Multi-scale Edge Detection + Template Matching**: Detect circular/elliptical crowns at multiple scales. Works well for coniferous species.
  - Reference: Pollock (1996), "The Automatic Recognition of Individual Trees in Aerial Images of Forests Based on a Synthetic Tree Crown Image Model," PhD thesis.

#### B. Object-Based Image Analysis (OBIA)
- **Multi-resolution Segmentation (eCognition/GRASS)**: Segment image into objects at multiple scales, classify tree objects using spectral, shape, and textural features.
  - Reference: Ke & Quackenbush (2011), "A Review of Methods for Automatic Individual Tree-Crown Detection and Delineation from Passive Remote Sensing," *International Journal of Remote Sensing*.
- **Mean Shift Segmentation**: Non-parametric clustering in spatial-spectral space to delineate crown boundaries.

#### C. Deep Learning — Semantic Segmentation
- **U-Net / ResU-Net on NAIP 4-band**: Binary or multi-class segmentation of tree canopy vs. non-canopy.
  - Reference: Wagner et al. (2020), "Using the U-Net Convolutional Network to Map Forest Types and Disturbance in the Atlantic Rainforest with Very High Resolution Images," *Remote Sensing of Environment*.
- **DeepLabv3+ with NAIP RGBNIR**: Achieves state-of-the-art canopy segmentation accuracy.
  - This is the approach used by many US urban tree canopy mapping programs.

#### D. Deep Learning — Instance Segmentation (Individual Tree Crowns)
- **DeepForest**: Purpose-built Python package for tree crown detection from aerial imagery.
  - Pre-trained on NEON airborne RGB data; fine-tunable for NAIP.
  - Uses RetinaNet architecture internally.
  - **URL**: https://deepforest.readthedocs.io/
  - **GitHub**: https://github.com/weecology/DeepForest
  - **Paper**: Weinstein et al. (2019), "Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks," *Remote Sensing*.
  - ```python
    from deepforest import main
    model = main.deepforest()
    model.use_release()  # Load pre-trained model
    predictions = model.predict_tile(raster_path="naip_tile.tif")
    ```
- **Mask R-CNN for Tree Crowns**: Instance segmentation producing per-tree masks.
  - Reference: Braga et al. (2020), "Tree Crown Delineation Algorithm Based on a Convolutional Neural Network," *Remote Sensing*.
- **Detectron2-based approaches**: Facebook AI's Detectron2 with Mask R-CNN backbone fine-tuned on tree crown annotations.
  - Reference: Ball et al. (2023), "Accurate Delineation of Individual Tree Crowns in Tropical Forests from Aerial RGB Imagery Using Mask R-CNN," *Remote Sensing in Ecology and Conservation*.
- **YOLO-based (YOLOv8/YOLOv9)**: Real-time object detection adapted for tree crown bounding boxes. Faster inference but box-level (not mask-level) predictions.

#### E. Canopy Height Integration
- When available, LiDAR-derived Canopy Height Models (CHMs) dramatically improve tree crown delineation.
- **3DEP LiDAR data** (USGS) can be paired with NAIP for height-informed segmentation.
- Microsoft's **Global Canopy Height Map** (from satellite imagery + AI): https://github.com/microsoft/GlobalCanopyHeight

#### F. Key Datasets
- **NEON Airborne Data**: High-res RGB + hyperspectral + LiDAR at ecological field sites across US. https://www.neonscience.org/
- **TreeForCaSt benchmark**: Aerial image tree crown segmentation benchmark.
- **Urban Tree Canopy (UTC) datasets**: USFS i-Tree and NLCD tree canopy layers.
- **NAIP itself**: Available via Planetary Computer or Earth Search (see Section 2).

### Recommended Pipeline for NAIP
```
NAIP 4-band COG → NDVI computation (band4-band1)/(band4+band1) → DeepForest or U-Net segmentation → Morphological post-processing → Vectorize crowns → Individual Tree Crown polygons (GeoJSON)
```

### Spectral Indices Useful for Tree Canopy
| Index | Formula | Purpose |
|---|---|---|
| **NDVI** | (NIR - Red) / (NIR + Red) | Vegetation vigor, canopy/non-canopy separation |
| **GNDVI** | (NIR - Green) / (NIR + Green) | Green chlorophyll sensitivity |
| **VARI** | (Green - Red) / (Green + Red - Blue) | Visible-band vegetation index |
| **ExG** | 2*Green - Red - Blue | Excess green index |

---

## 5. Tree Species Classification from 4-Band (RGBNIR) Imagery

### Overview
Tree species classification from 4-band RGBNIR imagery (e.g., NAIP) is achievable but challenging due to limited spectral resolution. Success depends on combining spectral features with spatial/textural information.

### State-of-the-Art Methods

#### A. Pixel-Level Classification
- **Random Forest on Spectral Features**: Extract per-pixel features (band values, spectral indices like NDVI, GNDVI, NDVI ratios) and classify with Random Forest.
  - Typically achieves 60–80% accuracy for 5–15 species classes with NAIP alone.
  - Reference: Immitzer et al. (2012), "Tree Species Classification with Random Forest Using Very High Spatial Resolution 8-Band WorldView-2 Satellite Data," *Remote Sensing*.
- **Gradient Boosted Trees (XGBoost/LightGBM)**: Often outperforms RF on tabular spectral features.
- **Support Vector Machines (SVM)**: Effective for small training sets with RBF kernel on spectral features.

#### B. Object-Based Classification
- **OBIA + Random Forest**: Segment crowns first (using methods from Section 4), then extract per-crown mean/std/min/max spectral values + shape metrics (crown area, perimeter, compactness) + texture (GLCM contrast, entropy, homogeneity). Classify crown objects.
  - Reference: Cho et al. (2012), "Mapping Tree Species Composition in South African Savannas Using an Integrated Airborne Spectral and LiDAR System," *Remote Sensing of Environment*.
  - This is the **most practical approach for NAIP-resolution imagery**.

#### C. Deep Learning — CNN Classifiers
- **CNN on Crown Patches**: Extract image chips centered on each detected tree crown, classify with ResNet/EfficientNet.
  - Input: 4-band (RGBNIR) patches of ~32×32 to 64×64 pixels (covering one crown at 0.6m resolution)
  - Reference: Hartling et al. (2019), "Urban Tree Species Classification Using a WorldView-2/3 and LiDAR Data Fusion Approach and Deep Learning," *Sensors*.
- **Vision Transformers (ViT)**: Emerging approach using self-attention on image patches for fine-grained species discrimination.
  - Reference: Ahlswede et al. (2023), "TreeSatAI Benchmark Archive: A Multi-Sensor, Multi-Label Dataset for Tree Species Classification in Remote Sensing," *Earth System Science Data*.
  - **TreeSatAI dataset**: https://zenodo.org/record/6780578

#### D. Deep Learning — End-to-End Segmentation + Classification
- **Multi-class Semantic Segmentation**: U-Net/DeepLabv3+ trained to segment pixels directly into species classes. Requires dense annotations.
  - Reference: Schiefer et al. (2020), "Mapping Forest Tree Species in High Resolution UAV-Based RGB-Imagery by Means of Convolutional Neural Networks," *Remote Sensing of Environment*.
- **Panoptic Segmentation**: Combines instance segmentation (individual crowns) with semantic segmentation (species class per crown). Cutting-edge but annotation-intensive.

#### E. Feature Engineering for RGBNIR Species Classification

**Spectral Features** (per-crown statistics):
| Feature | Description |
|---|---|
| Band means (R, G, B, NIR) | Mean reflectance per crown |
| Band ratios (NIR/R, NIR/G, G/R) | Spectral shape descriptors |
| NDVI, GNDVI | Vegetation indices |
| Band standard deviations | Within-crown spectral variability |

**Textural Features** (GLCM on individual bands):
| Feature | Description |
|---|---|
| Contrast | Local intensity variation |
| Homogeneity | Spatial uniformity |
| Entropy | Textural complexity |
| Correlation | Linear dependency of gray levels |
| Angular Second Moment (ASM) | Textural uniformity |

**Geometric Features** (from detected crowns):
| Feature | Description |
|---|---|
| Crown area | Species differ in crown size |
| Crown compactness | Circularity index (area/perimeter²) |
| Crown asymmetry | Ratio of semi-axes of fitted ellipse |

#### F. Practical Limitations with NAIP 4-Band
- **Limited spectral resolution**: 4 broad bands cannot distinguish species that differ only in narrow spectral regions (e.g., red-edge, SWIR)
- **No temporal phenology**: Single-date NAIP captures miss leaf-on/leaf-off differences. Multi-year NAIP composites can approximate phenological differences.
- **Accuracy ceiling**: Typically 70–85% for broad species groups (e.g., deciduous vs. coniferous, oak vs. maple vs. pine), lower for fine-grained species
- **Enhancement strategies**:
  - Fuse with **LiDAR** (crown shape, height, canopy structure) — improves accuracy by 10–20%
  - Fuse with **Sentinel-2 time series** for phenological features
  - Use **multi-year NAIP** to capture seasonal variation (different states captured in different seasons/years)

#### G. Key Software & Libraries
- **scikit-learn**: Random Forest, SVM, gradient boosting classifiers
- **XGBoost / LightGBM**: High-performance gradient boosted trees
- **PyTorch / TensorFlow**: CNN and ViT models
- **DeepForest**: Tree crown detection (pairs with classifier)
- **rasterio / rioxarray**: Raster I/O
- **GDAL**: Raster processing
- **eCognition**: Commercial OBIA software (widely used in forestry)

#### H. Key Datasets for Training
- **TreeSatAI**: Multi-sensor benchmark for tree species classification. https://zenodo.org/record/6780578
- **NEON Vegetation Structure Data**: Field-mapped tree species with co-located airborne imagery. https://data.neonscience.org/
- **FIA (Forest Inventory and Analysis)**: USFS plot-level species composition data. https://www.fia.fs.usda.gov/
- **iNaturalist / GBIF**: Citizen science tree occurrence records for label generation.
- **Auto Arborist**: Google's urban tree dataset with species labels and Street View imagery. https://google.github.io/auto-arborist/

### Recommended Pipeline
```
NAIP 4-band → Crown Detection (DeepForest) → Per-Crown Feature Extraction (spectral + textural + geometric) → Random Forest / XGBoost Classification → Species Map (GeoJSON with species attribute)
```

---

## Summary of Key URLs

| Resource | URL |
|---|---|
| Capella STAC Catalog | `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json` |
| Capella STAC Browser | `https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json` |
| Capella S3 Bucket | `s3://capella-open-data/data/` |
| Capella Interactive Map | `https://felt.com/map/Capella-Space-Open-Data-bB24xsH3SuiUlpMdDbVRaA` |
| Capella SLC Reader | `https://github.com/capellaspace/tools/tree/master/capella-slc-reader` |
| Capella Python SDK | `https://capella-console-client.readthedocs.io/en/main/` |
| Planetary Computer STAC API | `https://planetarycomputer.microsoft.com/api/stac/v1` |
| NAIP Collection (PC) | `https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip` |
| AWS Earth Search STAC API | `https://earth-search.aws.element84.com/v1` |
| SpaceNet 6 (SAR Buildings) | `https://spacenet.ai/sn6-challenge/` |
| DeepForest (Tree Detection) | `https://deepforest.readthedocs.io/` |
| TreeSatAI Dataset | `https://zenodo.org/record/6780578` |
| NEON Airborne Data | `https://www.neonscience.org/` |
| MS Global Canopy Height | `https://github.com/microsoft/GlobalCanopyHeight` |
| Auto Arborist (Tree Species) | `https://google.github.io/auto-arborist/` |
