# Quantum Flood Frequency Mapper

**Pseudo-Quantum Hybrid AI Flood Inundation Frequency Analysis**

A state-of-the-art tool that fuses **Landsat 8/9**, **Sentinel-2**, and **NAIP** satellite/aerial imagery with a novel **Quantum-Inspired Ensemble Classification (QIEC)** framework to produce per-pixel flood inundation frequency maps for Houston, TX.

## Overview

This tool addresses a gap in current flood mapping: most existing methods rely on single-sensor classification with classical thresholding. By combining **pseudo-quantum computation**, **machine-learning ensembles**, and **multi-temporal multi-sensor fusion**, this approach achieves more robust and nuanced flood detection — especially in challenging mixed-pixel environments like braided river channels, vegetated floodplains, and shadowed areas.

### Study Area

The default study area targets **Houston, Texas** (29.76°N, 95.37°W), covering Buffalo Bayou, Brays Bayou, and surrounding flood-prone watersheds within a 10 × 10 km bounding box.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANTUM FLOOD FREQUENCY MAPPER                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐                       │
│  │ Landsat   │  │ Sentinel-2 │  │   NAIP   │  ← Planetary Computer │
│  │ 8/9 (30m) │  │ L2A (20m)  │  │  (~1m)   │                      │
│  └─────┬─────┘  └─────┬──────┘  └─────┬────┘                      │
│        │              │               │                             │
│        ▼              ▼               ▼                             │
│  ┌─────────────────────────────────────────┐                       │
│  │        PREPROCESSING & ALIGNMENT         │                       │
│  │  • Downsample to common 30 m GSD         │                       │
│  │  • Anti-aliased cubic resampling         │                       │
│  │  • Cloud masking (QA_PIXEL / SCL)        │                       │
│  │  • Spectral normalisation to [0, 1]      │                       │
│  │  • Pixel-perfect grid co-registration    │                       │
│  └─────────────────┬───────────────────────┘                       │
│                    │                                                │
│                    ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │     QUANTUM-INSPIRED ENSEMBLE CLASSIFICATION (QIEC)      │      │
│  │                                                          │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │      │
│  │  │  Quantum      │  │  Quantum     │  │  Gradient-   │  │      │
│  │  │  Feature      │  │  Kernel      │  │  Boosted     │  │      │
│  │  │  Encoder      │  │  SVM         │  │  Spectral    │  │      │
│  │  │  (QFE)        │  │  (QK-SVM)    │  │  Ensemble    │  │      │
│  │  │              │  │              │  │  (GBSIE)     │  │      │
│  │  │  2-qubit     │  │  Quantum     │  │              │  │      │
│  │  │  Hilbert     │  │  kernel      │  │  200-tree    │  │      │
│  │  │  space       │  │  fidelity    │  │  gradient    │  │      │
│  │  │  simulation  │  │  matrix      │  │  boosting    │  │      │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │      │
│  │         │                 │                  │          │      │
│  │         ▼                 ▼                  ▼          │      │
│  │  ┌────────────────────────────────────────────────┐    │      │
│  │  │       BAYESIAN MODEL AVERAGING                  │    │      │
│  │  │  Adaptive confidence-weighted fusion            │    │      │
│  │  │  Sensor-specific reliability priors             │    │      │
│  │  └────────────────────┬───────────────────────────┘    │      │
│  └───────────────────────┼────────────────────────────────┘      │
│                          ▼                                        │
│  ┌─────────────────────────────────────────┐                     │
│  │    TEMPORAL FREQUENCY AGGREGATION        │                     │
│  │  • Per-pixel water count / obs count     │                     │
│  │  • Wilson score confidence intervals     │                     │
│  │  • Zone classification:                  │                     │
│  │    ≥90% permanent | 25-90% seasonal      │                     │
│  │    5-25% rare     | <5% dry              │                     │
│  └─────────────────┬───────────────────────┘                     │
│                    ▼                                              │
│  ┌─────────────────────────────────────────┐                     │
│  │    CARTOGRAPHIC OUTPUT                   │                     │
│  │  • Flood frequency GeoTIFF              │                     │
│  │  • FEMA NFHL comparison overlay         │                     │
│  │  • Interactive Leaflet map              │                     │
│  │  • Statistical summary charts           │                     │
│  └─────────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

## Scientific Methodology

### Spectral Water Indices

Three complementary water indices are computed for each observation:

| Index | Formula | Purpose |
|-------|---------|---------|
| **NDWI** | (Green − NIR) / (Green + NIR) | Open water detection (McFeeters 1996) |
| **MNDWI** | (Green − SWIR1) / (Green + SWIR1) | Built-up/water discrimination (Xu 2006) |
| **AWEI_sh** | 4(Green − SWIR1) − (0.25·NIR + 2.75·SWIR2) | Shadow-resistant water extraction (Feyisa 2014) |

### Pseudo-Quantum Feature Encoding (QFE)

Spectral indices are mapped onto a **simulated 2-qubit quantum state** in a 4-dimensional Hilbert space:

$$|\psi\rangle = U_{\text{mix}} \cdot (R_y(\theta_{\text{NDWI}}) \otimes R_y(\theta_{\text{MNDWI}})) \cdot e^{i\theta_{\text{AWEI}}} |00\rangle$$

The four basis states encode land-cover classes:
- $|00\rangle$ = water
- $|01\rangle$ = vegetation  
- $|10\rangle$ = bare soil
- $|11\rangle$ = shadow

The **Born rule** measurement extracts class probabilities:

$$P(\text{water}) = |\langle 00|\psi\rangle|^2$$

Key innovation: The **entangling mixing gate** creates quantum interference between indices — constructive interference amplifies consistent water evidence while destructive interference suppresses ambiguous mixed pixels.

### Quantum Kernel SVM (QK-SVM)

A support-vector classifier using a **quantum fidelity kernel**:

$$K(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$$

where $|\phi(x)\rangle = U_{\text{enc}}(x)|0\rangle$ is a parameterised quantum circuit embedding. This provides exponentially richer feature spaces than classical RBF kernels.

### Bayesian Model Averaging

The three classifiers are fused with adaptive, confidence-weighted Bayesian averaging:

$$p_{\text{final}} = w_q \cdot p_{\text{QFE}} + w_s \cdot p_{\text{SVM}} + w_g \cdot p_{\text{GB}}$$

Weights adapt per-pixel based on quantum measurement confidence and inter-model agreement.

### Statistical Confidence

Per-pixel **Wilson score intervals** quantify uncertainty in the frequency estimate:

$$\hat{p} \pm \frac{z}{1 + z^2/n} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$

## Multi-Sensor Harmonisation

| Sensor | Native GSD | Cloud Filter | Bands Used | Scale Factor |
|--------|-----------|-------------|-----------|-------------|
| Landsat 8/9 C2L2 | 30 m | QA_PIXEL bit flags | Blue, Green, Red, NIR, SWIR1, SWIR2 | DN × 0.0000275 − 0.2 |
| Sentinel-2 L2A | 20 m → **30 m** | SCL layer | B02–B04, B08, B11, B12 | ÷ 10000 |
| NAIP | ~1 m → **30 m** | N/A (clear-sky) | R, G, B, NIR | ÷ 255 |

All sensors are:
- **Downsampled** to 30 m using anti-aliased cubic interpolation
- **Co-registered** to the same UTM 15N (EPSG:32615) pixel grid
- **Clipped** to identical spatial extents
- **Normalised** to [0, 1] reflectance

## Output Products

| Output | Format | Description |
|--------|--------|-------------|
| `flood_frequency.tif` | GeoTIFF (float32) | Per-pixel inundation frequency [0–1] |
| `flood_zones.tif` | GeoTIFF (uint8) | Categorical zones: 0=dry, 1=rare, 2=seasonal, 3=permanent |
| `confidence_bounds.tif` | GeoTIFF (2-band) | Wilson 95% CI lower/upper bounds |
| `flood_frequency_map.png` | PNG | Publication-quality frequency map |
| `fema_comparison_map.png` | PNG | Frequency + FEMA NFHL overlay (40% transparent) |
| `interactive_flood_map.html` | HTML | Leaflet map with satellite base, frequency overlay, FEMA toggle |
| `observation_stats.png` | PNG | Sensor breakdown + frequency histogram |
| `fema_flood_zones.geojson` | GeoJSON | FEMA NFHL polygons for the study area |

## Installation

```bash
cd tools/python/quantum-flood-frequency
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Full pipeline with defaults (Houston, TX, 2015–2025, ≤15% cloud)
quantum-flood-frequency run --output ./outputs/flood --verbose

# Custom area and parameters
quantum-flood-frequency run \
    --center-lat 29.76 --center-lon -95.37 \
    --buffer-km 5 \
    --start-date 2018-01-01 --end-date 2025-12-31 \
    --max-cloud 10 \
    --output ./outputs/custom_flood

# Fast mode (skip quantum kernel SVM)
quantum-flood-frequency run --no-quantum-svm --output ./outputs/fast_flood
```

### Python API

```python
from quantum_flood_frequency import (
    AOIBuilder,
    MultiSensorAcquisition,
    ImagePreprocessor,
    QuantumHybridClassifier,
    FloodFrequencyEngine,
    FEMAFloodZones,
    FloodMapper,
)

# 1. Define AOI
aoi = AOIBuilder(center_lat=29.76, center_lon=-95.37, buffer_km=5).build()

# 2. Acquire imagery
stack = MultiSensorAcquisition(aoi, start_date="2015-01-01", end_date="2025-12-31").fetch_all()

# 3. Preprocess & align
aligned = ImagePreprocessor(aoi).align(stack.landsat, stack.sentinel2, stack.naip)

# 4. Classify water (quantum-hybrid)
classifications = QuantumHybridClassifier().classify_stack(aligned)

# 5. Compute flood frequency
engine = FloodFrequencyEngine(aligned)
result = engine.compute(classifications)
engine.save_frequency_raster(result, "outputs/flood_frequency.tif")

# 6. FEMA comparison
fema = FEMAFloodZones(aoi)
fema.fetch()

# 7. Maps
mapper = FloodMapper(result, aoi, fema, output_dir="outputs/")
mapper.generate_all()
```

## Dependencies

- **pystac-client / planetary-computer / stackstac** — STAC API imagery access
- **rioxarray / xarray / rasterio** — Raster I/O and processing
- **numpy / scipy** — Numerical computation + quantum simulation
- **scikit-learn** — SVM and gradient boosting classifiers
- **geopandas / shapely / pyproj** — Vector operations and CRS handling
- **matplotlib / folium / Pillow** — Visualisation
- **requests** — FEMA NFHL API access

## References

1. Havlíček, V. et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209–212.
2. McFeeters, S. K. (1996). "The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features." *Int. J. Remote Sensing*, 17(7), 1425–1432.
3. Xu, H. (2006). "Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery." *Int. J. Remote Sensing*, 27(14), 3025–3033.
4. Feyisa, G. L. et al. (2014). "Automated Water Extraction Index: A new technique for surface water mapping using Landsat imagery." *Remote Sensing of Environment*, 140, 23–35.
5. Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *J. American Statistical Association*, 22(158), 209–212.
6. Schuld, M. & Killoran, N. (2019). "Quantum Machine Learning in Feature Hilbert Spaces." *Physical Review Letters*, 122(4), 040504.

## License

MIT — see repository [LICENSE](../../../LICENSE).
