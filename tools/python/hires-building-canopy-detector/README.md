# Hi-Res Building Footprint + Tree Canopy Detector

**State-of-the-art building detection and tree canopy analysis using high-resolution Capella Space X-band SAR (0.5 m) fused with NAIP optical imagery (0.6 m).**

---

## Capabilities

| Feature | Method | Data source |
|---|---|---|
| **Building footprints** | Morphological Building Index (MBI) + local SAR contrast + edge density, fused with optical NDVI masking, regularised with seven-component MRR scoring | Capella SAR / Sentinel-1 RTC |
| **Tree canopy mapping** | NDVI segmentation with building exclusion | NAIP 4-band / Sentinel-2 |
| **Individual tree crowns** | Marker-controlled watershed on Gaussian-smoothed NDVI local maxima | NAIP NIR band |
| **Tree species grouping** | Unsupervised K-means spectral clustering on per-crown NAIP + SAR features | NAIP R/G/B/NIR + SAR |

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  AOIBuilder  (aoi.py)                                     │
│  ─ from_point(lon, lat, buffer_km)                        │
│  ─ WGS84 ↔ UTM projection                                │
└───────────────────────┬───────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────┐
│  HiResImageryFetcher  (fetcher.py)                        │
│  ─ Capella Open Data STAC (static catalog on S3)          │
│  ─ NAIP from Microsoft Planetary Computer                 │
│  ─ Sentinel-1 RTC fallback  •  Sentinel-2 fallback       │
│  ─ Copernicus GLO-30 DEM                                  │
│  ─ Resolution harmonisation → 1 m common grid             │
└───────────────────────┬───────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────┐
│  HiResAnalyser  (analysis.py)                             │
│  Step 1  SAR preprocessing (Lee sigma speckle filter)     │
│  Step 2  SAR features (MBI, contrast, edges, shadows)     │
│  Step 3  Optical features (NDVI, brightness)              │
│  Step 4  Building detection (fusion → vectorise → MRR)    │
│  Step 5  Canopy masking                                   │
│  Step 6  Crown delineation (watershed)                    │
│  Step 7  Species classification (K-means on 7 features)   │
└───────────────────────┬───────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────┐
│  HiResOutputWriter  (export.py)                           │
│  ─ GeoTIFF rasters (MBI, score, canopy, species …)        │
│  ─ GeoJSON vectors (buildings, crowns, species crowns)    │
│  ─ 12-panel summary PNG                                   │
│  ─ Building overlay on optical + SAR                      │
│  ─ Canopy + crown overlay                                 │
│  ─ Species classification map                             │
│  ─ 6-panel statistics dashboard                           │
└───────────────────────────────────────────────────────────┘
```

## Data Sources

### Capella Space Open Data (primary SAR)
- **Band**: X-band (9.65 GHz)
- **Resolution**: 0.3–1.0 m (Spotlight / Stripmap)
- **Polarisation**: single-pol (HH or VV)
- **Access**: static STAC catalog on S3 (`capella-open-data`)
- **Format**: GeoTIFF (GEC – Geocoded Ellipsoid Corrected)

### NAIP (primary optical)
- **Bands**: R, G, B, NIR (4-band)
- **Resolution**: 0.6 m
- **Coverage**: Continental US, 2010–2023
- **Access**: Microsoft Planetary Computer STAC

### Fallbacks
| Source | Type | Resolution | When used |
|---|---|---|---|
| Sentinel-1 RTC | C-band SAR | 10 m | No Capella data for AOI |
| Sentinel-2 L2A | Optical | 10 m | No NAIP data for AOI |
| Copernicus GLO-30 | DEM | 30 m | Always (slope context) |

## Algorithms

### Morphological Building Index (MBI)
Linear structuring elements at 0°, 45°, 90°, 135° with scales [3, 7, 15, 25] pixels.
White top-hat (image − opening) captures building wall/rooftop edge residuals.
The MBI response averages all scale × angle combinations and is percentile-normalised.

### Building Fusion Score
```
S = 0.30·MBI + 0.25·contrast + 0.15·edges + 0.20·(1−NDVI) + 0.10·shadow_proximity
```

### Building Regularisation (7-component)
1. Rectangularity (area / MRR area)
2. Compactness (Polsby–Popper)
3. Solidity (area / convex hull)
4. Edge sharpness (gradient on boundary)
5. Size appropriateness (log-normal peaked at typical houses)
6. Detection probability
7. Vegetation penalty (NDVI suppression)

### Crown Delineation
1. Gaussian-smooth NDVI (σ = 3.0)
2. Local maxima → tree-top seed markers
3. Marker-controlled watershed on inverted smoothed NDVI

### Species Classification
Per-crown features: NIR/Red ratio, green chromaticity, NDVI, NIR σ, NIR mean, SAR mean, log(crown area).
K-means (scipy `kmeans2`) with auto-labelling (deciduous broadleaf / conifer / mixed / dense canopy).

## Quick Start

```bash
# Install in development mode
pip install -e .

# Run the demo (Austin, TX)
python demo_test.py

# Or use the CLI
python -m hires_detector --lon -97.769 --lat 30.263 --buffer-km 0.75
```

## Outputs

| File | Description |
|---|---|
| `summary_panels.png` | 12-panel overview of every analysis layer |
| `building_overlay.png` | Side-by-side buildings on optical + SAR |
| `canopy_overlay.png` | Canopy mask + individual crown outlines |
| `species_map.png` | Species-coloured raster + vector overlays |
| `stats_dashboard.png` | 6-panel statistics (histograms, pie chart, land cover) |
| `building_footprints.geojson` | Regularised building polygons |
| `tree_crowns.geojson` | Individual tree crown polygons |
| `species_crowns.geojson` | Crowns with species ID + label |
| `*.tif` | GeoTIFF rasters (MBI, building score, NDVI, canopy, species …) |

## Dependencies

- numpy, scipy, scikit-image
- geopandas, shapely, rasterio
- matplotlib
- pystac, pystac-client, planetary-computer
- xarray, stackstac
- pyproj

## License

MIT — see repository root `LICENSE`.
