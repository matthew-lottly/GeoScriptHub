# NDWI Flood-Frequency Mapper — Mississippi Delta

> Google Earth Engine script that pulls a multi-year Sentinel-2 time
> series, computes per-image NDWI, classifies water pixels, and
> produces a flood-frequency composite showing how often each pixel
> was inundated (0–100 %).

| | |
|---|---|
| **Platform** | [Google Earth Engine Code Editor](https://code.earthengine.google.com/) |
| **Sensor** | Sentinel-2 Level-2A (Surface Reflectance) |
| **Index** | NDWI — McFeeters (1996): (Green − NIR) / (Green + NIR) |
| **Study Area** | Lower Mississippi Delta, ~10 mi inland from the Gulf coast |
| **Licence** | MIT |
| **Tool #** | 12 · GeoScriptHub |

---

## Features

| Capability | Details |
|---|---|
| **Adjustable cloud cover** | `MAX_CLOUD_COVER` parameter (0–100 %) filters the S2 collection |
| **SCL cloud mask** | Removes cloud shadow, medium/high cloud, and cirrus via the Scene Classification Layer |
| **NDWI threshold** | `NDWI_THRESHOLD` is user-tuneable — raise for fewer false positives |
| **Flood-frequency raster** | Per-pixel ratio of water observations to total clear observations |
| **Time-series chart** | Mean NDWI over time plotted in the console — shows seasonal cycles and anomalies |
| **Median NDWI composite** | Optional map layer for visual reference |
| **GeoTIFF export** | Commented-out `Export.image.toDrive` block ready to uncomment |
| **Zero dependencies** | Pure GEE JavaScript — paste into the Code Editor and click Run |

---

## Quick Start

1. Open the [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. Create a new script and paste the contents of [`script.js`](script.js).
3. **(Optional)** Adjust the parameters at the top of the file:

   | Parameter | Default | Description |
   |---|---|---|
   | `MAX_CLOUD_COVER` | `20` | Max cloud cover % per image (increase ⇒ more images, noisier) |
   | `START_DATE` | `2019-01-01` | Time series start |
   | `END_DATE` | `2024-12-31` | Time series end |
   | `NDWI_THRESHOLD` | `0.1` | Pixel classified as water when NDWI ≥ this value |
   | `AOI` | Lower MS Delta polygon | Modify coordinates to move the study area |

4. Click **Run**.
5. The map will display:
   - **Flood Frequency (%)** — yellow (rarely flooded) → blue (permanent water)
   - **Median NDWI** — toggled off by default
   - **Study Area** — red outline
6. Check the **Console** panel for the NDWI time-series chart and summary stats.

---

## How It Works

```
Sentinel-2 L2A Collection
  │
  ├─ filterBounds(AOI)
  ├─ filterDate(START_DATE, END_DATE)
  └─ filter(CLOUDY_PIXEL_PERCENTAGE < MAX_CLOUD_COVER)
  │
  ▼
Cloud Masking (SCL band)
  │  Remove classes 3 (shadow), 8/9 (cloud), 10 (cirrus)
  │
  ▼
Per-Image NDWI
  │  NDWI = (B3 − B8) / (B3 + B8)
  │
  ▼
Water Classification
  │  water = 1 if NDWI ≥ NDWI_THRESHOLD, else 0
  │
  ▼
Flood-Frequency Composite
  │  frequency = sum(water) / count(valid observations)
  │  Multiply by 100 → percentage
  │
  ▼
Visualisation
  ├─ Map: 6-colour YlGnBu flood-frequency raster
  ├─ Map: Median NDWI composite (optional)
  └─ Console: Mean NDWI time-series chart
```

### NDWI Formula

$$\text{NDWI} = \frac{B3_{\text{Green}} - B8_{\text{NIR}}}{B3_{\text{Green}} + B8_{\text{NIR}}}$$

- **NDWI > 0** → typically water
- **NDWI < 0** → typically vegetation / bare soil

The `NDWI_THRESHOLD` defaults to **0.1** (slightly conservative) to reduce
shadow false positives common in delta wetlands.

### Flood Frequency

$$\text{Flood Frequency (\%)} = \frac{\sum_{i=1}^{n} \text{water}_i}{\sum_{i=1}^{n} \text{valid}_i} \times 100$$

Where $n$ is the number of images in the filtered collection,
$\text{water}_i$ is 1 if the pixel was classified as water in image $i$,
and $\text{valid}_i$ is 1 if the pixel was cloud-free.

---

## Study Area

The default AOI covers the lower Mississippi River Delta from the
birdfoot passes northward, roughly 10 miles inland from the Gulf coast:

| Corner | Latitude | Longitude |
|---|---|---|
| NW | 29.45° N | 89.95° W |
| NE | 29.45° N | 88.85° W |
| SE | 28.85° N | 88.85° W |
| SW | 28.85° N | 89.95° W |

This captures the three main distributary passes (Southwest Pass, South
Pass, Pass a Loutre), the surrounding marshlands, and the Head of
Passes junction.

---

## Colour Palette

| Frequency | Colour | Meaning |
|---|---|---|
| 0 % | `#ffffcc` light yellow | Never classified as water |
| ~17 % | `#c7e9b4` light green | Rare inundation |
| ~33 % | `#7fcdbb` aqua | Occasional flooding |
| ~50 % | `#41b6c4` teal | Seasonal flooding |
| ~67 % | `#2c7fb8` blue | Frequently inundated |
| 100 % | `#253494` dark blue | Permanent water body |

---

## Exporting

Uncomment the `Export.image.toDrive` block at the bottom of the script
to save the flood-frequency raster as a 10 m GeoTIFF to your Google
Drive:

```js
Export.image.toDrive({
  image: frequencyPct.clip(AOI).toFloat(),
  description: 'NDWI_Flood_Frequency_Mississippi_Delta',
  folder: 'GEE_Exports',
  region: AOI,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});
```

---

## Requirements

- A [Google Earth Engine account](https://signup.earthengine.google.com/)
  (free for research, education, and non-commercial use)
- A modern browser (Chrome recommended)

---

## References

- McFeeters, S. K. (1996). *The use of the Normalized Difference Water
  Index (NDWI) in the delineation of open water features.* International
  Journal of Remote Sensing, 17(7), 1425–1432.
- ESA Sentinel-2 L2A product documentation:
  https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi

---

## Related Tools

| # | Tool | Description |
|---|---|---|
| 1 | [Batch Coordinate Transformer](../../../README.md) | Bulk CRS reprojection |
| 2 | [Shapefile Health Checker](../../../README.md) | Shapefile QA reports |
| 3 | [Batch Geocoder](../../../README.md) | Forward / reverse geocoding |
| 4 | [Raster Band Stats](../../../README.md) | Raster statistics & histograms |
| 5 | [Spectral Index Calculator](../../../README.md) | NDVI / NDWI / custom indices |
| 6 | [OSM Change Monitor](../../../README.md) | OpenStreetMap diff tracking |
| 7 | [FGDB Archive Publisher](../../../README.md) | ArcGIS FGDB archive pipeline |
| 8 | [Proximity Ring Analyzer](../../../README.md) | ExB buffer-ring spatial query widget |
| 9 | [Leaflet Widget Generator](../../../README.md) | Leaflet widget scaffolding |
| 10 | [Map Swiper](../../../README.md) | Before / after map comparison |
| 11 | [GeoJSON Diff Viewer](../../../README.md) | Side-by-side GeoJSON comparison |
| 12 | **NDWI Flood-Frequency Mapper** | ← you are here |
