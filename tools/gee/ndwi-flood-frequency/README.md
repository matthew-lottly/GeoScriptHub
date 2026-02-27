# NDWI Flood-Frequency Mapper — Mississippi Delta

> Google Earth Engine script that pulls a multi-year Sentinel-2 time
> series, computes per-image NDWI, classifies water pixels, and
> produces a flood-frequency composite showing how often each pixel
> was inundated (0–100 %).

![Platform](https://img.shields.io/badge/platform-Google%20Earth%20Engine-4285F4?logo=google&logoColor=white)
![Sensor](https://img.shields.io/badge/sensor-Sentinel--2%20L2A-00838F)
![Method](https://img.shields.io/badge/index-NDWI%20McFeeters%201996-2e7d32)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Tool](https://img.shields.io/badge/GeoScriptHub-Tool%2012-blueviolet)

| | |
|---|---|
| **Platform** | [Google Earth Engine Code Editor](https://code.earthengine.google.com/) |
| **Sensor** | Sentinel-2 Level-2A (Surface Reflectance) |
| **Index** | NDWI — McFeeters (1996): (Green − NIR) / (Green + NIR) |
| **Study Area** | Lower Mississippi Delta, ~10 mi inland from the Gulf coast |
| **Licence** | MIT |
| **Tool #** | 12 · GeoScriptHub |

---

## ✨ Features

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

## 🚀 Quick Start

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

## ⚙️ How It Works

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

- **NDWI > 0.3** → open water (lakes, permanent rivers)
- **NDWI 0–0.3** → mixed water or wet soil (tidal flats, shallow channels)
- **NDWI < 0** → land, vegetation, or bare soil

> **Tip:** The default threshold of **0.1** is slightly conservative for delta
> wetlands where shadows and turbid water push NDWI just above zero.  Raise
> to 0.2–0.3 to focus on clearly open water; lower to 0.0 to capture every
> hint of moisture (more false positives in shadow zones).

### Flood Frequency

$$\text{Flood Frequency (\%)} = \frac{\sum_{i=1}^{n} \text{water}_i}{\sum_{i=1}^{n} \text{valid}_i} \times 100$$

Where $n$ is the number of images in the filtered collection,
$\text{water}_i$ is 1 if the pixel was classified as water in image $i$,
and $\text{valid}_i$ is 1 if the pixel was cloud-free.

---

## 🗺️ Study Area

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

## 🎨 Colour Palette & Interpretation Guide

| Frequency | Colour | Hex | Interpretation |
|---|---|---|---|
| 0 % | 🟡 Light yellow | `#ffffcc` | High ground — never classified as water |
| ~17 % | 🟢 Light green | `#c7e9b4` | Rare inundation — seasonal marsh margin |
| ~33 % | 🩵 Aqua | `#7fcdbb` | Occasional flooding — backswamp / low levee |
| ~50 % | 🔵 Teal | `#41b6c4` | Seasonal flooding — periodically inundated floodplain |
| ~67 % | 🔵 Blue | `#2c7fb8` | Frequently inundated — semi-permanent wetland |
| 100 % | 🟣 Dark blue | `#253494` | Permanent open water — main channel / bay |

---

## 📈 Reading the Time-Series Chart

The console chart plots **mean NDWI over the AOI** through time.
Here is what to look for:

| Pattern | What it means |
|---|---|
| Regular seasonal waves | Normal — wetter winters, drier summers |
| Sharp upward spikes | Flood events: hurricanes, extreme precipitation |
| Rising multi-year trend | Possible wetland expansion or land subsidence |
| Flat, no seasonality | Very arid AOI or insufficient cloud-free images |
| Gaps in the trace | Periods of persistent cloud cover — no clear images passed the filter |

---

## 💾 Exporting

Uncomment the `Export.image.toDrive` block at the bottom of the script
to save the flood-frequency raster as a 10 m GeoTIFF to your Google Drive:

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

> **QGIS tip:** Import the GeoTIFF and apply a Singleband Pseudocolour
> renderer with the same YlGnBu hex values to reproduce the GEE view exactly.

---

## ⚠️ Known Limitations

| Limitation | Notes |
|---|---|
| **Cloud-heavy regions** | Pixels under persistent cloud may have fewer than 5 valid observations — frequency values will be noisy |
| **Shadow false positives** | Canals in deep forest shade can push NDWI near the threshold; SCL class 3 masking helps but is imperfect |
| **No SAR fusion** | Optical-only — Sentinel-1 SAR can supplement detection in persistently cloudy areas |
| **10 m pixel size** | Very narrow streams (< 10 m wide) may not be resolved as water pixels |
| **Static threshold** | A single global `NDWI_THRESHOLD` may need per-season or per-region tuning |

---

## 📋 Requirements

- A [Google Earth Engine account](https://signup.earthengine.google.com/)
  (free for research, education, and non-commercial use)
- A modern browser (Chrome recommended)

---

## 📚 References

- McFeeters, S. K. (1996). *The use of the Normalized Difference Water
  Index (NDWI) in the delineation of open water features.* International
  Journal of Remote Sensing, 17(7), 1425–1432.
- ESA Sentinel-2 L2A product documentation:
  https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi
- Brewer, C. A. (2003). ColorBrewer: A tool for selecting colour schemes
  for maps. *The Cartographic Journal*, 40(1), 27–37.

---

## 🔗 Related Tools

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
| 13 | [Sub-Canopy Structure Detector](../sub-canopy-structure-detector/README.md) | SAR–optical fusion for hidden buildings |
