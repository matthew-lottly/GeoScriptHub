# Sub-Canopy Structure Detector

> Google Earth Engine script that fuses **Sentinel-1 C-band SAR** with
> **Sentinel-2 optical** imagery to locate man-made structures (buildings,
> rooftops, concrete pads) hidden beneath forest canopy — structures
> that are invisible in standard satellite photos.

| | |
|---|---|
| **Platform** | [Google Earth Engine Code Editor](https://code.earthengine.google.com/) |
| **Sensors** | Sentinel-1 GRD (SAR) + Sentinel-2 L2A (optical) |
| **Technique** | Multi-indicator SAR–optical fusion |
| **Default AOI** | Petén, Guatemala — tropical forest with known hidden settlements |
| **Licence** | MIT |
| **Tool #** | 13 · GeoScriptHub |

---

## Why This Is Hard

Standard building detection relies on **optical imagery** — spectral
signatures of concrete and metal are easy to separate from vegetation.
But when a building sits **under a dense tree canopy**, the optical
sensor sees *only trees*.  Pixel-level classification, deep learning on
RGB, and even high-resolution commercial imagery all fail when the
structure is physically hidden by foliage.

**Radar changes the game.** C-band microwaves (5.4 GHz) partially
penetrate vegetation canopy and interact with hard surfaces underneath
through a mechanism called **double-bounce scattering** — the signal
bounces from the ground to a vertical wall and back to the satellite.
This creates a distinctive, temporally stable signature that forest
canopy alone does not produce.

This script is the first complete, open-source GEE implementation that
fuses **five independent SAR and optical indicators** into a single
probabilistic detection surface specifically designed for sub-canopy
structures.

---

## Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     SENTINEL-1 SAR (C-band)                     │
│       IW mode · VV + VH · configurable orbit direction          │
└──────────┬──────────┬──────────┬──────────┬─────────────────────┘
           │          │          │          │
     ┌─────▼─────┐ ┌─▼────────┐│   ┌──────▼──────┐
     │ Temporal  │ │Polarimet-││   │   Local     │
     │ Stability │ │ric Ratio ││   │ Backscatter │
     │ (1 − CoV) │ │ VH / VV  ││   │  Anomaly   │
     └─────┬─────┘ └─┬────────┘│   └──────┬──────┘
           │          │         │          │
           │ ①        │ ②       │ ③        │ ④
           │          │    ┌────▼─────┐    │
           │          │    │  GLCM    │    │
           │          │    │ Texture  │    │
           │          │    └────┬─────┘    │
           │          │         │          │
┌──────────┼──────────┼─────────┼──────────┼──────────────────────┐
│          │          │         │          │    SENTINEL-2        │
│          │          │         │          │    Optical           │
│          │          │         │          │                      │
│          │          │         │    ┌─────▼──────┐              │
│          │          │         │    │   NDBI     │              │
│          │          │         │    │ Micro-     │              │
│          │          │         │    │ Anomaly    │              │
│          │          │         │    └─────┬──────┘              │
│          │          │         │          │ ⑤                   │
│          │          │         │          │                      │
│   ┌──────┴──────────┴─────────┴──────────┴──────┐              │
│   │         WEIGHTED FUSION  (Σ wᵢ · sᵢ)       │              │
│   └──────────────────┬──────────────────────────┘              │
│                      │                                          │
│                 ┌────▼────────────┐                             │
│                 │   FOREST MASK   │  ← NDVI ≥ 0.55             │
│                 │  (restrict to   │    & not water              │
│                 │   canopy only)  │                             │
│                 └────┬────────────┘                             │
│                      │                                          │
│              ┌───────▼────────┐                                 │
│              │ Morphological  │                                 │
│              │   Cleanup      │                                 │
│              └───────┬────────┘                                 │
│                      │                                          │
│          ┌───────────▼───────────┐                              │
│          │  CONFIDENCE ZONES     │                              │
│          │  High / Medium / Low  │                              │
│          └───────────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Five Indicators

### ① SAR Temporal Stability

$$\text{Stability} = 1 - \text{CoV} = 1 - \frac{\sigma_{VV}}{\mu_{VV}}$$

Buildings are **persistent scatterers** — their backscatter barely
changes over weeks and months.  Forest canopy fluctuates with wind,
moisture, leaf phenology, and growth.  A stability value near 1.0
almost always indicates a hard, fixed surface.

### ② Polarimetric Double-Bounce Index

$$\text{Score} = 1 - \frac{VH / VV - 0.02}{0.30 - 0.02}$$

The cross-polarisation ratio (VH / VV) separates scattering
mechanisms.  **Volume scattering** (canopy) produces strong VH →
high ratio.  **Double-bounce** (wall + ground) produces strong VV →
low ratio.  Inverting the ratio gives a score where high = more
building-like.

### ③ GLCM Texture Contrast

$$\text{Contrast} = \sum_{i,j} (i - j)^2 \cdot P(i, j)$$

The Gray-Level Co-occurrence Matrix captures spatial regularity.
Man-made structures introduce sharp edges and geometric patterns
that produce high contrast in the SAR backscatter field.

### ④ Local Backscatter Anomaly

$$z = \frac{\text{pixel}_{VV} - \mu_{\text{local}}}{\sigma_{\text{local}}}$$

A building under canopy creates a localised VV hotspot.  We flag
pixels where the backscatter exceeds the neighbourhood mean by more
than $1.5\sigma$ (adjustable).

### ⑤ Optical NDBI Micro-Anomaly

$$\text{NDBI} = \frac{B11_{\text{SWIR}} - B8_{\text{NIR}}}{B11_{\text{SWIR}} + B8_{\text{NIR}}}$$

Even under canopy, mixed pixels containing roof material shift the
NDBI slightly higher than surrounding pure-forest pixels.  We detect
this as a local z-score anomaly above the forest baseline.

---

## Fusion & Confidence

The five scores are combined with user-tuneable weights:

$$P(\text{structure}) = w_1 \cdot S_{\text{stability}} + w_2 \cdot S_{\text{pol}} + w_3 \cdot S_{\text{texture}} + w_4 \cdot S_{\text{anomaly}} + w_5 \cdot S_{\text{optical}}$$

| Weight | Default | Indicator |
|---|---|---|
| $w_1$ | 0.30 | SAR temporal stability |
| $w_2$ | 0.20 | Polarimetric double-bounce |
| $w_3$ | 0.20 | GLCM texture contrast |
| $w_4$ | 0.20 | Local backscatter anomaly |
| $w_5$ | 0.10 | Optical NDBI micro-anomaly |

The probability surface is then:
1. **Masked** to forested pixels only (NDVI ≥ 0.55, not water)
2. **Terrain-corrected** — steep slopes (> 15°) excluded via Copernicus
   GLO-30 DEM to eliminate radar shadow/layover false positives
3. **Cleaned** with morphological opening (erode + dilate) to remove
   salt-and-pepper noise
4. **Classified** into confidence zones:
   - **High** (≥ 0.65) — very likely a hidden structure
   - **Medium** (≥ 0.45) — possible, needs ground truth
   - **Low** (< 0.45) — weak signal
5. **Cross-validated** against GHSL (Global Human Settlement Layer)

---

## Quick Start

1. Open the [GEE Code Editor](https://code.earthengine.google.com/).
2. Create a new script and paste [`script.js`](script.js).
3. Adjust the parameters at the top (or leave the defaults).
4. Click **Run**.
5. In the **Layers** panel:
   - **★ Detected Hidden Structures** — yellow to red probability
   - **★ Confidence Zones** — discrete High / Medium / Low
   - Toggle individual indicators ①–⑤ to understand each signal
   - Enable **GHSL**, **✓ Confirmed**, or **★ Novel** layers for
     cross-validation against the Global Human Settlement Layer
6. **Click any pixel** on the map → full per-indicator diagnostic
   printed to the Console (all 5 scores + fusion + confidence + GHSL).
7. Use the **🧪 Test Sites** panel (top-left) to navigate to six
   curated global locations with known sub-canopy structures.
8. Check the **Console** for two VV time-series charts (full AOI +
   high-confidence zones) and the probability histogram.

---

## User Parameters

| Parameter | Default | Description |
|---|---|---|
| `MAX_CLOUD_COVER` | `15` | S2 cloud-cover filter (%) |
| `START_DATE` / `END_DATE` | `2022-01-01` / `2024-12-31` | Analysis window |
| `FOREST_NDVI_THRESHOLD` | `0.55` | NDVI cutoff for the forest mask |
| `WATER_NDWI_THRESHOLD` | `0.15` | NDWI cutoff to exclude water |
| `STABILITY_FLOOR` | `0.70` | Minimum stability for indicator ① |
| `TEXTURE_KERNEL_RADIUS` | `3` | GLCM kernel size (pixels) |
| `ANOMALY_KERNEL_RADIUS` | `15` | Local-anomaly neighbourhood (pixels) |
| `ANOMALY_SIGMA` | `1.5` | σ multiplier for anomaly detection |
| `W_STABILITY` | `0.30` | Fusion weight — temporal stability |
| `W_POLRATIO` | `0.20` | Fusion weight — polarimetric ratio |
| `W_TEXTURE` | `0.20` | Fusion weight — GLCM texture |
| `W_ANOMALY` | `0.20` | Fusion weight — backscatter anomaly |
| `W_OPTICAL` | `0.10` | Fusion weight — optical NDBI |
| `THRESH_HIGH` | `0.65` | High-confidence floor |
| `THRESH_MEDIUM` | `0.45` | Medium-confidence floor |
| `ORBIT_DIRECTION` | `'ASCENDING'` | S1 orbit pass: `'ASCENDING'`, `'DESCENDING'`, or `'BOTH'` |
| `SLOPE_THRESHOLD` | `15` | Max terrain slope (°) — steeper pixels excluded |
| `POL_RATIO_MIN` | `0.02` | Min VH/VV for double-bounce normalisation |
| `POL_RATIO_MAX` | `0.30` | Max VH/VV for double-bounce normalisation |
| `AOI` | Petén, Guatemala | Study area polygon |

---

## Map Layers

| Layer | Default | Description |
|---|---|---|
| **Sentinel-2 True Colour** | ON | Optical base map |
| **★ Detected Hidden Structures** | ON | Cleaned probability surface (yellow → red) |
| **★ Confidence Zones** | ON | Discrete 1 / 2 / 3 classification |
| **Forest Mask** | off | Green overlay — NDVI ≥ threshold AND slope < threshold |
| **Terrain Slope** | off | Copernicus DEM slope (white → red) |
| **① SAR Temporal Stability** | off | Black → white (0 → 1) |
| **② Polarimetric Double-Bounce** | off | Viridis colour ramp |
| **③ GLCM Texture Score** | off | Black → cyan → white |
| **④ Local Backscatter Anomaly** | off | Black → orange → red |
| **⑤ Optical NDBI Micro-Anomaly** | off | Black → pink → magenta |
| **GHSL Built-up in Forest** | off | Cyan — GHSL reference built-up under canopy |
| **✓ Confirmed** | off | Green — our detection overlaps GHSL |
| **★ Novel Detection** | off | Magenta — our detection, not in GHSL |
| **Study Area Boundary** | ON | Cyan outline |

---

## Use Cases

| Domain | Application |
|---|---|
| **Deforestation monitoring** | Illegal settlements as early indicators of forest loss |
| **Tax / cadastral assessment** | Undeclared structures in rural wooded parcels |
| **Disaster response** | Locating buildings for rescue when canopy obscures aerial views |
| **Conservation enforcement** | Detecting encroachment in protected forest reserves |
| **Military / intelligence** | Identifying concealed infrastructure |
| **Urban planning** | Mapping informal peri-urban growth into forested hills |

---

## Validation & Testing

The script includes a built-in validation system:

### Click-to-Inspect

Click **any pixel** on the map.  The Console prints a full diagnostic:

```
═══════════════════════════════════════════════════
  PIXEL DIAGNOSTIC — 17.12500°N, -90.00500°E
═══════════════════════════════════════════════════
① Stability Score      : 0.8123
② Double-Bounce Score  : 0.6541
③ Texture Score        : 0.4210
④ Anomaly Score        : 0.5802
⑤ Optical (NDBI) Score : 0.1900
───────────────────────────────────────────────────
Fused Probability      : 0.5840
Confidence Zone        : MEDIUM
───────────────────────────────────────────────────
Forest Pixel           : YES
Terrain Slope          : 4.2100°
GHSL Built-up %        : 0.0000
═══════════════════════════════════════════════════
```

This lets you instantly verify whether each indicator is contributing
logically at any point worldwide.

### Global Test Sites

A **🧪 Test Sites** panel (top-left corner) lets you fly to six curated
locations with known sub-canopy structures:

| # | Site | Biome |
|---|------|-------|
| 1 | Petén, Guatemala | Tropical broadleaf jungle |
| 2 | Leticia, Colombia | Amazon riparian forest |
| 3 | Ulu Baram, Borneo | Tropical rainforest |
| 4 | Black Forest, Germany | Temperate conifer forest |
| 5 | Portland Metro, Oregon | Temperate mixed forest |
| 6 | Mt Halimun, Java | Montane tropical forest |

To run the full analysis at a different site, copy the preset AOI from
the commented-out library at the bottom of the script, replace the
default AOI, and click **Run**.

### GHSL Cross-Validation

The script loads the **Global Human Settlement Layer** (GHSL-BUILT-S
2020, 10 m) and computes overlap statistics:

| Layer | Meaning |
|---|---|
| **✓ Confirmed** (green) | Our detection AND GHSL shows built-up |
| **★ Novel** (magenta) | Our detection BUT GHSL shows nothing |
| **GHSL in Forest** (cyan) | GHSL built-up under our forest mask |

"Novel" detections are the most interesting — these are structures
that even GHSL missed, likely because they sit beneath dense canopy
where optical-only methods fail.

---

## Limitations & Caveats

- **C-band penetration is partial.** Very dense multi-layer tropical
  canopy attenuates the signal; L-band SAR (ALOS PALSAR) penetrates
  deeper but is not freely available in GEE at the same cadence.
- **Small structures** (< ~10 m footprint) may not produce a detectable
  double-bounce at Sentinel-1's 10–20 m ground resolution.
- **Terrain effects.** Steep slopes are now excluded by default using
  the Copernicus GLO-30 DEM (threshold: 15°).  Adjust `SLOPE_THRESHOLD`
  for different terrain.
- **False positives** can arise from exposed rock, power-line towers,
  large fallen trees, or any hard surface under canopy.
- **Ground truth** is essential to calibrate weights and thresholds for
  a specific region.  Use the click-to-inspect diagnostic and GHSL
  comparison to iteratively tune parameters.

---

## Exporting

Uncomment the `Export.image.toDrive` blocks at the bottom of the script
to save:

- **Hidden structure probability** (float, 10 m)
- **Confidence zones** (byte, 10 m) — values 1 / 2 / 3
- **Cleaned detections** (float, 10 m) — probability for medium+ pixels
- **GHSL validation raster** (byte, 10 m) — agreement + novel bands

---

## References

- Ferretti, A., Prati, C., & Rocca, F. (2001). *Permanent scatterers
  in SAR interferometry.* IEEE TGRS, 39(1), 8–20.
- Lee, J.-S., & Pottier, E. (2009). *Polarimetric Radar Imaging: From
  Basics to Applications.* CRC Press.
- McFeeters, S. K. (1996). *The use of the Normalized Difference Water
  Index.* Int. J. Remote Sensing, 17(7), 1425–1432.
- Zha, Y., Gao, J., & Ni, S. (2003). *Use of normalized difference
  built-up index in automatically mapping urban areas from TM imagery.*
  Int. J. Remote Sensing, 24(3), 583–594.
- Haralick, R. M. (1973). *Textural features for image classification.*
  IEEE Trans. Systems, Man, Cybernetics, 3(6), 610–621.

---

## Requirements

- A [Google Earth Engine account](https://signup.earthengine.google.com/)
- A modern browser (Chrome recommended)
- No local software required

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
| 12 | [NDWI Flood-Frequency Mapper](../../../README.md) | Sentinel-2 flood frequency |
| 13 | **Sub-Canopy Structure Detector** | ← you are here |
