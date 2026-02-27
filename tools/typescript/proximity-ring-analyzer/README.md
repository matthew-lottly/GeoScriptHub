# Proximity Ring Analyzer

> ArcGIS Experience Builder custom widget — drop a pin on the map, draw
> concentric buffer rings, and instantly discover which features from
> every visible layer fall within each ring.

| | |
|---|---|
| **Platform** | ArcGIS Experience Builder (Developer Edition) ≥ 1.14 |
| **Language** | TypeScript + React (Jimu framework) |
| **ArcGIS JS API** | `@arcgis/core` ≥ 4.28 |
| **Licence** | MIT |
| **Tool #** | 8 · GeoScriptHub |

---

## Features

| Capability | Details |
|---|---|
| **Click-to-analyse** | Click anywhere on the map to set the analysis centre |
| **Geodesic buffers** | True geodesic rings via `geometryEngine.geodesicBuffer` |
| **Multi-layer query** | Queries every visible FeatureLayer automatically |
| **Ring differencing** | Each ring only counts features *between* its boundary and the previous ring |
| **Accordion results** | Expandable per-ring, per-layer breakdown with OID + display field |
| **CSV export** | One-click download of the full result set |
| **Full settings panel** | Configure distances, units, colours, opacity, max results |
| **Lightweight** | Zero additional dependencies beyond what ExB provides |

---

## How It Works

```
Click map
  │
  ▼
Centre point recorded
  │
  ▼
Build geodesic buffer rings (sorted ascending)
  │
  ├─ Ring 1: 0 → d₁
  ├─ Ring 2: d₁ → d₂
  └─ Ring n: dₙ₋₁ → dₙ
  │
  ▼
For each ring × each visible FeatureLayer:
  │
  ├─ geometryEngine.difference(outer, inner) → ring geometry
  ├─ layer.queryFeatures({ geometry: ring, spatialRelationship: 'intersects' })
  └─ Collect OID + display field
  │
  ▼
Render results in accordion UI  +  draw graphics on map
```

### Ring Differencing

Features are attributed to the **tightest** ring that contains them.
The innermost ring covers the full buffer from the centre to the first
distance.  Every subsequent ring uses `geometryEngine.difference()` to
subtract the previous ring, ensuring no double-counting.

---

## Installation

1. **Download** or clone this repository.
2. Copy the `proximity-ring-analyzer/` folder into your Experience
   Builder app's custom widgets directory:
   ```
   <ExB app>/client/your-extensions/widgets/proximity-ring-analyzer/
   ```
3. Restart the ExB dev server (`npm start` in the `client/` folder).
4. Open the Experience Builder page editor — the widget will appear in
   the **Insert → Widget** panel.

---

## Configuration (Settings Panel)

| Setting | Type | Default | Description |
|---|---|---|---|
| **Map Widget** | selector | — | Map widget the analyser binds to |
| **Ring Distances** | number[] | `[0.25, 0.5, 1.0]` | Ordered buffer distances |
| **Distance Unit** | enum | `miles` | `miles` · `kilometers` · `meters` · `feet` |
| **Max Results** | number | `500` | Cap per layer per ring |
| **Ring Colours** | hex[] | `["#1a9641","#a6d96a","#fdae61","#d7191c"]` | Cycles if fewer colours than rings |
| **Ring Opacity** | float | `0.25` | Fill transparency (0–1) |
| **Centre Colour** | hex | `#e31a1c` | Marker colour at the click point |

---

## File Structure

```
proximity-ring-analyzer/
├── manifest.json              # ExB widget manifest
├── config.ts                  # Config interface + defaults
├── package.json               # Dev dependencies (@types/react)
├── tsconfig.json              # TypeScript config with path aliases
├── icon.svg                   # Widget icon (concentric rings)
├── README.md
├── typings/                   # Stub declarations for standalone editing
│   ├── arcgis-core.d.ts
│   ├── jimu-arcgis.d.ts
│   ├── jimu-core.d.ts
│   └── jimu-ui.d.ts
└── src/
    ├── runtime/
    │   ├── widget.tsx         # Main widget component
    │   └── style.css          # Widget styles
    ├── setting/
    │   ├── setting.tsx        # Builder-facing settings panel
    │   └── style.css          # Settings styles
    └── translations/
        └── default.ts         # English strings
```

---

## API Surfaces Used

All APIs used are **stable, production-grade** parts of the ArcGIS JS
API and Jimu framework:

| API | Purpose |
|---|---|
| `geometryEngine.geodesicBuffer` | Accurate buffers in any projection |
| `geometryEngine.difference` | Ring subtraction for per-ring counting |
| `FeatureLayer.queryFeatures` | Server-side spatial intersection query |
| `GraphicsLayer` / `Graphic` | Drawing rings and the centre marker |
| `SimpleFillSymbol` / `SimpleMarkerSymbol` | Ring and marker styling |
| `JimuMapViewComponent` | ExB ↔ MapView bridge |
| `MapWidgetSelector` | Settings panel map picker |
| `Collapse`, `Card`, `Button`, … | Jimu UI components |

---

## Example Workflow

1. Open your Experience Builder app.
2. Add a **Map** widget showing your data.
3. Add the **Proximity Ring Analyzer** widget.
4. In settings, select the Map widget and configure ring distances
   (e.g. `0.25, 0.5, 1.0` miles).
5. Preview the app, click **Start — Click Map**, then click on the map.
6. Concentric rings appear; the results panel shows feature counts per
   ring per layer with expandable detail rows.
7. Click **Export CSV** to download the full result set.

---

## Requirements

- ArcGIS Experience Builder **Developer Edition** ≥ 1.14
- Node.js ≥ 16 (for the ExB dev server)
- A web map with at least one visible **FeatureLayer**

---

## Related Tools

| # | Tool | Description |
|---|---|---|
| 1 | [Batch Coordinate Transformer](../../../README.md) | Bulk CRS reprojection |
| 2 | [Batch Geocoder](../../../README.md) | Forward / reverse geocoding |
| 3 | [OSM Change Monitor](../../../README.md) | OpenStreetMap diff tracking |
| 4 | [Raster Band Stats](../../../README.md) | Raster statistics & histograms |
| 5 | [Shapefile Health Checker](../../../README.md) | Shapefile QA reports |
| 6 | [Spectral Index Calculator](../../../README.md) | NDVI / NDWI / custom indices |
| 7 | [FGDB Archive Publisher](../../../README.md) | ArcGIS FGDB archive pipeline |
| 8 | **Proximity Ring Analyzer** | ← you are here |
| 9 | [GeoJSON Diff Viewer](../../../README.md) | Side-by-side GeoJSON comparison |
| 10 | [Leaflet Widget Generator](../../../README.md) | Leaflet widget scaffolding |
| 11 | [Map Swiper](../../../README.md) | Before / after map comparison |
