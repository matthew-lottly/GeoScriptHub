# 🗺 GeoJSON Diff Viewer

> **@geoscripthub/geojson-diff-viewer** — A Leaflet-based TypeScript widget that
> colour-codes the differences between two GeoJSON FeatureCollections directly on
> an interactive map.

[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9-199900?logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![Vite](https://img.shields.io/badge/Vite-5.x-646cff?logo=vite&logoColor=white)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)
[![GeoScriptHub](https://img.shields.io/badge/GeoScriptHub-Portfolio-00d9ff)](../../README.md)

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🟢 **Added** features | Highlighted in **green** (`#27ae60`) |
| 🔴 **Removed** features | Highlighted in **red** (`#e74c3c`) |
| ⚫ **Unchanged** features | Highlighted in **grey** (`#95a5a6`) |
| 📐 **Auto-fit** | Map pans + zooms to the combined feature extent |
| 📖 **Legend overlay** | Built-in Leaflet control showing counts per status |
| 🔑 **Flexible matching** | Match features by property key(s) or geometry (default) |
| 📦 **ESM + UMD builds** | Works in modern bundlers and plain `<script>` tags |
| 🧪 **Full test suite** | Pure DiffEngine unit tests + DOM tests via Vitest / jsdom |

---

## 📦 Installation

```bash
# pnpm (recommended)
pnpm add @geoscripthub/geojson-diff-viewer leaflet

# npm
npm install @geoscripthub/geojson-diff-viewer leaflet

# yarn
yarn add @geoscripthub/geojson-diff-viewer leaflet
```

> **Leaflet is a peer dependency** — you must install it separately and include
> its CSS in your HTML.

---

## 🚀 Quick Start

### ESM (Bundler / Vite / Webpack)

```typescript
import { GeoJsonDiffViewer } from "@geoscripthub/geojson-diff-viewer";
import "leaflet/dist/leaflet.css";

const viewer = new GeoJsonDiffViewer({
  containerId: "diff-map",
  height:      "500px",
  showLegend:  true,
});

viewer.mount();
viewer.update(beforeCollection, afterCollection);
```

### Plain HTML (UMD / CDN)

```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script src="dist/geojson-diff-viewer.umd.cjs"></script>

<div id="diff-map" style="height: 500px;"></div>

<script>
  const viewer = new GeoJsonDiffViewer.GeoJsonDiffViewer({
    containerId: "diff-map",
  });
  viewer.mount();
  viewer.update(myBeforeFC, myAfterFC);
</script>
```

---

## 🔧 API Reference

### `new GeoJsonDiffViewer(config: DiffViewerConfig)`

Create a new viewer instance. Throws if `containerId` is empty.

| Method | Signature | Description |
|---|---|---|
| `mount()` | `(): void` | Creates the Leaflet map inside the target container. Call this once before `update()`. |
| `update(before, after)` | `(FeatureCollection, FeatureCollection): void` | Re-runs the diff and refreshes all map layers. |
| `unmount()` | `(): void` | Destroys the Leaflet map and clears internal state. Safe to call repeatedly. |

---

### `new DiffEngine(options?: DiffEngineOptions)`

Pure diff logic, no DOM required.

```typescript
import { DiffEngine } from "@geoscripthub/geojson-diff-viewer";

const engine = new DiffEngine({ matchBy: "osm_id" });
const result = engine.diff(beforeFC, afterFC);

console.log(DiffEngine.summarise(result));
// Total features examined : 120
//   Added                 : 5
//   Removed               : 3
//   Unchanged             : 112
```

| Method | Signature | Description |
|---|---|---|
| `diff(a, b)` | `(FeatureCollection, FeatureCollection): DiffResult` | Compute added / removed / unchanged arrays. |
| `DiffEngine.summarise(result)` | `static (DiffResult): string` | Human-readable summary string. |

---

## 📋 DiffResult Shape

```typescript
interface DiffResult {
  added:     DiffFeature[];   // in b, not in a
  removed:   DiffFeature[];   // in a, not in b
  unchanged: DiffFeature[];   // in both a and b
}

interface DiffFeature {
  feature: GeoJSON.Feature;
  status:  "added" | "removed" | "unchanged";
}
```

---

## 🎨 Colour Reference

| Status | Default Colour | Hex |
|---|---|---|
| Added | Green | `#27ae60` |
| Removed | Red | `#e74c3c` |
| Unchanged | Grey | `#95a5a6` |

Override any colour via `DiffViewerConfig.styles`.

---

## 🛠 Customization Guide

Every configurable option is documented below.

| Setting | Location | What to Change |
|---|---|---|
| `containerId` | `DiffViewerConfig` | CSS `id` of your map `<div>` |
| `height` / `width` | `DiffViewerConfig` | CSS dimensions of the map container |
| `tileUrl` | `DiffViewerConfig` | XYZ tile URL for your preferred base map |
| `tileAttribution` | `DiffViewerConfig` | HTML attribution string for your tile provider |
| `styles.addedColor` | `DiffStyle` | Fill/stroke colour for added features |
| `styles.removedColor` | `DiffStyle` | Fill/stroke colour for removed features |
| `styles.unchangedColor` | `DiffStyle` | Fill/stroke colour for unchanged features |
| `styles.weight` | `DiffStyle` | Stroke width in pixels |
| `styles.fillOpacity` | `DiffStyle` | Polygon fill opacity (0–1) |
| `diffOptions.matchBy` | `DiffEngineOptions` | Property name(s) for feature matching, or `null` for geometry |
| `showLegend` | `DiffViewerConfig` | `false` to hide the legend overlay |

### Common Tile Providers

| Provider | URL Template |
|---|---|
| OpenStreetMap | `https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png` |
| OpenTopoMap | `https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png` |
| Stadia Smooth Dark | `https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png` |
| Esri World Imagery | `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}` |

> Always check the provider's terms of service and add the correct attribution.

---

## 🧪 Development

```bash
# Install dependencies
pnpm install

# Run tests (single pass)
pnpm test

# Run tests in watch mode
pnpm test:watch

# Type check
pnpm typecheck

# Build ESM + UMD bundles
pnpm build
```

### Project Structure

```
geojson-diff-viewer/
├── src/
│   ├── types.ts                # Type definitions and interfaces
│   ├── DiffEngine.ts           # Pure diff logic (no DOM)
│   ├── GeoJsonDiffViewer.ts    # Leaflet-based map widget
│   └── index.ts                # Public re-exports
├── tests/
│   ├── DiffEngine.test.ts      # Unit tests for diff logic
│   └── GeoJsonDiffViewer.test.ts # DOM tests with Leaflet mocked
├── demo/
│   └── index.html              # Interactive demo
├── package.json
├── tsconfig.json
├── vite.config.ts
└── eslint.config.js
```

---

## How Feature Matching Works

The `DiffEngine` needs a way to decide whether a feature in *before* corresponds
to a feature in *after*.

| `matchBy` value | Strategy |
|---|---|
| `null` (default) | Compare serialised geometry (rounded to 6 dp) |
| `"id"` | Compare `feature.properties.id` values |
| `["type", "ref"]` | Composite key — concatenate multiple property values |

> **Tip:** Use `matchBy: null` when features have no stable identifier and you
> want geometry-level change detection.

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for branch naming, commit rules,
and PR guidelines.

---

## License

MIT — see [LICENSE](../../LICENSE).

---

*Part of the [GeoScriptHub](../../README.md) portfolio — a curated collection of
production-ready GIS tools and widgets.*
