# 🗺️ Leaflet Widget Generator

> Generate self-contained Leaflet.js map widgets from a typed TypeScript configuration object.

![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript&logoColor=white)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9%2B-green?logo=leaflet&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![GeoScriptHub](https://img.shields.io/badge/GeoScriptHub-Widget%207%2F10-purple)](../../README.md)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Tile Providers](#tile-providers)
- [Customization Guide](#customization-guide)
- [Running Tests](#running-tests)
- [Building for Production](#building-for-production)
- [Architecture](#architecture)

---

## Overview

**`LeafletWidgetGenerator`** takes a single `WidgetConfig` object and returns a ready-to-insert 
HTML/JS string that initialises a fully interactive Leaflet map.

Two output modes:

| Mode | Description |
|------|-------------|
| **Inline** (default) | `<script>` block only — Leaflet must already be loaded on the page |
| **Self-contained** | Complete `<link>` + `<script>` + `<div>` — drop onto any page |

Key features:
- **Typed configuration** — every option has strict TypeScript types and inline JSDoc  
- **Four tile provider presets** — OpenStreetMap, OpenTopoMap, Stadia Alidade Smooth, custom  
- **GeoJSON overlays** — fetch any URL, apply custom fill/stroke colours  
- **Markers with popups** — arbitrary HTML popup content  
- **Layer control, scale bar, zoom control** — toggle with single booleans  
- **Vite lib mode** — ships as ESM + UMD bundles with `.d.ts` declarations  

---

## Installation

```bash
# In your project
pnpm add @geoscripthub/leaflet-widget-generator leaflet

# Or from this monorepo (dev)
cd tools/typescript/leaflet-widget-generator
pnpm install
```

---

## Quick Start

### TypeScript / ESM

```typescript
import { LeafletWidgetGenerator } from "@geoscripthub/leaflet-widget-generator";

const generator = new LeafletWidgetGenerator({
  containerId: "map",             // PLACEHOLDER: id of your <div>
  view: { lat: 51.505, lng: -0.09, zoom: 13 }, // PLACEHOLDER: your coordinates
  tileProvider: "openstreetmap",
  selfContained: true,
});

document.getElementById("output")!.innerHTML = generator.generate();
```

### Self-contained HTML drop-in

```typescript
const html = new LeafletWidgetGenerator({
  containerId: "my-map",
  height: "600px",
  view: { lat: 51.505, lng: -0.09, zoom: 13 },
  tileProvider: "openstreetmap",
  markers: [
    { lat: 51.505, lng: -0.09, popupHtml: "<b>Hello!</b>" },
  ],
  selfContained: true,
}).generate();

// `html` can be saved to a .html file or injected anywhere
console.log(html);
```

### Live demo

```bash
pnpm install
pnpm build
# Open demo/index.html in a browser — or serve with:
pnpm preview
```

---

## API Reference

### `new LeafletWidgetGenerator(config: WidgetConfig)`

Constructs a generator instance.  Throws on invalid `containerId`, out-of-range coordinates, 
or `tileProvider: "custom"` with a missing `tileUrl`.

### `generator.generate(): string`

Returns the HTML/JS widget string.  In `selfContained` mode this is a complete snippet; 
otherwise it is a `<script>` block only.

### `generator.toString(): string`

Alias for `generate()`.

---

## Configuration Reference

All `WidgetConfig` properties:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `containerId` | `string` | **required** | CSS id of the map `<div>` |
| `view` | `ViewConfig` | **required** | `{ lat, lng, zoom }` |
| `height` | `string` | `"500px"` | Container CSS height |
| `width` | `string` | `"100%"` | Container CSS width |
| `tileProvider` | `TileProvider` | `"openstreetmap"` | Preset tile source |
| `tileUrl` | `string` | — | Custom tile URL (when `tileProvider: "custom"`) |
| `tileAttribution` | `string` | — | Custom attribution HTML |
| `stadiaApiKey` | `string` | — | Stadia Maps API key |
| `geoJsonLayers` | `GeoJsonLayerConfig[]` | `[]` | GeoJSON overlay layers |
| `markers` | `MarkerConfig[]` | `[]` | Map markers |
| `showZoomControl` | `boolean` | `true` | Show/hide zoom ± buttons |
| `showLayerControl` | `boolean` | `true` | Show/hide layer toggle (when layers > 1) |
| `scaleControl` | `ScaleControlConfig` | — | Scale bar configuration |
| `selfContained` | `boolean` | `false` | Include CDN links in output |

### `ViewConfig`

```typescript
{ lat: number; lng: number; zoom: number }
// PLACEHOLDER: lat [-90,90], lng [-180,180], zoom [0,22]
```

### `GeoJsonLayerConfig`

```typescript
{
  url: string;          // PLACEHOLDER: GeoJSON file URL or data: URI
  name: string;         // PLACEHOLDER: Human-readable layer name
  fillColor?: string;   // PLACEHOLDER: e.g. "#e74c3c"
  color?: string;       // PLACEHOLDER: stroke colour
  weight?: number;      // PLACEHOLDER: stroke width in pixels
  visible?: boolean;    // default: true
}
```

### `MarkerConfig`

```typescript
{
  lat: number;          // PLACEHOLDER: marker latitude
  lng: number;          // PLACEHOLDER: marker longitude
  popupHtml?: string;   // PLACEHOLDER: HTML shown in click popup
}
```

---

## Tile Providers

| Key | Name | Free | API Key |
|-----|------|------|---------|
| `openstreetmap` | OpenStreetMap Standard | ✅ | None |
| `opentopomap` | OpenTopoMap | ✅ | None |
| `stadia-alidade-smooth` | Stadia Maps Light | ✅ dev | `stadiaApiKey` in production |
| `custom` | Your own tile server | — | `tileUrl` required |

---

## Customization Guide

| Placeholder | Location | Description | Example |
|-------------|----------|-------------|---------|
| `YOUR_NAME` / `YOUR_EMAIL` | `package.json` → `author` | Package metadata | `"Jane <jane@example.com>"` |
| `containerId` | `WidgetConfig` | id of your map `<div>` | `"map"`, `"city-map"` |
| `view.lat/lng/zoom` | `WidgetConfig.view` | Map centre + initial zoom | `{ lat: 48.85, lng: 2.35, zoom: 12 }` |
| `tileProvider` | `WidgetConfig` | Choose basemap | `"opentopomap"` |
| `tileUrl` | `WidgetConfig` | Custom tile server URL | `"https://{s}.tile.example.com/{z}/{x}/{y}.png"` |
| `stadiaApiKey` | `WidgetConfig` | Stadia Maps API key | `"abc123..."` (from stadiamaps.com) |
| `geoJsonLayers[].url` | `WidgetConfig` | GeoJSON data source | `"https://example.com/data.geojson"` |
| `geoJsonLayers[].fillColor` | `WidgetConfig` | Polygon fill colour | `"#27ae60"` |
| `markers[].popupHtml` | `WidgetConfig` | Popup inner HTML | `"<b>City Hall</b><br>Est. 1872"` |
| `height` / `width` | `WidgetConfig` | Container dimensions | `"600px"`, `"80vh"` |
| `LEAFLET_VERSION` | `LeafletWidgetGenerator.ts` | CDN Leaflet version | `"1.9.4"` (update as needed) |

---

## Running Tests

```bash
pnpm install
pnpm test        # single run
pnpm test:watch  # watch mode
```

Tests run in jsdom via Vitest — no browser required.

---

## Building for Production

```bash
pnpm build
```

Outputs to `dist/`:

```
dist/
├── leaflet-widget-generator.js        # ESM bundle
├── leaflet-widget-generator.umd.cjs   # UMD bundle (CommonJS / browser script)
└── index.d.ts                         # TypeScript declarations
```

---

## Architecture

```
leaflet-widget-generator/
├── src/
│   ├── types.ts                  # WidgetConfig + sub-types
│   ├── LeafletWidgetGenerator.ts # Main class
│   └── index.ts                  # Public API re-exports
├── tests/
│   └── LeafletWidgetGenerator.test.ts
├── demo/
│   └── index.html                # Interactive browser demo
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md                     # This file
```
