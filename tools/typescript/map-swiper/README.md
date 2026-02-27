# 🔀 Map Swiper

> A before/after map comparison widget with a draggable divider, built on MapLibre GL JS.

![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript&logoColor=white)
![MapLibre GL](https://img.shields.io/badge/MapLibre%20GL-4%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![GeoScriptHub](https://img.shields.io/badge/GeoScriptHub-Widget%208%2F10-purple)](../../README.md)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Free Style URLs](#free-style-urls)
- [Customization Guide](#customization-guide)
- [Running Tests](#running-tests)
- [Building for Production](#building-for-production)
- [Architecture](#architecture)

---

## Overview

**`MapSwiper`** renders two MapLibre GL maps inside a single container, divided by a draggable 
vertical handle.  The user can slide the divider to reveal how the same location looks with two 
different map styles — ideal for before/after comparisons (aerial ↔ street, historical ↔ current, 
day ↔ night style).

Key features:

- **Draggable divider** — mouse and touch support  
- **Two-map synchronisation** — pan/zoom locked together by default  
- **Any MapLibre style** — free public styles or your own tiles  
- **Programmatic control** — `setDividerPosition()` for animations  
- **Responsive** — `ResizeObserver` keeps both maps sized correctly  
- **Zero external CSS** — all styles injected at runtime  
- **Full TypeScript** — strict types, exported `.d.ts` declarations  

---

## Installation

```bash
pnpm add @geoscripthub/map-swiper maplibre-gl
```

> Leaflet and MapLibre are peer dependencies — you need to load MapLibre GL CSS separately.

```html
<link href="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css" rel="stylesheet" />
```

---

## Quick Start

```typescript
import { MapSwiper } from "@geoscripthub/map-swiper";

const swiper = new MapSwiper({
  containerId: "swiper",
  center: [-0.09, 51.505],
  zoom: 12,
  left: {
    style: "https://demotiles.maplibre.org/style.json",
    label: "Before",
  },
  right: {
    style: "https://demotiles.maplibre.org/style.json",
    label: "After",
  },
});

swiper.mount();
```

HTML:

```html
<div id="swiper" style="width:100%; height:500px;"></div>
```

---

## API Reference

### `new MapSwiper(config: MapSwiperConfig)`

Construct a swiper.  Throws on:
- empty `containerId`
- `center` latitude out of `[-90, 90]`
- `center` longitude out of `[-180, 180]`
- `zoom` out of `[0, 22]`

### `swiper.mount(): void`

Insert the two map panels and wires all event listeners into `containerId`.  
Throws if the container element is not found in the DOM.

### `swiper.setDividerPosition(fraction: number): void`

Move the divider programmatically.  `fraction` is `0.0–1.0` (clamped to `0.02–0.98`).

```typescript
swiper.setDividerPosition(0.3); // left panel shows 30% of width
```

### `swiper.unmount(): void`

Cleanly destroy both maps and remove all event listeners.

---

## Configuration Reference

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `containerId` | `string` | **required** | DOM element id |
| `center` | `[lng, lat]` | **required** | Map centre |
| `zoom` | `number` | **required** | Initial zoom |
| `left` | `PanelConfig` | **required** | Left panel style + label |
| `right` | `PanelConfig` | **required** | Right panel style + label |
| `initialDividerPosition` | `number` | `0.5` | Starting divider fraction |
| `syncMaps` | `boolean` | `true` | Lock pan/zoom together |
| `dividerWidth` | `number` | `4` | Handle line width (px) |
| `dividerColor` | `string` | `"#ffffff"` | Handle CSS colour |

### `PanelConfig`

```typescript
{
  style: string | object;  // MapLibre style URL or inline style object
  label?: string;          // e.g. "2015 Satellite", "Current OSM"
}
```

---

## Free Style URLs

These MapLibre-compatible style URLs require no API key:

| Style | URL |
|-------|-----|
| MapLibre Demo Tiles | `https://demotiles.maplibre.org/style.json` |
| OpenFreeMap Liberty | `https://tiles.openfreemap.org/styles/liberty` |
| OpenFreeMap Bright  | `https://tiles.openfreemap.org/styles/bright`  |
| OpenFreeMap Positron | `https://tiles.openfreemap.org/styles/positron` |

For commercial-quality tiles requiring an API key:
- **MapTiler** — `https://api.maptiler.com/maps/streets/style.json?key=YOUR_KEY`
- **Stadia Maps** — `https://tiles.stadiamaps.com/styles/alidade_smooth.json?api_key=YOUR_KEY`

---

## Customization Guide

| Setting | Location | Description | Example |
|-------------|----------|-------------|---------|
| `containerId` | `MapSwiperConfig` | DOM element id | `"my-map-swiper"` |
| `center` | `MapSwiperConfig` | Map centre `[lng, lat]` | `[2.35, 48.85]` (Paris) |
| `zoom` | `MapSwiperConfig` | Initial zoom | `12` |
| `left.style` | `PanelConfig` | Left panel style URL | Any MapLibre style URL |
| `right.style` | `PanelConfig` | Right panel style URL | Any MapLibre style URL |
| `left.label` / `right.label` | `PanelConfig` | Badge text | `"2015"`, `"2024"` |
| `initialDividerPosition` | `MapSwiperConfig` | Starting split fraction | `0.5` (centre), `0.3` |
| `dividerColor` | `MapSwiperConfig` | Handle colour | `"#fff"`, `"#e74c3c"` |
| `dividerWidth` | `MapSwiperConfig` | Handle line width in px | `4` (default), `2` (thin) |
| Container `height` / `width` | CSS / `demo/index.html` | Map dimensions | `"60vh"`, `"500px"` |

---

## Running Tests

```bash
pnpm install
pnpm test
```

MapLibre GL is mocked via Vitest — no WebGL or browser required.

```
✓ MapSwiper — construction > should construct without error with valid config
✓ MapSwiper — mount() > should render left and right panel divs
✓ MapSwiper — mount() > should render a divider element
...
```

---

## Building for Production

```bash
pnpm build
```

```
dist/
├── map-swiper.js        # ESM bundle (Leaflet/MapLibre excluded)
├── map-swiper.umd.cjs   # UMD bundle
└── index.d.ts           # TypeScript declarations
```

---

## Architecture

```
map-swiper/
├── src/
│   ├── types.ts       # MapSwiperConfig, PanelConfig types
│   ├── MapSwiper.ts   # Main widget class
│   └── index.ts       # Public re-exports
├── tests/
│   └── MapSwiper.test.ts
├── demo/
│   └── index.html     # Interactive browser demo
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md          # This file
```
