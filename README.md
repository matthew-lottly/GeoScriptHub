<div align="center">


<img src="docs/assets/banner.png" alt="GeoScriptHub Banner" width="900"/>

# GeoScriptHub

**A collection of open-source GIS tools built for analysts, developers, and the community.**

[![CI — Python](https://github.com/matthew-lottly/GeoScriptHub/actions/workflows/ci-python.yml/badge.svg)](https://github.com/matthew-lottly/GeoScriptHub/actions/workflows/ci-python.yml)
[![CI — TypeScript](https://github.com/matthew-lottly/GeoScriptHub/actions/workflows/ci-typescript.yml/badge.svg)](https://github.com/matthew-lottly/GeoScriptHub/actions/workflows/ci-typescript.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## Table of Contents

- [About](#about)
- [Tool Index](#tool-index)
  - [Python CLI Tools](#python-cli-tools)
  - [TypeScript Web Widgets](#typescript-web-widgets)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Python Tools — Quick Setup](#python-tools--quick-setup)
  - [TypeScript Widgets — Quick Setup](#typescript-widgets--quick-setup)
- [Shared Python Foundation](#shared-python-foundation)
- [Contributing](#contributing)
- [Customization Guide](#customization-guide)
- [License](#license)
- [Contact](#contact)

---

## About

**GeoScriptHub** is a monorepo of production-quality geospatial tools and interactive web widgets.  Every tool is written with:

- **Full OOP** — each Python tool inherits from a shared `GeoTool` abstract base class.
- **Clean architecture** — Template Method pattern, Strategy pattern, and a custom exception hierarchy keep code maintainable.
- **Google-style docstrings** on every class, method, and module.
- **Click-based CLIs** — all Python tools run from the command line with `--help` support.
- **TypeScript** web widgets built with Vite, typed end-to-end, and embeddable via `<script>` tag or npm import.

Whether you need to batch-reproject 10,000 coordinates, validate a shapefile before a data pipeline, or drop a before/after map swiper into a web page — there's a tool here for you.

---

## Tool Index

### Python CLI Tools

| # | Tool | Description | Docs |
|---|------|-------------|------|
| 1 | **Batch Coordinate Transformer** | Reproject CSV/GeoJSON coordinates between any two CRS using `pyproj` | [→ README](tools/python/batch-coordinate-transformer/README.md) |
| 2 | **Shapefile Health Checker** | Validate shapefiles & GeoJSON for null geometries, bad CRS, duplicates & more | [→ README](tools/python/shapefile-health-checker/README.md) |
| 3 | **Batch Geocoder** | Convert a CSV of addresses to GeoJSON points via Nominatim or Google | [→ README](tools/python/batch-geocoder/README.md) |
| 4 | **Raster Band Stats Reporter** | Per-band statistics (min/max/mean/std dev) for any GeoTIFF | [→ README](tools/python/raster-band-stats/README.md) |
| 5 | **Spectral Index Calculator** | Compute NDVI, NDWI, EVI, SAVI from Landsat 8/9 or Sentinel-2 bands | [→ README](tools/python/spectral-index-calculator/README.md) |
| 6 | **OSM Change Monitor** | Poll OpenStreetMap Overpass API for changes within a bounding box | [→ README](tools/python/osm-change-monitor/README.md) |

### TypeScript Web Widgets

| # | Widget | Description | Docs |
|---|--------|-------------|------|
| 7 | **Leaflet Widget Generator** | Paste GeoJSON → get a self-contained embeddable Leaflet HTML snippet | [→ README](tools/typescript/leaflet-widget-generator/README.md) |
| 8 | **Map Swiper** | Before/after swipe comparison of two map layers (MapLibre GL JS) | [→ README](tools/typescript/map-swiper/README.md) |
| 9 | **GeoJSON Diff Viewer** | Visual "git diff" for two GeoJSON files — added/removed/changed features | [→ README](tools/typescript/geojson-diff-viewer/README.md) |

---

## Repository Structure

```
GeoScriptHub/
├── .github/
│   ├── workflows/
│   │   ├── ci-python.yml        # Lint, type-check & test Python tools
│   │   └── ci-typescript.yml    # Build & lint TypeScript widgets
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
│
├── shared/
│   └── python/
│       ├── __init__.py          # Re-exports for clean imports
│       ├── base_tool.py         # GeoTool ABC — all Python tools inherit this
│       ├── exceptions.py        # Custom exception hierarchy
│       └── validators.py        # Shared precondition checks
│
├── tools/
│   ├── python/
│   │   ├── batch-coordinate-transformer/
│   │   ├── shapefile-health-checker/
│   │   ├── batch-geocoder/
│   │   ├── raster-band-stats/
│   │   ├── spectral-index-calculator/
│   │   └── osm-change-monitor/
│   └── typescript/
│       ├── leaflet-widget-generator/
│       ├── map-swiper/
│       └── geojson-diff-viewer/
│
├── docs/
│   └── assets/                  # Banner, screenshots, demo GIFs
│
├── .gitignore
├── LICENSE
└── README.md                    ← you are here
```

---

## Getting Started

### Python Tools — Quick Setup

Each Python tool is self-contained with its own `pyproject.toml`.  Install and run any tool individually:

```bash
# 1. Clone the repository
git clone https://github.com/matthew-lottly/GeoScriptHub.git
cd GeoScriptHub

# 2. Create a virtual environment (Python 3.11+ required)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install a specific tool (example: Batch Coordinate Transformer)
cd tools/python/batch-coordinate-transformer
pip install -e .

# 4. Run it
geo-transform --help
```

### TypeScript Widgets — Quick Setup

```bash
# Requires Node.js 18+ and pnpm (npm install -g pnpm)
cd tools/typescript/leaflet-widget-generator

# Install dependencies
pnpm install

# Start the local dev server
pnpm dev

# Build for production (outputs to dist/)
pnpm build
```

---

## Shared Python Foundation

All Python tools share a common foundation in `shared/python/`:

| Module | Purpose |
|--------|---------|
| `base_tool.py` | `GeoTool` abstract base class with Template Method `run()` pipeline |
| `exceptions.py` | `GeoScriptHubError` root + typed subclasses per failure domain |
| `validators.py` | Static precondition helpers (`assert_file_exists`, `assert_crs_valid`, etc.) |

To use the shared modules while developing a tool locally, add the repo root to `PYTHONPATH`:

```bash
# From the repo root
# Windows:
set PYTHONPATH=.
# macOS / Linux:
export PYTHONPATH=.
```

---

## Contributing

Contributions are welcome!  Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

1. **Fork** this repository.
2. **Create** a feature branch: `git checkout -b feature/my-new-tool`.
3. **Commit** your changes: `git commit -m "feat: add my new tool"`.
4. **Push** to your fork: `git push origin feature/my-new-tool`.
5. **Open** a Pull Request against `main`.

---

## Customization Guide

Each tool README includes a configuration reference table. Common values you may want to change:

| Setting | Where it appears | Notes |
|---------|-----------------|-------|
| `docs/assets/banner.png` | `<img>` tag at the top of this file | Replace with your real banner image |
| `docs/assets/demo-*.gif` | Each tool's README demo GIF | Screen-recorded GIF of the tool running |
| EPSG codes | Tool READMEs, code examples | Use the EPSG code for your data's CRS |
| Slack webhook URL | OSM Change Monitor | Your Slack incoming webhook URL |
| Google API key | Batch Geocoder | Your Google Maps Geocoding API key |

---

## License

Distributed under the MIT License.  See [LICENSE](LICENSE) for full text.

---

## Contact

**Author:** matthew-lottly  
**GitHub:** [github.com/matthew-lottly](https://github.com/matthew-lottly)

---

<div align="center">
  <sub>Built with Python 🐍 and TypeScript 🔷 — Made for the GIS community.</sub>
</div>
