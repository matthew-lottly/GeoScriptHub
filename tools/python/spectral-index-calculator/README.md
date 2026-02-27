# 🌿 Spectral Index Calculator

> Compute NDVI, NDWI, SAVI, and EVI from satellite band GeoTIFFs in one command.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![rasterio](https://img.shields.io/badge/rasterio-1.3%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![GeoScriptHub](https://img.shields.io/badge/GeoScriptHub-Tool%205%2F10-purple)](../../README.md)

---

## Table of Contents

- [Overview](#overview)
- [Supported Indices](#supported-indices)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Satellite Band Mapping](#satellite-band-mapping)
- [Output Format](#output-format)
- [Customization Guide](#customization-guide)
- [Running Tests](#running-tests)
- [Architecture](#architecture)

---

## Overview

The **Spectral Index Calculator** reads individual GeoTIFF band files from any satellite platform 
(Landsat 8/9, Sentinel-2, MODIS, etc.) and computes one or more spectral indices.  Each index is 
written as a single-band float32 GeoTIFF that preserves the original CRS, transform, and 
compression.

Key features:

- **Strategy pattern** — each index is an independent `IndexStrategy` subclass (add your own!)  
- **Batch mode** — compute multiple indices in one invocation  
- **CRS-preserving output** — output rasters inherit all projection metadata from the inputs  
- **nodata-safe** — zero-denominator pixels are set to −9999 rather than causing NaN across the array  
- **Full OOP** — inherits the `GeoTool` template-method pipeline for consistent error handling  

---

## Supported Indices

| Index | Full Name | Formula | Typical Range | Key Use |
|-------|-----------|---------|---------------|---------|
| **NDVI** | Normalized Difference Vegetation Index | `(NIR − Red) / (NIR + Red)` | −1 to +1 | Vegetation health & density |
| **NDWI** | Normalized Difference Water Index | `(Green − NIR) / (Green + NIR)` | −1 to +1 | Open water detection |
| **SAVI** | Soil-Adjusted Vegetation Index | `((NIR − Red) / (NIR + Red + L)) × (1 + L)` | ≈ −1 to +1 | Sparse vegetation on bare soil |
| **EVI** | Enhanced Vegetation Index | `2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1)` | −1 to +1 | High-biomass, corrected for atmosphere |

> **Add your own index** — see [Architecture](#architecture) for how to implement a custom `IndexStrategy`.

---

## Installation

```bash
# From the repo root (so shared/ is importable)
cd tools/python/spectral-index-calculator
pip install -e ".[dev]"
```

> **PYTHONPATH note:** the shared foundation at `shared/python/` must be on your path.  
> When running from the repo root this happens automatically; in CI it is set via  
> `env: PYTHONPATH: ${{ github.workspace }}`.

---

## Quick Start

```bash
# Compute NDVI from Landsat 8 bands
geo-spectral \
    --red  LC08_L2SP_B4.TIF \
    --nir  LC08_L2SP_B5.TIF \
    --index NDVI \
    --output-dir output/

# Compute NDVI + NDWI from Sentinel-2
geo-spectral \
    --red   T32UME_B04.tif \
    --nir   T32UME_B08.tif \
    --green T32UME_B03.tif \
    --index NDVI,NDWI \
    --output-dir results/

# Compute all four indices using Landsat 8
geo-spectral \
    --red   LC08_B4.TIF \
    --nir   LC08_B5.TIF \
    --green LC08_B3.TIF \
    --blue  LC08_B2.TIF \
    --index ALL \
    --output-dir /data/indices/
```

---

## CLI Reference

```
Usage: geo-spectral [OPTIONS]

  Compute spectral indices (NDVI, NDWI, SAVI, EVI) from satellite band
  GeoTIFFs. Each index is saved as a single-band float32 GeoTIFF.

Options:
  --red PATH         Red band file path (NDVI, SAVI, EVI)
  --nir PATH         NIR band file path (NDVI, NDWI, SAVI, EVI)
  --green PATH       Green band file path (NDWI)
  --blue PATH        Blue band file path (EVI)
  --index TEXT       Comma-separated indices or ALL  [default: NDVI]
  --savi-l FLOAT     SAVI soil brightness L factor   [default: 0.5]
  --output-dir PATH  Output directory               [default: output]
  --verbose          Enable DEBUG-level logging
  --help             Show this message and exit.
```

---

## Python API

```python
from pathlib import Path
from spectral_index_calculator.calculator import (
    SpectralIndexCalculator,
    BandFileMap,
    NDVIStrategy,
    NDWIStrategy,
    SAVIStrategy,
    EVIStrategy,
    ALL_STRATEGIES,
)

# --- Minimal: NDVI only ---
tool = SpectralIndexCalculator(
    band_files=BandFileMap(
        red=Path("data/LC08_B4.TIF"),    # Landsat 8 Red
        nir=Path("data/LC08_B5.TIF"),    # Landsat 8 NIR
    ),
    output_dir=Path("output/spectral"),
    strategies=[NDVIStrategy()],
)
tool.run()

# --- All indices ---
tool_all = SpectralIndexCalculator(
    band_files=BandFileMap(
        red=Path("data/LC08_B4.TIF"),
        nir=Path("data/LC08_B5.TIF"),
        green=Path("data/LC08_B3.TIF"),
        blue=Path("data/LC08_B2.TIF"),
    ),
    output_dir=Path("output/spectral"),
    strategies=ALL_STRATEGIES,
)
tool_all.run()

# Access results
for result in tool_all.results:
    print(result)
# NDVI: min=-0.1034 max=0.8923 mean=0.4512 → NDVI.tif
# NDWI: min=-0.7241 max=0.3817 mean=-0.1523 → NDWI.tif
# ...
```

### Custom strategy example

```python
from spectral_index_calculator.calculator import IndexStrategy
import numpy as np
import numpy.typing as npt

class NBR2Strategy(IndexStrategy):
    """NBR2 — Normalized Burn Ratio 2.
    
    Formula: (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    """

    @property
    def name(self) -> str:
        return "NBR2"

    @property
    def required_bands(self) -> list[str]:
        return ["swir1", "swir2"]

    def compute(self, bands: dict[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        swir1 = bands["swir1"].astype(np.float32)
        swir2 = bands["swir2"].astype(np.float32)
        denom = swir1 + swir2
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denom == 0, -9999.0, (swir1 - swir2) / denom).astype(np.float32)
```

---

## Satellite Band Mapping

### Landsat 8 / 9 (Collection 2, Level-2 Surface Reflectance)

| Band Key | Band Number | File suffix (example) |
|----------|-------------|----------------------|
| `blue`   | B2          | `*_B2.TIF`           |
| `green`  | B3          | `*_B3.TIF`           |
| `red`    | B4          | `*_B4.TIF`           |
| `nir`    | B5          | `*_B5.TIF`           |
| `swir1`  | B6          | `*_B6.TIF`           |
| `swir2`  | B7          | `*_B7.TIF`           |

### Sentinel-2 (Level-2A, 10 m & 20 m)

| Band Key | Band Number | Resolution | File suffix (example) |
|----------|-------------|------------|----------------------|
| `blue`   | B02         | 10 m       | `*_B02.tif`          |
| `green`  | B03         | 10 m       | `*_B03.tif`          |
| `red`    | B04         | 10 m       | `*_B04.tif`          |
| `nir`    | B08         | 10 m       | `*_B08.tif`          |
| `swir1`  | B11         | 20 m       | `*_B11.tif`          |
| `swir2`  | B12         | 20 m       | `*_B12.tif`          |

> **Note for Sentinel-2:** SWIR bands (B11, B12) are 20 m; you must resample them to 10 m before 
> passing them to this tool when mixing with 10 m bands.

---

## Output Format

Each computed index is saved as a **single-band float32 GeoTIFF** compressed with LZW:

```
output/
├── NDVI.tif
├── NDWI.tif
├── SAVI.tif
└── EVI.tif
```

- **Pixel values:** floating-point index values (e.g., −1 to +1 for normalized indices)  
- **nodata value:** −9999.0 (where denominator = 0 or where source was masked)  
- **CRS/Transform:** copied verbatim from the first input band  
- **Compression:** LZW (reduces file size ~50–70% vs uncompressed)

---

## Customization Guide

The table below maps each configurable setting to its location and describes valid values.

| Setting | File | Line ref | Description | Example values |
|-------------|------|----------|-------------|----------------|
| Red band path | `cli.py` comment / `calculator.py` docstring | `--red` | Path to red band GeoTIFF | `Path("LC08_B4.TIF")` |
| NIR band path | `cli.py` comment / `calculator.py` docstring | `--nir` | Path to NIR band GeoTIFF | `Path("B08.tif")` |
| Green band path | `cli.py` comment | `--green` | Path to green band (NDWI only) | `Path("LC08_B3.TIF")` |
| Blue band path | `cli.py` comment | `--blue` | Path to blue band (EVI only) | `Path("LC08_B2.TIF")` |
| `--index` value | CLI / `BandFileMap` | `--index` | Indices to compute | `"NDVI,NDWI"` or `"ALL"` |
| `--savi-l` value | `SAVIStrategy.__init__` | `soil_factor` | SAVI L factor (0.25–1.0) | `0.5` (default), `0.25` |
| `output_dir` | `SpectralIndexCalculator` | constructor | Where output GeoTIFFs are saved | `Path("results/spectral")` |

---

## Running Tests

```bash
# From the repo root
cd tools/python/spectral-index-calculator
PYTHONPATH=../../.. pytest tests/ -v --tb=short
```

Expected output (abbreviated):

```
tests/test_calculator.py::TestIndexStrategies::test_ndvi_typical_vegetation PASSED
tests/test_calculator.py::TestIndexStrategies::test_ndvi_negative_water PASSED
tests/test_calculator.py::TestIndexStrategies::test_ndvi_zero_denominator_returns_nodata PASSED
tests/test_calculator.py::TestIndexStrategies::test_ndwi_positive_water PASSED
tests/test_calculator.py::TestIndexStrategies::test_savi_default_l PASSED
tests/test_calculator.py::TestIndexStrategies::test_savi_custom_l PASSED
tests/test_calculator.py::TestIndexStrategies::test_evi_typical PASSED
tests/test_calculator.py::TestSpectralIndexCalculatorHappyPath::test_ndvi_output_file_created PASSED
...
20 passed in 1.42s
```

---

## Architecture

```
spectral-index-calculator/
├── src/
│   └── spectral_index_calculator/
│       ├── __init__.py          # Public API exports
│       ├── calculator.py        # GeoTool subclass + IndexStrategy hierarchy
│       └── cli.py               # Click CLI (geo-spectral)
├── tests/
│   └── test_calculator.py       # Pytest suite (synthetic rasters, no real imagery)
├── pyproject.toml               # hatchling build config + entry point
└── README.md                    # This file
```

### Class diagram (abridged)

```
GeoTool (ABC)
└── SpectralIndexCalculator
        │ uses list of
        ▼
IndexStrategy (ABC)
    ├── NDVIStrategy
    ├── NDWIStrategy
    ├── SAVIStrategy
    └── EVIStrategy
```

The **Strategy pattern** keeps each formula isolated — add a new index by subclassing `IndexStrategy` 
and registering it in `_STRATEGY_REGISTRY` in `cli.py`.  No changes to `SpectralIndexCalculator` 
are needed.
