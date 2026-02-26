# Raster Band Stats Reporter

<!-- PLACEHOLDER: replace YOUR_GITHUB_USERNAME with your actual GitHub username -->
[![CI — Python](https://github.com/YOUR_GITHUB_USERNAME/GeoScriptHub/actions/workflows/ci-python.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/GeoScriptHub/actions/workflows/ci-python.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../../../LICENSE)

> Compute **per-band statistics** (min, max, mean, std dev) for any GeoTIFF in one command — export to JSON or CSV.

<!-- PLACEHOLDER: replace with a demo GIF of the tool running -->
<!-- ![Demo](../../../../docs/assets/demo-raster-band-stats.gif) -->

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python API](#python-api)
- [Configuration Reference](#configuration-reference)
- [Output Formats](#output-formats)
- [Examples](#examples)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Nodata-aware** — pixels matching the raster's nodata value are excluded from statistics.
- **Band selection** — process all bands or specify a subset (e.g. `--bands 3,4` for red + NIR).
- **Two output formats** — JSON (structured, machine-readable) or CSV (open in Excel / QGIS).
- **Rich metadata** — output includes CRS, raster dimensions, and data type.
- **Fast** — uses numpy masked arrays; handles large rasters efficiently.

---

## Installation

```bash
cd tools/python/raster-band-stats

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

pip install -e .
geo-raster-stats --help
```

> Add the repo root to `PYTHONPATH`:
> ```bash
> # Windows: set PYTHONPATH=.
> # macOS / Linux: export PYTHONPATH=.
> ```

---

## Usage

### CLI

```bash
# All bands → JSON
geo-raster-stats --input data/landsat_scene.tif --output output/stats.json

# Specific bands → CSV
geo-raster-stats \
  --input  data/sentinel2.tif \
  --output output/stats.csv \
  --format csv \
  --bands  2,3,4,8

# Enable verbose logging
geo-raster-stats --input data/dem.tif --output stats.json --verbose
```

### Python API

```python
from pathlib import Path
from src.raster_band_stats.stats import BandStatsReporter, BandStatsConfig

config = BandStatsConfig(
    output_format="json",         # PLACEHOLDER: "json" or "csv"
    bands=[3, 4],                 # PLACEHOLDER: list of 1-based band indices, or None for all
)

tool = BandStatsReporter(
    input_path=Path("data/landsat.tif"),    # PLACEHOLDER: path to your raster
    output_path=Path("output/stats.json"),  # PLACEHOLDER: path for output file
    config=config,
)
tool.run()

# Inspect results
for band in tool.band_stats:
    print(band)
    # Band 3: min=0.0003 max=0.4521 mean=0.0934 std=0.0621 valid_px=7,921,024
```

---

## Configuration Reference

| Parameter | Type | Default | Description | Placeholder |
|-----------|------|---------|-------------|-------------|
| `--input` / `input_path` | `Path` | — | Path to the input raster file | **PLACEHOLDER** — path to your `.tif`, `.img`, or other rasterio-supported file |
| `--output` / `output_path` | `Path` | — | Path for the output stats file | **PLACEHOLDER** — `.json` for JSON, `.csv` for CSV |
| `--format` / `output_format` | `str` | `"json"` | Output format | **PLACEHOLDER** — `"json"` or `"csv"` |
| `--bands` / `bands` | `str` (CSV) | all | Band indices to process (1-based) | **PLACEHOLDER** — e.g. `1,4` for first and fourth bands<br>Landsat 8 Red=4, NIR=5; Sentinel-2 Red=4, NIR=8 |
| `--verbose` | `bool` | `False` | Debug logging | Pass `-v` for per-band debug output |

### Common Satellite Band Numbering

| Satellite | Blue | Green | Red | NIR | SWIR |
|-----------|------|-------|-----|-----|------|
| Landsat 8/9 | 2 | 3 | 4 | 5 | 6 |
| Sentinel-2 | 2 | 3 | 4 | 8 | 11 |

---

## Output Formats

### JSON
```json
{
  "source_file": "data/landsat_scene.tif",
  "crs": "EPSG:32614",
  "width": 7711,
  "height": 7861,
  "dtype": "uint16",
  "bands": [
    {
      "band_index": 1,
      "min": 7441.0,
      "max": 65535.0,
      "mean": 10234.8,
      "std_dev": 1827.3,
      "valid_pixels": 60617971,
      "nodata_pixels": 95029,
      "nodata_value": 0.0
    }
  ]
}
```

### CSV
```csv
band_index,min,max,mean,std_dev,valid_pixels,nodata_pixels,nodata_value
1,7441.0,65535.0,10234.8,1827.3,60617971,95029,0.0
2,7100.0,63200.0,9870.2,1650.1,60617971,95029,0.0
```

---

## Examples

```bash
# Landsat 8 scene — all 11 bands
geo-raster-stats --input LC08_band.tif --output stats.json

# PLACEHOLDER: replace LC08_band.tif with your actual Landsat file name
# Landsat 8 band file naming convention: LC08_L2SP_*_B1.TIF through B11.TIF

# Sentinel-2 — analyze only the 10m bands
geo-raster-stats --input S2A_MSI.tif --output stats.csv --format csv --bands 2,3,4,8
# PLACEHOLDER: replace S2A_MSI.tif with your actual Sentinel-2 filename
```

---

## Running Tests

```bash
export PYTHONPATH=../../../..  # macOS/Linux
set PYTHONPATH=../../../..     # Windows

pytest tests/ -v
```

---

## Contributing

See root [CONTRIBUTING.md](../../../../CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](../../../../LICENSE).
