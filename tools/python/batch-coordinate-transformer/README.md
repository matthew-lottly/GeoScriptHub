# Batch Coordinate Transformer

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../../LICENSE)

> Reproject coordinate pairs in a CSV file from **any source CRS** to **any target CRS** — in one command.

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

- **Any CRS → any CRS** — powered by [pyproj 3.x](https://pyproj4.github.io/pyproj/), supports all EPSG codes, PROJ strings, and WKT definitions.
- **Flexible column names** — configure which CSV columns hold X/Y values; defaults to `longitude` / `latitude`.
- **Two output formats** — CSV (same schema as input) or GeoJSON FeatureCollection.
- **Null-safe** — rows with missing or non-numeric coordinates are skipped and reported, not silently corrupted.
- **Full OOP** — `CoordinateTransformer` inherits from `GeoTool`; easily subclass it for custom pipelines.
- **Click CLI** — install once, run from any terminal with `geo-transform --help`.

---

## Installation

```bash
# From the GeoScriptHub repository root — Python 3.11+ required
cd tools/python/batch-coordinate-transformer

# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install the tool in editable mode
pip install -e .

# Verify the CLI is available
geo-transform --help
```

> **Note:** You must also add the repository root to `PYTHONPATH` so the shared
> base modules can be resolved:
>
> ```bash
> # Windows (from repo root):
> set PYTHONPATH=.
> # macOS / Linux:
> export PYTHONPATH=.
> ```

---

## Usage

### CLI

```bash
geo-transform \
  --input     data/survey_points.csv \
  --output    output/survey_wgs84.csv \
  --from-crs  EPSG:32614 \
  --to-crs    EPSG:4326 \
  --lon-col   easting \
  --lat-col   northing
```

Full option reference:

```
geo-transform --help

Options:
  -i, --input PATH         Path to the input CSV file.  [required]
  -o, --output PATH        Path for the output file.    [required]
  --from-crs TEXT          Source CRS (e.g. EPSG:32614).[required]
  --to-crs TEXT            Target CRS (e.g. EPSG:4326). [required]
  --lon-col TEXT           X/longitude/easting column. [default: longitude]
  --lat-col TEXT           Y/latitude/northing column.  [default: latitude]
  --format [csv|geojson]   Output format.               [default: csv]
  -v, --verbose            Enable debug logging.
  --help                   Show this message and exit.
```

### Python API

```python
from pathlib import Path
from src.batch_coord_transformer.transformer import CoordinateTransformer, TransformerConfig

config = TransformerConfig(
    from_crs="EPSG:32614",
    to_crs="EPSG:4326",
    lon_col="easting",
    lat_col="northing",
    output_format="geojson",
)

tool = CoordinateTransformer(
    input_path=Path("data/survey_points.csv"),
    output_path=Path("output/result.geojson"),
    config=config,
)
tool.run()

# Inspect the result
print(tool.result.summary())
```

---

## Configuration Reference

Every parameter you may need to change is listed below.

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `--input` / `input_path` | `Path` | — | Path to your input CSV file | `data/points.csv` |
| `--output` / `output_path` | `Path` | — | Path for the output file. Parent dirs are auto-created. | `output/points_wgs84.csv` |
| `--from-crs` / `from_crs` | `str` | — | EPSG code or PROJ/WKT string of the input CRS | `EPSG:32614` |
| `--to-crs` / `to_crs` | `str` | — | EPSG code or PROJ/WKT string of the desired output CRS | `EPSG:4326` |
| `--lon-col` / `lon_col` | `str` | `"longitude"` | Column name holding X / longitude / easting values | `"easting"` |
| `--lat-col` / `lat_col` | `str` | `"latitude"` | Column name holding Y / latitude / northing values | `"northing"` |
| `--format` / `output_format` | `"csv"` \| `"geojson"` | `"csv"` | Output file format | `"geojson"` |
| `--verbose` / `verbose` | `bool` | `False` | Print DEBUG-level log messages | `True` |

### Common CRS codes

| CRS | EPSG Code | When to use |
|-----|-----------|-------------|
| WGS84 (lat/lon) | `EPSG:4326` | Web maps, GPS data, most GeoJSON |
| Web Mercator | `EPSG:3857` | Tile-based web maps (Leaflet, Mapbox) |
| NAD83 | `EPSG:4269` | US federal datasets |
| UTM Zone 14N | `EPSG:32614` | South-central US (TX, OK, KS) |
| UTM Zone 10N | `EPSG:32610` | Pacific coast US |

> Find any CRS at [epsg.io](https://epsg.io).

---

## Output Formats

### CSV (default)
Same columns as input; coordinate columns are overwritten with reprojected values.

```csv
easting,northing,name
-96.8345,32.7799,A
-96.8254,32.7889,B
-96.8163,32.7979,C
```

### GeoJSON
A GeoJSON `FeatureCollection` where each row becomes a `Point` feature.  Non-coordinate columns become `properties`.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [-96.8345, 32.7799] },
      "properties": { "name": "A" }
    }
  ]
}
```

---

## Examples

### Reproject a US state-plane dataset to WGS84

```bash
geo-transform \
  --input   data/parcels_statePlane.csv \
  --output  output/parcels_wgs84.geojson \
  --from-crs EPSG:2276 \
  --to-crs   EPSG:4326 \
  --lon-col  x \
  --lat-col  y \
  --format   geojson
```

### Convert Web Mercator back to lat/lon

```bash
geo-transform \
  --input   data/clicks_webmercator.csv \
  --output  output/clicks_latlon.csv \
  --from-crs EPSG:3857 \
  --to-crs   EPSG:4326
```

---

## Running Tests

```bash
# From this tool's directory, with repo root on PYTHONPATH
export PYTHONPATH=../../../..   # macOS / Linux
set PYTHONPATH=../../../..      # Windows

pytest tests/ -v
```

---

## Contributing

See the root [CONTRIBUTING.md](../../../CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](../../../LICENSE).
