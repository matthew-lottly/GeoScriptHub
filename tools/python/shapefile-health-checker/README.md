# Shapefile Health Checker

<!-- PLACEHOLDER: replace YOUR_GITHUB_USERNAME with your actual GitHub username -->
[![CI — Python](https://github.com/YOUR_GITHUB_USERNAME/GeoScriptHub/actions/workflows/ci-python.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/GeoScriptHub/actions/workflows/ci-python.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../../../LICENSE)

> Validate Shapefiles, GeoJSON, and GeoPackage files against **six configurable health checks** — output a Markdown or HTML report in seconds.

<!-- PLACEHOLDER: replace with a demo GIF of the tool running in a terminal -->
<!-- ![Demo](../../../../docs/assets/demo-shapefile-health-checker.gif) -->

---

## Table of Contents

- [Features](#features)
- [Checks Performed](#checks-performed)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python API](#python-api)
  - [Custom Check Suite](#custom-check-suite)
- [Configuration Reference](#configuration-reference)
- [Sample Report Output](#sample-report-output)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Six built-in checks** covering the most common spatial data quality issues.
- **Pluggable** — pass your own list of `CheckStrategy` subclasses to run custom checks.
- **Two report formats** — Markdown (great for GitHub PRs) or HTML (browser-viewable).
- **Detailed findings** — each failed check lists the exact row indices affected.
- **CLI & Python API** — use from the command line or embed in a larger pipeline.

---

## Checks Performed

| # | Check | Severity | Description |
|---|-------|----------|-------------|
| 1 | **CRS Presence** | FAILED | Confirms the dataset has a defined coordinate reference system |
| 2 | **Null / Empty Geometry** | FAILED | Detects features with `None` or empty geometry objects |
| 3 | **Self-Intersection** | FAILED | Uses Shapely's `is_valid` to flag topologically invalid geometries |
| 4 | **Duplicate Features** | WARNING | Detects rows that share identical WKT geometries |
| 5 | **Attribute Encoding** | WARNING | Verifies all string attribute values encode cleanly to UTF-8 |
| 6 | **Extent Sanity** | FAILED | Flags geometries outside world bounds (±180° lon, ±90° lat) |

---

## Installation

```bash
cd tools/python/shapefile-health-checker

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

pip install -e .
geo-check --help
```

> **Note:** Add the repo root to `PYTHONPATH`:
> ```bash
> # Windows: set PYTHONPATH=.
> # macOS / Linux: export PYTHONPATH=.
> ```

---

## Usage

### CLI

```bash
# Basic usage — outputs a Markdown report
geo-check --input data/parcels.shp --output output/report.md

# HTML report
geo-check --input data/roads.geojson --output report.html --format html

# Skip a specific check
geo-check --input data/parcels.shp --output report.md \
          --skip-check duplicate-features

# Multiple skips
geo-check --input data.gpkg --output report.md \
          --skip-check encoding \
          --skip-check extent-sanity \
          --verbose
```

### Python API

```python
from pathlib import Path
from src.shapefile_health_checker.checker import ShapefileHealthChecker

tool = ShapefileHealthChecker(
    input_path=Path("data/parcels.shp"),    # PLACEHOLDER: path to your vector file
    output_path=Path("output/report.md"),   # PLACEHOLDER: path for your report
    report_format="markdown",               # PLACEHOLDER: "markdown" or "html"
)
tool.run()

# Inspect results programmatically
report = tool.report
print(f"Overall: {report.overall_status.name}")
print(f"Failed checks: {report.failed_count}")

for result in report.results:
    print(f"  {result.status_label}  {result.check_name}: {result.details}")
```

### Custom Check Suite

```python
from src.shapefile_health_checker.checker import (
    DEFAULT_CHECKS,
    NullGeometryCheck,
    SelfIntersectionCheck,
    ShapefileHealthChecker,
)

# Run only the two most important checks
my_checks = [NullGeometryCheck(), SelfIntersectionCheck()]

ShapefileHealthChecker(
    input_path=Path("data/parcels.shp"),
    output_path=Path("output/report.md"),
    checks=my_checks,
).run()
```

---

## Configuration Reference

| Parameter | Type | Default | Description | Placeholder |
|-----------|------|---------|-------------|-------------|
| `--input` / `input_path` | `Path` | — | Path to the input vector file | **PLACEHOLDER** — set to your `.shp`, `.geojson`, or `.gpkg` file |
| `--output` / `output_path` | `Path` | — | Path for the output report | **PLACEHOLDER** — set to your desired output path |
| `--format` / `report_format` | `str` | `"markdown"` | Report format | **PLACEHOLDER** — `"markdown"` produces `.md`, `"html"` produces `.html` |
| `--skip-check` | `str` (repeatable) | none | Checks to skip by name slug | **PLACEHOLDER** — add check names you want to omit (see check list above) |
| `--verbose` | `bool` | `False` | Debug logging | Set to `True` / pass `-v` to see per-feature detail |

---

## Sample Report Output

```markdown
# Health Report — `parcels.shp`

**Overall status:** WARNING
**File:** `data/parcels.shp`
**CRS:** `EPSG:4326`
**Feature count:** 1,432

## Summary

| Check | Status | Details |
|---|---|---|
| CRS Presence | ✅ PASSED | CRS is defined: 4326 |
| Null / Empty Geometry | ✅ PASSED | All geometries are non-null. |
| Self-Intersection | ✅ PASSED | All geometries are topologically valid. |
| Duplicate Features | ⚠️  WARNING | 14 feature(s) share identical geometries... |
| Attribute Encoding | ✅ PASSED | All attribute values encode cleanly to UTF-8. |
| Extent Sanity | ✅ PASSED | All features fall within world bounds. |
```

---

## Running Tests

```bash
export PYTHONPATH=../../../..    # macOS/Linux
set PYTHONPATH=../../../..       # Windows

pytest tests/ -v
```

---

## Contributing

See root [CONTRIBUTING.md](../../../../CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](../../../../LICENSE).
