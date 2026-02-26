# Batch Geocoder

<!-- PLACEHOLDER: replace YOUR_GITHUB_USERNAME with your actual GitHub username -->
[![CI — Python](https://github.com/YOUR_GITHUB_USERNAME/GeoScriptHub/actions/workflows/ci-python.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/GeoScriptHub/actions/workflows/ci-python.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../../../LICENSE)

> Convert a CSV of addresses to a **GeoJSON FeatureCollection** — free with Nominatim, or fast with Google Maps.

<!-- PLACEHOLDER: replace with a demo GIF -->
<!-- ![Demo](../../../../docs/assets/demo-batch-geocoder.gif) -->

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python API](#python-api)
- [Configuration Reference](#configuration-reference)
- [Backends](#backends)
- [Output Format](#output-format)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Two backends** — Nominatim (free, no API key) and Google Maps (fast, paid).
- **Pluggable** — implement `GeocoderBackend` to add any geocoding provider.
- **Null-safe output** — failed rows are included with `null` geometry so no data is lost.
- **Confidence scores** — Nominatim returns an `importance` score per result.
- **Extra columns passthrough** — carry any CSV columns through to GeoJSON properties.
- **Rate-limit aware** — configurable delay between requests; respects Nominatim's 1 req/s policy.

---

## Installation

```bash
cd tools/python/batch-geocoder

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

pip install -e .
geo-geocode --help
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
# Nominatim (free, default)
geo-geocode \
  --input      data/addresses.csv \
  --output     output/addresses.geojson \
  --address-col full_address \
  --user-agent "my-project/1.0" \
  --extra-cols name,city,zip

# Google Maps
geo-geocode \
  --input      data/addresses.csv \
  --output     output/addresses.geojson \
  --address-col address \
  --backend    google \
  --google-api-key YOUR_GOOGLE_API_KEY \
  --rate-limit 0.05
```

### Python API

```python
from pathlib import Path
from src.batch_geocoder.geocoder import BatchGeocoder, NominatimBackend

tool = BatchGeocoder(
    input_path=Path("data/customers.csv"),         # PLACEHOLDER: your CSV path
    output_path=Path("output/customers.geojson"),  # PLACEHOLDER: your output path
    address_col="full_address",                    # PLACEHOLDER: your address column name
    backend=NominatimBackend(
        user_agent="my-company/1.0",               # PLACEHOLDER: your app name
        rate_limit_seconds=1.1,                    # PLACEHOLDER: >= 1.0 for Nominatim
    ),
    extra_cols=["name", "city", "zip"],            # PLACEHOLDER: columns to include in output
)
tool.run()

# Inspect results
for result in tool.results:
    print(f"{result.address} → success={result.success}, confidence={result.confidence}")
```

---

## Configuration Reference

| Parameter | Type | Default | Description | Placeholder |
|-----------|------|---------|-------------|-------------|
| `--input` / `input_path` | `Path` | — | Path to the input CSV file | **PLACEHOLDER** — your CSV file path |
| `--output` / `output_path` | `Path` | — | Path for the output GeoJSON file | **PLACEHOLDER** — desired output path |
| `--address-col` / `address_col` | `str` | `"address"` | Column containing address strings | **PLACEHOLDER** — your address column name |
| `--backend` | `"nominatim"` \| `"google"` | `"nominatim"` | Geocoding provider | **PLACEHOLDER** — choose provider |
| `--user-agent` | `str` | `"geoscripthub-geocoder/1.0"` | App identifier for Nominatim | **PLACEHOLDER** — set to your app name |
| `--google-api-key` | `str` | env var | Google Maps API key | **PLACEHOLDER** — `YOUR_GOOGLE_API_KEY` or set `GOOGLE_MAPS_API_KEY` env var |
| `--rate-limit` | `float` | `1.1` | Seconds between requests | **PLACEHOLDER** — `>= 1.0` for Nominatim; `~0.05` for Google |
| `--extra-cols` | `str` (CSV) | `""` | Extra columns to carry into GeoJSON | **PLACEHOLDER** — comma-separated column names, e.g. `name,city,zip` |
| `--verbose` | `bool` | `False` | Debug logging | Pass `-v` to see per-address results |

---

## Backends

### Nominatim (default, free)
- Powered by OpenStreetMap data.
- No API key required.
- **Must** respect the [Nominatim Usage Policy](https://operations.osmfoundation.org/policies/nominatim/): keep `rate_limit_seconds >= 1.0` and set a descriptive `user_agent`.

### Google Maps Geocoding API
- Requires a Google Cloud project with the Geocoding API enabled.
- Generate an API key at [console.cloud.google.com](https://console.cloud.google.com/) → APIs & Services → Credentials.
- **PLACEHOLDER**: Store the key in an environment variable (`GOOGLE_MAPS_API_KEY`) — never commit it to version control.

### Custom Backend

```python
from src.batch_geocoder.geocoder import GeocoderBackend, GeocodeResult

class MyCustomBackend(GeocoderBackend):
    def geocode_one(self, address: str) -> GeocodeResult:
        # Your implementation here
        ...
```

---

## Output Format

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [-77.036, 38.897] },
      "properties": {
        "address": "1600 Pennsylvania Ave NW, Washington DC",
        "display_name": "White House, Washington, DC, USA",
        "confidence": 0.901,
        "geocode_success": true,
        "name": "White House"
      }
    },
    {
      "type": "Feature",
      "geometry": null,
      "properties": {
        "address": "zzz bad address",
        "geocode_success": false
      }
    }
  ]
}
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
