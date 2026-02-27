# FGDB Archive Publisher

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![arcpy](https://img.shields.io/badge/arcpy-ArcGIS%20Pro%203.x-00897B)](https://pro.arcgis.com/en/pro-app/latest/arcpy/main/arcgis-pro-arcpy-reference.htm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../../LICENSE)
[![GeoScriptHub](https://img.shields.io/badge/GeoScriptHub-Tool%207%2F10-purple)](../../../README.md)

> Automate the full lifecycle of ArcGIS data — **backup** portal layers to a File Geodatabase, **clean** schemas, **validate** with topology rules, and **republish** to your portal — all in one command.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
  - [Top-Level Keys](#top-level-keys)
  - [field_cleanup](#field_cleanup)
  - [domains](#domains)
  - [topology_rules](#topology_rules)
  - [republish](#republish)
- [How It Works](#how-it-works)
  - [1. Archive](#1-archive)
  - [2. Schema Cleanup](#2-schema-cleanup)
  - [3. Topology Validation](#3-topology-validation)
  - [4. Republish](#4-republish)
- [Topology Rules Reference](#topology-rules-reference)
- [Example Config](#example-config)
- [Architecture](#architecture)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Batch export** — queries portal layers in configurable page sizes to handle datasets of any size without memory pressure.
- **Parallel attachment downloads** — uses a thread pool to download feature attachments concurrently.
- **Schema cleanup** — delete, rename, and add fields; create and assign coded-value or range domains.
- **Topology validation** — user-defined topology rules are applied and errors are exported for review.
- **Portal republish** — pushes the cleaned data back to ArcGIS Online or Enterprise as a hosted feature service.
- **JSON config** — the entire pipeline is driven by a single configuration file.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| ArcGIS Pro | 3.x (provides `arcpy` and `arcgis` Python API) |
| Python | 3.11+ (bundled with ArcGIS Pro) |
| Click | 8.1+ |

> **Note:** This tool must run in the **ArcGIS Pro Python environment** (`arcgispro-py3`) because it depends on `arcpy`, which is only available inside that environment.

---

## Installation

```bash
# 1. Open the ArcGIS Pro Python Command Prompt (or activate arcgispro-py3)
# 2. Navigate to the tool directory
cd GeoScriptHub/tools/python/fgdb-archive-publisher

# 3. Install in editable mode
pip install -e .

# 4. Verify
geo-archive --help
```

---

## Quick Start

```bash
# Create a config file (see example below), then run:
geo-archive --config my_config.json --output ./backups --verbose
```

The tool will:
1. Connect to your portal
2. Export each layer to a local `.gdb`
3. Clean up fields and apply domains
4. Validate topology rules
5. Republish to the target portal

---

## Configuration Reference

The pipeline is driven by a single JSON file.

### Top-Level Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `portal_url` | `string` | *required* | Source portal URL (e.g. `https://myorg.maps.arcgis.com`) |
| `service_urls` | `string[]` | *required* | Feature layer REST URLs to archive |
| `output_gdb_name` | `string` | `"archive.gdb"` | Name for the output File Geodatabase |
| `batch_size` | `integer` | `5000` | Features per query page (tune for large datasets) |
| `max_workers` | `integer` | `4` | Thread pool size for attachment downloads |
| `include_attachments` | `boolean` | `true` | Download feature attachments |
| `spatial_reference` | `integer` | `4326` | WKID for the output spatial reference |

### field_cleanup

| Key | Type | Description |
|-----|------|-------------|
| `delete_fields` | `string[]` | Field names to remove after archiving |
| `rename_fields` | `{old: new}` | Field rename mapping |
| `add_fields` | `FieldDefinition[]` | New fields to create (see below) |

**FieldDefinition:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | `string` | *required* | Field name |
| `type` | `string` | *required* | `TEXT`, `LONG`, `SHORT`, `DOUBLE`, `FLOAT`, `DATE` |
| `length` | `integer` | `255` | String length (TEXT fields only) |
| `alias` | `string` | `""` | Human-readable alias |
| `domain` | `string` | `""` | Domain name to assign |

### domains

Each entry creates a geodatabase domain.

| Key | Type | Description |
|-----|------|-------------|
| `name` | `string` | Unique domain name |
| `domain_type` | `string` | `"CODED"` or `"RANGE"` |
| `field_type` | `string` | ArcGIS field type (`TEXT`, `SHORT`, etc.) |
| `description` | `string` | Optional description |
| `values` | `object` | Coded: `{code: description}` — Range: `{min: n, max: n}` |

### topology_rules

Each entry adds a rule to the validation topology.

| Key | Type | Description |
|-----|------|-------------|
| `rule` | `string` | ArcGIS topology rule name (see reference below) |
| `feature_class` | `string` | Target feature class name |
| `subtype` | `string` | Optional subtype code |
| `covering_class` | `string` | For two-class rules — the covering feature class |
| `covering_subtype` | `string` | Subtype of the covering class |

### republish

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `target_portal` | `string` | `""` | Portal URL to publish to (leave empty to skip) |
| `folder` | `string` | `""` | Portal folder for the service |
| `service_name` | `string` | `""` | Service name (defaults to GDB name) |
| `overwrite` | `boolean` | `true` | Overwrite existing service |

---

## How It Works

### 1. Archive

The archiver connects to your portal and exports each feature layer into a local File Geodatabase:

- **Offset-based batching**: features are queried in pages of `batch_size` rows using `resultOffset` / `resultRecordCount` — this avoids memory pressure and portal query limits on large datasets (100k+ features).
- **Thread-pool attachments**: when `include_attachments` is `true`, a `ThreadPoolExecutor` with `max_workers` threads downloads all attachment blobs in parallel, organized into `<layer>_attachments/<OID>/` folders.
- The first batch creates the feature class schema via `JSONToFeatures`, subsequent batches append via `arcpy.da.InsertCursor` for maximum throughput.

### 2. Schema Cleanup

After the backup is created, the `SchemaManager` applies changes **in this order**:

1. **Create domains** — coded-value and range domains are added to the geodatabase.
2. **Delete fields** — unwanted columns (duplicates, temp fields, join artifacts) are removed. System fields (OID, Shape, GlobalID) are protected.
3. **Rename fields** — `arcpy.management.AlterField` applies cleaner names.
4. **Add fields** — new columns are created with type, alias, and optionally a domain assignment.

### 3. Topology Validation

The `TopologyChecker` creates a feature dataset, imports the relevant feature classes, builds a topology with all user-defined rules, and validates it:

1. Feature classes are copied into a `TopologyValidation` feature dataset.
2. Rules are added via `arcpy.management.AddRuleToTopology`.
3. `arcpy.management.ValidateTopology` runs the check.
4. Errors are exported to a `TopologyErrors` feature dataset with point, line, and polygon error feature classes.

### 4. Republish

If `republish.target_portal` is set, the cleaned FGDB is published:

- **Primary**: Service Definition workflow via `arcpy.sharing` (most reliable, supports overwrite).
- **Fallback**: ArcGIS API for Python — zips the FGDB and overwrites/creates a hosted feature service.

---

## Topology Rules Reference

Common rules you can use in your config:

| Rule | Geometry | Description |
|------|----------|-------------|
| `Must Not Overlap (Area)` | Polygon | No two features may share interior area |
| `Must Not Have Gaps (Area)` | Polygon | No voids between adjacent polygons |
| `Must Be Single Part (Area)` | Polygon | Multipart polygons are flagged |
| `Must Not Self-Overlap (Line)` | Line | Line features cannot overlap themselves |
| `Must Not Self-Intersect (Line)` | Line | Lines cannot cross themselves |
| `Must Not Have Dangles (Line)` | Line | Line endpoints must touch another line |
| `Must Not Intersect (Line)` | Line | No two lines may cross |
| `Must Be Covered By Feature Class Of (Area-Area)` | Polygon | Every feature must fall within a covering class |
| `Boundary Must Be Covered By (Area-Line)` | Mixed | Polygon boundaries must follow line features |
| `Must Not Overlap (Line)` | Line | No two lines may share segments |

See the [ArcGIS topology rules documentation](https://pro.arcgis.com/en/pro-app/latest/help/data/topologies/topology-rules.htm) for the full list.

---

## Example Config

```json
{
  "portal_url": "https://myorg.maps.arcgis.com",
  "service_urls": [
    "https://services.arcgis.com/abc123/arcgis/rest/services/Parcels/FeatureServer/0",
    "https://services.arcgis.com/abc123/arcgis/rest/services/Buildings/FeatureServer/0"
  ],
  "output_gdb_name": "city_backup_2026.gdb",
  "batch_size": 5000,
  "max_workers": 4,
  "include_attachments": true,
  "spatial_reference": 2263,

  "field_cleanup": {
    "delete_fields": ["GlobalID_1", "OBJECTID_1", "temp_flag"],
    "rename_fields": {
      "addr": "address",
      "zn": "zone_code"
    },
    "add_fields": [
      {
        "name": "STATUS",
        "type": "TEXT",
        "length": 50,
        "alias": "Status",
        "domain": "StatusDomain"
      },
      {
        "name": "PRIORITY",
        "type": "SHORT",
        "alias": "Priority Level"
      }
    ]
  },

  "domains": [
    {
      "name": "StatusDomain",
      "domain_type": "CODED",
      "field_type": "TEXT",
      "description": "Feature lifecycle status",
      "values": {
        "Active": "Active",
        "Inactive": "Inactive",
        "Pending": "Pending Review"
      }
    },
    {
      "name": "PriorityRange",
      "domain_type": "RANGE",
      "field_type": "SHORT",
      "description": "1-5 priority scale",
      "values": { "min": 1, "max": 5 }
    }
  ],

  "topology_rules": [
    {
      "rule": "Must Not Overlap (Area)",
      "feature_class": "Parcels"
    },
    {
      "rule": "Must Not Have Gaps (Area)",
      "feature_class": "Parcels"
    },
    {
      "rule": "Must Be Covered By Feature Class Of (Area-Area)",
      "feature_class": "Buildings",
      "covering_class": "Parcels"
    }
  ],

  "republish": {
    "target_portal": "https://myorg.maps.arcgis.com",
    "folder": "Published",
    "service_name": "Parcels_Clean",
    "overwrite": true
  }
}
```

---

## Architecture

```
fgdb-archive-publisher/
├── pyproject.toml
├── README.md
├── src/
│   └── fgdb_archive_publisher/
│       ├── __init__.py
│       ├── cli.py               # Click CLI — `geo-archive` command
│       ├── pipeline.py          # Orchestrator (GeoTool subclass) + config dataclasses
│       ├── archiver.py          # Batch export with thread-pool attachments
│       ├── schema_manager.py    # Field cleanup, domains, schema modifications
│       ├── topology_checker.py  # User-defined topology validation
│       └── publisher.py         # Republish to portal (SD + API fallback)
└── tests/
    └── test_pipeline.py         # Config parsing and validation tests
```

All modules use `arcpy` for geodatabase operations and follow the shared
`GeoTool` abstract base class pattern from `shared/python/base_tool.py`.

---

## Testing

```bash
# From the tool directory (inside ArcGIS Pro Python environment)
set PYTHONPATH=..\..\..
pytest tests/ -v --tb=short
```

> **Note:** The test suite validates config parsing and input validation only.
> Integration tests that exercise `arcpy` operations require an active ArcGIS Pro
> license and a portal connection.

---

## Contributing

See the root [CONTRIBUTING.md](../../../CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](../../../LICENSE).
