# 🗺️ OSM Change Monitor

> Watch any OpenStreetMap bounding box for added or removed features and get notified via Slack, email, or a JSON log.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![overpy](https://img.shields.io/badge/overpy-0.7%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![GeoScriptHub](https://img.shields.io/badge/GeoScriptHub-Tool%206%2F10-purple)](../../README.md)

---

## Table of Contents

- [Overview](#overview)
- [Use Cases](#use-cases)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Notification Backends](#notification-backends)
- [Scheduling](#scheduling)
- [State & Persistence](#state--persistence)
- [Customization Guide](#customization-guide)
- [Running Tests](#running-tests)
- [Architecture](#architecture)

---

## Overview

The **OSM Change Monitor** polls the [Overpass API](https://overpass-api.de/) on a configurable 
schedule to detect when OpenStreetMap features matching a tag (e.g. `amenity=hospital`) are 
**added** or **removed** from a bounding box.

On each run the tool:

1. Queries Overpass for the current set of matching features.
2. Diffs the result against the previously saved snapshot.
3. Notifies all registered backends if changes are detected.
4. Saves the new snapshot for the next run.

Key features:

- **Strategy pattern** for notifiers — plug in Slack, email, JSON file, or your own backend  
- **First-run safe** — creates a baseline snapshot on first run without false-positive alerts  
- **Retry logic** — Overpass queries are retried up to 3× on transient failures  
- **Scheduler** — built-in `schedule`-based polling loop, or use any external cron/scheduler  
- **Full OOP** — inherits `GeoTool` template-method pipeline  

---

## Use Cases

| Scenario | Tag | Alert when |
|----------|-----|------------|
| Hospital availability tracking | `amenity=hospital` | New clinic opened, existing one closes |
| Café / restaurant map freshness | `amenity=cafe` | New venues appear, old ones removed |
| Traffic signal monitoring | `highway=traffic_signals` | Signal added/removed from junction |
| Park boundary changes | `leisure=park` | New parks added, existing ones removed |
| Construction permits (informal) | `landuse=construction` | Sites opened or cleared |

---

## Installation

```bash
# From the repo root (so shared/ is importable)
cd tools/python/osm-change-monitor
pip install -e ".[dev]"
```

---

## Quick Start

```bash
# One-shot poll for hospitals in central London
geo-osm-monitor \
    --south 51.47 --west -0.15 --north 51.52 --east -0.08 \
    --tag "amenity=hospital" \
    --output-dir data/monitor

# Schedule polling every 60 minutes (Ctrl-C to stop)
geo-osm-monitor \
    --south 51.47 --west -0.15 --north 51.52 --east -0.08 \
    --tag "amenity=hospital" \
    --output-dir data/monitor \
    --schedule 60

# Add Slack notifications via env var
export OSM_SLACK_WEBHOOK="https://hooks.slack.com/services/T00/B00/XXXXXX"
geo-osm-monitor \
    --south 51.47 --west -0.15 --north 51.52 --east -0.08 \
    --tag "amenity=hospital" \
    --schedule 60
```

> **First run:** The tool saves a baseline snapshot and prints a message.  
> **Second run:** The tool diffs against the baseline and reports changes.

---

## CLI Reference

```
Usage: geo-osm-monitor [OPTIONS]

  Monitor an OSM bounding box for feature changes and send notifications.

Options:
  --south FLOAT          South latitude of bounding box      [required]
  --west FLOAT           West longitude of bounding box      [required]
  --north FLOAT          North latitude of bounding box      [required]
  --east FLOAT           East longitude of bounding box      [required]
  --tag TEXT             OSM tag in "key=value" format        [required]
  --output-dir PATH      State & log directory   [default: osm-monitor-data]
  --schedule INTEGER     Poll interval (minutes). Omit for one-shot run.
  --slack-webhook TEXT   Slack Incoming Webhook URL (or OSM_SLACK_WEBHOOK)
  --slack-min-changes N  Min changes before Slack alert       [default: 1]
  --overpass-url TEXT    Overpass API URL
  --verbose              Enable DEBUG logging
  --help                 Show this message and exit.
```

---

## Python API

```python
from pathlib import Path
from osm_change_monitor.monitor import (
    OSMChangeMonitor,
    BoundingBox,
    JsonFileNotifier,
    SlackNotifier,
    EmailNotifier,
)
from osm_change_monitor.scheduler import MonitorScheduler

# --- Configure bounding box ---
bbox = BoundingBox(
    south=51.47,
    west=-0.15,
    north=51.52,
    east=-0.08,
)

# --- Configure notifiers ---
notifiers = [
    JsonFileNotifier(Path("data/monitor/changes.jsonl")),
    SlackNotifier(
        webhook_url="https://hooks.slack.com/services/T00/B00/XXXXXX",
        min_changes=1,
    ),
]

# --- Create and run monitor ---
monitor = OSMChangeMonitor(
    bbox=bbox,
    osm_tag="amenity=hospital",
    output_dir=Path("data/monitor"),
    notifiers=notifiers,
)

# One-shot poll
monitor.run()

# Or schedule continuous polling every 30 minutes
MonitorScheduler(monitor, interval_minutes=30).start()
```

---

## Notification Backends

### JsonFileNotifier (default)

Appends each `ChangeSet` as a JSON line to a `.jsonl` file — always enabled.

```jsonl
{"osm_tag": "amenity=hospital", "bbox": {...}, "polled_at": "2025-01-15T12:30:00+00:00", "added": [...], "removed": []}
{"osm_tag": "amenity=hospital", "bbox": {...}, "polled_at": "2025-01-15T13:30:00+00:00", "added": [], "removed": [...]}
```

### SlackNotifier

Posts a formatted Slack message via an [Incoming Webhook](https://api.slack.com/messaging/webhooks).

```
🗺️ *OSM Change Alert*
[2025-01-15 12:30 UTC] amenity=hospital in 51.47,-0.15,51.52,-0.08: +1 added, -0 removed

*Added (1):*
  • King's College Hospital (way)
```

### EmailNotifier

Sends an HTML email via SMTP. Configure via the Python API (see `EmailNotifier` docstring).

### Custom backend

```python
from osm_change_monitor.monitor import NotifierBackend, ChangeSet

class WebhookNotifier(NotifierBackend):
    def __init__(self, url: str) -> None:
        self.url = url

    def send(self, change_set: ChangeSet) -> None:
        import requests
        requests.post(self.url, json=change_set.to_dict(), timeout=10)
```

---

## Scheduling

### Built-in scheduler

```python
MonitorScheduler(monitor, interval_minutes=60).start()
```

Runs forever, intercepting SIGINT/SIGTERM for graceful shutdown.

### Cron (recommended for production)

```cron
# Poll every hour at :00
0 * * * * cd /path/to/repo && PYTHONPATH=. python -m osm_change_monitor.cli \
    --south 51.47 --west -0.15 --north 51.52 --east -0.08 \
    --tag "amenity=hospital" --output-dir /data/monitor >> /var/log/osm-monitor.log 2>&1
```

### Systemd service

```ini
[Unit]
Description=OSM Change Monitor
After=network.target

[Service]
WorkingDirectory=/path/to/GeoScriptHub
ExecStart=/path/to/venv/bin/geo-osm-monitor \
    --south 51.47 --west -0.15 --north 51.52 --east -0.08 \
    --tag "amenity=hospital" --output-dir /data/monitor --schedule 60
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## State & Persistence

The tool stores its state in `output_dir/`:

```
osm-monitor-data/
├── latest_snapshot.json   ← Current known state of features
└── changes.jsonl          ← Append-only change log (one JSON object per line)
```

`latest_snapshot.json` is **overwritten on every run**.  `changes.jsonl` is **append-only**.

---

## Customization Guide

| Setting | File | Description | Example values |
|-------------|------|-------------|----------------|
| Bounding box `south/west/north/east` | `cli.py` / `monitor.py` docstring | Area of interest | `51.47, -0.15, 51.52, -0.08` |
| `osm_tag` | `monitor.py` / `cli.py` | OSM feature tag to watch | `"amenity=hospital"`, `"shop=supermarket"` |
| `output_dir` | `monitor.py` / `cli.py` | State & log directory | `Path("data/osm-monitor")` |
| Slack webhook URL | `SlackNotifier` / `OSM_SLACK_WEBHOOK` env | Slack Incoming Webhook | `"https://hooks.slack.com/..."` |
| `min_changes` | `SlackNotifier` | Minimum changes before Slack post | `1` (default), `5` to reduce noise |
| SMTP settings | `EmailNotifier` | All email delivery settings | See `EmailNotifier` docstring |
| Overpass API URL | `OverpassClient.DEFAULT_API_URL` | Overpass endpoint | `"https://overpass.kumi.systems/api/interpreter"` |
| `interval_minutes` | `MonitorScheduler` | Poll frequency in minutes | `60` (hourly), `30` (twice hourly) |

---

## Running Tests

```bash
cd tools/python/osm-change-monitor
PYTHONPATH=../../.. pytest tests/ -v --tb=short
```

All Overpass API calls are mocked — no internet connection required.

```
tests/test_monitor.py::TestBoundingBox::test_valid_bbox PASSED
tests/test_monitor.py::TestBoundingBox::test_south_north_inverted_raises PASSED
tests/test_monitor.py::TestOSMFeatureSnapshot::test_to_dict_round_trip PASSED
...
20 passed in 0.41s
```

---

## Architecture

```
osm-change-monitor/
├── src/
│   └── osm_change_monitor/
│       ├── __init__.py      # Public API exports
│       ├── monitor.py       # GeoTool subclass + NotifierBackend hierarchy
│       ├── scheduler.py     # schedule-based polling loop
│       └── cli.py           # Click CLI (geo-osm-monitor)
├── tests/
│   └── test_monitor.py      # Pytest suite (mocked Overpass)
├── pyproject.toml           # hatchling build config + entry point
└── README.md                # This file
```

### Class diagram (abridged)

```
GeoTool (ABC)
└── OSMChangeMonitor
        │ uses list of
        ▼
NotifierBackend (ABC)
    ├── JsonFileNotifier
    ├── SlackNotifier
    └── EmailNotifier

MonitorScheduler
    └── wraps OSMChangeMonitor

OverpassClient
    └── wraps overpy.API
```
