"""
OSM Change Monitor — CLI Entry Point
======================================
Exposes :class:`~osm_change_monitor.monitor.OSMChangeMonitor` as the
``geo-osm-monitor`` command with both one-shot and scheduled polling modes.

Usage::

    # Single poll (useful for cron jobs)
    geo-osm-monitor \\
        --south 51.47 --west -0.15 --north 51.52 --east -0.08 \\
        --tag "amenity=hospital" \\
        --output-dir data/monitor

    # Continuous polling every 30 minutes
    geo-osm-monitor \\
        --south 51.47 --west -0.15 --north 51.52 --east -0.08 \\
        --tag "amenity=hospital" \\
        --output-dir data/monitor \\
        --schedule 30

Run ``geo-osm-monitor --help`` for full option list.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from osm_change_monitor.monitor import (
    BoundingBox,
    EmailNotifier,
    JsonFileNotifier,
    OSMChangeMonitor,
    OverpassClient,
    SlackNotifier,
)
from osm_change_monitor.scheduler import MonitorScheduler

logger = logging.getLogger("geoscripthub.osm_change_monitor.cli")


@click.command("geo-osm-monitor")
# Bounding box
@click.option("--south", required=True, type=float,
              help="South latitude of the bounding box.")
# PLACEHOLDER: replace with your bounding box south latitude, e.g. 51.47
@click.option("--west", required=True, type=float,
              help="West longitude of the bounding box.")
# PLACEHOLDER: replace with your bounding box west longitude, e.g. -0.15
@click.option("--north", required=True, type=float,
              help="North latitude of the bounding box.")
# PLACEHOLDER: replace with your bounding box north latitude, e.g. 51.52
@click.option("--east", required=True, type=float,
              help="East longitude of the bounding box.")
# PLACEHOLDER: replace with your bounding box east longitude, e.g. -0.08
# OSM tag
@click.option(
    "--tag", "osm_tag", required=True,
    # PLACEHOLDER: OSM "key=value" tag to monitor, e.g. "amenity=hospital"
    help='OSM tag to monitor, in "key=value" format (e.g. amenity=hospital).',
)
# Output
@click.option(
    "--output-dir", default="osm-monitor-data", show_default=True,
    # PLACEHOLDER: Directory for snapshot and change log files
    help="Directory for snapshot JSON and changes.jsonl.",
)
# Scheduler
@click.option(
    "--schedule", "interval_minutes", default=None, type=int,
    # PLACEHOLDER: Poll interval in minutes; omit for a single one-shot run
    help="Poll interval (minutes). Omit for a single one-shot run.",
)
# Notifier overrides
@click.option(
    "--slack-webhook", default=None, envvar="OSM_SLACK_WEBHOOK",
    # PLACEHOLDER: Set via env var OSM_SLACK_WEBHOOK or pass directly
    help="Slack Incoming Webhook URL (or set OSM_SLACK_WEBHOOK env var).",
)
@click.option(
    "--slack-min-changes", default=1, show_default=True, type=int,
    # PLACEHOLDER: Minimum changed features before posting to Slack
    help="Minimum number of changes before triggering Slack notification.",
)
@click.option(
    "--overpass-url", default=OverpassClient.DEFAULT_API_URL, show_default=True,
    # PLACEHOLDER: Override if you run a private Overpass instance
    help="Overpass API URL.",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def cli(
    south: float,
    west: float,
    north: float,
    east: float,
    osm_tag: str,
    output_dir: str,
    interval_minutes: int | None,
    slack_webhook: str | None,
    slack_min_changes: int,
    overpass_url: str,
    verbose: bool,
) -> None:
    """Monitor an OSM bounding box for feature changes and send notifications.

    \b
    Examples:
        # One-shot poll for hospitals in central London
        geo-osm-monitor \\
            --south 51.47 --west -0.15 --north 51.52 --east -0.08 \\
            --tag "amenity=hospital" --output-dir data/monitor

        # Schedule polling every 60 minutes with Slack alerts
        OSM_SLACK_WEBHOOK=https://hooks.slack.com/... \\
        geo-osm-monitor \\
            --south 51.47 --west -0.15 --north 51.52 --east -0.08 \\
            --tag "amenity=hospital" --schedule 60 --output-dir data/monitor
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        bbox = BoundingBox(south=south, west=west, north=north, east=east)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Build notifier list
    notifiers = [JsonFileNotifier(Path(output_dir) / "changes.jsonl")]
    if slack_webhook:
        notifiers.append(SlackNotifier(webhook_url=slack_webhook, min_changes=slack_min_changes))

    overpass_client = OverpassClient(api_url=overpass_url)

    monitor = OSMChangeMonitor(
        bbox=bbox,
        osm_tag=osm_tag,
        output_dir=Path(output_dir),
        notifiers=notifiers,
        overpass_client=overpass_client,
        verbose=verbose,
    )

    if interval_minutes is not None:
        # Continuous mode
        MonitorScheduler(monitor, interval_minutes=interval_minutes).start()
    else:
        # One-shot mode
        try:
            monitor.run()
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

        cs = monitor.last_change_set
        if cs is None:
            click.echo("First run: baseline snapshot saved. Re-run to detect changes.")
        else:
            click.echo(cs.summary())
            if not cs.has_changes:
                click.echo("No changes detected.")


if __name__ == "__main__":
    cli()
