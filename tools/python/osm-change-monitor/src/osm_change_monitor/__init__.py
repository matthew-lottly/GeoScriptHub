"""
OSM Change Monitor
===================
Watch an OpenStreetMap bounding box for feature changes and send notifications.
"""

from osm_change_monitor.monitor import (
    BoundingBox,
    ChangeSet,
    EmailNotifier,
    JsonFileNotifier,
    NotifierBackend,
    OSMChangeMonitor,
    OSMFeatureSnapshot,
    OverpassClient,
    SlackNotifier,
)
from osm_change_monitor.scheduler import MonitorScheduler

__all__ = [
    "OSMChangeMonitor",
    "BoundingBox",
    "OSMFeatureSnapshot",
    "ChangeSet",
    "NotifierBackend",
    "JsonFileNotifier",
    "SlackNotifier",
    "EmailNotifier",
    "OverpassClient",
    "MonitorScheduler",
]
