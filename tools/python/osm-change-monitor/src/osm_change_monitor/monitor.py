"""
OSM Change Monitor — Core Module
==================================
Polls the OpenStreetMap Overpass API for features inside a bounding box and
notifies configured backends when features are added or removed between runs.

Design:
    * :class:`OSMFeatureSnapshot` — immutable record of a queried feature.
    * :class:`ChangeSet` — diff of added/removed features since the last poll.
    * :class:`NotifierBackend` (ABC) — sends a :class:`ChangeSet` somewhere.
      Concrete implementations: :class:`SlackNotifier`, :class:`EmailNotifier`,
      :class:`JsonFileNotifier`.
    * :class:`OverpassClient` — thin wrapper around `overpy` with retry logic.
    * :class:`OSMChangeMonitor` (:class:`~shared.python.GeoTool`) — orchestrates
      the Overpass query, diff, notification, and state persistence.
    * :class:`MonitorScheduler` — thin wrapper using the ``schedule`` library.

Typical workflow::

    monitor = OSMChangeMonitor(
        bbox=BoundingBox(south=51.47, west=-0.15, north=51.52, east=-0.08),
        osm_tag="amenity=hospital",
        output_dir=Path("monitor-state"),
        notifiers=[JsonFileNotifier(Path("changes.json"))],
    )
    monitor.run()                    # single poll
    # or: MonitorScheduler(monitor, interval_minutes=60).start()
"""

from __future__ import annotations

import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import requests

try:
    import overpy  # type: ignore[import-untyped]
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "overpy is required: pip install overpy"
    ) from exc

from shared.python.base_tool import GeoTool
from shared.python.exceptions import InputValidationError, OutputWriteError
from shared.python.validators import Validators

logger = logging.getLogger("geoscripthub.osm_change_monitor")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box for an Overpass query.

    Args:
        south: Southern latitude boundary.
               <!-- PLACEHOLDER: Replace with your area's south latitude, e.g. 51.47 -->
        west:  Western longitude boundary.
               <!-- PLACEHOLDER: Replace with your area's west longitude, e.g. -0.15 -->
        north: Northern latitude boundary.
               <!-- PLACEHOLDER: Replace with your area's north latitude, e.g. 51.52 -->
        east:  Eastern longitude boundary.
               <!-- PLACEHOLDER: Replace with your area's east longitude, e.g. -0.08 -->
    """

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not (-90 <= self.south < self.north <= 90):
            raise InputValidationError(
                f"Invalid latitude range: south={self.south}, north={self.north}"
            )
        if not (-180 <= self.west < self.east <= 180):
            raise InputValidationError(
                f"Invalid longitude range: west={self.west}, east={self.east}"
            )

    def to_overpass_str(self) -> str:
        """Return Overpass API bbox string: ``south,west,north,east``."""
        return f"{self.south},{self.west},{self.north},{self.east}"


@dataclass(frozen=True)
class OSMFeatureSnapshot:
    """Immutable snapshot of a single OSM feature returned by Overpass.

    Attributes:
        feature_id: OSM node/way/relation identifier.
        feature_type: One of ``"node"``, ``"way"``, ``"relation"``.
        tags: Dict of OSM tags (e.g. ``{"amenity": "hospital", "name": "..."}``)
        lat: Latitude (centre for ways/relations, exact for nodes).
        lon: Longitude (centre for ways/relations, exact for nodes).
    """

    feature_id: int
    feature_type: str
    tags: dict[str, str]
    lat: float | None
    lon: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON persistence."""
        return {
            "id": self.feature_id,
            "type": self.feature_type,
            "tags": self.tags,
            "lat": self.lat,
            "lon": self.lon,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OSMFeatureSnapshot":
        """Deserialise from a plain dict (inverse of :meth:`to_dict`)."""
        return cls(
            feature_id=int(d["id"]),
            feature_type=str(d["type"]),
            tags=dict(d.get("tags", {})),
            lat=d.get("lat"),
            lon=d.get("lon"),
        )


@dataclass
class ChangeSet:
    """Diff between two consecutive feature snapshots.

    Attributes:
        osm_tag: The monitored OSM "key=value" tag string.
        bbox: The bounding box used for the query.
        polled_at: UTC timestamp of the poll that produced this diff.
        added: Features present in the new snapshot but not the previous one.
        removed: Features present in the previous snapshot but not the new one.
    """

    osm_tag: str
    bbox: BoundingBox
    polled_at: datetime
    added: list[OSMFeatureSnapshot] = field(default_factory=list)
    removed: list[OSMFeatureSnapshot] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """``True`` when at least one feature was added or removed."""
        return bool(self.added or self.removed)

    def summary(self) -> str:
        """Human-readable one-line description of the change set."""
        return (
            f"[{self.polled_at.strftime('%Y-%m-%d %H:%M UTC')}] "
            f"{self.osm_tag} in {self.bbox.to_overpass_str()}: "
            f"+{len(self.added)} added, -{len(self.removed)} removed"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serialisable dict."""
        return {
            "osm_tag": self.osm_tag,
            "bbox": {
                "south": self.bbox.south,
                "west": self.bbox.west,
                "north": self.bbox.north,
                "east": self.bbox.east,
            },
            "polled_at": self.polled_at.isoformat(),
            "added": [f.to_dict() for f in self.added],
            "removed": [f.to_dict() for f in self.removed],
        }


# ---------------------------------------------------------------------------
# Notifier ABC + concrete backends
# ---------------------------------------------------------------------------


class NotifierBackend(ABC):
    """Abstract base for all notification delivery backends.

    Subclasses implement :meth:`send` to deliver a :class:`ChangeSet` to a
    specific destination (Slack, email, file, etc.).
    """

    @abstractmethod
    def send(self, change_set: ChangeSet) -> None:
        """Deliver the change set notification.

        Args:
            change_set: The diff to report.  Only called when
                        ``change_set.has_changes`` is ``True``.
        """


class JsonFileNotifier(NotifierBackend):
    """Append each :class:`ChangeSet` as a JSON line to a file.

    Args:
        output_file: Path to the JSONL output file.
                     <!-- PLACEHOLDER: Set to your desired log path,
                          e.g. Path("logs/osm_changes.jsonl") -->
    """

    def __init__(self, output_file: Path) -> None:
        # PLACEHOLDER: change this path to your preferred output location
        self.output_file = Path(output_file)

    def send(self, change_set: ChangeSet) -> None:
        """Append the change set as a JSON line to :attr:`output_file`.

        Args:
            change_set: ChangeSet to write.

        Raises:
            OutputWriteError: If the file cannot be written.
        """
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.output_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(change_set.to_dict()) + "\n")
        except OSError as exc:
            raise OutputWriteError(str(self.output_file), str(exc)) from exc
        logger.info("JsonFileNotifier: wrote change to %s", self.output_file)


class SlackNotifier(NotifierBackend):
    """Post a change summary to a Slack Incoming Webhook URL.

    Args:
        webhook_url: Slack Incoming Webhook URL.
                     <!-- PLACEHOLDER: Create a webhook at
                          https://api.slack.com/messaging/webhooks and paste
                          the URL here, e.g.
                          "https://hooks.slack.com/services/T00/B00/XXXXXX" -->
        min_changes: Minimum number of changed features before posting (default 1).
                     <!-- PLACEHOLDER: Set higher to reduce noise, e.g. 5 -->
    """

    def __init__(
        self,
        webhook_url: str,
        min_changes: int = 1,
    ) -> None:
        # PLACEHOLDER: replace with your actual Slack Incoming Webhook URL
        self.webhook_url = webhook_url
        self.min_changes = min_changes

    def send(self, change_set: ChangeSet) -> None:
        """POST a formatted Slack message to the webhook.

        Args:
            change_set: ChangeSet to report.

        Raises:
            requests.HTTPError: If the Slack API returns a non-2xx response.
        """
        total = len(change_set.added) + len(change_set.removed)
        if total < self.min_changes:
            logger.debug("SlackNotifier: skipped (total changes %d < %d)", total, self.min_changes)
            return

        text = (
            f":world_map: *OSM Change Alert*\n"
            f"{change_set.summary()}\n"
        )
        if change_set.added:
            text += f"\n*Added ({len(change_set.added)}):*\n"
            for feat in change_set.added[:5]:  # cap at 5 for readability
                name = feat.tags.get("name", f"id:{feat.feature_id}")
                text += f"  • `{name}` ({feat.feature_type})\n"
        if change_set.removed:
            text += f"\n*Removed ({len(change_set.removed)}):*\n"
            for feat in change_set.removed[:5]:
                name = feat.tags.get("name", f"id:{feat.feature_id}")
                text += f"  • `{name}` ({feat.feature_type})\n"

        payload = {"text": text}
        response = requests.post(self.webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("SlackNotifier: message sent (%d changes)", total)


class EmailNotifier(NotifierBackend):
    """Send a change summary via SMTP email.

    Args:
        smtp_host: SMTP server hostname.
                   <!-- PLACEHOLDER: e.g. "smtp.gmail.com" -->
        smtp_port: SMTP server port (commonly 587 for STARTTLS).
                   <!-- PLACEHOLDER: 587 for STARTTLS, 465 for SSL, 25 for plain -->
        sender: Sender email address.
                <!-- PLACEHOLDER: e.g. "monitor@example.com" -->
        recipients: List of recipient email addresses.
                    <!-- PLACEHOLDER: e.g. ["team@example.com", "gis@example.com"] -->
        username: SMTP login username (often same as sender).
                  <!-- PLACEHOLDER: your SMTP account username -->
        password: SMTP login password or app password.
                  <!-- PLACEHOLDER: use an env var — os.environ["SMTP_PASSWORD"] -->
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipients: list[str],
        username: str,
        password: str,
    ) -> None:
        # PLACEHOLDER: configure all SMTP settings for your email provider
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.recipients = recipients
        self.username = username
        self.password = password

    def send(self, change_set: ChangeSet) -> None:
        """Send an HTML email summarising the change set.

        Args:
            change_set: ChangeSet to report.

        Raises:
            smtplib.SMTPException: On SMTP connection or authentication failure.
        """
        subject = f"OSM Change Alert: {change_set.osm_tag} ({len(change_set.added)} added, {len(change_set.removed)} removed)"
        body_lines = [
            f"<h2>OSM Change Alert</h2>",
            f"<p><b>Tag:</b> {change_set.osm_tag}<br>",
            f"<b>BBox:</b> {change_set.bbox.to_overpass_str()}<br>",
            f"<b>Poll time:</b> {change_set.polled_at.isoformat()}</p>",
        ]

        if change_set.added:
            body_lines.append(f"<h3>Added ({len(change_set.added)})</h3><ul>")
            for feat in change_set.added:
                name = feat.tags.get("name", f"id:{feat.feature_id}")
                body_lines.append(f"<li>{name} ({feat.feature_type})</li>")
            body_lines.append("</ul>")

        if change_set.removed:
            body_lines.append(f"<h3>Removed ({len(change_set.removed)})</h3><ul>")
            for feat in change_set.removed:
                name = feat.tags.get("name", f"id:{feat.feature_id}")
                body_lines.append(f"<li>{name} ({feat.feature_type})</li>")
            body_lines.append("</ul>")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.attach(MIMEText("\n".join(body_lines), "html"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.sender, self.recipients, msg.as_string())

        logger.info("EmailNotifier: sent to %s", self.recipients)


# ---------------------------------------------------------------------------
# Overpass client (thin wrapper with retry)
# ---------------------------------------------------------------------------


class OverpassClient:
    """Thin wrapper around :mod:`overpy` with configurable retry logic.

    Args:
        api_url: Overpass API endpoint.
                 <!-- PLACEHOLDER: defaults to the main public instance;
                      change to a private instance if rate limits are a concern,
                      e.g. "https://overpass.kumi.systems/api/interpreter" -->
        max_retries: Number of retry attempts on transient errors.
        retry_delay: Seconds to wait between retries.
    """

    # PLACEHOLDER: swap to a private Overpass instance if the public one is too slow
    DEFAULT_API_URL = "https://overpass-api.de/api/interpreter"

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self.api = overpy.API(url=api_url)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def query_tag_in_bbox(self, osm_tag: str, bbox: BoundingBox) -> list[OSMFeatureSnapshot]:
        """Query all nodes, ways, and relations with ``osm_tag`` inside ``bbox``.

        Args:
            osm_tag: Tag as ``"key=value"``, e.g. ``"amenity=hospital"``.
                     <!-- PLACEHOLDER: Any valid OSM tag, e.g.
                          "amenity=cafe", "shop=supermarket", "highway=traffic_signals" -->
            bbox: Geographic bounding box.

        Returns:
            List of :class:`OSMFeatureSnapshot` objects.

        Raises:
            overpy.exception.OverPyException: On API error after all retries.
        """
        key, _, value = osm_tag.partition("=")
        if not key or not value:
            raise InputValidationError(
                f"osm_tag must be 'key=value', got: {osm_tag!r}"
            )

        bbox_str = bbox.to_overpass_str()
        query = (
            f'[out:json][timeout:60];\n'
            f'(\n'
            f'  node["{key}"="{value}"]({bbox_str});\n'
            f'  way["{key}"="{value}"]({bbox_str});\n'
            f'  relation["{key}"="{value}"]({bbox_str});\n'
            f');\n'
            f'out center;'
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.api.query(query)
                return self._parse_result(result)
            except overpy.exception.OverPyException as exc:
                logger.warning("Overpass attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

        return []  # unreachable but satisfies mypy

    @staticmethod
    def _parse_result(result: "overpy.Result") -> list[OSMFeatureSnapshot]:
        """Convert overpy Result into :class:`OSMFeatureSnapshot` objects."""
        snapshots: list[OSMFeatureSnapshot] = []

        for node in result.nodes:
            snapshots.append(OSMFeatureSnapshot(
                feature_id=int(node.id),
                feature_type="node",
                tags=dict(node.tags),
                lat=float(node.lat),
                lon=float(node.lon),
            ))

        for way in result.ways:
            lat = float(way.center_lat) if way.center_lat is not None else None
            lon = float(way.center_lon) if way.center_lon is not None else None
            snapshots.append(OSMFeatureSnapshot(
                feature_id=int(way.id),
                feature_type="way",
                tags=dict(way.tags),
                lat=lat,
                lon=lon,
            ))

        for rel in result.relations:
            lat = float(rel.center_lat) if rel.center_lat is not None else None
            lon = float(rel.center_lon) if rel.center_lon is not None else None
            snapshots.append(OSMFeatureSnapshot(
                feature_id=int(rel.id),
                feature_type="relation",
                tags=dict(rel.tags),
                lat=lat,
                lon=lon,
            ))

        return snapshots


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class OSMChangeMonitor(GeoTool):
    """Monitor an OSM bounding box for added/removed features and notify on change.

    Inherits the Template Method pipeline from :class:`~shared.python.GeoTool`.

    State between runs is persisted in a JSON snapshot file inside
    ``output_dir``.  On the first run there is no previous snapshot so the
    tool creates one without sending notifications.  Subsequent runs diff the
    fresh query against the stored snapshot.

    Args:
        bbox: Geographic bounding box.
              <!-- PLACEHOLDER: Replace all four coordinates with your area
                   of interest, e.g. BoundingBox(51.47, -0.15, 51.52, -0.08)
                   for central London -->
        osm_tag: ``"key=value"`` tag to watch.
                 <!-- PLACEHOLDER: Any valid OSM tag, examples:
                      "amenity=hospital", "shop=supermarket",
                      "leisure=park", "highway=traffic_signals" -->
        output_dir: Directory for snapshot JSON and change logs.
                    <!-- PLACEHOLDER: Set to a persistent directory,
                         e.g. Path("data/osm-monitor") — this dir must
                         survive between runs for the diff to work -->
        notifiers: List of :class:`NotifierBackend` instances to invoke
                   when changes are detected.  Defaults to a
                   :class:`JsonFileNotifier` writing to ``output_dir/changes.jsonl``.
                   <!-- PLACEHOLDER: Configure Slack/email/file as needed -->
        overpass_client: Optional pre-configured :class:`OverpassClient`.
                         Defaults to a new client using the public API.
                         <!-- PLACEHOLDER: Pass a custom client if you need a
                              private Overpass instance or custom retry settings -->
        verbose: Enable DEBUG-level logging.
    """

    _SNAPSHOT_FILENAME = "latest_snapshot.json"

    def __init__(
        self,
        bbox: BoundingBox,
        osm_tag: str,
        output_dir: Path,
        notifiers: list[NotifierBackend] | None = None,
        overpass_client: OverpassClient | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__(output_dir, output_dir, verbose=verbose)
        self.bbox = bbox
        self.osm_tag = osm_tag
        self.output_dir = Path(output_dir)
        self.overpass_client = overpass_client or OverpassClient()
        self.notifiers: list[NotifierBackend] = notifiers or [
            # PLACEHOLDER: replace with your preferred notifier(s)
            JsonFileNotifier(self.output_dir / "changes.jsonl"),
        ]
        self._last_change_set: ChangeSet | None = None

    # ------------------------------------------------------------------
    # GeoTool abstract method implementations
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate the bounding box and tag string.

        Raises:
            InputValidationError: If the tag is not in ``key=value`` format or
                bbox coordinates are out of range.
        """
        key, _, value = self.osm_tag.partition("=")
        if not key or not value:
            raise InputValidationError(
                f"osm_tag must be 'key=value', got: {self.osm_tag!r}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Validated tag=%r bbox=%s", self.osm_tag, self.bbox.to_overpass_str())

    def process(self) -> None:
        """Query Overpass, diff against the previous snapshot, and notify.

        On first run (no snapshot file) saves a baseline without notifying.
        On subsequent runs diffs and notifies all backends if changes exist.
        """
        logger.info(
            "Polling Overpass for tag=%r in bbox=%s ...",
            self.osm_tag,
            self.bbox.to_overpass_str(),
        )
        fresh_features = self.overpass_client.query_tag_in_bbox(self.osm_tag, self.bbox)
        now = datetime.now(tz=timezone.utc)
        snapshot_path = self.output_dir / self._SNAPSHOT_FILENAME

        if not snapshot_path.exists():
            # First run — save baseline
            self._save_snapshot(fresh_features, snapshot_path)
            logger.info(
                "First run: saved baseline snapshot (%d features). "
                "Re-run to detect changes.",
                len(fresh_features),
            )
            return

        # Load previous snapshot
        previous_features = self._load_snapshot(snapshot_path)
        change_set = self._compute_diff(self.osm_tag, self.bbox, now, previous_features, fresh_features)

        logger.info(change_set.summary())

        if change_set.has_changes:
            for notifier in self.notifiers:
                try:
                    notifier.send(change_set)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Notifier %s failed: %s", type(notifier).__name__, exc)

        # Always save the fresh snapshot for next run
        self._save_snapshot(fresh_features, snapshot_path)
        self._last_change_set = change_set

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_snapshot(
        self,
        features: list[OSMFeatureSnapshot],
        path: Path,
    ) -> None:
        """Persist a feature list to a JSON file.

        Args:
            features: Features to serialise.
            path: Destination file path.
        """
        payload = {
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "features": [f.to_dict() for f in features],
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise OutputWriteError(str(path), str(exc)) from exc
        logger.debug("Snapshot saved: %d features → %s", len(features), path)

    @staticmethod
    def _load_snapshot(path: Path) -> list[OSMFeatureSnapshot]:
        """Load a previously saved snapshot.

        Args:
            path: JSON snapshot file path.

        Returns:
            List of :class:`OSMFeatureSnapshot` objects.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return [OSMFeatureSnapshot.from_dict(d) for d in data.get("features", [])]

    @staticmethod
    def _compute_diff(
        osm_tag: str,
        bbox: BoundingBox,
        polled_at: datetime,
        previous: list[OSMFeatureSnapshot],
        current: list[OSMFeatureSnapshot],
    ) -> ChangeSet:
        """Compute added/removed features between two snapshots.

        Args:
            osm_tag: Tag being monitored.
            bbox: Bounding box used for the query.
            polled_at: Timestamp of the current poll.
            previous: Feature list from the last snapshot.
            current: Feature list from the current query.

        Returns:
            A :class:`ChangeSet` object with populated ``added``/``removed``.
        """
        prev_ids = {(f.feature_type, f.feature_id) for f in previous}
        curr_ids = {(f.feature_type, f.feature_id) for f in current}

        prev_map = {(f.feature_type, f.feature_id): f for f in previous}
        curr_map = {(f.feature_type, f.feature_id): f for f in current}

        added = [curr_map[k] for k in (curr_ids - prev_ids)]
        removed = [prev_map[k] for k in (prev_ids - curr_ids)]

        return ChangeSet(
            osm_tag=osm_tag,
            bbox=bbox,
            polled_at=polled_at,
            added=added,
            removed=removed,
        )

    @property
    def last_change_set(self) -> ChangeSet | None:
        """The most recent :class:`ChangeSet`, or ``None`` on first run."""
        return self._last_change_set
