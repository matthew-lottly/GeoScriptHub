"""
Tests for OSM Change Monitor
==============================
All Overpass API calls are mocked with ``unittest.mock`` so no real HTTP
requests are made.

Test classes:
    TestBoundingBox                  BoundingBox validation.
    TestOSMFeatureSnapshot           Serialisation round-trip.
    TestChangeSet                    Diff logic and summary string.
    TestJsonFileNotifier             File output from JsonFileNotifier.
    TestOSMChangeMonitorHappyPath    Full pipeline with mocked OverpassClient.
    TestOSMChangeMonitorValidation   Error conditions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osm_change_monitor.monitor import (
    BoundingBox,
    ChangeSet,
    JsonFileNotifier,
    NotifierBackend,
    OSMChangeMonitor,
    OSMFeatureSnapshot,
    OverpassClient,
)
from shared.python.exceptions import InputValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature(fid: int = 1, ftype: str = "node", name: str = "Test") -> OSMFeatureSnapshot:
    """Create a dummy OSMFeatureSnapshot."""
    return OSMFeatureSnapshot(
        feature_id=fid,
        feature_type=ftype,
        tags={"amenity": "hospital", "name": name},
        lat=51.5,
        lon=-0.1,
    )


def _make_monitor(tmp_path: Path, notifiers=None, features=None) -> tuple[OSMChangeMonitor, MagicMock]:
    """
    Build an OSMChangeMonitor with a mocked OverpassClient.

    Returns (monitor, mock_client).
    """
    if features is None:
        features = [_make_feature(1)]

    mock_client = MagicMock(spec=OverpassClient)
    mock_client.query_tag_in_bbox.return_value = features

    monitor = OSMChangeMonitor(
        bbox=BoundingBox(51.47, -0.15, 51.52, -0.08),
        osm_tag="amenity=hospital",
        output_dir=tmp_path / "monitor",
        notifiers=notifiers or [],
        overpass_client=mock_client,
    )
    return monitor, mock_client


# ---------------------------------------------------------------------------
# BoundingBox tests
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_valid_bbox(self) -> None:
        bbox = BoundingBox(51.0, -1.0, 52.0, 0.0)
        assert bbox.to_overpass_str() == "51.0,-1.0,52.0,0.0"

    def test_south_north_inverted_raises(self) -> None:
        with pytest.raises(InputValidationError):
            BoundingBox(south=52.0, west=-1.0, north=51.0, east=0.0)  # south > north

    def test_west_east_inverted_raises(self) -> None:
        with pytest.raises(InputValidationError):
            BoundingBox(south=51.0, west=1.0, north=52.0, east=-1.0)  # west > east

    def test_overpass_str_format(self) -> None:
        bbox = BoundingBox(51.47, -0.15, 51.52, -0.08)
        assert bbox.to_overpass_str() == "51.47,-0.15,51.52,-0.08"


# ---------------------------------------------------------------------------
# OSMFeatureSnapshot tests
# ---------------------------------------------------------------------------


class TestOSMFeatureSnapshot:
    def test_to_dict_round_trip(self) -> None:
        feat = _make_feature(42, "way", "King's College Hospital")
        d = feat.to_dict()
        restored = OSMFeatureSnapshot.from_dict(d)
        assert restored == feat

    def test_from_dict_missing_tags(self) -> None:
        d = {"id": 1, "type": "node", "lat": 51.5, "lon": -0.1}
        feat = OSMFeatureSnapshot.from_dict(d)
        assert feat.tags == {}


# ---------------------------------------------------------------------------
# ChangeSet diff/summary tests
# ---------------------------------------------------------------------------


class TestChangeSet:
    def _bbox(self) -> BoundingBox:
        return BoundingBox(51.47, -0.15, 51.52, -0.08)

    def test_has_changes_false(self) -> None:
        cs = ChangeSet("amenity=hospital", self._bbox(), datetime.now(tz=timezone.utc))
        assert not cs.has_changes

    def test_has_changes_added(self) -> None:
        cs = ChangeSet(
            "amenity=hospital", self._bbox(), datetime.now(tz=timezone.utc),
            added=[_make_feature()],
        )
        assert cs.has_changes

    def test_has_changes_removed(self) -> None:
        cs = ChangeSet(
            "amenity=hospital", self._bbox(), datetime.now(tz=timezone.utc),
            removed=[_make_feature()],
        )
        assert cs.has_changes

    def test_summary_contains_tag(self) -> None:
        cs = ChangeSet(
            "amenity=hospital", self._bbox(), datetime.now(tz=timezone.utc),
            added=[_make_feature()],
        )
        assert "amenity=hospital" in cs.summary()

    def test_to_dict_serialisable(self) -> None:
        cs = ChangeSet(
            "amenity=hospital", self._bbox(), datetime.now(tz=timezone.utc),
            added=[_make_feature()], removed=[],
        )
        d = cs.to_dict()
        # Must be JSON-roundtrippable
        assert json.loads(json.dumps(d))["osm_tag"] == "amenity=hospital"


# ---------------------------------------------------------------------------
# JsonFileNotifier tests
# ---------------------------------------------------------------------------


class TestJsonFileNotifier:
    def test_creates_file_on_send(self, tmp_path: Path) -> None:
        out = tmp_path / "changes.jsonl"
        notifier = JsonFileNotifier(out)
        bbox = BoundingBox(51.0, -1.0, 52.0, 0.0)
        cs = ChangeSet("amenity=cafe", bbox, datetime.now(tz=timezone.utc), added=[_make_feature()])
        notifier.send(cs)
        assert out.exists()

    def test_appends_json_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "changes.jsonl"
        notifier = JsonFileNotifier(out)
        bbox = BoundingBox(51.0, -1.0, 52.0, 0.0)
        cs = ChangeSet("amenity=cafe", bbox, datetime.now(tz=timezone.utc), added=[_make_feature()])
        notifier.send(cs)
        notifier.send(cs)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "changes.jsonl"
        notifier = JsonFileNotifier(out)
        bbox = BoundingBox(51.0, -1.0, 52.0, 0.0)
        cs = ChangeSet("amenity=cafe", bbox, datetime.now(tz=timezone.utc), added=[_make_feature()])
        notifier.send(cs)
        assert out.exists()


# ---------------------------------------------------------------------------
# OSMChangeMonitor happy-path tests
# ---------------------------------------------------------------------------


class TestOSMChangeMonitorHappyPath:
    def test_first_run_saves_snapshot(self, tmp_path: Path) -> None:
        """First run must save a baseline snapshot file."""
        monitor, _ = _make_monitor(tmp_path)
        monitor.run()
        snapshot = tmp_path / "monitor" / "latest_snapshot.json"
        assert snapshot.exists()

    def test_first_run_no_change_set(self, tmp_path: Path) -> None:
        """On first run last_change_set should be None (no diff possible)."""
        monitor, _ = _make_monitor(tmp_path)
        monitor.run()
        assert monitor.last_change_set is None

    def test_second_run_no_changes(self, tmp_path: Path) -> None:
        """Identical queries on consecutive runs → no changes."""
        features = [_make_feature(1)]
        monitor, mock_client = _make_monitor(tmp_path, features=features)
        monitor.run()  # first run
        monitor.run()  # second run (same features)
        assert monitor.last_change_set is not None
        assert not monitor.last_change_set.has_changes

    def test_second_run_detects_added_feature(self, tmp_path: Path) -> None:
        """Feature added between runs should appear in change_set.added."""
        monitor, mock_client = _make_monitor(tmp_path, features=[_make_feature(1)])
        monitor.run()  # baseline: 1 feature

        # Simulate an added feature in the second query
        mock_client.query_tag_in_bbox.return_value = [
            _make_feature(1),
            _make_feature(2, name="New Hospital"),
        ]
        monitor.run()

        cs = monitor.last_change_set
        assert cs is not None
        assert len(cs.added) == 1
        assert cs.added[0].feature_id == 2

    def test_second_run_detects_removed_feature(self, tmp_path: Path) -> None:
        """Feature absent in second query should appear in change_set.removed."""
        monitor, mock_client = _make_monitor(
            tmp_path,
            features=[_make_feature(1), _make_feature(2, name="Old Clinic")],
        )
        monitor.run()  # baseline: 2 features

        mock_client.query_tag_in_bbox.return_value = [_make_feature(1)]
        monitor.run()

        cs = monitor.last_change_set
        assert cs is not None
        assert len(cs.removed) == 1
        assert cs.removed[0].feature_id == 2

    def test_notifier_called_on_change(self, tmp_path: Path) -> None:
        """NotifierBackend.send() should be called when changes are detected."""
        mock_notifier = MagicMock(spec=NotifierBackend)
        monitor, mock_client = _make_monitor(tmp_path, notifiers=[mock_notifier], features=[_make_feature(1)])
        monitor.run()  # baseline

        mock_client.query_tag_in_bbox.return_value = [_make_feature(1), _make_feature(2)]
        monitor.run()

        mock_notifier.send.assert_called_once()

    def test_notifier_not_called_when_no_changes(self, tmp_path: Path) -> None:
        """NotifierBackend.send() should NOT be called when nothing changed."""
        mock_notifier = MagicMock(spec=NotifierBackend)
        features = [_make_feature(1)]
        monitor, _ = _make_monitor(tmp_path, notifiers=[mock_notifier], features=features)
        monitor.run()  # baseline
        monitor.run()  # identical

        mock_notifier.send.assert_not_called()


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestOSMChangeMonitorValidation:
    def test_invalid_tag_format_raises(self, tmp_path: Path) -> None:
        """Tags without '=' should raise InputValidationError."""
        mock_client = MagicMock(spec=OverpassClient)
        monitor = OSMChangeMonitor(
            bbox=BoundingBox(51.47, -0.15, 51.52, -0.08),
            osm_tag="amenityhospital",  # missing '='
            output_dir=tmp_path / "out",
            notifiers=[],
            overpass_client=mock_client,
        )
        with pytest.raises(InputValidationError, match="key=value"):
            monitor.run()
