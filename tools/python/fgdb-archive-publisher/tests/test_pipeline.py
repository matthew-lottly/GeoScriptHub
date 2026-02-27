"""
Tests — FGDB Archive Publisher
================================
Unit tests for the pipeline configuration parsing and validation logic.

These tests do NOT require ``arcpy`` — they test the pure-Python parts
of the pipeline (config loading, dataclass construction, input
validation).  Integration tests that exercise arcpy operations should
be run inside an ArcGIS Pro Python environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.fgdb_archive_publisher.pipeline import (
    ArchivePipeline,
    DomainDefinition,
    FieldCleanupConfig,
    FieldDefinition,
    PipelineConfig,
    RepublishConfig,
    TopologyRule,
    load_config,
)
from shared.python.exceptions import InputValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config(tmp_path: Path) -> Path:
    """Write a minimal valid pipeline config JSON and return the path."""
    config = {
        "portal_url": "https://myorg.maps.arcgis.com",
        "service_urls": [
            "https://services.arcgis.com/abc/arcgis/rest/services/Parcels/FeatureServer/0"
        ],
        "output_gdb_name": "test_backup.gdb",
        "batch_size": 1000,
        "max_workers": 2,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


@pytest.fixture()
def full_config(tmp_path: Path) -> Path:
    """Write a fully-populated pipeline config JSON and return the path."""
    config = {
        "portal_url": "https://myorg.maps.arcgis.com",
        "service_urls": [
            "https://services.arcgis.com/abc/arcgis/rest/services/Parcels/FeatureServer/0",
            "https://services.arcgis.com/abc/arcgis/rest/services/Buildings/FeatureServer/0",
        ],
        "output_gdb_name": "full_backup.gdb",
        "batch_size": 5000,
        "max_workers": 4,
        "include_attachments": True,
        "spatial_reference": 2263,
        "field_cleanup": {
            "delete_fields": ["GlobalID_1", "OBJECTID_1", "temp_flag"],
            "rename_fields": {"addr": "address", "zn": "zone_code"},
            "add_fields": [
                {"name": "STATUS", "type": "TEXT", "length": 50, "alias": "Status", "domain": "StatusDomain"},
                {"name": "PRIORITY", "type": "SHORT", "alias": "Priority Level"},
            ],
        },
        "domains": [
            {
                "name": "StatusDomain",
                "domain_type": "CODED",
                "field_type": "TEXT",
                "description": "Feature lifecycle status",
                "values": {"Active": "Active", "Inactive": "Inactive", "Pending": "Pending Review"},
            },
            {
                "name": "PriorityRange",
                "domain_type": "RANGE",
                "field_type": "SHORT",
                "description": "1-5 priority scale",
                "values": {"min": 1, "max": 5},
            },
        ],
        "topology_rules": [
            {"rule": "Must Not Overlap (Area)", "feature_class": "Parcels"},
            {"rule": "Must Not Have Gaps (Area)", "feature_class": "Parcels"},
            {
                "rule": "Must Be Covered By Feature Class Of (Area-Area)",
                "feature_class": "Buildings",
                "covering_class": "Parcels",
            },
        ],
        "republish": {
            "target_portal": "https://myorg.maps.arcgis.com",
            "folder": "Published",
            "service_name": "Parcels_Clean",
            "overwrite": True,
        },
    }
    config_path = tmp_path / "config_full.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the JSON config parser."""

    def test_minimal_config_loads(self, minimal_config: Path) -> None:
        """A minimal config with only required keys should parse cleanly."""
        config = load_config(minimal_config)

        assert config.portal_url == "https://myorg.maps.arcgis.com"
        assert len(config.service_urls) == 1
        assert config.output_gdb_name == "test_backup.gdb"
        assert config.batch_size == 1000
        assert config.max_workers == 2

    def test_full_config_loads(self, full_config: Path) -> None:
        """A fully populated config should parse all nested objects."""
        config = load_config(full_config)

        assert len(config.service_urls) == 2
        assert config.spatial_reference == 2263
        assert config.include_attachments is True

        # Field cleanup
        assert config.field_cleanup.delete_fields == ["GlobalID_1", "OBJECTID_1", "temp_flag"]
        assert config.field_cleanup.rename_fields == {"addr": "address", "zn": "zone_code"}
        assert len(config.field_cleanup.add_fields) == 2
        assert config.field_cleanup.add_fields[0].name == "STATUS"
        assert config.field_cleanup.add_fields[0].domain == "StatusDomain"

        # Domains
        assert len(config.domains) == 2
        assert config.domains[0].name == "StatusDomain"
        assert config.domains[0].domain_type == "CODED"
        assert config.domains[1].name == "PriorityRange"
        assert config.domains[1].values == {"min": 1, "max": 5}

        # Topology rules
        assert len(config.topology_rules) == 3
        assert config.topology_rules[2].covering_class == "Parcels"

        # Republish
        assert config.republish.target_portal == "https://myorg.maps.arcgis.com"
        assert config.republish.overwrite is True

    def test_defaults_applied(self, minimal_config: Path) -> None:
        """Missing optional keys should get sensible defaults."""
        config = load_config(minimal_config)

        assert config.include_attachments is True
        assert config.spatial_reference == 4326
        assert config.field_cleanup.delete_fields == []
        assert config.field_cleanup.rename_fields == {}
        assert config.field_cleanup.add_fields == []
        assert config.domains == []
        assert config.topology_rules == []
        assert config.republish.target_portal == ""

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON should raise InputValidationError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json !!!", encoding="utf-8")

        with pytest.raises(InputValidationError, match="Failed to read config"):
            load_config(bad)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A non-existent config file should raise InputValidationError."""
        with pytest.raises(InputValidationError, match="Failed to read config"):
            load_config(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for the configuration dataclasses."""

    def test_field_definition_defaults(self) -> None:
        """FieldDefinition should have sensible defaults."""
        fd = FieldDefinition(name="test", type="TEXT")
        assert fd.length == 255
        assert fd.alias == ""
        assert fd.domain == ""

    def test_domain_definition(self) -> None:
        """DomainDefinition should store values correctly."""
        dd = DomainDefinition(
            name="TestDomain",
            domain_type="CODED",
            field_type="TEXT",
            values={"A": "Active", "I": "Inactive"},
        )
        assert dd.values == {"A": "Active", "I": "Inactive"}

    def test_topology_rule_defaults(self) -> None:
        """TopologyRule optional fields should default to empty strings."""
        tr = TopologyRule(rule="Must Not Overlap (Area)", feature_class="Parcels")
        assert tr.subtype == ""
        assert tr.covering_class == ""
        assert tr.covering_subtype == ""

    def test_pipeline_config_defaults(self) -> None:
        """PipelineConfig should have production-safe defaults."""
        pc = PipelineConfig()
        assert pc.batch_size == 5000
        assert pc.max_workers == 4
        assert pc.spatial_reference == 4326
        assert pc.include_attachments is True


# ---------------------------------------------------------------------------
# Pipeline validation tests
# ---------------------------------------------------------------------------


class TestPipelineValidation:
    """Tests for ArchivePipeline.validate_inputs()."""

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        """A non-existent config path should raise InputValidationError."""
        pipeline = ArchivePipeline(
            input_path=tmp_path / "nope.json",
            output_path=tmp_path / "out",
        )
        with pytest.raises(InputValidationError, match="Config file not found"):
            pipeline.validate_inputs()

    def test_empty_portal_url_raises(self, tmp_path: Path) -> None:
        """A config with an empty portal_url should fail validation."""
        config_path = tmp_path / "empty_portal.json"
        config_path.write_text(json.dumps({
            "portal_url": "",
            "service_urls": ["https://example.com/FeatureServer/0"],
        }), encoding="utf-8")

        pipeline = ArchivePipeline(
            input_path=config_path,
            output_path=tmp_path / "out",
        )
        with pytest.raises(InputValidationError, match="portal_url"):
            pipeline.validate_inputs()

    def test_empty_service_urls_raises(self, tmp_path: Path) -> None:
        """A config with no service URLs should fail validation."""
        config_path = tmp_path / "no_urls.json"
        config_path.write_text(json.dumps({
            "portal_url": "https://myportal.com",
            "service_urls": [],
        }), encoding="utf-8")

        pipeline = ArchivePipeline(
            input_path=config_path,
            output_path=tmp_path / "out",
        )
        with pytest.raises(InputValidationError, match="service_urls"):
            pipeline.validate_inputs()

    def test_invalid_batch_size_raises(self, tmp_path: Path) -> None:
        """batch_size < 1 should fail validation."""
        config_path = tmp_path / "bad_batch.json"
        config_path.write_text(json.dumps({
            "portal_url": "https://myportal.com",
            "service_urls": ["https://example.com/FeatureServer/0"],
            "batch_size": 0,
        }), encoding="utf-8")

        pipeline = ArchivePipeline(
            input_path=config_path,
            output_path=tmp_path / "out",
        )
        with pytest.raises(InputValidationError, match="batch_size"):
            pipeline.validate_inputs()

    def test_valid_config_passes(self, minimal_config: Path, tmp_path: Path) -> None:
        """A valid minimal config should pass validation without error."""
        pipeline = ArchivePipeline(
            input_path=minimal_config,
            output_path=tmp_path / "out",
        )
        pipeline.validate_inputs()

        assert pipeline.config is not None
        assert pipeline.config.portal_url == "https://myorg.maps.arcgis.com"
        assert (tmp_path / "out").exists()
