"""
FGDB Archive Publisher — Pipeline Orchestrator
===============================================
Main pipeline class that coordinates the full archive-clean-validate-publish
workflow.  Inherits from :class:`~shared.python.base_tool.GeoTool` and
implements the Template Method pattern.

Usage::

    from pathlib import Path
    from fgdb_archive_publisher.pipeline import ArchivePipeline

    pipeline = ArchivePipeline(
        input_path=Path("config.json"),
        output_path=Path("backups/"),
        verbose=True,
    )
    pipeline.run()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.python.base_tool import GeoTool
from shared.python.exceptions import InputValidationError

from src.fgdb_archive_publisher.archiver import Archiver
from src.fgdb_archive_publisher.publisher import Publisher
from src.fgdb_archive_publisher.schema_manager import SchemaManager
from src.fgdb_archive_publisher.topology_checker import TopologyChecker

logger = logging.getLogger("geoscripthub.fgdb_archive_publisher")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FieldDefinition:
    """Schema definition for a field to add to a feature class.

    Attributes:
        name: Field name (no spaces, max 64 characters).
        type: ArcGIS field type — ``TEXT``, ``LONG``, ``SHORT``,
              ``DOUBLE``, ``FLOAT``, ``DATE``, ``BLOB``, ``GUID``.
        length: String length (only relevant for ``TEXT`` fields).
        alias: Optional human-readable alias for the field.
        domain: Optional domain name to assign to this field.
    """

    name: str
    type: str
    length: int = 255
    alias: str = ""
    domain: str = ""


@dataclass
class DomainDefinition:
    """Schema definition for a coded-value or range domain.

    Attributes:
        name: Domain name (unique within the geodatabase).
        domain_type: ``CODED`` for coded-value domains or ``RANGE``
                     for range domains.
        field_type: ArcGIS field type the domain applies to.
        values: For coded domains — ``{code: description}`` mapping.
                For range domains — ``{"min": number, "max": number}``.
        description: Optional human-readable description.
    """

    name: str
    domain_type: str
    field_type: str
    values: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class TopologyRule:
    """A single topology rule to apply during validation.

    Attributes:
        rule: ArcGIS topology rule name, e.g.
              ``"Must Not Overlap (Area)"``,
              ``"Must Not Have Gaps (Area)"``,
              ``"Must Not Self-Overlap (Line)"``.
        feature_class: Name of the feature class the rule targets.
        subtype: Optional subtype code (default ``""`` = all).
        covering_class: For two-class rules (e.g.
              ``"Must Be Covered By Feature Class Of"``), the
              name of the covering feature class.
        covering_subtype: Subtype code for the covering class.
    """

    rule: str
    feature_class: str
    subtype: str = ""
    covering_class: str = ""
    covering_subtype: str = ""


@dataclass
class FieldCleanupConfig:
    """Configuration for field-level cleanup operations.

    Attributes:
        delete_fields: Field names to remove after archiving.
        rename_fields: ``{old_name: new_name}`` mapping for renames.
        add_fields: New field definitions to create.
    """

    delete_fields: list[str] = field(default_factory=list)
    rename_fields: dict[str, str] = field(default_factory=dict)
    add_fields: list[FieldDefinition] = field(default_factory=list)


@dataclass
class RepublishConfig:
    """Configuration for republishing the cleaned data to a portal.

    Attributes:
        target_portal: Portal URL (e.g. ``https://myorg.maps.arcgis.com``).
        folder: Portal folder to publish into.
        service_name: Name for the hosted feature service.
        overwrite: When ``True``, overwrite an existing service
                   with the same name.
    """

    target_portal: str = ""
    folder: str = ""
    service_name: str = ""
    overwrite: bool = True


@dataclass
class PipelineConfig:
    """Full pipeline configuration parsed from a JSON file.

    Attributes:
        portal_url: Source portal URL for data download.
        service_urls: List of feature service layer URLs to archive.
        output_gdb_name: Name for the output File Geodatabase
                         (e.g. ``"backup_2024.gdb"``).
        batch_size: Number of features to fetch per query batch.
        max_workers: Thread pool size for parallel attachment downloads.
        include_attachments: Whether to download feature attachments.
        field_cleanup: Field cleanup configuration.
        domains: Domain definitions to create and assign.
        topology_rules: Topology rules for data validation.
        republish: Portal republish configuration.
        spatial_reference: WKID for the output feature dataset
                          (default ``4326`` = WGS 84).
    """

    portal_url: str = ""
    service_urls: list[str] = field(default_factory=list)
    output_gdb_name: str = "archive.gdb"
    batch_size: int = 5000
    max_workers: int = 4
    include_attachments: bool = True
    field_cleanup: FieldCleanupConfig = field(default_factory=FieldCleanupConfig)
    domains: list[DomainDefinition] = field(default_factory=list)
    topology_rules: list[TopologyRule] = field(default_factory=list)
    republish: RepublishConfig = field(default_factory=RepublishConfig)
    spatial_reference: int = 4326


# ---------------------------------------------------------------------------
# Config parser
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> PipelineConfig:
    """Parse a JSON configuration file into a :class:`PipelineConfig`.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        A fully populated ``PipelineConfig`` instance.

    Raises:
        InputValidationError: If the file cannot be read or parsed.
    """
    try:
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise InputValidationError(
            f"Failed to read config file '{config_path}': {exc}"
        ) from exc

    cleanup_raw = raw.get("field_cleanup", {})
    field_cleanup = FieldCleanupConfig(
        delete_fields=cleanup_raw.get("delete_fields", []),
        rename_fields=cleanup_raw.get("rename_fields", {}),
        add_fields=[
            FieldDefinition(**f) for f in cleanup_raw.get("add_fields", [])
        ],
    )

    domains = [DomainDefinition(**d) for d in raw.get("domains", [])]
    topology_rules = [TopologyRule(**r) for r in raw.get("topology_rules", [])]

    republish_raw = raw.get("republish", {})
    republish = RepublishConfig(
        target_portal=republish_raw.get("target_portal", ""),
        folder=republish_raw.get("folder", ""),
        service_name=republish_raw.get("service_name", ""),
        overwrite=republish_raw.get("overwrite", True),
    )

    return PipelineConfig(
        portal_url=raw.get("portal_url", ""),
        service_urls=raw.get("service_urls", []),
        output_gdb_name=raw.get("output_gdb_name", "archive.gdb"),
        batch_size=raw.get("batch_size", 5000),
        max_workers=raw.get("max_workers", 4),
        include_attachments=raw.get("include_attachments", True),
        field_cleanup=field_cleanup,
        domains=domains,
        topology_rules=topology_rules,
        republish=republish,
        spatial_reference=raw.get("spatial_reference", 4326),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ArchivePipeline(GeoTool):
    """Full archive-clean-validate-publish pipeline.

    Reads a JSON config, exports portal feature layers to a local FGDB,
    applies schema changes and topology validation, then optionally
    republishes the result.

    Attributes:
        config: Parsed pipeline configuration.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialise the pipeline.

        Args:
            input_path: Path to the JSON configuration file.
            output_path: Directory where the output FGDB will be created.
            verbose: Enable debug-level logging.
        """
        super().__init__(input_path=input_path, output_path=output_path, verbose=verbose)
        self.config: PipelineConfig | None = None

    # ------------------------------------------------------------------
    # GeoTool interface
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate the config file and its contents.

        Raises:
            InputValidationError: On missing/invalid config values.
        """
        if not self.input_path.exists():
            raise InputValidationError(
                f"Config file not found: '{self.input_path}'."
            )

        self.config = load_config(self.input_path)

        if not self.config.portal_url:
            raise InputValidationError(
                "Config key 'portal_url' is required."
            )
        if not self.config.service_urls:
            raise InputValidationError(
                "Config key 'service_urls' must contain at least one URL."
            )
        if self.config.batch_size < 1:
            raise InputValidationError(
                f"'batch_size' must be >= 1, got {self.config.batch_size}."
            )
        if self.config.max_workers < 1:
            raise InputValidationError(
                f"'max_workers' must be >= 1, got {self.config.max_workers}."
            )

        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Configuration validated — %d layer(s) to archive.", len(self.config.service_urls))

    def process(self) -> None:
        """Execute the full pipeline: archive → schema → topology → publish.

        Each stage logs progress and is safe to interrupt between stages.
        """
        assert self.config is not None, "Call validate_inputs() first."

        gdb_path = self._archive()
        feature_classes = self._list_feature_classes(gdb_path)

        if self.config.domains or self.config.field_cleanup.add_fields or \
           self.config.field_cleanup.delete_fields or self.config.field_cleanup.rename_fields:
            self._apply_schema(gdb_path, feature_classes)

        if self.config.topology_rules:
            self._validate_topology(gdb_path, feature_classes)

        if self.config.republish.target_portal:
            self._publish(gdb_path)

        logger.info("Pipeline complete — output at '%s'.", gdb_path)

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _archive(self) -> Path:
        """Run the archive/export stage and return the FGDB path."""
        assert self.config is not None
        archiver = Archiver(
            portal_url=self.config.portal_url,
            service_urls=self.config.service_urls,
            output_dir=self.output_path,
            gdb_name=self.config.output_gdb_name,
            batch_size=self.config.batch_size,
            max_workers=self.config.max_workers,
            include_attachments=self.config.include_attachments,
            spatial_reference=self.config.spatial_reference,
        )
        return archiver.export()

    def _list_feature_classes(self, gdb_path: Path) -> list[str]:
        """Return a list of feature class paths inside the FGDB."""
        import arcpy  # type: ignore[import-unresolved]

        arcpy.env.workspace = str(gdb_path)
        fcs: list[str] = []

        # Top-level feature classes
        for fc in arcpy.ListFeatureClasses() or []:
            fcs.append(str(gdb_path / fc))

        # Feature classes inside feature datasets
        for fds in arcpy.ListDatasets(feature_type="Feature") or []:
            for fc in arcpy.ListFeatureClasses(feature_dataset=fds) or []:
                fcs.append(str(gdb_path / fds / fc))

        logger.info("Found %d feature class(es) in '%s'.", len(fcs), gdb_path.name)
        return fcs

    def _apply_schema(self, gdb_path: Path, feature_classes: list[str]) -> None:
        """Run schema cleanup and domain assignment."""
        assert self.config is not None
        manager = SchemaManager(gdb_path=gdb_path, config=self.config)
        manager.apply(feature_classes)

    def _validate_topology(self, gdb_path: Path, feature_classes: list[str]) -> None:
        """Create and validate a topology dataset using user-defined rules."""
        assert self.config is not None
        checker = TopologyChecker(
            gdb_path=gdb_path,
            rules=self.config.topology_rules,
            spatial_reference=self.config.spatial_reference,
        )
        checker.validate(feature_classes)

    def _publish(self, gdb_path: Path) -> None:
        """Republish the cleaned FGDB to the target portal."""
        assert self.config is not None
        publisher = Publisher(
            gdb_path=gdb_path,
            config=self.config.republish,
        )
        publisher.publish()
