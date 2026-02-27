"""
FGDB Archive Publisher — Schema Manager
========================================
Handles all post-archive schema modifications:

- **Delete fields** — remove unwanted columns inherited from the source.
- **Rename fields** — apply cleaner names using ``arcpy.management.AlterField``.
- **Add fields** — create new columns with type, alias, and domain.
- **Create domains** — coded-value or range domains on the geodatabase.
- **Assign domains** — link domains to the appropriate fields.

All operations use ``arcpy`` so domain rules, subtypes, and field
properties are fully honoured.

Usage::

    manager = SchemaManager(gdb_path=Path("backup.gdb"), config=pipeline_config)
    manager.apply(["backup.gdb/Parcels", "backup.gdb/Buildings"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import arcpy  # type: ignore[import-unresolved]

from shared.python.exceptions import GeoScriptHubError

if TYPE_CHECKING:
    from src.fgdb_archive_publisher.pipeline import PipelineConfig

logger = logging.getLogger("geoscripthub.fgdb_archive_publisher.schema_manager")


class SchemaError(GeoScriptHubError):
    """Raised when a schema modification operation fails."""


class SchemaManager:
    """Apply field cleanup, domains, and schema modifications to feature classes.

    Attributes:
        gdb_path: Absolute path to the File Geodatabase.
        config: Full pipeline config (field cleanup + domains).
    """

    def __init__(self, gdb_path: Path, config: PipelineConfig) -> None:
        self.gdb_path = gdb_path
        self.config = config

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply(self, feature_classes: list[str]) -> None:
        """Apply all schema modifications to the given feature classes.

        Operations run in order: create domains → delete fields →
        rename fields → add fields → assign domains.

        Args:
            feature_classes: Full paths to each feature class in the FGDB.
        """
        self._create_domains()

        for fc_path in feature_classes:
            fc_name = Path(fc_path).name
            logger.info("Applying schema changes to '%s'.", fc_name)

            self._delete_fields(fc_path)
            self._rename_fields(fc_path)
            self._add_fields(fc_path)
            self._assign_domains(fc_path)

        logger.info("Schema modifications complete for %d feature class(es).", len(feature_classes))

    # ------------------------------------------------------------------
    # Domain creation
    # ------------------------------------------------------------------

    def _create_domains(self) -> None:
        """Create all coded-value and range domains in the geodatabase.

        Existing domains with the same name are skipped with a warning.
        """
        gdb = str(self.gdb_path)
        existing_domains = {d.name for d in arcpy.da.ListDomains(gdb)}

        for domain_def in self.config.domains:
            if domain_def.name in existing_domains:
                logger.warning("Domain '%s' already exists — skipping.", domain_def.name)
                continue

            try:
                arcpy.management.CreateDomain(
                    in_workspace=gdb,
                    domain_name=domain_def.name,
                    domain_description=domain_def.description or domain_def.name,
                    field_type=domain_def.field_type,
                    domain_type=domain_def.domain_type,
                )

                if domain_def.domain_type.upper() == "CODED":
                    for code, desc in domain_def.values.items():
                        arcpy.management.AddCodedValueToDomain(
                            in_workspace=gdb,
                            domain_name=domain_def.name,
                            code=code,
                            code_description=str(desc),
                        )
                elif domain_def.domain_type.upper() == "RANGE":
                    arcpy.management.SetValueForRangeDomain(
                        in_workspace=gdb,
                        domain_name=domain_def.name,
                        min_value=domain_def.values.get("min", 0),
                        max_value=domain_def.values.get("max", 999999),
                    )

                logger.info("Created domain '%s' (%s).", domain_def.name, domain_def.domain_type)

            except arcpy.ExecuteError as exc:
                raise SchemaError(
                    f"Failed to create domain '{domain_def.name}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Field deletion
    # ------------------------------------------------------------------

    def _delete_fields(self, fc_path: str) -> None:
        """Remove unwanted fields from a feature class.

        System fields (OID, Shape, GlobalID) are automatically excluded
        from deletion even if listed.

        Args:
            fc_path: Full path to the feature class.
        """
        fields_to_delete = self.config.field_cleanup.delete_fields
        if not fields_to_delete:
            return

        existing = {f.name for f in arcpy.ListFields(fc_path)}
        safe_to_delete = [
            f for f in fields_to_delete
            if f in existing and f.upper() not in ("OBJECTID", "SHAPE", "GLOBALID", "SHAPE_LENGTH", "SHAPE_AREA")
        ]

        if not safe_to_delete:
            return

        try:
            arcpy.management.DeleteField(fc_path, safe_to_delete)
            logger.info("Deleted %d field(s) from '%s': %s", len(safe_to_delete), Path(fc_path).name, safe_to_delete)
        except arcpy.ExecuteError as exc:
            raise SchemaError(f"Failed to delete fields from '{fc_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Field renaming
    # ------------------------------------------------------------------

    def _rename_fields(self, fc_path: str) -> None:
        """Rename fields using ``arcpy.management.AlterField``.

        Args:
            fc_path: Full path to the feature class.
        """
        rename_map = self.config.field_cleanup.rename_fields
        if not rename_map:
            return

        existing = {f.name for f in arcpy.ListFields(fc_path)}

        for old_name, new_name in rename_map.items():
            if old_name not in existing:
                logger.warning("Field '%s' not found in '%s' — skipping rename.", old_name, Path(fc_path).name)
                continue
            try:
                arcpy.management.AlterField(
                    in_table=fc_path,
                    field=old_name,
                    new_field_name=new_name,
                    new_field_alias=new_name,
                )
                logger.info("Renamed '%s' → '%s' in '%s'.", old_name, new_name, Path(fc_path).name)
            except arcpy.ExecuteError as exc:
                raise SchemaError(
                    f"Failed to rename field '{old_name}' in '{fc_path}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Field addition
    # ------------------------------------------------------------------

    def _add_fields(self, fc_path: str) -> None:
        """Add new fields to the feature class.

        Args:
            fc_path: Full path to the feature class.
        """
        fields_to_add = self.config.field_cleanup.add_fields
        if not fields_to_add:
            return

        existing = {f.name for f in arcpy.ListFields(fc_path)}

        for field_def in fields_to_add:
            if field_def.name in existing:
                logger.warning("Field '%s' already exists in '%s' — skipping.", field_def.name, Path(fc_path).name)
                continue
            try:
                arcpy.management.AddField(
                    in_table=fc_path,
                    field_name=field_def.name,
                    field_type=field_def.type,
                    field_length=field_def.length if field_def.type.upper() == "TEXT" else None,
                    field_alias=field_def.alias or field_def.name,
                )
                logger.info("Added field '%s' (%s) to '%s'.", field_def.name, field_def.type, Path(fc_path).name)
            except arcpy.ExecuteError as exc:
                raise SchemaError(
                    f"Failed to add field '{field_def.name}' to '{fc_path}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Domain assignment
    # ------------------------------------------------------------------

    def _assign_domains(self, fc_path: str) -> None:
        """Assign domains to fields based on the ``domain`` property in add_fields.

        Only applies to fields that were listed in ``add_fields`` AND
        have a non-empty ``domain`` value.

        Args:
            fc_path: Full path to the feature class.
        """
        for field_def in self.config.field_cleanup.add_fields:
            if not field_def.domain:
                continue
            try:
                arcpy.management.AssignDomainToField(
                    in_table=fc_path,
                    field_name=field_def.name,
                    domain_name=field_def.domain,
                )
                logger.info(
                    "Assigned domain '%s' to field '%s' in '%s'.",
                    field_def.domain, field_def.name, Path(fc_path).name,
                )
            except arcpy.ExecuteError as exc:
                raise SchemaError(
                    f"Failed to assign domain '{field_def.domain}' to "
                    f"'{field_def.name}' in '{fc_path}': {exc}"
                ) from exc
