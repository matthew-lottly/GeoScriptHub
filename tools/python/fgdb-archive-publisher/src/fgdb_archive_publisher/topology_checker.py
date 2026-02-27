"""
FGDB Archive Publisher — Topology Checker
==========================================
Creates a topology inside the File Geodatabase, adds user-defined rules,
validates, and exports any errors to a human-readable log.

Supported topology rules are any rule string accepted by
``arcpy.management.AddRuleToTopology``, for example:

- ``"Must Not Overlap (Area)"``
- ``"Must Not Have Gaps (Area)"``
- ``"Must Not Overlap (Line)"``
- ``"Must Not Self-Overlap (Line)"``
- ``"Must Not Self-Intersect (Line)"``
- ``"Must Be Single Part (Area)"``
- ``"Must Be Covered By Feature Class Of (Area-Area)"``
- ``"Must Not Have Dangles (Line)"``
- ``"Boundary Must Be Covered By (Area-Line)"``

The user specifies rules in the JSON config and the checker applies them
automatically, validates the topology, and writes any errors to a
``topology_errors/`` subdirectory inside the FGDB parent folder.

Usage::

    checker = TopologyChecker(
        gdb_path=Path("backup.gdb"),
        rules=[TopologyRule(rule="Must Not Overlap (Area)", feature_class="Parcels")],
        spatial_reference=4326,
    )
    checker.validate(["backup.gdb/Parcels"])
"""

from __future__ import annotations

import logging
from pathlib import Path

import arcpy  # type: ignore[import-unresolved]

from shared.python.exceptions import GeoScriptHubError

from src.fgdb_archive_publisher.pipeline import TopologyRule

logger = logging.getLogger("geoscripthub.fgdb_archive_publisher.topology_checker")

# Name for the feature dataset and topology created during validation.
_TOPOLOGY_FDS_NAME = "TopologyValidation"
_TOPOLOGY_NAME = "DataQuality_Topology"
_CLUSTER_TOLERANCE = 0.001


class TopologyError(GeoScriptHubError):
    """Raised when topology creation or validation fails."""


class TopologyChecker:
    """Create, populate, and validate a geodatabase topology.

    Attributes:
        gdb_path: Absolute path to the File Geodatabase.
        rules: User-defined topology rules.
        spatial_reference: WKID for the feature dataset.
    """

    def __init__(
        self,
        gdb_path: Path,
        rules: list[TopologyRule],
        spatial_reference: int = 4326,
    ) -> None:
        self.gdb_path = gdb_path
        self.rules = rules
        self.spatial_reference = spatial_reference

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate(self, feature_classes: list[str]) -> Path:
        """Run the full topology validation workflow.

        Steps:
            1. Create a feature dataset inside the FGDB.
            2. Import feature classes into the feature dataset.
            3. Create a topology and add the user-defined rules.
            4. Validate the topology.
            5. Export any errors to a subfolder.

        Args:
            feature_classes: Full paths to the source feature classes.

        Returns:
            Path to the topology errors directory.

        Raises:
            TopologyError: On any arcpy failure.
        """
        fds_path = self._create_feature_dataset()
        fc_map = self._import_feature_classes(feature_classes, fds_path)
        topo_path = self._create_topology(fds_path)
        self._add_rules(topo_path, fc_map)
        self._validate_topology(topo_path)
        error_dir = self._export_errors(topo_path)
        return error_dir

    # ------------------------------------------------------------------
    # Step 1 — Feature dataset
    # ------------------------------------------------------------------

    def _create_feature_dataset(self) -> str:
        """Create (or recreate) the topology feature dataset.

        Returns:
            Full path to the feature dataset.
        """
        fds_path = str(self.gdb_path / _TOPOLOGY_FDS_NAME)

        if arcpy.Exists(fds_path):
            arcpy.management.Delete(fds_path)

        sr = arcpy.SpatialReference(self.spatial_reference)
        arcpy.management.CreateFeatureDataset(
            out_dataset_path=str(self.gdb_path),
            out_name=_TOPOLOGY_FDS_NAME,
            spatial_reference=sr,
        )
        logger.info("Created feature dataset '%s'.", _TOPOLOGY_FDS_NAME)
        return fds_path

    # ------------------------------------------------------------------
    # Step 2 — Import feature classes
    # ------------------------------------------------------------------

    def _import_feature_classes(
        self,
        feature_classes: list[str],
        fds_path: str,
    ) -> dict[str, str]:
        """Copy feature classes into the topology feature dataset.

        Returns a mapping of ``{fc_name: full_fds_path}`` so rules can
        reference feature classes by name.

        Args:
            feature_classes: Source feature class paths.
            fds_path: Destination feature dataset.

        Returns:
            Name-to-path mapping for the imported feature classes.
        """
        fc_map: dict[str, str] = {}
        for src_fc in feature_classes:
            fc_name = Path(src_fc).name
            dest = f"{fds_path}/{fc_name}"

            if arcpy.Exists(dest):
                arcpy.management.Delete(dest)

            arcpy.conversion.ExportFeatures(src_fc, dest)
            fc_map[fc_name] = dest
            logger.debug("Imported '%s' into topology dataset.", fc_name)

        logger.info("Imported %d feature class(es) into '%s'.", len(fc_map), _TOPOLOGY_FDS_NAME)
        return fc_map

    # ------------------------------------------------------------------
    # Step 3 — Create topology
    # ------------------------------------------------------------------

    def _create_topology(self, fds_path: str) -> str:
        """Create an empty topology inside the feature dataset.

        Returns:
            Full path to the topology.
        """
        topo_path = f"{fds_path}/{_TOPOLOGY_NAME}"

        if arcpy.Exists(topo_path):
            arcpy.management.Delete(topo_path)

        arcpy.management.CreateTopology(
            in_dataset=fds_path,
            out_name=_TOPOLOGY_NAME,
            in_cluster_tolerance=_CLUSTER_TOLERANCE,
        )
        logger.info("Created topology '%s'.", _TOPOLOGY_NAME)
        return topo_path

    # ------------------------------------------------------------------
    # Step 4 — Add rules
    # ------------------------------------------------------------------

    def _add_rules(self, topo_path: str, fc_map: dict[str, str]) -> None:
        """Add each user-defined rule to the topology.

        Feature classes referenced by a rule must have been imported in
        Step 2.  Missing references produce a warning and are skipped.

        Args:
            topo_path: Full path to the topology.
            fc_map: Name-to-path mapping from :meth:`_import_feature_classes`.
        """
        for rule in self.rules:
            fc_path = fc_map.get(rule.feature_class)
            if not fc_path:
                logger.warning(
                    "Feature class '%s' not found in the topology dataset — "
                    "skipping rule '%s'.",
                    rule.feature_class, rule.rule,
                )
                continue

            # Add the feature class to the topology (idempotent)
            try:
                arcpy.management.AddFeatureClassToTopology(
                    in_topology=topo_path,
                    in_featureclass=fc_path,
                    xy_rank=1,
                    z_rank=1,
                )
            except arcpy.ExecuteError:
                pass  # Already added — safe to continue

            # Resolve covering class for two-class rules
            covering_path = ""
            if rule.covering_class:
                covering_path = fc_map.get(rule.covering_class, "")
                if not covering_path:
                    logger.warning(
                        "Covering class '%s' not found — skipping rule '%s'.",
                        rule.covering_class, rule.rule,
                    )
                    continue

                try:
                    arcpy.management.AddFeatureClassToTopology(
                        in_topology=topo_path,
                        in_featureclass=covering_path,
                        xy_rank=1,
                        z_rank=1,
                    )
                except arcpy.ExecuteError:
                    pass

            try:
                if covering_path:
                    arcpy.management.AddRuleToTopology(
                        in_topology=topo_path,
                        rule_type=rule.rule,
                        in_featureclass=fc_path,
                        subtype=rule.subtype or "",
                        in_featureclass2=covering_path,
                        subtype2=rule.covering_subtype or "",
                    )
                else:
                    arcpy.management.AddRuleToTopology(
                        in_topology=topo_path,
                        rule_type=rule.rule,
                        in_featureclass=fc_path,
                        subtype=rule.subtype or "",
                    )
                logger.info("Added rule: '%s' on '%s'.", rule.rule, rule.feature_class)
            except arcpy.ExecuteError as exc:
                raise TopologyError(
                    f"Failed to add topology rule '{rule.rule}' on "
                    f"'{rule.feature_class}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Step 5 — Validate
    # ------------------------------------------------------------------

    def _validate_topology(self, topo_path: str) -> None:
        """Validate the topology and log the result.

        Args:
            topo_path: Full path to the topology.

        Raises:
            TopologyError: If validation itself fails to execute.
        """
        try:
            arcpy.management.ValidateTopology(topo_path)
            logger.info("Topology validated successfully.")
        except arcpy.ExecuteError as exc:
            raise TopologyError(f"Topology validation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 6 — Export errors
    # ------------------------------------------------------------------

    def _export_errors(self, topo_path: str) -> Path:
        """Export topology errors to feature classes in a subfolder.

        Creates three feature classes (point, line, polygon errors) in the
        FGDB under a ``TopologyErrors`` feature dataset.

        Args:
            topo_path: Full path to the validated topology.

        Returns:
            Path to the error feature dataset.
        """
        error_fds = str(self.gdb_path / "TopologyErrors")

        if arcpy.Exists(error_fds):
            arcpy.management.Delete(error_fds)

        sr = arcpy.SpatialReference(self.spatial_reference)
        arcpy.management.CreateFeatureDataset(
            out_dataset_path=str(self.gdb_path),
            out_name="TopologyErrors",
            spatial_reference=sr,
        )

        try:
            arcpy.management.ExportTopologyErrors(
                in_topology=topo_path,
                out_path=error_fds,
                out_basename="errors",
            )
        except arcpy.ExecuteError as exc:
            raise TopologyError(f"Failed to export topology errors: {exc}") from exc

        # Count errors for logging
        for suffix in ("_point", "_line", "_poly"):
            error_fc = f"{error_fds}/errors{suffix}"
            if arcpy.Exists(error_fc):
                count = int(arcpy.management.GetCount(error_fc)[0])
                if count > 0:
                    logger.warning("Topology found %d %s error(s).", count, suffix.strip("_"))
                else:
                    logger.info("No %s topology errors.", suffix.strip("_"))

        logger.info("Topology errors exported to '%s'.", error_fds)
        return Path(error_fds)
