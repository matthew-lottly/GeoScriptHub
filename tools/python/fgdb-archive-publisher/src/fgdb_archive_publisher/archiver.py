"""
FGDB Archive Publisher — Archiver
==================================
Exports feature layers from an ArcGIS portal to a local File Geodatabase.

For large datasets and datasets with attachments the archiver uses:

- **Offset-based batching** — queries features in chunks of ``batch_size``
  rows using ``resultOffset`` / ``resultRecordCount`` to avoid memory
  pressure and portal query limits.
- **Thread-pool parallelism** — attachment blobs are downloaded
  concurrently via :class:`concurrent.futures.ThreadPoolExecutor`.

Usage::

    archiver = Archiver(
        portal_url="https://myorg.maps.arcgis.com",
        service_urls=["https://services.arcgis.com/.../FeatureServer/0"],
        output_dir=Path("backups"),
        gdb_name="archive.gdb",
        batch_size=5000,
        max_workers=4,
    )
    gdb_path = archiver.export()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import arcpy  # type: ignore[import-unresolved]
from arcgis.features import FeatureLayer  # type: ignore[import-unresolved]
from arcgis.gis import GIS  # type: ignore[import-unresolved]

from shared.python.exceptions import GeoScriptHubError

logger = logging.getLogger("geoscripthub.fgdb_archive_publisher.archiver")


class ArchiverError(GeoScriptHubError):
    """Raised when the archive/export operation fails."""


class Archiver:
    """Batch-export portal feature layers into a local File Geodatabase.

    Attributes:
        portal_url: ArcGIS Enterprise or ArcGIS Online URL.
        service_urls: Feature layer REST endpoint URLs.
        output_dir: Local directory for the output FGDB.
        gdb_name: File Geodatabase name (e.g. ``"archive.gdb"``).
        batch_size: Features per query batch.
        max_workers: Thread pool size for attachment downloads.
        include_attachments: Download attachments when ``True``.
        spatial_reference: Output spatial reference WKID.
    """

    def __init__(
        self,
        portal_url: str,
        service_urls: list[str],
        output_dir: Path,
        gdb_name: str,
        batch_size: int = 5000,
        max_workers: int = 4,
        include_attachments: bool = True,
        spatial_reference: int = 4326,
    ) -> None:
        self.portal_url = portal_url
        self.service_urls = service_urls
        self.output_dir = Path(output_dir)
        self.gdb_name = gdb_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.include_attachments = include_attachments
        self.spatial_reference = spatial_reference

        self._gis: GIS | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def export(self) -> Path:
        """Run the full export and return the path to the output FGDB.

        Returns:
            Absolute path to the created File Geodatabase.

        Raises:
            ArchiverError: If portal connection or data export fails.
        """
        self._connect()
        gdb_path = self._create_gdb()

        for url in self.service_urls:
            self._export_layer(url, gdb_path)

        logger.info("Archive complete — %d layer(s) saved to '%s'.", len(self.service_urls), gdb_path)
        return gdb_path

    # ------------------------------------------------------------------
    # Portal connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Authenticate to the portal using the active ArcGIS Pro session.

        Falls back to anonymous access if no active session is available.

        Raises:
            ArchiverError: If the connection cannot be established.
        """
        try:
            self._gis = GIS(self.portal_url)
            logger.info("Connected to portal: %s", self.portal_url)
        except Exception as exc:
            raise ArchiverError(
                f"Could not connect to portal '{self.portal_url}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # FGDB creation
    # ------------------------------------------------------------------

    def _create_gdb(self) -> Path:
        """Create the output File Geodatabase.

        Returns:
            Path to the newly created ``.gdb`` directory.

        Raises:
            ArchiverError: If geodatabase creation fails.
        """
        gdb_path = self.output_dir / self.gdb_name

        if gdb_path.exists():
            logger.warning("FGDB already exists — overwriting: '%s'.", gdb_path)
            arcpy.management.Delete(str(gdb_path))

        try:
            arcpy.management.CreateFileGDB(
                out_folder_path=str(self.output_dir),
                out_name=self.gdb_name.replace(".gdb", ""),
            )
            logger.info("Created FGDB: '%s'.", gdb_path)
        except arcpy.ExecuteError as exc:
            raise ArchiverError(f"Failed to create FGDB: {exc}") from exc

        return gdb_path

    # ------------------------------------------------------------------
    # Layer export (batched)
    # ------------------------------------------------------------------

    def _export_layer(self, service_url: str, gdb_path: Path) -> None:
        """Export a single feature layer to the FGDB using batched queries.

        Features are queried in pages of ``batch_size`` rows using
        ``resultOffset``.  Each page is appended to the target feature
        class via ``arcpy.da.InsertCursor``.

        Args:
            service_url: REST endpoint of the feature layer.
            gdb_path: Path to the output FGDB.

        Raises:
            ArchiverError: On query or write failure.
        """
        try:
            fl = FeatureLayer(service_url, gis=self._gis)
            layer_name = fl.properties.get("name", "layer") or "layer"
            layer_name = arcpy.ValidateTableName(layer_name, str(gdb_path))
        except Exception as exc:
            raise ArchiverError(f"Cannot access layer at '{service_url}': {exc}") from exc

        total_count = fl.query(where="1=1", return_count_only=True)
        logger.info(
            "Exporting '%s' — %d feature(s), batch_size=%d.",
            layer_name, total_count, self.batch_size,
        )

        fc_path = str(gdb_path / layer_name)
        fc_created = False
        offset = 0

        while offset < total_count:
            feature_set = fl.query(
                where="1=1",
                out_sr=self.spatial_reference,
                result_offset=offset,
                result_record_count=self.batch_size,
                return_geometry=True,
            )

            if not feature_set.features:
                break

            if not fc_created:
                self._create_feature_class_from_set(feature_set, fc_path, gdb_path)
                fc_created = True

            self._insert_features(feature_set, fc_path)

            offset += self.batch_size
            logger.debug(
                "  '%s' progress: %d / %d features.",
                layer_name, min(offset, total_count), total_count,
            )

        # Attachments
        if self.include_attachments and self._layer_has_attachments(fl):
            self._download_attachments(fl, gdb_path, layer_name)

        logger.info("Finished exporting '%s'.", layer_name)

    # ------------------------------------------------------------------
    # Feature class creation from first batch
    # ------------------------------------------------------------------

    def _create_feature_class_from_set(
        self,
        feature_set: Any,
        fc_path: str,
        gdb_path: Path,
    ) -> None:
        """Create the target feature class by converting the first batch.

        Uses ``arcpy.conversion.JSONToFeatures`` to create the schema
        from the first batch of query results.

        Args:
            feature_set: An ``arcgis.features.FeatureSet``.
            fc_path: Full path to the output feature class.
            gdb_path: Path to the FGDB (workspace).
        """
        geojson_str = json.dumps(feature_set.to_geojson)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False, encoding="utf-8",
        )
        try:
            tmp.write(geojson_str)
            tmp.close()
            arcpy.conversion.JSONToFeatures(tmp.name, fc_path)
        finally:
            os.unlink(tmp.name)

    # ------------------------------------------------------------------
    # Batch insert via InsertCursor
    # ------------------------------------------------------------------

    def _insert_features(self, feature_set: Any, fc_path: str) -> None:
        """Append features from a FeatureSet into an existing feature class.

        Uses ``arcpy.da.InsertCursor`` for fast row insertion.

        Args:
            feature_set: An ``arcgis.features.FeatureSet``.
            fc_path: Full path to the target feature class.
        """
        sdf = feature_set.sdf  # Spatially-enabled DataFrame
        if sdf.empty:
            return

        field_names = [f.name for f in arcpy.ListFields(fc_path) if f.type != "OID"]
        shape_field = "SHAPE@"

        cursor_fields = [
            name for name in field_names if name.upper() not in ("SHAPE", "SHAPE_LENGTH", "SHAPE_AREA")
        ]
        cursor_fields.append(shape_field)

        with arcpy.da.InsertCursor(fc_path, cursor_fields) as cursor:
            for _, row in sdf.iterrows():
                values = []
                for fname in cursor_fields:
                    if fname == shape_field:
                        geom = row.get("SHAPE")
                        values.append(geom)
                    else:
                        values.append(row.get(fname))
                cursor.insertRow(values)

    # ------------------------------------------------------------------
    # Attachment handling (parallelised)
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_has_attachments(fl: FeatureLayer) -> bool:
        """Check whether the feature layer supports attachments."""
        return bool(fl.properties.get("hasAttachments", False))

    def _download_attachments(
        self,
        fl: FeatureLayer,
        gdb_path: Path,
        layer_name: str,
    ) -> None:
        """Download all attachments for every feature using a thread pool.

        Attachments are stored in a folder alongside the FGDB named
        ``<layer_name>_attachments/``.

        Args:
            fl: The source feature layer.
            gdb_path: FGDB path (used to resolve the attachment directory).
            layer_name: Name of the feature class.
        """
        attachment_dir = gdb_path.parent / f"{layer_name}_attachments"
        attachment_dir.mkdir(parents=True, exist_ok=True)

        oid_result = fl.query(where="1=1", return_ids_only=True)
        oid_list: list[int] = oid_result.get("objectIds", []) if isinstance(oid_result, dict) else []

        if not oid_list:
            logger.info("No features with OIDs found — skipping attachments.")
            return

        logger.info(
            "Downloading attachments for '%s' (%d features, %d workers).",
            layer_name, len(oid_list), self.max_workers,
        )

        def _download_one(oid: int) -> int:
            """Download all attachments for a single feature. Returns count."""
            try:
                att_list = fl.attachments.get_list(oid)
            except Exception:
                return 0

            count = 0
            for att in att_list:
                att_id = att["id"]
                att_name = att.get("name", f"attachment_{att_id}")
                dest = attachment_dir / str(oid)
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    fl.attachments.download(oid=oid, attachment_id=att_id, save_path=str(dest))
                    count += 1
                except Exception as exc:
                    logger.warning("Failed to download attachment %d/%d: %s", oid, att_id, exc)
            return count

        total_downloaded = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_download_one, oid): oid for oid in oid_list}
            for future in as_completed(futures):
                total_downloaded += future.result()

        logger.info("Downloaded %d attachment(s) to '%s'.", total_downloaded, attachment_dir)
