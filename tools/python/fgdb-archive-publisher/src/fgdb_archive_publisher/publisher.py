"""
FGDB Archive Publisher — Publisher
===================================
Republishes the cleaned and validated File Geodatabase back to an
ArcGIS Enterprise or ArcGIS Online portal as a hosted feature service.

Two strategies are supported:

1. **Service Definition (SD) workflow** — uses ``arcpy.sharing`` to create
   a service definition, stage it, then upload.  This is the most
   reliable method and supports overwriting existing services.

2. **ArcGIS API for Python** — uses ``arcgis.features`` to overwrite an
   existing hosted feature layer directly from the FGDB.

The publisher defaults to the SD workflow and falls back to the API
approach when ``arcpy.sharing`` is unavailable.

Usage::

    publisher = Publisher(
        gdb_path=Path("backup.gdb"),
        config=RepublishConfig(
            target_portal="https://myorg.maps.arcgis.com",
            service_name="Parcels_Clean",
            folder="Published",
            overwrite=True,
        ),
    )
    publisher.publish()
"""

from __future__ import annotations

import logging
from pathlib import Path

import arcpy  # type: ignore[import-unresolved]
from arcgis.gis import GIS  # type: ignore[import-unresolved]

from shared.python.exceptions import GeoScriptHubError

from src.fgdb_archive_publisher.pipeline import RepublishConfig

logger = logging.getLogger("geoscripthub.fgdb_archive_publisher.publisher")


class PublishError(GeoScriptHubError):
    """Raised when publishing to the portal fails."""


class Publisher:
    """Publish a File Geodatabase to an ArcGIS portal.

    Attributes:
        gdb_path: Path to the cleaned FGDB to publish.
        config: Republish target configuration.
    """

    def __init__(self, gdb_path: Path, config: RepublishConfig) -> None:
        self.gdb_path = gdb_path
        self.config = config

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def publish(self) -> str:
        """Publish the FGDB and return the service URL.

        Returns:
            The REST URL of the published (or overwritten) feature service.

        Raises:
            PublishError: On any publishing failure.
        """
        logger.info(
            "Publishing '%s' to '%s' (folder='%s', overwrite=%s).",
            self.gdb_path.name,
            self.config.target_portal,
            self.config.folder,
            self.config.overwrite,
        )

        try:
            return self._publish_via_sd()
        except Exception as sd_exc:
            logger.warning("SD workflow failed (%s) — trying API fallback.", sd_exc)
            try:
                return self._publish_via_api()
            except Exception as api_exc:
                raise PublishError(
                    f"Publishing failed.  SD error: {sd_exc}  |  API error: {api_exc}"
                ) from api_exc

    # ------------------------------------------------------------------
    # Strategy 1 — Service Definition
    # ------------------------------------------------------------------

    def _publish_via_sd(self) -> str:
        """Create a service definition from the FGDB and upload it.

        This method creates a temporary ``.sddraft`` and ``.sd`` file,
        stages the SD, and publishes to the portal.

        Returns:
            REST URL of the published service.
        """
        import tempfile

        sd_dir = Path(tempfile.mkdtemp(prefix="geoscripthub_sd_"))
        service_name = self.config.service_name or self.gdb_path.stem
        sddraft_path = sd_dir / f"{service_name}.sddraft"
        sd_path = sd_dir / f"{service_name}.sd"

        # Sign in to the portal in ArcGIS Pro
        arcpy.SignInToPortal(self.config.target_portal)

        # List feature classes to include
        arcpy.env.workspace = str(self.gdb_path)
        feature_classes = arcpy.ListFeatureClasses() or []

        # Also include FCs inside feature datasets
        for fds in arcpy.ListDatasets(feature_type="Feature") or []:
            for fc in arcpy.ListFeatureClasses(feature_dataset=fds) or []:
                feature_classes.append(f"{fds}/{fc}")

        if not feature_classes:
            raise PublishError("No feature classes found in the FGDB to publish.")

        # Build layers for the sharing draft
        layers = []
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        mp = aprx.listMaps()[0] if aprx.listMaps() else None

        if mp is None:
            raise PublishError(
                "No map found in the current ArcGIS Pro project.  "
                "Open a project with at least one map before publishing."
            )

        # Add each feature class as a layer
        for fc_name in feature_classes:
            fc_full = str(self.gdb_path / fc_name)
            mp.addDataFromPath(fc_full)

        layers = mp.listLayers()

        # Create the sharing draft
        sharing_draft = mp.getWebLayerSharingDraft(
            server_type="HOSTING_SERVER",
            service_type="FEATURE",
            service_name=service_name,
            layers_and_tables=layers,
        )
        sharing_draft.overwriteExistingService = self.config.overwrite
        sharing_draft.portalFolder = self.config.folder
        sharing_draft.exportToSDDraft(str(sddraft_path))

        # Stage and upload
        arcpy.server.StageService(str(sddraft_path), str(sd_path))
        arcpy.server.UploadServiceDefinition(
            str(sd_path),
            "HOSTING_SERVER",
        )

        service_url = f"{self.config.target_portal}/rest/services/{service_name}/FeatureServer"
        logger.info("Published via SD workflow: %s", service_url)
        return service_url

    # ------------------------------------------------------------------
    # Strategy 2 — ArcGIS API overwrite
    # ------------------------------------------------------------------

    def _publish_via_api(self) -> str:
        """Overwrite an existing feature service using the ArcGIS API.

        Connects to the portal, searches for the existing service by
        name, and overwrites it with the FGDB contents.

        Returns:
            REST URL of the overwritten service.
        """
        gis = GIS(self.config.target_portal)
        service_name = self.config.service_name or self.gdb_path.stem

        # Search for the existing item
        query = f'title:"{service_name}" AND type:"Feature Service" AND owner:{gis.users.me.username}'
        results = gis.content.search(query, max_items=5)

        target_item = None
        for item in results:
            if item.title == service_name:
                target_item = item
                break

        if target_item is None and self.config.overwrite:
            raise PublishError(
                f"Cannot overwrite — no existing service named '{service_name}' "
                f"found for user '{gis.users.me.username}'."
            )

        if target_item and self.config.overwrite:
            # Upload the FGDB as a zip and overwrite
            gdb_zip = self._zip_gdb()
            target_item.update(data=str(gdb_zip))
            flc = target_item.publish(overwrite=True)
            service_url = flc.url
            logger.info("Overwritten via API: %s", service_url)
            return service_url

        # Publish as a new item
        gdb_zip = self._zip_gdb()
        uploaded = gis.content.add(
            item_properties={
                "title": service_name,
                "type": "File Geodatabase",
                "tags": "GeoScriptHub, archive, backup",
            },
            data=str(gdb_zip),
            folder=self.config.folder,
        )
        published = uploaded.publish()
        logger.info("Published via API: %s", published.url)
        return published.url

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zip_gdb(self) -> Path:
        """Zip the FGDB for upload via the ArcGIS API.

        Returns:
            Path to the ``.zip`` file.
        """
        import shutil

        zip_path = self.gdb_path.parent / self.gdb_path.stem
        shutil.make_archive(str(zip_path), "zip", str(self.gdb_path.parent), self.gdb_path.name)
        result = zip_path.with_suffix(".zip")
        logger.debug("Zipped FGDB to '%s'.", result)
        return result
