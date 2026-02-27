"""
FGDB Archive Publisher
======================
Automated pipeline for backing up ArcGIS portal data to a File
Geodatabase, cleaning schemas, validating with topology rules, and
republishing to a portal.

Public API::

    from fgdb_archive_publisher.pipeline import ArchivePipeline, PipelineConfig

Modules:
    pipeline        — Main orchestrator (``GeoTool`` subclass)
    archiver        — Batch export from portal to FGDB
    schema_manager  — Field cleanup, domains, and schema modifications
    topology_checker— User-defined topology rule validation
    publisher       — Republish cleaned data back to portal
"""

from __future__ import annotations
