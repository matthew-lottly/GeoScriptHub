"""
Shapefile Health Checker — Core Module
========================================
Runs six validation checks against a vector file (Shapefile, GeoJSON,
GeoPackage) and collects results into a structured :class:`HealthReport`.

Classes:
    CheckResult         Immutable result for a single check.
    HealthReport        Aggregated results for all checks on one file.
    CheckStrategy       Abstract base for individual check implementations.
    NullGeometryCheck   Detects features with null / empty geometries.
    SelfIntersectionCheck  Detects geometrically self-intersecting polygons.
    DuplicateFeaturesCheck Detects rows with completely duplicate geometries.
    CRSPresenceCheck    Confirms a CRS is defined on the dataset.
    EncodingCheck       Reads every attribute value to catch encoding errors.
    ExtentSanityCheck   Flags geometries that fall outside the CRS's valid bounds.
    ShapefileHealthChecker  Primary tool class (inherits GeoTool).

Usage::

    from pathlib import Path
    from src.shapefile_health_checker.checker import ShapefileHealthChecker

    tool = ShapefileHealthChecker(
        input_path=Path("data/parcels.shp"),
        output_path=Path("output/report.md"),
        report_format="markdown",
    )
    tool.run()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import geopandas as gpd

from shared.python.base_tool import GeoTool
from shared.python.exceptions import InputValidationError, OutputWriteError
from shared.python.validators import Validators

logger = logging.getLogger("geoscripthub.shapefile_health_checker")


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class CheckStatus(Enum):
    """Status of a single health check."""

    PASSED = auto()
    WARNING = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass(frozen=True)
class CheckResult:
    """Immutable result for one health check.

    Attributes:
        check_name: Human-readable name of the check.
        status: One of :class:`CheckStatus`.
        details: Prose description of findings.
        affected_rows: Indices of affected rows (empty if none).
    """

    check_name: str
    status: CheckStatus
    details: str
    affected_rows: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` if status is PASSED."""
        return self.status == CheckStatus.PASSED

    @property
    def status_label(self) -> str:
        """Return an emoji + label string for display."""
        labels = {
            CheckStatus.PASSED: "✅ PASSED",
            CheckStatus.WARNING: "⚠️  WARNING",
            CheckStatus.FAILED: "❌ FAILED",
            CheckStatus.SKIPPED: "⏭️  SKIPPED",
        }
        return labels[self.status]


@dataclass
class HealthReport:
    """Aggregated health check results for a single vector file.

    Attributes:
        file_path: Path to the file that was checked.
        crs: CRS string found on the dataset, or ``"None"`` if absent.
        feature_count: Total number of features in the dataset.
        results: Ordered list of :class:`CheckResult` objects.
    """

    file_path: Path
    crs: str
    feature_count: int
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        """Number of checks that passed."""
        return sum(1 for r in self.results if r.status == CheckStatus.PASSED)

    @property
    def failed_count(self) -> int:
        """Number of checks that failed."""
        return sum(1 for r in self.results if r.status == CheckStatus.FAILED)

    @property
    def warning_count(self) -> int:
        """Number of checks that produced a warning."""
        return sum(1 for r in self.results if r.status == CheckStatus.WARNING)

    @property
    def overall_status(self) -> CheckStatus:
        """Highest severity status across all checks."""
        if self.failed_count:
            return CheckStatus.FAILED
        if self.warning_count:
            return CheckStatus.WARNING
        return CheckStatus.PASSED


# ---------------------------------------------------------------------------
# Check strategy ABC + implementations
# ---------------------------------------------------------------------------


class CheckStrategy(ABC):
    """Abstract base for a single health check.

    Subclasses implement :meth:`run` to inspect a GeoDataFrame and return
    a :class:`CheckResult`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this check."""

    @abstractmethod
    def run(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Execute the check against *gdf* and return the result.

        Args:
            gdf: The GeoDataFrame to inspect.

        Returns:
            A populated :class:`CheckResult`.
        """


class NullGeometryCheck(CheckStrategy):
    """Detect features with null or empty geometries."""

    @property
    def name(self) -> str:
        return "Null / Empty Geometry"

    def run(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Flag any rows where geometry is null or an empty geometry object."""
        null_mask = gdf.geometry.is_empty | gdf.geometry.isna()
        bad_indices = gdf[null_mask].index.tolist()

        if bad_indices:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.FAILED,
                details=f"{len(bad_indices)} feature(s) have null or empty geometries.",
                affected_rows=bad_indices,
            )
        return CheckResult(check_name=self.name, status=CheckStatus.PASSED, details="All geometries are non-null.")


class SelfIntersectionCheck(CheckStrategy):
    """Detect polygon/multipolygon features that are not valid (e.g. self-intersecting)."""

    @property
    def name(self) -> str:
        return "Self-Intersection (Geometry Validity)"

    def run(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Use Shapely's ``is_valid`` to flag invalid geometries."""
        invalid_mask = ~gdf.geometry.is_valid
        bad_indices = gdf[invalid_mask].index.tolist()

        if bad_indices:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.FAILED,
                details=f"{len(bad_indices)} feature(s) have invalid geometries (self-intersections or other topology errors).",
                affected_rows=bad_indices,
            )
        return CheckResult(check_name=self.name, status=CheckStatus.PASSED, details="All geometries are topologically valid.")


class DuplicateFeaturesCheck(CheckStrategy):
    """Detect rows with completely duplicate WKT geometries."""

    @property
    def name(self) -> str:
        return "Duplicate Features"

    def run(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Compare WKT representations to find exact geometry duplicates."""
        wkt_series = gdf.geometry.apply(lambda g: g.wkt if g else None)
        duplicated = wkt_series.duplicated(keep=False)
        bad_indices = gdf[duplicated].index.tolist()

        if bad_indices:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.WARNING,
                details=f"{len(bad_indices)} feature(s) share identical geometries with at least one other feature.",
                affected_rows=bad_indices,
            )
        return CheckResult(check_name=self.name, status=CheckStatus.PASSED, details="No duplicate geometries found.")


class CRSPresenceCheck(CheckStrategy):
    """Confirm the dataset has a defined CRS."""

    @property
    def name(self) -> str:
        return "CRS Presence"

    def run(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Fail if ``gdf.crs`` is ``None``."""
        if gdf.crs is None:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.FAILED,
                details="Dataset has no defined CRS.  Assign one with gdf.set_crs().",
            )
        return CheckResult(
            check_name=self.name,
            status=CheckStatus.PASSED,
            details=f"CRS is defined: {gdf.crs.to_epsg() or gdf.crs.name}",
        )


class EncodingCheck(CheckStrategy):
    """Read every attribute value to catch encoding / decoding errors.

    When GeoPandas successfully loads a file, encoding errors have typically
    already been caught.  This check attempts to re-encode all string values
    to UTF-8 as an extra safety net.
    """

    @property
    def name(self) -> str:
        return "Attribute Encoding"

    def run(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Attempt to UTF-8 encode all string attribute values."""
        bad_cols: list[str] = []
        for col in gdf.columns:
            if col == gdf.geometry.name:
                continue
            try:
                gdf[col].astype(str).apply(lambda v: v.encode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                bad_cols.append(col)

        if bad_cols:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.WARNING,
                details=f"Encoding issues found in column(s): {', '.join(bad_cols)}",
            )
        return CheckResult(check_name=self.name, status=CheckStatus.PASSED, details="All attribute values encode cleanly to UTF-8.")


class ExtentSanityCheck(CheckStrategy):
    """Flag geometries whose coordinates fall outside standard world bounds.

    Uses ±180° longitude and ±90° latitude as the sanity window.  Reprojects
    to WGS84 before checking if the dataset CRS is defined.
    """

    @property
    def name(self) -> str:
        return "Extent Sanity (World Bounds)"

    def run(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check that all geometries fall within [-180, -90, 180, 90]."""
        try:
            check_gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
        except Exception:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.SKIPPED,
                details="Skipped: could not reproject to WGS84 for extent check.",
            )

        bounds = check_gdf.geometry.bounds
        out_of_bounds = (
            (bounds["minx"] < -180)
            | (bounds["maxx"] > 180)
            | (bounds["miny"] < -90)
            | (bounds["maxy"] > 90)
        )
        bad_indices = gdf[out_of_bounds].index.tolist()

        if bad_indices:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.FAILED,
                details=f"{len(bad_indices)} feature(s) have coordinates outside [-180, -90, 180, 90].",
                affected_rows=bad_indices,
            )
        return CheckResult(check_name=self.name, status=CheckStatus.PASSED, details="All features fall within world bounds.")


# ---------------------------------------------------------------------------
# Default check suite
# ---------------------------------------------------------------------------

DEFAULT_CHECKS: list[CheckStrategy] = [
    CRSPresenceCheck(),
    NullGeometryCheck(),
    SelfIntersectionCheck(),
    DuplicateFeaturesCheck(),
    EncodingCheck(),
    ExtentSanityCheck(),
]


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class ShapefileHealthChecker(GeoTool):
    """Validate a vector file against a configurable suite of health checks.

    Inherits the Template Method pipeline from :class:`~shared.python.GeoTool`:
    ``validate_inputs`` → ``process`` → ``_report_success``.

    Args:
        input_path: Path to the vector file to check.
        output_path: Path where the report will be written.
        report_format: Output format for the report: ``"markdown"`` or
                       ``"html"``.
        checks: List of :class:`CheckStrategy` instances to run.  Defaults
                to the six built-in checks.  Pass a custom list to skip or
                add checks.
        verbose: Enable DEBUG-level logging.

    Example::

        ShapefileHealthChecker(
            input_path=Path("data/parcels.shp"),
            output_path=Path("output/report.md"),
        ).run()
    """

    SUPPORTED_EXTENSIONS = [".shp", ".geojson", ".json", ".gpkg"]

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        report_format: Literal["markdown", "html"] = "markdown",
        checks: list[CheckStrategy] | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__(input_path, output_path, verbose=verbose)
        self.report_format: Literal["markdown", "html"] = report_format
        self.checks: list[CheckStrategy] = checks if checks is not None else DEFAULT_CHECKS

        self._report: HealthReport | None = None

    # ------------------------------------------------------------------
    # GeoTool abstract method implementations
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate the input file and output path before processing.

        Raises:
            InputValidationError: If the file does not exist, has an
                unsupported extension, or the output dir is not writable.
        """
        Validators.assert_file_exists(self.input_path)
        Validators.assert_supported_extension(self.input_path, self.SUPPORTED_EXTENSIONS)
        Validators.assert_output_dir_writable(self.output_path)
        logger.debug("Input validated: %s", self.input_path)

    def process(self) -> None:
        """Load the vector file, run all checks, and write the report.

        Raises:
            OutputWriteError: If writing the report to disk fails.
        """
        logger.info("Loading vector file: %s", self.input_path)
        gdf = gpd.read_file(self.input_path)

        # Build the HealthReport
        crs_string = str(gdf.crs) if gdf.crs else "None"
        report = HealthReport(
            file_path=self.input_path,
            crs=crs_string,
            feature_count=len(gdf),
        )

        # Run every check strategy
        for check in self.checks:
            logger.debug("Running check: %s", check.name)
            result = check.run(gdf)
            report.results.append(result)
            logger.debug("  %s — %s", check.name, result.status_label)

        self._report = report

        # Delegate writing to the appropriate reporter
        from src.shapefile_health_checker.reporter import (  # noqa: PLC0415
            HtmlReporter,
            MarkdownReporter,
        )

        reporter = (
            HtmlReporter(report, self.output_path)
            if self.report_format == "html"
            else MarkdownReporter(report, self.output_path)
        )

        try:
            reporter.write()
        except OSError as exc:
            raise OutputWriteError(str(self.output_path), str(exc)) from exc

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def report(self) -> HealthReport | None:
        """The :class:`HealthReport` from the last :meth:`run` call, or ``None``."""
        return self._report
