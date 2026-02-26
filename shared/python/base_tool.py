"""
GeoScriptHub — Shared Base Tool
================================
Abstract base class that every GeoScriptHub Python tool inherits from.

Design Pattern:
    Template Method — the public ``run()`` method defines a fixed
    pipeline (validate → process → report) that subclasses fill in
    by implementing the abstract methods ``validate_inputs`` and
    ``process``.

Usage:
    Do NOT instantiate this class directly.  Subclass it and implement
    the two abstract methods::

        from shared.python.base_tool import GeoTool

        class MyTool(GeoTool):
            def validate_inputs(self) -> None:
                ...
            def process(self) -> None:
                ...
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level logger — each tool gets its own child logger via
#   logging.getLogger(__name__) inside its own module.
# ---------------------------------------------------------------------------
logger = logging.getLogger("geoscripthub")


class GeoTool(ABC):
    """Abstract base class for all GeoScriptHub geospatial tools.

    Every concrete tool must inherit from this class and implement
    :meth:`validate_inputs` and :meth:`process`.  Calling :meth:`run`
    executes the full pipeline in the correct order.

    Attributes:
        input_path: Absolute path to the primary input file or directory.
        output_path: Absolute path where output will be written.
        verbose: When ``True`` the tool logs DEBUG-level messages in
            addition to INFO/WARNING/ERROR.

    Example::

        tool = BatchCoordinateTransformer(
            input_path=Path("coords.csv"),
            output_path=Path("coords_wgs84.csv"),
            from_crs="EPSG:32614",
            to_crs="EPSG:4326",
        )
        tool.run()
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialise the base tool.

        Args:
            input_path: Path to the primary input file or directory.
                        <!-- PLACEHOLDER: replace with the actual path
                             to your input data, e.g. Path("data/my_file.shp") -->
            output_path: Path where the tool will write its output.
                         <!-- PLACEHOLDER: replace with your desired
                              output path, e.g. Path("output/result.geojson") -->
            verbose: Set to ``True`` to enable debug-level console
                     logging during the run.  Defaults to ``False``.
        """
        self.input_path: Path = Path(input_path)
        self.output_path: Path = Path(output_path)
        self.verbose: bool = verbose

        self._configure_logging()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def validate_inputs(self) -> None:
        """Validate all inputs before processing begins.

        Subclasses should raise :class:`~shared.python.exceptions.InputValidationError`
        (or a subclass) when any input condition is not satisfied.

        Raises:
            InputValidationError: If a required file is missing, a
                column does not exist, or a CRS string is invalid.
        """

    @abstractmethod
    def process(self) -> None:
        """Execute the core geospatial processing logic.

        This method is called by :meth:`run` after :meth:`validate_inputs`
        has succeeded.  Any exception raised here will propagate up
        through :meth:`run`.
        """

    # ------------------------------------------------------------------
    # Template method — the public API callers use
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full tool pipeline.

        Runs the steps in order:

        1. :meth:`validate_inputs` — verify all preconditions.
        2. :meth:`process` — perform the geospatial work.
        3. :meth:`_report_success` — log the elapsed time and output path.

        Raises:
            Any exception raised by ``validate_inputs`` or ``process``
            propagates unchanged so callers can handle it appropriately.
        """
        logger.info("Starting %s", self.__class__.__name__)
        start = time.perf_counter()

        self.validate_inputs()
        self.process()

        elapsed = time.perf_counter() - start
        self._report_success(elapsed)

    # ------------------------------------------------------------------
    # Protected helpers — subclasses may override if needed
    # ------------------------------------------------------------------

    def _report_success(self, elapsed: float) -> None:
        """Log a success message with the elapsed time and output path.

        Args:
            elapsed: Seconds taken for the full run, as returned by
                     ``time.perf_counter()``.
        """
        logger.info(
            "%s completed in %.2fs → %s",
            self.__class__.__name__,
            elapsed,
            self.output_path,
        )

    def _configure_logging(self) -> None:
        """Set up console logging for this tool instance.

        Attaches a :class:`logging.StreamHandler` to the root
        ``geoscripthub`` logger if no handlers are already present.
        Uses DEBUG level when ``self.verbose`` is ``True``, otherwise INFO.
        """
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_path={self.input_path!r}, "
            f"output_path={self.output_path!r})"
        )
