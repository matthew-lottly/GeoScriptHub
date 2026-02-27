"""
GeoScriptHub — Shared Input Validators
=======================================
Static utility methods used across every GeoScriptHub Python tool to
validate common preconditions before processing begins.

All methods raise an appropriate exception from
:mod:`shared.python.exceptions` rather than returning booleans — this
makes ``validate_inputs`` implementations in each tool simple and
readable::

    class MyTool(GeoTool):
        def validate_inputs(self) -> None:
            Validators.assert_file_exists(self.input_path)
            Validators.assert_supported_extension(self.input_path, [".csv"])
            Validators.assert_crs_valid(self.target_crs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

# Lazy imports for heavy libraries so tools that do not use them avoid
# the import cost at startup.
#   pyproj → assert_crs_valid
#   pandas → assert_columns_exist

from shared.python.exceptions import (
    BandIndexError,
    ColumnNotFoundError,
    CRSError,
    InputValidationError,
    OutputWriteError,
)


class Validators:
    """Collection of static precondition checks shared across all tools.

    All methods are ``@staticmethod`` — this class is never instantiated.
    It exists purely as a logical namespace.
    """

    # ------------------------------------------------------------------
    # File-system checks
    # ------------------------------------------------------------------

    @staticmethod
    def assert_file_exists(path: Path) -> None:
        """Assert that *path* points to an existing regular file.

        Args:
            path: Path object to check.

        Raises:
            InputValidationError: If *path* does not exist or is a
                directory rather than a file.

        Example::

            Validators.assert_file_exists(Path("data/input.shp"))
        """
        path = Path(path)
        if not path.exists():
            raise InputValidationError(
                f"Input file not found: '{path}'. "
                "Check that the path is correct and the file exists."
            )
        if path.is_dir():
            raise InputValidationError(
                f"Expected a file but got a directory: '{path}'."
            )

    @staticmethod
    def assert_directory_exists(path: Path) -> None:
        """Assert that *path* is an existing directory.

        Args:
            path: Path object to check.

        Raises:
            InputValidationError: If *path* does not exist or is not a
                directory.
        """
        path = Path(path)
        if not path.exists():
            raise InputValidationError(
                f"Input directory not found: '{path}'."
            )
        if not path.is_dir():
            raise InputValidationError(
                f"Expected a directory but got a file: '{path}'."
            )

    @staticmethod
    def assert_output_dir_writable(output_path: Path) -> None:
        """Assert that the parent directory of *output_path* is writable.

        Creates the parent directory (and any missing parents) if it does
        not yet exist, so authors never have to pre-create output dirs.

        Args:
            output_path: Intended output file path.  The parent directory
                         is created if absent.

        Raises:
            OutputWriteError: If the parent directory cannot be created
                or is not writable.
        """
        parent = Path(output_path).parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputWriteError(str(output_path), str(exc)) from exc

    @staticmethod
    def assert_supported_extension(path: Path, extensions: Sequence[str]) -> None:
        """Assert that *path* has one of the allowed file extensions.

        Args:
            path: File path to check.
            extensions: Sequence of allowed extensions, each starting with
                        a dot (e.g. ``[".shp", ".geojson", ".gpkg"]``).

        Raises:
            InputValidationError: If the file extension is not in
                *extensions*.

        Example::

            Validators.assert_supported_extension(
                Path("data/parcels.gpkg"),
                [".shp", ".geojson", ".gpkg"],
            )
        """
        path = Path(path)
        suffix = path.suffix.lower()
        allowed = [ext.lower() for ext in extensions]
        if suffix not in allowed:
            raise InputValidationError(
                f"Unsupported file extension '{suffix}' for '{path.name}'. "
                f"Accepted extensions: {', '.join(allowed)}"
            )

    # ------------------------------------------------------------------
    # CRS / projection checks
    # ------------------------------------------------------------------

    @staticmethod
    def assert_crs_valid(crs_string: str) -> None:
        """Assert that *crs_string* can be parsed as a valid CRS.

        Uses :mod:`pyproj` to attempt parsing.  Accepts EPSG codes
        (``"EPSG:4326"``), PROJ strings, and WKT strings.

        Args:
            crs_string: The CRS identifier to validate (e.g. ``"EPSG:4326"``).

        Raises:
            CRSError: If *crs_string* is not recognised by pyproj.

        Example::

            Validators.assert_crs_valid("EPSG:4326")
        """
        try:
            from pyproj import CRS  # noqa: PLC0415

            CRS.from_user_input(crs_string)
        except Exception as exc:
            raise CRSError(crs_string) from exc

    # ------------------------------------------------------------------
    # Tabular data checks
    # ------------------------------------------------------------------

    @staticmethod
    def assert_columns_exist(
        df: object,  # pandas DataFrame — typed loosely to avoid hard dep
        required_columns: Sequence[str],
    ) -> None:
        """Assert that all *required_columns* are present in *df*.

        Args:
            df: A ``pandas.DataFrame`` (typed as ``object`` here to avoid
                importing pandas at module load time).
            required_columns: List of column names that must be present.

        Raises:
            ColumnNotFoundError: On the first missing column found.

        Example::

            Validators.assert_columns_exist(df, ["latitude", "longitude"])
        """
        available = list(df.columns)  # type: ignore[union-attr]
        for col in required_columns:
            if col not in available:
                raise ColumnNotFoundError(col, available)

    # ------------------------------------------------------------------
    # Raster checks
    # ------------------------------------------------------------------

    @staticmethod
    def assert_band_index_valid(band_index: int, total_bands: int) -> None:
        """Assert that *band_index* is within the valid range for a raster.

        Args:
            band_index: 1-based band index requested by the user.
            total_bands: Total number of bands in the raster file.

        Raises:
            BandIndexError: If *band_index* is less than 1 or exceeds
                *total_bands*.

        Example::

            Validators.assert_band_index_valid(band_index=3, total_bands=4)
        """
        if band_index < 1 or band_index > total_bands:
            raise BandIndexError(band_index, total_bands)

    @staticmethod
    def assert_raster_shapes_match(
        shape_a: tuple[int, int],
        shape_b: tuple[int, int],
        label_a: str = "Band A",
        label_b: str = "Band B",
    ) -> None:
        """Assert that two raster arrays have identical (rows, cols) shapes.

        This is required before any pixel-wise arithmetic (e.g. NDVI).

        Args:
            shape_a: ``(rows, cols)`` of the first raster array.
            shape_b: ``(rows, cols)`` of the second raster array.
            label_a: Human-readable name for the first band (used in the
                     error message).
            label_b: Human-readable name for the second band.

        Raises:
            InputValidationError: If the shapes do not match.

        Example::

            Validators.assert_raster_shapes_match(
                nir_array.shape, red_array.shape, "NIR", "Red"
            )
        """
        if shape_a != shape_b:
            raise InputValidationError(
                f"Raster shape mismatch: {label_a} is {shape_a} but "
                f"{label_b} is {shape_b}. "
                "All input bands must have identical dimensions."
            )
