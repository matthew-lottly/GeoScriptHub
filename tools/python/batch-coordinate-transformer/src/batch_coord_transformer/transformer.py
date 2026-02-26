"""
Batch Coordinate Transformer — Core Module
===========================================
Provides the :class:`CoordinateTransformer` class, which reads a CSV (or
GeoJSON) file containing coordinate columns, reprojects every point from a
source CRS to a target CRS, and writes the result to a new file.

Classes:
    CoordinateTransformer   Primary tool class (inherits GeoTool).

Typical usage::

    from pathlib import Path
    from src.batch_coord_transformer.transformer import CoordinateTransformer

    tool = CoordinateTransformer(
        input_path=Path("data/coordinates.csv"),
        output_path=Path("output/coordinates_wgs84.csv"),
        from_crs="EPSG:32614",
        to_crs="EPSG:4326",
        lon_col="easting",
        lat_col="northing",
    )
    tool.run()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import pyproj

# ---------------------------------------------------------------------------
# Shared GeoScriptHub foundation — add repo root to PYTHONPATH to resolve.
# ---------------------------------------------------------------------------
from shared.python.base_tool import GeoTool
from shared.python.exceptions import InputValidationError, OutputWriteError
from shared.python.validators import Validators

logger = logging.getLogger("geoscripthub.batch_coord_transformer")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformResult:
    """Immutable container for a completed transformation run.

    Attributes:
        rows_processed: Total number of coordinate rows transformed.
        rows_skipped: Rows skipped due to null or non-numeric coordinates.
        from_crs: Source CRS string as provided by the user.
        to_crs: Target CRS string as provided by the user.
        output_path: Path where the transformed file was written.
    """

    rows_processed: int
    rows_skipped: int
    from_crs: str
    to_crs: str
    output_path: Path

    def summary(self) -> str:
        """Return a human-readable summary string for logging or display."""
        return (
            f"Transformed {self.rows_processed} rows "
            f"({self.rows_skipped} skipped) | "
            f"{self.from_crs} → {self.to_crs} | "
            f"Output: {self.output_path}"
        )


@dataclass
class TransformerConfig:
    """Configuration bundle for :class:`CoordinateTransformer`.

    Attributes:
        from_crs: Source CRS string.
                  <!-- PLACEHOLDER: set to the CRS your input data is in,
                       e.g. "EPSG:32614" for UTM Zone 14N,
                            "EPSG:4269" for NAD83,
                            "EPSG:4326" for WGS84 -->
        to_crs: Target CRS string.
                <!-- PLACEHOLDER: set to the CRS you want the output in,
                     e.g. "EPSG:4326" for WGS84 (lat/lon) -->
        lon_col: Name of the column containing X / longitude / easting values.
                 <!-- PLACEHOLDER: replace with the actual column name in
                      your CSV, e.g. "longitude", "x", "easting" -->
        lat_col: Name of the column containing Y / latitude / northing values.
                 <!-- PLACEHOLDER: replace with the actual column name in
                      your CSV, e.g. "latitude", "y", "northing" -->
        output_format: Output file format.  One of ``"csv"`` or ``"geojson"``.
        always_xy: When ``True``, enforce (longitude, latitude) axis order
                   regardless of CRS definition.  Recommended to keep as
                   ``True`` for predictable results.
    """

    from_crs: str
    to_crs: str
    lon_col: str = "longitude"
    lat_col: str = "latitude"
    output_format: Literal["csv", "geojson"] = "csv"
    always_xy: bool = True
    extra_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class CoordinateTransformer(GeoTool):
    """Reproject coordinate pairs in a CSV (or GeoJSON) to a new CRS.

    Inherits the Template Method pipeline from :class:`~shared.python.GeoTool`:
    ``validate_inputs`` → ``process`` → ``_report_success``.

    Supports CSV inputs with arbitrary coordinate column names and optional
    passthrough of additional attribute columns.  Outputs to CSV or GeoJSON.

    Args:
        input_path: Path to the input CSV file.
                    <!-- PLACEHOLDER: path to your input file,
                         e.g. Path("data/survey_points.csv") -->
        output_path: Path where the reprojected output will be written.
                     <!-- PLACEHOLDER: desired output path,
                          e.g. Path("output/survey_points_wgs84.csv") -->
        config: A :class:`TransformerConfig` instance containing CRS strings,
                column names, and output format settings.
        verbose: Enable DEBUG-level logging.  Defaults to ``False``.

    Example::

        from pathlib import Path
        from src.batch_coord_transformer.transformer import (
            CoordinateTransformer,
            TransformerConfig,
        )

        cfg = TransformerConfig(
            from_crs="EPSG:32614",
            to_crs="EPSG:4326",
            lon_col="easting",
            lat_col="northing",
            output_format="geojson",
        )
        CoordinateTransformer(
            Path("data/points.csv"), Path("output/points.geojson"), cfg
        ).run()
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        config: TransformerConfig,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__(input_path, output_path, verbose=verbose)
        self.config: TransformerConfig = config

        # Set after validate_inputs() — populated in process()
        self._result: TransformResult | None = None

    # ------------------------------------------------------------------
    # GeoTool abstract method implementations
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate the input file and configuration before processing.

        Checks:
        - Input file exists and has a ``.csv`` extension.
        - ``from_crs`` and ``to_crs`` are valid parseable CRS strings.
        - ``lon_col`` and ``lat_col`` are present in the CSV header.
        - Output directory is writable (created if absent).

        Raises:
            InputValidationError: On any precondition failure.
            CRSError: If either CRS string cannot be parsed by pyproj.
            OutputWriteError: If the output directory cannot be created.
        """
        Validators.assert_file_exists(self.input_path)
        Validators.assert_supported_extension(self.input_path, [".csv"])
        Validators.assert_crs_valid(self.config.from_crs)
        Validators.assert_crs_valid(self.config.to_crs)
        Validators.assert_output_dir_writable(self.output_path)

        # Peek at the header row to confirm coordinate columns exist
        df_peek = pd.read_csv(self.input_path, nrows=0)
        Validators.assert_columns_exist(
            df_peek, [self.config.lon_col, self.config.lat_col]
        )

        logger.debug("Inputs validated successfully.")

    def process(self) -> None:
        """Read the CSV, reproject coordinates, and write the output file.

        Uses :mod:`pyproj` for the actual CRS transformation.  Rows with
        null or non-numeric coordinate values are dropped and counted in
        :attr:`_result`.

        Raises:
            OutputWriteError: If writing the output file fails.
        """
        df = pd.read_csv(self.input_path)
        original_len = len(df)

        # Drop rows where either coordinate column is null or non-numeric
        df = self._drop_invalid_rows(df)
        rows_skipped = original_len - len(df)

        if rows_skipped:
            logger.warning(
                "Dropped %d row(s) with null or non-numeric coordinate values.",
                rows_skipped,
            )

        # Build the pyproj transformer
        transformer = pyproj.Transformer.from_crs(
            self.config.from_crs,
            self.config.to_crs,
            always_xy=self.config.always_xy,
        )

        # Reproject — pyproj.Transformer.transform accepts numpy arrays
        xs = df[self.config.lon_col].to_numpy(dtype=float)
        ys = df[self.config.lat_col].to_numpy(dtype=float)
        new_xs, new_ys = transformer.transform(xs, ys)

        df[self.config.lon_col] = new_xs
        df[self.config.lat_col] = new_ys

        # Write output in the requested format
        try:
            if self.config.output_format == "geojson":
                self._write_geojson(df)
            else:
                df.to_csv(self.output_path, index=False)
        except OSError as exc:
            raise OutputWriteError(str(self.output_path), str(exc)) from exc

        self._result = TransformResult(
            rows_processed=len(df),
            rows_skipped=rows_skipped,
            from_crs=self.config.from_crs,
            to_crs=self.config.to_crs,
            output_path=self.output_path,
        )
        logger.info(self._result.summary())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where the coordinate columns are null or non-numeric.

        Args:
            df: Input DataFrame loaded from the CSV.

        Returns:
            A filtered DataFrame with only valid coordinate rows.
        """
        lon = pd.to_numeric(df[self.config.lon_col], errors="coerce")
        lat = pd.to_numeric(df[self.config.lat_col], errors="coerce")
        mask = lon.notna() & lat.notna()
        return df[mask].copy()

    def _write_geojson(self, df: pd.DataFrame) -> None:
        """Serialise the DataFrame as a GeoJSON FeatureCollection.

        Coordinate columns become the ``geometry`` of each Feature.
        All remaining columns are stored in the ``properties`` dict.

        Args:
            df: DataFrame with valid, reprojected coordinate values.
        """
        lon_col = self.config.lon_col
        lat_col = self.config.lat_col

        features = []
        for _, row in df.iterrows():
            prop_cols = [c for c in df.columns if c not in (lon_col, lat_col)]
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row[lon_col], row[lat_col]],
                },
                "properties": {c: row[c] for c in prop_cols},
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(geojson, fh, indent=2, default=str)

    @property
    def result(self) -> TransformResult | None:
        """The :class:`TransformResult` from the last :meth:`run` call.

        Returns ``None`` if :meth:`run` has not been called yet.
        """
        return self._result
