"""
Raster Band Stats Reporter — Core Module
==========================================
Computes per-band statistics for any GeoTIFF (or rasterio-supported raster)
and writes the results to JSON or CSV.

Classes:
    BandStats           Immutable stats for one raster band.
    BandStatsConfig     Configuration bundle for the reporter.
    BandStatsReporter   Primary tool class (inherits GeoTool).

Usage::

    from pathlib import Path
    from src.raster_band_stats.stats import BandStatsReporter, BandStatsConfig

    tool = BandStatsReporter(
        input_path=Path("data/landsat_scene.tif"),
        output_path=Path("output/band_stats.json"),
        config=BandStatsConfig(output_format="json"),
    )
    tool.run()

    for band in tool.band_stats:
        print(band)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import rasterio

from shared.python.base_tool import GeoTool
from shared.python.exceptions import BandIndexError, OutputWriteError, RasterError
from shared.python.validators import Validators

logger = logging.getLogger("geoscripthub.raster_band_stats")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandStats:
    """Immutable statistics for one raster band.

    Attributes:
        band_index: 1-based band number.
        min: Minimum pixel value (nodata pixels excluded).
        max: Maximum pixel value (nodata pixels excluded).
        mean: Arithmetic mean of valid pixel values.
        std_dev: Standard deviation of valid pixel values.
        valid_pixels: Count of non-nodata pixels used in statistics.
        nodata_pixels: Count of nodata / masked pixels.
        nodata_value: The nodata sentinel value defined in the raster,
                      or ``None`` if not defined.
    """

    band_index: int
    min: float
    max: float
    mean: float
    std_dev: float
    valid_pixels: int
    nodata_pixels: int
    nodata_value: float | None

    def __str__(self) -> str:
        return (
            f"Band {self.band_index}: "
            f"min={self.min:.4f} max={self.max:.4f} "
            f"mean={self.mean:.4f} std={self.std_dev:.4f} "
            f"valid_px={self.valid_pixels:,}"
        )


@dataclass
class BandStatsConfig:
    """Configuration for :class:`BandStatsReporter`.

    Attributes:
        output_format: Output file format — ``"json"`` or ``"csv"``.
        bands: Specific 1-based band indices to process.  ``None`` means
               all bands.
        compute_histogram: When ``True``, also compute a 256-bin histogram
                           for each band.  Adds processing time for large rasters.
        indent: JSON indentation level.  Only used when ``output_format="json"``.
    """

    output_format: Literal["json", "csv"] = "json"
    bands: list[int] | None = None
    compute_histogram: bool = False
    indent: int = 2


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class BandStatsReporter(GeoTool):
    """Compute per-band statistics for a GeoTIFF and write the results.

    Inherits the Template Method pipeline from :class:`~shared.python.GeoTool`.

    Opens the raster with :mod:`rasterio`, reads each requested band as a
    masked numpy array (respecting the nodata value), and computes min, max,
    mean, and std deviation using :mod:`numpy`.

    Args:
        input_path: Path to the input raster file (GeoTIFF recommended).
        output_path: Path for the output stats file.
        config: A :class:`BandStatsConfig` instance.
        verbose: Enable DEBUG-level logging.

    Example::

        BandStatsReporter(
            Path("data/sentinel2.tif"),
            Path("output/stats.json"),
            BandStatsConfig(bands=[1, 2, 3, 4]),
        ).run()
    """

    SUPPORTED_EXTENSIONS = [".tif", ".tiff", ".img", ".vrt", ".nc"]

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        config: BandStatsConfig | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__(input_path, output_path, verbose=verbose)
        self.config = config or BandStatsConfig()
        self._band_stats: list[BandStats] = []
        self._meta: dict = {}

    # ------------------------------------------------------------------
    # GeoTool abstract method implementations
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate input raster path and band index requests.

        Raises:
            InputValidationError: If file is missing or has unsupported extension.
            BandIndexError: If a requested band index is out of range.
            OutputWriteError: If the output directory cannot be created.
        """
        Validators.assert_file_exists(self.input_path)
        Validators.assert_supported_extension(self.input_path, self.SUPPORTED_EXTENSIONS)
        Validators.assert_output_dir_writable(self.output_path)

        if self.config.bands:
            with rasterio.open(self.input_path) as src:
                for b in self.config.bands:
                    Validators.assert_band_index_valid(b, src.count)

        logger.debug("Inputs validated.")

    def process(self) -> None:
        """Open the raster, compute statistics for each band, and write output.

        Raises:
            RasterError: If the raster cannot be read by rasterio.
            OutputWriteError: If writing the output file fails.
        """
        try:
            with rasterio.open(self.input_path) as src:
                self._meta = dict(src.meta)
                band_indices = self.config.bands or list(range(1, src.count + 1))
                logger.info(
                    "Processing %d band(s) from %s", len(band_indices), self.input_path.name
                )

                stats_list: list[BandStats] = []
                for band_index in band_indices:
                    logger.debug("Computing stats for band %d...", band_index)
                    band_array = src.read(band_index, masked=True)
                    stats = self._compute_stats(band_array, band_index, src.nodata)
                    stats_list.append(stats)
                    logger.debug("  %s", stats)

        except rasterio.errors.RasterioIOError as exc:
            raise RasterError(f"Could not open raster '{self.input_path}': {exc}") from exc

        self._band_stats = stats_list

        try:
            if self.config.output_format == "csv":
                self._write_csv(stats_list)
            else:
                self._write_json(stats_list)
        except OSError as exc:
            raise OutputWriteError(str(self.output_path), str(exc)) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(
        array: npt.NDArray,
        band_index: int,
        nodata: float | None,
    ) -> BandStats:
        """Compute statistics for one band array.

        Args:
            array: A rasterio masked array for the band.
            band_index: 1-based band number.
            nodata: The nodata value from the raster metadata.

        Returns:
            A populated :class:`BandStats` dataclass.
        """
        # Use the masked array — valid data only
        valid = array.compressed() if hasattr(array, "compressed") else array.ravel()
        nodata_count = int(np.sum(array.mask)) if hasattr(array, "mask") else 0

        if valid.size == 0:
            return BandStats(
                band_index=band_index, min=float("nan"), max=float("nan"),
                mean=float("nan"), std_dev=float("nan"),
                valid_pixels=0, nodata_pixels=nodata_count, nodata_value=nodata,
            )

        return BandStats(
            band_index=band_index,
            min=float(np.min(valid)),
            max=float(np.max(valid)),
            mean=float(np.mean(valid)),
            std_dev=float(np.std(valid)),
            valid_pixels=int(valid.size),
            nodata_pixels=nodata_count,
            nodata_value=nodata,
        )

    def _write_json(self, stats_list: list[BandStats]) -> None:
        """Serialise stats to JSON.

        Args:
            stats_list: List of :class:`BandStats` to write.
        """
        output = {
            "source_file": str(self.input_path),
            "crs": str(self._meta.get("crs", "undefined")),
            "width": self._meta.get("width"),
            "height": self._meta.get("height"),
            "dtype": str(self._meta.get("dtype", "unknown")),
            "bands": [asdict(s) for s in stats_list],
        }
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=self.config.indent, default=str)

    def _write_csv(self, stats_list: list[BandStats]) -> None:
        """Serialise stats to CSV.

        Args:
            stats_list: List of :class:`BandStats` to write.
        """
        import csv  # noqa: PLC0415

        fieldnames = [
            "band_index", "min", "max", "mean", "std_dev",
            "valid_pixels", "nodata_pixels", "nodata_value",
        ]
        with open(self.output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for s in stats_list:
                writer.writerow(asdict(s))

    @property
    def band_stats(self) -> list[BandStats]:
        """List of :class:`BandStats` from the last run, or ``[]``."""
        return self._band_stats
