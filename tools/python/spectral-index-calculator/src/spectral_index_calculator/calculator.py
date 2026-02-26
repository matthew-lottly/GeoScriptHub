"""
Spectral Index Calculator — Core Module
=========================================
Calculates spectral vegetation/water/soil indices from multi-band raster files
(Landsat 8/9 or Sentinel-2).

Each index is implemented as a :class:`IndexStrategy` subclass following the
Strategy design pattern.  The :class:`SpectralIndexCalculator` orchestrator
accepts any list of strategies and runs them all in one pass.

Supported indices:
    - NDVI   Normalized Difference Vegetation Index
    - NDWI   Normalized Difference Water Index
    - SAVI   Soil-Adjusted Vegetation Index
    - EVI    Enhanced Vegetation Index

Classes:
    BandFileMap         Typed dict of band name → file path.
    IndexResult         Immutable result for one computed index.
    IndexStrategy       Abstract base for all index strategies.
    NDVIStrategy        NDVI = (NIR - Red) / (NIR + Red)
    NDWIStrategy        NDWI = (Green - NIR) / (Green + NIR)
    SAVIStrategy        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    EVIStrategy         EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    SpectralIndexCalculator  Primary tool class (inherits GeoTool).

Usage::

    from pathlib import Path
    from src.spectral_index_calculator.calculator import (
        SpectralIndexCalculator, BandFileMap, NDVIStrategy
    )

    tool = SpectralIndexCalculator(
        band_files=BandFileMap(
            red=Path("data/LC08_B4.TIF"),
            nir=Path("data/LC08_B5.TIF"),
        ),
        output_dir=Path("output/indices"),
        strategies=[NDVIStrategy()],
    )
    tool.run()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio import profiles

from shared.python.base_tool import GeoTool
from shared.python.exceptions import (
    InputValidationError,
    OutputWriteError,
    SpectralIndexError,
)
from shared.python.validators import Validators

logger = logging.getLogger("geoscripthub.spectral_index_calculator")


# ---------------------------------------------------------------------------
# Typed dict for band file paths
# ---------------------------------------------------------------------------


class BandFileMap(TypedDict, total=False):
    """Mapping of spectral band names to their GeoTIFF file paths.

    Not all keys are required for every index — each strategy declares
    which bands it needs.

    Keys:
        blue: Blue band file.
               <!-- PLACEHOLDER: Landsat 8/9 → B2.TIF, Sentinel-2 → B02.tif -->
        green: Green band file.
               <!-- PLACEHOLDER: Landsat 8/9 → B3.TIF, Sentinel-2 → B03.tif -->
        red: Red band file.
             <!-- PLACEHOLDER: Landsat 8/9 → B4.TIF, Sentinel-2 → B04.tif -->
        nir: Near-infrared band file.
             <!-- PLACEHOLDER: Landsat 8/9 → B5.TIF, Sentinel-2 → B08.tif -->
        swir1: Shortwave infrared 1 band file.
               <!-- PLACEHOLDER: Landsat 8/9 → B6.TIF, Sentinel-2 → B11.tif -->
        swir2: Shortwave infrared 2 band file.
               <!-- PLACEHOLDER: Landsat 8/9 → B7.TIF, Sentinel-2 → B12.tif -->
    """

    blue: Path
    green: Path
    red: Path
    nir: Path
    swir1: Path
    swir2: Path


@dataclass(frozen=True)
class IndexResult:
    """Immutable result for one computed spectral index.

    Attributes:
        index_name: Short name of the index (e.g. ``"NDVI"``).
        output_path: Path where the result GeoTIFF was written.
        min_value: Minimum valid index value in the output raster.
        max_value: Maximum valid index value in the output raster.
        mean_value: Mean valid index value.
    """

    index_name: str
    output_path: Path
    min_value: float
    max_value: float
    mean_value: float

    def __str__(self) -> str:
        return (
            f"{self.index_name}: min={self.min_value:.4f} "
            f"max={self.max_value:.4f} mean={self.mean_value:.4f} "
            f"→ {self.output_path.name}"
        )


# ---------------------------------------------------------------------------
# Index strategy ABC + concrete implementations
# ---------------------------------------------------------------------------


class IndexStrategy(ABC):
    """Abstract base for a single spectral index computation.

    Subclasses implement :meth:`required_bands` to declare their inputs
    and :meth:`compute` to run the formula on numpy arrays.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name of the index (e.g. ``"NDVI"``)."""

    @property
    @abstractmethod
    def required_bands(self) -> list[str]:
        """List of band keys from :class:`BandFileMap` that this index needs.

        Example: ``["red", "nir"]``
        """

    @abstractmethod
    def compute(self, bands: dict[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Compute the index from a dict of band arrays.

        Args:
            bands: Dict mapping band name → float32 numpy array.  Only the
                   bands declared in :attr:`required_bands` are guaranteed present.

        Returns:
            A float32 numpy array of index values (typically in [-1, 1]).
        """


class NDVIStrategy(IndexStrategy):
    """NDVI — Normalized Difference Vegetation Index.

    Formula: ``NDVI = (NIR - Red) / (NIR + Red)``

    Range: -1 to +1.  Higher values indicate denser, healthier vegetation.
    Typical thresholds: < 0.1 bare soil/water, 0.1–0.3 sparse, > 0.5 dense.

    Required bands: ``red``, ``nir``
    """

    @property
    def name(self) -> str:
        return "NDVI"

    @property
    def required_bands(self) -> list[str]:
        return ["red", "nir"]

    def compute(self, bands: dict[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Calculate NDVI from red and NIR band arrays."""
        red = bands["red"].astype(np.float32)
        nir = bands["nir"].astype(np.float32)
        denominator = nir + red
        # Avoid division-by-zero — set to nodata (-9999) where denom == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(denominator == 0, -9999.0, (nir - red) / denominator)
        return ndvi.astype(np.float32)


class NDWIStrategy(IndexStrategy):
    """NDWI — Normalized Difference Water Index.

    Formula: ``NDWI = (Green - NIR) / (Green + NIR)``

    Range: -1 to +1.  Positive values indicate open water features.
    Negative values indicate vegetation or soil.

    Required bands: ``green``, ``nir``
    """

    @property
    def name(self) -> str:
        return "NDWI"

    @property
    def required_bands(self) -> list[str]:
        return ["green", "nir"]

    def compute(self, bands: dict[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Calculate NDWI from green and NIR band arrays."""
        green = bands["green"].astype(np.float32)
        nir = bands["nir"].astype(np.float32)
        denominator = green + nir
        with np.errstate(invalid="ignore", divide="ignore"):
            ndwi = np.where(denominator == 0, -9999.0, (green - nir) / denominator)
        return ndwi.astype(np.float32)


class SAVIStrategy(IndexStrategy):
    """SAVI — Soil-Adjusted Vegetation Index.

    Formula: ``SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)``

    The soil brightness correction factor ``L`` defaults to 0.5, which is
    appropriate for intermediate vegetation cover.

    Required bands: ``red``, ``nir``

    Args:
        soil_factor: The ``L`` correction factor.
                     <!-- PLACEHOLDER: set to 0.5 for intermediate cover,
                          0.25 for dense vegetation, 1.0 for very sparse cover -->
    """

    def __init__(self, soil_factor: float = 0.5) -> None:
        # PLACEHOLDER: adjust L based on your study area's vegetation density
        self.soil_factor = soil_factor

    @property
    def name(self) -> str:
        return "SAVI"

    @property
    def required_bands(self) -> list[str]:
        return ["red", "nir"]

    def compute(self, bands: dict[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Calculate SAVI with the configured soil brightness factor."""
        red = bands["red"].astype(np.float32)
        nir = bands["nir"].astype(np.float32)
        L = self.soil_factor
        denominator = nir + red + L
        with np.errstate(invalid="ignore", divide="ignore"):
            savi = np.where(
                denominator == 0, -9999.0,
                ((nir - red) / denominator) * (1.0 + L),
            )
        return savi.astype(np.float32)


class EVIStrategy(IndexStrategy):
    """EVI — Enhanced Vegetation Index.

    Formula:
        ``EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)``

    Corrects for atmospheric and soil background distortions.  Better than
    NDVI in high-biomass regions and over the Amazon.

    Required bands: ``blue``, ``red``, ``nir``
    """

    @property
    def name(self) -> str:
        return "EVI"

    @property
    def required_bands(self) -> list[str]:
        return ["blue", "red", "nir"]

    def compute(self, bands: dict[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Calculate EVI from blue, red, and NIR band arrays."""
        blue = bands["blue"].astype(np.float32)
        red = bands["red"].astype(np.float32)
        nir = bands["nir"].astype(np.float32)
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            evi = np.where(
                denominator == 0, -9999.0,
                2.5 * (nir - red) / denominator,
            )
        return evi.astype(np.float32)


# ---------------------------------------------------------------------------
# Default strategy suite
# ---------------------------------------------------------------------------

ALL_STRATEGIES: list[IndexStrategy] = [
    NDVIStrategy(),
    NDWIStrategy(),
    SAVIStrategy(),
    EVIStrategy(),
]


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class SpectralIndexCalculator(GeoTool):
    """Compute one or more spectral indices from satellite band GeoTIFF files.

    Inherits the Template Method pipeline from :class:`~shared.python.GeoTool`.

    Each requested :class:`IndexStrategy` is executed in sequence.  The
    output for each index is written as a single-band float32 GeoTIFF to
    ``output_dir``, named ``<INDEX_NAME>.tif`` (e.g. ``NDVI.tif``).

    Args:
        band_files: A :class:`BandFileMap` dict.
                    <!-- PLACEHOLDER: provide Path objects for each band needed
                         by your selected strategies, e.g.:
                         BandFileMap(
                             red=Path("LC08_B4.TIF"),   # Landsat 8 Red
                             nir=Path("LC08_B5.TIF"),   # Landsat 8 NIR
                         ) -->
        output_dir: Directory where output GeoTIFFs will be written.
                    <!-- PLACEHOLDER: e.g. Path("output/indices") -->
        strategies: List of :class:`IndexStrategy` instances to run.
                    <!-- PLACEHOLDER: pick from NDVIStrategy, NDWIStrategy,
                         SAVIStrategy, EVIStrategy — or pass them all:
                         strategies=ALL_STRATEGIES -->
        verbose: Enable DEBUG-level logging.

    Note:
        The ``input_path`` argument inherited from ``GeoTool`` is set to
        the first band file's path for logging/repr purposes.
    """

    def __init__(
        self,
        band_files: BandFileMap,
        output_dir: Path,
        strategies: list[IndexStrategy] | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        # Use the first provided band path as the canonical input_path for GeoTool
        first_band = next(iter(band_files.values())) if band_files else Path(".")
        super().__init__(Path(first_band), Path(output_dir), verbose=verbose)

        self.band_files: BandFileMap = band_files
        self.output_dir: Path = Path(output_dir)
        self.strategies: list[IndexStrategy] = strategies or [NDVIStrategy()]
        self._results: list[IndexResult] = []

    # ------------------------------------------------------------------
    # GeoTool abstract method implementations
    # ------------------------------------------------------------------

    def validate_inputs(self) -> None:
        """Validate all band files exist and the output directory is writable.

        Raises:
            InputValidationError: If any required band file is missing or
                raster dimensions do not match across bands.
            SpectralIndexError: If a strategy requires a band not in ``band_files``.
            OutputWriteError: If the output directory cannot be created.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate all provided band files exist
        for band_name, band_path in self.band_files.items():
            Validators.assert_file_exists(Path(band_path))
            Validators.assert_supported_extension(Path(band_path), [".tif", ".tiff", ".img"])

        # Validate that strategies have their required bands available
        available_bands = set(self.band_files.keys())
        for strategy in self.strategies:
            missing = set(strategy.required_bands) - available_bands
            if missing:
                raise SpectralIndexError(
                    strategy.name,
                    f"Required band(s) not provided: {', '.join(missing)}",
                )

        # Check that all provided rasters share the same shape
        shapes: list[tuple[int, int]] = []
        for band_name, band_path in self.band_files.items():
            with rasterio.open(band_path) as src:
                shapes.append((src.height, src.width))

        if len(set(shapes)) > 1:
            raise InputValidationError(
                "Band rasters have mismatched dimensions: "
                + ", ".join(f"{k}={v}" for k, v in zip(self.band_files.keys(), shapes))
            )

        logger.debug("Inputs validated: %d band(s), %d strategy/strategies.", len(self.band_files), len(self.strategies))

    def process(self) -> None:
        """Load band arrays, compute each index, and write output GeoTIFFs.

        Raises:
            SpectralIndexError: If a strategy computation fails.
            OutputWriteError: If writing an output file fails.
        """
        # Load all bands needed across all strategies into memory
        needed_bands = {b for s in self.strategies for b in s.required_bands}
        band_arrays: dict[str, npt.NDArray[np.float32]] = {}
        reference_profile: profiles.Profile | None = None

        for band_name in needed_bands:
            if band_name not in self.band_files:
                continue
            band_path = Path(self.band_files[band_name])  # type: ignore[index]
            with rasterio.open(band_path) as src:
                band_arrays[band_name] = src.read(1).astype(np.float32)
                if reference_profile is None:
                    reference_profile = src.profile.copy()

        if reference_profile is None:
            raise SpectralIndexError("unknown", "No bands were loaded.")

        # Run each strategy and write results
        results: list[IndexResult] = []
        for strategy in self.strategies:
            logger.info("Computing %s...", strategy.name)
            try:
                index_array = strategy.compute(band_arrays)
            except Exception as exc:
                raise SpectralIndexError(strategy.name, str(exc)) from exc

            output_path = self.output_dir / f"{strategy.name}.tif"
            self._write_index_raster(index_array, output_path, reference_profile)

            # Compute summary stats (ignore -9999 nodata)
            valid = index_array[index_array != -9999.0]
            results.append(IndexResult(
                index_name=strategy.name,
                output_path=output_path,
                min_value=float(valid.min()) if valid.size else float("nan"),
                max_value=float(valid.max()) if valid.size else float("nan"),
                mean_value=float(valid.mean()) if valid.size else float("nan"),
            ))
            logger.info("  %s", results[-1])

        self._results = results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_index_raster(
        array: npt.NDArray[np.float32],
        output_path: Path,
        reference_profile: profiles.Profile,
    ) -> None:
        """Write a single-band float32 GeoTIFF.

        Args:
            array: The index value array to write.
            output_path: Destination file path.
            reference_profile: rasterio profile from the input bands
                               (provides CRS, transform, etc.).

        Raises:
            OutputWriteError: If the file cannot be written.
        """
        profile = reference_profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="lzw",
        )
        try:
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(array, 1)
        except OSError as exc:
            raise OutputWriteError(str(output_path), str(exc)) from exc

    @property
    def results(self) -> list[IndexResult]:
        """List of :class:`IndexResult` from the last run, or ``[]``."""
        return self._results
