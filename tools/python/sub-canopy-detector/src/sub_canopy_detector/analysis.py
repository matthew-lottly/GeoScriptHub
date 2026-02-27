"""
analysis.py
===========
Core sub-canopy structure detection algorithm.

Mirrors the processing pipeline of the companion GEE script:

  1.  SAR temporal statistics  (mean + std of VV and VH linear power)
  2.  Stability indicator      (1 - CoV, high when backscatter is steady)
  3.  Polarimetric ratio       (VH / VV normalized to [POL_RATIO_MIN, MAX])
  4.  SAR texture              (GLCM homogeneity proxy via local variance)
  5.  SAR anomaly              (Gaussian residual z-score in dB)
  6.  S2 cloud masking         (SCL-based per-pixel masking)
  7.  S2 spectral indices      (NDVI, NDWI, NDBI from cloud-masked median)
  8.  NDBI anomaly             (same Gaussian z-score approach as SAR)
  9.  Forest + terrain mask    (NDVI threshold + slope limit)
  10. Weighted fusion          (5-indicator probability score)
  11. Confidence zones         (HIGH / MEDIUM thresholds)
  12. Morphological opening    (remove salt-and-pepper noise)
  13. Footprint vectorisation  (polygonise connected blobs, filter area)

All heavy computation is triggered by a single `run()` call.  The result
is an ``AnalysisResult`` dataclass whose fields are plain numpy arrays
(or geopandas GeoDataFrames) aligned to a shared geospatial grid.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import geopandas as gpd
import xarray as xr
import dask
from scipy.ndimage import (
    gaussian_filter,
    binary_erosion,
    binary_dilation,
    uniform_filter,
    label as ndi_label,
)
from skimage.measure import regionprops
import rasterio
import rasterio.transform
from rasterio.features import shapes
from rasterio.transform import Affine, from_bounds
from shapely.geometry import shape

from .aoi import AOIResult
from .fetcher import ImageryData


# ---------------------------------------------------------------------------
# Default analysis parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict = dict(
    forest_ndvi_threshold=0.55,
    water_ndwi_threshold=0.15,
    stability_floor=0.70,
    texture_kernel_radius=3,         # pixels
    anomaly_kernel_radius=15,        # pixels (Gaussian sigma)
    anomaly_sigma=1.5,
    # -- Fusion weights (must sum to 1.0) --
    w_stability=0.18,                # SAR temporal stability
    w_polratio=0.12,                 # VH/VV polarimetric ratio
    w_texture=0.12,                  # GLCM homogeneity proxy
    w_anomaly=0.13,                  # Gaussian residual z-score
    w_optical=0.10,                  # S2 NDBI anomaly + inverse NDVI
    w_entropy=0.13,                  # cross-polarization entropy
    w_coherence=0.12,                # amplitude dispersion coherence proxy
    w_seasonal=0.10,                 # seasonal invariance score
    thresh_high=0.65,
    thresh_medium=0.45,
    min_footprint_area=80,           # m^2
    slope_threshold=15,              # degrees
    pol_ratio_min=0.02,
    pol_ratio_max=0.30,
    edge_erosion_px=2,               # extra edge erosion for validity mask
    # -- Building regularisation ---
    min_compactness=0.15,            # Polsby-Popper floor
    min_rectangularity=0.35,         # area / MRR area floor
    max_aspect_ratio=10.0,           # MRR length / width ceiling
    max_footprint_area=25000,        # m^2 -- filter out huge clearings
    min_building_score=0.35,         # composite building-likeness floor
    min_solidity=0.40,               # area / convex-hull area floor
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """All outputs produced by the detection pipeline."""

    # ---- Core probability raster ----------------------------------------
    # 2-D float32 numpy array, values 0.0-1.0.  Pixels outside forest are NaN.
    probability: np.ndarray

    # ---- Confidence zone raster -----------------------------------------
    # Integer: 0 = background, 1 = low, 2 = medium, 3 = high
    confidence: np.ndarray

    # ---- Cleaned probability (morphological opening applied) ------------
    cleaned_prob: np.ndarray

    # ---- Individual indicators (all normalised 0-1) ---------------------
    stability: np.ndarray
    pol_ratio_norm: np.ndarray
    texture: np.ndarray
    sar_anomaly: np.ndarray
    optical_indicator: np.ndarray

    # ---- Advanced SAR indicators (eight-channel fusion) -----------------
    cross_pol_entropy: np.ndarray     # temporal VH/VV stability
    coherence_proxy: np.ndarray       # amplitude dispersion index
    seasonal_invariance: np.ndarray   # inter-season backscatter stability

    # ---- Common validity mask -------------------------------------------
    validity_mask: np.ndarray         # bool -- all sensors valid + edge-eroded

    # ---- Spectral indices -----------------------------------------------
    ndvi: np.ndarray
    ndwi: np.ndarray
    ndbi: np.ndarray

    # ---- RGB true-colour composite --------------------------------------
    rgb_composite: np.ndarray         # (H, W, 3) float32, 0-1

    # ---- Masks ----------------------------------------------------------
    forest_mask: np.ndarray     # bool -- pixels included in analysis
    slope_mask: np.ndarray      # bool -- pixels below slope threshold

    # ---- Detected footprints as vector ----------------------------------
    footprints: gpd.GeoDataFrame

    # ---- Regularised building footprints --------------------------------
    regularized_footprints: gpd.GeoDataFrame

    # ---- Number of S1 scenes used ---------------------------------------
    s1_scene_count: int

    # ---- Geospatial grid metadata (for export) --------------------------
    transform: Affine = field(repr=False)
    crs_wkt: str = field(repr=False)
    height: int = 0
    width: int = 0

    # ---- Parameters snapshot -------------------------------------------
    params: Dict = field(default_factory=dict, repr=False)

    def summary(self) -> None:
        """Print detection summary to the console."""
        high_n   = int(np.sum(self.confidence == 3))
        med_n    = int(np.sum(self.confidence == 2))
        fp_n     = len(self.footprints)
        fp_area  = self.footprints["area_m2"].sum() if fp_n else 0
        reg_n    = len(self.regularized_footprints)
        reg_area = self.regularized_footprints["area_m2"].sum() if reg_n else 0
        rej_n    = fp_n - reg_n
        avg_sc   = float(self.regularized_footprints["building_score"].mean()) if reg_n else 0.0
        print("=== Detection Summary ====================================")
        print(f"  S1 scenes used         : {self.s1_scene_count}")
        print(f"  High-confidence pixels : {high_n:,}")
        print(f"  Medium-confidence px   : {med_n:,}")
        print(f"  Raw footprints         : {fp_n:,}")
        print(f"  Raw footprint area     : {fp_area:,.0f} m2")
        print(f"  Building footprints    : {reg_n:,}")
        print(f"  Building area          : {reg_area:,.0f} m2")
        print(f"  Rejected as noise      : {rej_n:,}")
        print(f"  Avg building score     : {avg_sc:.3f}")
        print("==========================================================")


# ---------------------------------------------------------------------------
# Cloud masking helpers
# ---------------------------------------------------------------------------

# Sentinel-2 SCL classes that represent clear-sky observations
_S2_CLEAR_SCL = {4, 5, 6, 7, 11}   # vegetation, non-veg, water, unclassified, snow

def _mask_s2_clouds(s2_stack: xr.DataArray) -> xr.DataArray:
    """Return the S2 stack with cloudy and shadow pixels set to NaN.

    The SCL (Scene Classification Layer) band is used as a per-pixel quality
    indicator.  Only pixels classified as clear (SCL in {4,5,6,7,11}) are
    retained.  The SCL band is dropped from the output.
    """
    # Select SCL separately before masking reflectance bands
    scl = s2_stack.sel(band="SCL")
    refl = s2_stack.sel(band=[b for b in s2_stack.band.values if b != "SCL"])

    # Build boolean clear mask (time x y x) -- broadcast across bands
    clear = xr.zeros_like(scl, dtype=bool)
    for cls in _S2_CLEAR_SCL:
        clear = clear | (scl == cls)

    # Expand to match reflectance dims (time, band, y, x)
    clear_expanded = clear.expand_dims("band", axis=1).broadcast_like(refl)
    return refl.where(clear_expanded)


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class SubCanopyAnalyser:
    """Runs the full sub-canopy detection pipeline.

    Parameters
    ----------
    aoi:
        Resolved AOI from ``AOIBuilder``.
    imagery:
        Fetched imagery from ``ImageryFetcher.fetch_all()``.
    params:
        Override any default analysis parameter by name.  Unspecified
        parameters retain their defaults (see ``DEFAULT_PARAMS``).
    """

    def __init__(
        self,
        aoi: AOIResult,
        imagery: ImageryData,
        **params,
    ) -> None:
        self.aoi = aoi
        self.imagery = imagery
        self.params = {**DEFAULT_PARAMS, **params}
        p = self.params  # shorthand

        # Validate weight sum
        total_w = (
            p["w_stability"] + p["w_polratio"] + p["w_texture"]
            + p["w_anomaly"] + p["w_optical"]
            + p["w_entropy"] + p["w_coherence"] + p["w_seasonal"]
        )
        if abs(total_w - 1.0) > 0.01:
            raise ValueError(
                f"Fusion weights must sum to 1.0 (got {total_w:.4f}).  "
                "Adjust w_stability ... w_seasonal."
            )

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> AnalysisResult:
        """Execute the full 8-indicator pipeline and return an AnalysisResult.

        Computation is eager: all dask arrays are materialised inside this
        method.  For large AOIs this may take several minutes.

        Indicator channels fused
        ------------------------
        Classic (5):  stability, pol-ratio, texture, SAR anomaly, optical
        Advanced (3): cross-pol entropy, coherence proxy, seasonal invariance
        """
        p = self.params

        # -- Step 1: Compute S1 temporal statistics ----------------------
        if verbose:
            print("[1/9] Computing SAR temporal statistics ...")
        vv_mean, vh_mean, vv_std, s1_time_arr, vv_stack, vh_stack = (
            self._compute_s1_stats()
        )

        # -- Step 2: All SAR-derived indicators --------------------------
        if verbose:
            print("[2/9] Deriving SAR indicators (8 channels) ...")
        stability = self._stability(vv_mean, vv_std)
        pol_ratio_norm = self._pol_ratio(vv_mean, vh_mean)
        vv_mean_db = 10.0 * np.log10(np.maximum(vv_mean, 1e-10))
        texture = self._texture(vv_mean_db, p["texture_kernel_radius"])
        sar_anomaly = self._anomaly(vv_mean_db, p["anomaly_kernel_radius"], p["anomaly_sigma"])

        # Advanced indicators computed from full temporal stacks
        cross_pol_entropy = self._cross_pol_entropy(vv_stack, vh_stack)
        coherence_proxy = self._coherence_proxy(vv_stack)
        seasonal_inv = self._seasonal_invariance(vv_stack, s1_time_arr)

        # Free the large temporal stacks (hundreds of MB)
        del vv_stack, vh_stack

        # -- Step 3: S2 cloud masking + median compositing ---------------
        if verbose:
            print("[3/9] Cloud-masking Sentinel-2, computing spectral indices ...")
        ndvi, ndwi, ndbi, rgb = self._compute_s2_indices()
        ndbi_anomaly = self._anomaly(ndbi, p["anomaly_kernel_radius"], p["anomaly_sigma"])
        # Optical indicator: high NDBI Z-score + low NDVI (structure in forest gap)
        optical = np.clip(0.5 * ndbi_anomaly + 0.5 * (1.0 - np.clip(ndvi, 0, 1)), 0, 1)

        # -- Step 4: Validity + forest + terrain masks -------------------
        if verbose:
            print("[4/9] Building validity, forest, and terrain masks ...")
        validity_mask = self._build_validity_mask(vv_mean, ndvi)
        forest_mask = self._forest_mask(ndvi, ndwi)
        slope_mask = self._slope_mask()
        combined_mask = validity_mask & forest_mask & slope_mask

        # -- Step 5: 8-indicator weighted fusion -------------------------
        if verbose:
            print("[5/9] Fusing 8 indicators into probability score ...")
        fusion = (
            p["w_stability"] * stability
            + p["w_polratio"] * pol_ratio_norm
            + p["w_texture"] * texture
            + p["w_anomaly"] * sar_anomaly
            + p["w_optical"] * optical
            + p["w_entropy"] * cross_pol_entropy
            + p["w_coherence"] * coherence_proxy
            + p["w_seasonal"] * seasonal_inv
        )
        probability = np.where(combined_mask, fusion, np.nan)

        # -- Step 6: Confidence zones ------------------------------------
        if verbose:
            print("[6/9] Assigning confidence zones ...")
        confidence = np.zeros_like(probability, dtype=np.int8)
        confidence = np.where(probability >= p["thresh_medium"], 2, confidence)
        confidence = np.where(probability >= p["thresh_high"], 3, confidence)
        confidence = np.where(np.isnan(probability), 0, confidence)

        # -- Step 7: Morphological opening -------------------------------
        if verbose:
            print("[7/9] Morphological cleaning ...")
        clean_prob = self._morphological_open(probability, p["thresh_medium"])

        # -- Step 8: Vectorise footprints --------------------------------
        if verbose:
            print("[8/9] Extracting footprints ...")
        transform, crs_wkt = self._get_geotransform()
        footprints = self._extract_footprints(
            clean_prob, transform, crs_wkt, p["thresh_medium"], p["min_footprint_area"]
        )

        # -- Step 9: Building regularisation -----------------------------
        if verbose:
            print("[9/9] Regularising building footprints ...")
        regularized = self._regularize_footprints(
            footprints, probability, ndwi, transform
        )

        if verbose:
            print("Done.\n")

        result = AnalysisResult(
            probability=probability,
            confidence=confidence,
            cleaned_prob=clean_prob,
            stability=stability,
            pol_ratio_norm=pol_ratio_norm,
            texture=texture,
            sar_anomaly=sar_anomaly,
            optical_indicator=optical,
            cross_pol_entropy=cross_pol_entropy,
            coherence_proxy=coherence_proxy,
            seasonal_invariance=seasonal_inv,
            validity_mask=validity_mask,
            ndvi=ndvi,
            ndwi=ndwi,
            ndbi=ndbi,
            rgb_composite=rgb,
            forest_mask=forest_mask,
            slope_mask=slope_mask,
            footprints=footprints,
            regularized_footprints=regularized,
            s1_scene_count=self.imagery.s1_count,
            transform=transform,
            crs_wkt=crs_wkt,
            height=probability.shape[0],
            width=probability.shape[1],
            params=self.params,
        )
        result.summary()
        return result

    # ------------------------------------------------------------------
    # S1 statistics
    # ------------------------------------------------------------------

    def _compute_s1_stats(self):
        """Return (vv_mean, vh_mean, vv_std, times, vv_stack, vh_stack).

        S1 RTC values are already in linear power (gamma0).  The raw
        temporal stacks are returned so that advanced indicators
        (entropy, coherence, seasonal) can operate on the full time
        series without re-downloading.
        """
        s1 = self.imagery.s1

        # Parse band names -- stackstac lowercases asset keys
        band_names = [str(b) for b in s1.band.values]
        vv_band = next((b for b in band_names if "vv" in b.lower()), None)
        vh_band = next((b for b in band_names if "vh" in b.lower()), None)

        if vv_band is None or vh_band is None:
            raise KeyError(
                f"Expected 'vv' and 'vh' bands in S1 data; got {band_names}."
            )

        vv_stack = s1.sel(band=vv_band).compute(scheduler='synchronous').values.astype(np.float32)
        vh_stack = s1.sel(band=vh_band).compute(scheduler='synchronous').values.astype(np.float32)

        # Temporal statistics (time axis = 0)
        vv_mean = np.nanmean(vv_stack, axis=0)
        vv_std  = np.nanstd(vv_stack,  axis=0)
        vh_mean = np.nanmean(vh_stack, axis=0)

        time_vals = s1.time.values

        return vv_mean, vh_mean, vv_std, time_vals, vv_stack, vh_stack

    # ------------------------------------------------------------------
    # Individual indicators
    # ------------------------------------------------------------------

    def _stability(self, vv_mean: np.ndarray, vv_std: np.ndarray) -> np.ndarray:
        """SAR temporal stability: 1 - coefficient of variation.

        Persistent sub-canopy structures reflect consistently, yielding a
        high stability score even when the surrounding canopy fluctuates.
        A stability_floor clamps very low-confidence pixels.
        """
        cov = vv_std / np.maximum(vv_mean, 1e-10)
        raw = 1.0 - np.clip(cov, 0.0, 1.0)
        return np.clip(raw, self.params["stability_floor"], 1.0)

    def _pol_ratio(self, vv_mean: np.ndarray, vh_mean: np.ndarray) -> np.ndarray:
        """Polarimetric VH/VV ratio, normalised to [0, 1].

        Built structures and thin vegetation produce a distinct VH/VV ratio
        compared to dense forest: double-bounce raises VV more than VH,
        while volume scattering raises both equally.
        """
        ratio = vh_mean / np.maximum(vv_mean, 1e-10)
        cmin = self.params["pol_ratio_min"]
        cmax = self.params["pol_ratio_max"]
        return np.clip((ratio - cmin) / (cmax - cmin + 1e-10), 0.0, 1.0)

    def _texture(self, vv_mean_db: np.ndarray, kernel_radius: int) -> np.ndarray:
        """GLCM homogeneity proxy computed from local variance in dB.

        A sliding-window ratio (local mean / local variance) approximates
        GEE's GLCM Angular Second Moment: high homogeneity indicates uniform
        man-made surfaces beneath the canopy; low homogeneity suggests natural
        volume scattering from forest branches and trunks.

        We scale the VV dB image to 0-255 (matching GEE's uint8 GLCM input)
        before computing local statistics.
        """
        p02, p98 = np.nanpercentile(vv_mean_db, 2), np.nanpercentile(vv_mean_db, 98)
        scaled = np.clip(
            (vv_mean_db - p02) / (p98 - p02 + 1e-10) * 255.0, 0.0, 255.0
        )

        size = 2 * kernel_radius + 1
        # Fill NaN before applying uniform filter
        filled = np.where(np.isnan(scaled), 0.0, scaled)
        local_mean = uniform_filter(filled, size=size)
        local_sq   = uniform_filter(filled ** 2, size=size)
        local_var  = np.maximum(local_sq - local_mean ** 2, 0.0)

        # Homogeneity-like measure: high mean + low variance
        homogeneity = local_mean / (local_var + 1.0)

        # Normalise to 0-1
        h2, h98 = np.nanpercentile(homogeneity, 2), np.nanpercentile(homogeneity, 98)
        return np.clip((homogeneity - h2) / (h98 - h2 + 1e-10), 0.0, 1.0)

    def _anomaly(
        self, image: np.ndarray, kernel_radius: int, sigma_scale: float
    ) -> np.ndarray:
        """Gaussian residual z-score: how much a pixel deviates from its neighbourhood.

        The low-frequency background is estimated with a Gaussian blur.
        The signed residual is then divided by (sigma_scale * regional_std)
        and clipped to [0, 1].  Bright residuals indicate localised
        backscatter elevation relative to the surroundings, a key signature
        of structures under canopy.
        """
        filled = np.where(np.isnan(image), np.nanmean(image), image)
        smooth = gaussian_filter(filled, sigma=float(kernel_radius))
        residual = filled - smooth
        r_std = np.nanstd(residual)
        z = residual / (sigma_scale * r_std + 1e-10)
        # Map [-3, +3] to [0, 1]; bright anomalies score high
        return np.clip((z + 3.0) / 6.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Advanced SAR indicators (novel eight-channel fusion)
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_pol_entropy(
        vv_stack: np.ndarray, vh_stack: np.ndarray,
    ) -> np.ndarray:
        """Cross-polarization scattering-mechanism stability.

        Computes the temporal coefficient of variation of the VH/VV ratio.
        Persistent scatterers (man-made structures) maintain a stable
        polarimetric response over time, producing a low CV and therefore
        a high indicator score.

        This approximates the Shannon entropy of the scattering mechanism
        without requiring full-polarimetric (quad-pol) data -- a novel
        approach for dual-pol GRD data that captures similar physical
        information to the H-Alpha decomposition used in PolSAR.
        """
        ratio = vh_stack / np.maximum(vv_stack, 1e-10)
        mean_r = np.nanmean(ratio, axis=0)
        std_r = np.nanstd(ratio, axis=0)
        cv = std_r / np.maximum(mean_r, 1e-10)
        # Invert and normalise: low CV (stable mechanism) -> high score
        return np.clip(1.0 - np.clip(cv, 0.0, 2.0) / 2.0, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _coherence_proxy(vv_stack: np.ndarray) -> np.ndarray:
        """Amplitude Dispersion Index as interferometric coherence proxy.

        D_A = std(amplitude) / mean(amplitude) is inversely related to
        interferometric phase stability.  In Persistent Scatterer
        Interferometry (PSI) literature, D_A < 0.25 identifies stable PS
        candidates.

        This proxy extends PS-InSAR concepts to GRD (detected) amplitude
        data, enabling coherence-like analysis without requiring complex
        SLC products -- a significant advantage for operational pipelines.
        """
        amp = np.sqrt(np.maximum(vv_stack, 0.0))
        mean_a = np.nanmean(amp, axis=0)
        std_a = np.nanstd(amp, axis=0)
        da = std_a / np.maximum(mean_a, 1e-10)
        # Invert: low DA -> high score (persistent scatterer)
        return np.clip(1.0 - np.clip(da, 0.0, 1.0), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _seasonal_invariance(
        vv_stack: np.ndarray, times: np.ndarray,
    ) -> np.ndarray:
        """Seasonal stability: low inter-seasonal variance indicates permanence.

        Permanent structures exhibit consistent backscatter across all
        meteorological quarters, while vegetation undergoes phenological
        cycles (leaf-on vs leaf-off, growth vs dormancy) that modulate the
        SAR signal.

        The coefficient of variation across quarterly means isolates the
        seasonal component.  A high invariance score means the pixel's
        seasonal amplitude is small relative to its annual mean --
        characteristic of man-made features beneath forest canopy.
        """
        import pandas as pd

        months = pd.DatetimeIndex(times).month
        quarter_means = []
        for q_months in [(12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]:
            mask = np.isin(months, q_months)
            if mask.sum() >= 3:   # need enough scenes for a stable mean
                quarter_means.append(np.nanmean(vv_stack[mask], axis=0))

        if len(quarter_means) < 2:
            # Not enough seasonal coverage -- return neutral score
            return np.full(vv_stack.shape[1:], 0.5, dtype=np.float32)

        q_stack = np.stack(quarter_means, axis=0)
        seasonal_cv = np.nanstd(q_stack, axis=0) / np.maximum(
            np.nanmean(q_stack, axis=0), 1e-10,
        )
        return np.clip(
            1.0 - np.clip(seasonal_cv, 0.0, 1.0), 0.0, 1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Common validity mask
    # ------------------------------------------------------------------

    def _build_validity_mask(
        self, vv_mean: np.ndarray, ndvi: np.ndarray,
    ) -> np.ndarray:
        """Intersect valid pixels from all sensors plus edge erosion.

        Multi-sensor fusion is only meaningful where *every* input sensor
        has valid data.  This mask enforces that constraint and further
        erodes the boundary by ``edge_erosion_px`` pixels to remove
        reprojection-boundary artefacts where partial-pixel coverage
        degrades accuracy.
        """
        s1_valid = ~np.isnan(vv_mean)
        s2_valid = ~np.isnan(ndvi)

        # DEM validity (already at 10 m after harmonization)
        try:
            dem_arr = self.imagery.dem.isel(band=0).values
            dem_valid = ~np.isnan(dem_arr)
            # Resize if needed (defensive -- should be identical after harmonization)
            if dem_valid.shape != s1_valid.shape:
                dem_valid = np.ones_like(s1_valid, dtype=bool)
        except Exception:
            dem_valid = np.ones_like(s1_valid, dtype=bool)

        combined = s1_valid & s2_valid & dem_valid

        # Edge erosion
        edge_px = self.params.get("edge_erosion_px", 2)
        if edge_px > 0:
            struct = np.ones((3, 3), dtype=bool)
            combined = binary_erosion(
                combined, structure=struct, iterations=edge_px,
            )

        return np.asarray(combined, dtype=bool)

    # ------------------------------------------------------------------
    # Sentinel-2 processing
    # ------------------------------------------------------------------

    def _compute_s2_indices(self):
        """Cloud-mask S2, compute temporal median, return NDVI/NDWI/NDBI + RGB."""
        s2 = self.imagery.s2
        s2_masked = _mask_s2_clouds(s2)

        # Temporal median (nanmedian across time, lazy then compute once)
        s2_med = s2_masked.median(dim="time", skipna=True).compute(scheduler='synchronous')

        def _band(name: str) -> np.ndarray:
            """Extract a band and return as float32 in [0, 1]."""
            return s2_med.sel(band=name).values.astype(np.float32) / 10000.0

        red   = _band("B04")
        green = _band("B03")
        blue  = _band("B02")
        nir   = _band("B08")
        swir  = _band("B11")

        ndvi = (nir - red)   / np.where(np.abs(nir + red)   > 0, nir + red,   1e-10)
        ndwi = (green - nir) / np.where(np.abs(green + nir) > 0, green + nir, 1e-10)
        ndbi = (swir - nir)  / np.where(np.abs(swir + nir)  > 0, swir + nir,  1e-10)

        # True-colour composite (percentile-stretch + gamma for display)
        rgb = np.stack([red, green, blue], axis=-1)
        p02, p98 = np.nanpercentile(rgb, [2, 98])
        rgb = np.clip((rgb - p02) / max(float(p98 - p02), 1e-6), 0.0, 1.0)
        rgb = np.power(rgb, 0.85)  # mild gamma correction

        return (
            np.clip(ndvi, -1.0, 1.0),
            np.clip(ndwi, -1.0, 1.0),
            np.clip(ndbi, -1.0, 1.0),
            np.nan_to_num(rgb, nan=0.0).astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Forest and terrain masks
    # ------------------------------------------------------------------

    def _forest_mask(self, ndvi: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
        """Boolean mask: True where NDVI >= threshold AND NDWI < threshold.

        Limiting the analysis to forested, non-waterlogged pixels suppresses
        false positives from urban areas and open water, keeping the
        probability score meaningful.
        """
        p = self.params
        return (
            (ndvi >= p["forest_ndvi_threshold"])
            & (ndwi < p["water_ndwi_threshold"])
        )

    def _slope_mask(self) -> np.ndarray:
        """Boolean mask: True where terrain slope < SLOPE_THRESHOLD degrees.

        Steep slopes produce SAR layover and foreshortening artefacts that
        mimic sub-canopy structure signatures.  Excluding high-slope terrain
        removes most of these confounders.

        The DEM is pre-harmonized to the S1 10 m master grid by the fetcher's
        ``_harmonize_grids()`` method using bilinear interpolation.  Pixel
        spacing is derived from the actual coordinate arrays rather than from
        the AOI bounding box, ensuring correct results even after edge
        trimming.
        """
        dem_arr = self.imagery.dem.isel(band=0).compute(scheduler='synchronous').values.astype(np.float32)

        # Fallback resampling if DEM was not pre-harmonized (defensive)
        s1_shape = (self.imagery.s1.sizes["y"], self.imagery.s1.sizes["x"])
        if dem_arr.shape != s1_shape:
            from scipy.ndimage import zoom
            warnings.warn(
                "DEM grid does not match S1 -- resampling on-the-fly. "
                "Use ImageryFetcher.fetch_all() for proper harmonization.",
                stacklevel=2,
            )
            zy = s1_shape[0] / dem_arr.shape[0]
            zx = s1_shape[1] / dem_arr.shape[1]
            dem_arr = zoom(dem_arr, (zy, zx), order=1)

        # Pixel spacing from actual S1 coordinate arrays (robust to trimming)
        y_coords = self.imagery.s1["y"].values
        x_coords = self.imagery.s1["x"].values
        res_x = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 10.0
        res_y = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 10.0

        dy, dx = np.gradient(np.asarray(dem_arr), res_y, res_x)  # type: ignore[arg-type]
        slope_deg = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

        return slope_deg < self.params["slope_threshold"]

    # ------------------------------------------------------------------
    # Morphological cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _morphological_open(
        probability: np.ndarray, threshold: float, iterations: int = 1
    ) -> np.ndarray:
        """Remove isolated speckle via morphological opening (erosion + dilation).

        Opening erodes small isolated detections that are unlikely to represent
        real structures, then restores the shape of larger connected blobs.
        The threshold determines which pixels enter the binary mask.
        """
        binary = (~np.isnan(probability)) & (probability >= threshold)
        struct = np.ones((3, 3), dtype=bool)
        eroded  = binary_erosion(binary, structure=struct, iterations=iterations)
        dilated = binary_dilation(eroded, structure=struct, iterations=iterations)
        # Re-apply probability values only where the cleaned mask is True
        return np.where(dilated, probability, np.nan)

    # ------------------------------------------------------------------
    # Footprint extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_footprints(
        clean_prob: np.ndarray,
        transform: Affine,
        crs_wkt: str,
        threshold: float,
        min_area: float,
    ) -> gpd.GeoDataFrame:
        """Vectorise connected blobs above *threshold* and filter by area.

        Each blob becomes one polygon in the output GeoDataFrame, with
        attributes: area_m2, prob_mean, prob_max.
        """
        binary = (~np.isnan(clean_prob)) & (clean_prob >= threshold)
        float_arr = np.where(binary, clean_prob, 0.0).astype(np.float32)

        # Vectorise via rasterio.features.shapes -- yields (geojson_dict, value) tuples
        labeled, n_feats = ndi_label(binary)  # type: ignore[misc]
        if n_feats == 0:
            return gpd.GeoDataFrame(
                columns=["geometry", "area_m2", "prob_mean", "prob_max"],
                geometry="geometry",
            ).set_crs(crs_wkt)

        records = []
        for region in regionprops(labeled, intensity_image=clean_prob):
            geom_mask = labeled == region.label
            geom_arr  = geom_mask.astype(np.uint8)

            # Get single polygon for this blob
            polys = list(shapes(geom_arr, mask=geom_arr, transform=transform))
            if not polys:
                continue
            geom = shape(polys[0][0])

            # Pixel resolution for area (use pixel_areas from labels)
            pixs = clean_prob[geom_mask]
            area_px = region.area
            # rasterio transform: pixel_x_size is transform.a
            pixel_area = abs(transform.a * transform.e)
            area_m2 = area_px * pixel_area

            if area_m2 < min_area:
                continue

            records.append({
                "geometry": geom,
                "area_m2":  round(area_m2, 1),
                "prob_mean": round(float(np.nanmean(pixs)), 4),
                "prob_max":  round(float(np.nanmax(pixs)),  4),
            })

        if not records:
            return gpd.GeoDataFrame(
                columns=["geometry", "area_m2", "prob_mean", "prob_max"],
                geometry="geometry",
            ).set_crs(crs_wkt)

        gdf = gpd.GeoDataFrame(records, geometry="geometry").set_crs(crs_wkt)
        return gdf

    # ------------------------------------------------------------------
    # Building regularisation
    # ------------------------------------------------------------------

    def _regularize_footprints(
        self,
        footprints: gpd.GeoDataFrame,
        probability: np.ndarray,
        ndwi: np.ndarray,
        transform: Affine,
    ) -> gpd.GeoDataFrame:
        """Regularise raw blobs into building-like polygons.

        Each raw footprint is scored for *building-likeness* using seven
        geometric, radiometric, and contextual criteria.  This multi-criteria
        approach effectively separates real buildings from common false
        positives such as tidal flats, river channels, clearings, and other
        non-building landscape features.

        Scoring components
        ------------------
        1. **Rectangularity** -- blob area / MRR area.  Buildings fill
           their bounding rectangle; natural features do not.
        2. **Compactness** (Polsby--Popper) -- 4 pi A / P^2.  Compact
           shapes score high; fractal coastlines score very low.
        3. **Solidity** -- blob area / convex-hull area.  Solid, simple
           shapes (> 0.7) are building-like; jagged, branching tidal or
           river-channel shapes have solidity < 0.4.
        4. **Edge sharpness** -- mean probability-gradient magnitude along
           the polygon boundary.  Buildings produce sharp edges; canopy
           or tidal transitions are gradual.
        5. **Size appropriateness** -- bell-curve peaking at 100 -- 3 000 m²
           (typical building footprints).  Very large blobs are heavily
           penalised because they usually represent landscape features.
        6. **Detection probability** -- mean probability within the blob.
        7. **Water penalty** -- mean NDWI sampled within the polygon.
           Positive NDWI indicates water or saturated wetland, strongly
           suppressing the building score.
        """
        _EMPTY_COLS = [
            "geometry", "area_m2", "prob_mean", "prob_max",
            "compactness", "rectangularity", "solidity",
            "aspect_ratio", "edge_sharpness", "ndwi_mean",
            "building_score", "is_rectangular",
        ]

        if footprints.empty:
            return gpd.GeoDataFrame(
                columns=_EMPTY_COLS, geometry="geometry",
            ).set_crs(footprints.crs or "EPSG:4326")

        p = self.params

        # Pre-compute probability gradient for edge-sharpness scoring
        prob_clean = np.nan_to_num(probability, nan=0.0)
        gy, gx = np.gradient(prob_clean)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # NDWI array for water penalty
        ndwi_clean = np.nan_to_num(ndwi, nan=0.0)

        records = []
        for _, row in footprints.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            area = geom.area
            perimeter = geom.length

            # -- Compactness (Polsby--Popper) ---------------------------
            compactness = (
                (4.0 * np.pi * area) / (perimeter ** 2)
                if perimeter > 0 else 0.0
            )

            # -- Solidity (area / convex hull area) ---------------------
            ch = geom.convex_hull
            solidity = area / max(ch.area, 1e-10)

            # -- Minimum Rotated Rectangle ------------------------------
            mrr = geom.minimum_rotated_rectangle
            mrr_area = mrr.area if mrr.area > 0 else 1e-10
            rectangularity = area / mrr_area

            # -- Aspect ratio from MRR edges ----------------------------
            coords = list(mrr.exterior.coords)
            edge_lens = []
            for i in range(len(coords) - 1):
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
                edge_lens.append(np.sqrt(dx ** 2 + dy ** 2))
            edge_lens.sort()
            aspect_ratio = (
                edge_lens[-1] / max(edge_lens[0], 0.01)
                if edge_lens else 1.0
            )

            # -- Edge sharpness from probability gradient ---------------
            boundary = geom.boundary
            n_pts = max(int(boundary.length / abs(transform.a)), 8)
            sharp_sum, n_valid = 0.0, 0
            for frac in np.linspace(0.0, 1.0, n_pts, endpoint=False):
                pt = boundary.interpolate(frac, normalized=True)
                c_idx = int((pt.x - transform.c) / transform.a)
                r_idx = int((pt.y - transform.f) / transform.e)
                if 0 <= r_idx < grad_mag.shape[0] and 0 <= c_idx < grad_mag.shape[1]:
                    sharp_sum += grad_mag[r_idx, c_idx]
                    n_valid += 1
            edge_sharpness = sharp_sum / max(n_valid, 1)

            # -- NDWI sampling within polygon ---------------------------
            cent = geom.centroid
            ndwi_samples: list[float] = []
            sample_pts = [cent] + [
                boundary.interpolate(f, normalized=True)
                for f in np.linspace(0, 1, 8, endpoint=False)
            ]
            for pt in sample_pts:
                ci = int((pt.x - transform.c) / transform.a)
                ri = int((pt.y - transform.f) / transform.e)
                if 0 <= ri < ndwi_clean.shape[0] and 0 <= ci < ndwi_clean.shape[1]:
                    ndwi_samples.append(float(ndwi_clean[ri, ci]))
            ndwi_mean = float(np.mean(ndwi_samples)) if ndwi_samples else 0.0

            # == Composite building score (7 components, weights sum=1) ==
            rect_score    = min(rectangularity / 0.70, 1.0)
            compact_score = min(compactness / 0.50, 1.0)
            solid_score   = min(solidity / 0.75, 1.0)
            edge_score    = min(edge_sharpness * 10.0, 1.0)
            prob_score    = min(row["prob_mean"] / 0.60, 1.0)

            # Size: bell curve -- buildings are typically 30-5000 m^2
            a_m2 = row["area_m2"]
            if a_m2 < 30:
                size_score = 0.1
            elif a_m2 <= 5000:
                size_score = 1.0
            elif a_m2 <= 20000:
                size_score = max(0.0, 1.0 - (a_m2 - 5000) / 15000)
            else:
                size_score = 0.0

            # Water penalty: positive NDWI = water-like environment
            water_score = max(0.0, 1.0 - max(ndwi_mean, 0.0) * 5.0)

            building_score = (
                0.18 * rect_score
                + 0.14 * compact_score
                + 0.14 * solid_score
                + 0.13 * edge_score
                + 0.13 * size_score
                + 0.12 * prob_score
                + 0.16 * water_score
            )

            # -- Multi-criteria hard filter ----------------------------
            if area > p["max_footprint_area"]:
                continue
            if compactness < p["min_compactness"]:
                continue
            if aspect_ratio > p["max_aspect_ratio"]:
                continue
            if solidity < p.get("min_solidity", 0.40):
                continue
            if building_score < p["min_building_score"]:
                continue

            # -- Regularise geometry ------------------------------------
            is_rectangular = (
                rectangularity > p["min_rectangularity"]
                and compactness > 0.20
                and 1.0 <= aspect_ratio <= 8.0
            )
            reg_geom = mrr if is_rectangular else ch

            records.append({
                "geometry": reg_geom,
                "area_m2": row["area_m2"],
                "prob_mean": row["prob_mean"],
                "prob_max": row["prob_max"],
                "compactness": round(float(compactness), 4),
                "rectangularity": round(float(rectangularity), 4),
                "solidity": round(float(solidity), 4),
                "aspect_ratio": round(float(aspect_ratio), 2),
                "edge_sharpness": round(float(edge_sharpness), 4),
                "ndwi_mean": round(float(ndwi_mean), 4),
                "building_score": round(float(building_score), 4),
                "is_rectangular": is_rectangular,
            })

        if not records:
            return gpd.GeoDataFrame(
                columns=_EMPTY_COLS, geometry="geometry",
            ).set_crs(footprints.crs or "EPSG:4326")

        gdf = gpd.GeoDataFrame(records, geometry="geometry").set_crs(
            footprints.crs or "EPSG:4326"
        )
        return gdf.sort_values(
            "building_score", ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Geotransform helper
    # ------------------------------------------------------------------

    def _get_geotransform(self):
        """Derive the Affine transform and CRS WKT from S1 stack coordinates."""
        s1 = self.imagery.s1

        # Infer pixel size from coordinate arrays
        y_coords = s1["y"].values
        x_coords = s1["x"].values

        res_x = float(x_coords[1] - x_coords[0])  if len(x_coords) > 1 else 10.0
        res_y = float(y_coords[1] - y_coords[0])  if len(y_coords) > 1 else -10.0

        x_min = float(x_coords[0]) - res_x / 2.0
        y_max = float(y_coords[0]) - res_y / 2.0   # res_y is negative

        transform = rasterio.transform.from_origin(x_min, y_max, abs(res_x), abs(res_y))
        crs_wkt = self.aoi.utm_crs.to_wkt()
        return transform, crs_wkt
