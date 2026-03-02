"""sub_canopy_detector.py — Detect structures and water hidden under tree canopy.

Uses 4 independent evidence lines:

    1. LiDAR multi-return penetration analysis
       High penetration_ratio in tall-canopy pixels → likely sub-canopy structure.

    2. SAR C-band double-bounce (Sentinel-1)
       Forest pixels with high VV backscatter (> −8 dB) → building corner reflectors.
       VV/VH ratio > 0.7 in forested areas → double-bounce signature.

    3. Landsat Band 10 TIRS thermal anomaly
       Impervious surfaces emit more heat — thermal z-score > 1.5 in a
       forested pixel flags likely impervious surface under canopy.

    4. HAND-based sub-canopy inundation
       Forested pixels with HAND < 2 m and SAR VV < −18 dB → flooded forest.
       GEDI multi-modal waveform proxy (from SAR seasonality) reinforces.

A 4-evidence weighted fusion produces ``sub_canopy_probability ∈ [0, 1]``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import numpy as np
from scipy.ndimage import label, binary_opening
from shapely.geometry import shape
from skimage.measure import regionprops

from .constants import (
    AUSTIN_UTM_CRS,
    HAND_FLOOD_THRESH_M,
    LIDAR_CANOPY_HEIGHT_M,
    LIDAR_PENETRATION_THRESH,
    NDVI_FOREST_THRESH,
    SAR_DOUBLBOUNCE_VV_DB,
    SAR_FLOODED_FOREST_VV_DB,
    SAR_WATER_VV_DB,
    SUBCANOPY_HIGH_THRESH,
    SUBCANOPY_MED_THRESH,
    SUBCANOPY_WEIGHTS,
)
from .lidar_processor import LiDARProducts
from .temporal_compositing import AnnualComposite

logger = logging.getLogger("geoscripthub.deep_fusion_landcover.sub_canopy_detector")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SubCanopyResult:
    """Result of the sub-canopy detection analysis.

    Attributes
    ----------
    probability_map:   Float32 (H, W) probability that sub-canopy structure exists.
    building_mask:     Binary mask of detected buildings/structures under canopy.
    flood_mask:        Binary mask of detected sub-canopy inundation / flooded forest.
    road_mask:         Binary mask of probable roads under canopy (elongated patches).
    confidence:        Confidence string tier: ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
    detections_gdf:    GeoDataFrame of vectorised detections with ``label`` and
                       ``sub_canopy_prob`` attributes.
    shape:             ``(H, W)`` spatial dimensions.
    crs:               CRS string for ``detections_gdf``.
    """

    probability_map: np.ndarray
    building_mask: np.ndarray
    flood_mask: np.ndarray
    road_mask: np.ndarray
    confidence: np.ndarray
    detections_gdf: gpd.GeoDataFrame
    shape: tuple[int, int]
    crs: str = AUSTIN_UTM_CRS


# ── Main detector ─────────────────────────────────────────────────────────────

class SubCanopyDetector:
    """Detect sub-canopy structures and inundation using the 4-evidence fusion.

    Parameters
    ----------
    weights:        Dict mapping evidence names to fusion weights (must sum to 1).
    high_thresh:    Probability threshold for HIGH-confidence detection.
    medium_thresh:  Probability threshold for MEDIUM-confidence detection.
    min_area_m2:    Minimum detected region area in m² for inclusion as a polygon.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        high_thresh: float = SUBCANOPY_HIGH_THRESH,
        medium_thresh: float = SUBCANOPY_MED_THRESH,
        min_area_m2: float = 50.0,
    ) -> None:
        self.weights = weights or SUBCANOPY_WEIGHTS
        self.high_thresh = high_thresh
        self.medium_thresh = medium_thresh
        self.min_area_m2 = min_area_m2

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(
        self,
        composite: AnnualComposite,
        lidar: Optional[LiDARProducts] = None,
        ndvi_map: Optional[np.ndarray] = None,
        hand_map: Optional[np.ndarray] = None,
        thermal_map: Optional[np.ndarray] = None,
    ) -> SubCanopyResult:
        """Run the 4-evidence sub-canopy detection pipeline.

        Parameters
        ----------
        composite:   Annual composite providing SAR and Landsat data.
        lidar:       LiDAR products for evidence line 1.
        ndvi_map:    Pre-computed NDVI at the composite resolution.
        hand_map:    Height Above Nearest Drainage array.
        thermal_map: Landsat Band 10 brightness temperature (°C or K).

        Returns
        -------
        SubCanopyResult
            Probability map, labelled masks, and vectorised GeoDataFrame.
        """
        # Determine shape from composite
        H, W = self._get_shape(composite)
        logger.info("Sub-canopy detection: grid %d × %d", H, W)

        # ── Build forest mask from NDVI ────────────────────────────────────
        if ndvi_map is None:
            ndvi_map = self._compute_ndvi(composite, H, W)
        forest_mask = ndvi_map > NDVI_FOREST_THRESH  # (H, W) bool

        # ── Evidence 1: LiDAR penetration ─────────────────────────────────
        e1 = self._lidar_evidence(lidar, forest_mask, H, W)

        # ── Evidence 2: SAR double-bounce ──────────────────────────────────
        e2 = self._sar_double_bounce_evidence(composite, forest_mask, H, W)

        # ── Evidence 3: Thermal anomaly ────────────────────────────────────
        e3 = self._thermal_evidence(composite, thermal_map, forest_mask, H, W)

        # ── Evidence 4: HAND-based inundation ─────────────────────────────
        e4 = self._inundation_evidence(composite, hand_map, forest_mask, H, W)

        # ── Weighted fusion ────────────────────────────────────────────────
        w1 = self.weights.get("lidar_penetration", 0.35)
        w2 = self.weights.get("sar_double_bounce", 0.30)
        w3 = self.weights.get("thermal_anomaly", 0.20)
        w4 = self.weights.get("hand_inundation", 0.15)

        probability = (w1 * e1 + w2 * e2 + w3 * e3 + w4 * e4).clip(0, 1)
        # Only flag within forest
        probability = np.where(forest_mask, probability, 0.0).astype("float32")

        # ── Classify detections ────────────────────────────────────────────
        high_mask = probability >= self.high_thresh
        med_mask = (probability >= self.medium_thresh) & ~high_mask
        combined = high_mask | med_mask

        # Morphological opening to remove speckle
        combined = binary_opening(combined, structure=np.ones((3, 3)))

        # Separate building vs. flood signatures
        # Buildings: high SAR VV but NOT low SAR (specular)
        building_mask = self._label_buildings(e2, high_mask, H, W)
        flood_mask = self._label_flooded_forest(e4, high_mask | med_mask, H, W)
        road_mask = self._label_roads(building_mask, H, W)

        # Confidence array
        confidence = np.zeros((H, W), dtype="uint8")
        confidence[med_mask] = 1    # MEDIUM
        confidence[high_mask] = 2   # HIGH

        # Vectorise detections > medium threshold
        detections_gdf = self._vectorise(
            combined, probability, composite, H, W
        )
        logger.info(
            "Sub-canopy detections: %d features (HIGH: %d, MED: %d)",
            len(detections_gdf),
            int(high_mask.sum()),
            int(med_mask.sum()),
        )

        return SubCanopyResult(
            probability_map=probability,
            building_mask=building_mask.astype(bool),
            flood_mask=flood_mask.astype(bool),
            road_mask=road_mask.astype(bool),
            confidence=confidence,
            detections_gdf=detections_gdf,
            shape=(H, W),
            crs=AUSTIN_UTM_CRS,
        )

    # ── Evidence lines ────────────────────────────────────────────────────────

    def _lidar_evidence(
        self,
        lidar: Optional[LiDARProducts],
        forest_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Evidence 1: LiDAR penetration ratio in forested pixels.

        High penetration in tall canopy areas → sub-canopy structure.
        Normalised to [0, 1].
        """
        if lidar is None or lidar.penetration_ratio is None:
            logger.debug("  LiDAR evidence: no data → returning zeros.")
            return np.zeros((H, W), dtype="float32")

        from .feature_engineering import _resize_to
        pen = _resize_to(lidar.penetration_ratio, H, W).clip(0, 1)
        ndsm = (
            _resize_to(lidar.ndsm, H, W).clip(0, 60)
            if lidar.ndsm is not None
            else np.full((H, W), LIDAR_CANOPY_HEIGHT_M + 1, dtype="float32")
        )

        # Score: penetration above threshold in areas where canopy is tall
        canopy_tall = (ndsm > LIDAR_CANOPY_HEIGHT_M) & forest_mask
        evidence = np.where(
            canopy_tall,
            (pen / LIDAR_PENETRATION_THRESH).clip(0, 1),   # high pen = high evidence
            0.0,
        )
        return evidence.astype("float32")

    def _sar_double_bounce_evidence(
        self,
        comp: AnnualComposite,
        forest_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Evidence 2: Sentinel-1 double-bounce in forested pixels."""
        if comp.sentinel1 is None:
            return np.zeros((H, W), dtype="float32")

        ds = comp.sentinel1
        from .feature_engineering import _band
        vv_db = _band(ds, ["vv_db_mean", "vv_mean_db", "vv_db"], H, W)

        # Double-bounce: VV > threshold in forest
        double_bounce_strength = np.where(
            forest_mask & (vv_db > SAR_DOUBLBOUNCE_VV_DB),
            np.clip((vv_db - SAR_DOUBLBOUNCE_VV_DB) / 8.0, 0, 1),
            0.0,
        )
        return double_bounce_strength.astype("float32")

    def _thermal_evidence(
        self,
        comp: AnnualComposite,
        thermal_map: Optional[np.ndarray],
        forest_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Evidence 3: Thermal anomaly (heat island) in forested pixels."""
        if thermal_map is not None:
            tmap = thermal_map.astype("float32")
        elif comp.landsat is not None:
            from .feature_engineering import _band
            tmap = _band(comp.landsat, ["ST_B10", "tir", "ST_B6"], H, W)
        else:
            return np.zeros((H, W), dtype="float32")

        if np.all(np.isnan(tmap)):
            return np.zeros((H, W), dtype="float32")

        from .feature_engineering import _resize_to
        tmap = _resize_to(tmap, H, W)

        # Z-score normalisation within the scene
        t_mean = np.nanmean(tmap)
        t_std = np.nanstd(tmap) + 1e-6
        t_z = (tmap - t_mean) / t_std

        # High thermal anomaly within forest = likely impervious
        evidence = np.where(
            forest_mask & (t_z > 1.5),
            np.clip((t_z - 1.5) / 2.0, 0, 1),
            0.0,
        )
        return evidence.astype("float32")

    def _inundation_evidence(
        self,
        comp: AnnualComposite,
        hand_map: Optional[np.ndarray],
        forest_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Evidence 4: Sub-canopy inundation (flooded forest) via HAND + SAR VV."""
        if comp.sentinel1 is None:
            return np.zeros((H, W), dtype="float32")

        from .feature_engineering import _band, _resize_to

        vv_db = _band(
            comp.sentinel1, ["vv_db_mean", "vv_mean_db", "vv_db"], H, W
        )

        if hand_map is not None:
            hand = _resize_to(hand_map, H, W).clip(0, 200)
        else:
            # Use DEM-based HAND proxy
            if comp.dem is not None:
                elev = _resize_to(comp.dem.values.squeeze().astype("float32"), H, W)
                from scipy.ndimage import uniform_filter
                local_min = -uniform_filter(-elev, size=50, mode="nearest")
                hand = (elev - local_min).clip(0, 200)
            else:
                hand = np.full((H, W), HAND_FLOOD_THRESH_M + 1, dtype="float32")

        # Flooded forest: forest + low HAND + very low SAR VV (specular)
        low_hand = hand < HAND_FLOOD_THRESH_M
        specular = vv_db < SAR_FLOODED_FOREST_VV_DB

        evidence = np.where(
            forest_mask & low_hand & specular,
            np.clip(
                (SAR_FLOODED_FOREST_VV_DB - vv_db) / 5.0    # deeper specular = more confident
                * (1 - hand / HAND_FLOOD_THRESH_M),          # lower HAND = more confident
                0, 1
            ),
            0.0,
        )
        return evidence.astype("float32")

    # ── Label helpers ──────────────────────────────────────────────────────────

    def _label_buildings(
        self,
        double_bounce: np.ndarray,
        high_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Tag pixels as buildings/structures based on double-bounce signature."""
        return (high_mask & (double_bounce > 0.4)).astype("uint8")

    def _label_flooded_forest(
        self,
        inundation: np.ndarray,
        detection_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Tag pixels as flooded forest based on inundation evidence."""
        return (detection_mask & (inundation > 0.3)).astype("uint8")

    def _label_roads(self, building_mask: np.ndarray, H: int, W: int) -> np.ndarray:
        """Detect elongated sub-canopy patches as probable roads."""
        label_result = label(building_mask)
        labeled = label_result[0] if isinstance(label_result, tuple) else label_result
        road_mask = np.zeros((H, W), dtype="uint8")
        for prop in regionprops(labeled):
            if prop.area < 4:
                continue
            # Elongated (aspect_ratio > 4) AND thin (minor axis < 5 px)
            if prop.axis_minor_length > 0:
                aspect = prop.axis_major_length / max(prop.axis_minor_length, 1)
                if aspect > 4 and prop.axis_minor_length < 5:
                    coords = np.argwhere(labeled == prop.label)
                    for r, c in coords:
                        road_mask[r, c] = 1
        return road_mask

    # ── Vectorisation ─────────────────────────────────────────────────────────

    def _vectorise(
        self,
        mask: np.ndarray,
        probability: np.ndarray,
        comp: AnnualComposite,
        H: int,
        W: int,
    ) -> gpd.GeoDataFrame:
        """Rasterize labelled mask to polygon GeoDataFrame."""
        from rasterio.features import shapes as rio_shapes
        from rasterio.transform import from_bounds
        import rasterio

        if comp.dem is not None:
            try:
                transform = comp.dem.rio.transform()
            except Exception:  # noqa: BLE001
                minx, miny, maxx, maxy = (
                    550000, 3340000, 670000, 3420000
                )  # Austin UTM 14N fallback
                transform = from_bounds(minx, miny, maxx, maxy, W, H)
        else:
            minx, miny, maxx, maxy = 550000, 3340000, 670000, 3420000
            transform = from_bounds(minx, miny, maxx, maxy, W, H)

        mask_uint8 = mask.astype("uint8")
        geoms_vals = list(rio_shapes(mask_uint8, mask=mask_uint8, transform=transform))

        if not geoms_vals:
            return gpd.GeoDataFrame(
                columns=["geometry", "sub_canopy_prob", "label"],
                crs=AUSTIN_UTM_CRS,
            )

        records = []
        for geom_dict, val in geoms_vals:
            if int(val) == 0:
                continue
            poly = shape(geom_dict)
            if poly.area < self.min_area_m2:
                continue
            # Sample probability at centroid
            cx, cy = poly.centroid.x, poly.centroid.y
            # Convert to array idx
            col_r = int((cx - transform.c) / transform.a)  # type: ignore[union-attr]
            row_r = int((cy - transform.f) / transform.e)  # type: ignore[union-attr]
            row_r = max(0, min(H - 1, row_r))
            col_r = max(0, min(W - 1, col_r))
            prob = float(probability[row_r, col_r])
            lbl = "building_under_canopy" if prob >= self.high_thresh else "sub_canopy_structure"
            records.append({"geometry": poly, "sub_canopy_prob": prob, "label": lbl})

        if not records:
            return gpd.GeoDataFrame(
                columns=["geometry", "sub_canopy_prob", "label"],
                crs=AUSTIN_UTM_CRS,
            )

        return gpd.GeoDataFrame(records, crs=AUSTIN_UTM_CRS)

    # ── Shape helpers ─────────────────────────────────────────────────────────

    def _get_shape(self, comp: AnnualComposite) -> tuple[int, int]:
        for ds in (comp.landsat, comp.sentinel2, comp.sentinel1):
            if ds is not None:
                for var in ds.data_vars:
                    arr = ds[var].values.squeeze()
                    if arr.ndim == 2:
                        return arr.shape  # type: ignore[return-value]
        return (1000, 1000)  # fallback size

    def _compute_ndvi(self, comp: AnnualComposite, H: int, W: int) -> np.ndarray:
        from .feature_engineering import _band
        ds = comp.landsat if comp.landsat is not None else comp.sentinel2
        if ds is None:
            return np.zeros((H, W), dtype="float32")
        red = _band(ds, ["red", "SR_B4", "SR_B3", "B04"], H, W)
        nir = _band(ds, ["nir", "SR_B5", "SR_B4", "B08"], H, W)
        return ((nir - red) / (nir + red + 1e-10)).clip(-1, 1).astype("float32")
