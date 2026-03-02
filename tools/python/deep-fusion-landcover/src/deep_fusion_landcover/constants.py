"""constants.py — Central-Texas 12-class scheme, sensor configs, and algorithm thresholds.

All tunable parameters for the deep-fusion landcover pipeline live here
so that the rest of the codebase never contains magic numbers.
"""
from __future__ import annotations

# ── Austin metro AOI ──────────────────────────────────────────────────────────
AUSTIN_CENTER_LAT: float = 30.2672   # °N  (Capitol building)
AUSTIN_CENTER_LON: float = -97.7431  # °E
AUSTIN_BUFFER_KM: float = 55.0       # covers ~110×110 km metro area
AUSTIN_UTM_CRS: str = "EPSG:32614"   # UTM Zone 14N
AUSTIN_WGS84_CRS: str = "EPSG:4326"

# Tight bounding box for STAC queries (minx, miny, maxx, maxy)
AUSTIN_BBOX_WGS84: tuple[float, float, float, float] = (
    -98.20, 29.90, -97.30, 30.60
)

# ── Temporal range ────────────────────────────────────────────────────────────
FIRST_YEAR: int = 1990
LAST_YEAR: int = 2025

# ── Target resolutions ────────────────────────────────────────────────────────
RESOLUTION_CHANGE_M: int = 10    # 10 m — change-history rasters
RESOLUTION_NAIP_M: float = 1.0   # 1 m  — present-day NAIP layer

# ── 12-class Austin scheme ────────────────────────────────────────────────────
#  Code → (display_name, hex_colour, nlcd_equiv_codes)
CLASS_DEFINITIONS: dict[int, tuple[str, str, list[int]]] = {
    1:  ("Open Water",            "#4169E1", [11]),
    2:  ("Emergent Wetland",      "#7BC8F6", [95]),
    3:  ("Woody Wetland",         "#A9C4A0", [90]),
    4:  ("Upland Juniper Forest", "#1A5C2E", [42]),
    5:  ("Upland Oak/Elm Forest", "#2D8A4E", [41, 43]),
    6:  ("Cedar Brush/Shrubland", "#8DB361", [52]),
    7:  ("Grassland/Savannah",    "#D4C27A", [71]),
    8:  ("Agriculture",           "#D2A67A", [81, 82]),
    9:  ("Low Intensity Dev.",    "#F4B183", [21, 22]),
    10: ("High Intensity Dev.",   "#C0392B", [23, 24]),
    11: ("Impervious/Rooftop",    "#8E0000", [31]),   # repurposed barren code
    12: ("Barren/Quarry",         "#C8B9A0", [31]),
}

NUM_CLASSES: int = len(CLASS_DEFINITIONS)
CLASS_NAMES: list[str] = [CLASS_DEFINITIONS[i][0] for i in range(1, NUM_CLASSES + 1)]
CLASS_COLORS: list[str] = [CLASS_DEFINITIONS[i][1] for i in range(1, NUM_CLASSES + 1)]

# NLCD class → Austin class mapping (used for validation)
NLCD_TO_AUSTIN: dict[int, int] = {
    11: 1, 12: 1,
    90: 3, 95: 2,
    41: 5, 42: 4, 43: 5,
    52: 6,
    71: 7, 72: 7, 73: 7, 74: 7,
    81: 8, 82: 8,
    21: 9, 22: 9,
    23: 10, 24: 10,
    31: 12,
}

# ── Sensor / band configurations ──────────────────────────────────────────────
# Planetary Computer collection IDs
PC_COLLECTIONS: dict[str, str] = {
    "landsat":    "landsat-c2-l2",
    "sentinel2":  "sentinel-2-l2a",
    "sentinel1":  "sentinel-1-rtc",
    "naip":       "naip",
    "cop_dem":    "cop-dem-glo-30",
}

# Landsat band names by spacecraft (Collection 2 Level-2 asset names)
LANDSAT_BANDS: dict[str, dict[str, str]] = {
    # Landsat 4/5 TM
    "LT04": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
             "nir": "SR_B4", "swir1": "SR_B5", "tir": "ST_B6", "swir2": "SR_B7"},
    "LT05": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
             "nir": "SR_B4", "swir1": "SR_B5", "tir": "ST_B6", "swir2": "SR_B7"},
    # Landsat 7 ETM+
    "LE07": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
             "nir": "SR_B4", "swir1": "SR_B5", "tir": "ST_B6", "swir2": "SR_B7"},
    # Landsat 8/9 OLI
    "LC08": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
             "nir": "SR_B5", "swir1": "SR_B6", "tir": "ST_B10", "swir2": "SR_B7"},
    "LC09": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
             "nir": "SR_B5", "swir1": "SR_B6", "tir": "ST_B10", "swir2": "SR_B7"},
}

# Sentinel-2 band names (L2A)
S2_BANDS: dict[str, str] = {
    "blue":   "B02",
    "green":  "B03",
    "red":    "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir":    "B08",
    "nir_narrow": "B8A",
    "swir1":  "B11",
    "swir2":  "B12",
    "scl":    "SCL",    # scene classification layer
}

# Sentinel-2 SCL cloud / shadow classes to mask
S2_CLOUD_CLASSES: list[int] = [1, 2, 3, 7, 8, 9, 10, 11]

# Landsat C2 QA_PIXEL bit masks
LANDSAT_CLOUD_BIT: int = 3   # bit 3 = cloud
LANDSAT_SHADOW_BIT: int = 4  # bit 4 = cloud shadow
LANDSAT_FILL_BIT: int = 0    # bit 0 = fill / nodata

# Landsat Collection 2 surface reflectance scale + offset (applied to SR_Bx)
LANDSAT_SR_SCALE: float = 0.0000275
LANDSAT_SR_OFFSET: float = -0.2

# OLI → TM band harmonisation coefficients (Roy et al. 2016)
# Format: band → (slope, intercept)  →  TM = slope * OLI + intercept
OLI_TO_TM_COEFFS: dict[str, tuple[float, float]] = {
    "blue":  (0.8474, 0.0003),
    "green": (0.8483, 0.0088),
    "red":   (0.9047, 0.0061),
    "nir":   (0.8462, 0.0412),
    "swir1": (0.8937, 0.0254),
    "swir2": (0.9071, 0.0172),
}

# ── Quantum VQC configuration ─────────────────────────────────────────────────
N_QUBITS: int = 8
VQC_LAYERS: int = 3           # variational ansatz depth
VQC_FEATURE_DIM: int = 8      # PCA target dimensionality fed to VQC
VQC_LR: float = 0.01
VQC_EPOCHS: int = 80
ENTANGLEMENT_STRENGTH: float = 0.95

# ── CNN encoder configuration ─────────────────────────────────────────────────
CNN_CHIP_SIZE: int = 256       # px — tile size fed to encoder
CNN_OVERLAP_PX: int = 32       # overlap between adjacent CNN tiles
CNN_IN_CHANNELS: int = 9       # S2×6 + SAR×2 + nDSM×1
CNN_EMBEDDING_DIM: int = 128   # projected embedding size
CNN_BACKBONE: str = "resnet50" # torchgeo backbone name

# ── Spectral index thresholds ─────────────────────────────────────────────────
NDVI_FOREST_THRESH: float = 0.55
NDVI_VEG_THRESH: float = 0.35
NDWI_WATER_THRESH: float = 0.1
MNDWI_WATER_THRESH: float = 0.2
NDBI_URBAN_THRESH: float = 0.1
BSI_BARREN_THRESH: float = 0.15
SAR_DOUBLBOUNCE_VV_DB: float = -8.0    # dB threshold for building double-bounce
SAR_WATER_VV_DB: float = -18.0         # dB threshold for specular water
SAR_FLOODED_FOREST_VV_DB: float = -15.0

# ── LiDAR configuration ───────────────────────────────────────────────────────
LIDAR_PENETRATION_THRESH: float = 0.35
LIDAR_CANOPY_HEIGHT_M: float = 5.0     # min height to be classified as canopy
LIDAR_CHIPS_BUFFER_M: float = 100.0    # buffer around AOI to avoid edge effects

# ── Sub-canopy detection weights ─────────────────────────────────────────────
SUBCANOPY_WEIGHTS: dict[str, float] = {
    "lidar_penetration":   0.35,
    "sar_double_bounce":   0.30,
    "thermal_anomaly":     0.20,
    "hand_inundation":     0.15,
}
SUBCANOPY_HIGH_THRESH: float = 0.60
SUBCANOPY_MED_THRESH: float = 0.40

# ── Terrain ───────────────────────────────────────────────────────────────────
HAND_FLOOD_THRESH_M: float = 2.0       # Height Above Nearest Drainage
SLOPE_FOREST_MAX_DEG: float = 45.0     # steep slopes unlikely to be flooded

# ── Dask / tiling ─────────────────────────────────────────────────────────────
TILE_SIZE_PX: int = 256
TILE_OVERLAP_PX: int = 32
DASK_WORKERS: int = 4
DASK_THREADS_PER_WORKER: int = 2

# ── Feature engineering ───────────────────────────────────────────────────────
GLCM_WINDOW_PX: int = 7        # GLCM window size
PHENOLOGY_WINDOW_DAYS: int = 16  # smoothing window for seasonal NDVI
N_FEATURE_BANDS: int = 65      # base feature count per annual composite (before temporal)
N_TEMPORAL_FEATURES: int = 7   # NDVI statistics across years
N_LIDAR_FEATURES: int = 9
N_TERRAIN_FEATURES: int = 9
N_TEXTURE_FEATURES: int = 21
TOTAL_FEATURES: int = 150      # approximate — actual depends on data availability

# ── Accuracy assessment ───────────────────────────────────────────────────────
NLCD_VALIDATION_YEARS: list[int] = [2001, 2006, 2011, 2016, 2021]
VALIDATION_SAMPLE_N: int = 50_000   # stratified per-class sample size

# ── NLCD download base URL ────────────────────────────────────────────────────
NLCD_S3_BASE: str = "https://s3-us-west-2.amazonaws.com/mrlc"

# ── USGS 3DEP WCS endpoint ────────────────────────────────────────────────────
THREEDEP_WCS_URL: str = (
    "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/"
    "ImageServer/WCSServer"
)
THREEDEP_STAC_URL: str = "https://stac.opentopography.org/api/stac/v1"

# ── Planetary Computer endpoint ───────────────────────────────────────────────
PC_STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
