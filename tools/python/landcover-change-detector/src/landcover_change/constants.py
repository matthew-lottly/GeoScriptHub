"""Land-cover classes, colours, thresholds, and global constants.

v1.0 — Quantum Land-Cover Change Detector

Eight-class scheme loosely based on Anderson Level II, optimised
for the Houston TX coastal-plain landscape.
"""
from __future__ import annotations

import numpy as np

# ── Target resolution ─────────────────────────────────────────────
TARGET_RESOLUTION: int = 30  # metres — Landsat native

# ── Default AOI (San Jacinto River, East Houston) ─────────────────
DEFAULT_CENTER_LAT: float = 29.75645586726091
DEFAULT_CENTER_LON: float = -95.08540365044576
DEFAULT_BUFFER_KM: float = 3.0
DEFAULT_CRS: str = "EPSG:32615"  # UTM Zone 15N

# ── Cloud-cover ceiling for optical queries ───────────────────────
MAX_CLOUD_COVER: int = 15  # percent

# ── Epoch boundaries ─────────────────────────────────────────────
EPOCHS: dict[str, tuple[str, str]] = {
    "1990s": ("1990-01-01", "1999-12-31"),
    "2000s": ("2000-01-01", "2009-12-31"),
    "2010s": ("2010-01-01", "2019-12-31"),
    "2020s": ("2020-01-01", "2026-12-31"),
}

# ── Land-cover class definitions ─────────────────────────────────
#    ID : (short_name, display_name, hex_colour, NLCD_codes)
LANDCOVER_CLASSES: dict[int, tuple[str, str, str, list[int]]] = {
    0: ("water",        "Water",              "#2563eb", [11]),
    1: ("wetland",      "Wetland",            "#06b6d4", [90, 95]),
    2: ("forest",       "Forest / Tree",      "#16a34a", [41, 42, 43]),
    3: ("shrub_grass",  "Shrub / Grassland",  "#84cc16", [51, 52, 71, 72, 73, 74]),
    4: ("agriculture",  "Agriculture",        "#eab308", [81, 82]),
    5: ("dev_low",      "Developed (Low)",    "#f97316", [21, 22]),
    6: ("dev_high",     "Developed (High)",   "#dc2626", [23, 24]),
    7: ("barren",       "Barren / Bare Soil", "#a8a29e", [31]),
}

NUM_CLASSES: int = len(LANDCOVER_CLASSES)

# Convenience arrays
CLASS_IDS: list[int] = list(LANDCOVER_CLASSES.keys())
CLASS_NAMES: list[str] = [v[0] for v in LANDCOVER_CLASSES.values()]
CLASS_LABELS: list[str] = [v[1] for v in LANDCOVER_CLASSES.values()]
CLASS_COLOURS: list[str] = [v[2] for v in LANDCOVER_CLASSES.values()]

# NLCD → our class mapping
NLCD_TO_CLASS: dict[int, int] = {}
for cid, (_, _, _, nlcd_codes) in LANDCOVER_CLASSES.items():
    for nlcd in nlcd_codes:
        NLCD_TO_CLASS[nlcd] = cid

# ── Matplotlib colour map (for raster display) ───────────────────
CLASS_CMAP_RGBA: np.ndarray = np.array(
    [
        [0.149, 0.388, 0.922, 1.0],  # Water — blue
        [0.024, 0.714, 0.831, 1.0],  # Wetland — cyan
        [0.086, 0.639, 0.290, 1.0],  # Forest — green
        [0.518, 0.800, 0.086, 1.0],  # Shrub/grass — lime
        [0.918, 0.702, 0.031, 1.0],  # Agriculture — yellow
        [0.976, 0.451, 0.086, 1.0],  # Dev Low — orange
        [0.863, 0.149, 0.149, 1.0],  # Dev High — red
        [0.659, 0.635, 0.624, 1.0],  # Barren — stone
    ],
    dtype="float32",
)

# ── Quantum parameters ───────────────────────────────────────────
N_QUBITS: int = 4
HILBERT_DIM: int = 2 ** N_QUBITS  # 16
ENTANGLEMENT_STRENGTH: float = float(np.pi / 4)

# ── Spectral index thresholds ────────────────────────────────────
NDVI_FOREST_THRESH: float = 0.50
NDVI_VEG_THRESH: float = 0.25
NDWI_WATER_THRESH: float = 0.10
NDBI_URBAN_THRESH: float = 0.05
BSI_BARREN_THRESH: float = 0.15

# ── SAR thresholds (dB) ─────────────────────────────────────────
SAR_WATER_VV_THRESH: float = -15.0
SAR_URBAN_VV_THRESH: float = -5.0
SAR_FOREST_VH_VV_THRESH: float = -6.0

# ── Physics: allowed transitions matrix ──────────────────────────
#    1 = transition allowed in a single year, 0 = forbidden
#    Rows = from class, Cols = to class
TRANSITION_ALLOWED: np.ndarray = np.array(
    [
        # Wa  We  Fo  SG  Ag  DL  DH  Ba
        [1,  1,  0,  0,  0,  0,  0,  1],  # Water    → wetland, barren ok
        [1,  1,  1,  1,  0,  0,  0,  0],  # Wetland  → water, forest, grass
        [0,  1,  1,  1,  1,  1,  0,  0],  # Forest   → wetland, grass, ag, dev_low
        [0,  1,  1,  1,  1,  1,  0,  1],  # Shrub    → most things
        [0,  0,  0,  1,  1,  1,  0,  1],  # Ag       → grass, dev, barren
        [0,  0,  0,  0,  0,  1,  1,  0],  # Dev Low  → dev high only
        [0,  0,  0,  0,  0,  0,  1,  0],  # Dev High → stays (permanent)
        [0,  0,  0,  1,  1,  1,  0,  1],  # Barren   → grass, ag, dev
    ],
    dtype=np.int8,
)

# ── Compositing settings ─────────────────────────────────────────
MIN_CLEAR_FRACTION: float = 0.50  # reject scenes < 50% clear
NODATA: float = float("nan")

# ── NLCD reference years (for accuracy assessment) ────────────────
NLCD_YEARS: list[int] = [1992, 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]
