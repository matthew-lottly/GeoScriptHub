/*
 * ══════════════════════════════════════════════════════════════════════
 *  Sub-Canopy Structure Detector  v2.0
 *  SAR–Optical Fusion for Hidden Building Detection in Forested Areas
 * ══════════════════════════════════════════════════════════════════════
 *
 *  Combines Sentinel-1 C-band SAR with Sentinel-2 optical imagery to
 *  locate man-made structures (buildings, rooftops, concrete pads)
 *  that are obscured by forest canopy and therefore invisible in
 *  standard optical composites.
 *
 *  The core idea: radar penetrates canopy and bounces off hard
 *  surfaces in a distinctive way (double-bounce scattering, high
 *  temporal stability, anomalous polarimetric ratio).  By fusing
 *  multiple SAR-derived indicators with an optical forest mask and
 *  terrain correction we can isolate pixels that *look* like forest
 *  in photos but *behave* like built structures in radar.
 *
 *  ─── DETECTION PIPELINE ──────────────────────────────────────────
 *
 *  1. SAR Temporal Stability     — persistent scatterers (low CoV)
 *  2. Polarimetric Ratio         — VH/VV double-bounce separation
 *  3. SAR Texture (GLCM)         — geometric regularity detection
 *  4. Local Backscatter Anomaly  — VV hotspots vs. neighbourhood
 *  5. Optical NDBI Micro-Anomaly — built-up index shift under canopy
 *  6. Forest Mask + Terrain Correction — restrict to valid canopy
 *  7. Weighted Fusion → probability surface
 *  8. Confidence Zones (High / Medium / Low)
 *  9. Morphological Cleanup
 *
 *  ─── VALIDATION SYSTEM ───────────────────────────────────────────
 *
 *  • Click any pixel → per-indicator diagnostic in Console
 *  • Six global test-site presets with navigation panel
 *  • GHSL cross-validation layer for accuracy assessment
 *  • Time-series charts for full AOI and detection zones
 *
 *  ─── HOW TO USE ──────────────────────────────────────────────────
 *
 *  Paste into the GEE Code Editor (code.earthengine.google.com),
 *  adjust the USER PARAMETERS block, click Run.
 *
 * ══════════════════════════════════════════════════════════════════════
 */


// ═══════════════════════════════════════════════════════════════════
//  1. USER PARAMETERS — change these, click Run
// ═══════════════════════════════════════════════════════════════════

/** Maximum cloud-cover percentage for Sentinel-2 images (0–100).
 *  Lower = cleaner optical data but fewer images.                  */
var MAX_CLOUD_COVER = 15;

/** Date range.  A full year gives good SAR temporal statistics;
 *  multi-year improves stability estimates.                        */
var START_DATE = '2022-01-01';
var END_DATE   = '2024-12-31';

/** NDVI threshold that defines "forest".  Pixels with median
 *  NDVI >= this value are treated as forested canopy.              */
var FOREST_NDVI_THRESHOLD = 0.55;

/** NDWI threshold to exclude open water from the forest mask.
 *  Pixels with NDWI > this are removed.                           */
var WATER_NDWI_THRESHOLD = 0.15;

/** Minimum SAR temporal stability (0–1) for the persistent-
 *  scatterer indicator.  Higher = stricter (fewer detections).     */
var STABILITY_FLOOR = 0.70;

/** Kernel radius (pixels) for GLCM texture.  Larger kernels
 *  smooth more but can blur small structures.                      */
var TEXTURE_KERNEL_RADIUS = 3;

/** Kernel radius (pixels) for the local-anomaly detector.
 *  Defines the neighbourhood used to compute local mean/stdDev.    */
var ANOMALY_KERNEL_RADIUS = 15;

/** Standard-deviation multiplier for the backscatter anomaly.
 *  A pixel counts as anomalous if its VV exceeds the local mean
 *  by this many σ.  Lower = more detections, more false positives. */
var ANOMALY_SIGMA = 1.5;

/** Weights for the final fusion score.  MUST sum to 1.0.          */
var W_STABILITY  = 0.30;   // SAR temporal stability
var W_POLRATIO   = 0.20;   // Polarimetric ratio (double-bounce)
var W_TEXTURE    = 0.20;   // GLCM contrast texture
var W_ANOMALY    = 0.20;   // Local backscatter anomaly
var W_OPTICAL    = 0.10;   // Optical canopy-gap micro-anomaly

/** Probability thresholds for the confidence zones.               */
var THRESH_HIGH   = 0.65;  // >= this → High confidence
var THRESH_MEDIUM = 0.45;  // >= this → Medium confidence
                            // below   → Low / no detection

/** Orbit direction for Sentinel-1 filtering.
 *  'ASCENDING'  — morning passes (east-looking geometry)
 *  'DESCENDING' — evening passes (west-looking geometry)
 *  'BOTH'       — combines both (more data, may add geometric noise)
 *  Tip: higher latitudes often have better DESCENDING coverage.    */
var ORBIT_DIRECTION = 'ASCENDING';

/** Maximum terrain slope in degrees.  Steep slopes cause radar
 *  shadow and layover that mimic structure signatures → false
 *  positives.  Pixels steeper than this are excluded.              */
var SLOPE_THRESHOLD = 15;

/** Minimum and maximum VH/VV ratios (linear) for the double-
 *  bounce normalisation.
 *  Buildings ≈ 0.03–0.08,  forest canopy ≈ 0.10–0.30.
 *  Widen for diverse biomes;  narrow for tropical forest.          */
var POL_RATIO_MIN = 0.02;
var POL_RATIO_MAX = 0.30;

/** Study area — default: tropical forest edge in Petén, Guatemala
 *  (known area with settlements hidden beneath dense canopy).
 *  Replace with your own ee.Geometry for a different region.       */
var AOI = ee.Geometry.Polygon([
  [
    [-90.20, 17.30],
    [-89.80, 17.30],
    [-89.80, 16.95],
    [-90.20, 16.95],
    [-90.20, 17.30]
  ]
]);


// ═══════════════════════════════════════════════════════════════════
//  2. WEIGHT VALIDATION
// ═══════════════════════════════════════════════════════════════════

var weightSum = W_STABILITY + W_POLRATIO + W_TEXTURE + W_ANOMALY + W_OPTICAL;
if (Math.abs(weightSum - 1.0) > 0.001) {
  print('⚠ WARNING: Fusion weights sum to ' + weightSum.toFixed(3) +
        ' — expected 1.0.  Results may be incorrectly scaled.');
  print('  Adjust W_STABILITY + W_POLRATIO + W_TEXTURE + W_ANOMALY + W_OPTICAL');
}


// ═══════════════════════════════════════════════════════════════════
//  3. SENTINEL-1 SAR COLLECTION
// ═══════════════════════════════════════════════════════════════════

/**
 * Load Sentinel-1 GRD (Ground Range Detected) IW-mode images
 * with both VV and VH polarisations.  GRD is already terrain-
 * corrected and calibrated to σ⁰ (backscatter coefficient in dB).
 */
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(AOI)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'));

// Apply orbit direction filter unless user selected BOTH
if (ORBIT_DIRECTION !== 'BOTH') {
  s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', ORBIT_DIRECTION));
}

s1 = s1.select(['VV', 'VH']);

print('Sentinel-1 SAR images:', s1.size());

/**
 * Convert dB → linear power for statistical operations.
 * Statistics (mean, stdDev, CoV) are only valid in linear space.
 *
 *   linear = 10^(dB / 10)
 */
function dbToLinear(image) {
  var lin = ee.Image(10).pow(image.divide(10));
  return lin.rename(image.bandNames())
    .copyProperties(image, ['system:time_start']);
}

var s1Linear = s1.map(dbToLinear);


// ═══════════════════════════════════════════════════════════════════
//  4. INDICATOR 1 — SAR TEMPORAL STABILITY
// ═══════════════════════════════════════════════════════════════════

/**
 * Persistent scatterers (buildings, metal roofs, concrete) produce
 * very stable backscatter over time.  Natural surfaces (vegetation,
 * water, bare soil) fluctuate with moisture, growth, wind, etc.
 *
 * We quantify stability as:
 *
 *     Stability = 1 − CoV = 1 − (σ / μ)
 *
 * where CoV = Coefficient of Variation computed on the linear-power
 * VV time stack.  Stability near 1 = perfectly constant (built).
 * Stability near 0 = highly variable (natural).
 *
 * Using linear power (not dB) is critical: dB values are logarithmic
 * and standard deviation in dB space is not physically meaningful
 * for the coefficient of variation.
 */
var vvMean   = s1Linear.select('VV').mean();
var vvStdDev = s1Linear.select('VV').reduce(ee.Reducer.stdDev());

// Guard against division by zero in homogeneous areas
var cov = vvStdDev.divide(vvMean.max(1e-10));
var stability = ee.Image(1).subtract(cov).rename('stability');

// Clamp to [0, 1]
stability = stability.max(0).min(1);

// Score: pixels below the stability floor get 0; above it, linearly
// scaled from 0 to 1.  Example with floor=0.70:
//   stability 0.70 → score 0.0
//   stability 0.85 → score 0.5
//   stability 1.00 → score 1.0
var stabilityScore = stability
  .subtract(STABILITY_FLOOR)
  .divide(ee.Number(1).subtract(STABILITY_FLOOR))
  .max(0).min(1)
  .rename('stability_score');


// ═══════════════════════════════════════════════════════════════════
//  5. INDICATOR 2 — POLARIMETRIC RATIO (DOUBLE-BOUNCE INDEX)
// ═══════════════════════════════════════════════════════════════════

/**
 * The cross-polarisation ratio VH/VV separates scattering mechanisms:
 *
 *  • Volume scattering (forest canopy): strong VH → HIGH VH/VV
 *  • Double-bounce (wall + ground):     strong VV → LOW  VH/VV
 *  • Surface scattering (bare soil):    low both  → moderate ratio
 *
 * Buildings under canopy produce localised double-bounce anomalies
 * where VV is disproportionately strong relative to VH.
 *
 * We compute the temporal mean in LINEAR space (not dB) and invert
 * the ratio so high score = more double-bounce = more building-like.
 *
 * The normalisation bounds (POL_RATIO_MIN / POL_RATIO_MAX) are
 * user-configurable for different biomes.
 */
var vhMean = s1Linear.select('VH').mean();
var polRatio = vhMean.divide(vvMean.max(1e-10)).rename('pol_ratio');

var doubleBounceScore = ee.Image(POL_RATIO_MAX).subtract(polRatio)
  .divide(POL_RATIO_MAX - POL_RATIO_MIN)
  .max(0).min(1)
  .rename('double_bounce_score');


// ═══════════════════════════════════════════════════════════════════
//  6. INDICATOR 3 — SAR TEXTURE (GLCM CONTRAST)
// ═══════════════════════════════════════════════════════════════════

/**
 * Man-made structures introduce geometric regularity and sharp
 * edges into the backscatter field.  The Gray-Level Co-occurrence
 * Matrix (GLCM) captures this as high *contrast* in a local window.
 *
 * We compute GLCM on the temporal median VV image (in dB).
 * GEE's glcmTexture expects integer-typed input, so we rescale
 * the dB values to an 8-bit range first.
 */
var vvMedianDb = s1.select('VV').median();

// Rescale dB → 0–255 (typical forest VV range: −25 to 0 dB)
var vvScaled = vvMedianDb
  .unitScale(-25, 0)
  .multiply(255)
  .toUint8()
  .rename('VV_scaled');

var glcm = vvScaled.glcmTexture({ size: TEXTURE_KERNEL_RADIUS });
var contrast = glcm.select('VV_scaled_contrast').rename('glcm_contrast');

// Normalise to [0, 1] using a 2nd–98th percentile stretch so that
// outliers don't compress the useful range.
var contrastPcts = contrast.reduceRegion({
  reducer: ee.Reducer.percentile([2, 98]),
  geometry: AOI,
  scale: 10,
  maxPixels: 1e8,
  bestEffort: true
});

var contrastMin = ee.Number(contrastPcts.get('glcm_contrast_p2'));
var contrastMax = ee.Number(contrastPcts.get('glcm_contrast_p98'));

var textureScore = contrast
  .subtract(contrastMin)
  .divide(contrastMax.subtract(contrastMin).max(1e-10))
  .max(0).min(1)
  .rename('texture_score');


// ═══════════════════════════════════════════════════════════════════
//  7. INDICATOR 4 — LOCAL BACKSCATTER ANOMALY
// ═══════════════════════════════════════════════════════════════════

/**
 * A building under trees creates a localised VV hotspot relative
 * to the surrounding forest.  We compute a z-score for each pixel
 * against its circular neighbourhood:
 *
 *     z = (pixel − local_mean) / local_stdDev
 *
 * Pixels with z > ANOMALY_SIGMA are flagged as anomalous.
 *
 * We use the median VV image in dB.  Although dB is logarithmic,
 * it is appropriate here because we are looking for *relative*
 * local departures, and dB stabilises variance across brightness
 * levels (homoscedasticity).
 */
var vvMeanDb = s1.select('VV').mean();
var anomalyKernel = ee.Kernel.circle(ANOMALY_KERNEL_RADIUS, 'pixels');
var localMean = vvMeanDb.reduceNeighborhood(ee.Reducer.mean(),   anomalyKernel);
var localStd  = vvMeanDb.reduceNeighborhood(ee.Reducer.stdDev(), anomalyKernel);

// Guard against division by zero in perfectly homogeneous patches
var zScore = vvMeanDb.subtract(localMean)
  .divide(localStd.max(0.01))
  .rename('z_score');

// Score mapping: z = ANOMALY_SIGMA → 0.5,  z = 2×ANOMALY_SIGMA → 1.0
var anomalyScore = zScore
  .subtract(ANOMALY_SIGMA)
  .divide(ANOMALY_SIGMA)
  .max(0).min(1)
  .rename('anomaly_score');


// ═══════════════════════════════════════════════════════════════════
//  8. SENTINEL-2 COLLECTION + INDICATOR 5 — OPTICAL NDBI
// ═══════════════════════════════════════════════════════════════════

/**
 * Even dense forest has subtle spectral clues above hidden structures:
 *
 *  • Micro-reduction in canopy NDVI (heat island, altered hydrology)
 *  • Elevated NDBI (Normalized Difference Built-up Index), even
 *    slightly, from mixed pixels where roof material blends with
 *    surrounding canopy.
 *
 * We compute:
 *   NDBI = (SWIR1 − NIR) / (SWIR1 + NIR)   [Sentinel-2 B11 & B8]
 *
 * In pure forest NDBI is strongly negative.  A mixed pixel containing
 * roof material pushes NDBI higher.  We capture this relative shift
 * as a local z-score anomaly against the forest baseline.
 */
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER));

print('Sentinel-2 optical images:', s2.size());

/**
 * Cloud mask using the Scene Classification Layer (SCL).
 * Removes cloud shadow (3), cloud medium (8), cloud high (9),
 * and cirrus (10).
 */
function maskS2Clouds(image) {
  var scl = image.select('SCL');
  var clear = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(clear);
}

var s2clean = s2.map(maskS2Clouds);
var s2Median = s2clean.median().clip(AOI);

// Derived indices
var ndvi = s2Median.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = s2Median.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndbi = s2Median.normalizedDifference(['B11', 'B8']).rename('NDBI');

// Local NDBI anomaly: z-score against circular neighbourhood
var ndbiKernel   = ee.Kernel.circle(ANOMALY_KERNEL_RADIUS, 'pixels');
var ndbiLocal    = ndbi.reduceNeighborhood(ee.Reducer.mean(),   ndbiKernel);
var ndbiLocalStd = ndbi.reduceNeighborhood(ee.Reducer.stdDev(), ndbiKernel);

// Guard against division by zero in spectrally homogeneous areas
var ndbiAnomaly = ndbi.subtract(ndbiLocal)
  .divide(ndbiLocalStd.max(0.001))
  .max(0).min(3).divide(3)
  .rename('optical_score');


// ═══════════════════════════════════════════════════════════════════
//  9. FOREST MASK + TERRAIN CORRECTION
// ═══════════════════════════════════════════════════════════════════

/**
 * Restrict detections to pixels that are:
 *   (a) Forested in optical data (high NDVI, not water)
 *   (b) On gentle terrain (slope < threshold)
 *
 * The forest mask ensures we only flag *hidden* structures — if a
 * building is openly visible in optical imagery, standard methods
 * already detect it.
 *
 * The terrain mask removes steep slopes where radar shadow and
 * layover produce strong backscatter anomalies that look like
 * buildings but are purely topographic artefacts.
 */
var forestMask = ndvi.gte(FOREST_NDVI_THRESHOLD)
  .and(ndwi.lt(WATER_NDWI_THRESHOLD))
  .rename('forest');

// Copernicus GLO-30 DEM for slope calculation
var dem   = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .filterBounds(AOI)
  .select('DEM')
  .mosaic();
var slope = ee.Terrain.slope(dem);
var slopeMask = slope.lt(SLOPE_THRESHOLD);

// Combined mask: forest AND gentle slope
forestMask = forestMask.and(slopeMask).rename('forest');


// ═══════════════════════════════════════════════════════════════════
//  10. FUSION — HIDDEN STRUCTURE PROBABILITY
// ═══════════════════════════════════════════════════════════════════

/**
 * Weighted sum of all five indicators → single probability surface.
 *
 *   P(structure) = w₁·stability + w₂·doubleBounce + w₃·texture
 *                + w₄·anomaly   + w₅·optical
 *
 * Masked to forested, gently-sloped areas only.
 */
var fusionScore = stabilityScore.multiply(W_STABILITY)
  .add(doubleBounceScore.multiply(W_POLRATIO))
  .add(textureScore.multiply(W_TEXTURE))
  .add(anomalyScore.multiply(W_ANOMALY))
  .add(ndbiAnomaly.multiply(W_OPTICAL))
  .rename('structure_probability');

var hiddenStructureProb = fusionScore.updateMask(forestMask);


// ═══════════════════════════════════════════════════════════════════
//  11. CONFIDENCE ZONES
// ═══════════════════════════════════════════════════════════════════

/**
 * Classify the probability surface into discrete confidence levels:
 *   3 = High   (P ≥ 0.65)  — very likely a hidden structure
 *   2 = Medium (P ≥ 0.45)  — possible, needs ground truthing
 *   1 = Low    (P < 0.45)  — weak signal
 */
var confidenceZones = ee.Image(0)
  .where(hiddenStructureProb.gte(0),              1)
  .where(hiddenStructureProb.gte(THRESH_MEDIUM),  2)
  .where(hiddenStructureProb.gte(THRESH_HIGH),    3)
  .updateMask(forestMask)
  .rename('confidence');

var highCount = confidenceZones.eq(3).selfMask()
  .reduceRegion({
    reducer: ee.Reducer.count(), geometry: AOI,
    scale: 10, maxPixels: 1e9, bestEffort: true
  });
var medCount = confidenceZones.eq(2).selfMask()
  .reduceRegion({
    reducer: ee.Reducer.count(), geometry: AOI,
    scale: 10, maxPixels: 1e9, bestEffort: true
  });


// ═══════════════════════════════════════════════════════════════════
//  12. MORPHOLOGICAL CLEANUP
// ═══════════════════════════════════════════════════════════════════

/**
 * Morphological opening (erode → dilate) removes single-pixel
 * salt-and-pepper false positives and keeps contiguous clusters
 * that are more likely to be real structures.
 */
var highMask = hiddenStructureProb.gte(THRESH_MEDIUM);
var morphKernel = ee.Kernel.circle(1, 'pixels');
var opened = highMask
  .focal_min({ kernel: morphKernel })   // erode
  .focal_max({ kernel: morphKernel })   // dilate
  .selfMask();

var cleanDetections = hiddenStructureProb.updateMask(opened)
  .rename('clean_probability');


// ═══════════════════════════════════════════════════════════════════
//  13. GHSL VALIDATION LAYER
// ═══════════════════════════════════════════════════════════════════

/**
 * Cross-validate detections against the Global Human Settlement
 * Layer (GHSL-BUILT-S 2020, 10 m).  This dataset uses multi-
 * temporal Sentinel data and Landsat to map built-up surface
 * fraction globally.
 *
 * Interpretation:
 *   ● Our detection AND GHSL > 0       → Confirmed hidden structure
 *   ● Our detection AND GHSL = 0       → Novel detection (not in GHSL)
 *   ● No detection  AND GHSL > 0       → Possible miss (check manually)
 *   ● No detection  AND GHSL = 0       → True negative (pure forest)
 *
 * GHSL may also miss sub-canopy structures, so "novel detections"
 * are not necessarily false positives — they may be genuinely new
 * findings that SAR-only methods can reveal.
 */
var ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface_total')
  .clip(AOI);

// GHSL pixels with any built-up fraction inside our forest mask
var ghslInForest = ghsl.gt(0).and(forestMask).selfMask()
  .rename('ghsl_built_in_forest');

// Agreement: our HIGH/MEDIUM detections that overlap with GHSL
var agreement = cleanDetections.gt(0).and(ghsl.gt(0)).selfMask()
  .rename('agreed');

// Novel: our detections where GHSL shows nothing
var novel = cleanDetections.gt(0).and(ghsl.eq(0)).selfMask()
  .rename('novel');

// Compute agreement statistics
var agreementCount = agreement.reduceRegion({
  reducer: ee.Reducer.count(), geometry: AOI,
  scale: 10, maxPixels: 1e9, bestEffort: true
});
var novelCount = novel.reduceRegion({
  reducer: ee.Reducer.count(), geometry: AOI,
  scale: 10, maxPixels: 1e9, bestEffort: true
});
var ghslForestCount = ghslInForest.reduceRegion({
  reducer: ee.Reducer.count(), geometry: AOI,
  scale: 10, maxPixels: 1e9, bestEffort: true
});


// ═══════════════════════════════════════════════════════════════════
//  14. VISUALISATION
// ═══════════════════════════════════════════════════════════════════

Map.centerObject(AOI, 13);
Map.setOptions('SATELLITE');

// ── Base layers ────────────────────────────────────────────────────

Map.addLayer(
  s2Median.select(['B4', 'B3', 'B2']),
  { min: 0, max: 2500 },
  'Sentinel-2 True Colour',
  true
);

Map.addLayer(
  forestMask.selfMask(),
  { palette: ['228B22'] },
  'Forest Mask (NDVI ≥ ' + FOREST_NDVI_THRESHOLD + ', slope < ' + SLOPE_THRESHOLD + '°)',
  false
);

Map.addLayer(
  slope.clip(AOI),
  { min: 0, max: 30, palette: ['white', 'orange', 'red'] },
  'Terrain Slope (Copernicus DEM)',
  false
);

// ── Individual SAR indicators (off by default) ─────────────────────

Map.addLayer(
  stability.clip(AOI),
  { min: 0.3, max: 1, palette: ['black', 'yellow', 'white'] },
  '① SAR Temporal Stability',
  false
);

Map.addLayer(
  doubleBounceScore.clip(AOI),
  { min: 0, max: 1,
    palette: ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'] },
  '② Polarimetric Double-Bounce Score',
  false
);

Map.addLayer(
  textureScore.clip(AOI),
  { min: 0, max: 1, palette: ['black', 'cyan', 'white'] },
  '③ GLCM Texture Score',
  false
);

Map.addLayer(
  anomalyScore.clip(AOI),
  { min: 0, max: 1, palette: ['black', 'orange', 'red'] },
  '④ Local Backscatter Anomaly',
  false
);

Map.addLayer(
  ndbiAnomaly.clip(AOI),
  { min: 0, max: 1, palette: ['black', 'pink', 'magenta'] },
  '⑤ Optical NDBI Micro-Anomaly',
  false
);

// ── Primary result layers (on by default) ──────────────────────────

Map.addLayer(
  hiddenStructureProb.clip(AOI),
  {
    min: 0, max: 1,
    palette: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
  },
  'Hidden Structure Probability (raw)',
  false
);

Map.addLayer(
  cleanDetections.clip(AOI),
  {
    min: THRESH_MEDIUM, max: 1,
    palette: ['#fee08b', '#fc8d59', '#d73027']
  },
  '★ Detected Hidden Structures (cleaned)',
  true
);

Map.addLayer(
  confidenceZones.clip(AOI),
  { min: 1, max: 3, palette: ['#fee08b', '#fc8d59', '#d73027'] },
  '★ Confidence Zones (Low / Med / High)',
  true
);

// ── Validation layers ──────────────────────────────────────────────

Map.addLayer(
  ghslInForest,
  { palette: ['00ffff'] },
  'GHSL Built-up in Forest (reference)',
  false
);

Map.addLayer(
  agreement,
  { palette: ['00ff00'] },
  '✓ Confirmed (our detection + GHSL)',
  false
);

Map.addLayer(
  novel,
  { palette: ['ff00ff'] },
  '★ Novel Detection (ours only, not in GHSL)',
  false
);

// ── AOI outline ────────────────────────────────────────────────────

Map.addLayer(
  ee.Image().paint(AOI, 0, 2),
  { palette: ['cyan'] },
  'Study Area Boundary'
);


// ═══════════════════════════════════════════════════════════════════
//  15. CLICK-TO-INSPECT DIAGNOSTIC
// ═══════════════════════════════════════════════════════════════════

/**
 * Click anywhere on the map to see a full per-indicator breakdown
 * printed to the Console.  This is the primary way to validate
 * the detector at any point the user chooses.
 *
 * For each clicked pixel you get:
 *   • Each indicator's raw score  (0–1)
 *   • The fused probability score (0–1)
 *   • The confidence zone label   (High / Medium / Low / None)
 *   • Whether the pixel is classified as forest
 *   • The terrain slope in degrees
 *   • The GHSL built-up fraction  (0–100 %)
 */
var diagnosticStack = stabilityScore
  .addBands(doubleBounceScore)
  .addBands(textureScore)
  .addBands(anomalyScore)
  .addBands(ndbiAnomaly)
  .addBands(hiddenStructureProb)
  .addBands(confidenceZones)
  .addBands(forestMask)
  .addBands(slope.rename('slope'))
  .addBands(ghsl.rename('ghsl'));

Map.onClick(function(coords) {
  var point = ee.Geometry.Point([coords.lon, coords.lat]);

  diagnosticStack.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 10
  }).evaluate(function(v) {
    if (!v) {
      print('⚠ No data at this location (outside S1/S2 coverage).');
      return;
    }

    var conf = v.confidence === 3 ? 'HIGH'
             : v.confidence === 2 ? 'MEDIUM'
             : v.confidence === 1 ? 'LOW'
             : 'NONE';

    var isForest = v.forest === 1 ? 'YES' : 'NO';

    print('═══════════════════════════════════════════════════');
    print('  PIXEL DIAGNOSTIC — '
      + coords.lat.toFixed(5) + '°N, ' + coords.lon.toFixed(5) + '°E');
    print('═══════════════════════════════════════════════════');
    print('① Stability Score      : ' + fmt(v.stability_score));
    print('② Double-Bounce Score  : ' + fmt(v.double_bounce_score));
    print('③ Texture Score        : ' + fmt(v.texture_score));
    print('④ Anomaly Score        : ' + fmt(v.anomaly_score));
    print('⑤ Optical (NDBI) Score : ' + fmt(v.optical_score));
    print('───────────────────────────────────────────────────');
    print('Fused Probability      : ' + fmt(v.structure_probability));
    print('Confidence Zone        : ' + conf);
    print('───────────────────────────────────────────────────');
    print('Forest Pixel           : ' + isForest);
    print('Terrain Slope          : ' + fmt(v.slope) + '°');
    print('GHSL Built-up %        : ' + fmt(v.ghsl));
    print('═══════════════════════════════════════════════════');
  });
});

/** Format a number to 4 decimal places, or show "—" if null. */
function fmt(val) {
  return (val !== null && val !== undefined) ? val.toFixed(4) : '—';
}


// ═══════════════════════════════════════════════════════════════════
//  16. GLOBAL TEST-SITE NAVIGATOR
// ═══════════════════════════════════════════════════════════════════

/**
 * Panel with six curated test sites around the world where sub-
 * canopy structures are known to exist.  Click a button to fly
 * to that location.
 *
 * NOTE: The detection layers are computed only for the current AOI's
 * Sentinel-1/2 footprint.  To run the full analysis at a different
 * site, change the AOI variable at the top and click Run again.
 * The navigator helps you explore the *concept* at different
 * locations using the base satellite imagery + GHSL reference.
 */
var testSiteNames = [
  '1. Petén, Guatemala — jungle settlements',
  '2. Leticia, Colombia — Amazon river towns',
  '3. Ulu Baram, Borneo — logging camps',
  '4. Black Forest, Germany — forest villages',
  '5. Portland Metro, Oregon — forest suburbs',
  '6. Mt Halimun, Java — montane villages'
];

var testSiteCoords = [
  [-90.00, 17.15],
  [-69.94, -4.20],
  [114.80,  3.10],
  [  8.15, 48.10],
  [-122.75, 45.50],
  [106.50, -6.73]
];

var sitePanel = ui.Panel({
  style: { position: 'top-left', padding: '8px 12px', width: '330px' }
});

sitePanel.add(ui.Label('🧪 Test Sites', {
  fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'
}));
sitePanel.add(ui.Label(
  'Click to fly.  To analyse a site, copy its coords into AOI and re-run.',
  { fontSize: '11px', color: '#666', margin: '0 0 8px 0' }
));

for (var i = 0; i < testSiteNames.length; i++) {
  (function(idx) {
    sitePanel.add(ui.Button({
      label: testSiteNames[idx],
      style: { stretch: 'horizontal' },
      onClick: function() {
        Map.setCenter(testSiteCoords[idx][0], testSiteCoords[idx][1], 14);
      }
    }));
  })(i);
}

Map.add(sitePanel);

// ── Preset AOI Library (uncomment one to replace the default) ──────
//
// To run the full detector at a different site, uncomment one of
// these AOI definitions (and comment out the original above), then
// click Run.
//
// var AOI = ee.Geometry.Rectangle([-70.05, -4.30, -69.83, -4.10]);
//     // ↑ Leticia, Colombia — Amazon riparian settlements
//
// var AOI = ee.Geometry.Rectangle([114.65, 2.95, 114.95, 3.25]);
//     // ↑ Ulu Baram, Borneo — forest logging camps
//
// var AOI = ee.Geometry.Rectangle([7.95, 47.95, 8.35, 48.25]);
//     // ↑ Black Forest, Germany — houses under conifer canopy
//
// var AOI = ee.Geometry.Rectangle([-122.90, 45.38, -122.60, 45.58]);
//     // ↑ Portland Metro, Oregon — suburban lots in dense forest
//
// var AOI = ee.Geometry.Rectangle([106.38, -6.82, 106.62, -6.62]);
//     // ↑ Mt Halimun, Java — villages in montane forest


// ═══════════════════════════════════════════════════════════════════
//  17. SAR TIME-SERIES CHARTS
// ═══════════════════════════════════════════════════════════════════

/**
 * Chart A — Mean VV backscatter across the FULL AOI over time.
 * Chart B — Mean VV for HIGH-confidence detection zones only.
 *
 * Compare the two: persistent flat lines in Chart B confirm stable
 * scatterers (buildings).  Seasonal curves indicate forest leakage.
 */

// Chart A: full AOI
var chartFull = ui.Chart.image.series({
  imageCollection: s1.select('VV'),
  region: AOI,
  reducer: ee.Reducer.mean(),
  scale: 100,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Chart A — Mean VV Backscatter (full AOI)',
  hAxis: { title: 'Date' },
  vAxis: { title: 'VV (dB)' },
  lineWidth: 1,
  pointSize: 2,
  series: { 0: { color: '#3b528b' } }
});

print(chartFull);

// Chart B: high-confidence detections only
var highZone = confidenceZones.eq(3).selfMask();
var s1HighMasked = s1.select('VV').map(function(img) {
  return img.updateMask(highZone)
    .copyProperties(img, ['system:time_start']);
});

var chartHigh = ui.Chart.image.series({
  imageCollection: s1HighMasked,
  region: AOI,
  reducer: ee.Reducer.mean(),
  scale: 100,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Chart B — Mean VV Backscatter (high-confidence detections only)',
  hAxis: { title: 'Date' },
  vAxis: { title: 'VV (dB)' },
  lineWidth: 1,
  pointSize: 2,
  series: { 0: { color: '#d73027' } }
});

print(chartHigh);


// ═══════════════════════════════════════════════════════════════════
//  18. CONSOLE SUMMARY
// ═══════════════════════════════════════════════════════════════════

print('══════════════════════════════════════════════════════');
print('  Sub-Canopy Structure Detector v2.0 — Results');
print('══════════════════════════════════════════════════════');
print('Study area         : Custom AOI (see map)');
print('Date range         :', START_DATE, '→', END_DATE);
print('S1 images          :', s1.size());
print('S2 images          :', s2.size());
print('Orbit direction    :', ORBIT_DIRECTION);
print('Cloud cover limit  :', MAX_CLOUD_COVER, '%');
print('Forest NDVI thresh :', FOREST_NDVI_THRESHOLD);
print('Slope threshold    :', SLOPE_THRESHOLD, '°');
print('Stability floor    :', STABILITY_FLOOR);
print('Anomaly sigma      :', ANOMALY_SIGMA);
print('Pol. ratio range   :', POL_RATIO_MIN, '–', POL_RATIO_MAX);
print('');
print('Fusion weights     : (sum = ' + weightSum.toFixed(3) + ')');
print('  Stability        :', W_STABILITY);
print('  Polarimetric     :', W_POLRATIO);
print('  Texture (GLCM)   :', W_TEXTURE);
print('  Local Anomaly    :', W_ANOMALY);
print('  Optical NDBI     :', W_OPTICAL);
print('');
print('Detection counts:');
print('  High-confidence pixels  :', highCount);
print('  Medium-confidence pixels:', medCount);
print('');
print('GHSL Validation:');
print('  GHSL built-up in forest :', ghslForestCount);
print('  Confirmed (ours + GHSL) :', agreementCount);
print('  Novel (ours only)       :', novelCount);
print('');
print('How to test:');
print('  • Click any pixel → per-indicator diagnostic in Console');
print('  • Toggle indicators ①–⑤ in the Layers panel');
print('  • Use the 🧪 Test Sites panel to explore global locations');
print('  • Enable GHSL / Agreement / Novel layers for validation');
print('══════════════════════════════════════════════════════');


// ═══════════════════════════════════════════════════════════════════
//  19. HISTOGRAM — structure probability distribution
// ═══════════════════════════════════════════════════════════════════

var probHist = ui.Chart.image.histogram({
  image: hiddenStructureProb.clip(AOI),
  region: AOI,
  scale: 30,
  maxPixels: 1e8,
  maxBuckets: 50
}).setOptions({
  title: 'Distribution of Hidden-Structure Probability (forested pixels)',
  hAxis: { title: 'Probability Score' },
  vAxis: { title: 'Pixel Count' },
  colors: ['#d73027']
});

print(probHist);


// ═══════════════════════════════════════════════════════════════════
//  20. EXPORT (optional — uncomment to save)
// ═══════════════════════════════════════════════════════════════════

// ── Export probability raster at 10 m ──────────────────────────────
// Export.image.toDrive({
//   image: hiddenStructureProb.clip(AOI).toFloat(),
//   description: 'Hidden_Structure_Probability',
//   folder: 'GEE_Exports',
//   region: AOI,
//   scale: 10,
//   crs: 'EPSG:4326',
//   maxPixels: 1e10
// });

// ── Export confidence zones at 10 m ────────────────────────────────
// Export.image.toDrive({
//   image: confidenceZones.clip(AOI).toByte(),
//   description: 'Hidden_Structure_Confidence_Zones',
//   folder: 'GEE_Exports',
//   region: AOI,
//   scale: 10,
//   crs: 'EPSG:4326',
//   maxPixels: 1e10
// });

// ── Export cleaned detections at 10 m ──────────────────────────────
// Export.image.toDrive({
//   image: cleanDetections.clip(AOI).toFloat(),
//   description: 'Hidden_Structure_Detections_Cleaned',
//   folder: 'GEE_Exports',
//   region: AOI,
//   scale: 10,
//   crs: 'EPSG:4326',
//   maxPixels: 1e10
// });

// ── Export GHSL validation raster at 10 m ──────────────────────────
// Export.image.toDrive({
//   image: agreement.clip(AOI).toByte()
//            .addBands(novel.clip(AOI).toByte()),
//   description: 'Hidden_Structure_GHSL_Validation',
//   folder: 'GEE_Exports',
//   region: AOI,
//   scale: 10,
//   crs: 'EPSG:4326',
//   maxPixels: 1e10
// });
