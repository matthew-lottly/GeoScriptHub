/*
 * ══════════════════════════════════════════════════════════════════════
 *  NDWI Flood-Frequency Mapper — Mississippi River Delta
 * ══════════════════════════════════════════════════════════════════════
 *
 *  Pulls a Sentinel-2 time series over the lower Mississippi delta
 *  (~10 mi / 16 km inside from the coast), computes per-image NDWI,
 *  classifies water pixels, and creates a composite that shows how
 *  often each pixel was flagged as water (flood frequency 0–100 %).
 *
 *  Paste this entire file into the Google Earth Engine Code Editor:
 *      https://code.earthengine.google.com/
 *
 *  ─── USER-ADJUSTABLE PARAMETERS ─────────────────────────────────
 *  All tuneable values live in the block below.  Change them and
 *  click "Run" — no other code changes are needed.
 * ══════════════════════════════════════════════════════════════════════
 */

// ─── User Parameters ───────────────────────────────────────────────

/** Maximum cloud cover percentage (0–100).  Increase to keep more
 *  images; decrease for cleaner data (fewer images).               */
var MAX_CLOUD_COVER = 20;

/** Date range for the time series.                                 */
var START_DATE = '2019-01-01';
var END_DATE   = '2024-12-31';

/** NDWI threshold.  Pixels with NDWI >= this value are classified
 *  as water.  McFeeters (2006) recommends 0; bump up to 0.1–0.3
 *  to be more conservative (fewer false positives in shadows).     */
var NDWI_THRESHOLD = 0.1;

/** Study area — lower Mississippi delta, ~10 miles inland from the
 *  Gulf coast.  The rectangle covers the main passes and
 *  surrounding wetlands.  Adjust coordinates if needed.            */
var AOI = ee.Geometry.Polygon([
  [
    [-89.95, 29.45],   // NW corner
    [-88.85, 29.45],   // NE corner
    [-88.85, 28.85],   // SE corner
    [-89.95, 28.85],   // SW corner
    [-89.95, 29.45]    // close ring
  ]
]);

// ─── Sentinel-2 Collection ─────────────────────────────────────────

/**
 * Load the Sentinel-2 Level-2A (Surface Reflectance) harmonised
 * collection, filtered to the AOI, date range, and cloud cover cap.
 */
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER));

print('Sentinel-2 images in collection:', s2.size());

// ─── Cloud Masking ─────────────────────────────────────────────────

/**
 * Mask clouds and cirrus using the SCL (Scene Classification Layer)
 * band that ships with every S2 L2A product.
 *
 * SCL classes removed:
 *   3 = Cloud shadow
 *   8 = Cloud (medium probability)
 *   9 = Cloud (high probability)
 *  10 = Thin cirrus
 */
function maskS2Clouds(image) {
  var scl = image.select('SCL');
  var clearMask = scl.neq(3)    // not cloud shadow
    .and(scl.neq(8))            // not cloud med
    .and(scl.neq(9))            // not cloud high
    .and(scl.neq(10));          // not cirrus
  return image.updateMask(clearMask);
}

var s2Masked = s2.map(maskS2Clouds);

// ─── NDWI Computation ──────────────────────────────────────────────

/**
 * Compute NDWI per image using the McFeeters formula:
 *
 *     NDWI = (Green − NIR) / (Green + NIR)
 *
 * Sentinel-2 bands:
 *   Green = B3  (560 nm, 10 m resolution)
 *   NIR   = B8  (842 nm, 10 m resolution)
 *
 * Returns a single-band image named "NDWI".
 */
function computeNDWI(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  return ndwi.copyProperties(image, ['system:time_start']);
}

var ndwiCollection = s2Masked.map(computeNDWI);

// ─── Water Classification ──────────────────────────────────────────

/**
 * Classify each pixel as water (1) or non-water (0) based on the
 * NDWI threshold.
 */
function classifyWater(image) {
  var water = image.select('NDWI').gte(NDWI_THRESHOLD).rename('water');
  return water.copyProperties(image, ['system:time_start']);
}

var waterCollection = ndwiCollection.map(classifyWater);

// ─── Flood-Frequency Composite ─────────────────────────────────────

/**
 * Sum all water masks and divide by the count of valid (unmasked)
 * observations to produce a per-pixel frequency from 0 to 1.
 *
 * frequency = (# of times classified as water) / (# of clear observations)
 */
var waterSum   = waterCollection.sum();
var validCount = waterCollection.count();
var frequency  = waterSum.divide(validCount).rename('flood_frequency');

/** Convert to percentage (0–100). */
var frequencyPct = frequency.multiply(100).rename('flood_frequency_pct');

// ─── Median NDWI Composite ────────────────────────────────────────

/** A single median-composite NDWI image for visual reference. */
var ndwiMedian = ndwiCollection.median().clip(AOI);

// ─── Visualisation ─────────────────────────────────────────────────

Map.centerObject(AOI, 11);
Map.setOptions('SATELLITE');

/** Flood-frequency layer — blue = always water, red = rarely water,
 *  transparent = never classified as water.                        */
var freqVis = {
  min: 0,
  max: 100,
  palette: [
    '#ffffcc',   //   0 % — very rare flooding (light yellow)
    '#c7e9b4',   //  ~17 %
    '#7fcdbb',   //  ~33 %
    '#41b6c4',   //  ~50 %
    '#2c7fb8',   //  ~67 %
    '#253494'    // 100 % — permanent water (dark blue)
  ]
};

Map.addLayer(
  frequencyPct.clip(AOI),
  freqVis,
  'Flood Frequency (%)'
);

/** Median NDWI for context. */
Map.addLayer(
  ndwiMedian,
  { min: -0.5, max: 0.8, palette: ['brown', 'white', 'cyan', 'blue'] },
  'Median NDWI',
  false   // initially off
);

/** AOI outline. */
Map.addLayer(
  ee.Image().paint(AOI, 0, 2),
  { palette: ['red'] },
  'Study Area'
);

// ─── Legend (Console) ──────────────────────────────────────────────

print('════════════════════════════════════════');
print('  NDWI Flood-Frequency Mapper');
print('════════════════════════════════════════');
print('Study area      : Lower Mississippi Delta (~10 mi inland)');
print('Date range       :', START_DATE, '→', END_DATE);
print('Max cloud cover  :', MAX_CLOUD_COVER, '%');
print('NDWI threshold   :', NDWI_THRESHOLD);
print('Images after filter:', s2.size());
print('');
print('Flood frequency palette:');
print('  Light yellow = rarely flooded');
print('  Dark blue    = permanent water / always flooded');
print('════════════════════════════════════════');

// ─── Time-Series Chart ─────────────────────────────────────────────

/**
 * Plot the mean NDWI over the AOI through time so you can see
 * seasonal wetting / drying cycles and anomalous flood events.
 */
var ndwiChart = ui.Chart.image.series({
  imageCollection: ndwiCollection,
  region: AOI,
  reducer: ee.Reducer.mean(),
  scale: 100,
  xProperty: 'system:time_start'
})
.setOptions({
  title: 'Mean NDWI Over Time — Mississippi Delta',
  hAxis: { title: 'Date' },
  vAxis: { title: 'Mean NDWI', viewWindow: { min: -0.4, max: 0.6 } },
  lineWidth: 1,
  pointSize: 2,
  series: { 0: { color: '#2166ac' } }
});

print(ndwiChart);

// ─── Export (optional) ─────────────────────────────────────────────

/**
 * Uncomment the block below to export the flood-frequency raster to
 * Google Drive as a GeoTIFF.
 */
// Export.image.toDrive({
//   image: frequencyPct.clip(AOI).toFloat(),
//   description: 'NDWI_Flood_Frequency_Mississippi_Delta',
//   folder: 'GEE_Exports',
//   region: AOI,
//   scale: 10,
//   crs: 'EPSG:4326',
//   maxPixels: 1e10
// });
