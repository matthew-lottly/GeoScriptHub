/**
 * Leaflet Widget Generator — Type Definitions
 * =============================================
 * All configuration types consumed by {@link LeafletWidgetGenerator}.
 * Every property has a JSDoc comment explaining its purpose and any
 * placeholder values the user should replace.
 */

/** Supported Leaflet tile provider presets. */
export type TileProvider =
  | "openstreetmap"        // OpenStreetMap standard tiles (free, no key required)
  | "opentopomap"          // OpenTopoMap hillshade tiles (free, no key required)
  | "stadia-alidade-smooth" // Stadia Maps smooth light basemap
  | "custom";              // Provide your own `tileUrl` and `tileAttribution`

/** Initial map view configuration. */
export interface ViewConfig {
  /**
   * Initial map centre latitude.
   * <!-- PLACEHOLDER: Replace with your area of interest latitude,
   *      e.g. 51.505 for London, 40.7128 for New York, -33.8688 for Sydney -->
   */
  lat: number;

  /**
   * Initial map centre longitude.
   * <!-- PLACEHOLDER: Replace with your area of interest longitude,
   *      e.g. -0.09 for London, -74.006 for New York, 151.2093 for Sydney -->
   */
  lng: number;

  /**
   * Initial zoom level (1 = world, 18 = building level).
   * <!-- PLACEHOLDER: 13 is a good city-district zoom. Range: 1–18 -->
   */
  zoom: number;
}

/** A GeoJSON layer to add on top of the basemap. */
export interface GeoJsonLayerConfig {
  /**
   * URL to a GeoJSON file or API endpoint.
   * <!-- PLACEHOLDER: Replace with your GeoJSON endpoint, e.g.
   *      "https://example.com/data/features.geojson"
   *      or use a data: URI for inline data -->
   */
  url: string;

  /**
   * Human-readable layer name shown in the layer control.
   * <!-- PLACEHOLDER: Replace with a descriptive name, e.g. "Hospitals" -->
   */
  name: string;

  /**
   * Hex fill colour for polygon/point features.
   * <!-- PLACEHOLDER: Any CSS colour string, e.g. "#e74c3c", "rgba(0,0,0,0.5)" -->
   */
  fillColor?: string;

  /**
   * Hex stroke colour for polygon/point features.
   * <!-- PLACEHOLDER: e.g. "#c0392b" -->
   */
  color?: string;

  /**
   * Stroke weight in pixels.
   * <!-- PLACEHOLDER: 1-4 for thin outlines, 5+ for thick boundaries -->
   */
  weight?: number;

  /** Initial visibility of this layer (default: true). */
  visible?: boolean;
}

/** Configuration for a single map marker. */
export interface MarkerConfig {
  /** Marker latitude. <!-- PLACEHOLDER: Decimal degrees, e.g. 51.505 --> */
  lat: number;
  /** Marker longitude. <!-- PLACEHOLDER: Decimal degrees, e.g. -0.09 --> */
  lng: number;
  /**
   * Popup HTML content shown when the marker is clicked.
   * <!-- PLACEHOLDER: Any HTML string, e.g. "<b>My Location</b><br>Pop. 10,000" -->
   */
  popupHtml?: string;
}

/** Scale control configuration. */
export interface ScaleControlConfig {
  /** Position of the scale control on the map. */
  position?: "bottomleft" | "bottomright" | "topleft" | "topright";
  /** Display metric scale (metres/kilometres). Default: true. */
  metric?: boolean;
  /** Display imperial scale (feet/miles). Default: false. */
  imperial?: boolean;
}

/** Full configuration object for {@link LeafletWidgetGenerator}. */
export interface WidgetConfig {
  /**
   * CSS selector or HTML id for the map container element.
   * <!-- PLACEHOLDER: Matches the `id` attribute of your map div,
   *      e.g. "map", "#myMap" — must be a unique id on the page -->
   */
  containerId: string;

  /**
   * Map container height.
   * <!-- PLACEHOLDER: Any valid CSS height, e.g. "500px", "80vh", "100%" -->
   */
  height?: string;

  /**
   * Map container width.
   * <!-- PLACEHOLDER: Any valid CSS width, e.g. "100%", "800px" -->
   */
  width?: string;

  /** Initial view (centre + zoom). */
  view: ViewConfig;

  /**
   * Tile provider preset.
   * <!-- PLACEHOLDER: "openstreetmap" | "opentopomap" | "stadia-alidade-smooth" | "custom" -->
   */
  tileProvider?: TileProvider;

  /**
   * Custom tile URL template (used only when tileProvider = "custom").
   * <!-- PLACEHOLDER: Tile URL with {z}/{x}/{y} placeholders, e.g.
   *      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" -->
   */
  tileUrl?: string;

  /**
   * Custom tile attribution (used only when tileProvider = "custom").
   * <!-- PLACEHOLDER: e.g. '© <a href="https://openstreetmap.org">OpenStreetMap</a>' -->
   */
  tileAttribution?: string;

  /**
   * Optional Stadia Maps API key (required for Stadia tile layers in production).
   * <!-- PLACEHOLDER: Get a free key at https://client.stadiamaps.com/ -->
   */
  stadiaApiKey?: string;

  /** GeoJSON overlay layers to add on top of the basemap. */
  geoJsonLayers?: GeoJsonLayerConfig[];

  /** Markers to pin on the map. */
  markers?: MarkerConfig[];

  /** Whether to show a zoom control (default: true). */
  showZoomControl?: boolean;

  /** Whether to show a layer toggle control (default: true when layers > 1). */
  showLayerControl?: boolean;

  /** Scale bar configuration. */
  scaleControl?: ScaleControlConfig;

  /**
   * Whether to generate a self-contained HTML string with Leaflet bundled
   * from CDN (default: false — assumes Leaflet is already on the page).
   * <!-- PLACEHOLDER: Set to true to get a fully portable HTML snippet -->
   */
  selfContained?: boolean;
}
