/**
 * Leaflet Widget Generator — Type Definitions
 * =============================================
 * All configuration types consumed by {@link LeafletWidgetGenerator}.
 */

/** Supported Leaflet tile provider presets. */
export type TileProvider =
  | "openstreetmap"        // OpenStreetMap standard tiles (free, no key required)
  | "opentopomap"          // OpenTopoMap hillshade tiles (free, no key required)
  | "stadia-alidade-smooth" // Stadia Maps smooth light basemap
  | "custom";              // Provide your own `tileUrl` and `tileAttribution`

/** Initial map view configuration. */
export interface ViewConfig {
  /** Initial map centre latitude (e.g. `51.505` for London). */
  lat: number;

  /** Initial map centre longitude (e.g. `-0.09` for London). */
  lng: number;

  /** Initial zoom level (1 = world, 18 = building level). */
  zoom: number;
}

/** A GeoJSON layer to add on top of the basemap. */
export interface GeoJsonLayerConfig {
  /** URL to a GeoJSON file or API endpoint. */
  url: string;

  /** Human-readable layer name shown in the layer control. */
  name: string;

  /** Hex fill colour for polygon/point features. */
  fillColor?: string;

  /** Hex stroke colour for polygon/point features. */
  color?: string;

  /** Stroke weight in pixels. */
  weight?: number;

  /** Initial visibility of this layer (default: true). */
  visible?: boolean;
}

/** Configuration for a single map marker. */
export interface MarkerConfig {
  /** Marker latitude (decimal degrees). */
  lat: number;
  /** Marker longitude (decimal degrees). */
  lng: number;
  /** Popup HTML content shown when the marker is clicked. */
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
  /** CSS selector or HTML id for the map container element. */
  containerId: string;

  /** Map container height (e.g. `"500px"`, `"80vh"`). */
  height?: string;

  /** Map container width (e.g. `"100%"`, `"800px"`). */
  width?: string;

  /** Initial view (centre + zoom). */
  view: ViewConfig;

  /** Tile provider preset. */
  tileProvider?: TileProvider;

  /** Custom tile URL template (used only when `tileProvider = "custom"`). */
  tileUrl?: string;

  /** Custom tile attribution (used only when `tileProvider = "custom"`). */
  tileAttribution?: string;

  /** Optional Stadia Maps API key (required for Stadia tile layers in production). */
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
   */
  selfContained?: boolean;
}
