/**
 * Map Swiper — Type Definitions
 * ================================
 * Configuration types for {@link MapSwiper}.
 * Every placeholder-bearing property has a JSDoc comment explaining what to change.
 */

/** A MapLibre GL style source — either a full style URL or inline style object. */
export type MapStyleSource = string | Record<string, unknown>;

/** Configuration for one side of the swiper (left or right panel). */
export interface PanelConfig {
  /**
   * MapLibre GL style URL or inline style object for this panel.
   *
   * <!-- PLACEHOLDER: Set to any MapLibre-compatible style URL, e.g.:
   *      "https://demotiles.maplibre.org/style.json"           (free, no key)
   *      "https://api.maptiler.com/maps/streets/style.json?key=YOUR_KEY"
   *      "https://api.mapbox.com/mapbox-gl-js/v3/mapbox.min.js" (needs access token)
   *      Or pass an inline style object for full control. -->
   */
  style: MapStyleSource;

  /**
   * Human-readable label shown in the panel badge overlay (optional).
   * <!-- PLACEHOLDER: e.g. "Before 2015", "Satellite", "Street Map" -->
   */
  label?: string;
}

/** Full configuration for {@link MapSwiper}. */
export interface MapSwiperConfig {
  /**
   * CSS selector (with #) or plain id string for the container element.
   * The container must have an explicit width and height in CSS.
   *
   * <!-- PLACEHOLDER: Match the id of your <div>, e.g. "swiper", "#mapSwiper" -->
   */
  containerId: string;

  /** Left/before panel configuration. */
  left: PanelConfig;

  /** Right/after panel configuration. */
  right: PanelConfig;

  /**
   * Initial map centre [longitude, latitude].
   * <!-- PLACEHOLDER: Replace with your area of interest, e.g. [-0.09, 51.505] -->
   */
  center: [number, number];

  /**
   * Initial zoom level (0=world, 22=building).
   * <!-- PLACEHOLDER: 12 is a good city-scale zoom -->
   */
  zoom: number;

  /**
   * Initial divider position as a fraction of total width (0.0–1.0).
   * <!-- PLACEHOLDER: 0.5 = middle (default), 0.3 = favour the left panel -->
   */
  initialDividerPosition?: number;

  /**
   * Whether to synchronise pitch and bearing between the two maps.
   * <!-- PLACEHOLDER: true (default) keeps both maps locked together -->
   */
  syncMaps?: boolean;

  /**
   * Thickness of the divider line in pixels.
   * <!-- PLACEHOLDER: 4 is a good default; 2–8 for thin/thick handles -->
   */
  dividerWidth?: number;

  /**
   * CSS colour of the divider handle.
   * <!-- PLACEHOLDER: Any CSS colour, e.g. "#ffffff", "rgba(255,255,255,0.9)" -->
   */
  dividerColor?: string;
}
