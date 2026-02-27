/**
 * Map Swiper — Type Definitions
 * ================================
 * Configuration types for {@link MapSwiper}.
 */

/** A MapLibre GL style source — either a full style URL or inline style object. */
export type MapStyleSource = string | Record<string, unknown>;

/** Configuration for one side of the swiper (left or right panel). */
export interface PanelConfig {
  /**
   * MapLibre GL style URL or inline style object for this panel.
   */
  style: MapStyleSource;

  /**
   * Human-readable label shown in the panel badge overlay (optional).
   */
  label?: string;
}

/** Full configuration for {@link MapSwiper}. */
export interface MapSwiperConfig {
  /**
   * CSS selector (with `#`) or plain id string for the container element.
   * The container must have an explicit width and height in CSS.
   */
  containerId: string;

  /** Left/before panel configuration. */
  left: PanelConfig;

  /** Right/after panel configuration. */
  right: PanelConfig;

  /**
   * Initial map centre as `[longitude, latitude]`.
   */
  center: [number, number];

  /**
   * Initial zoom level (0 = world, 22 = building).
   */
  zoom: number;

  /**
   * Initial divider position as a fraction of total width (0.0–1.0).
   */
  initialDividerPosition?: number;

  /**
   * Whether to synchronise pan/zoom between the two maps (default: true).
   */
  syncMaps?: boolean;

  /**
   * Thickness of the divider line in pixels (default: 4).
   */
  dividerWidth?: number;

  /**
   * CSS colour of the divider handle (default: `"#ffffff"`).
   */
  dividerColor?: string;
}
