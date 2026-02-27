/**
 * GeoJSON Diff Viewer — Type Definitions
 * =========================================
 * Types consumed by {@link DiffEngine} and {@link GeoJsonDiffViewer}.
 */

import type { Feature, FeatureCollection, Geometry } from "geojson";

/** Re-export GeoJSON types for convenience. */
export type { Feature, FeatureCollection, Geometry };

/** The three possible states of a GeoJSON feature after diffing. */
export type DiffStatus = "added" | "removed" | "unchanged";

/** A GeoJSON Feature annotated with its diff status. */
export interface DiffFeature<G extends Geometry = Geometry> {
  /** The original GeoJSON Feature. */
  feature: Feature<G>;
  /** Diff status assigned by the engine. */
  status: DiffStatus;
}

/** Result produced by {@link DiffEngine.diff}. */
export interface DiffResult {
  /** Features present in `b` but not `a`. */
  added: DiffFeature[];
  /** Features present in `a` but not `b`. */
  removed: DiffFeature[];
  /** Features present in both `a` and `b` (unchanged). */
  unchanged: DiffFeature[];
}

/** Options for {@link DiffEngine}. */
export interface DiffEngineOptions {
  /**
   * Property name(s) used to match features across the two collections.
   * If a feature in `a` has `properties.id === 42`, a feature
   * in `b` with the same `id` value is considered the same feature.
   *
   * If `null` (default), features are matched by their serialised geometry
   * string (coordinate-level comparison, regardless of properties).
   */
  matchBy: string | string[] | null;
}

/** Visual style settings per diff status. */
export interface DiffStyle {
  /**
   * Fill + stroke colour for added features.
   */
  addedColor: string;

  /**
   * Fill + stroke colour for removed features.
   */
  removedColor: string;

  /**
   * Fill + stroke colour for unchanged features.
   */
  unchangedColor: string;

  /** Stroke weight in pixels (default: 2). */
  weight?: number;

  /** Fill opacity (default: 0.35). */
  fillOpacity?: number;
}

/** Full configuration for {@link GeoJsonDiffViewer}. */
export interface DiffViewerConfig {
  /**
   * CSS id of the container element (without `#`).
   */
  containerId: string;

  /**
   * Container height (CSS string, e.g. `"500px"`).
   */
  height?: string;

  /**
   * Container width (CSS string, e.g. `"100%"`).
   */
  width?: string;

  /**
   * Leaflet tile URL template.
   */
  tileUrl?: string;

  /**
   * Tile attribution HTML.
   */
  tileAttribution?: string;

  /** Visual styles for each diff status. */
  styles?: Partial<DiffStyle>;

  /** Whether to show a legend overlay (default: true). */
  showLegend?: boolean;

  /** Options passed to {@link DiffEngine}. */
  diffOptions?: DiffEngineOptions;
}
