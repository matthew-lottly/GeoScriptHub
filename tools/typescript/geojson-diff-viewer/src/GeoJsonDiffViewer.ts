/**
 * GeoJsonDiffViewer
 * =================
 * Leaflet-based map widget that visualises the difference between two
 * GeoJSON FeatureCollections using colour-coded layers and a legend overlay.
 *
 * @example
 * ```ts
 * const viewer = new GeoJsonDiffViewer({ containerId: "diff-map" });
 * viewer.mount();
 * viewer.update(beforeCollection, afterCollection);
 * ```
 */

import * as L from "leaflet";

import { DiffEngine } from "./DiffEngine.js";
import type {
  DiffResult,
  DiffStyle,
  DiffViewerConfig,
  FeatureCollection,
} from "./types.js";

// ---------------------------------------------------------------------------
// Default configuration
// ---------------------------------------------------------------------------

const DEFAULT_TILE_URL =
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

const DEFAULT_ATTRIBUTION =
  '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const DEFAULT_STYLES: Required<DiffStyle> = {
  addedColor: "#27ae60",      // green
  removedColor: "#e74c3c",    // red
  unchangedColor: "#95a5a6",  // grey
  weight: 2,
  fillOpacity: 0.35,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Inject the viewer's base CSS once per page.
 */
function injectStyles(): void {
  const STYLE_ID = "geoscripthub-geojson-diff-viewer-styles";
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.gsh-diff-legend {
  background: white;
  padding: 8px 12px;
  border-radius: 4px;
  box-shadow: 0 1px 4px rgba(0,0,0,.25);
  font: 13px/1.6 sans-serif;
  min-width: 140px;
}
.gsh-diff-legend h4 {
  margin: 0 0 6px;
  font-size: 13px;
  font-weight: 600;
}
.gsh-diff-legend-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.gsh-diff-legend-swatch {
  width: 14px;
  height: 14px;
  border-radius: 2px;
  flex-shrink: 0;
}
`;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Legend control
// ---------------------------------------------------------------------------

/**
 * Build a Leaflet custom control that renders the diff legend.
 */
function buildLegend(
  styles: Required<DiffStyle>,
  result: DiffResult,
): L.Control {
  const LegendControl = L.Control.extend({
    onAdd(): HTMLElement {
      const div = L.DomUtil.create("div", "gsh-diff-legend");
      div.innerHTML = `
        <h4>Diff Legend</h4>
        <div class="gsh-diff-legend-row">
          <span class="gsh-diff-legend-swatch" style="background:${styles.addedColor}"></span>
          <span>Added (${result.added.length})</span>
        </div>
        <div class="gsh-diff-legend-row">
          <span class="gsh-diff-legend-swatch" style="background:${styles.removedColor}"></span>
          <span>Removed (${result.removed.length})</span>
        </div>
        <div class="gsh-diff-legend-row">
          <span class="gsh-diff-legend-swatch" style="background:${styles.unchangedColor}"></span>
          <span>Unchanged (${result.unchanged.length})</span>
        </div>`;
      return div;
    },
  });
  return new LegendControl({ position: "bottomright" });
}

// ---------------------------------------------------------------------------
// Main class
// ---------------------------------------------------------------------------

/**
 * Leaflet map widget for visualising GeoJSON diffs.
 */
export class GeoJsonDiffViewer {
  private readonly cfg: Required<DiffViewerConfig>;
  private readonly styles: Required<DiffStyle>;
  private map: L.Map | null = null;
  private layers: L.GeoJSON[] = [];
  private legendControl: L.Control | null = null;
  private readonly engine: DiffEngine;

  /**
   * @param config - Viewer configuration.  See {@link DiffViewerConfig}.
   * @throws {Error} If `containerId` is empty.
   */
  constructor(config: DiffViewerConfig) {
    if (!config.containerId) {
      throw new Error("GeoJsonDiffViewer: containerId must not be empty.");
    }

    this.cfg = {
      containerId: config.containerId,
      height: config.height ?? "500px",
      width: config.width ?? "100%",
      tileUrl: config.tileUrl ?? DEFAULT_TILE_URL,
      tileAttribution: config.tileAttribution ?? DEFAULT_ATTRIBUTION,
      styles: config.styles ?? {},
      showLegend: config.showLegend ?? true,
      diffOptions: config.diffOptions ?? { matchBy: null },
    };

    this.styles = { ...DEFAULT_STYLES, ...this.cfg.styles };
    this.engine = new DiffEngine(this.cfg.diffOptions);
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Mount the viewer into the DOM.  Must be called before {@link update}.
   *
   * @throws {Error} If the container element is not found.
   */
  public mount(): void {
    if (this.map) return; // already mounted

    injectStyles();

    const container = document.getElementById(this.cfg.containerId);
    if (!container) {
      throw new Error(
        `GeoJsonDiffViewer: element with id="${this.cfg.containerId}" not found.`,
      );
    }

    container.style.height = this.cfg.height;
    container.style.width = this.cfg.width;

    this.map = L.map(container, { preferCanvas: true });

    L.tileLayer(this.cfg.tileUrl, {
      attribution: this.cfg.tileAttribution,
      maxZoom: 19,
    }).addTo(this.map);
  }

  /**
   * Run the diff and re-render the map layers.
   *
   * @param before - The "before" / reference FeatureCollection.
   * @param after  - The "after" / updated FeatureCollection.
   * @throws {Error} If the viewer has not been mounted.
   */
  public update(before: FeatureCollection, after: FeatureCollection): void {
    if (!this.map) {
      throw new Error(
        "GeoJsonDiffViewer: call mount() before calling update().",
      );
    }

    // Remove old layers + legend
    this.layers.forEach((l) => l.remove());
    this.layers = [];
    if (this.legendControl) {
      this.legendControl.remove();
      this.legendControl = null;
    }

    const result = this.engine.diff(before, after);
    const { addedColor, removedColor, unchangedColor, weight, fillOpacity } =
      this.styles;

    const makeLayer = (
      features: typeof result.added,
      color: string,
    ): L.GeoJSON => {
      const fc: FeatureCollection = {
        type: "FeatureCollection",
        features: features.map((df) => df.feature),
      };
      return L.geoJSON(fc, {
        style: () => ({
          color,
          fillColor: color,
          weight,
          fillOpacity,
        }),
        pointToLayer: (_feature: GeoJSON.Feature, latlng: L.LatLng) =>
          L.circleMarker(latlng, {
            radius: 7,
            color,
            fillColor: color,
            fillOpacity,
            weight,
          }),
      });
    };

    const unchangedLayer = makeLayer(result.unchanged, unchangedColor);
    const addedLayer = makeLayer(result.added, addedColor);
    const removedLayer = makeLayer(result.removed, removedColor);

    [unchangedLayer, removedLayer, addedLayer].forEach((l) => {
      if (this.map) {
        l.addTo(this.map);
      }
      this.layers.push(l);
    });

    // Fit bounds to all features combined
    const allBounds: L.LatLngBounds[] = [];
    [unchangedLayer, addedLayer, removedLayer].forEach((l) => {
      try {
        const b = l.getBounds();
        if (b.isValid()) allBounds.push(b);
      } catch {
        // empty layer — skip
      }
    });

    if (allBounds.length > 0) {
      let combined = allBounds[0];
      if (combined) {
        for (let i = 1; i < allBounds.length; i++) {
          const next = allBounds[i];
          if (next) {
            combined = combined.extend(next);
          }
        }
        this.map.fitBounds(combined, { padding: [20, 20] });
      }
    }

    // Add legend
    if (this.cfg.showLegend) {
      this.legendControl = buildLegend(this.styles, result);
      this.legendControl.addTo(this.map);
    }
  }

  /**
   * Unmount the viewer: remove the Leaflet map and clear the container.
   */
  public unmount(): void {
    if (!this.map) return;
    this.map.remove();
    this.map = null;
    this.layers = [];
    this.legendControl = null;
  }
}
