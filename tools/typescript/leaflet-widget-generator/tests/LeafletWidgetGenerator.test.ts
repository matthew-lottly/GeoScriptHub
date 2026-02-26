/**
 * Tests for LeafletWidgetGenerator
 * ===================================
 * All tests are pure unit tests — no DOM manipulation, no browser required.
 * We inspect the generated HTML/JS string directly.
 */

import { describe, expect, it } from "vitest";
import { LeafletWidgetGenerator } from "../src/LeafletWidgetGenerator.js";
import type { WidgetConfig } from "../src/types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const BASE_CONFIG: WidgetConfig = {
  containerId: "map",
  view: { lat: 51.505, lng: -0.09, zoom: 13 },
};

function gen(overrides: Partial<WidgetConfig> = {}): LeafletWidgetGenerator {
  return new LeafletWidgetGenerator({ ...BASE_CONFIG, ...overrides });
}

// ---------------------------------------------------------------------------
// Construction & validation tests
// ---------------------------------------------------------------------------

describe("LeafletWidgetGenerator — construction", () => {
  it("should construct without error given valid config", () => {
    expect(() => gen()).not.toThrow();
  });

  it("should throw if containerId is empty", () => {
    expect(() => gen({ containerId: "" })).toThrow(/containerId/i);
  });

  it("should throw if lat is out of range", () => {
    expect(() => gen({ view: { lat: 999, lng: 0, zoom: 10 } })).toThrow(/lat/);
  });

  it("should throw if lng is out of range", () => {
    expect(() => gen({ view: { lat: 51, lng: -999, zoom: 10 } })).toThrow(/lng/);
  });

  it("should throw if zoom is out of range", () => {
    expect(() => gen({ view: { lat: 51, lng: -0.09, zoom: 99 } })).toThrow(/zoom/);
  });
});

// ---------------------------------------------------------------------------
// generate() content tests (inline script, no selfContained)
// ---------------------------------------------------------------------------

describe("LeafletWidgetGenerator — generate() inline mode", () => {
  it("should return a <script> tag", () => {
    const output = gen().generate();
    expect(output).toMatch(/^<script>/);
    expect(output).toMatch(/<\/script>$/);
  });

  it("should include the containerId in initialisation", () => {
    const output = gen().generate();
    expect(output).toContain("'map'");
  });

  it("should include map center coordinates", () => {
    const output = gen().generate();
    expect(output).toContain("51.505");
    expect(output).toContain("-0.09");
    expect(output).toContain("zoom: 13");
  });

  it("should include openstreetmap tile URL by default", () => {
    const output = gen().generate();
    expect(output).toContain("tile.openstreetmap.org");
  });

  it("should include opentopomap tile URL when specified", () => {
    const output = gen({ tileProvider: "opentopomap" }).generate();
    expect(output).toContain("opentopomap.org");
  });

  it("should include custom tile URL when provider is custom", () => {
    const output = gen({
      tileProvider: "custom",
      tileUrl: "https://my-tiles.example.com/{z}/{x}/{y}.png",
      tileAttribution: "My Tiles",
    }).generate();
    expect(output).toContain("my-tiles.example.com");
  });

  it("should throw if tileProvider=custom but no tileUrl", () => {
    expect(() => gen({ tileProvider: "custom" }).generate()).toThrow(/tileUrl/);
  });
});

// ---------------------------------------------------------------------------
// Self-contained output tests
// ---------------------------------------------------------------------------

describe("LeafletWidgetGenerator — selfContained mode", () => {
  it("should include Leaflet CSS CDN link", () => {
    const output = gen({ selfContained: true }).generate();
    expect(output).toContain("leaflet.css");
  });

  it("should include Leaflet JS CDN script", () => {
    const output = gen({ selfContained: true }).generate();
    expect(output).toContain("leaflet.js");
  });

  it("should include a <div> with the containerId", () => {
    const output = gen({ selfContained: true }).generate();
    expect(output).toContain(`id="map"`);
  });

  it("should apply custom height to the div style", () => {
    const output = gen({ selfContained: true, height: "800px" }).generate();
    expect(output).toContain("height: 800px");
  });
});

// ---------------------------------------------------------------------------
// Markers tests
// ---------------------------------------------------------------------------

describe("LeafletWidgetGenerator — markers", () => {
  it("should include L.marker for a configured marker", () => {
    const output = gen({
      markers: [{ lat: 51.5, lng: -0.1, popupHtml: "<b>Test</b>" }],
    }).generate();
    expect(output).toContain("L.marker([51.5, -0.1])");
  });

  it("should include bindPopup when popupHtml is set", () => {
    const output = gen({
      markers: [{ lat: 51.5, lng: -0.1, popupHtml: "<b>Test</b>" }],
    }).generate();
    expect(output).toContain("bindPopup");
    expect(output).toContain("<b>Test</b>");
  });

  it("should not include bindPopup when popupHtml is absent", () => {
    const output = gen({
      markers: [{ lat: 51.5, lng: -0.1 }],
    }).generate();
    expect(output).not.toContain("bindPopup");
  });
});

// ---------------------------------------------------------------------------
// GeoJSON layer tests
// ---------------------------------------------------------------------------

describe("LeafletWidgetGenerator — GeoJSON layers", () => {
  it("should include fetch() call for a GeoJSON layer URL", () => {
    const output = gen({
      geoJsonLayers: [
        { url: "https://example.com/data.geojson", name: "Hospitals" },
      ],
    }).generate();
    expect(output).toContain("fetch('https://example.com/data.geojson')");
  });

  it("should include layer control when multiple GeoJSON layers are defined", () => {
    const output = gen({
      geoJsonLayers: [
        { url: "a.geojson", name: "Layer A" },
        { url: "b.geojson", name: "Layer B" },
      ],
    }).generate();
    expect(output).toContain("L.control.layers");
  });

  it("should use custom fillColor when specified", () => {
    const output = gen({
      geoJsonLayers: [
        { url: "a.geojson", name: "Layer A", fillColor: "#ff0000" },
      ],
    }).generate();
    expect(output).toContain("#ff0000");
  });
});

// ---------------------------------------------------------------------------
// Scale control tests
// ---------------------------------------------------------------------------

describe("LeafletWidgetGenerator — scale control", () => {
  it("should include L.control.scale when scaleControl is configured", () => {
    const output = gen({
      scaleControl: { position: "bottomright", metric: true, imperial: false },
    }).generate();
    expect(output).toContain("L.control.scale");
    expect(output).toContain("bottomright");
  });

  it("should not include scale control by default", () => {
    const output = gen().generate();
    expect(output).not.toContain("L.control.scale");
  });
});
