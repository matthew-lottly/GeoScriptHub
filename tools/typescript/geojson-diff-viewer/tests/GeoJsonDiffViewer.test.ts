/**
 * GeoJsonDiffViewer — DOM Tests
 * ==============================
 * Leaflet is mocked so no real canvas / tile requests are made.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { FeatureCollection } from "../src/types.js";

// ---------------------------------------------------------------------------
// Mock Leaflet before importing GeoJsonDiffViewer
// ---------------------------------------------------------------------------

const mockGeoJSONLayer = {
  addTo: vi.fn().mockReturnThis(),
  remove: vi.fn(),
  getBounds: vi.fn().mockReturnValue({
    isValid: vi.fn().mockReturnValue(false),
    extend: vi.fn().mockReturnThis(),
  }),
};

const mockTileLayer = { addTo: vi.fn().mockReturnThis() };

const mockLegendControl = {
  addTo: vi.fn().mockReturnThis(),
  remove: vi.fn(),
};

const LegendExtend = vi.fn().mockReturnValue(mockLegendControl);

const mockMap = {
  remove: vi.fn(),
  fitBounds: vi.fn(),
};

vi.mock("leaflet", () => ({
  default: {},
  map: vi.fn().mockReturnValue(mockMap),
  tileLayer: vi.fn().mockReturnValue(mockTileLayer),
  geoJSON: vi.fn().mockReturnValue(mockGeoJSONLayer),
  circleMarker: vi.fn().mockReturnValue({}),
  DomUtil: { create: vi.fn().mockReturnValue(document.createElement("div")) },
  Control: {
    extend: vi.fn().mockReturnValue(LegendExtend),
  },
}));

// ---------------------------------------------------------------------------
// Import after mock
// ---------------------------------------------------------------------------

import { GeoJsonDiffViewer } from "../src/GeoJsonDiffViewer.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const EMPTY_FC: FeatureCollection = { type: "FeatureCollection", features: [] };
const ONE_POINT: FeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {},
      geometry: { type: "Point", coordinates: [0, 0] },
    },
  ],
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GeoJsonDiffViewer", () => {
  let container: HTMLDivElement;

  beforeEach(() => {
    container = document.createElement("div");
    container.id = "test-map";
    document.body.appendChild(container);
    vi.clearAllMocks();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------

  it("throws if containerId is empty", () => {
    expect(() => new GeoJsonDiffViewer({ containerId: "" })).toThrow(
      /containerId must not be empty/,
    );
  });

  it("constructs without error for valid config", () => {
    expect(
      () => new GeoJsonDiffViewer({ containerId: "test-map" }),
    ).not.toThrow();
  });

  // -------------------------------------------------------------------------
  // mount()
  // -------------------------------------------------------------------------

  it("mount() throws if container element not found", () => {
    const viewer = new GeoJsonDiffViewer({ containerId: "no-such-element" });
    expect(() => viewer.mount()).toThrow(/not found/);
  });

  it("mount() sets container dimensions from config", () => {
    const viewer = new GeoJsonDiffViewer({
      containerId: "test-map",
      height: "300px",
      width: "600px",
    });
    viewer.mount();
    expect(container.style.height).toBe("300px");
    expect(container.style.width).toBe("600px");
  });

  it("mount() is idempotent (second call is a no-op)", async () => {
    const { map } = await import("leaflet");
    const viewer = new GeoJsonDiffViewer({ containerId: "test-map" });
    viewer.mount();
    viewer.mount(); // second call
    expect(map).toHaveBeenCalledTimes(1);
  });

  // -------------------------------------------------------------------------
  // update()
  // -------------------------------------------------------------------------

  it("update() throws if not mounted", () => {
    const viewer = new GeoJsonDiffViewer({ containerId: "test-map" });
    expect(() => viewer.update(EMPTY_FC, EMPTY_FC)).toThrow(/mount\(\)/);
  });

  it("update() does not throw for valid collections", () => {
    const viewer = new GeoJsonDiffViewer({ containerId: "test-map" });
    viewer.mount();
    expect(() => viewer.update(EMPTY_FC, ONE_POINT)).not.toThrow();
  });

  it("update() calls geoJSON three times (added, removed, unchanged layers)", async () => {
    const L = await import("leaflet");
    const viewer = new GeoJsonDiffViewer({ containerId: "test-map" });
    viewer.mount();
    viewer.update(EMPTY_FC, EMPTY_FC);
    // One call per status bucket (added, removed, unchanged)
    expect(L.geoJSON).toHaveBeenCalledTimes(3);
  });

  // -------------------------------------------------------------------------
  // unmount()
  // -------------------------------------------------------------------------

  it("unmount() removes the map", () => {
    const viewer = new GeoJsonDiffViewer({ containerId: "test-map" });
    viewer.mount();
    viewer.unmount();
    expect(mockMap.remove).toHaveBeenCalledTimes(1);
  });

  it("unmount() is idempotent when not mounted", () => {
    const viewer = new GeoJsonDiffViewer({ containerId: "test-map" });
    expect(() => viewer.unmount()).not.toThrow();
    expect(mockMap.remove).not.toHaveBeenCalled();
  });
});
