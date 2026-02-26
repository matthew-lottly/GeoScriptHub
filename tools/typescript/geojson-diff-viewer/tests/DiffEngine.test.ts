/**
 * DiffEngine — Unit Tests
 * ========================
 * Pure logic tests — no DOM, no Leaflet.
 */

import { describe, it, expect } from "vitest";
import { DiffEngine } from "../src/DiffEngine.js";
import type { FeatureCollection } from "../src/types.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const pointA: FeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "1",
      properties: { name: "Alpha" },
      geometry: { type: "Point", coordinates: [0, 0] },
    },
    {
      type: "Feature",
      id: "2",
      properties: { name: "Beta" },
      geometry: { type: "Point", coordinates: [1, 1] },
    },
  ],
};

const pointBAllSame: FeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "1",
      properties: { name: "Alpha" },
      geometry: { type: "Point", coordinates: [0, 0] },
    },
    {
      type: "Feature",
      id: "2",
      properties: { name: "Beta" },
      geometry: { type: "Point", coordinates: [1, 1] },
    },
  ],
};

const pointBNewFeature: FeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "1",
      properties: { name: "Alpha" },
      geometry: { type: "Point", coordinates: [0, 0] },
    },
    {
      type: "Feature",
      id: "2",
      properties: { name: "Beta" },
      geometry: { type: "Point", coordinates: [1, 1] },
    },
    {
      type: "Feature",
      id: "3",
      properties: { name: "Gamma" },
      geometry: { type: "Point", coordinates: [2, 2] },
    },
  ],
};

const pointBRemovedFeature: FeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "1",
      properties: { name: "Alpha" },
      geometry: { type: "Point", coordinates: [0, 0] },
    },
    // id "2" removed
  ],
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DiffEngine (geometry-match mode)", () => {
  const engine = new DiffEngine({ matchBy: null });

  it("produces no added/removed for identical collections", () => {
    const result = engine.diff(pointA, pointBAllSame);
    expect(result.added).toHaveLength(0);
    expect(result.removed).toHaveLength(0);
    expect(result.unchanged).toHaveLength(2);
  });

  it("detects an added feature", () => {
    const result = engine.diff(pointA, pointBNewFeature);
    expect(result.added).toHaveLength(1);
    expect(result.added[0]!.feature.geometry).toMatchObject({
      type: "Point",
      coordinates: [2, 2],
    });
    expect(result.removed).toHaveLength(0);
    expect(result.unchanged).toHaveLength(2);
  });

  it("detects a removed feature", () => {
    const result = engine.diff(pointA, pointBRemovedFeature);
    expect(result.removed).toHaveLength(1);
    expect(result.removed[0]!.feature.geometry).toMatchObject({
      type: "Point",
      coordinates: [1, 1],
    });
    expect(result.added).toHaveLength(0);
    expect(result.unchanged).toHaveLength(1);
  });

  it("handles empty before collection", () => {
    const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
    const result = engine.diff(empty, pointA);
    expect(result.added).toHaveLength(2);
    expect(result.removed).toHaveLength(0);
    expect(result.unchanged).toHaveLength(0);
  });

  it("handles empty after collection", () => {
    const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
    const result = engine.diff(pointA, empty);
    expect(result.added).toHaveLength(0);
    expect(result.removed).toHaveLength(2);
    expect(result.unchanged).toHaveLength(0);
  });

  it("handles both collections empty", () => {
    const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
    const result = engine.diff(empty, empty);
    expect(result.added).toHaveLength(0);
    expect(result.removed).toHaveLength(0);
    expect(result.unchanged).toHaveLength(0);
  });

  it("assigns correct DiffStatus values", () => {
    const result = engine.diff(pointA, pointBNewFeature);
    for (const df of result.added) expect(df.status).toBe("added");
    for (const df of result.unchanged) expect(df.status).toBe("unchanged");
  });
});

describe("DiffEngine (property-match mode)", () => {
  const engine = new DiffEngine({ matchBy: "name" });

  it("matches features by property, not geometry", () => {
    const before: FeatureCollection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { name: "River" },
          geometry: { type: "Point", coordinates: [10, 20] },
        },
      ],
    };
    // Same "name" but different coordinates → still "unchanged" by property match
    const after: FeatureCollection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { name: "River" },
          geometry: { type: "Point", coordinates: [10.0001, 20.0001] },
        },
      ],
    };
    const result = engine.diff(before, after);
    expect(result.unchanged).toHaveLength(1);
    expect(result.added).toHaveLength(0);
    expect(result.removed).toHaveLength(0);
  });

  it("detects rename as remove + add", () => {
    const before: FeatureCollection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { name: "OldName" },
          geometry: { type: "Point", coordinates: [0, 0] },
        },
      ],
    };
    const after: FeatureCollection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { name: "NewName" },
          geometry: { type: "Point", coordinates: [0, 0] },
        },
      ],
    };
    const result = engine.diff(before, after);
    expect(result.removed).toHaveLength(1);
    expect(result.added).toHaveLength(1);
    expect(result.unchanged).toHaveLength(0);
  });
});

describe("DiffEngine.summarise()", () => {
  it("returns correct count lines", () => {
    const engine = new DiffEngine();
    const result = engine.diff(pointA, pointBNewFeature);
    const summary = DiffEngine.summarise(result);
    expect(summary).toContain("Added                 : 1");
    expect(summary).toContain("Removed               : 0");
    expect(summary).toContain("Unchanged             : 2");
  });
});
