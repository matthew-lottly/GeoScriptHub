/**
 * Tests for MapSwiper
 * ====================
 * Uses happy-dom (DOM environment) via Vitest.
 * MapLibre GL Map calls are mocked so no WebGL context is needed.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MapSwiperConfig } from "../src/types.js";

// ---------------------------------------------------------------------------
// Mock maplibre-gl before importing MapSwiper
// ---------------------------------------------------------------------------

vi.mock("maplibre-gl", () => {
  const MockMap = vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    remove: vi.fn(),
    resize: vi.fn(),
    getCenter: vi.fn().mockReturnValue({ lng: -0.09, lat: 51.505 }),
    getZoom: vi.fn().mockReturnValue(12),
    getBearing: vi.fn().mockReturnValue(0),
    getPitch: vi.fn().mockReturnValue(0),
    jumpTo: vi.fn(),
  }));
  return { default: { Map: MockMap }, Map: MockMap };
});

import { MapSwiper } from "../src/MapSwiper.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const BASE_CONFIG: MapSwiperConfig = {
  containerId: "swiper",
  center: [-0.09, 51.505],
  zoom: 12,
  left: { style: "https://demotiles.maplibre.org/style.json", label: "Before" },
  right: { style: "https://demotiles.maplibre.org/style.json", label: "After" },
};

function createContainerDiv(id: string = "swiper"): HTMLDivElement {
  const div = document.createElement("div");
  div.id = id;
  div.style.width = "800px";
  div.style.height = "500px";
  document.body.appendChild(div);
  return div;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MapSwiper — construction", () => {
  it("should construct without error with valid config", () => {
    expect(() => new MapSwiper(BASE_CONFIG)).not.toThrow();
  });

  it("should throw if containerId is empty", () => {
    expect(() => new MapSwiper({ ...BASE_CONFIG, containerId: "" })).toThrow(/containerId/i);
  });

  it("should throw if latitude is out of range", () => {
    expect(
      () => new MapSwiper({ ...BASE_CONFIG, center: [-0.09, 999] })
    ).toThrow(/latitude/);
  });

  it("should throw if longitude is out of range", () => {
    expect(
      () => new MapSwiper({ ...BASE_CONFIG, center: [-999, 51.5] })
    ).toThrow(/longitude/);
  });

  it("should throw if zoom is out of range", () => {
    expect(
      () => new MapSwiper({ ...BASE_CONFIG, zoom: 99 })
    ).toThrow(/zoom/);
  });
});

describe("MapSwiper — mount()", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  it("should throw if container element is not found", () => {
    const swiper = new MapSwiper({ ...BASE_CONFIG, containerId: "does-not-exist" });
    expect(() => swiper.mount()).toThrow(/does-not-exist/);
  });

  it("should render left and right panel divs", () => {
    createContainerDiv("swiper");
    const swiper = new MapSwiper(BASE_CONFIG);
    swiper.mount();

    const container = document.getElementById("swiper")!;
    const leftPanel = container.querySelector(".gs-swiper-panel-left");
    const rightPanel = container.querySelector(".gs-swiper-panel-right");
    expect(leftPanel).toBeTruthy();
    expect(rightPanel).toBeTruthy();
  });

  it("should render a divider element", () => {
    createContainerDiv("swiper");
    const swiper = new MapSwiper(BASE_CONFIG);
    swiper.mount();

    const container = document.getElementById("swiper")!;
    const divider = container.querySelector(".gs-swiper-divider");
    expect(divider).toBeTruthy();
  });

  it("should inject styles into <head>", () => {
    createContainerDiv("swiper");
    new MapSwiper(BASE_CONFIG).mount();
    const styleEl = document.getElementById("geoscripthub-map-swiper-styles");
    expect(styleEl).toBeTruthy();
  });
});

describe("MapSwiper — setDividerPosition()", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  it("should clamp position to [0.02, 0.98]", () => {
    createContainerDiv("swiper");
    const swiper = new MapSwiper(BASE_CONFIG);
    swiper.mount();
    // Should not throw on extreme values
    expect(() => swiper.setDividerPosition(0)).not.toThrow();
    expect(() => swiper.setDividerPosition(1)).not.toThrow();
    expect(() => swiper.setDividerPosition(0.75)).not.toThrow();
  });
});

describe("MapSwiper — unmount()", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  it("should clear the container on unmount", () => {
    createContainerDiv("swiper");
    const swiper = new MapSwiper(BASE_CONFIG);
    swiper.mount();
    swiper.unmount();

    const container = document.getElementById("swiper")!;
    expect(container.innerHTML).toBe("");
  });
});

describe("MapSwiper — config defaults", () => {
  it("should default dividerPosition to 0.5", () => {
    createContainerDiv("swiper");
    const swiper = new MapSwiper(BASE_CONFIG);
    swiper.mount();

    const container = document.getElementById("swiper")!;
    const leftPanel = container.querySelector<HTMLElement>(".gs-swiper-panel-left")!;
    // Left panel's right edge should be at ~50%
    expect(leftPanel.style.right).toBe("50%");
  });

  it("should respect custom initialDividerPosition", () => {
    createContainerDiv("swiper");
    const swiper = new MapSwiper({ ...BASE_CONFIG, initialDividerPosition: 0.3 });
    swiper.mount();

    const container = document.getElementById("swiper")!;
    const leftPanel = container.querySelector<HTMLElement>(".gs-swiper-panel-left")!;
    expect(leftPanel.style.right).toBe("70%");
  });
});
