/**
 * Map Swiper
 * ===========
 * A before/after map comparison widget built on MapLibre GL JS.
 *
 * Renders two maps side-by-side inside a single container, connected by a
 * draggable vertical divider.  Both maps are synchronised on pan/zoom by
 * default.
 *
 * @example
 * ```typescript
 * import { MapSwiper } from './MapSwiper';
 *
 * const swiper = new MapSwiper({
 *   containerId: 'swiper',
 *   center: [-0.09, 51.505],
 *   zoom: 12,
 *   left:  { style: 'https://demotiles.maplibre.org/style.json', label: 'Before' },
 *   right: { style: 'https://demotiles.maplibre.org/style.json', label: 'After'  },
 * });
 * swiper.mount();
 * ```
 *
 * @remarks
 * Requires MapLibre GL JS to be installed as a peer dependency and its CSS to
 * be loaded on the page.
 */

import maplibregl, { Map as MLMap } from "maplibre-gl";
import type { MapSwiperConfig, PanelConfig } from "./types.js";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Clamp a value to [min, max]. */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Inject the swiper CSS scoped to this widget (idempotent). */
function injectStyles(dividerColor: string, dividerWidth: number): void {
  const styleId = "geoscripthub-map-swiper-styles";
  if (document.getElementById(styleId)) return;

  const css = `
    .gs-swiper-container {
      position: relative;
      overflow: hidden;
      width: 100%;
      height: 100%;
    }
    .gs-swiper-panel {
      position: absolute;
      top: 0;
      bottom: 0;
    }
    .gs-swiper-panel-left  { left: 0; }
    .gs-swiper-panel-right { right: 0; }
    .gs-swiper-divider {
      position: absolute;
      top: 0;
      bottom: 0;
      width: ${dividerWidth}px;
      background: ${dividerColor};
      cursor: ew-resize;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .gs-swiper-handle {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background: ${dividerColor};
      border: 3px solid rgba(0,0,0,0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      user-select: none;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .gs-swiper-label {
      position: absolute;
      top: 12px;
      background: rgba(0,0,0,0.55);
      color: #fff;
      font-family: system-ui, sans-serif;
      font-size: 13px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 4px;
      pointer-events: none;
      user-select: none;
      z-index: 5;
    }
    .gs-swiper-label-left  { left: 12px; }
    .gs-swiper-label-right { right: 12px; }
  `;

  const el = document.createElement("style");
  el.id = styleId;
  el.textContent = css;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// MapSwiper class
// ---------------------------------------------------------------------------

/**
 * Before/after map comparison widget with a draggable divider.
 *
 * Call {@link mount} to render, {@link unmount} to destroy cleanly.
 */
export class MapSwiper {
  private readonly config: Required<MapSwiperConfig>;
  private container: HTMLElement | null = null;
  private leftMap: MLMap | null = null;
  private rightMap: MLMap | null = null;
  private divider: HTMLElement | null = null;
  private isDragging = false;
  private dividerPosition: number; // fraction [0,1]

  // Bound event handler references (needed for removeEventListener)
  private _onMouseDown: (e: MouseEvent | TouchEvent) => void;
  private _onMouseMove: (e: MouseEvent | TouchEvent) => void;
  private _onMouseUp: () => void;

  /**
   * @param config - Swiper configuration.  See {@link MapSwiperConfig} for
   *   placeholder descriptions.
   */
  constructor(config: MapSwiperConfig) {
    this.config = {
      initialDividerPosition: 0.5,
      syncMaps: true,
      dividerWidth: 4,
      dividerColor: "#ffffff",
      ...config,
    };
    this.dividerPosition = this.config.initialDividerPosition;
    this._onMouseDown = this._handleDragStart.bind(this);
    this._onMouseMove = this._handleDragMove.bind(this);
    this._onMouseUp = this._handleDragEnd.bind(this);
    this._validate();
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Mount the swiper into the configured container element.
   * Creates two MapLibre maps and wires up all event listeners.
   *
   * @throws {Error} If the container element cannot be found in the DOM.
   */
  public mount(): void {
    const raw = this.config.containerId;
    const id = raw.startsWith("#") ? raw.slice(1) : raw;
    const containerEl = document.getElementById(id);
    if (!containerEl) {
      throw new Error(
        `MapSwiper: container element #${id} not found. ` +
          "<!-- PLACEHOLDER: ensure the id matches an element in your HTML -->",
      );
    }

    this.container = containerEl;
    injectStyles(this.config.dividerColor, this.config.dividerWidth);

    // Wrap the container
    containerEl.style.position = "relative";
    containerEl.style.overflow = "hidden";

    const { left, right } = this.config;

    // Create left panel
    const leftPanel = this._createPanel("left", left);
    containerEl.appendChild(leftPanel);

    // Create right panel
    const rightPanel = this._createPanel("right", right);
    containerEl.appendChild(rightPanel);

    // Create divider handle
    this.divider = this._createDivider();
    containerEl.appendChild(this.divider);

    // Initialise MapLibre maps
    this.leftMap = new maplibregl.Map({
      container: leftPanel.querySelector(".gs-swiper-map") as HTMLElement,
      style: left.style as maplibregl.StyleSpecification | string,
      center: this.config.center,
      zoom: this.config.zoom,
      attributionControl: true,
    });

    this.rightMap = new maplibregl.Map({
      container: rightPanel.querySelector(".gs-swiper-map") as HTMLElement,
      style: right.style as maplibregl.StyleSpecification | string,
      center: this.config.center,
      zoom: this.config.zoom,
      attributionControl: false,
    });

    // Sync maps
    if (this.config.syncMaps) {
      this._wireSynchronisation();
    }

    // Initial layout
    this._updateDivider(this.dividerPosition);

    // Resize observer to handle container size changes
    new ResizeObserver(() => {
      this.leftMap?.resize();
      this.rightMap?.resize();
    }).observe(containerEl);
  }

  /**
   * Programmatically move the divider to a new position.
   *
   * @param fraction - New position as a fraction of container width (0.0–1.0).
   */
  public setDividerPosition(fraction: number): void {
    this.dividerPosition = clamp(fraction, 0.02, 0.98);
    this._updateDivider(this.dividerPosition);
  }

  /**
   * Cleanly destroy both maps and remove all event listeners.
   */
  public unmount(): void {
    document.removeEventListener("mousemove", this._onMouseMove as EventListener);
    document.removeEventListener("mouseup", this._onMouseUp);
    document.removeEventListener("touchmove", this._onMouseMove as EventListener);
    document.removeEventListener("touchend", this._onMouseUp);
    this.leftMap?.remove();
    this.rightMap?.remove();
    if (this.container) {
      this.container.innerHTML = "";
    }
    this.leftMap = null;
    this.rightMap = null;
  }

  // ------------------------------------------------------------------
  // Private builders
  // ------------------------------------------------------------------

  private _createPanel(side: "left" | "right", _panelConfig: PanelConfig): HTMLElement {
    const panel = document.createElement("div");
    panel.className = `gs-swiper-panel gs-swiper-panel-${side}`;
    panel.style.cssText = "position:absolute;top:0;bottom:0;overflow:hidden;";
    if (side === "left") {
      panel.style.left = "0";
      panel.style.right = `${(1 - this.dividerPosition) * 100}%`;
    } else {
      panel.style.left = `${this.dividerPosition * 100}%`;
      panel.style.right = "0";
    }

    // Inner map div
    const mapDiv = document.createElement("div");
    mapDiv.className = "gs-swiper-map";
    mapDiv.style.cssText = "width:100%;height:100%;";
    panel.appendChild(mapDiv);

    return panel;
  }

  private _createDivider(): HTMLElement {
    const divider = document.createElement("div");
    divider.className = "gs-swiper-divider";
    divider.style.left = `calc(${this.dividerPosition * 100}% - ${this.config.dividerWidth / 2}px)`;

    const handle = document.createElement("div");
    handle.className = "gs-swiper-handle";
    handle.textContent = "⇔";
    divider.appendChild(handle);

    divider.addEventListener("mousedown", this._onMouseDown as EventListener);
    divider.addEventListener("touchstart", this._onMouseDown as EventListener, { passive: true });

    return divider;
  }

  // ------------------------------------------------------------------
  // Drag event handlers
  // ------------------------------------------------------------------

  private _handleDragStart(_e: MouseEvent | TouchEvent): void {
    this.isDragging = true;
    document.addEventListener("mousemove", this._onMouseMove as EventListener);
    document.addEventListener("mouseup", this._onMouseUp);
    document.addEventListener("touchmove", this._onMouseMove as EventListener);
    document.addEventListener("touchend", this._onMouseUp);
  }

  private _handleDragMove(e: MouseEvent | TouchEvent): void {
    if (!this.isDragging || !this.container) return;

    const rect = this.container.getBoundingClientRect();
    const clientX =
      e instanceof MouseEvent ? e.clientX : (e as TouchEvent).touches[0]?.clientX ?? 0;
    const fraction = clamp((clientX - rect.left) / rect.width, 0.02, 0.98);
    this._updateDivider(fraction);
    this.dividerPosition = fraction;
  }

  private _handleDragEnd(): void {
    this.isDragging = false;
    document.removeEventListener("mousemove", this._onMouseMove as EventListener);
    document.removeEventListener("mouseup", this._onMouseUp);
    document.removeEventListener("touchmove", this._onMouseMove as EventListener);
    document.removeEventListener("touchend", this._onMouseUp);
  }

  // ------------------------------------------------------------------
  // Layout update
  // ------------------------------------------------------------------

  private _updateDivider(fraction: number): void {
    if (!this.container) return;

    const pct = fraction * 100;
    const leftPanel = this.container.querySelector(".gs-swiper-panel-left") as HTMLElement | null;
    const rightPanel = this.container.querySelector(".gs-swiper-panel-right") as HTMLElement | null;

    if (leftPanel) leftPanel.style.right = `${100 - pct}%`;
    if (rightPanel) rightPanel.style.left = `${pct}%`;
    if (this.divider) {
      this.divider.style.left = `calc(${pct}% - ${this.config.dividerWidth / 2}px)`;
    }

    this.leftMap?.resize();
    this.rightMap?.resize();
  }

  // ------------------------------------------------------------------
  // Map synchronisation
  // ------------------------------------------------------------------

  private _wireSynchronisation(): void {
    if (!this.leftMap || !this.rightMap) return;

    let syncing = false;

    const syncTo = (source: MLMap, target: MLMap): void => {
      if (syncing) return;
      syncing = true;
      target.jumpTo({
        center: source.getCenter(),
        zoom: source.getZoom(),
        bearing: source.getBearing(),
        pitch: source.getPitch(),
      });
      syncing = false;
    };

    this.leftMap.on("move", () => syncTo(this.leftMap!, this.rightMap!));
    this.rightMap.on("move", () => syncTo(this.rightMap!, this.leftMap!));
  }

  // ------------------------------------------------------------------
  // Validation
  // ------------------------------------------------------------------

  private _validate(): void {
    if (!this.config.containerId) {
      throw new Error(
        "containerId is required. <!-- PLACEHOLDER: set containerId in MapSwiperConfig -->",
      );
    }
    const [lng, lat] = this.config.center;
    if (lat < -90 || lat > 90) throw new Error(`center latitude must be in [-90,90], got ${lat}`);
    if (lng < -180 || lng > 180)
      throw new Error(`center longitude must be in [-180,180], got ${lng}`);
    if (this.config.zoom < 0 || this.config.zoom > 22)
      throw new Error(`zoom must be in [0,22], got ${this.config.zoom}`);
  }
}
