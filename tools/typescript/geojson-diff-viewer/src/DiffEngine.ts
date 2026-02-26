/**
 * DiffEngine
 * ==========
 * Pure (DOM-free) GeoJSON diff logic.
 *
 * Two {@link GeoJSON.FeatureCollection}s are compared and every feature is
 * classified as **added**, **removed**, or **unchanged**.
 *
 * @example
 * ```ts
 * const engine = new DiffEngine({ matchBy: "id" });
 * const result = engine.diff(beforeCollection, afterCollection);
 * console.log(result.added.length, "new features");
 * ```
 */

import type {
  Feature,
  FeatureCollection,
  DiffResult,
  DiffFeature,
  DiffEngineOptions,
} from "./types.js";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Serialise a GeoJSON geometry to a stable string key.
 * Rounds coordinates to 6 decimal places so floating-point noise does not
 * produce false positives.
 */
function geometryKey(feature: Feature): string {
  const geom = feature.geometry;
  if (!geom) return "null-geometry";

  const rounded = JSON.stringify(geom, (_key, value) =>
    typeof value === "number" ? parseFloat(value.toFixed(6)) : value,
  );
  return rounded;
}

/**
 * Derive a match key for a feature using the supplied `matchBy` option.
 *
 * @param feature   - GeoJSON Feature
 * @param matchBy   - Property name, array of names, or null (geometry fallback)
 * @returns A string key identifying this feature within a collection.
 */
function featureKey(
  feature: Feature,
  matchBy: string | string[] | null,
): string {
  if (matchBy === null) {
    return geometryKey(feature);
  }

  const props = feature.properties ?? {};
  const keys = Array.isArray(matchBy) ? matchBy : [matchBy];
  const parts = keys.map((k) => String(props[k] ?? ""));
  return parts.join("|");
}

// ---------------------------------------------------------------------------
// DiffEngine class
// ---------------------------------------------------------------------------

/**
 * Computes a diff between two GeoJSON FeatureCollections.
 */
export class DiffEngine {
  /** Resolved options for this engine instance. */
  private readonly options: Required<DiffEngineOptions>;

  /**
   * @param options - Engine options.  See {@link DiffEngineOptions}.
   */
  constructor(options: Partial<DiffEngineOptions> = {}) {
    this.options = {
      matchBy: options.matchBy ?? null,
    };
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Compute the diff between two feature collections.
   *
   * Algorithm
   * ---------
   * 1.  Build a keyed map of `a` features.
   * 2.  Walk through `b`:
   *     - If the key exists in `a` → **unchanged** (remove from `a` map).
   *     - Otherwise → **added**.
   * 3.  Remaining entries in the `a` map → **removed**.
   *
   * @param a - The "before" collection (older / reference state).
   * @param b - The "after" collection (newer / updated state).
   * @returns A {@link DiffResult} with three categorised arrays.
   */
  public diff(a: FeatureCollection, b: FeatureCollection): DiffResult {
    const { matchBy } = this.options;

    // Build index of "before" features
    const aMap = new Map<string, Feature>();
    for (const feature of a.features) {
      const key = featureKey(feature, matchBy);
      aMap.set(key, feature);
    }

    const added: DiffFeature[] = [];
    const unchanged: DiffFeature[] = [];

    for (const feature of b.features) {
      const key = featureKey(feature, matchBy);
      if (aMap.has(key)) {
        unchanged.push({ feature, status: "unchanged" });
        aMap.delete(key); // consumed
      } else {
        added.push({ feature, status: "added" });
      }
    }

    // Whatever remains in aMap was not found in b → removed
    const removed: DiffFeature[] = [];
    for (const feature of aMap.values()) {
      removed.push({ feature, status: "removed" });
    }

    return { added, removed, unchanged };
  }

  /**
   * Human-readable summary of a {@link DiffResult}.
   *
   * @param result - Result from {@link diff}.
   * @returns Multi-line string summary.
   */
  public static summarise(result: DiffResult): string {
    const total =
      result.added.length + result.removed.length + result.unchanged.length;
    return [
      `Total features examined : ${total}`,
      `  Added                 : ${result.added.length}`,
      `  Removed               : ${result.removed.length}`,
      `  Unchanged             : ${result.unchanged.length}`,
    ].join("\n");
  }
}
