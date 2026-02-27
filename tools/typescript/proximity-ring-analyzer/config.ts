import { type ImmutableObject } from 'jimu-core'

/**
 * Proximity Ring Analyzer — widget configuration.
 *
 * These values are set in the Experience Builder *Settings* panel
 * and consumed by the runtime widget.
 */
export interface Config {
  /** Ordered list of buffer distances drawn outward from the centre point. */
  ringDistances: number[]

  /** Unit for every distance in `ringDistances`. */
  distanceUnit: 'miles' | 'kilometers' | 'meters' | 'feet'

  /** Maximum number of features returned per layer per ring. */
  maxResults: number

  /**
   * Fill colours for each ring, applied from innermost to outermost.
   * Each colour is a hex string (e.g. `"#1a9641"`).
   * If fewer colours than rings are provided the list cycles.
   */
  ringColors: string[]

  /** Fill opacity for the ring graphics (0–1). */
  ringOpacity: number

  /** Colour of the centre-point marker. */
  centerColor: string
}

/** Immutable version consumed by Jimu runtime. */
export type IMConfig = ImmutableObject<Config>

/**
 * Sensible defaults — three rings at 0.25, 0.5, and 1.0 miles.
 */
export const defaultConfig: Config = {
  ringDistances: [0.25, 0.5, 1.0],
  distanceUnit: 'miles',
  maxResults: 500,
  ringColors: ['#1a9641', '#a6d96a', '#fdae61', '#d7191c'],
  ringOpacity: 0.25,
  centerColor: '#e31a1c',
}
