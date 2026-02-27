/**
 * Stub type declarations for @arcgis/core.
 *
 * In the Experience Builder environment the full ArcGIS Maps SDK for
 * JavaScript types are available.  These stubs provide just enough
 * surface for the widget source to type-check outside ExB.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

declare namespace __esri {
  interface Handle {
    remove: () => void
  }
}

declare module '@arcgis/core/layers/GraphicsLayer' {
  export default class GraphicsLayer {
    title: string
    constructor (props?: { title?: string })
    add (graphic: any): void
    removeAll (): void
  }
}

declare module '@arcgis/core/Graphic' {
  export default class Graphic {
    constructor (props?: { geometry?: any; symbol?: any; attributes?: any })
  }
}

declare module '@arcgis/core/geometry/Point' {
  export default class Point {
    longitude: number
    latitude: number
    spatialReference: any
    constructor (props?: any)
  }
}

declare module '@arcgis/core/geometry/Polygon' {
  export default class Polygon {
    rings: number[][][]
    spatialReference: any
  }
}

declare module '@arcgis/core/geometry/geometryEngine' {
  export function geodesicBuffer (
    geometry: any,
    distance: number | number[],
    unit: string,
  ): any
  export function difference (geometry1: any, geometry2: any): any
}

declare module '@arcgis/core/symbols/SimpleFillSymbol' {
  export default class SimpleFillSymbol {
    constructor (props?: { color?: any; outline?: any })
  }
}

declare module '@arcgis/core/symbols/SimpleMarkerSymbol' {
  export default class SimpleMarkerSymbol {
    constructor (props?: { style?: string; color?: any; size?: number; outline?: any })
  }
}

declare module '@arcgis/core/layers/FeatureLayer' {
  export default class FeatureLayer {
    id: string
    title: string
    type: string
    visible: boolean
    objectIdField: string
    displayField: string
    createQuery (): any
    queryFeatures (query: any): Promise<any>
    properties: any
  }
}

declare module '@arcgis/core/views/MapView' {
  export default class MapView {
    map: any
    on (event: string, callback: (e: any) => void): __esri.Handle
  }
}
