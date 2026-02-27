/**
 * Stub type declarations for jimu-arcgis.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
declare module 'jimu-arcgis' {
  import * as ReactNS from 'react'

  export interface JimuMapView {
    view: any
    [key: string]: any
  }

  export class JimuMapViewComponent extends ReactNS.Component<{
    useMapWidgetId: string
    onActiveViewChange: (jmv: JimuMapView) => void
  }> {}
}
