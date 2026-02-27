/** @jsx jsx */
/**
 * Proximity Ring Analyzer — Runtime Widget
 * ==========================================
 * Click anywhere on the map to drop a centre point.  Concentric buffer
 * rings are drawn at the distances defined in the widget config, and
 * every visible FeatureLayer is queried to count (and list) the features
 * that fall within each ring.
 *
 * Key ArcGIS JS API surfaces used
 * --------------------------------
 * - `geometryEngine.geodesicBuffer` — accurate geodesic buffers
 * - `FeatureLayer.queryFeatures`    — server-side spatial queries
 * - `GraphicsLayer`                 — ring / centre-point visualisation
 * - `JimuMapView`                   — ExB ↔ Map SDK bridge
 */
import {
  React,
  jsx,
  type AllWidgetProps,
  DataSourceComponent,
} from 'jimu-core'
import { JimuMapViewComponent, type JimuMapView } from 'jimu-arcgis'
import { Button, Loading, Card, CardBody, Alert, Collapse } from 'jimu-ui'

import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer'
import Graphic from '@arcgis/core/Graphic'
import Point from '@arcgis/core/geometry/Point'
import * as geometryEngine from '@arcgis/core/geometry/geometryEngine'
import SimpleFillSymbol from '@arcgis/core/symbols/SimpleFillSymbol'
import SimpleMarkerSymbol from '@arcgis/core/symbols/SimpleMarkerSymbol'
import type FeatureLayer from '@arcgis/core/layers/FeatureLayer'
import type MapView from '@arcgis/core/views/MapView'
import type Polygon from '@arcgis/core/geometry/Polygon'

import { type IMConfig, defaultConfig } from '../../config'
import './style.css'

// ─── Types ───────────────────────────────────────────────────────────
/** Features found in a single ring for one layer. */
interface RingResult {
  ringIndex: number
  distance: number
  layerTitle: string
  count: number
  features: Array<{ oid: number; label: string }>
}

/** Full analysis result set. */
interface AnalysisResult {
  center: { longitude: number; latitude: number }
  rings: RingResult[]
  timestamp: Date
}

// ─── Component ───────────────────────────────────────────────────────

interface State {
  /** Active JimuMapView reference. */
  jimuMapView: JimuMapView | null
  /** True while queries are in-flight. */
  loading: boolean
  /** Latest analysis result (null before first click). */
  result: AnalysisResult | null
  /** Error message, if any. */
  error: string | null
  /** Tracks which result rows are expanded in the accordion. */
  expandedRows: Set<string>
  /** Whether the click listener is active. */
  listening: boolean
}

export default class ProximityRingAnalyzer extends React.PureComponent<
  AllWidgetProps<IMConfig>,
  State
> {
  /** Dedicated graphics layer for ring / marker visuals. */
  private graphicsLayer: GraphicsLayer | null = null
  /** Handle returned by `view.on('click')` so we can remove it. */
  private clickHandle: __esri.Handle | null = null

  state: State = {
    jimuMapView: null,
    loading: false,
    result: null,
    error: null,
    expandedRows: new Set(),
    listening: false,
  }

  // ── Lifecycle ────────────────────────────────────────────────────

  componentWillUnmount (): void {
    this.cleanup()
  }

  // ── Map view callbacks ───────────────────────────────────────────

  private onActiveViewChange = (jmv: JimuMapView): void => {
    this.cleanup()
    this.setState({ jimuMapView: jmv, result: null, error: null })
  }

  // ── Click listener management ────────────────────────────────────

  private startListening = (): void => {
    const { jimuMapView } = this.state
    if (!jimuMapView) return

    const view = jimuMapView.view as MapView
    this.graphicsLayer = new GraphicsLayer({ title: 'Proximity Rings' })
    view.map.add(this.graphicsLayer)

    this.clickHandle = view.on('click', (event) => {
      this.runAnalysis(event.mapPoint)
    })

    this.setState({ listening: true })
  }

  private stopListening = (): void => {
    this.cleanup()
    this.setState({ listening: false, result: null })
  }

  private cleanup (): void {
    if (this.clickHandle) {
      this.clickHandle.remove()
      this.clickHandle = null
    }
    if (this.graphicsLayer) {
      const view = this.state.jimuMapView?.view as MapView | undefined
      if (view) {
        view.map.remove(this.graphicsLayer)
      }
      this.graphicsLayer = null
    }
  }

  // ── Core analysis ────────────────────────────────────────────────

  private async runAnalysis (mapPoint: Point): Promise<void> {
    const { jimuMapView } = this.state
    if (!jimuMapView) return

    const cfg = { ...defaultConfig, ...this.props.config }
    const view = jimuMapView.view as MapView

    this.setState({ loading: true, error: null, result: null, expandedRows: new Set() })

    // Clear previous graphics
    if (this.graphicsLayer) {
      this.graphicsLayer.removeAll()
    }

    try {
      // 1 — Centre marker
      this.drawCenterMarker(mapPoint, cfg.centerColor)

      // 2 — Build geodesic buffer rings (sorted ascending)
      const distances = [...cfg.ringDistances].sort((a, b) => a - b)
      const unitMap: Record<string, 'miles' | 'kilometers' | 'meters' | 'feet'> = {
        miles: 'miles',
        kilometers: 'kilometers',
        meters: 'meters',
        feet: 'feet',
      }
      const unit = unitMap[cfg.distanceUnit] ?? 'miles'

      const ringPolygons = distances.map((d) =>
        geometryEngine.geodesicBuffer(mapPoint, d, unit) as Polygon,
      )

      // 3 — Draw the ring graphics (outermost first so inner rings render on top)
      for (let i = ringPolygons.length - 1; i >= 0; i--) {
        this.drawRing(ringPolygons[i], i, cfg)
      }

      // 4 — Discover queryable feature layers in the map
      const featureLayers = view.map.allLayers
        .filter((l) => l.type === 'feature' && l.visible)
        .toArray() as FeatureLayer[]

      // 5 — Query each layer × each ring
      const allResults: RingResult[] = []

      for (let ri = 0; ri < ringPolygons.length; ri++) {
        const innerGeom = ri === 0 ? null : ringPolygons[ri - 1]
        const outerGeom = ringPolygons[ri]

        for (const layer of featureLayers) {
          try {
            const result = await this.queryRing(
              layer,
              outerGeom,
              innerGeom,
              cfg.maxResults,
            )
            allResults.push({
              ringIndex: ri,
              distance: distances[ri],
              layerTitle: layer.title ?? layer.id,
              count: result.count,
              features: result.features,
            })
          } catch {
            // Skip layers that fail to query (e.g. non-spatial tables)
          }
        }
      }

      this.setState({
        loading: false,
        result: {
          center: { longitude: mapPoint.longitude, latitude: mapPoint.latitude },
          rings: allResults,
          timestamp: new Date(),
        },
      })
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      this.setState({ loading: false, error: message })
    }
  }

  // ── Spatial queries ──────────────────────────────────────────────

  /**
   * Query a single feature layer for features that fall within the
   * ring defined by `outerGeom` but outside `innerGeom` (if provided).
   *
   * When `innerGeom` is null the result covers the full buffer (the
   * innermost ring).
   */
  private async queryRing (
    layer: FeatureLayer,
    outerGeom: Polygon,
    innerGeom: Polygon | null,
    maxResults: number,
  ): Promise<{ count: number; features: Array<{ oid: number; label: string }> }> {
    const queryGeom = innerGeom
      ? geometryEngine.difference(outerGeom, innerGeom) as Polygon
      : outerGeom

    if (!queryGeom) {
      return { count: 0, features: [] }
    }

    const query = layer.createQuery()
    query.geometry = queryGeom
    query.spatialRelationship = 'intersects'
    query.returnGeometry = false
    query.outFields = [layer.objectIdField, layer.displayField ?? '*']
    query.num = maxResults

    const featureSet = await layer.queryFeatures(query)

    const features = featureSet.features.map((f) => ({
      oid: f.attributes[layer.objectIdField] as number,
      label: f.attributes[layer.displayField] as string
        ?? `OID ${f.attributes[layer.objectIdField]}`,
    }))

    return { count: featureSet.features.length, features }
  }

  // ── Graphics helpers ─────────────────────────────────────────────

  private drawCenterMarker (point: Point, color: string): void {
    if (!this.graphicsLayer) return

    this.graphicsLayer.add(
      new Graphic({
        geometry: point,
        symbol: new SimpleMarkerSymbol({
          style: 'circle',
          color: color,
          size: 10,
          outline: { color: '#ffffff', width: 2 },
        }),
      }),
    )
  }

  private drawRing (polygon: Polygon, index: number, cfg: typeof defaultConfig): void {
    if (!this.graphicsLayer) return

    const colorIndex = index % cfg.ringColors.length
    const fillColor = cfg.ringColors[colorIndex]

    this.graphicsLayer.add(
      new Graphic({
        geometry: polygon,
        symbol: new SimpleFillSymbol({
          color: this.hexToRgba(fillColor, cfg.ringOpacity),
          outline: { color: fillColor, width: 1.5 },
        }),
      }),
    )
  }

  /** Convert "#rrggbb" + opacity to `[r, g, b, a]`. */
  private hexToRgba (hex: string, opacity: number): [number, number, number, number] {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return [r, g, b, opacity]
  }

  // ── Accordion helpers ────────────────────────────────────────────

  private toggleRow = (key: string): void => {
    this.setState((prev) => {
      const next = new Set(prev.expandedRows)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return { expandedRows: next }
    })
  }

  // ── Export ────────────────────────────────────────────────────────

  private exportCsv = (): void => {
    const { result } = this.state
    if (!result) return

    const cfg = { ...defaultConfig, ...this.props.config }
    const header = 'Ring,Distance,Unit,Layer,Count,Feature Label,OID'
    const rows: string[] = [header]

    for (const ring of result.rings) {
      if (ring.features.length === 0) {
        rows.push(
          `${ring.ringIndex + 1},${ring.distance},${cfg.distanceUnit},${ring.layerTitle},0,,`,
        )
      } else {
        for (const f of ring.features) {
          rows.push(
            `${ring.ringIndex + 1},${ring.distance},${cfg.distanceUnit},${ring.layerTitle},${ring.count},"${f.label}",${f.oid}`,
          )
        }
      }
    }

    const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `proximity_analysis_${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ── Render ───────────────────────────────────────────────────────

  render (): React.ReactNode {
    const { useMapWidgetIds } = this.props
    const { loading, result, error, listening, jimuMapView, expandedRows } = this.state
    const cfg = { ...defaultConfig, ...this.props.config }

    return (
      <div className='proximity-ring-analyzer p-3'>
        <h5 className='mb-3'>Proximity Ring Analyzer</h5>

        {/* Map view binding */}
        {useMapWidgetIds && useMapWidgetIds.length > 0 && (
          <JimuMapViewComponent
            useMapWidgetId={useMapWidgetIds[0]}
            onActiveViewChange={this.onActiveViewChange}
          />
        )}

        {!jimuMapView && (
          <Alert type='warning' className='mb-3' open>
            Connect a Map widget in the widget settings.
          </Alert>
        )}

        {/* Controls */}
        {jimuMapView && (
          <div className='mb-3 d-flex controls-row'>
            {!listening
              ? (
                <Button type='primary' onClick={this.startListening}>
                  Start — Click Map
                </Button>
                )
              : (
                <Button type='danger' onClick={this.stopListening}>
                  Stop Listening
                </Button>
                )}
            {result && (
              <Button type='default' onClick={this.exportCsv}>
                Export CSV
              </Button>
            )}
          </div>
        )}

        {/* Loading indicator */}
        {loading && <Loading type='SECONDARY' />}

        {/* Error display */}
        {error && (
          <Alert type='error' className='mb-3' open>
            {error}
          </Alert>
        )}

        {/* Results */}
        {result && this.renderResults(result, cfg, expandedRows)}
      </div>
    )
  }

  // ── Result renderer ──────────────────────────────────────────────

  private renderResults (
    result: AnalysisResult,
    cfg: typeof defaultConfig,
    expandedRows: Set<string>,
  ): React.ReactNode {
    const { rings, center } = result
    const grouped = this.groupByRing(rings)
    const distances = [...cfg.ringDistances].sort((a, b) => a - b)

    return (
      <div>
        <p className='text-muted mb-2 center-label'>
          Centre: {center.latitude.toFixed(5)}, {center.longitude.toFixed(5)}
        </p>

        {distances.map((dist, ri) => {
          const layersInRing = grouped.get(ri) ?? []
          const totalCount = layersInRing.reduce((sum, r) => sum + r.count, 0)
          const ringKey = `ring-${ri}`
          const isExpanded = expandedRows.has(ringKey)

          return (
            <Card key={ri} className='mb-2'>
              <CardBody>
                <div
                  role='button'
                  tabIndex={0}
                  className='ring-header'
                  onClick={() => this.toggleRow(ringKey)}
                  onKeyDown={(e) => { if (e.key === 'Enter') this.toggleRow(ringKey) }}
                >
                  <div className='d-flex justify-content-between align-items-center'>
                    <strong>
                      Ring {ri + 1} — {dist} {cfg.distanceUnit}
                    </strong>
                    <span className='badge bg-primary'>{totalCount} feature(s)</span>
                  </div>
                </div>

                <Collapse isOpen={isExpanded}>
                  {layersInRing.length === 0 && (
                    <p className='text-muted mt-2 mb-0 empty-ring-msg'>
                      No features found.
                    </p>
                  )}
                  {layersInRing.map((lr) => {
                    const layerKey = `ring-${ri}-${lr.layerTitle}`
                    const layerExpanded = expandedRows.has(layerKey)

                    return (
                      <div key={lr.layerTitle} className='mt-2'>
                        <div
                          role='button'
                          tabIndex={0}
                          className='layer-header'
                          onClick={() => this.toggleRow(layerKey)}
                          onKeyDown={(e) => { if (e.key === 'Enter') this.toggleRow(layerKey) }}
                        >
                          <span>{lr.layerTitle}</span>
                          <span className='badge bg-secondary ms-2'>{lr.count}</span>
                        </div>

                        <Collapse isOpen={layerExpanded}>
                          <ul className='list-unstyled ms-3 mt-1 mb-0 feature-list'>
                            {lr.features.map((f) => (
                              <li key={f.oid}>
                                OID {f.oid} — {f.label}
                              </li>
                            ))}
                          </ul>
                        </Collapse>
                      </div>
                    )
                  })}
                </Collapse>
              </CardBody>
            </Card>
          )
        })}
      </div>
    )
  }

  /** Group ring results by ring index for easy rendering. */
  private groupByRing (rings: RingResult[]): Map<number, RingResult[]> {
    const map = new Map<number, RingResult[]>()
    for (const r of rings) {
      const existing = map.get(r.ringIndex) ?? []
      existing.push(r)
      map.set(r.ringIndex, existing)
    }
    return map
  }
}
