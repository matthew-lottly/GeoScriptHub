/** @jsx jsx */
/**
 * Proximity Ring Analyzer — Settings Panel
 * ==========================================
 * Allows the app builder to:
 *
 * 1. Select a Map widget to bind to.
 * 2. Edit the list of ring distances.
 * 3. Choose the distance unit.
 * 4. Adjust ring colours and opacity.
 * 5. Set a maximum-results cap per layer per ring.
 */
import {
  React,
  jsx,
  type AllWidgetSettingProps,
  Immutable,
} from 'jimu-core'
import { MapWidgetSelector } from 'jimu-ui/advanced/setting-components'
import {
  TextInput,
  NumericInput,
  Select,
  Option,
  Label,
  Button,
  Alert,
} from 'jimu-ui'

import { type IMConfig, defaultConfig } from '../../config'
import './style.css'

interface State {
  /** Temporary new-distance input value. */
  newDistance: number
}

export default class Setting extends React.PureComponent<
  AllWidgetSettingProps<IMConfig>,
  State
> {
  state: State = { newDistance: 1 }

  // ── Helpers ──────────────────────────────────────────────────────

  /** Merge a partial config update into the widget config. */
  private updateConfig (partial: Record<string, unknown>): void {
    const current = this.props.config ?? Immutable(defaultConfig)
    this.props.onSettingChange({
      id: this.props.id,
      config: current.merge(partial) as IMConfig,
    })
  }

  // ── Ring distance management ─────────────────────────────────────

  private addDistance = (): void => {
    const current = (this.props.config?.ringDistances as unknown as number[])
      ?? defaultConfig.ringDistances
    const next = [...current, this.state.newDistance].sort((a, b) => a - b)
    this.updateConfig({ ringDistances: next })
  }

  private removeDistance = (index: number): void => {
    const current = (this.props.config?.ringDistances as unknown as number[])
      ?? defaultConfig.ringDistances
    const next = current.filter((_, i) => i !== index)
    this.updateConfig({ ringDistances: next })
  }

  // ── Colour management ────────────────────────────────────────────

  private updateColor = (index: number, hex: string): void => {
    const current = (this.props.config?.ringColors as unknown as string[])
      ?? defaultConfig.ringColors
    const next = [...current]
    next[index] = hex
    this.updateConfig({ ringColors: next })
  }

  private addColor = (): void => {
    const current = (this.props.config?.ringColors as unknown as string[])
      ?? defaultConfig.ringColors
    this.updateConfig({ ringColors: [...current, '#888888'] })
  }

  private removeColor = (index: number): void => {
    const current = (this.props.config?.ringColors as unknown as string[])
      ?? defaultConfig.ringColors
    const next = current.filter((_, i) => i !== index)
    this.updateConfig({ ringColors: next })
  }

  // ── Render ───────────────────────────────────────────────────────

  render (): React.ReactNode {
    const config = {
      ...defaultConfig,
      ...(this.props.config?.asMutable?.({ deep: true }) ?? {}),
    }

    return (
      <div className='proximity-ring-settings p-3'>
        {/* ─── Map Widget ─── */}
        <Label className='mb-1 fw-bold'>Map Widget</Label>
        <MapWidgetSelector
          useMapWidgetIds={this.props.useMapWidgetIds}
          onSelect={(ids: string[]) => {
            this.props.onSettingChange({
              id: this.props.id,
              useMapWidgetIds: ids,
            })
          }}
        />

        <hr />

        {/* ─── Ring Distances ─── */}
        <Label className='mb-1 fw-bold'>Ring Distances</Label>
        {config.ringDistances.map((d: number, i: number) => (
          <div key={i} className='d-flex align-items-center mb-1 setting-row'>
            <span className='distance-label'>{d}</span>
            <Button size='sm' type='danger' onClick={() => this.removeDistance(i)}>
              ✕
            </Button>
          </div>
        ))}
        <div className='d-flex align-items-center mt-1 setting-row'>
          <NumericInput
            size='sm'
            value={this.state.newDistance}
            min={0.01}
            step={0.25}
            onChange={(value) => this.setState({ newDistance: value })}
          />
          <Button size='sm' type='primary' onClick={this.addDistance}>
            Add
          </Button>
        </div>

        <hr />

        {/* ─── Distance Unit ─── */}
        <Label className='mb-1 fw-bold'>Distance Unit</Label>
        <Select
          value={config.distanceUnit}
          onChange={(e) => this.updateConfig({ distanceUnit: e.target.value })}
        >
          <Option value='miles'>Miles</Option>
          <Option value='kilometers'>Kilometres</Option>
          <Option value='meters'>Metres</Option>
          <Option value='feet'>Feet</Option>
        </Select>

        <hr />

        {/* ─── Max Results ─── */}
        <Label className='mb-1 fw-bold'>Max Features per Layer per Ring</Label>
        <NumericInput
          value={config.maxResults}
          min={1}
          max={5000}
          step={100}
          onChange={(value) => this.updateConfig({ maxResults: value })}
        />

        <hr />

        {/* ─── Ring Colours ─── */}
        <Label className='mb-1 fw-bold'>Ring Colours</Label>
        {config.ringColors.map((c: string, i: number) => (
          <div key={i} className='d-flex align-items-center mb-1 setting-row'>
            <input
              type='color'
              title={`Ring colour ${i + 1}`}
              value={c}
              className='color-picker'
              onChange={(e) => this.updateColor(i, e.target.value)}
            />
            <TextInput
              size='sm'
              value={c}
              onChange={(e) => this.updateColor(i, e.target.value)}
            />
            <Button size='sm' type='danger' onClick={() => this.removeColor(i)}>
              ✕
            </Button>
          </div>
        ))}
        <Button size='sm' type='default' onClick={this.addColor} className='mt-1'>
          + Add Colour
        </Button>

        <hr />

        {/* ─── Ring Opacity ─── */}
        <Label className='mb-1 fw-bold'>Ring Fill Opacity</Label>
        <NumericInput
          value={config.ringOpacity}
          min={0}
          max={1}
          step={0.05}
          onChange={(value) => this.updateConfig({ ringOpacity: value })}
        />

        <hr />

        {/* ─── Centre Colour ─── */}
        <Label className='mb-1 fw-bold'>Centre Point Colour</Label>
        <div className='d-flex align-items-center setting-row'>
          <input
            type='color'
            title='Centre point colour'
            value={config.centerColor}
            className='color-picker'
            onChange={(e) => this.updateConfig({ centerColor: e.target.value })}
          />
          <TextInput
            size='sm'
            value={config.centerColor}
            onChange={(e) => this.updateConfig({ centerColor: e.target.value })}
          />
        </div>

        {config.ringDistances.length === 0 && (
          <Alert type='warning' className='mt-3' open>
            Add at least one ring distance to use the widget.
          </Alert>
        )}
      </div>
    )
  }
}
