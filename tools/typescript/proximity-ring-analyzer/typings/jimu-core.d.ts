/**
 * Stub type declarations for jimu-core.
 *
 * The full types are provided by the Experience Builder framework at
 * build time.  These stubs exist so the widget source compiles cleanly
 * in a standalone editor without the ExB SDK installed.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
declare module 'jimu-core' {
  import * as ReactNS from 'react'

  export const React: typeof ReactNS
  export function jsx (tag: any, props: any, ...children: any[]): any

  export type AllWidgetProps<T> = {
    id: string
    config: T
    useMapWidgetIds?: string[]
    [key: string]: any
  }

  export type AllWidgetSettingProps<T> = {
    id: string
    config: T
    useMapWidgetIds?: string[]
    onSettingChange: (setting: { id: string; config?: T; useMapWidgetIds?: string[] }) => void
    [key: string]: any
  }

  export function Immutable<T> (obj: T): T & { merge: (partial: any) => T; asMutable: (opts?: any) => T }

  export class DataSourceComponent extends ReactNS.Component<any, any> {}

  export interface ImmutableObject<T> {
    [key: string]: any
    merge: (partial: any) => ImmutableObject<T>
    asMutable: (opts?: any) => T
  }
}
