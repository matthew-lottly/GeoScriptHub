/**
 * Stub type declarations for jimu-ui and jimu-ui/advanced/setting-components.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
declare module 'jimu-ui' {
  import * as ReactNS from 'react'

  export class Button extends ReactNS.Component<any, any> {}
  export class Loading extends ReactNS.Component<any, any> {}
  export class Card extends ReactNS.Component<any, any> {}
  export class CardBody extends ReactNS.Component<any, any> {}
  export class Alert extends ReactNS.Component<any, any> {}
  export class Collapse extends ReactNS.Component<{ isOpen: boolean; children?: any }, any> {}
  export class TextInput extends ReactNS.Component<any, any> {}
  export class NumericInput extends ReactNS.Component<any, any> {}
  export class Select extends ReactNS.Component<any, any> {}
  export class Option extends ReactNS.Component<any, any> {}
  export class Label extends ReactNS.Component<any, any> {}
}

declare module 'jimu-ui/advanced/setting-components' {
  import * as ReactNS from 'react'

  export class MapWidgetSelector extends ReactNS.Component<{
    useMapWidgetIds?: string[]
    onSelect: (ids: string[]) => void
  }> {}
}
