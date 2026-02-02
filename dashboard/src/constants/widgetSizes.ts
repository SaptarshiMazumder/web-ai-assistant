/**
 * Widget width/height per size. Must match backend api/widget/widget.js sizeMap
 * so Design preview, Testing iframe, and live website all show the same dimensions.
 */
export const WIDGET_SIZE_DIMENSIONS = {
  small: { width: 320, height: 420 },
  medium: { width: 380, height: 560 },
  large: { width: 440, height: 640 },
} as const

export type WidgetSizeKey = keyof typeof WIDGET_SIZE_DIMENSIONS
