import type { LucideIcon } from 'lucide-react'
import {
  Badge,
  Brain,
  Check,
  CheckCircle,
  Code,
  Copy,
  Files,
  Globe,
  Link,
  MousePointerClick,
  Palette,
  PartyPopper,
  Play,
  Plus,
  Printer,
  SquareStop,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react'

const ICONS: Record<string, LucideIcon> = {
  add: Plus,
  ads_click: MousePointerClick,
  badge: Badge,
  celebration: PartyPopper,
  check: Check,
  check_circle: CheckCircle,
  cloud_upload: UploadCloud,
  close: X,
  code: Code,
  content_copy: Copy,
  delete: Trash2,
  language: Globe,
  link: Link,
  model_training: Brain,
  palette: Palette,
  play_arrow: Play,
  print: Printer,
  source: Files,
  stop: SquareStop,
}

export type FlowIconName = keyof typeof ICONS

type FlowIconProps = {
  name: FlowIconName
  filled?: boolean
  size?: 'sm' | 'xs' | 'md'
  className?: string
  style?: React.CSSProperties
  'aria-hidden'?: boolean
}

function resolveSizePx(size: FlowIconProps['size'], fontSize: React.CSSProperties['fontSize']) {
  const baseSize = size === 'sm' ? 18 : size === 'xs' ? 16 : 20
  if (typeof fontSize === 'number') return fontSize
  if (typeof fontSize === 'string') {
    const value = parseFloat(fontSize)
    if (!Number.isNaN(value)) {
      return fontSize.endsWith('rem') ? value * 16 : value
    }
  }
  return baseSize
}

export function FlowIcon({
  name,
  filled = false,
  size = 'md',
  className = '',
  style,
  'aria-hidden': ariaHidden = true,
}: FlowIconProps) {
  const Icon = ICONS[name]
  if (!Icon) return null

  const { color, fontSize, ...restStyle } = style ?? {}
  const resolvedSize = resolveSizePx(size, fontSize)

  const sizeClass = size === 'sm' ? ' flow-icon--sm' : size === 'xs' ? ' flow-icon--xs' : ''
  const filledClass = filled ? ' flow-icon--filled' : ''

  return (
    <Icon
      className={`flow-icon${filledClass}${sizeClass} ${className}`.trim()}
      size={resolvedSize}
      color={color}
      strokeWidth={1.8}
      fill={filled ? 'currentColor' : 'none'}
      style={restStyle}
      aria-hidden={ariaHidden}
    />
  )
}
