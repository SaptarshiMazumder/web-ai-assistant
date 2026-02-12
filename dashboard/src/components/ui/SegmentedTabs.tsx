import type { ReactNode } from 'react'

export type SegmentedTabOption<T extends string> = {
  id: T
  label: string
  icon?: ReactNode
  disabled?: boolean
}

type SegmentedTabsProps<T extends string> = {
  value: T
  onChange: (value: T) => void
  options: ReadonlyArray<SegmentedTabOption<T>>
  ariaLabel?: string
}

export function SegmentedTabs<T extends string>({
  value,
  onChange,
  options,
  ariaLabel = 'Tabs',
}: SegmentedTabsProps<T>) {
  return (
    <div className="ui-segmented-tabs" role="tablist" aria-label={ariaLabel}>
      {options.map((option) => {
        const active = option.id === value
        return (
          <button
            key={option.id}
            type="button"
            role="tab"
            aria-selected={active}
            aria-current={active ? 'page' : undefined}
            disabled={option.disabled}
            className={`ui-segmented-tab ${active ? 'is-active' : ''}`}
            onClick={() => onChange(option.id)}
          >
            {option.icon ? <span className="ui-segmented-tab-icon">{option.icon}</span> : null}
            <span>{option.label}</span>
          </button>
        )
      })}
    </div>
  )
}
