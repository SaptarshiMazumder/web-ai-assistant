import { useEffect, useId, useRef, useState } from 'react'

type FlowSelectOption = {
  value: string
  label: string
  disabled?: boolean
}

type FlowSelectProps = {
  value: string
  onChange: (value: string) => void
  options: FlowSelectOption[]
  placeholder?: string
  className?: string
  menuClassName?: string
  disabled?: boolean
}

export function FlowSelect({
  value,
  onChange,
  options,
  placeholder = 'Select...',
  className = '',
  menuClassName = '',
  disabled = false,
}: FlowSelectProps) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const rootRef = useRef<HTMLDivElement | null>(null)
  const selected = options.find((opt) => opt.value === value)
  const label = selected?.label || placeholder
  const isPlaceholder = !selected || selected.value === ''

  useEffect(() => {
    if (!open) return
    const handleClick = (event: MouseEvent) => {
      if (!rootRef.current) return
      if (!rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKey)
    }
  }, [open])

  return (
    <div ref={rootRef} className={`flow-select ${className}`.trim()}>
      <button
        type="button"
        className={`flow-select-trigger ${isPlaceholder ? 'is-placeholder' : ''}`.trim()}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={id}
        disabled={disabled}
        onClick={() => {
          if (!disabled) setOpen((prev) => !prev)
        }}
      >
        <span>{label}</span>
        <svg className="flow-select-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div id={id} role="listbox" className={`flow-select-menu ${menuClassName}`.trim()}>
          {options.map((opt) => {
            const isActive = opt.value === value
            return (
              <button
                type="button"
                key={opt.value}
                role="option"
                aria-selected={isActive}
                className={`flow-select-option ${isActive ? 'active' : ''}`.trim()}
                disabled={opt.disabled}
                onClick={() => {
                  if (opt.disabled) return
                  onChange(opt.value)
                  setOpen(false)
                }}
              >
                {opt.label}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
