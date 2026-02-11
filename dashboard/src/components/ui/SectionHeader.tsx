import type { ReactNode } from 'react'

type SectionHeaderProps = {
  eyebrow?: string
  title: string
  titleAccessory?: ReactNode
  subtitle?: string
  action?: ReactNode
  className?: string
}

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(' ')
}

export function SectionHeader({ eyebrow, title, titleAccessory, subtitle, action, className }: SectionHeaderProps) {
  return (
    <div className={cx('ui-section-header', className)}>
      <div className="ui-section-header-copy">
        {eyebrow ? <span className="ui-section-header-eyebrow">{eyebrow}</span> : null}
        <div className="ui-section-header-title-row">
          <h2 className="ui-section-header-title">{title}</h2>
          {titleAccessory ? <span className="ui-section-header-title-accessory">{titleAccessory}</span> : null}
        </div>
        {subtitle ? <p className="ui-section-header-subtitle">{subtitle}</p> : null}
      </div>
      {action ? <div className="ui-section-header-action">{action}</div> : null}
    </div>
  )
}
