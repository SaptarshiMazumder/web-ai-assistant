import type { ReactNode } from 'react'

type EmptyStateProps = {
  title: string
  description: string
  action?: ReactNode
  className?: string
}

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(' ')
}

export function EmptyState({ title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cx('ui-empty-state', className)}>
      <div className="ui-empty-state-visual" aria-hidden>
        <span className="ui-empty-state-orb ui-empty-state-orb--one" />
        <span className="ui-empty-state-orb ui-empty-state-orb--two" />
        <span className="ui-empty-state-orb ui-empty-state-orb--three" />
      </div>
      <h3 className="ui-empty-state-title">{title}</h3>
      <p className="ui-empty-state-description">{description}</p>
      {action ? <div className="ui-empty-state-action">{action}</div> : null}
    </div>
  )
}
