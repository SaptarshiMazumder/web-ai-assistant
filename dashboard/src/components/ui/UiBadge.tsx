import type { HTMLAttributes, ReactNode } from 'react'

type UiBadgeProps = HTMLAttributes<HTMLSpanElement> & {
  children: ReactNode
}

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}

export function UiBadge({ className, children, ...props }: UiBadgeProps) {
  return (
    <span className={cx('ui-flow-chip', className)} {...props}>
      {children}
    </span>
  )
}
