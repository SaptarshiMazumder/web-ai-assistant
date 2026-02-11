import type { HTMLAttributes, ReactNode } from 'react'

type UiCardProps = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode
}

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}

export function UiCard({ className, children, ...props }: UiCardProps) {
  return (
    <div className={cx('ui-flow-card', className)} {...props}>
      {children}
    </div>
  )
}
