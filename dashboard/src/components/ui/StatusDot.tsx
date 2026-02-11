import type { HTMLAttributes } from 'react'

type StatusTone = 'success' | 'warning' | 'danger' | 'neutral'

type StatusDotProps = HTMLAttributes<HTMLSpanElement> & {
  tone?: StatusTone
}

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(' ')
}

export function StatusDot({ tone = 'success', className, ...props }: StatusDotProps) {
  return <span className={cx('status-dot', `status-dot--${tone}`, className)} {...props} />
}
