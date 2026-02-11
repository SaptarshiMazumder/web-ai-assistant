import type { ButtonHTMLAttributes, ReactNode } from 'react'

export type UiButtonVariant = 'primary' | 'secondary' | 'ghost'

type UiButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
  variant?: UiButtonVariant
  children: ReactNode
}

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}

export function UiButton({
  type = 'button',
  variant = 'primary',
  className,
  children,
  ...props
}: UiButtonProps) {
  return (
    <button
      type={type}
      className={cx('ui-flow-btn', `ui-flow-btn--${variant}`, className)}
      {...props}
    >
      {children}
    </button>
  )
}
