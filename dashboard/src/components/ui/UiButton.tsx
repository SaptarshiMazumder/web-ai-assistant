import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'

export type UiButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

type UiButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
  variant?: UiButtonVariant
  children: ReactNode
}

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}

export const UiButton = forwardRef<HTMLButtonElement, UiButtonProps>(function UiButton(
  {
    type = 'button',
    variant = 'primary',
    className,
    children,
    ...props
  },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx('ui-flow-btn', `ui-flow-btn--${variant}`, className)}
      {...props}
    >
      {children}
    </button>
  )
})
