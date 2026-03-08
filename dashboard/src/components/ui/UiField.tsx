import { forwardRef, type InputHTMLAttributes, type TextareaHTMLAttributes } from 'react'

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}

export const UiInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function UiInput(
  { className, ...props },
  ref
) {
  return <input ref={ref} className={cx('ui-flow-field-input', className)} {...props} />
})

export const UiTextArea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function UiTextArea(
  { className, ...props },
  ref
) {
  return <textarea ref={ref} className={cx('ui-flow-field-input', className)} {...props} />
})
