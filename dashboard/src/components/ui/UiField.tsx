import type { InputHTMLAttributes, TextareaHTMLAttributes } from 'react'

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}

export function UiInput({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cx('ui-flow-field-input', className)} {...props} />
}

export function UiTextArea({ className, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cx('ui-flow-field-input', className)} {...props} />
}
