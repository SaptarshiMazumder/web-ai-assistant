import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { GlassField, UiButton, UiInput } from '../components/ui'

type DialogTone = 'default' | 'danger'

type ConfirmDialogOptions = {
  title: string
  description?: string
  confirmLabel?: string
  cancelLabel?: string
  tone?: DialogTone
}

type AlertDialogOptions = {
  title: string
  description?: string
  confirmLabel?: string
  tone?: DialogTone
}

type PromptDialogOptions = {
  title: string
  description?: string
  label?: string
  placeholder?: string
  defaultValue?: string
  confirmLabel?: string
  cancelLabel?: string
  tone?: DialogTone
  required?: boolean
}

type PendingConfirmDialog = {
  id: number
  kind: 'confirm'
  options: ConfirmDialogOptions
  resolve: (value: boolean) => void
  returnFocusTo: HTMLElement | null
}

type PendingAlertDialog = {
  id: number
  kind: 'alert'
  options: AlertDialogOptions
  resolve: () => void
  returnFocusTo: HTMLElement | null
}

type PendingPromptDialog = {
  id: number
  kind: 'prompt'
  options: PromptDialogOptions
  resolve: (value: string | null) => void
  returnFocusTo: HTMLElement | null
}

type PendingDialog = PendingConfirmDialog | PendingAlertDialog | PendingPromptDialog

type DialogContextValue = {
  confirm: (options: ConfirmDialogOptions) => Promise<boolean>
  alert: (options: AlertDialogOptions) => Promise<void>
  prompt: (options: PromptDialogOptions) => Promise<string | null>
}

const DialogContext = createContext<DialogContextValue | undefined>(undefined)

function getActiveElement(): HTMLElement | null {
  if (typeof document === 'undefined') return null
  return document.activeElement instanceof HTMLElement ? document.activeElement : null
}

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const nodes = container.querySelectorAll<HTMLElement>(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )
  return Array.from(nodes).filter((element) => {
    if (element.hasAttribute('disabled')) return false
    if (element.getAttribute('aria-hidden') === 'true') return false
    if (element.tabIndex === -1) return false
    return element.getClientRects().length > 0
  })
}

function DialogHost({
  request,
  closeRequest,
}: {
  request: PendingDialog
  closeRequest: (request: PendingDialog, result?: boolean | string | null) => void
}) {
  const { t } = useTranslation()
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const confirmButtonRef = useRef<HTMLButtonElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const titleId = `dashboard-dialog-title-${request.id}`
  const descriptionId = `dashboard-dialog-description-${request.id}`
  const inputId = `dashboard-dialog-input-${request.id}`
  const [promptValue, setPromptValue] = useState(request.kind === 'prompt' ? request.options.defaultValue ?? '' : '')

  useEffect(() => {
    if (request.kind === 'prompt') {
      setPromptValue(request.options.defaultValue ?? '')
      window.requestAnimationFrame(() => {
        inputRef.current?.focus({ preventScroll: true })
        inputRef.current?.select()
      })
      return
    }
    setPromptValue('')
    window.requestAnimationFrame(() => {
      confirmButtonRef.current?.focus({ preventScroll: true })
    })
  }, [request])

  useEffect(() => {
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault()
        if (request.kind === 'alert') {
          closeRequest(request)
        } else if (request.kind === 'confirm') {
          closeRequest(request, false)
        } else {
          closeRequest(request, null)
        }
        return
      }
      if (event.key !== 'Tab') return
      const dialog = dialogRef.current
      if (!dialog) return
      const focusable = getFocusableElements(dialog)
      if (focusable.length === 0) {
        event.preventDefault()
        return
      }
      const active = document.activeElement instanceof HTMLElement ? document.activeElement : null
      const currentIndex = active ? focusable.indexOf(active) : -1
      if (event.shiftKey) {
        if (currentIndex <= 0) {
          focusable[focusable.length - 1]?.focus()
          event.preventDefault()
        }
        return
      }
      if (currentIndex === -1 || currentIndex === focusable.length - 1) {
        focusable[0]?.focus()
        event.preventDefault()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [request, closeRequest])

  const isDanger = (request.options.tone ?? 'default') === 'danger'
  const description = request.options.description?.trim() || ''
  const cancelLabel =
    request.kind === 'alert'
      ? null
      : request.options.cancelLabel || t('common.cancel', 'Cancel')
  const confirmLabel =
    request.options.confirmLabel
    || (request.kind === 'alert'
      ? t('common.ok', 'OK')
      : t('common.confirm', 'Confirm'))
  const promptRequired = request.kind === 'prompt' ? request.options.required === true : false
  const canSubmitPrompt = !promptRequired || promptValue.trim().length > 0

  const handleBackdropClick = () => {
    if (request.kind === 'alert') {
      closeRequest(request)
    } else if (request.kind === 'confirm') {
      closeRequest(request, false)
    } else {
      closeRequest(request, null)
    }
  }

  const handleConfirm = () => {
    if (request.kind === 'alert') {
      closeRequest(request)
      return
    }
    if (request.kind === 'confirm') {
      closeRequest(request, true)
      return
    }
    if (!canSubmitPrompt) return
    closeRequest(request, promptValue.trim())
  }

  return createPortal(
    <div className="app-dialog-overlay" onClick={handleBackdropClick}>
      <div
        ref={dialogRef}
        className={`app-dialog${isDanger ? ' app-dialog--danger' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descriptionId : undefined}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="app-dialog__accent" />
        <div className="app-dialog__header">
          <div className="app-dialog__content">
            <h2 id={titleId} className="app-dialog__title">
              {request.options.title}
            </h2>
            {description && (
              <p id={descriptionId} className="app-dialog__description">
                {description}
              </p>
            )}
          </div>
        </div>
        {request.kind === 'prompt' && (
          <div className="app-dialog__body">
            <GlassField
              label={request.options.label || t('common.value', 'Value')}
              className="app-dialog__field"
              style={{ maxWidth: '100%' }}
            >
              <UiInput
                ref={inputRef}
                id={inputId}
                className="app-dialog__input"
                value={promptValue}
                placeholder={request.options.placeholder}
                onChange={(event) => setPromptValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && canSubmitPrompt) {
                    event.preventDefault()
                    handleConfirm()
                  }
                }}
              />
            </GlassField>
          </div>
        )}
        <div className="app-dialog__actions">
          {cancelLabel && (
            <UiButton
              variant="ghost"
              onClick={() => {
                if (request.kind === 'confirm') {
                  closeRequest(request, false)
                  return
                }
                closeRequest(request, null)
              }}
            >
              {cancelLabel}
            </UiButton>
          )}
          <UiButton
            ref={confirmButtonRef}
            variant={isDanger ? 'danger' : 'primary'}
            onClick={handleConfirm}
            disabled={request.kind === 'prompt' ? !canSubmitPrompt : false}
          >
            {confirmLabel}
          </UiButton>
        </div>
      </div>
    </div>,
    document.body
  )
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const nextIdRef = useRef(1)
  const [queue, setQueue] = useState<PendingDialog[]>([])

  const openConfirm = useCallback((options: ConfirmDialogOptions) => {
    return new Promise<boolean>((resolve) => {
      setQueue((prev) => [
        ...prev,
        {
          id: nextIdRef.current++,
          kind: 'confirm',
          options,
          resolve,
          returnFocusTo: getActiveElement(),
        },
      ])
    })
  }, [])

  const openAlert = useCallback((options: AlertDialogOptions) => {
    return new Promise<void>((resolve) => {
      setQueue((prev) => [
        ...prev,
        {
          id: nextIdRef.current++,
          kind: 'alert',
          options,
          resolve,
          returnFocusTo: getActiveElement(),
        },
      ])
    })
  }, [])

  const openPrompt = useCallback((options: PromptDialogOptions) => {
    return new Promise<string | null>((resolve) => {
      setQueue((prev) => [
        ...prev,
        {
          id: nextIdRef.current++,
          kind: 'prompt',
          options,
          resolve,
          returnFocusTo: getActiveElement(),
        },
      ])
    })
  }, [])

  const closeRequest = useCallback((request: PendingDialog, result?: boolean | string | null) => {
    setQueue((prev) => {
      const current = prev.find((item) => item.id === request.id)
      if (!current) return prev
      window.setTimeout(() => {
        if (current.kind === 'alert') {
          current.resolve()
        } else if (current.kind === 'confirm') {
          current.resolve(result === true)
        } else {
          current.resolve(typeof result === 'string' ? result : null)
        }
        if (current.returnFocusTo && current.returnFocusTo.isConnected) {
          current.returnFocusTo.focus({ preventScroll: true })
        }
      }, 0)
      return prev.filter((item) => item.id !== request.id)
    })
  }, [])

  const activeRequest = queue[0] ?? null
  const value = useMemo<DialogContextValue>(() => ({
    confirm: openConfirm,
    alert: openAlert,
    prompt: openPrompt,
  }), [openAlert, openConfirm, openPrompt])

  return (
    <DialogContext.Provider value={value}>
      {children}
      {activeRequest && <DialogHost request={activeRequest} closeRequest={closeRequest} />}
    </DialogContext.Provider>
  )
}

export function useDialog() {
  const context = useContext(DialogContext)
  if (!context) {
    throw new Error('useDialog must be used within DialogProvider')
  }
  return context
}
