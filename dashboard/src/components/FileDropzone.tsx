import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { FlowIcon } from './FlowIcon'

type Props = {
  label?: string
  helperText?: string
  files: File[]
  setFiles: (next: File[]) => void
  accept?: string
  multiple?: boolean
  maxFiles?: number
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export function FileDropzone({
  label = 'Upload files',
  helperText,
  files,
  setFiles,
  accept = 'application/pdf',
  multiple = true,
  maxFiles = 20,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isMobilePickerUi, setIsMobilePickerUi] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const media = window.matchMedia('(max-width: 768px), (pointer: coarse)')
    const update = () => setIsMobilePickerUi(media.matches)
    update()
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', update)
      return () => media.removeEventListener('change', update)
    }
    media.addListener(update)
    return () => media.removeListener(update)
  }, [])

  const acceptLower = accept.toLowerCase()
  const isPdfOnly = acceptLower.includes('pdf')

  const addFiles = useCallback(
    (incoming: File[]) => {
      const cleaned = (incoming || []).filter(Boolean)
      const onlyAccepted = cleaned.filter((f) => {
        if (!isPdfOnly) return true
        const nameOk = (f.name || '').toLowerCase().endsWith('.pdf')
        const typeOk = (f.type || '').toLowerCase().includes('pdf')
        return nameOk || typeOk
      })
      const merged = [...(files || []), ...onlyAccepted]
      const deduped = Array.from(new Map(merged.map((f) => [`${f.name}:${f.size}:${f.lastModified}`, f])).values())
      setFiles(deduped.slice(0, maxFiles))
    },
    [files, isPdfOnly, maxFiles, setFiles]
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      if (isMobilePickerUi) return
      e.preventDefault()
      setIsDragging(false)
      const list = Array.from(e.dataTransfer.files || [])
      addFiles(list)
    },
    [addFiles, isMobilePickerUi]
  )

  const onBrowse = () => inputRef.current?.click()

  const effectiveLabel = useMemo(() => {
    if (!isMobilePickerUi) return label
    return isPdfOnly ? 'Select PDF files' : 'Select files'
  }, [isMobilePickerUi, isPdfOnly, label])

  const subtitle = useMemo(() => {
    if (isMobilePickerUi) {
      if (isPdfOnly) return `Choose up to ${maxFiles} PDF files from your device.`
      return `Choose up to ${maxFiles} files from your device.`
    }
    if (helperText) return helperText
    if (isPdfOnly) return 'Drag & drop PDFs here, or browse.'
    return 'Drag & drop files here, or browse.'
  }, [helperText, isMobilePickerUi, isPdfOnly, maxFiles])

  const browseLabel = isMobilePickerUi
    ? multiple
      ? 'Choose files'
      : 'Choose file'
    : 'Browse files'

  const selectedLabel = files.length > 0
    ? `${files.length} selected`
    : isMobilePickerUi
      ? 'No files selected'
      : '0 selected'

  return (
    <div className="file-dropzone" data-mobile-picker={isMobilePickerUi ? 'true' : 'false'}>
      <div className="design-form-label file-dropzone-label" style={{ marginBottom: '6px' }}>{effectiveLabel}</div>
      <div
        onDragOver={(e) => {
          if (isMobilePickerUi) return
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => {
          if (isMobilePickerUi) return
          setIsDragging(false)
        }}
        onDrop={onDrop}
        className="file-dropzone-surface"
        style={{
          border: isMobilePickerUi
            ? `1px solid var(--flow-border, #f2d8d2)`
            : `2px dashed ${isDragging ? 'var(--flow-accent, #e4587a)' : 'var(--flow-border, #f2d8d2)'}`,
          borderRadius: '14px',
          padding: isMobilePickerUi ? '12px' : '16px',
          background: isMobilePickerUi
            ? 'var(--flow-surface, #fff)'
            : isDragging
              ? 'var(--flow-accent-soft, #fff1ef)'
              : 'var(--flow-surface, #fff)',
          transition: 'border-color 0.2s, background 0.2s',
        }}
      >
        <div className="muted file-dropzone-subtitle" style={{ marginBottom: '10px' }}>{subtitle}</div>
        <div className="row file-dropzone-actions" style={{ gap: '0.75rem', flexWrap: 'wrap' }}>
          <button type="button" className="secondary" onClick={onBrowse}>
            {browseLabel}
          </button>
          <div className="muted file-dropzone-selected">{selectedLabel}</div>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          style={{ display: 'none' }}
          onChange={(e) => {
            const list = Array.from(e.target.files || [])
            addFiles(list)
            // allow selecting same file again
            e.currentTarget.value = ''
          }}
        />

        {files.length > 0 && (
          <div style={{ marginTop: '12px', borderTop: '1px solid var(--flow-border, #e2e8f0)', paddingTop: '12px' }}>
            {files.map((f) => (
              <div key={`${f.name}:${f.size}:${f.lastModified}`} className="row" style={{ justifyContent: 'space-between', gap: '0.75rem', padding: '6px 0', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                  <FlowIcon name="check_circle" size="sm" style={{ color: 'var(--flow-accent, #e4587a)', flexShrink: 0 }} />
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 500, color: 'var(--flow-text, #0f172a)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {f.name}
                    </div>
                    <div className="muted" style={{ fontSize: '0.85rem' }}>{formatBytes(f.size)}</div>
                  </div>
                </div>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => setFiles(files.filter((x) => x !== f))}
                  aria-label="Delete"
                  title="Delete"
                  style={{ padding: '0.4rem', color: 'var(--flow-muted, #64748b)' }}
                >
                  <FlowIcon name="delete" size="sm" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
