import { useCallback, useMemo, useState } from 'react'
import { Plus } from 'lucide-react'
import { FlowIcon } from './FlowIcon'
import type { SuggestedMessageConfig } from './WidgetDesignForm'

type SuggestedMessagesEditorProps = {
  suggestedMessages: SuggestedMessageConfig[]
  onChange: (next: SuggestedMessageConfig[]) => void
  title?: string
  subtitle?: string
  addButtonPlacement?: 'top' | 'bottom'
  maxItems?: number
  actions?: React.ReactNode
}

export function SuggestedMessagesEditor({
  suggestedMessages,
  onChange,
  title = 'Suggested messages',
  subtitle = 'Quick actions shown to users when the chat opens.',
  addButtonPlacement = 'top',
  maxItems,
  actions,
}: SuggestedMessagesEditorProps) {
  const [editingSuggestion, setEditingSuggestion] = useState<SuggestedMessageConfig | null>(null)
  const [suggestionDraft, setSuggestionDraft] = useState<SuggestedMessageConfig | null>(null)
  const [isSuggestionModalOpen, setIsSuggestionModalOpen] = useState(false)
  const canAdd = useMemo(
    () => (typeof maxItems === 'number' ? suggestedMessages.length < maxItems : true),
    [maxItems, suggestedMessages.length]
  )

  const normalizeOneUrl = useCallback((entry: string): string => {
    const raw = (entry || '').trim()
    if (!raw) return ''
    try {
      const parsed = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`)
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return ''
      return parsed.toString()
    } catch {
      return ''
    }
  }, [])

  const openSuggestionModal = useCallback((item?: SuggestedMessageConfig) => {
    if (!item && !canAdd) return
    const base: SuggestedMessageConfig = item
      ? { ...item, type: 'ai_response', urls: Array.isArray(item.urls) ? item.urls : [] }
      : { id: `suggest_${Date.now()}`, label: '', type: 'ai_response', urls: [] }
    setEditingSuggestion(item || null)
    setSuggestionDraft(base)
    setIsSuggestionModalOpen(true)
  }, [canAdd])

  const closeSuggestionModal = useCallback(() => {
    setIsSuggestionModalOpen(false)
    setEditingSuggestion(null)
    setSuggestionDraft(null)
  }, [])

  const saveSuggestion = useCallback(() => {
    if (!suggestionDraft) return
    if (!suggestionDraft.label.trim()) {
      closeSuggestionModal()
      return
    }
    const normalizedUrls = Array.from(
      new Set(
        (Array.isArray(suggestionDraft.urls) ? suggestionDraft.urls : [])
          .map((url) => normalizeOneUrl(url))
          .filter(Boolean)
      )
    )
    const next: SuggestedMessageConfig = {
      ...suggestionDraft,
      type: suggestionDraft.type === 'escalate' ? 'escalate' : 'ai_response',
      message: undefined,
      urls: suggestionDraft.type === 'ai_response' ? normalizedUrls : undefined,
      prompt: suggestionDraft.type === 'ai_response' ? suggestionDraft.prompt : undefined,
    }
    if (!next) {
      closeSuggestionModal()
      return
    }
    if (!editingSuggestion && typeof maxItems === 'number' && suggestedMessages.length >= maxItems) {
      closeSuggestionModal()
      return
    }
    const updated = editingSuggestion
      ? suggestedMessages.map((msg) => (msg.id === editingSuggestion.id ? next : msg))
      : [...suggestedMessages, next]
    onChange(updated)
    closeSuggestionModal()
  }, [suggestionDraft, editingSuggestion, maxItems, normalizeOneUrl, suggestedMessages, onChange, closeSuggestionModal])

  const removeSuggestion = useCallback(
    (id: string) => {
      onChange(suggestedMessages.filter((msg) => msg.id !== id))
    },
    [suggestedMessages, onChange]
  )

  return (
    <>
      <div className="design-form-field design-form-field-full">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: title || subtitle ? 'space-between' : 'flex-end', gap: '0.75rem' }}>
          {(title || subtitle) && (
            <div className="stacked-title">
              <label className="design-form-label">{title}</label>
              <span className="design-form-hint">{subtitle}</span>
            </div>
          )}
          {addButtonPlacement === 'top' && (
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="primary"
                onClick={() => openSuggestionModal()}
                disabled={!canAdd}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}
              >
                <Plus size={15} />
                Add
              </button>
            </div>
          )}
        </div>
        <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.75rem' }}>
          {suggestedMessages.length === 0 && <span className="muted">No suggested messages yet.</span>}
          {suggestedMessages.map((msg) => (
            <div key={msg.id} className="list-row" style={{ background: 'var(--flow-surface, #fff)', border: '1px solid var(--flow-border, #f2d8d2)', borderRadius: 'var(--flow-radius, 10px)' }}>
              <div>
                <div style={{ fontWeight: 600 }}>{msg.label}</div>
                <div className="muted" style={{ fontSize: '0.85rem' }}>
                  {`AI response${Array.isArray(msg.urls) && msg.urls.length ? ` · ${msg.urls.length} URL${msg.urls.length === 1 ? '' : 's'}` : ''}`}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button type="button" className="secondary" onClick={() => openSuggestionModal(msg)}>
                  Edit
                </button>
                <button type="button" className="ghost" onClick={() => removeSuggestion(msg.id)} aria-label="Delete" title="Delete" style={{ padding: '0.4rem', color: 'var(--flow-muted, #64748b)' }}>
                  <FlowIcon name="delete" size="sm" />
                </button>
              </div>
            </div>
          ))}
        </div>
        {(addButtonPlacement === 'bottom' || actions) && (
          <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
              {addButtonPlacement === 'bottom' && (
                <button
                  type="button"
                  className="primary"
                  onClick={() => openSuggestionModal()}
                  disabled={!canAdd}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}
                >
                  <Plus size={15} />
                  Add
                </button>
              )}
              {typeof maxItems === 'number' && (
                <span className="muted" style={{ fontSize: '0.85rem' }}>
                  {suggestedMessages.length}/{maxItems}
                </span>
              )}
            </div>
            {actions ? <div>{actions}</div> : null}
          </div>
        )}
      </div>

      {isSuggestionModalOpen && suggestionDraft && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <div>
                <div className="modal-title">
                  {editingSuggestion ? 'Edit Suggested Message' : 'Add Suggested Message'}
                </div>
                <div className="modal-subtitle">Update the suggested message details.</div>
              </div>
              <button type="button" className="modal-close modal-close--circle" onClick={closeSuggestionModal} aria-label="Close">
                <FlowIcon name="close" />
              </button>
            </div>
            <div className="modal-body">
              <label className="design-form-label">Name</label>
              <input
                type="text"
                className="design-form-input"
                value={suggestionDraft.label}
                onChange={(e) =>
                  setSuggestionDraft((prev) => (prev ? { ...prev, label: e.target.value } : prev))
                }
                placeholder="Where are success stories?"
              />
              <label className="design-form-label" style={{ marginTop: '1rem' }}>
                Prompt
              </label>
              <textarea
                className="design-form-input"
                rows={3}
                value={suggestionDraft.prompt || ''}
                onChange={(e) =>
                  setSuggestionDraft((prev) => (prev ? { ...prev, prompt: e.target.value } : prev))
                }
                placeholder="Can you show me some user success stories?"
              />
              <label className="design-form-label" style={{ marginTop: '1rem' }}>
                URLs (optional)
              </label>
              <textarea
                className="design-form-input"
                rows={3}
                value={Array.isArray(suggestionDraft.urls) ? suggestionDraft.urls.join('\n') : ''}
                onChange={(e) =>
                  setSuggestionDraft((prev) =>
                    prev ? { ...prev, urls: e.target.value.split('\n').map((line) => line.trim()).filter(Boolean) } : prev
                  )
                }
                placeholder={'https://example.com/pricing\nhttps://example.com/faq'}
              />
              <div className="muted" style={{ marginTop: '0.45rem', fontSize: '0.8rem' }}>
                One URL per line.
              </div>
            </div>
            <div className="modal-actions">
              <button type="button" className="secondary" onClick={closeSuggestionModal}>
                Cancel
              </button>
              <button type="button" className="primary" onClick={saveSuggestion}>
                {editingSuggestion ? 'Update' : 'Add'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
