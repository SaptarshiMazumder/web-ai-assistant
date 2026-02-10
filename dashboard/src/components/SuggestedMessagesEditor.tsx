import { useCallback, useState } from 'react'
import { FlowIcon } from './FlowIcon'
import type { SuggestedMessageConfig } from './WidgetDesignForm'

type SuggestedMessagesEditorProps = {
  suggestedMessages: SuggestedMessageConfig[]
  onChange: (next: SuggestedMessageConfig[]) => void
  title?: string
  subtitle?: string
}

export function SuggestedMessagesEditor({
  suggestedMessages,
  onChange,
  title = 'Suggested messages',
  subtitle = 'Quick actions shown to users when the chat opens.',
}: SuggestedMessagesEditorProps) {
  const [editingSuggestion, setEditingSuggestion] = useState<SuggestedMessageConfig | null>(null)
  const [suggestionDraft, setSuggestionDraft] = useState<SuggestedMessageConfig | null>(null)
  const [isSuggestionModalOpen, setIsSuggestionModalOpen] = useState(false)

  const openSuggestionModal = useCallback((item?: SuggestedMessageConfig) => {
    const base: SuggestedMessageConfig = item
      ? { ...item }
      : {
          id: `suggest_${Date.now()}`,
          label: '',
          type: 'user_message',
          message: '',
        }
    setEditingSuggestion(item || null)
    setSuggestionDraft(base)
    setIsSuggestionModalOpen(true)
  }, [])

  const closeSuggestionModal = useCallback(() => {
    setIsSuggestionModalOpen(false)
    setEditingSuggestion(null)
    setSuggestionDraft(null)
  }, [])

  const saveSuggestion = useCallback(() => {
    if (!suggestionDraft) return
    const next = suggestionDraft.label.trim() ? suggestionDraft : null
    if (!next) {
      closeSuggestionModal()
      return
    }
    const updated = editingSuggestion
      ? suggestedMessages.map((msg) => (msg.id === editingSuggestion.id ? next : msg))
      : [...suggestedMessages, next]
    onChange(updated)
    closeSuggestionModal()
  }, [suggestionDraft, editingSuggestion, suggestedMessages, onChange, closeSuggestionModal])

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
            <div>
              <label className="design-form-label">{title}</label>
              <span className="design-form-hint">{subtitle}</span>
            </div>
          )}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button type="button" className="primary" onClick={() => openSuggestionModal()}>
              Add
            </button>
          </div>
        </div>
        <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.75rem' }}>
          {suggestedMessages.length === 0 && <span className="muted">No suggested messages yet.</span>}
          {suggestedMessages.map((msg) => (
            <div key={msg.id} className="list-row" style={{ background: 'var(--flow-surface, #fff)', border: '1px solid var(--flow-border, #f2d8d2)', borderRadius: 'var(--flow-radius, 10px)' }}>
              <div>
                <div style={{ fontWeight: 600 }}>{msg.label}</div>
                <div className="muted" style={{ fontSize: '0.85rem' }}>
                  {msg.type === 'ai_response'
                    ? 'AI response'
                    : msg.type === 'escalate'
                    ? 'Escalate to support'
                    : msg.type === 'availability'
                    ? 'Check availability form'
                    : 'User message'}
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
                Type
              </label>
              <select
                className="design-form-input"
                value={suggestionDraft.type}
                onChange={(e) => {
                  const newType = e.target.value as SuggestedMessageConfig['type']
                  setSuggestionDraft((prev) => {
                    if (!prev) return prev
                    const next = { ...prev, type: newType }
                    if (newType === 'availability' && !prev.label.trim()) {
                      next.label = 'Check availability'
                    }
                    return next
                  })
                }}
              >
                <option value="user_message">User message</option>
                <option value="ai_response">AI response</option>
                <option value="escalate">Escalate to support</option>
                <option value="availability">Check availability form</option>
              </select>
              {suggestionDraft.type === 'ai_response' && (
                <>
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
                </>
              )}
              {suggestionDraft.type === 'user_message' && (
                <>
                  <label className="design-form-label" style={{ marginTop: '1rem' }}>
                    Message
                  </label>
                  <textarea
                    className="design-form-input"
                    rows={2}
                    value={suggestionDraft.message || ''}
                    onChange={(e) =>
                      setSuggestionDraft((prev) => (prev ? { ...prev, message: e.target.value } : prev))
                    }
                    placeholder="What can you do?"
                  />
                </>
              )}
              {suggestionDraft.type === 'escalate' && (
                <div className="muted" style={{ marginTop: '0.75rem' }}>
                  Visitors will be prompted for their email before escalation is submitted.
                </div>
              )}
              {suggestionDraft.type === 'availability' && (
                <div className="muted" style={{ marginTop: '0.75rem' }}>
                  Visitors will see a form to enter check-in, check-out dates, and guest details before the availability check runs.
                </div>
              )}
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
