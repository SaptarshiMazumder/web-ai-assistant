import { useCallback, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Plus, Settings } from 'lucide-react'
import { useTranslation } from 'react-i18next'
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
  /** Config-driven: only these types appear in the dropdown. Default: all (ai_response, show_menu, escalate). */
  availableTypes?: Array<'ai_response' | 'show_menu' | 'escalate'>
  /** When set, shows a link to Human Support settings when editing an escalate-type message. */
  botId?: string
}

const ALL_TYPES: Array<'ai_response' | 'show_menu' | 'escalate'> = ['ai_response', 'show_menu', 'escalate']

export function SuggestedMessagesEditor({
  suggestedMessages,
  onChange,
  title,
  subtitle,
  addButtonPlacement = 'top',
  maxItems,
  actions,
  availableTypes = ALL_TYPES,
  botId,
}: SuggestedMessagesEditorProps) {
  const { t } = useTranslation()
  const resolvedTitle = title ?? t('suggestedMessagesEditor.title', 'Suggested messages')
  const resolvedSubtitle = subtitle ?? t('suggestedMessagesEditor.subtitle', 'Quick actions shown to users when the chat opens.')

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
    const defaultType = (availableTypes[0] || 'ai_response') as SuggestedMessageConfig['type']
    const base: SuggestedMessageConfig = item
      ? {
          ...item,
          urls: Array.isArray(item.urls) ? item.urls : [],
          type: availableTypes.includes(item.type) ? item.type : defaultType,
        }
      : { id: `suggest_${Date.now()}`, label: '', type: defaultType, urls: [] }
    setEditingSuggestion(item || null)
    setSuggestionDraft(base)
    setIsSuggestionModalOpen(true)
  }, [canAdd, availableTypes])

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
      type: suggestionDraft.type === 'escalate' ? 'escalate' : suggestionDraft.type === 'show_menu' ? 'show_menu' : 'ai_response',
      message: undefined,
      urls: suggestionDraft.type === 'ai_response' ? normalizedUrls : undefined,
      prompt: suggestionDraft.type === 'ai_response' ? suggestionDraft.prompt : undefined,
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
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: resolvedTitle || resolvedSubtitle ? 'space-between' : 'flex-end', gap: '0.75rem' }}>
          {(resolvedTitle || resolvedSubtitle) && (
            <div className="stacked-title">
              <label className="design-form-label">{resolvedTitle}</label>
              <span className="design-form-hint">{resolvedSubtitle}</span>
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
                {t('suggestedMessagesEditor.add', 'Add')}
              </button>
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.75rem' }}>
          {suggestedMessages.length === 0 && <span className="muted">{t('suggestedMessagesEditor.noneYet', 'No suggested messages yet.')}</span>}
          {suggestedMessages.map((msg) => {
            const urlCount = Array.isArray(msg.urls) ? msg.urls.length : 0
            const typeLabel =
              msg.type === 'escalate'
                ? t('suggestedMessagesEditor.humanSupport', 'Human support')
                : msg.type === 'show_menu'
                  ? t('suggestedMessagesEditor.showMenu', 'Show menu')
                  : urlCount > 0
                    ? `${t('suggestedMessagesEditor.aiResponse', 'AI response')} - ${t(
                        urlCount === 1 ? 'suggestedMessagesEditor.urlSingular' : 'suggestedMessagesEditor.urlPlural',
                        urlCount === 1 ? '{{count}} URL' : '{{count}} URLs',
                        { count: urlCount }
                      )}`
                    : t('suggestedMessagesEditor.aiResponse', 'AI response')

            return (
              <div key={msg.id} className="list-row" style={{ background: 'var(--flow-surface, #fff)', border: '1px solid var(--flow-border, #f2d8d2)', borderRadius: 'var(--flow-radius, 10px)' }}>
                <div>
                  <div style={{ fontWeight: 600 }}>{msg.label}</div>
                  <div className="muted" style={{ fontSize: '0.85rem' }}>
                    {typeLabel}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button type="button" className="secondary" onClick={() => openSuggestionModal(msg)}>
                    {t('suggestedMessagesEditor.edit', 'Edit')}
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => removeSuggestion(msg.id)}
                    aria-label={t('suggestedMessagesEditor.delete', 'Delete')}
                    title={t('suggestedMessagesEditor.delete', 'Delete')}
                    style={{ padding: '0.4rem', color: 'var(--flow-muted, #64748b)' }}
                  >
                    <FlowIcon name="delete" size="sm" />
                  </button>
                </div>
              </div>
            )
          })}
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
                  {t('suggestedMessagesEditor.add', 'Add')}
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
                  {editingSuggestion
                    ? t('suggestedMessagesEditor.modalEditTitle', 'Edit Suggested Message')
                    : t('suggestedMessagesEditor.modalAddTitle', 'Add Suggested Message')}
                </div>
                <div className="modal-subtitle">{t('suggestedMessagesEditor.modalSubtitle', 'Update the suggested message details.')}</div>
              </div>
              <button type="button" className="modal-close modal-close--circle" onClick={closeSuggestionModal} aria-label={t('suggestedMessagesEditor.close', 'Close')}>
                <FlowIcon name="close" />
              </button>
            </div>

            <div className="modal-body">
              <label className="design-form-label">{t('suggestedMessagesEditor.typeLabel', 'Type')}</label>
              <select
                className="design-form-input"
                value={availableTypes.includes(suggestionDraft.type) ? suggestionDraft.type : availableTypes[0]}
                onChange={(e) =>
                  setSuggestionDraft((prev) =>
                    prev ? { ...prev, type: e.target.value as SuggestedMessageConfig['type'] } : prev
                  )
                }
              >
                {availableTypes.includes('ai_response') && (
                  <option value="ai_response">{t('suggestedMessagesEditor.aiResponse', 'AI response')}</option>
                )}
                {availableTypes.includes('show_menu') && (
                  <option value="show_menu">{t('suggestedMessagesEditor.showMenu', 'Show menu')}</option>
                )}
                {availableTypes.includes('escalate') && (
                  <option value="escalate">{t('suggestedMessagesEditor.humanSupport', 'Human support')}</option>
                )}
              </select>

              <label className="design-form-label" style={{ marginTop: '1rem' }}>{t('suggestedMessagesEditor.nameLabel', 'Name')}</label>
              <input
                type="text"
                className="design-form-input"
                value={suggestionDraft.label}
                onChange={(e) => setSuggestionDraft((prev) => (prev ? { ...prev, label: e.target.value } : prev))}
                placeholder={
                  suggestionDraft.type === 'show_menu'
                    ? t('suggestedMessagesEditor.namePlaceholderMenu', 'Menu')
                    : suggestionDraft.type === 'escalate'
                      ? t('suggestedMessagesEditor.namePlaceholderSupport', 'Request human support')
                      : t('suggestedMessagesEditor.namePlaceholder', 'Where are success stories?')
                }
              />

              {suggestionDraft.type === 'escalate' && botId && (
                <div style={{ marginTop: '1rem', padding: '0.75rem', borderRadius: 8, background: 'rgba(255,241,239,0.5)', border: '1px solid var(--ui-flow-border)' }}>
                  <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--ui-flow-text)' }}>
                    {t('suggestedMessagesEditor.configureSupportHint', 'Configure email notifications and button label in Human Support settings.')}
                  </p>
                  <Link
                    to={`/bots/${botId}/human-support`}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--ui-flow-accent)' }}
                  >
                    <Settings size={14} />
                    {t('suggestedMessagesEditor.openSupportSettings', 'Open Human Support settings')}
                  </Link>
                </div>
              )}

              {suggestionDraft.type === 'ai_response' && (
                <>
              <label className="design-form-label" style={{ marginTop: '1rem' }}>
                {t('suggestedMessagesEditor.promptLabel', 'Prompt')}
              </label>
              <textarea
                className="design-form-input"
                rows={3}
                value={suggestionDraft.prompt || ''}
                onChange={(e) => setSuggestionDraft((prev) => (prev ? { ...prev, prompt: e.target.value } : prev))}
                placeholder={t('suggestedMessagesEditor.promptPlaceholder', 'Can you show me some user success stories?')}
              />

              <label className="design-form-label" style={{ marginTop: '1rem' }}>
                {t('suggestedMessagesEditor.urlsLabel', 'URLs (optional)')}
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
                placeholder={t('suggestedMessagesEditor.urlsPlaceholder', 'https://example.com/pricing\nhttps://example.com/faq')}
              />
              <div className="muted" style={{ marginTop: '0.45rem', fontSize: '0.8rem' }}>
                {t('suggestedMessagesEditor.urlsHelper', 'One URL per line.')}
              </div>
                </>
              )}
            </div>

            <div className="modal-actions">
              <button type="button" className="secondary" onClick={closeSuggestionModal}>
                {t('suggestedMessagesEditor.cancel', 'Cancel')}
              </button>
              <button type="button" className="primary" onClick={saveSuggestion}>
                {editingSuggestion ? t('suggestedMessagesEditor.update', 'Update') : t('suggestedMessagesEditor.add', 'Add')}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
