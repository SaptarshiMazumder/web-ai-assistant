import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check, Loader2, Save, Sparkles } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  widgetConfigToState,
  stateToWidgetConfig,
  type WidgetDesignState,
  type SuggestedMessageConfig,
} from '../../components/WidgetDesignForm'
import { SuggestedMessagesEditor } from '../../components/SuggestedMessagesEditor'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'

const SAVED_FEEDBACK_MS = 2000

export default function BotSuggestedMessagesTab() {
  const { t } = useTranslation()
  const { botId } = useParams()
  const {
    selectedBot,
    selectedBotWidgetConfig,
    saveWidgetConfig,
    loading,
    generateSuggestedMessages,
    generatingSuggestions,
  } = useDashboardData()
  const [state, setState] = useState<WidgetDesignState>(() => DEFAULT_WIDGET_DESIGN_STATE)
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)

  useEffect(() => {
    setState(widgetConfigToState(selectedBotWidgetConfig ?? null))
  }, [selectedBotWidgetConfig])

  const update = useCallback(<K extends keyof WidgetDesignState>(key: K, value: WidgetDesignState[K]) => {
    setState((prev) => ({ ...prev, [key]: value }))
  }, [])

  // Only ai_response messages go into the editor; escalate is managed in the Human Support tab.
  const aiMessages = state.suggestedMessages.filter((m) => m.type !== 'escalate')
  const escalateMessages = state.suggestedMessages.filter((m) => m.type === 'escalate')

  const handleAiMessagesChange = (next: SuggestedMessageConfig[]) => {
    update('suggestedMessages', [...next, ...escalateMessages])
  }

  const handleToggleEnabled = async (enabled: boolean) => {
    if (!botId) return
    update('suggestedMessagesEnabled', enabled)
    setSaving(true)
    try {
      const newState = { ...state, suggestedMessagesEnabled: enabled }
      await saveWidgetConfig(botId, stateToWidgetConfig(newState))
    } finally {
      setSaving(false)
    }
  }

  const handleSave = async () => {
    if (!botId || saving || savedJustNow) return
    setSaving(true)
    setSavedJustNow(false)
    try {
      await saveWidgetConfig(botId, stateToWidgetConfig(state))
      setSavedJustNow(true)
      setTimeout(() => setSavedJustNow(false), SAVED_FEEDBACK_MS)
    } finally {
      setSaving(false)
    }
  }

  const handleAutoGenerate = async () => {
    if (!botId || generatingSuggestions) return
    await generateSuggestedMessages(botId)
  }

  if (!botId) return <div className="empty-panel">{t('botSuggestedMessages.selectBot', 'Select a bot.')}</div>
  if (loading && !selectedBot) return <div className="empty-panel">{t('common.working', 'Working...')}</div>
  if (selectedBot?.bot_id !== botId) return <div className="empty-panel">{t('common.working', 'Working...')}</div>

  const isGenerating = generatingSuggestions
  const isEnabled = state.suggestedMessagesEnabled

  const saveAction = (
    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
      <UiButton
        variant="secondary"
        onClick={() => void handleAutoGenerate()}
        disabled={isGenerating || saving || !isEnabled}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
      >
        {isGenerating ? (
          <>
            <Loader2 size={15} className="spin" />
            {t('botSuggestedMessages.generating', 'Generating...')}
          </>
        ) : (
          <>
            <Sparkles size={15} />
            {t('botSuggestedMessages.autoGenerate', 'Auto-generate')}
          </>
        )}
      </UiButton>
      <UiButton
        variant="primary"
        onClick={() => void handleSave()}
        disabled={saving || savedJustNow || isGenerating || !isEnabled}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
      >
        {saving ? (
          t('botSuggestedMessages.saving', 'Saving...')
        ) : savedJustNow ? (
          <>
            <Check size={18} strokeWidth={2.5} aria-hidden />
            <span>{t('botSuggestedMessages.saved', 'Saved')}</span>
          </>
        ) : (
          <>
            <Save size={16} />
            {t('botSuggestedMessages.save', 'Save')}
          </>
        )}
      </UiButton>
    </div>
  )

  return (
    <AnimatedPage>
      <SectionHeader
        title={t('botSuggestedMessages.title', 'Suggested messages')}
        subtitle={t('botSuggestedMessages.subtitle', 'Quick actions shown when the chat opens. Auto-generate from your trained content or add manually.')}
      />

      <GlassCard style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>{t('botSuggestedMessages.sectionTitle', 'Suggested messages')}</div>
            <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
              {t('botSuggestedMessages.sectionSubtitle', 'Quick actions shown when the chat opens.')}
            </p>
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={isEnabled}
              onChange={(e) => void handleToggleEnabled(e.target.checked)}
              disabled={saving}
            />
            <span className="toggle-slider" />
          </label>
        </div>

        {isGenerating && (
          <div
            style={{
              marginTop: '1rem',
              padding: '0.75rem 1rem',
              background: 'var(--flow-surface-alt, #fef3f0)',
              border: '1px solid var(--flow-border, #f2d8d2)',
              borderRadius: 'var(--flow-radius, 10px)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              fontSize: '0.9rem',
              color: 'var(--flow-text, #1e293b)',
            }}
          >
            <Loader2 size={18} className="spin" style={{ color: 'var(--flow-primary, #e8614d)' }} />
            {t('botSuggestedMessages.generatingFromContent', 'Generating suggested messages from your trained content...')}
          </div>
        )}

        <div style={{ marginTop: '1rem', opacity: isEnabled ? 1 : 0.45, pointerEvents: isEnabled ? 'auto' : 'none' }}>
          <SuggestedMessagesEditor
            suggestedMessages={aiMessages}
            onChange={handleAiMessagesChange}
            title=""
            subtitle=""
            addButtonPlacement="bottom"
            maxItems={10}
            actions={saveAction}
          />
        </div>
      </GlassCard>
    </AnimatedPage>
  )
}
