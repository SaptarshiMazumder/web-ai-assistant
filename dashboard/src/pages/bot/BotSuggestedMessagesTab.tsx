import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check, Loader2, Save, Sparkles } from 'lucide-react'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  widgetConfigToState,
  stateToWidgetConfig,
  type WidgetDesignState,
} from '../../components/WidgetDesignForm'
import { SuggestedMessagesEditor } from '../../components/SuggestedMessagesEditor'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'

const SAVED_FEEDBACK_MS = 2000

export default function BotSuggestedMessagesTab() {
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

  if (!botId) return <div className="empty-panel">Select a bot.</div>
  if (loading && !selectedBot) return <div className="empty-panel">Loading...</div>
  if (selectedBot?.bot_id !== botId) return <div className="empty-panel">Loading...</div>

  const isGenerating = generatingSuggestions

  const saveAction = (
    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
      <UiButton
        variant="secondary"
        onClick={() => void handleAutoGenerate()}
        disabled={isGenerating || saving}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
      >
        {isGenerating ? (
          <>
            <Loader2 size={15} className="spin" />
            Generating...
          </>
        ) : (
          <>
            <Sparkles size={15} />
            Auto-generate
          </>
        )}
      </UiButton>
      <UiButton
        variant="primary"
        onClick={() => void handleSave()}
        disabled={saving || savedJustNow || isGenerating}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
      >
        {saving ? (
          'Saving...'
        ) : savedJustNow ? (
          <>
            <Check size={18} strokeWidth={2.5} aria-hidden />
            <span>Saved</span>
          </>
        ) : (
          <>
            <Save size={16} />
            Save
          </>
        )}
      </UiButton>
    </div>
  )

  return (
    <AnimatedPage>
      <SectionHeader
        title="Suggested messages"
        subtitle="Quick actions shown when the chat opens. Auto-generate from your trained content or add manually."
      />

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
          Generating suggested messages from your trained content...
        </div>
      )}

      <GlassCard style={{ marginTop: '1rem' }}>
        <SuggestedMessagesEditor
          suggestedMessages={state.suggestedMessages}
          onChange={(next) => update('suggestedMessages', next)}
          title=""
          subtitle=""
          addButtonPlacement="bottom"
          maxItems={10}
          actions={saveAction}
        />
      </GlassCard>
    </AnimatedPage>
  )
}
