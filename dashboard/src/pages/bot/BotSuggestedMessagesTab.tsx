import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check, Save } from 'lucide-react'
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
  const { selectedBot, selectedBotWidgetConfig, saveWidgetConfig, loading } = useDashboardData()
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

  if (!botId) return <div className="empty-panel">Select a bot.</div>
  if (loading && !selectedBot) return <div className="empty-panel">Loading...</div>
  if (selectedBot?.bot_id !== botId) return <div className="empty-panel">Loading...</div>

  const saveAction = (
    <UiButton
      variant="primary"
      onClick={() => void handleSave()}
      disabled={saving || savedJustNow}
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
  )

  return (
    <AnimatedPage>
      <SectionHeader
        title="Suggested messages"
        subtitle="Quick actions shown to users when the chat opens. Add, edit, or remove them here."
      />

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
