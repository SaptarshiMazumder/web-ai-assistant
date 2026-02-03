import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check } from 'lucide-react'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  widgetConfigToState,
  stateToWidgetConfig,
  type WidgetDesignState,
} from '../../components/WidgetDesignForm'
import { SuggestedMessagesEditor } from '../../components/SuggestedMessagesEditor'
import { WidgetPreview } from '../createBot/WidgetPreview'
import { useDashboardData } from '../../hooks/useDashboardData'

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

  useEffect(() => {
    if (selectedBot?.display_name && state.widgetTitle === 'Chat') {
      setState((prev) => ({ ...prev, widgetTitle: selectedBot.display_name.trim() }))
    }
  }, [selectedBot?.display_name, state.widgetTitle])

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

  if (!botId) {
    return <div className="empty-panel">Select a bot to edit suggested messages.</div>
  }

  if (loading && !selectedBot) {
    return <div className="empty-panel">Loading...</div>
  }

  if (selectedBot?.bot_id !== botId) {
    return <div className="empty-panel">Loading...</div>
  }

  return (
    <div className="flow-panel-body">
      <div>
        <h2 className="card-title" style={{ marginBottom: '0.25rem' }}>Suggested messages</h2>
        <p className="card-subtitle" style={{ margin: 0 }}>
          Configure quick replies that appear in the chat widget.
        </p>
      </div>

      <div className="widget-design-grid">
        <div className="design-form">
          <section className="card">
            <div className="card-title">Suggestions</div>
            <div className="design-form-section">
              <SuggestedMessagesEditor
                suggestedMessages={state.suggestedMessages}
                onChange={(next) => update('suggestedMessages', next)}
                title="Suggested messages"
                subtitle="These appear above the input when the widget opens."
              />
            </div>
          </section>

          <div className="flow-actions">
            <button
              type="button"
              className="primary"
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
                'Save'
              )}
            </button>
          </div>
        </div>

        <WidgetPreview
          position={state.widgetPosition}
          primaryColor={state.widgetPrimaryColor}
          title={state.widgetTitle || 'Chat'}
          size={state.widgetSize}
          welcomeMessage={state.welcomeMessage}
          placeholder={state.placeholder}
          footerMessage={state.footerMessage}
          theme={state.theme}
          textColor={state.textColor}
          headerIconUrl={state.headerIconUrl}
          launcherIconUrl={state.launcherIconUrl}
          launcherText={state.launcherText}
          maxHeight={state.maxHeight}
          fontSize={state.fontSize}
          headerSize={state.headerSize}
          suggestedMessages={state.suggestedMessages.map((msg) => ({ id: msg.id, label: msg.label }))}
        />
      </div>
    </div>
  )
}
