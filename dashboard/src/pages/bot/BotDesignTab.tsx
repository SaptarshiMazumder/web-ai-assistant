import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check, Paintbrush } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  WidgetDesignForm,
  widgetConfigToState,
  stateToWidgetConfig,
  getDefaultsForLanguage,
  type WidgetDesignState,
} from '../../components/WidgetDesignForm'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, SectionHeader, UiButton } from '../../components/ui'

const SAVED_FEEDBACK_MS = 2000

export default function BotDesignTab() {
  const { t, i18n } = useTranslation()
  const { botId } = useParams()
  const { selectedBot, selectedBotWidgetConfig, saveWidgetConfig, loading } = useDashboardData()
  const [state, setState] = useState<WidgetDesignState>(() => DEFAULT_WIDGET_DESIGN_STATE)
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)

  useEffect(() => {
    const parsed = widgetConfigToState(selectedBotWidgetConfig ?? null)
    // If bot has no explicit language set, inherit from dashboard UI language
    if (!selectedBotWidgetConfig?.language) {
      const appLang: 'en' | 'ja' = i18n.language?.startsWith('ja') ? 'ja' : 'en'
      if (appLang !== parsed.botLanguage) {
        const defaults = getDefaultsForLanguage(appLang)
        Object.assign(parsed, defaults)
      }
    }
    setState(parsed)
  }, [selectedBotWidgetConfig, i18n.language])

  useEffect(() => {
    if (selectedBot?.display_name && (state.widgetTitle === 'Chat' || state.widgetTitle === 'チャット')) {
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
    return <div className="empty-panel">{t('botDesign.selectBot', 'Select a bot to edit design.')}</div>
  }

  if (loading && !selectedBot) {
    return <div className="empty-panel">{t('common.working', 'Working...')}</div>
  }

  if (selectedBot?.bot_id !== botId) {
    return <div className="empty-panel">{t('common.working', 'Working...')}</div>
  }

  return (
    <AnimatedPage>
      <SectionHeader
        title={t('botDesign.title', 'Design the chat widget')}
        subtitle={t('botDesign.subtitle', 'Customize how the widget appears. Changes update the preview on the right.')}
      />

      <WidgetDesignForm
        value={state}
        onChange={update}
        actions={
          <UiButton
            variant="primary"
            onClick={() => void handleSave()}
            disabled={saving || savedJustNow}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
          >
            {saving ? (
              t('botDesign.saving', 'Saving...')
            ) : savedJustNow ? (
              <>
                <Check size={18} strokeWidth={2.5} aria-hidden />
                <span>{t('botDesign.saved', 'Saved')}</span>
              </>
            ) : (
              <>
                <Paintbrush size={16} />
                {t('botDesign.saveDesign', 'Save design')}
              </>
            )}
          </UiButton>
        }
      />
    </AnimatedPage>
  )
}
