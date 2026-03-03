import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check, Save } from 'lucide-react'
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
    fetchPlatformConfig,
    loading,
  } = useDashboardData()
  const [state, setState] = useState<WidgetDesignState>(() => DEFAULT_WIDGET_DESIGN_STATE)
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)
  const [availableTypes, setAvailableTypes] = useState<Array<'ai_response' | 'show_menu' | 'escalate'>>(['ai_response'])

  useEffect(() => {
    setState(widgetConfigToState(selectedBotWidgetConfig ?? null))
  }, [selectedBotWidgetConfig])

  useEffect(() => {
    fetchPlatformConfig().then((r) => {
      const platformId = String((selectedBotWidgetConfig as Record<string, unknown>)?.reservationPlatform ?? '').toLowerCase()
      if (platformId) {
        const platform = r.platforms.find((p) => p.id.toLowerCase() === platformId)
        const types = (platform?.availableSuggestedMessageTypes ?? r.defaultAvailableSuggestedMessageTypes) as Array<'ai_response' | 'show_menu' | 'escalate'>
        setAvailableTypes(types.length ? types : ['ai_response'])
      } else {
        setAvailableTypes(r.defaultAvailableSuggestedMessageTypes as Array<'ai_response' | 'show_menu' | 'escalate'>)
      }
    })
  }, [fetchPlatformConfig, selectedBotWidgetConfig])

  const update = useCallback(<K extends keyof WidgetDesignState>(key: K, value: WidgetDesignState[K]) => {
    setState((prev) => ({ ...prev, [key]: value }))
  }, [])

  const handleSuggestedMessagesChange = (next: SuggestedMessageConfig[]) => {
    update('suggestedMessages', next)
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

  if (!botId) return <div className="empty-panel">{t('botSuggestedMessages.selectBot', 'Select a bot.')}</div>
  if (loading && !selectedBot) return <div className="empty-panel">{t('common.working', 'Working...')}</div>
  if (selectedBot?.bot_id !== botId) return <div className="empty-panel">{t('common.working', 'Working...')}</div>

  const saveAction = (
    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
      <UiButton
        variant="primary"
        onClick={() => void handleSave()}
        disabled={saving || savedJustNow}
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
        subtitle={t('botSuggestedMessages.subtitle', 'Quick actions shown when the chat opens. Configured per platform in config YAML (e.g. Tabelog: Menu, Human support).')}
      />

      <GlassCard style={{ marginTop: '1rem' }}>
        <SuggestedMessagesEditor
          suggestedMessages={state.suggestedMessages}
          onChange={handleSuggestedMessagesChange}
          title=""
          subtitle=""
          addButtonPlacement="bottom"
          maxItems={10}
          actions={saveAction}
          availableTypes={availableTypes}
          botId={botId}
        />
      </GlassCard>
    </AnimatedPage>
  )
}

