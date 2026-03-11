import { useCallback, useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check, Globe, MessageCircle, Paintbrush } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  WidgetDesignForm,
  widgetConfigToState,
  stateToWidgetConfig,
  getDefaultsForLanguage,
  type WidgetDesignState,
} from '../../components/WidgetDesignForm'
import { LineDesignForm } from '../../components/LineDesignForm'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, SectionHeader, SegmentedTabs, UiButton } from '../../components/ui'

const SAVED_FEEDBACK_MS = 2000
type DesignTab = 'web' | 'line'

export default function BotDesignTab() {
  const { t, i18n } = useTranslation()
  const { botId } = useParams()
  const {
    selectedBot,
    selectedBotWidgetConfig,
    saveWidgetConfig,
    loading,
    fetchPlatformConfig,
    getLineDesign,
    saveLineDesign,
  } = useDashboardData()
  const [activeTab, setActiveTab] = useState<DesignTab>('web')
  const [state, setState] = useState<WidgetDesignState>(() => DEFAULT_WIDGET_DESIGN_STATE)
  const [lineDesignOverrides, setLineDesignOverrides] = useState<Record<string, unknown> | null>(null)
  const [lineDesignProfile, setLineDesignProfile] = useState<Record<string, unknown> | null>(null)
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)

  useEffect(() => {
    const parsed = widgetConfigToState(selectedBotWidgetConfig ?? null)
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

  useEffect(() => {
    if (!botId) return
    let mounted = true
    void getLineDesign(botId).then((result) => {
      if (!mounted) return
      setLineDesignOverrides(result?.overrides || {})
    })
    return () => {
      mounted = false
    }
  }, [botId, getLineDesign])

  useEffect(() => {
    let mounted = true
    const lang = i18n.language?.startsWith('ja') ? 'ja' : 'en'
    void fetchPlatformConfig(lang).then((result) => {
      if (!mounted) return
      const profile =
        result.lineDesignProfile &&
        typeof result.lineDesignProfile === 'object' &&
        !Array.isArray(result.lineDesignProfile)
          ? result.lineDesignProfile
          : {}
      setLineDesignProfile(profile)
    })
    return () => {
      mounted = false
    }
  }, [fetchPlatformConfig, i18n.language])

  const update = useCallback(<K extends keyof WidgetDesignState>(key: K, value: WidgetDesignState[K]) => {
    setState((prev) => ({ ...prev, [key]: value }))
  }, [])

  const tabs = useMemo(
    () => [
      { id: 'web' as const, label: t('botDesign.webTab', 'Website'), icon: <Globe size={15} /> },
      { id: 'line' as const, label: t('botDesign.lineTab', 'LINE'), icon: <MessageCircle size={15} /> },
    ],
    [t]
  )
  const suggestedPreviewMessages = useMemo(
    () => state.suggestedMessages,
    [state.suggestedMessages]
  )

  const handleSave = async () => {
    if (!botId || saving || savedJustNow) return
    setSaving(true)
    setSavedJustNow(false)
    try {
      if (activeTab === 'line') {
        await saveLineDesign(botId, lineDesignOverrides || {})
      } else {
        await saveWidgetConfig(botId, stateToWidgetConfig(state, { includeWelcomeMessage: false }))
      }
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

  const saveButton = (
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
  )

  return (
    <AnimatedPage>
      <SectionHeader
        title={t('botDesign.title', 'Design the chat widget')}
        subtitle={t('botDesign.subtitle', 'Customize how the widget appears. Changes update the preview on the right.')}
      />

      <div style={{ marginBottom: '1rem' }}>
        <SegmentedTabs value={activeTab} onChange={setActiveTab} options={tabs} ariaLabel="Design tabs" />
      </div>

      {activeTab === 'line' ? (
        <LineDesignForm
          profile={lineDesignProfile}
          value={lineDesignOverrides}
          onChange={setLineDesignOverrides}
          suggestedPreviewMessages={suggestedPreviewMessages}
          actions={saveButton}
          botName={selectedBot?.display_name || state.widgetTitle || 'Bot'}
        />
      ) : (
        <WidgetDesignForm
          value={state}
          onChange={update}
          showWelcomeMessage={false}
          actions={saveButton}
        />
      )}
    </AnimatedPage>
  )
}
