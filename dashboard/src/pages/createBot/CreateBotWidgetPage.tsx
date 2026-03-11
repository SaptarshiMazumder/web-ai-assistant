import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Check, Globe, MessageCircle } from 'lucide-react'
import { SectionHeader, SegmentedTabs, UiButton } from '../../components/ui'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'
import { WidgetDesignForm, stateToWidgetConfig, type WidgetDesignState } from '../../components/WidgetDesignForm'
import { LineDesignForm } from '../../components/LineDesignForm'

type DesignTab = 'web' | 'line'

export default function CreateBotWidgetPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { saveWidgetConfig, fetchPlatformConfig, getLineDesign, saveLineDesign } = useDashboardData()
  const { step1, step2, step3, step4, flow } = useCreateBotFlow()
  const [activeTab, setActiveTab] = useState<DesignTab>('web')
  const [saving, setSaving] = useState(false)
  const [lineDesignOverrides, setLineDesignOverrides] = useState<Record<string, unknown> | null>(null)
  const [lineDesignProfile, setLineDesignProfile] = useState<Record<string, unknown> | null>(null)
  const { botName } = step1
  const { botId, trainingStage, localError: trainingError } = step3
  const { contentHosting } = step2

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  useEffect(() => {
    if (botName && (step4.widgetTitle === 'Chat' || step4.widgetTitle === 'チャット')) {
      step4.setWidgetTitle(botName.trim())
    }
  }, [botName, step4.widgetTitle, step4.setWidgetTitle])

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
    void fetchPlatformConfig(step4.botLanguage).then((result) => {
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
  }, [fetchPlatformConfig, step4.botLanguage])

  const value: WidgetDesignState = {
    botLanguage: step4.botLanguage,
    widgetPosition: step4.widgetPosition,
    widgetPrimaryColor: step4.widgetPrimaryColor,
    widgetTitle: step4.widgetTitle,
    widgetSize: step4.widgetSize,
    welcomeMessage: step4.welcomeMessage,
    placeholder: step4.placeholder,
    footerMessage: step4.footerMessage,
    theme: step4.theme,
    textColor: step4.textColor,
    launcherIconUrl: step4.launcherIconUrl,
    launcherText: step4.launcherText,
    headerIconUrl: step4.headerIconUrl,
    shareIconUrl: step4.shareIconUrl,
    maxHeight: step4.maxHeight,
    fontSize: step4.fontSize,
    headerSize: step4.headerSize,
    autoPopupWelcome: step4.autoPopupWelcome,
    autoScrollNewMessages: step4.autoScrollNewMessages,
    displaySourcesInMessages: step4.displaySourcesInMessages,
    sourcesLabel: step4.sourcesLabel,
    suggestedMessages: step4.suggestedMessages,
  }

  const onChange = useCallback(<K extends keyof WidgetDesignState>(key: K, val: WidgetDesignState[K]) => {
    const setterKey = ('set' + key.charAt(0).toUpperCase() + key.slice(1)) as keyof typeof step4
    const setter = step4[setterKey]
    if (typeof setter === 'function') (setter as (v: WidgetDesignState[K]) => void)(val)
  }, [step4])

  const handleContinue = async () => {
    if (!botId || !flow.nextPath || saving) return
    setSaving(true)
    try {
      await saveWidgetConfig(botId, {
        ...stateToWidgetConfig(value),
        businessType: step1.businessType || undefined,
        contentHosting: contentHosting || undefined,
      })
      await saveLineDesign(botId, lineDesignOverrides || {})
      navigate(flow.nextPath)
    } catch {
      setSaving(false)
    }
  }

  const banner = (
    <>
      {trainingStage === 'training' && (
        <div className="design-form-training-in-progress" style={{ marginBottom: 0, color: step4.widgetPrimaryColor }}>
          <span className="discovery-loading-dots" aria-hidden>
            <span />
            <span />
            <span />
          </span>
          <span>{t('createBot.agentGettingReadyDesignWhileWait', 'Your agent is getting ready. Design the chat while you wait.')}</span>
        </div>
      )}
      {trainingStage === 'complete' && !trainingError && (
        <div
          className="design-form-training-done"
          style={{ color: step4.widgetPrimaryColor, marginBottom: 0 }}
        >
          <Check size={20} strokeWidth={2.5} aria-hidden />
          <span>{t('createBot.agentLearnedFromContent', 'Your agent has learned from your content.')}</span>
        </div>
      )}
      {trainingError && (
        <div className="alert error" style={{ marginBottom: 0 }}>
          {trainingError}
        </div>
      )}
    </>
  )

  const tabs = useMemo(
    () => [
      { id: 'web' as const, label: t('botDesign.webTab', 'Website'), icon: <Globe size={15} /> },
      { id: 'line' as const, label: t('botDesign.lineTab', 'LINE'), icon: <MessageCircle size={15} /> },
    ],
    [t]
  )
  const suggestedPreviewMessages = useMemo(
    () => step4.suggestedMessages,
    [step4.suggestedMessages]
  )

  const actions = (
    <>
      <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
        {t('common.back', 'Back')}
      </UiButton>
      <UiButton variant="primary" onClick={() => void handleContinue()} disabled={saving}>
        {saving ? t('botDesign.saving', 'Saving...') : t('common.continue', 'Continue')}
      </UiButton>
    </>
  )

  return (
    <>
      <SectionHeader
        title={t('botDesign.title', 'Design the chat widget')}
        subtitle={t('botDesign.subtitle', 'Customize how the widget appears. Changes update the preview on the right.')}
      />
      <div style={{ marginBottom: '1rem' }}>{banner}</div>
      <div style={{ marginBottom: '1rem' }}>
        <SegmentedTabs value={activeTab} onChange={setActiveTab} options={tabs} ariaLabel="Design tabs" />
      </div>
      {activeTab === 'line' ? (
        <LineDesignForm
          profile={lineDesignProfile}
          value={lineDesignOverrides}
          onChange={setLineDesignOverrides}
          suggestedPreviewMessages={suggestedPreviewMessages}
          actions={actions}
          botName={botName || step4.widgetTitle || 'Bot'}
        />
      ) : (
        <WidgetDesignForm
          value={value}
          onChange={onChange}
          actions={actions}
          welcomeDefaultsByLanguage={step4.welcomeDefaultsByLanguage}
        />
      )}
    </>
  )
}
