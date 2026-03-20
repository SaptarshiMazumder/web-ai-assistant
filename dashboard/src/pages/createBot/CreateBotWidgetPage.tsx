import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Check, CheckCircle2, Globe, MessageCircle } from 'lucide-react'
import { SectionHeader, UiButton } from '../../components/ui'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'
import { WidgetDesignForm, stateToWidgetConfig, type WidgetDesignState } from '../../components/WidgetDesignForm'
import { LineDesignForm } from '../../components/LineDesignForm'
import { LineIcon } from '../../assets/icons/LineIcon'

type DesignView = 'hub' | 'website' | 'line'
type SavingScope = 'website' | 'line' | 'all' | null

export default function CreateBotWidgetPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { saveWidgetConfig, fetchPlatformConfig, getLineDesign, saveLineDesign } = useDashboardData()
  const { step1, step2, step3, step4, flow } = useCreateBotFlow()
  const [designView, setDesignView] = useState<DesignView>('hub')
  const [savingScope, setSavingScope] = useState<SavingScope>(null)
  const [websiteConfigured, setWebsiteConfigured] = useState(false)
  const [lineConfigured, setLineConfigured] = useState(false)
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
      const overrides =
        result?.overrides && typeof result.overrides === 'object' && !Array.isArray(result.overrides)
          ? result.overrides
          : {}
      setLineDesignOverrides(overrides)
      setLineConfigured(Object.keys(overrides).length > 0)
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

  const persistWebsiteDesign = useCallback(async () => {
    if (!botId) return
    await saveWidgetConfig(botId, {
      ...stateToWidgetConfig(value),
      businessType: step1.businessType || undefined,
      contentHosting: contentHosting || undefined,
    })
  }, [botId, contentHosting, saveWidgetConfig, step1.businessType, value])

  const persistLineDesign = useCallback(async () => {
    if (!botId) return
    await saveLineDesign(botId, lineDesignOverrides || {})
  }, [botId, lineDesignOverrides, saveLineDesign])

  const handleWebsiteDone = async () => {
    if (!botId || savingScope) return
    setSavingScope('website')
    try {
      await persistWebsiteDesign()
      setWebsiteConfigured(true)
      setDesignView('hub')
    } catch {
      // Keep user on the current flow so they can retry.
    } finally {
      setSavingScope(null)
    }
  }

  const handleLineDone = async () => {
    if (!botId || savingScope) return
    setSavingScope('line')
    try {
      await persistLineDesign()
      setLineConfigured(true)
      setDesignView('hub')
    } catch {
      // Keep user on the current flow so they can retry.
    } finally {
      setSavingScope(null)
    }
  }

  const handleContinue = async () => {
    if (!botId || !flow.nextPath || savingScope) return
    setSavingScope('all')
    try {
      await Promise.all([persistWebsiteDesign(), persistLineDesign()])
      navigate(flow.nextPath)
    } catch {
      setSavingScope(null)
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
          <span>{t('createBot.agentGettingReadyDesignWhileWait', 'Your agent is getting ready. Design Website and LINE appearance while you wait.')}</span>
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

  const suggestedPreviewMessages = useMemo(
    () => step4.suggestedMessages,
    [step4.suggestedMessages]
  )

  const continueActions = (
    <>
      <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
        {t('common.back', 'Back')}
      </UiButton>
      <UiButton variant="primary" onClick={() => void handleContinue()} disabled={Boolean(savingScope)}>
        {savingScope === 'all' ? t('botDesign.saving', 'Saving...') : t('common.continue', 'Continue')}
      </UiButton>
    </>
  )

  const websiteActions = (
    <>
      <UiButton variant="secondary" onClick={() => setDesignView('hub')} disabled={Boolean(savingScope)}>
        {t('common.back', 'Back')}
      </UiButton>
      <UiButton variant="primary" onClick={() => void handleWebsiteDone()} disabled={Boolean(savingScope)}>
        {savingScope === 'website' ? t('botDesign.saving', 'Saving...') : t('createBot.doneForWebsiteAppearance', 'Done for Website')}
      </UiButton>
    </>
  )

  const lineActions = (
    <>
      <UiButton variant="secondary" onClick={() => setDesignView('hub')} disabled={Boolean(savingScope)}>
        {t('common.back', 'Back')}
      </UiButton>
      <UiButton variant="primary" onClick={() => void handleLineDone()} disabled={Boolean(savingScope)}>
        {savingScope === 'line' ? t('botDesign.saving', 'Saving...') : t('createBot.doneForLineAppearance', 'Done for LINE')}
      </UiButton>
    </>
  )

  if (designView === 'website') {
    return (
      <>
        <SectionHeader
          title={t('botDesign.title', 'Design your agent appearance')}
          subtitle={t('botDesign.subtitle', 'Customize how your agent looks on Website and LINE. Changes update the preview on the right.')}
        />
        <div style={{ marginBottom: '1rem' }}>{banner}</div>
        <WidgetDesignForm
          value={value}
          onChange={onChange}
          actions={websiteActions}
          welcomeDefaultsByLanguage={step4.welcomeDefaultsByLanguage}
          leftAligned
        />
      </>
    )
  }

  if (designView === 'line') {
    return (
      <>
        <SectionHeader
          title={t('botDesign.title', 'Design your agent appearance')}
          subtitle={t('botDesign.subtitle', 'Customize how your agent looks on Website and LINE. Changes update the preview on the right.')}
        />
        <div style={{ marginBottom: '1rem' }}>{banner}</div>
        <LineDesignForm
          profile={lineDesignProfile}
          value={lineDesignOverrides}
          onChange={setLineDesignOverrides}
          suggestedPreviewMessages={suggestedPreviewMessages}
          actions={lineActions}
          botName={botName || step4.widgetTitle || 'Bot'}
          leftAligned
        />
      </>
    )
  }

  return (
    <div className="flow-panel-body">
      <SectionHeader
        title={t('botDesign.title', 'Design your agent appearance')}
        subtitle={t('botDesign.subtitle', 'Customize how your agent looks on Website and LINE. Changes update the preview on the right.')}
      />
      <div style={{ marginBottom: '1rem' }}>{banner}</div>

      <div>
        <div className="card-title">{t('createBot.appearanceChannelsTitle', 'Design your AI agent appearance')}</div>
        <div className="card-subtitle">
          {t('createBot.appearanceChannelsSubtitle', 'Choose a platform card to open appearance setup.')}
        </div>
      </div>

      <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
        <button
          type="button"
          onClick={() => setDesignView('website')}
          style={{
            textAlign: 'left',
            border: websiteConfigured ? '2px solid #22c55e' : '1px solid var(--flow-border, #f2d8d2)',
            borderRadius: 16,
            background: websiteConfigured ? 'rgba(34,197,94,0.08)' : 'var(--flow-surface, #fff)',
            padding: '1rem',
            cursor: 'pointer',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.7rem' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: 'var(--flow-text)' }}>
              <Globe size={18} />
              {t('createBot.websiteCardTitle', 'Website')}
            </div>
            {websiteConfigured ? <CheckCircle2 size={18} color="#22c55e" /> : null}
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--flow-muted)', lineHeight: 1.5 }}>
            {websiteConfigured
              ? t('createBot.websiteAppearanceConfigured', 'Appearance configured')
              : t('createBot.websiteAppearanceNotConfigured', 'Not configured')}
          </div>
        </button>

        <button
          type="button"
          onClick={() => setDesignView('line')}
          style={{
            textAlign: 'left',
            border: lineConfigured ? '2px solid #22c55e' : '1px solid var(--flow-border, #f2d8d2)',
            borderRadius: 16,
            background: lineConfigured ? 'rgba(34,197,94,0.08)' : 'var(--flow-surface, #fff)',
            padding: '1rem',
            cursor: 'pointer',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.7rem' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: 'var(--flow-text)' }}>
              <LineIcon size={18} />
              {t('createBot.lineCardTitle', 'LINE')}
            </div>
            {lineConfigured ? <CheckCircle2 size={18} color="#22c55e" /> : null}
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--flow-muted)', lineHeight: 1.5 }}>
            {lineConfigured
              ? t('createBot.lineAppearanceConfigured', 'Appearance configured')
              : t('createBot.lineAppearanceNotConfigured', 'Not configured')}
          </div>
        </button>
      </div>

      <div style={{ border: '1px dashed var(--flow-border, #f2d8d2)', borderRadius: 12, padding: '0.9rem 1rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', color: 'var(--flow-muted)', fontSize: '0.9rem' }}>
        <MessageCircle size={16} />
        {t('createBot.appearanceOptionalHint', 'Skip this now and set up later.')}
      </div>

      <div className="flow-actions">{continueActions}</div>
    </div>
  )
}
