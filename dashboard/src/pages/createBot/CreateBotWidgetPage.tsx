import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Check } from 'lucide-react'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'
import { WidgetDesignForm, stateToWidgetConfig, type WidgetDesignState } from '../../components/WidgetDesignForm'

export default function CreateBotWidgetPage() {
  const navigate = useNavigate()
  const { saveWidgetConfig } = useDashboardData()
  const { step1, step2, step3, step4, flow } = useCreateBotFlow()
  const [saving, setSaving] = useState(false)
  const { botName, businessType: step1BusinessType } = step1
  const { botId, trainingStage, localError: trainingError } = step3
  const { contentHosting } = step2

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  useEffect(() => {
    if (botName && step4.widgetTitle === 'Chat') {
      step4.setWidgetTitle(botName.trim())
    }
  }, [botName, step4.widgetTitle, step4.setWidgetTitle])

  useEffect(() => {
    if (step1BusinessType && !step4.businessType) {
      step4.setBusinessType(step1BusinessType)
    }
  }, [step1BusinessType, step4.businessType, step4.setBusinessType])

  const value: WidgetDesignState = {
    widgetPosition: step4.widgetPosition,
    widgetPrimaryColor: step4.widgetPrimaryColor,
    businessType: step4.businessType,
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
        businessType: step1.businessType || step4.businessType || undefined,
        contentHosting: contentHosting || undefined,
      })
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
          <span>Your agent is getting ready. Design the chat while you wait.</span>
        </div>
      )}
      {trainingStage === 'complete' && !trainingError && (
        <div
          className="design-form-training-done"
          style={{ color: step4.widgetPrimaryColor, marginBottom: 0 }}
        >
          <Check size={20} strokeWidth={2.5} aria-hidden />
          <span>Your agent has learned from your content.</span>
        </div>
      )}
      {trainingError && (
        <div className="alert error" style={{ marginBottom: 0 }}>
          {trainingError}
        </div>
      )}
    </>
  )

  const actions = (
    <>
      <button type="button" className="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
        Back
      </button>
      <button type="button" className="primary" onClick={() => void handleContinue()} disabled={saving}>
        {saving ? 'Saving...' : 'Continue'}
      </button>
    </>
  )

  return (
    <WidgetDesignForm
      value={value}
      onChange={onChange}
      banner={banner}
      actions={actions}
    />
  )
}
