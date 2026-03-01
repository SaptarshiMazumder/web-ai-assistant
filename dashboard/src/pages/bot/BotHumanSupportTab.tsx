import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { BellRing, Check, MessageSquare, Save } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  stateToWidgetConfig,
  type SuggestedMessageConfig,
  widgetConfigToState,
} from '../../components/WidgetDesignForm'
import { useDashboardData, type EscalationConfig } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, GlassField, SectionHeader, UiButton } from '../../components/ui'

const SAVED_FEEDBACK_MS = 2000

const DEFAULT_ESCALATION_CONFIG: EscalationConfig = {
  enabled: false,
  notify_enabled: false,
  notification_emails: '',
}

export default function BotHumanSupportTab() {
  const { t } = useTranslation()
  const { botId } = useParams()
  const {
    selectedBot,
    selectedBotWidgetConfig,
    saveWidgetConfig,
    getEscalationConfig,
    saveEscalationConfig,
    loading,
  } = useDashboardData()

  const defaultEscalationBtnLabel = t('botSuggestedMessages.defaultEscalationBtnLabel', 'Request human support')

  const [suggestedMessages, setSuggestedMessages] = useState<SuggestedMessageConfig[]>([])
  const [escalationConfig, setEscalationConfig] = useState<EscalationConfig>(DEFAULT_ESCALATION_CONFIG)
  const [escalationBtnLabel, setEscalationBtnLabel] = useState(defaultEscalationBtnLabel)
  const [savingEscalation, setSavingEscalation] = useState(false)
  const [escalationSavedJustNow, setEscalationSavedJustNow] = useState(false)
  useEffect(() => {
    const next = widgetConfigToState(selectedBotWidgetConfig ?? null)
    setSuggestedMessages(next.suggestedMessages)
    const existingEscalateMsg = next.suggestedMessages.find((m) => m.type === 'escalate')
    setEscalationBtnLabel(existingEscalateMsg?.label || defaultEscalationBtnLabel)
  }, [selectedBotWidgetConfig, defaultEscalationBtnLabel])

  useEffect(() => {
    if (!botId) return
    void getEscalationConfig(botId).then((data) => {
      if (data) setEscalationConfig(data)
    })
  }, [botId, getEscalationConfig])

  const saveMessages = async (next: SuggestedMessageConfig[]) => {
    if (!botId) return
    setSuggestedMessages(next)
    const currentState = widgetConfigToState(selectedBotWidgetConfig ?? null)
    currentState.suggestedMessages = next
    await saveWidgetConfig(botId, stateToWidgetConfig(currentState))
  }

  const handleSave = async () => {
    if (!botId || savingEscalation || escalationSavedJustNow) return
    setSavingEscalation(true)
    try {
      const toSave = { ...escalationConfig, enabled: true }
      const saved = await saveEscalationConfig(botId, toSave)
      if (saved) setEscalationConfig(saved)

      const updatedMessages = suggestedMessages.map((m) =>
        m.type === 'escalate' ? { ...m, label: escalationBtnLabel } : m
      )
      await saveMessages(updatedMessages)

      setEscalationSavedJustNow(true)
      setTimeout(() => setEscalationSavedJustNow(false), SAVED_FEEDBACK_MS)
    } finally {
      setSavingEscalation(false)
    }
  }

  if (!botId) {
    return <div className="empty-panel">{t('botHumanSupport.selectBot', 'Select a bot to configure human support.')}</div>
  }
  if (loading && !selectedBot) return <div className="empty-panel">{t('common.working', 'Working...')}</div>
  if (selectedBot?.bot_id !== botId) return <div className="empty-panel">{t('common.working', 'Working...')}</div>

  return (
    <AnimatedPage>
      <SectionHeader
        title={t('botHumanSupport.title', 'Human support')}
        subtitle={t('botHumanSupport.subtitle', 'Let visitors request human support and notify your team by email.')}
      />

      <GlassCard style={{ display: 'grid', gap: '1.25rem', marginTop: '0.5rem' }}>
        <div style={{ padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.75rem' }}>
                <MessageSquare size={18} style={{ color: 'var(--ui-flow-accent)' }} />
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>{t('botSuggestedMessages.humanSupportButtonTitle', 'Human support button')}</div>
                  <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                    {t('botSuggestedMessages.humanSupportButtonSubtitle', 'A button shown in the chat that lets visitors request human support.')}
                  </p>
                </div>
              </div>
              <GlassField label={t('botSuggestedMessages.buttonLabel', 'Button label')} style={{ marginTop: '1rem' }}>
                <input
                  type="text"
                  placeholder={t('botSuggestedMessages.defaultEscalationBtnLabel', 'Request human support')}
                  value={escalationBtnLabel}
                  onChange={(e) => setEscalationBtnLabel(e.target.value)}
                />
              </GlassField>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <BellRing size={18} style={{ color: 'var(--ui-flow-accent)' }} />
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>{t('botSuggestedMessages.emailNotificationsTitle', 'Email notifications')}</div>
                  <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                    {t('botSuggestedMessages.emailNotificationsSubtitle', 'Receive notifications when visitors request human support.')}
                  </p>
                </div>
              </div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={escalationConfig.notify_enabled}
                  onChange={(e) =>
                    setEscalationConfig((prev) => ({ ...prev, notify_enabled: e.target.checked }))
                  }
                />
                <span className="toggle-slider" />
              </label>
            </div>

            <GlassField
              label={t('botSuggestedMessages.notificationEmailsLabel', 'Notification email(s)')}
              helper={t('botSuggestedMessages.notificationEmailsHelper', 'Separate multiple emails with semicolons (;).')}
            >
              <input
                type="text"
                placeholder={t('botSuggestedMessages.notificationEmailsPlaceholder', 'team@company.com; support@company.com')}
                value={escalationConfig.notification_emails}
                onChange={(e) =>
                  setEscalationConfig((prev) => ({ ...prev, notification_emails: e.target.value }))
                }
              />
            </GlassField>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <UiButton
            variant="primary"
            onClick={() => void handleSave()}
            disabled={savingEscalation || escalationSavedJustNow}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
          >
            {savingEscalation ? (
              t('botSuggestedMessages.saving', 'Saving...')
            ) : escalationSavedJustNow ? (
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
      </GlassCard>
    </AnimatedPage>
  )
}
