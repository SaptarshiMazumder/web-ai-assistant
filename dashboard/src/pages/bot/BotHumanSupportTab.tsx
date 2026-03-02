import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { BellRing, Check, Globe, MessageSquare, Save } from 'lucide-react'
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
  notify_website: false,
  notify_instagram: false,
  notify_line: false,
  notification_emails: '',
}

type ChannelTab = 'website' | 'instagram' | 'line'

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
  const [activeChannelTab, setActiveChannelTab] = useState<ChannelTab>('website')
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
      if (data) {
        setEscalationConfig({
          ...DEFAULT_ESCALATION_CONFIG,
          ...data,
          notify_website: data.notify_website ?? data.notify_enabled ?? false,
          notify_instagram: data.notify_instagram ?? data.notify_enabled ?? false,
          notify_line: data.notify_line ?? data.notify_enabled ?? false,
        })
      }
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

            <div style={{ padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,241,239,0.3)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.75rem' }}>
                <BellRing size={18} style={{ color: 'var(--ui-flow-accent)' }} />
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>{t('botSuggestedMessages.emailNotificationsTitle', 'Email notifications')}</div>
                  <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                    {t('botSuggestedMessages.emailNotificationsSubtitle', 'Receive notifications when visitors request human support.')}
                  </p>
                </div>
              </div>
              <GlassField
                label={t('botSuggestedMessages.notificationEmailsLabel', 'Notification email(s)')}
                helper={t('botSuggestedMessages.notificationEmailsHelper', 'Separate multiple emails with semicolons (;).')}
                style={{ marginBottom: '1rem' }}
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
              <div style={{ marginBottom: '0.5rem', fontSize: '0.9rem', fontWeight: 500, color: 'var(--ui-flow-text)' }}>
                {t('botHumanSupport.notifyByChannel', 'Notify by channel')}
              </div>
              <div style={{ display: 'flex', gap: '0.25rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                {(['website', 'instagram', 'line'] as const).map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setActiveChannelTab(tab)}
                    style={{
                      padding: '0.5rem 1rem',
                      borderRadius: 8,
                      border: `1px solid ${activeChannelTab === tab ? 'var(--ui-flow-accent)' : 'var(--ui-flow-border)'}`,
                      background: activeChannelTab === tab ? 'rgba(var(--ui-flow-accent-rgb, 234, 88, 12), 0.15)' : 'transparent',
                      color: 'var(--ui-flow-text)',
                      cursor: 'pointer',
                      fontSize: '0.9rem',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                    }}
                  >
                    {tab === 'website' && <Globe size={16} />}
                    {tab === 'instagram' && (
                      <svg width={16} height={16} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                        <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />
                      </svg>
                    )}
                    {tab === 'line' && (
                      <svg width={16} height={16} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                        <path d="M19.365 9.863c.349 0 .63.285.63.631 0 .345-.281.63-.63.63H17.61v1.125h1.755c.349 0 .63.283.63.63 0 .344-.281.629-.63.629h-2.386c-.345 0-.627-.285-.627-.629V8.108c0-.345.282-.63.63-.63h2.386c.346 0 .627.285.627.63 0 .349-.281.63-.63.63H17.61v1.125h1.755zm-3.855 3.016c0 .27-.174.51-.432.596-.064.021-.133.031-.199.031-.211 0-.391-.09-.51-.25l-2.443-3.317v2.94c0 .344-.279.629-.631.629-.346 0-.626-.285-.626-.629V8.108c0-.27.173-.51.43-.595.06-.023.136-.033.194-.033.195 0 .375.104.495.254l2.462 3.33V8.108c0-.345.282-.63.63-.63.345 0 .63.285.63.63v4.771zm-5.741 0c0 .344-.282.629-.631.629-.345 0-.627-.285-.627-.629V8.108c0-.345.282-.63.63-.63.346 0 .628.285.628.63v4.771zm-2.466.629H4.917c-.345 0-.63-.285-.63-.629V8.108c0-.345.285-.63.63-.63.348 0 .63.285.63.63v4.141h1.756c.348 0 .629.283.629.63 0 .344-.282.629-.629.629M24 10.314C24 4.943 18.615.572 12 .572S0 4.943 0 10.314c0 4.811 4.27 8.842 10.035 9.608.391.082.923.258 1.058.59.12.301.079.766.038 1.08l-.164 1.02c-.045.301-.24 1.186 1.049.645 1.291-.539 6.916-4.078 9.436-6.975C23.176 14.393 24 12.458 24 10.314" />
                      </svg>
                    )}
                    {tab === 'website' && t('botHumanSupport.channelWebsite', 'Website')}
                    {tab === 'instagram' && t('botHumanSupport.channelInstagram', 'Instagram')}
                    {tab === 'line' && t('botHumanSupport.channelLine', 'LINE')}
                  </button>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.75rem', borderRadius: 10, background: 'rgba(0,0,0,0.03)' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--ui-flow-text)' }}>
                  {activeChannelTab === 'website' && t('botHumanSupport.notifyWebsiteDesc', 'Email when visitors escalate from the website chat widget')}
                  {activeChannelTab === 'instagram' && t('botHumanSupport.notifyInstagramDesc', 'Email when visitors escalate via Instagram DM')}
                  {activeChannelTab === 'line' && t('botHumanSupport.notifyLineDesc', 'Email when visitors escalate via LINE')}
                </span>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={
                      activeChannelTab === 'website' ? escalationConfig.notify_website :
                      activeChannelTab === 'instagram' ? escalationConfig.notify_instagram :
                      escalationConfig.notify_line
                    }
                    onChange={(e) => {
                      const key = activeChannelTab === 'website' ? 'notify_website' : activeChannelTab === 'instagram' ? 'notify_instagram' : 'notify_line'
                      setEscalationConfig((prev) => ({ ...prev, [key]: e.target.checked, notify_enabled: true }))
                    }}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>
            </div>

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
