import { useEffect, useMemo, useRef, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { Check, Globe, Info, MessageSquare, Save, Sparkles } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { LineIcon } from '../../assets/icons/LineIcon'
import { SuggestedMessagesEditor } from '../../components/SuggestedMessagesEditor'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  type SuggestedMessageConfig,
} from '../../components/WidgetDesignForm'
import { useDashboardData } from '../../hooks/useDashboardData'
import {
  AnimatedPage,
  GlassCard,
  GlassField,
  SectionHeader,
  SegmentedTabs,
  type SegmentedTabOption,
  UiButton,
} from '../../components/ui'
import {
  getBotSuggestedMessagesBaseLanguage,
  getSuggestedMessagesForLanguageFromConfig,
  setSuggestedMessagesForLanguageInConfig,
  type SuggestedMessagesLanguage,
} from '../../utils/suggestedMessagesConfig'
import {
  getBotWelcomeMessagesBaseLanguage,
  getWelcomeMessageForChannelFromConfig,
  setWelcomeMessageForChannelInConfig,
} from '../../utils/welcomeMessagesConfig'

type MessagesTabId = 'welcome' | 'suggested'

const SAVED_FEEDBACK_MS = 2000

export default function BotMessagesTab() {
  const { t } = useTranslation()
  const { botId } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const {
    selectedBot,
    selectedBotWidgetConfig,
    saveWidgetConfig,
    fetchPlatformConfig,
    loading,
  } = useDashboardData()

  const [editorLanguage, setEditorLanguage] = useState<SuggestedMessagesLanguage>('en')
  const [webWelcomeMessage, setWebWelcomeMessage] = useState('')
  const [suggestedMessages, setSuggestedMessages] = useState<SuggestedMessageConfig[]>(
    () => DEFAULT_WIDGET_DESIGN_STATE.suggestedMessages
  )
  const [availableTypes, setAvailableTypes] = useState<Array<'ai_response' | 'show_menu' | 'escalate'>>(['ai_response'])
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)
  const initializedBotIdRef = useRef<string | null>(null)

  const activeTab: MessagesTabId = searchParams.get('tab') === 'suggested' ? 'suggested' : 'welcome'

  const messageTabs = useMemo<ReadonlyArray<SegmentedTabOption<MessagesTabId>>>(
    () => [
      {
        id: 'welcome',
        label: t('botMessages.welcomeTab', 'Welcome'),
        icon: <MessageSquare size={16} />,
      },
      {
        id: 'suggested',
        label: t('botMessages.suggestedTab', 'Suggested'),
        icon: <Sparkles size={16} />,
      },
    ],
    [t]
  )

  useEffect(() => {
    if (!botId || !selectedBotWidgetConfig) return
    if (selectedBot?.bot_id !== botId) return
    if (initializedBotIdRef.current === botId) return
    initializedBotIdRef.current = botId
    const baseLanguage = getBotSuggestedMessagesBaseLanguage(selectedBotWidgetConfig)
    setEditorLanguage(baseLanguage || getBotWelcomeMessagesBaseLanguage(selectedBotWidgetConfig))
  }, [selectedBotWidgetConfig, selectedBot?.bot_id, botId])

  useEffect(() => {
    setWebWelcomeMessage(getWelcomeMessageForChannelFromConfig(selectedBotWidgetConfig, 'web', editorLanguage))
  }, [selectedBotWidgetConfig, editorLanguage])

  useEffect(() => {
    setSuggestedMessages(getSuggestedMessagesForLanguageFromConfig(selectedBotWidgetConfig, editorLanguage))
  }, [selectedBotWidgetConfig, editorLanguage])

  useEffect(() => {
    let cancelled = false

    void fetchPlatformConfig().then((result) => {
      if (cancelled) return
      const platformId = String((selectedBotWidgetConfig as Record<string, unknown> | null)?.reservationPlatform ?? '').toLowerCase()
      if (platformId) {
        const platform = result.platforms.find((entry) => entry.id.toLowerCase() === platformId)
        const nextTypes = (platform?.availableSuggestedMessageTypes ?? result.defaultAvailableSuggestedMessageTypes) as Array<
          'ai_response' | 'show_menu' | 'escalate'
        >
        setAvailableTypes(nextTypes.length ? nextTypes : ['ai_response'])
        return
      }
      setAvailableTypes(result.defaultAvailableSuggestedMessageTypes as Array<'ai_response' | 'show_menu' | 'escalate'>)
    })

    return () => {
      cancelled = true
    }
  }, [fetchPlatformConfig, selectedBotWidgetConfig])

  const handleSave = async () => {
    if (!botId || saving || savedJustNow) return

    setSaving(true)
    setSavedJustNow(false)

    try {
      const withWelcome = setWelcomeMessageForChannelInConfig(
        selectedBotWidgetConfig ?? {},
        'web',
        editorLanguage,
        webWelcomeMessage
      )
      const nextConfig = setSuggestedMessagesForLanguageInConfig(
        withWelcome,
        editorLanguage,
        suggestedMessages
      )

      await saveWidgetConfig(botId, {
        welcomeMessagesByChannel: nextConfig.welcomeMessagesByChannel as Record<string, unknown>,
        welcomeMessage: nextConfig.welcomeMessage as string | undefined,
        suggestedMessagesByLanguage: nextConfig.suggestedMessagesByLanguage as Record<string, unknown>,
      })

      setSavedJustNow(true)
      setTimeout(() => setSavedJustNow(false), SAVED_FEEDBACK_MS)
    } finally {
      setSaving(false)
    }
  }

  const renderSaveAction = () => (
    <UiButton
      variant="primary"
      onClick={() => void handleSave()}
      disabled={saving || savedJustNow}
      style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
    >
      {saving ? (
        t('botMessages.saving', 'Saving...')
      ) : savedJustNow ? (
        <>
          <Check size={18} strokeWidth={2.5} aria-hidden />
          <span>{t('botMessages.saved', 'Saved')}</span>
        </>
      ) : (
        <>
          <Save size={16} />
          {t('botMessages.save', 'Save')}
        </>
      )}
    </UiButton>
  )

  if (!botId) {
    return <div className="empty-panel">{t('botMessages.selectBot', 'Select a bot to edit messages.')}</div>
  }
  if (loading && !selectedBot) return <div className="empty-panel">{t('common.working', 'Working...')}</div>
  if (selectedBot?.bot_id !== botId) return <div className="empty-panel">{t('common.working', 'Working...')}</div>

  return (
    <AnimatedPage>
      <SectionHeader
        title={t('botMessages.title', 'Messages')}
        subtitle={t(
          'botMessages.subtitle',
          'Manage the first greeting and the quick actions visitors see when the chat opens.'
        )}
      />

      <GlassCard style={{ display: 'grid', gap: '1.25rem', marginTop: '0.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <SegmentedTabs
            value={activeTab}
            onChange={(nextTab) => setSearchParams({ tab: nextTab }, { replace: true })}
            options={messageTabs}
            ariaLabel={t('botMessages.tabAria', 'Message editor tabs')}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--ui-flow-text)' }}>
              {t('widgetDesign.botLanguage', 'Bot language')}
            </div>
            <div style={{ display: 'inline-flex', gap: '0.35rem', flexWrap: 'wrap' }}>
              {([
                { id: 'en', label: 'English' },
                { id: 'ja', label: 'Japanese' },
              ] as const).map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setEditorLanguage(option.id)}
                  style={{
                    padding: '0.5rem 0.85rem',
                    borderRadius: 999,
                    border: `1px solid ${editorLanguage === option.id ? 'var(--ui-flow-accent)' : 'var(--ui-flow-border)'}`,
                    background: editorLanguage === option.id ? 'var(--ui-flow-accent)' : 'transparent',
                    color: editorLanguage === option.id ? '#fff' : 'var(--ui-flow-text)',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {activeTab === 'welcome' ? (
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div className="stacked-title">
              <label className="design-form-label">{t('botWelcomeMessages.title', 'Welcome messages')}</label>
              <span className="design-form-hint">
                {t(
                  'botWelcomeMessages.subtitle',
                  'Control the first message shown in the web widget before the visitor sends their first message.'
                )}
              </span>
            </div>

            <div style={{ display: 'grid', gap: '1rem' }}>
              <div style={{ padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,255,255,0.55)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.75rem' }}>
                  <Globe size={18} style={{ color: 'var(--ui-flow-accent)' }} />
                  <div>
                    <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>
                      {t('botWelcomeMessages.webTitle', 'Web widget')}
                    </div>
                    <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                      {t(
                        'botWelcomeMessages.webSubtitle',
                        'Shown as the first bot bubble before the visitor sends their first message.'
                      )}
                    </p>
                  </div>
                </div>
                <GlassField label={t('botWelcomeMessages.messageLabel', 'Message')}>
                  <textarea
                    className="design-form-input"
                    rows={3}
                    value={webWelcomeMessage}
                    onChange={(event) => setWebWelcomeMessage(event.target.value)}
                    placeholder={t('widgetDesign.welcomePlaceholder', 'Welcome! How can I help you today?')}
                    style={{ resize: 'vertical', width: '100%' }}
                  />
                </GlassField>
              </div>

              <div style={{ padding: '1rem', borderRadius: 14, border: '1px solid var(--ui-flow-border)', background: 'rgba(255,255,255,0.55)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.75rem' }}>
                  <LineIcon className="nav-icon" aria-hidden />
                  <div>
                    <div style={{ fontWeight: 600, color: 'var(--ui-flow-text)' }}>
                      {t('botWelcomeMessages.lineTitle', 'LINE')}
                    </div>
                    <p className="card-subtitle" style={{ margin: 0, fontSize: '0.85rem' }}>
                      {t(
                        'botWelcomeMessages.lineManagedSubtitle',
                        'The first LINE message is currently handled by LINE Official Account greeting settings, not by the app.'
                      )}
                    </p>
                  </div>
                </div>
                <div
                  style={{
                    display: 'flex',
                    gap: '0.65rem',
                    alignItems: 'flex-start',
                    padding: '0.75rem 0.9rem',
                    borderRadius: 10,
                    background: 'rgba(15, 23, 42, 0.04)',
                    color: 'var(--ui-flow-muted)',
                    fontSize: '0.85rem',
                  }}
                >
                  <Info size={16} style={{ flex: '0 0 auto', marginTop: 1 }} />
                  <span>
                    {t(
                      'botWelcomeMessages.lineManagedNote',
                      'We are not sending a custom follow welcome from the app right now. If you want a first message on LINE, manage it in LINE Official Account Manager.'
                    )}
                  </span>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              {renderSaveAction()}
            </div>
          </div>
        ) : (
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div className="stacked-title">
              <label className="design-form-label">{t('botSuggestedMessages.title', 'Suggested messages')}</label>
              <span className="design-form-hint">
                {t(
                  'botSuggestedMessages.subtitle',
                  'Quick actions shown when the chat opens. Auto-generate from your trained content or add manually.'
                )}
              </span>
            </div>

            <SuggestedMessagesEditor
              suggestedMessages={suggestedMessages}
              onChange={setSuggestedMessages}
              title=""
              subtitle=""
              addButtonPlacement="bottom"
              maxItems={10}
              actions={renderSaveAction()}
              availableTypes={availableTypes}
              botId={botId}
            />
          </div>
        )}
      </GlassCard>
    </AnimatedPage>
  )
}
