import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Check, Save } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  DEFAULT_WIDGET_DESIGN_STATE,
  type SuggestedMessageConfig,
} from '../../components/WidgetDesignForm'
import { SuggestedMessagesEditor } from '../../components/SuggestedMessagesEditor'
import { useDashboardData } from '../../hooks/useDashboardData'
import { AnimatedPage, GlassCard, SectionHeader, UiButton } from '../../components/ui'
import {
  getBotSuggestedMessagesBaseLanguage,
  getSuggestedMessagesForLanguageFromConfig,
  setSuggestedMessagesForLanguageInConfig,
  type SuggestedMessagesLanguage,
} from '../../utils/suggestedMessagesConfig'

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
  const [suggestedMessages, setSuggestedMessages] = useState<SuggestedMessageConfig[]>(() => DEFAULT_WIDGET_DESIGN_STATE.suggestedMessages)
  const [editorLanguage, setEditorLanguage] = useState<SuggestedMessagesLanguage>('en')
  const [saving, setSaving] = useState(false)
  const [savedJustNow, setSavedJustNow] = useState(false)
  const [availableTypes, setAvailableTypes] = useState<Array<'ai_response' | 'show_menu' | 'escalate'>>(['ai_response'])

  useEffect(() => {
    setEditorLanguage(getBotSuggestedMessagesBaseLanguage(selectedBotWidgetConfig))
  }, [selectedBotWidgetConfig, botId])

  useEffect(() => {
    setSuggestedMessages(getSuggestedMessagesForLanguageFromConfig(selectedBotWidgetConfig, editorLanguage))
  }, [selectedBotWidgetConfig, editorLanguage])

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

  const handleSuggestedMessagesChange = (next: SuggestedMessageConfig[]) => {
    setSuggestedMessages(next)
  }

  const handleSave = async () => {
    if (!botId || saving || savedJustNow) return
    setSaving(true)
    setSavedJustNow(false)
    try {
      const nextConfig = setSuggestedMessagesForLanguageInConfig(
        selectedBotWidgetConfig ?? {},
        editorLanguage,
        suggestedMessages
      )
      await saveWidgetConfig(botId, {
        suggestedMessagesByLanguage: nextConfig.suggestedMessagesByLanguage as Record<string, unknown>,
      })
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

  const languageToggle = (
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
  )

  return (
    <AnimatedPage>
      <SectionHeader
        title={t('botSuggestedMessages.title', 'Suggested messages')}
        subtitle={t('botSuggestedMessages.subtitle', 'Quick actions shown when the chat opens. Configured per platform in config YAML (e.g. Tabelog: Menu, Human support).')}
      />

      <GlassCard style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap', marginBottom: '1rem' }}>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--ui-flow-text)' }}>
            {t('widgetDesign.botLanguage', 'Bot language')}
          </div>
          {languageToggle}
        </div>
        <SuggestedMessagesEditor
          suggestedMessages={suggestedMessages}
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
