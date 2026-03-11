import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { SectionHeader, UiButton } from '../../components/ui'
import { SuggestedMessagesEditor } from '../../components/SuggestedMessagesEditor'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useCreateBotFlow } from './CreateBotContext'

type SuggestedMessageType = 'ai_response' | 'show_menu' | 'escalate'

export default function CreateBotSuggestedMessagesPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { fetchPlatformConfig } = useDashboardData()
  const { step2, step3, step4, flow } = useCreateBotFlow()
  const { reservationPlatform } = step2
  const { botId } = step3
  const [availableTypes, setAvailableTypes] = useState<SuggestedMessageType[]>(['ai_response'])

  useEffect(() => {
    if (!botId) {
      navigate(flow.firstPath)
    }
  }, [botId, navigate, flow.firstPath])

  useEffect(() => {
    let mounted = true
    void fetchPlatformConfig(step4.botLanguage).then((result) => {
      if (!mounted) return
      const selectedPlatformId = String(reservationPlatform || '').trim().toLowerCase()
      const selectedPlatform = result.platforms.find((platform) => platform.id.toLowerCase() === selectedPlatformId)
      const configuredTypes = (selectedPlatform?.availableSuggestedMessageTypes || result.defaultAvailableSuggestedMessageTypes || [])
        .filter((type): type is SuggestedMessageType => type === 'ai_response' || type === 'show_menu' || type === 'escalate')
      setAvailableTypes(configuredTypes.length > 0 ? configuredTypes : ['ai_response'])
    })
    return () => {
      mounted = false
    }
  }, [fetchPlatformConfig, reservationPlatform, step4.botLanguage])

  return (
    <div className="flow-panel-body">
      <SectionHeader
        title={t('botSuggestedMessages.title', 'Suggested messages')}
        subtitle={t(
          'botSuggestedMessages.subtitle',
          'Quick actions shown when the chat opens. Configured per platform in config YAML (e.g. Tabelog: Menu, Human support).'
        )}
      />
      <section className="ui-glass-card">
        <SuggestedMessagesEditor
          suggestedMessages={step4.suggestedMessages}
          onChange={step4.setSuggestedMessages}
          title=""
          subtitle=""
          addButtonPlacement="bottom"
          maxItems={10}
          availableTypes={availableTypes}
          botId={botId || undefined}
        />
      </section>
      <div className="flow-actions">
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          {t('common.back', 'Back')}
        </UiButton>
        <UiButton variant="primary" onClick={() => flow.nextPath && navigate(flow.nextPath)}>
          {t('common.continue', 'Continue')}
        </UiButton>
      </div>
    </div>
  )
}
