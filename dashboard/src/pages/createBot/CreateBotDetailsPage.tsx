import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { FlowSelect } from '../../components/FlowSelect'
import { GlassField, UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotDetailsPage() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const { step1, flow } = useCreateBotFlow()
  const {
    botName,
    setBotName,
    businessType,
    setBusinessType,
    localError,
  } = step1

  const handleContinue = async () => {
    if (flow.nextPath) navigate(flow.nextPath)
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">{t('createBot.nameAgent', 'Name your agent')}</div>
        <div className="card-subtitle">
          {t('createBot.nameAgentSubtitle', 'Choose a name your customers will see when they open the chat.')}
        </div>
      </div>

      <GlassField label={t('createBot.agentName', 'Agent name')}>
        <input
          type="text"
          value={botName}
          onChange={(event) => setBotName(event.target.value)}
          placeholder={t('createBot.agentNamePlaceholder', 'e.g. Concierge, Support, Luna...')}
        />
      </GlassField>

      <GlassField
        label={t('createBot.businessType', 'Business type (optional)')}
        helper={t('createBot.businessTypeHelper', 'Helps us tailor suggestions for your industry.')}
      >
        <FlowSelect
          value={businessType}
          onChange={(next) => setBusinessType((next || '') as '' | 'hotel' | 'restaurant' | 'other')}
          options={[
            { value: '', label: t('common.blank', '--') },
            { value: 'hotel', label: t('createBot.hotel', 'Hotel') },
            { value: 'restaurant', label: t('createBot.restaurant', 'Restaurant') },
            { value: 'other', label: t('createBot.other', 'Other') },
          ]}
        />
      </GlassField>

      {localError && (
        <div className="alert error">
          <div style={{ whiteSpace: 'pre-line' }}>{localError}</div>
        </div>
      )}

      <div className="flow-actions">
        <UiButton
          variant="secondary"
          onClick={() => {
            if (flow.prevPath) {
              navigate(flow.prevPath)
              return
            }
            navigate('/bots')
          }}
        >
          {t('common.back', 'Back')}
        </UiButton>
        <UiButton
          variant="primary"
          onClick={() => void handleContinue()}
          disabled={!botName.trim()}
        >
          {t('common.continue', 'Continue')}
        </UiButton>
      </div>
    </div>
  )
}
