import { useNavigate } from 'react-router-dom'
import { FlowSelect } from '../../components/FlowSelect'
import { GlassField, UiButton } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'

export default function CreateBotDetailsPage() {
  const navigate = useNavigate()
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
        <div className="card-title">Name your agent</div>
        <div className="card-subtitle">
          Choose a name your customers will see when they open the chat.
        </div>
      </div>

      <GlassField label="Agent name">
        <input
          type="text"
          value={botName}
          onChange={(event) => setBotName(event.target.value)}
          placeholder="e.g. Concierge, Support, Luna..."
        />
      </GlassField>

      <GlassField
        label="Business type (optional)"
        helper="Helps us tailor suggestions for your industry."
      >
        <FlowSelect
          value={businessType}
          onChange={(next) => setBusinessType((next || '') as '' | 'hotel' | 'other')}
          options={[
            { value: '', label: '--' },
            { value: 'hotel', label: 'Hotel' },
            { value: 'other', label: 'Other' },
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
          variant="primary"
          onClick={() => void handleContinue()}
          disabled={!botName.trim()}
        >
          Continue
        </UiButton>
      </div>
    </div>
  )
}
