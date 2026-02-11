import { useNavigate } from 'react-router-dom'
import { FlowSelect } from '../../components/FlowSelect'
import { UiButton } from '../../components/ui'
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

      <div className="flow-field">
        <label className="flow-field-label">Agent name</label>
        <div className="flow-field-input-wrap">
          <input
            type="text"
            value={botName}
            onChange={(event) => setBotName(event.target.value)}
            placeholder="e.g. Concierge, Support, Luna..."
          />
        </div>
      </div>

      <div className="flow-field">
        <label className="flow-field-label">Business type (optional)</label>
        <div className="flow-field-input-wrap">
          <FlowSelect
            value={businessType}
            onChange={(next) => setBusinessType((next || '') as '' | 'hotel' | 'other')}
            options={[
              { value: '', label: '--' },
              { value: 'hotel', label: 'Hotel' },
              { value: 'other', label: 'Other' },
            ]}
          />
        </div>
        <span className="flow-field-helper">Helps us tailor suggestions for your industry.</span>
      </div>

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
