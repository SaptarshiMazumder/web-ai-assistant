import { useNavigate } from 'react-router-dom'
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
        <div className="card-title">Name your helper</div>
        <div className="card-subtitle">This is the name customers will see.</div>
        <input value={botName} onChange={(event) => setBotName(event.target.value)} placeholder="Web AI Assistant" />
      </div>

      <div>
        <div className="card-title">Business type (optional)</div>
        <div className="card-subtitle">This helps us show a few helpful suggestions.</div>
        <select
          value={businessType}
          onChange={(e) => setBusinessType((e.target.value || '') as '' | 'hotel' | 'other')}
          style={{ width: '100%', maxWidth: '320px', padding: '0.5rem' }}
        >
          <option value="">—</option>
          <option value="hotel">Hotel</option>
          <option value="other">Other</option>
        </select>
      </div>

      {localError && (
        <div className="alert error">
          <div style={{ whiteSpace: 'pre-line' }}>{localError}</div>
        </div>
      )}

      <div className="flow-actions">
        <button
          type="button"
          className="primary"
          onClick={() => void handleContinue()}
          disabled={!botName.trim()}
        >
          Continue
        </button>
      </div>
    </div>
  )
}
