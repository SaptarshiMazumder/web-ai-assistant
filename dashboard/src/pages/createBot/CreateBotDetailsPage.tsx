import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'
import { PlayIcon, StopIcon } from './DiscoveryIcons'

export default function CreateBotDetailsPage() {
  const navigate = useNavigate()
  const { step1, flow } = useCreateBotFlow()
  const { botName, setBotName, websiteUrl, setWebsiteUrl, businessType, setBusinessType, isDiscovering, localError, discoverUrls, stopDiscovery } = step1
  const [starting, setStarting] = useState(false)

  const handleContinue = async () => {
    if (starting || isDiscovering) return
    setStarting(true)
    try {
      const ok = await discoverUrls()
      if (ok && flow.nextPath) {
        navigate(flow.nextPath)
      }
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Name your bot</div>
        <div className="card-subtitle">Give your chatbot a friendly, customer-facing name.</div>
        <input value={botName} onChange={(event) => setBotName(event.target.value)} placeholder="Web AI Assistant" />
      </div>

      <div>
        <div className="card-title">Website to learn from</div>
        <div className="card-subtitle">We will scan this site and suggest pages to include.</div>
        <input value={websiteUrl} onChange={(event) => setWebsiteUrl(event.target.value)} placeholder="https://yourwebsite.com" />
      </div>

      <div>
        <div className="card-title">Business type (optional)</div>
        <div className="card-subtitle">Hotel bots can use booking and availability features.</div>
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
        {isDiscovering ? (
          <button type="button" className="primary" onClick={stopDiscovery} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <StopIcon />
            Stop
          </button>
        ) : (
          <button type="button" className="primary" onClick={() => void handleContinue()} disabled={starting} style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <PlayIcon />
            {starting ? 'Starting…' : 'Start bot training'}
          </button>
        )}
      </div>
    </div>
  )
}
