import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateBotFlow } from './CreateBotContext'
import { PlayIcon, StopIcon } from './DiscoveryIcons'

export default function CreateBotDetailsPage() {
  const navigate = useNavigate()
  const { step1, flow } = useCreateBotFlow()
  const {
    botName,
    setBotName,
    websiteUrl,
    setWebsiteUrl,
    contentHosting,
    setContentHosting,
    businessType,
    setBusinessType,
    isDiscovering,
    localError,
    continueWithoutSources,
    discoverUrls,
    stopDiscovery,
  } = step1
  const [starting, setStarting] = useState(false)

  const handleContinue = async () => {
    if (starting || isDiscovering) return
    setStarting(true)
    try {
      if (contentHosting === 'shared') {
        if (flow.nextPath) navigate(flow.nextPath)
        return
      }
      const ok = await discoverUrls()
      if (ok && flow.nextPath) navigate(flow.nextPath)
    } finally {
      setStarting(false)
    }
  }

  const handleSkip = async () => {
    if (starting || isDiscovering) return
    setStarting(true)
    try {
      const botId = await continueWithoutSources()
      if (botId && flow.nextPath) {
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
        <div className="card-title">Where is your content hosted?</div>
        <div className="card-subtitle">
          If your site is hosted on a shared provider with multiple clients, we&apos;ll only fetch the exact URLs you paste.
        </div>
        <div className="design-form-radio-group" style={{ marginTop: '0.5rem' }}>
          <label className="design-form-radio-card">
            <input
              type="radio"
              name="contentHosting"
              value="own"
              checked={contentHosting === 'own'}
              onChange={() => setContentHosting('own')}
            />
            <span>My own website</span>
          </label>
          <label className="design-form-radio-card">
            <input
              type="radio"
              name="contentHosting"
              value="shared"
              checked={contentHosting === 'shared'}
              onChange={() => setContentHosting('shared')}
            />
            <span>Shared service provider</span>
          </label>
        </div>
      </div>

      {contentHosting === 'own' ? (
        <div>
          <div className="card-title">Website to learn from</div>
          <div className="card-subtitle">We will scan this site and suggest pages to include.</div>
          <input value={websiteUrl} onChange={(event) => setWebsiteUrl(event.target.value)} placeholder="https://yourwebsite.com" />
        </div>
      ) : (
        <div className="alert info" style={{ marginTop: '0.25rem' }}>
          Shared-provider mode: we will <b>not</b> crawl your whole site. Next, you&apos;ll paste the exact page URLs to train on.
        </div>
      )}

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
          <div className="row" style={{ gap: '0.75rem', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="primary"
              onClick={() => void handleContinue()}
              disabled={starting}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
            >
              <PlayIcon />
              {starting ? 'Starting…' : contentHosting === 'shared' ? 'Continue' : 'Start bot training'}
            </button>
            <button type="button" className="ghost" onClick={() => void handleSkip()} disabled={starting}>
              Skip sources for now
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
